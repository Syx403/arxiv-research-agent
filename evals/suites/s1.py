"""S1 retrieval (DESIGN §11.2): every QASPER question is searched within its own paper by each arm,
and the arm's ranking of the paper's paragraphs is graded against the annotators' evidence."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ara.db.pool import Pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.rag import search
from ara.rag.embed import Vector, embed
from ara.rag.ingest import ingest
from ara.rag.sources import ParsedPaper, qasper_paper
from evals import qasper

MANIFEST = Path(__file__).parents[1] / "datasets" / "s1_qasper.json"
SEED, PAPERS, PER_PAPER, DEV_PAPERS = 20261007, 10, 3, 4
ARMS = ("bm25", "bm25_stemmed", "fts", "dense", "rrf", "rrf_rerank")
# Paired comparisons in the report, (a, b) read as a - b: the tokenizer (D15), fusion against each
# of its inputs, and what reranking adds.
PAIRS = (
    ("bm25_stemmed", "bm25"),
    ("rrf", "bm25_stemmed"),
    ("rrf", "dense"),
    ("rrf_rerank", "rrf"),
)
RERANK_POOL = 30  # RRF candidates sent to the reranker


@dataclass(frozen=True)
class Item:
    id: str
    paper: str  # arXiv id
    split: str  # "dev" for tuning, "test" for reporting
    question: str
    references: tuple[frozenset[int], ...]


def build_manifest() -> dict[str, Any]:
    """Draw the items once (fixed seed); the manifest is committed, the data is not."""
    digest = qasper.download()
    picks = qasper.select(qasper.records(), seed=SEED, papers=PAPERS, per_paper=PER_PAPER)
    return {
        "suite": "s1",
        "dataset": {"name": "allenai/qasper", "split": "validation", "sha256": digest},
        "seed": SEED,
        "items": [
            {"id": qid, "paper": paper, "split": "dev" if n < DEV_PAPERS else "test"}
            for n, (paper, qids) in enumerate(picks)
            for qid in qids
        ],
    }


def manifest_hash() -> str:
    return sha256(MANIFEST.read_bytes()).hexdigest()[:12]


def splits() -> dict[str, str]:
    """Item id → "dev" or "test", from the committed manifest."""
    return {e["id"]: e["split"] for e in json.loads(MANIFEST.read_text())["items"]}


def load(papers: int | None = None) -> tuple[list[Item], dict[str, ParsedPaper]]:
    """The manifest's items (the first `papers` papers only, if given) and their parsed papers.
    The local data must be the exact file the manifest was drawn from."""
    manifest = json.loads(MANIFEST.read_text())
    if qasper.download() != manifest["dataset"]["sha256"]:
        raise ValueError(f"{qasper.PATH} differs from the file the S1 manifest was drawn from")
    order = list(dict.fromkeys(entry["paper"] for entry in manifest["items"]))[:papers]
    data = qasper.records()
    found = {q.id: q for paper in order for q in qasper.questions(data[paper])}
    items = [
        Item(e["id"], e["paper"], e["split"], found[e["id"]].text, found[e["id"]].references)
        for e in manifest["items"]
        if e["paper"] in order
    ]
    return items, {paper: qasper_paper(data[paper]) for paper in order}


@dataclass(frozen=True)
class ChunkInfo:
    paragraph: int
    search_text: str


class S1:
    """Holds what the arms share: the ingested documents and their chunks."""

    def __init__(self, pool: Pool, gateway: Gateway, scope: Scope) -> None:
        self.pool, self.gateway, self.scope = pool, gateway, scope
        self.documents: dict[str, int] = {}
        self.chunks: dict[int, ChunkInfo] = {}

    async def prepare(self, items: Sequence[Item], papers: dict[str, ParsedPaper]) -> None:
        """Ingest the papers and embed every question in one batch, so the per-item searches
        find their query vectors in the cache instead of sending one request each."""
        for arxiv_id, paper in papers.items():
            self.documents[arxiv_id] = await ingest(
                paper, pool=self.pool, gateway=self.gateway, scope=self.scope
            )
        await self._embed([item.question for item in items])
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT id, paragraph, search_text FROM chunks WHERE document_id = ANY(%s)",
                (list(self.documents.values()),),
            )
            for row in await cursor.fetchall():
                self.chunks[row["id"]] = ChunkInfo(row["paragraph"], row["search_text"])

    async def rankings(self, item: Item) -> dict[str, list[int]]:
        """Each arm's ranking: the source paragraph of each retrieved chunk, best first."""
        documents = [self.documents[item.paper]]
        [vector] = await self._embed([item.question])
        async with self.pool.connection() as conn:
            chunk_lists = {
                "bm25": await search.bm25(conn, item.question, documents, stemmed=False),
                "bm25_stemmed": await search.bm25(conn, item.question, documents, stemmed=True),
                "fts": await search.fts(conn, item.question, documents),
                "dense": await search.dense(conn, vector, documents),
            }
        fused = search.rrf([chunk_lists["bm25_stemmed"], chunk_lists["dense"]])
        chunk_lists["rrf"] = fused
        pool = fused[:RERANK_POOL]
        reranked = await self.gateway.rerank(
            item.question,
            [self.chunks[c].search_text for c in pool],
            top_n=len(pool),
            scope=self.scope,
        )
        chunk_lists["rrf_rerank"] = [pool[result.index] for result in reranked]
        return {arm: self._paragraphs(chunk_lists[arm]) for arm in ARMS}

    def _paragraphs(self, chunk_ids: Sequence[int]) -> list[int]:
        return [self.chunks[c].paragraph for c in chunk_ids]

    async def _embed(self, texts: list[str]) -> list[Vector]:
        return await embed(texts, pool=self.pool, gateway=self.gateway, scope=self.scope)
