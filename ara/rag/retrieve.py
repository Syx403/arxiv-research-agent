"""The product's retrieval (D17, D18, D30): per document, stemmed BM25 and dense search fused by
RRF, its top RERANK_POOL; one Cohere rerank of every document's candidates (a requery skips it,
D40); each document keeps its best TOP_K passages, with their sentence offsets."""

from collections.abc import Sequence
from dataclasses import dataclass

from ara.db.pool import Connection, Pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.rag import search
from ara.rag.embed import Vector, embed

RERANK_POOL = 30  # candidates per document
TOP_K = 8  # passages per document (S1, one paper: recall@8 1.00, @4 0.88; D30)

PASSAGES = """
SELECT c.id, c.document_id, d.paper_id, c.paragraph, c.heading_path, c.text, c.search_text,
       c.sentences
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.id = ANY(%s)
"""


@dataclass(frozen=True)
class Passage:
    chunk_id: int
    document_id: int
    paper_id: str
    paragraph: int
    heading_path: str
    text: str
    search_text: str
    sentences: tuple[int, ...]  # start offsets in `text`

    def sentence_texts(self) -> list[str]:
        ends = (*self.sentences[1:], len(self.text))
        return [self.text[a:b].strip() for a, b in zip(self.sentences, ends, strict=True)]


async def retrieve(
    query: str,
    documents: Sequence[int],
    *,
    pool: Pool,
    gateway: Gateway,
    scope: Scope,
    rerank: bool = True,
) -> dict[int, list[Passage]]:
    """The best passages of each document for `query`, best first. Each document is searched on
    its own, so no paper crowds out another; one rerank call scores all candidates. A reranker
    scores each (query, passage) pair on its own, so one call orders each document's passages as
    separate calls would (inferred from how cross-encoders score; not measured on Cohere, D30).
    Without `rerank`, each document's passages keep their RRF order (D40)."""
    [vector] = await embed([query], pool=pool, gateway=gateway, scope=scope)
    async with pool.connection() as conn:
        found = await candidates(conn, query, vector, documents)
        rows = await (await conn.execute(PASSAGES, (found,))).fetchall() if found else []
    if not found:
        return {d: [] for d in documents}
    by_id = {
        row["id"]: Passage(
            row["id"],
            row["document_id"],
            row["paper_id"],
            row["paragraph"],
            row["heading_path"],
            row["text"],
            row["search_text"],
            tuple(row["sentences"]),
        )
        for row in rows
    }
    if not rerank:
        return best_of_each([by_id[c] for c in found], documents)
    ranked = await gateway.rerank(
        query, [by_id[c].search_text for c in found], top_n=len(found), scope=scope
    )
    return best_of_each([by_id[found[r.index]] for r in ranked], documents)


async def candidates(
    conn: Connection, query: str, vector: Vector, documents: Sequence[int]
) -> list[int]:
    """Each document's own top RERANK_POOL chunks (BM25 and dense, fused by RRF), in turn."""
    found: list[int] = []
    for document in documents:
        lexical = await search.bm25(conn, query, [document], stemmed=True)
        semantic = await search.dense(conn, vector, [document])
        found += search.rrf([lexical, semantic])[:RERANK_POOL]
    return found


def best_of_each(ranked: list[Passage], documents: Sequence[int]) -> dict[int, list[Passage]]:
    """The first TOP_K passages of each document in the reranked order."""
    kept: dict[int, list[Passage]] = {d: [] for d in documents}
    for passage in ranked:
        if len(kept[passage.document_id]) < TOP_K:
            kept[passage.document_id].append(passage)
    return kept
