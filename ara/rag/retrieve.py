"""The product's retrieval (D17, D18): stemmed BM25 and dense search, fused by RRF, the top
RERANK_POOL reranked by Cohere, the best TOP_K passages returned with their sentence offsets."""

from collections.abc import Sequence
from dataclasses import dataclass

from ara.db.pool import Pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.rag import search
from ara.rag.embed import embed

RERANK_POOL = 30
TOP_K = 8

PASSAGES = """
SELECT c.id, d.paper_id, c.paragraph, c.heading_path, c.text, c.search_text, c.sentences
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.id = ANY(%s)
"""


@dataclass(frozen=True)
class Passage:
    chunk_id: int
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
    query: str, documents: Sequence[int], *, pool: Pool, gateway: Gateway, scope: Scope
) -> list[Passage]:
    [vector] = await embed([query], pool=pool, gateway=gateway, scope=scope)
    async with pool.connection() as conn:
        lexical = await search.bm25(conn, query, documents, stemmed=True)
        semantic = await search.dense(conn, vector, documents)
        candidates = search.rrf([lexical, semantic])[:RERANK_POOL]
        if not candidates:
            return []
        rows = await (await conn.execute(PASSAGES, (candidates,))).fetchall()
    by_id = {
        row["id"]: Passage(
            row["id"],
            row["paper_id"],
            row["paragraph"],
            row["heading_path"],
            row["text"],
            row["search_text"],
            tuple(row["sentences"]),
        )
        for row in rows
    }
    ranked = await gateway.rerank(
        query,
        [by_id[c].search_text for c in candidates],
        top_n=min(TOP_K, len(candidates)),
        scope=scope,
    )
    return [by_id[candidates[result.index]] for result in ranked]
