from __future__ import annotations

from src.core.types import Message, SufficiencyVerdict
from src.llm.client import get_chat_client
from src.retrieval.index.types import Hit


async def is_sufficient(question: str, evidence: list[Hit]) -> SufficiencyVerdict:
    client = get_chat_client("main")
    evidence_text = "\n\n".join(
        f"[{hit.paper_id}#{hit.chunk_id}] {hit.section or 'Unknown'}\n{hit.text}" for hit in evidence
    )
    response = await client.chat(
        [
            Message(
                role="system",
                content=(
                    "Decide whether the evidence is sufficient to answer the question. "
                    'Return only JSON: {"sufficient": true, "missing_aspects": []}'
                ),
            ),
            Message(role="user", content=f"Question:\n{question}\n\nEvidence:\n{evidence_text}"),
        ],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return SufficiencyVerdict.model_validate_json(response.content or "")
