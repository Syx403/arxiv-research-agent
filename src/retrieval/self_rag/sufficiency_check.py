from __future__ import annotations

from pydantic import ValidationError

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
        max_tokens=420,
        response_format={"type": "json_object"},
    )
    broken_output = response.content or ""
    try:
        return SufficiencyVerdict.model_validate_json(broken_output)
    except ValidationError:
        repair = await client.chat(
            [
                Message(
                    role="system",
                    content='Fix invalid JSON. Return compact valid JSON only: {"sufficient": true, "missing_aspects": []}',
                ),
                Message(role="user", content=f"Invalid output:\n{broken_output}"),
            ],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        try:
            return SufficiencyVerdict.model_validate_json(repair.content or "")
        except ValidationError:
            return SufficiencyVerdict(
                sufficient=True,
                missing_aspects=[],
            )
