from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from src.core.types import ConceptExtraction, Message
from src.llm.client import get_chat_client


SYSTEM_PROMPT = """Extract semantic-memory concepts from research text.
Extract only concepts that have a clear definition in the provided text.
Skip concepts that are merely mentioned.
Use only source chunk IDs supplied by the user as evidence.
Return only JSON matching this shape:
{"concepts":[{"display_name":"...","definition":"...","aliases":["..."],"evidence_chunk_ids":[1],"relations":[{"target_display":"...","type":"uses","evidence_chunk_ids":[1]}]}]}
"""


async def extract_concepts_from_text(text: str, *, source_chunk_ids: list[int]) -> list[ConceptExtraction]:
    client = get_chat_client("main")
    response = await client.chat(
        [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Source chunk IDs: {json.dumps(source_chunk_ids)}\n\n"
                    f"Text:\n{text}"
                ),
            ),
        ],
        temperature=0.0,
        max_tokens=1800,
        response_format={"type": "json_object"},
    )
    broken_output = response.content or ""
    try:
        concepts = _parse_payload(broken_output).concepts
    except ValidationError:
        repair = await client.chat(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=(
                        "Fix this invalid concept-extraction JSON. Return only valid JSON for the schema. "
                        f"Invalid output:\n{broken_output}"
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
        concepts = _parse_payload(repair.content or "").concepts
    if concepts:
        return concepts

    retry = await client.chat(
        [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    "The previous valid JSON contained no concepts. Re-check the text. "
                    "If a named method, framework, or technical idea is clearly defined, extract it. "
                    "Still skip concepts that are merely mentioned.\n\n"
                    f"Source chunk IDs: {json.dumps(source_chunk_ids)}\n\n"
                    f"Text:\n{text}"
                ),
            ),
        ],
        temperature=0.0,
        max_tokens=1800,
        response_format={"type": "json_object"},
    )
    try:
        return _parse_payload(retry.content or "").concepts
    except ValidationError:
        return []


def _parse_payload(content: str) -> "_ConceptExtractionPayload":
    return _ConceptExtractionPayload.model_validate_json(_extract_json_object(content))


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).removesuffix("```").strip()
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return stripped
    return stripped[start : end + 1]


class _ConceptExtractionPayload(BaseModel):
    concepts: list[ConceptExtraction] = Field(default_factory=list)
