from __future__ import annotations

import re

from pydantic import ValidationError

from src.core.types import Decomposition, MAX_SUB_QUESTIONS, Message, SubQuestion
from src.llm.client import get_chat_client


SYSTEM_PROMPT = """You decompose research questions for retrieval.
Return only JSON matching this shape:
{"sub_questions":[{"text":"...","depends_on":[0]}]}
Dependencies are zero-based indexes and may only point to earlier sub-questions.
"""


async def decompose(question: str, *, max_subq: int = MAX_SUB_QUESTIONS) -> Decomposition:
    question = question.strip()
    comparison_entities = _detect_comparison_entities(question)
    if not comparison_entities and _is_trivial_question(question):
        return Decomposition(sub_questions=[SubQuestion(text=question, depends_on=[])])

    client = get_chat_client("main")
    decomposition = await _decompose_with_repair(
        client,
        SYSTEM_PROMPT,
        (
            f"Decompose this question into at most {max_subq} retrieval sub-questions. "
            f"Question: {question}"
        ),
    )
    if not comparison_entities or _contains_all_entities(decomposition, comparison_entities):
        return decomposition

    retry_system = (
        f"{SYSTEM_PROMPT}\n"
        "Decompose this comparison question. The decomposition MUST include explicit sub-questions "
        f"covering each of these named entities: {', '.join(comparison_entities)}. "
        "Return at least one sub-question per entity, plus optionally a contrast/comparison sub-question."
    )
    try:
        retry = await _decompose_with_repair(
            client,
            retry_system,
            f"Question: {question}",
        )
    except ValidationError:
        return _fallback_comparison_decomposition(comparison_entities)
    if _contains_all_entities(retry, comparison_entities):
        return retry
    return _fallback_comparison_decomposition(comparison_entities)


async def _decompose_with_repair(client, system_prompt: str, user_prompt: str) -> Decomposition:
    response = await client.chat(
        [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ],
        temperature=0.0,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    broken_output = response.content or ""
    try:
        return Decomposition.model_validate_json(broken_output)
    except ValidationError:
        repair = await client.chat(
            [
                Message(role="system", content=system_prompt),
                Message(
                    role="user",
                    content=(
                        "Fix this invalid decomposition JSON. Return only valid JSON for the schema. "
                        f"Invalid output:\n{broken_output}"
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        return Decomposition.model_validate_json(repair.content or "")


def _is_trivial_question(question: str) -> bool:
    tokens = re.findall(r"\S+", question)
    if len(tokens) < 8:
        return True
    stripped = question.strip()
    return stripped.endswith("?") and len(re.findall(r"[.!?]", stripped[:-1])) == 0


def _detect_comparison_entities(question: str) -> list[str]:
    stripped = question.strip().rstrip("?.!")
    patterns = [
        r"^how does\s+(?P<x>.+?)\s+differ from\s+(?P<y>.+?)(?:\s+(?:for|in|on|with|about)\b.*)?$",
        r"^compare\s+(?P<x>.+?)\s+and\s+(?P<y>.+?)(?:\s+(?:as|for|in|on|with|about)\b.*)?$",
        r"^(?P<x>.+?)\s+(?:vs\.?|versus)\s+(?P<y>.+?)$",
        r"^difference between\s+(?P<x>.+?)\s+and\s+(?P<y>.+?)(?:\s+(?:for|in|on|with|about)\b.*)?$",
        r"^(?P<x>.+?)\s+compared to\s+(?P<y>.+?)(?:\s+(?:for|in|on|with|about)\b.*)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match is None:
            continue
        entities = [_clean_entity(match.group("x")), _clean_entity(match.group("y"))]
        if all(entities):
            return entities
    return []


def _clean_entity(entity: str) -> str:
    return entity.strip(" \t\n\r,;:\"'()[]{}")


def _contains_all_entities(decomposition: Decomposition, entities: list[str]) -> bool:
    subq_text = "\n".join(sub_question.text.lower() for sub_question in decomposition.sub_questions)
    return all(entity.lower() in subq_text for entity in entities)


def _fallback_comparison_decomposition(entities: list[str]) -> Decomposition:
    sub_questions = [SubQuestion(text=f"What is {entity}?", depends_on=[]) for entity in entities]
    compare_text = f"How do {' and '.join(entities)} compare?"
    sub_questions.append(SubQuestion(text=compare_text, depends_on=list(range(len(entities)))))
    return Decomposition(sub_questions=sub_questions)
