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
    if _is_trivial_question(question):
        return Decomposition(sub_questions=[SubQuestion(text=question, depends_on=[])])

    client = get_chat_client("main")
    response = await client.chat(
        [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Decompose this question into at most {max_subq} retrieval sub-questions. "
                    f"Question: {question}"
                ),
            ),
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
                Message(role="system", content=SYSTEM_PROMPT),
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
