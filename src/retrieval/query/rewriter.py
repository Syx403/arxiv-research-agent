from __future__ import annotations

import re

from src.core.types import Message
from src.llm.client import get_chat_client


_KEYWORD_QUERY_RE = re.compile(r"^[A-Za-z0-9_:/().,+-]+(?:\s+[A-Za-z0-9_:/().,+-]+){0,7}$")


async def rewrite_for_retrieval(subq: str) -> str:
    client = get_chat_client("fast")
    response = await client.chat(
        [
            Message(
                role="system",
                content="Rewrite research sub-questions into concise keyword-rich retrieval queries.",
            ),
            Message(role="user", content=f"Rewrite for retrieval:\n{subq.strip()}"),
        ],
        temperature=0.0,
        max_tokens=120,
    )
    rewritten = (response.content or "").strip()
    return rewritten or subq.strip()


async def hyde(subq: str) -> str:
    if not should_use_hyde(subq):
        return subq.strip()
    client = get_chat_client("fast")
    response = await client.chat(
        [
            Message(
                role="system",
                content="Write a short hypothetical answer paragraph that would help retrieve relevant papers.",
            ),
            Message(role="user", content=subq.strip()),
        ],
        temperature=0.0,
        max_tokens=220,
    )
    hypothetical = (response.content or "").strip()
    return hypothetical or subq.strip()


def should_use_hyde(subq: str) -> bool:
    text = subq.strip()
    if len(re.findall(r"\S+", text)) < 8:
        return False
    if _KEYWORD_QUERY_RE.fullmatch(text):
        return False
    return True
