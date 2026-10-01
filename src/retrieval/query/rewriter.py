from __future__ import annotations


from src.core.types import Message
from src.llm.client import get_chat_client
from src.core.trace import record_event




async def rewrite_for_retrieval(subq: str) -> str:
    client = get_chat_client("fast")
    response = await client.chat(
        [
            Message(
                role="system",
                content="Rewrite this one research sub-question into ONE concise English keyword-rich retrieval query, at most 60 words. Return only the query as a single plain-text line, no Markdown, code fences, bullets, alternatives or explanation. Preserve the selected paper and requested mechanisms; omit application constraints without searchable paper concepts. A scoped single-paper task must not expand to other papers or the later whole comparison.",
            ),
            Message(role="user", content=f"Rewrite for retrieval:\n{subq.strip()}"),
        ],
        temperature=0.0,
        max_tokens=120,
        thinking=False,
     stage="query")
    rewritten = (response.content or "").strip()
    if (response.finish_reason != "stop" or not rewritten or "\n" in rewritten
            or len(rewritten.split()) > 80 or rewritten.startswith(("`", "- ", "* ", "{"))):
        record_event("retrieval_rewrite", status="original_question_fallback", finish_reason=response.finish_reason)
        return subq.strip()
    return rewritten
