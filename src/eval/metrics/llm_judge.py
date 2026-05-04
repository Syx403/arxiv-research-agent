from __future__ import annotations

import json

from pydantic import ValidationError

from src.core.types import JudgeVerdict, Message
from src.eval.datasets.gold_questions import GoldQuestion
from src.llm.client import get_chat_client


SYSTEM_PROMPT = """You are an evaluation judge for a research assistant.
Score the answer on 1-5 integer scales:
- correctness: matches the expected papers and core facts.
- groundedness: includes the required concepts and does not contradict the corpus.
- completeness: answers the question with enough useful detail.
Return only JSON: {"correctness": 1, "groundedness": 1, "completeness": 1, "rationale": "..."}.
"""
JUDGE_MAX_TOKENS = 2000
JUDGE_REPAIR_MAX_TOKENS = 1000


async def judge_answer(question: str, answer: str, gold: GoldQuestion) -> JudgeVerdict:
    client = get_chat_client("main")
    response = await client.chat(
        [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Question:\n{question}\n\n"
                    f"Answer:\n{answer}\n\n"
                    f"Expected paper IDs:\n{json.dumps(gold.expected_paper_ids)}\n\n"
                    f"Must mention:\n{json.dumps(gold.answer_must_mention)}\n\n"
                    f"Must not mention:\n{json.dumps(gold.answer_must_not_mention)}"
                ),
            ),
        ],
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    broken_output = response.content or ""
    try:
        return JudgeVerdict.model_validate_json(broken_output)
    except ValidationError:
        repair = await client.chat(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=(
                        "Fix this invalid judge JSON. Return only valid JSON for the schema. "
                        f"Invalid output:\n{broken_output}"
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=JUDGE_REPAIR_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        try:
            return JudgeVerdict.model_validate_json(repair.content or "")
        except ValidationError:
            return JudgeVerdict(
                correctness=1,
                groundedness=1,
                completeness=1,
                rationale="Judge returned invalid JSON twice; scored conservatively.",
            )
