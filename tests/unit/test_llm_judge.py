from __future__ import annotations

import pytest

from src.core.types import ChatResponse
from src.eval.datasets.gold_questions import GOLD_QUESTIONS
from src.eval.metrics import llm_judge


class _BrokenJudgeClient:
    async def chat(self, *args, **kwargs) -> ChatResponse:
        return ChatResponse(content="", model="fake", finish_reason="length")


@pytest.mark.asyncio
async def test_judge_answer_scores_conservatively_after_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(llm_judge, "get_chat_client", lambda name: _BrokenJudgeClient())

    verdict = await llm_judge.judge_answer("question", "answer", GOLD_QUESTIONS[0])

    assert verdict.correctness == 1
    assert verdict.groundedness == 1
    assert verdict.completeness == 1
    assert "invalid JSON twice" in verdict.rationale
