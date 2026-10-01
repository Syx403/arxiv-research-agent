import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.corpus.http import ResponseCache
from src.llm.budget import BudgetExceeded, RequestBudget, use_turn_budget, set_turn_budget
from src.llm.providers.openai_embed import OpenAIEmbeddingClient


async def test_embedding_cache_reuses_exact_content_and_model_and_only_pays_for_misses(tmp_path):
    async def create(**kw):
        return {
            "data": [{"embedding": [float(len(t))] * 1536} for t in kw["input"]],
            "usage": {"prompt_tokens": 10},
        }

    method = AsyncMock(side_effect=create)
    client = OpenAIEmbeddingClient(
        api_key="test",
        client=SimpleNamespace(embeddings=SimpleNamespace(create=method)),
        cache=ResponseCache(tmp_path),
    )
    first = await client.embed(["one", "two", "one"])
    assert first[0] == first[2]
    assert method.call_args.kwargs["input"] == ["one", "two"]
    again = await client.embed(["two", "one"])
    assert again == [first[1], first[0]] and client.last_prompt_tokens == 0
    await client.embed(["one", "three"])
    assert method.call_args.kwargs["input"] == ["three"]
    await client.embed(["one"], model="new-model")
    assert method.await_count == 3


async def test_turn_budget_is_shared_by_parallel_tasks_and_survives_resume(tmp_path):
    budget = RequestBudget(2, path=tmp_path / "ledger.json")

    async def reserve():
        try:
            return budget.reserve("deepseek_native", "deepseek-flash", 100, 10000)
        except BudgetExceeded:
            return None

    with budget.activate(), use_turn_budget(0.025, identifier="same-turn"):
        results = await asyncio.gather(*(reserve() for _ in range(4)))
        assert sum(r is not None for r in results) == 2
    resumed = RequestBudget(2, path=tmp_path / "ledger.json")
    with resumed.activate(), use_turn_budget(0.025, identifier="same-turn"):
        with pytest.raises(BudgetExceeded):
            resumed.reserve("deepseek_native", "deepseek-flash", 100, 10000)
        set_turn_budget(0.15)
        resumed.reserve("deepseek_native", "deepseek-flash", 100, 10000)
    assert resumed.snapshot()["committed_usd"] < 0.15
    assert len(resumed.snapshot()["requests"]) == 3
