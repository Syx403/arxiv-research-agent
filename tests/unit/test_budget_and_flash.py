from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src.core.types import Message
from src.llm.budget import BudgetExceeded, RequestBudget
from src.llm.client import _RoutedChatClient
from src.llm.providers.deepseek import DeepSeekChatClient
from src.llm.registry import ReasoningSettings, get_route, snapshot_chat_routes, use_chat_routes


@pytest.mark.asyncio
async def test_flash_roles_send_same_model_different_effort_with_reasoning_headroom():
    captured = []

    async def transport(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "deepseek-flash", "choices": [{"message": {"content": "ok", "reasoning_content": "private reasoning"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22,
                      "prompt_cache_hit_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 10}},
        })

    async with httpx.AsyncClient(base_url="https://api.deepseek.test/v1", transport=httpx.MockTransport(transport)) as http:
        provider = DeepSeekChatClient(api_key="test-only", http_client=http)
        for role in ("main", "fast"):
            result = await _RoutedChatClient(role, get_route(role), provider).chat([Message(role="user", content="hi")], max_tokens=320)
            assert result.usage.completion_tokens_details["reasoning_tokens"] == 10
    assert [c["model"] for c in captured] == ["deepseek-flash"] * 2
    assert [c["reasoning_effort"] for c in captured] == ["high", "low"]
    assert [c["max_tokens"] for c in captured] == [8512, 2368]
    assert all("temperature" not in c and c["thinking"] == {"type": "enabled"} for c in captured)


@pytest.mark.asyncio
async def test_non_thinking_recovery_is_scoped_to_one_request():
    captured = []

    async def transport(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [
            {"message": {"content": "{}"}, "finish_reason": "stop"}
        ]})

    async with httpx.AsyncClient(base_url="https://api.deepseek.test/v1", transport=httpx.MockTransport(transport)) as http:
        routed = _RoutedChatClient("fast", get_route("fast"), DeepSeekChatClient(api_key="test-only", http_client=http))
        messages = [Message(role="user", content="Return JSON.")]
        await routed.chat(messages, max_tokens=700, thinking=False, response_format={"type": "json_object"})
        await routed.chat(messages, max_tokens=700)

    assert captured[0]["thinking"] == {"type": "disabled"}
    assert captured[0]["max_tokens"] == 700
    assert "reasoning_effort" not in captured[0]
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[1]["thinking"] == {"type": "enabled"}
    assert captured[1]["reasoning_effort"] == "low"
    assert captured[1]["max_tokens"] == 2748


@pytest.mark.asyncio
async def test_tool_calls_are_valid_outputs_and_round_trip_reasoning():
    captured = []

    async def transport(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "reasoning_content": "private reasoning",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"paper":"ReAct"}'}}],
        }}]})

    async with httpx.AsyncClient(base_url="https://api.deepseek.test/v1", transport=httpx.MockTransport(transport)) as http:
        provider = DeepSeekChatClient(api_key="test-only", http_client=http)
        routed = _RoutedChatClient("fast", get_route("fast"), provider)
        response = await routed.chat([Message(role="user", content="lookup")])
        assert len(captured) == 1  # Tool-only output is not an empty answer retry.
        await routed.chat([Message(role="assistant", tool_calls=response.tool_calls, reasoning_content=response.reasoning_content)])
    message = captured[-1]["messages"][0]
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"paper": "ReAct"}
    assert message["reasoning_content"] == "private reasoning"


@pytest.mark.asyncio
async def test_concurrent_reservations_and_resume_cannot_overspend(tmp_path):
    path = tmp_path / "budget.json"
    budget = RequestBudget(.025, path=path)
    async def request():
        await asyncio.sleep(0)
        try:
            return budget.reserve("deepseek_native", "deepseek-flash", 100, 10000)
        except BudgetExceeded:
            return None
    attempts = await asyncio.gather(*(request() for _ in range(8)))
    assert sum(item is not None for item in attempts) == 2
    assert budget.snapshot()["committed_usd"] < .025
    restored = RequestBudget(.025, path=path)
    with pytest.raises(BudgetExceeded):
        restored.reserve("deepseek_native", "deepseek-flash", 100, 10000)


@pytest.mark.asyncio
async def test_transport_retry_retains_unknown_charge_and_admits_each_attempt(monkeypatch):
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_INITIAL_SECONDS", .001)
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_MAX_SECONDS", .001)
    calls = 0

    async def transport(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timeout", request=request)

    budget = RequestBudget(.016)
    async with httpx.AsyncClient(base_url="https://api.deepseek.test/v1", transport=httpx.MockTransport(transport)) as http:
        client = _RoutedChatClient("main", get_route("main"), DeepSeekChatClient(api_key="test-only", http_client=http))
        with budget.activate(), pytest.raises(BudgetExceeded):
            await client.chat([Message(role="user", content="hi")], max_tokens=1000)
    assert calls == 1
    assert budget.snapshot()["uncertain_or_reserved_usd"] > 0


@pytest.mark.asyncio
async def test_exhausted_budget_blocks_request_before_transport():
    async def transport(request):
        raise AssertionError("No request should be sent")
    async with httpx.AsyncClient(base_url="https://api.deepseek.test/v1", transport=httpx.MockTransport(transport)) as http:
        client = DeepSeekChatClient(api_key="test-only", http_client=http)
        with RequestBudget(.00001).activate(), pytest.raises(BudgetExceeded):
            await client.chat([Message(role="user", content="hi")], model="deepseek-flash", max_tokens=1000)


def test_unknown_paid_rerank_plan_is_not_assumed_free():
    budget = RequestBudget(1)
    with pytest.raises(BudgetExceeded):
        budget.reserve("cohere_native", "rerank-v4.0-pro", 100, 0)


@pytest.mark.asyncio
async def test_parallel_failure_waits_for_already_admitted_requests_to_settle():
    from src.core.async_utils import gather_requests
    settled = []
    async def failure():
        raise BudgetExceeded("fixture exhausted")
    async def in_flight():
        await asyncio.sleep(.01)
        settled.append(True)
    with pytest.raises(BudgetExceeded):
        await gather_requests(failure(), in_flight())
    assert settled == [True]


@pytest.mark.asyncio
async def test_cached_clients_honor_per_run_effort_in_parallel(monkeypatch):
    from src.llm import client as clients
    captured = []
    async def transport(request):
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json={"model": "deepseek-flash", "choices": [
            {"message": {"content": "ok"}, "finish_reason": "stop"}
        ], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
    async with httpx.AsyncClient(base_url="https://api.deepseek.test/v1", transport=httpx.MockTransport(transport)) as http:
        provider = DeepSeekChatClient(api_key="test-only", http_client=http)
        monkeypatch.setattr(clients, "_provider_clients", {"deepseek_native": provider})
        monkeypatch.setattr(clients, "_routed_clients", {})
        async def request(effort):
            routes = snapshot_chat_routes(ReasoningSettings(main=effort, fast="low"))
            with use_chat_routes(routes):
                routed = clients.get_chat_client("main")
                await asyncio.sleep(0)
                assert clients.get_chat_client("main") is routed
                await routed.chat([Message(role="user", content=effort)], max_tokens=100)
        await asyncio.gather(request("low"), request("max"), request("high"))
    assert {c["reasoning_effort"]: c["max_tokens"] for c in captured} == {
        "low": 2148, "high": 8292, "max": 16484,
    }
    assert all(c["messages"][0]["content"] == c["reasoning_effort"] for c in captured)
    assert get_route("main").reasoning_effort == "high"
