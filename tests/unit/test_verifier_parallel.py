from __future__ import annotations

import asyncio

import pytest

from src.core.types import ChatResponse, EvidenceChunk
from src.graph.nodes.synthesize import _parse_citations
from src.retrieval.self_rag import verifier


class _CountingChatClient:
    def __init__(self, expected_calls: int) -> None:
        self.expected_calls = expected_calls
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.all_started = asyncio.Event()

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.calls == self.expected_calls:
            self.all_started.set()
        await self.all_started.wait()
        await asyncio.sleep(0)
        self.active -= 1
        return ChatResponse(
            content='{"claim_kind":"paper_fact","scope_status":"requested","all_assertions_supported":true,"no_unstated_assumptions":true,"all_citations_contribute":true,"supports": true, "rationale": "supported"}',
            model="fake",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_verify_citations_fans_out_with_bounded_concurrency(monkeypatch) -> None:
    client = _CountingChatClient(expected_calls=7)
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: client)
    answer = " ".join(f"Claim {index} [p#{index}]." for index in range(7))
    citations = _parse_citations(answer)
    evidence_lookup = {index: EvidenceChunk(paper_id="p", chunk_id=index, text=f"evidence {index}") for index in range(7)}

    report = await asyncio.wait_for(verifier.verify_citations(answer, citations, evidence_lookup), timeout=1)

    assert report.passed is True
    assert client.calls == 7
    assert client.max_active == 7


@pytest.mark.asyncio
async def test_compound_claim_uses_one_joint_check_without_borrowing_uncited_evidence(monkeypatch):
    from src.core.types import ChatResponse
    messages = []
    class Client:
        async def chat(self, prompt, **kwargs):
            messages.append(prompt)
            return ChatResponse(content='{"claim_kind":"paper_fact","scope_status":"requested","all_assertions_supported":true,"no_unstated_assumptions":true,"all_citations_contribute":true,"supports":true,"rationale":"joint support"}', model="fake", finish_reason="stop")
    monkeypatch.setattr(verifier, "get_chat_client", lambda role: Client())
    answer = "A stores memories [a#1], while B updates them [b#2]."
    report = await verifier.verify_citations(answer, _parse_citations(answer), {
        1: EvidenceChunk(paper_id="a", chunk_id=1, text="A stores memories", published_at="2026-01-01"),
        2: EvidenceChunk(paper_id="b", chunk_id=2, text="B updates memories"),
        3: EvidenceChunk(paper_id="c", chunk_id=3, text="UNREFERENCED SECRET EVIDENCE"),
    })
    assert report.passed and len(report.verdicts) == 2 and len(messages) == 1
    prompt = messages[0][1].content
    assert "A stores memories" in prompt and "B updates memories" in prompt
    assert "2026-01-01" in prompt
    assert "UNREFERENCED SECRET EVIDENCE" not in prompt


@pytest.mark.asyncio
async def test_joint_claim_cannot_hide_a_missing_or_wrong_identity_source(monkeypatch):
    def forbidden(role):
        raise AssertionError("Identity mismatch must be rejected without a model call")
    monkeypatch.setattr(verifier, "get_chat_client", forbidden)
    answer = "A stores memories [a#1], while B updates them [b#2]."
    report = await verifier.verify_citations(answer, _parse_citations(answer), {
        1: EvidenceChunk(paper_id="a", chunk_id=1, text="A stores memories"),
        2: EvidenceChunk(paper_id="wrong-paper", chunk_id=2, text="B updates memories"),
    })
    assert not report.passed
    assert all(not verdict.supports for verdict in report.verdicts)


@pytest.mark.asyncio
async def test_cached_verdict_is_invalidated_when_antecedent_or_source_changes(monkeypatch):
    client = _CountingChatClient(expected_calls=2)
    monkeypatch.setattr(verifier, "get_chat_client", lambda _: client)
    answer = "Persistent cache falls to one eighth [p#1]. This reduction combines two factors [p#2]."
    lookup = {i: EvidenceChunk(paper_id="p", chunk_id=i, text="Original persistent cache evidence")
              for i in (1, 2)}
    first = await verifier.verify_citations(answer, _parse_citations(answer), lookup)
    await verifier.verify_citations(answer, _parse_citations(answer), lookup, cached_verdicts=first.verdicts)
    assert client.calls == 2  # Identical inputs can reuse the result.

    rewritten = answer.replace("Persistent cache falls to one eighth", "Global cache falls to one quarter")
    second = await verifier.verify_citations(rewritten, _parse_citations(rewritten), lookup,
                                             cached_verdicts=first.verdicts)
    assert client.calls == 4  # The unchanged 'This reduction' must also be rechecked.

    new_lookup = {**lookup, 2: lookup[2].model_copy(update={"text": "Original with added counterevidence"})}
    await verifier.verify_citations(rewritten, _parse_citations(rewritten), new_lookup,
                                    cached_verdicts=second.verdicts)
    assert client.calls == 5  # Only the claim whose evidence changed needs another check.


@pytest.mark.asyncio
async def test_discovery_chronology_only_attaches_to_the_report_that_was_checked(monkeypatch):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    chat = AsyncMock(return_value=ChatResponse(content='{"claim_kind":"paper_fact","scope_status":"requested","all_assertions_supported":true,"no_unstated_assumptions":true,"all_citations_contribute":true,"supports":true,"rationale":"fixture"}', model="fake", finish_reason="stop"))
    monkeypatch.setattr(verifier, "get_chat_client", lambda _: SimpleNamespace(chat=chat))
    answer = "Current report [report:current#1]. Background [arxiv:older#2]."
    await verifier.verify_citations(answer, _parse_citations(answer), {
        1: EvidenceChunk(paper_id="report:current", chunk_id=1, text="Current mechanism"),
        2: EvidenceChunk(paper_id="arxiv:older", chunk_id=2, text="Older mechanism"),
    }, discovery_context={"attempted":True, "as_of":"2026-09-15", "current_report_ids":["report:current"],
        "publisher_check":{"status":"complete", "release_date":"2026-09-10", "latest_title":"Current report",
                           "index_urls":["https://www.deepseek.com/en/news/"]}})
    contexts = [json.loads(call.args[0][1].content.split("Source discovery provenance (bibliographic scope only):\n")[1].split("\n\nCited evidence set:")[0])
                for call in chat.await_args_list]
    assert sum(bool(context) for context in contexts) == 1
    assert next(context for context in contexts if context)["report:current"]["checked_at"] == "2026-09-15"
