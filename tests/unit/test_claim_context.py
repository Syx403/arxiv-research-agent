"""Claim context is conditional evidence: unverified ancestors cannot authorize delivery."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from src.core.types import ChatResponse, CitationVerdict, EvidenceChunk, VerificationReport
from src.graph.delivery import supported_text
from src.graph.nodes.self_rag import _preserve_unaffected_facts
from src.graph.nodes.synthesize import _parse_citations
from src.retrieval.self_rag import verifier


ANSWER = (
    "For this workload, A fits better than B [a#1][b#2]. "
    "A, the better fit, uses a dependency queue [a#1]. "
    "This queue receives task outputs [a#3]."
)
LOOKUP = {
    1: EvidenceChunk(paper_id="a", chunk_id=1, text="A schedules dependencies."),
    2: EvidenceChunk(paper_id="b", chunk_id=2, text="B stores observations."),
    3: EvidenceChunk(paper_id="a", chunk_id=3, text="A's queue receives outputs."),
}


def scope_result():
    return verifier._AnswerScope(
        items=[
            verifier._ScopeItem(
                id=i,
                scope_status="requested",
                reason="fixture",
                context_ids=[] if i == 0 else [i - 1],
            )
            for i in range(3)
        ],
        coverage=[verifier._Coverage(requirement_id=0, statement_ids=[0, 1, 2])],
    )


@pytest.mark.parametrize("base_supported", [True, False])
async def test_parallel_judgments_are_gated_by_all_antecedents(monkeypatch, base_supported):
    started = asyncio.Event()
    calls = []
    monkeypatch.setattr(verifier, "_answer_scope", AsyncMock(return_value=scope_result()))

    async def check(client, semaphore, answer, citation, lookup, **kwargs):
        calls.append((citation, kwargs))
        if len(calls) == 3:
            started.set()
        await started.wait()  # Dependencies must not serialize paid source calls.
        supported = base_supported if citation.claim_text.startswith("For this") else True
        return CitationVerdict(
            citation=citation,
            supports=supported,
            rationale="source result",
            claim_kind="paper_fact",
        )

    monkeypatch.setattr(verifier, "_verify_one_citation", check)
    report = await asyncio.wait_for(
        verifier.verify_citations(
            ANSWER,
            _parse_citations(ANSWER),
            LOOKUP,
            discovery_context={"research_question": "Compare mechanisms"},
        ),
        1,
    )
    assert report.passed is base_supported and len(calls) == 3
    assert calls[1][1]["contextual_claims"][0]["text"].startswith("For this workload")
    assert len(calls[2][1]["contextual_claims"]) == 2
    if base_supported:
        assert "dependency queue" in supported_text(ANSWER, report)
    else:
        assert not supported_text(ANSWER, report)
        assert all(not v.supports for v in report.verdicts)
        assert all(not v.context_verified for v in report.verdicts[2:])


async def test_context_evidence_changes_invalidate_dependents_not_just_local_source(monkeypatch):
    monkeypatch.setattr(verifier, "_answer_scope", AsyncMock(return_value=scope_result()))

    async def source(*args, **kwargs):
        return CitationVerdict(
            citation=args[3], supports=True, rationale="source", claim_kind="paper_fact"
        )

    check = AsyncMock(side_effect=source)
    monkeypatch.setattr(verifier, "_verify_one_citation", check)
    options = {"discovery_context": {"research_question": "Compare"}}
    first = await verifier.verify_citations(ANSWER, _parse_citations(ANSWER), LOOKUP, **options)
    await verifier.verify_citations(
        ANSWER, _parse_citations(ANSWER), LOOKUP, cached_verdicts=first.verdicts, **options
    )
    assert check.await_count == 3
    changed = {
        **LOOKUP,
        2: LOOKUP[2].model_copy(update={"text": "B actually supports dependency scheduling too."}),
    }
    await verifier.verify_citations(
        ANSWER, _parse_citations(ANSWER), changed, cached_verdicts=first.verdicts, **options
    )
    assert check.await_count == 6  # The ranking and both transitive dependents need rechecking.


async def test_unrelated_prose_repair_reuses_independent_claims_and_checks_new_claim(monkeypatch):
    scope = scope_result()
    for item in scope.items:
        item.context_ids = []
    monkeypatch.setattr(verifier, "_answer_scope", AsyncMock(return_value=scope))
    check = AsyncMock(side_effect=lambda *a, **k: CitationVerdict(
        citation=a[3], supports=True, rationale="fixture", claim_kind="paper_fact"))
    monkeypatch.setattr(verifier, "_verify_one_citation", check)
    answer = "A has a queue [a#1]. B stores observations [b#2]. A returns outputs [a#3]."
    options = {"discovery_context": {"research_question": "Explain A and B"}}
    first = await verifier.verify_citations(answer, _parse_citations(answer), LOOKUP, **options)
    changed = answer.replace("B stores observations", "B persists observations in an external store")
    second = await verifier.verify_citations(changed, _parse_citations(changed), LOOKUP,
                                            cached_verdicts=first.verdicts, **options)
    assert second.passed and check.await_count == 4
    assert check.call_args.args[3].claim_text.startswith("B persists")
    # Changing the antecedent must invalidate its entire dependency closure.
    scope.items[1].context_ids = [0]
    scope.items[2].context_ids = [1]
    await verifier.verify_citations(changed, _parse_citations(changed), LOOKUP,
                                   cached_verdicts=second.verdicts, **options)
    assert check.await_count == 6


async def test_scoped_source_prompt_excludes_unrelated_answer_and_keeps_declared_context(monkeypatch):
    scope = scope_result()
    scope.items[1].context_ids = []
    scope.items[2].context_ids = [0]
    monkeypatch.setattr(verifier, "_answer_scope", AsyncMock(return_value=scope))
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps({
        "supports": True, "rationale": "source", "claim_kind": "paper_fact", "scope_status": "requested",
        "all_assertions_supported": True, "no_unstated_assumptions": True, "all_citations_contribute": True,
    }), model="fixture", finish_reason="stop")
    monkeypatch.setattr(verifier, "get_chat_client", lambda _: client)
    answer = "A uses a dependency queue [a#1]. UNRELATED_SENTENCE [b#2]. This queue receives outputs [a#3]."
    await verifier.verify_citations(answer, _parse_citations(answer), LOOKUP,
                                   discovery_context={"research_question": "Explain A and B"})
    target = next(c.args[0][1].content for c in client.chat.await_args_list
                  if "Specific claim being verified:\nThis queue" in c.args[0][1].content)
    assert "UNRELATED_SENTENCE" not in target and "A uses a dependency queue" in target
    assert "Answer:\n" not in target


async def test_table_header_change_invalidates_scoped_check(monkeypatch):
    scope = verifier._AnswerScope(items=[verifier._ScopeItem(id=0, scope_status="requested", reason="row")], coverage=[])
    monkeypatch.setattr(verifier, "_answer_scope", AsyncMock(return_value=scope))
    check = AsyncMock(side_effect=lambda *a, **k: CitationVerdict(citation=a[3], supports=True, rationale="fixture"))
    monkeypatch.setattr(verifier, "_verify_one_citation", check)
    answer = "| Method | Width (bits) |\n| --- | --- |\n| A | 4 [a#1] |"
    options = {"discovery_context": {"research_question": "Show the width"}}
    first = await verifier.verify_citations(answer, _parse_citations(answer), LOOKUP, **options)
    changed = answer.replace("Width (bits)", "Number of formats")
    await verifier.verify_citations(changed, _parse_citations(changed), LOOKUP,
                                   cached_verdicts=first.verdicts, **options)
    assert check.await_count == 2


async def test_rewording_scope_reason_cannot_rejudge_an_unchanged_rejected_claim(monkeypatch):
    first_scope = scope_result()
    second_scope = first_scope.model_copy(deep=True)
    for item in second_scope.items:
        item.reason = 'Different phrasing of the same scope decision'
    second_scope.items[0].scope_status = 'context_needed'
    monkeypatch.setattr(verifier, '_answer_scope', AsyncMock(side_effect=[first_scope, second_scope]))
    source = AsyncMock(side_effect=lambda *a, **k: CitationVerdict(
        citation=a[3], supports=False, rationale='Original source does not establish the claim'))
    monkeypatch.setattr(verifier, '_verify_one_citation', source)
    first = await verifier.verify_citations(ANSWER, _parse_citations(ANSWER), LOOKUP,
        discovery_context={'research_question': 'Compare'})
    await verifier.verify_citations(ANSWER, _parse_citations(ANSWER), LOOKUP,
        cached_verdicts=first.verdicts, discovery_context={'research_question': 'Compare'})
    # The independent rejected ranking is reusable; context-failed dependents
    # may be reassessed, but cannot pass while the ranking still fails.
    assert sum(c.args[3].claim_text.startswith('For this') for c in source.await_args_list) == 1


def test_delivery_cannot_keep_a_dependent_after_its_context_is_removed():
    citations = _parse_citations(ANSWER)
    first = citations[0].claim_text
    dependent = next(c for c in citations if "better fit" in c.claim_text)
    report = VerificationReport(
        passed=True,
        verdicts=[
            CitationVerdict(
                citation=dependent, supports=True, rationale="conditional", context_claims=[first]
            )
        ],
    )
    assert not supported_text(ANSWER, report)


def test_evidence_recovery_never_preserves_context_dependent_facts_as_independent():
    citations = _parse_citations(ANSWER)
    report = VerificationReport(
        passed=True,
        verdicts=[
            CitationVerdict(
                citation=c,
                supports=True,
                rationale="source",
                claim_kind="paper_fact",
                context_claims=[]
                if c.claim_text.startswith("For this")
                else [citations[0].claim_text],
            )
            for c in citations
        ],
    )
    state = {
        "answer": ANSWER,
        "question": "Compare",
        "evidence_lookup": LOOKUP,
        "verification": report,
    }
    changed = {**LOOKUP, 2: LOOKUP[2].model_copy(update={"text": "New comparison evidence."})}
    update = _preserve_unaffected_facts(state, changed)
    assert update["verification"] is None and update["best_verified"] == {}


@pytest.mark.parametrize(
    "context_ids,cited", [([1], {0, 1}), ([2], {0, 1}), ([0, 0], {0, 1}), ([0], {1})]
)
async def test_scope_rejects_self_future_duplicate_or_uncited_context(
    monkeypatch, context_ids, cited
):
    raw = {
        "items": [
            {"id": 0, "scope_status": "requested", "reason": "base", "context_ids": []},
            {
                "id": 1,
                "scope_status": "requested",
                "reason": "dependent",
                "context_ids": context_ids,
            },
        ],
        "coverage": [{"requirement_id": 0, "statement_ids": [0, 1]}],
    }
    model = AsyncMock()
    model.chat.return_value = ChatResponse(
        content=json.dumps(raw), model="fixture", finish_reason="stop"
    )
    monkeypatch.setattr(verifier, "get_chat_client", lambda _: model)
    assert (
        await verifier._answer_scope("Compare", ["base", "dependent"], cited_statement_ids=cited)
        is None
    )
    assert model.chat.await_count == 2
