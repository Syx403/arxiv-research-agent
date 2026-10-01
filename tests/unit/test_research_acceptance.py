from pathlib import Path
from unittest.mock import AsyncMock
from contextlib import asynccontextmanager

import pytest
from pydantic import ValidationError

from src.eval.research_acceptance import TurnSpec, audit_sources, check_turn, load_suite


def spec(route="discover", **kwargs):
    return TurnSpec(question="fixture", route=route, rubric=["Ground claims in sources"], **kwargs)


def result(**kwargs):
    return {"status": "complete", "stop_reason": "papers_found",
            "paper_results": [{"paper_id": "arxiv:2408.15664", "published_at": "2024-08-28",
                               "date_confirmed": True}],
            "paper_search": {"queries": [{"scope": "identity"}]}, **kwargs}


def nodes(*names):
    return [{"event": "node_start", "node": name} for name in names]


def test_frozen_suite_has_task_variety_and_does_not_claim_independent_human_labels():
    suite = load_suite(Path("tests/fixtures/research_acceptance.json"))
    assert len(suite.cases) == 12
    assert sum(len(c.turns) for c in suite.cases) == 15
    assert sum(c.split == "calibration" for c in suite.cases) == 4
    assert "pending" in suite.label_origin


def test_mechanical_success_never_implies_semantic_success():
    checked = check_turn(spec(required_ids=["2408.15664"]), result(), [])
    assert checked["mechanical_pass"]
    assert checked["semantic_review"] == "pending"


def test_exact_paper_lookup_rejects_unnecessary_reading():
    checked = check_turn(spec(), result(), nodes("read_papers", "retrieve"))
    assert "unnecessary_full_text" in checked["failures"]


def test_clarification_cannot_search_before_asking():
    checked = check_turn(spec("clarify", allowed_statuses=["needs_input"]),
                         result(status="needs_input", stop_reason="clarification_required"), [])
    assert "acquisition_before_clarification" in checked["failures"]


def test_first_paper_followup_preserves_exact_version():
    s = spec("read", first_from_previous=True)
    prior = result(paper_results=[{"paper_id": "arxiv:2408.15664", "version_id": "2408.15664v1"}])
    out = result(sources=[{"paper_id": "arxiv:2408.15664v2"}])
    checked = check_turn(s, out, nodes("read_papers", "retrieve"), prior)
    assert "missing_required_paper:2408.15664v1" in checked["failures"]


@pytest.mark.parametrize("version,passes", [("2407.01489v1", True), ("2407.01489v2", False)])
def test_exact_discovery_checks_version_metadata(version, passes):
    out = result(paper_results=[{"paper_id": "arxiv:2407.01489", "version_id": version}])
    checked = check_turn(spec(required_ids=["2407.01489v1"], allowed_ids=["2407.01489v1"]), out, [])
    assert checked["mechanical_pass"] is passes
    assert checked["selected_version_ids"] == [version]


@pytest.mark.parametrize("cited,passes", [
    (["arxiv:2407.01489v1"], True),
    (["arxiv:2405.15793v3"], False),
    (["arxiv:2407.01489v2"], False),
    (["arxiv:2407.01489v1", "arxiv:2405.15793v3"], False),
])
def test_second_paper_followup_checks_identity_version_and_exclusive_scope(cited, passes):
    s = spec("read", previous_paper_position=2, only_previous_paper=True)
    prior = result(paper_results=[
        {"paper_id": "arxiv:2405.15793", "version_id": "2405.15793v3"},
        {"paper_id": "arxiv:2407.01489", "version_id": "2407.01489v1"},
    ])
    out = result(sources=[{"paper_id": pid} for pid in cited])
    checked = check_turn(s, out, nodes("read_papers", "retrieve"), prior)
    assert checked["mechanical_pass"] is passes
    # The reference is to the actual displayed order, not to a hard-coded paper.
    prior["paper_results"].reverse()
    checked = check_turn(s, out, nodes("read_papers", "retrieve"), prior)
    assert checked["mechanical_pass"] is (cited == ["arxiv:2405.15793v3"])


def test_missing_second_result_cannot_fall_back_to_the_first():
    checked = check_turn(
        spec("read", previous_paper_position=2),
        result(sources=[{"paper_id": "arxiv:2408.15664v1"}]),
        nodes("read_papers", "retrieve"), result(),
    )
    assert "previous_paper_missing" in checked["failures"]


@pytest.mark.parametrize("scope", [
    {"previous_paper_position": 0},
    {"previous_paper_position": 6},
    {"only_previous_paper": True},
    {"first_from_previous": True, "previous_paper_position": 2},
])
def test_invalid_reference_contract_is_rejected(scope):
    with pytest.raises(ValidationError):
        spec("read", **scope)


def test_generalization_cases_have_source_backing_without_reusing_old_questions():
    from src.eval.datasets.gold_questions import GOLD_QUESTIONS

    suite = load_suite(Path("tests/fixtures/research_generalization.json"))
    questions = [t.question.strip().casefold() for c in suite.cases for t in c.turns]
    old_questions = {q.question.strip().casefold() for q in GOLD_QUESTIONS}
    for name in ("research_acceptance", "research_quality"):
        old = load_suite(Path(f"tests/fixtures/{name}.json"))
        old_questions.update(t.question.strip().casefold() for c in old.cases for t in c.turns)
    assert len(questions) == len(set(questions))
    assert not set(questions) & old_questions
    for case in suite.cases:
        if any(t.route != "clarify" for t in case.turns):
            assert case.references, case.id
            assert all(r["url"].startswith("https://arxiv.org/") and r["check"] for r in case.references)
    assert {c.category for c in suite.cases if c.split == "acceptance"} == {
        "discovery", "reading", "conversation", "boundary",
    }
    # Model verification and assistant-authored rubrics must not become independent labels.
    assert "no independent human" in suite.label_origin


def test_recent_revision_is_not_a_recent_first_publication():
    checked = check_turn(spec(date_from="2026-01-01", date_to="2026-09-16"), result(), [])
    assert "date_violation:arxiv:2408.15664" in checked["failures"]


def test_recency_is_checked_within_fit_groups_and_only_direct_results_count():
    papers = [
        {"paper_id": "arxiv:2501.00001", "published_at": "2025-01-01", "date_confirmed": True, "fit": "direct"},
        {"paper_id": "arxiv:2601.00001", "published_at": "2026-01-01", "date_confirmed": True, "fit": "partial"},
    ]
    checked = check_turn(spec(recent_order=True, min_results=1, max_results=1), result(paper_results=papers), [])
    assert checked["mechanical_pass"]
    checked = check_turn(spec(min_results=2), result(paper_results=papers), [])
    assert "result_count" in checked["failures"]


@pytest.mark.parametrize("paper,text,fail", [
    ("arxiv:2408.15664v1", "abc\n[…]\nghi", False),
    ("arxiv:2408.15664v2", "abc\n[…]\nghi", True),
    ("arxiv:2408.15664v1", "fabricated", True),
])
async def test_source_audit_checks_original_identity_and_spans(monkeypatch, paper, text, fail):
    from src.core import db

    @asynccontextmanager
    async def connection():
        yield AsyncMock(fetch=AsyncMock(return_value=[{
            "id": 10, "paper_id": "arxiv:2408.15664v1", "text": "abcdefghi",
        }]))

    monkeypatch.setattr(db, "acquire_app", connection)
    errors = await audit_sources([{"chunk_id": 10, "paper_id": paper,
                                  "text": text, "source_spans": [[0, 3], [6, 9]]}])
    assert bool(errors) == fail
