from __future__ import annotations

import pytest

from src.core.types import SufficiencyVerdict
from src.retrieval.index.types import Hit
from src.retrieval.self_rag import sufficiency_check


@pytest.mark.asyncio
async def test_survey_question_type_routes_to_survey_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_survey(*args, **kwargs) -> SufficiencyVerdict:
        return SufficiencyVerdict(sufficient=True, missing_aspects=[])

    async def fail_generic(*args, **kwargs) -> SufficiencyVerdict:
        raise AssertionError("generic path should not run for survey")

    monkeypatch.setattr(sufficiency_check, "_survey_sufficiency", fake_survey)
    monkeypatch.setattr(sufficiency_check, "_generic_sufficiency", fail_generic)

    verdict = await sufficiency_check.is_sufficient("What approaches exist?", [_hit()], question_type="survey")

    assert verdict.sufficient is True


@pytest.mark.asyncio
async def test_non_survey_question_type_routes_to_generic_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_survey(*args, **kwargs) -> SufficiencyVerdict:
        raise AssertionError("survey path should not run")

    async def fake_generic(*args, **kwargs) -> SufficiencyVerdict:
        return SufficiencyVerdict(sufficient=False, missing_aspects=["missing"])

    monkeypatch.setattr(sufficiency_check, "_survey_sufficiency", fail_survey)
    monkeypatch.setattr(sufficiency_check, "_generic_sufficiency", fake_generic)

    verdict = await sufficiency_check.is_sufficient("How does ReAct work?", [_hit()], question_type="multi_hop")

    assert verdict.missing_aspects == ["missing"]


def test_survey_coverage_threshold_and_missing_aspects() -> None:
    aspects = [_aspect(idx) for idx in range(5)]

    sufficient = sufficiency_check._survey_coverage_decision(aspects, [True, True, True, False, False], requery_round=0)
    insufficient = sufficiency_check._survey_coverage_decision(aspects, [True, True, False, False, False], requery_round=0)
    none_covered = sufficiency_check._survey_coverage_decision(aspects, [False, False, False, False, False], requery_round=0)

    assert sufficient.sufficient is True
    assert insufficient.sufficient is False
    assert insufficient.missing_aspects == ["description 2", "description 3", "description 4"]
    assert none_covered.missing_aspects == [f"description {idx}" for idx in range(5)]


@pytest.mark.asyncio
async def test_survey_decomposition_failure_falls_back_to_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken_decompose(*args, **kwargs):
        raise sufficiency_check._SurveyAspectDecompositionError("bad json")

    async def fake_generic(*args, **kwargs) -> SufficiencyVerdict:
        return SufficiencyVerdict(sufficient=False, missing_aspects=["generic fallback"])

    monkeypatch.setattr(sufficiency_check, "_get_or_decompose_survey_aspects", broken_decompose)
    monkeypatch.setattr(sufficiency_check, "_generic_sufficiency", fake_generic)

    verdict = await sufficiency_check.is_sufficient("What approaches exist?", [_hit()], question_type="survey")

    assert verdict.missing_aspects == ["generic fallback"]


@pytest.mark.asyncio
async def test_keyword_mention_does_not_prove_semantic_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_judge(*args, **kwargs) -> bool:
        return False

    monkeypatch.setattr(sufficiency_check, "_judge_aspect_with_llm", fail_judge)
    evidence = [Hit(chunk_id=1, paper_id="p", section="S", text="This chunk discusses tool APIs.", score=1.0)]

    covered = await sufficiency_check._is_aspect_covered(_aspect(0, keywords=["APIs"]), evidence)

    assert covered is False


def test_exhausting_requery_rounds_does_not_create_evidence() -> None:
    aspects = [_aspect(idx) for idx in range(5)]

    verdict = sufficiency_check._survey_coverage_decision(
        aspects,
        [False, False, False, False, False],
        requery_round=2,
    )

    assert verdict.sufficient is False
    assert verdict.missing_aspects == [f"description {idx}" for idx in range(5)]


def _aspect(idx: int, *, keywords: list[str] | None = None):
    return sufficiency_check._SurveyAspect(
        name=f"aspect {idx}",
        keywords=keywords or [f"keyword{idx}"],
        description=f"description {idx}",
    )


def _hit():
    return Hit(chunk_id=1, paper_id="p", section="", text="Research evidence", score=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("question_type", ["definition", "survey", "comparison"])
async def test_empty_evidence_never_spends_a_sufficiency_call(monkeypatch, question_type):
    def no_call(role):
        raise AssertionError("No evidence is deterministically insufficient")
    monkeypatch.setattr(sufficiency_check, "get_chat_client", no_call)
    result = await sufficiency_check.is_sufficient("question", [], question_type=question_type)
    assert not result.sufficient
    assert result.missing_aspects


@pytest.mark.asyncio
async def test_new_survey_on_same_thread_does_not_reuse_previous_aspects(monkeypatch):
    calls = []
    async def decompose(question):
        calls.append(question)
        return [_aspect(len(calls))]
    monkeypatch.setattr(sufficiency_check, "_SURVEY_ASPECT_CACHE", {})
    monkeypatch.setattr(sufficiency_check, "_decompose_survey_aspects", decompose)
    first = await sufficiency_check._get_or_decompose_survey_aspects("Agent memory?", question_id="same-thread")
    second = await sufficiency_check._get_or_decompose_survey_aspects("MoE routing?", question_id="same-thread")
    repeated = await sufficiency_check._get_or_decompose_survey_aspects("Agent memory?", question_id="same-thread")
    assert first != second
    assert first == repeated
    assert calls == ["Agent memory?", "MoE routing?"]
