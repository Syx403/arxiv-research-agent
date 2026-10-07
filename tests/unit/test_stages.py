"""The stage table encodes the confirmed allocation (D10); these tests keep its principles true."""

from ara.llm.stages import EFFORTS, STAGES


def test_every_stage_uses_an_effort_its_provider_accepts() -> None:
    for stage in STAGES.values():
        assert stage.effort in EFFORTS[stage.provider], stage.name


def test_cache_breakpoints_are_set_only_for_openai() -> None:
    for stage in STAGES.values():
        assert stage.provider == "openai" or not stage.breakpoints, stage.name


def test_the_checker_differs_from_the_writer() -> None:
    assert STAGES["verify"].provider != STAGES["synthesize"].provider


def test_a_judge_never_shares_a_family_with_what_it_grades() -> None:
    assert STAGES["judge_answer"].provider != STAGES["synthesize"].provider  # writes answers
    assert STAGES["judge_relevance"].provider != STAGES["screen"].provider  # selects papers
