import json

import pytest

from src.eval.agent_metrics import ndcg, score_turn, summarize
from src.llm.registry import get_route
from src.llm.stages import STAGES, StageSettings, resolve, use_stage_settings
from src.core.run_context import RunContext, use_run_context
from src.retrieval.paper_search import cache_key


def test_stage_profiles_do_not_disable_semantic_repairs():
    for profile in ("semantic_low", "critical_high", "semantic_high", "ui"):
        with use_stage_settings(StageSettings(profile=profile)):
            for stage in STAGES:
                role, route, thinking = resolve(stage, "fast", get_route("fast"), False)
                assert route.vendor_model == "deepseek-flash"
                assert thinking is (stage != "format")
                if stage == "semantic_repair" and profile in {"critical_high", "semantic_high"}:
                    assert role == "main" and route.reasoning_effort == "high"
    with use_stage_settings(StageSettings(profile="legacy")):
        assert resolve("verify", "main", get_route("main"), False)[2] is False


def test_model_cache_is_bound_to_stage_settings_and_date():
    from datetime import date

    with use_stage_settings(StageSettings(profile="semantic_low")):
        low = cache_key("plan", "fast", {"q": "question"})
    with use_stage_settings(StageSettings(profile="critical_high")):
        high = cache_key("plan", "fast", {"q": "question"})
        with use_run_context(RunContext(as_of=date(2024, 1, 1))):
            old = cache_key("plan", "fast", {"q": "question"})
    assert len({low, high, old}) == 3


def test_ndcg_does_not_reward_duplicates():
    grades = {"2312.04511": 3, "2305.18323": 1, "2305.13245": 0}
    assert ndcg(list(grades), grades, 3) == 1
    assert ndcg(["2305.18323", "2312.04511", "2305.13245"], grades, 3) == pytest.approx(0.7098097)
    assert ndcg(["2312.04511", "2312.04511", "2312.04511"], grades, 3) < 1


def test_legacy_fact_dictionary_is_not_mistaken_for_a_verified_result_contract():
    spec = {
        "expected_route": "read",
        "expected_status": "complete",
        "programmatic_checks": [],
        "gold": {"required_facts": {"heads": 3}},
        "review_only": [],
    }
    result = {
        "route": "read",
        "output": {"status": "complete", "sources": [{"paper_id": "arxiv:2305.13245v3"}]},
    }
    score = score_turn(spec, result)
    assert score["fact_graded_count"] == 0
    assert score["task_pass"] is None and score["content_review"] == "pending"
    result["output"]["structured_facts"] = {"heads": 2}
    score = score_turn(spec, result)
    assert score["fact_graded_count"] == 0 and score["task_pass"] is None


def test_exact_requested_version_and_missing_results_fail():
    spec = {
        "expected_route": "read",
        "expected_status": "complete",
        "programmatic_checks": [],
        "gold": {"required_version_ids": ["2407.01489v1"]},
        "review_only": [],
    }
    result = {
        "route": "read",
        "output": {"status": "complete", "sources": [{"paper_id": "arxiv:2407.01489v2"}]},
    }
    assert score_turn(spec, result)["reference_coverage"] == 0


def test_review_dataset_and_summary_have_explicit_denominators():
    from pathlib import Path

    suite = json.loads(Path("docs/eval_design/review_cases_v1.json").read_text())
    assert len(suite["cases"]) == 12
    assert sum(len(c["turns"]) for c in suite["cases"]) == 15
    empty = summarize([], {"legacy": 3})["profiles"]["legacy"]
    assert empty["executed_turns"] == 0 and empty["not_run_or_interrupted"] == 3
    assert empty["route_accuracy"] == {"value": None, "n": 0}
