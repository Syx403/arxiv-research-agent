import argparse
import json
import pytest

from scripts import eval_agent
from src.agent.runtime import AgentRuntime
from src.llm.budget import RequestBudget


async def test_standalone_execute_resume_and_rescore_do_not_repeat_agent(tmp_path, monkeypatch):
    dataset = tmp_path / "suite.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": "ara-gold-review-1",
                "sources": {},
                "cases": [
                    {
                        "id": "one-case",
                        "source_refs": [],
                        "turns": [
                            {
                                "question": "Find a paper",
                                "expected_route": "discover",
                                "expected_status": "complete",
                                "gold": {},
                                "programmatic_checks": ["route"],
                                "review_only": [],
                            }
                        ],
                    }
                ],
            }
        )
    )
    calls = []

    async def runner(question, **kwargs):
        calls.append(question)
        return {
            "answer": "A result",
            "result_kind": "papers",
            "status": "complete",
            "stop_reason": "papers_found",
        }

    monkeypatch.setattr(
        eval_agent,
        "AgentRuntime",
        lambda directory, budget: AgentRuntime(directory, budget, runner=runner),
    )

    async def audit(sources):
        return []

    monkeypatch.setattr(eval_agent, "audit_sources", audit)
    shared_ledger = tmp_path / 'shared-budget.json'
    earlier = RequestBudget(1, path=shared_ledger, cohere_trial=True)
    earlier.reserve('deepseek_native', 'deepseek-flash', 100, 1000).settle({'prompt_tokens': 100, 'completion_tokens': 100})
    original_ledger = shared_ledger.read_bytes()
    args = argparse.Namespace(
        dataset=dataset,
        output=tmp_path / "out",
        case=None,
        profile=["legacy", "critical_high"],
        repeat=1,
        concurrency=2,
        budget=1,
        ledger=shared_ledger,
        as_of="2026-09-17",
        context="structured",
        materials=None,
        total_output_cap=None,
        execute=True,
        rescore=False,
        prepare_materials=False,
        phase_case=None,
        phase_profile=["legacy"],
    )
    await eval_agent.execute(args)
    assert len(calls) == 1
    args.phase_profile = None
    await eval_agent.execute(args)
    assert len(calls) == 2
    await eval_agent.execute(args)
    assert len(calls) == 2
    args.execute, args.rescore = False, True
    await eval_agent.execute(args)
    assert len(calls) == 2
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["profiles"]["legacy"]["executed_turns"] == 1
    assert summary["profiles"]["critical_high"]["executed_turns"] == 1
    assert (args.output / "summary.csv").read_text().startswith("profile,")
    assert shared_ledger.read_bytes() == original_ledger
    assert not (args.output / 'budget.json').exists()
    assert json.loads((args.output / 'manifest.json').read_text())['ledger_path'] == str(shared_ledger)


async def test_structured_cli_public_input_group_report_and_offline_rescore(tmp_path, monkeypatch):
    from tests.unit.test_native_structured_answer import spec, fixture_state
    from src.core.run_context import current_run
    turn = spec()
    turn["gold"]["private_sentinel"] = "do_not_send_this_to_agent"
    suite = {"schema_version": "ara-gold-review-1", "evaluation_group": "structured", "sources": {},
             "cases": [{"id": "typed", "source_refs": [], "turns": [turn]}]}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(suite))
    seen = []

    async def runner(question, **kwargs):
        assert "do_not_send_this_to_agent" not in question
        assert '"expected"' not in question and '"value": 3' not in question
        assert current_run().response_fields[0].name == "sensor_count"
        seen.append(question)
        return fixture_state()

    monkeypatch.setattr(eval_agent, "AgentRuntime", lambda d, b: AgentRuntime(d, b, runner=runner))
    async def audit(sources):
        return []
    monkeypatch.setattr(eval_agent, "audit_sources", audit)
    args = argparse.Namespace(dataset=path, output=tmp_path / "run", case=None, profile=["semantic_low"],
        repeat=1, concurrency=1, budget=.01, ledger=None, as_of="2026-09-17", context="structured",
        materials=None, total_output_cap=4096, execute=True, rescore=False, prepare_materials=False,
        phase_case=None, phase_profile=None, typed_observations=False)
    await eval_agent.execute(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["evaluation_group"] == "structured"
    assert summary["profiles"]["semantic_low"]["fact_accuracy"] == {"value": 1, "n": 1}
    assert summary["profiles"]["semantic_low"]["content_reviews_pending"] == 1
    saved = {p: p.read_bytes() for p in (args.output / "records").glob("*.json")}
    assert not (args.output / "budget.json").exists()  # no provider reservations in this fixture
    args.execute, args.rescore = False, True
    await eval_agent.execute(args)
    assert len(seen) == 1 and all(p.read_bytes() == b for p, b in saved.items())
    assert not (args.output / "budget.json").exists()
    assert summary == json.loads((args.output / "summary.json").read_text())
    assert "structured" in (args.output / "report.md").read_text()
    # Contradictory private/public schemas fail before runtime execution.
    suite["cases"][0]["turns"][0]["response_fields"][0]["kind"] = "boolean"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="scopes disagree"):
        eval_agent.load_dataset(path)
