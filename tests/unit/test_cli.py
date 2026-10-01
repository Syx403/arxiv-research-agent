from __future__ import annotations

import json
import subprocess
import sys

import pytest

from src import cli
from src.core.citations import parse_citations
from src.core.types import CitationVerdict, VerificationReport
from src.retrieval.index.types import Hit


def test_example_resolves_a_real_paper_without_claiming_a_model_run():
    report = cli.example_report()
    assert report["mode"] == "example"
    assert report["verification_passed"] is None
    assert report["citations"][0]["resolved"] is True
    assert report["citations"][0]["supports"] is None
    assert report["citations"][0]["paper_url"] == "https://arxiv.org/abs/2210.03629"
    assert "illustrative" in report["note"]


def test_report_preserves_failed_citation_in_the_draft_for_inspection():
    answer = "An unsupported claim [p#7]."
    citation = parse_citations(answer)[0]
    report = cli.build_report(
        {
            "question": "Q",
            "answer": answer,
            "citations": [],  # graph retains only supported citations
            "evidence": [
                Hit(chunk_id=7, paper_id="p", text="Different fact.", score=1, section="Body")
            ],
            "verification": VerificationReport(
                passed=False,
                verdicts=[
                    CitationVerdict(citation=citation, supports=False, rationale="Unsupported.")
                ],
            ),
        }
    )
    assert report["citations"][0]["supports"] is False
    assert "UNVERIFIED DRAFT" in cli.render_text(report)


def test_cli_example_runs_as_documented():
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--example", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["mode"] == "example"


@pytest.mark.parametrize("argv", [[], ["   "], ["--example", "a live question"]])
def test_cli_rejects_ambiguous_or_empty_input(argv):
    with pytest.raises(SystemExit) as error:
        cli.parse_args(argv)
    assert error.value.code == 2


@pytest.mark.asyncio
async def test_live_cli_uses_shared_runtime_budget_and_session(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from uuid import uuid4
    from src import agent
    from src.graph import builder

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    session_id = uuid4()
    shutdown_calls = []

    class Runtime:
        def __init__(self, directory, budget):
            assert directory == tmp_path / "data/cli"
            assert budget.path == tmp_path / "data/eval_outputs/round1-budget.json"
            assert budget.limit_usd == 2.5

        async def query(self, request):
            assert request.question == "Q"
            assert request.session_id == session_id
            assert request.stages.profile == "ui"
            assert request.reasoning.main == request.reasoning.fast == "low"
            return SimpleNamespace(output={"status": "incomplete", "answer": ""},
                                   question=request.question, session_id=str(session_id),
                                   turn_id="test-turn", cost={"requests": 0}, elapsed_s=0,
                                   trace_path="test-trace", model_dump_json=lambda **kw: '{}')

    async def shutdown():
        shutdown_calls.append(True)

    monkeypatch.setattr(agent, "AgentRuntime", Runtime)
    monkeypatch.setattr(builder, "shutdown", shutdown)
    report = await cli.run_question(" Q ", session_id=session_id)
    assert shutdown_calls == [True]
    assert report["status"] == "incomplete"
    assert (tmp_path / "data/cli/test-turn.json").exists()


@pytest.mark.parametrize(("status", "exit_code"), [
    ("complete", 0), ("needs_input", 0), ("partial", 2), ("incomplete", 2),
])
def test_live_cli_preserves_delivery_status(monkeypatch, capsys, status, exit_code):
    async def fake_run(*args, **kwargs):
        return {
            "mode": "live",
            "question": "Q",
            "answer": "Unverified.",
            "status": status,
            "session_id": "test", "trace_path": "trace", "cost": {}, "elapsed_s": 1,
        }

    monkeypatch.setattr(cli, "run_question", fake_run)
    assert cli.main(["Q"]) == exit_code
    assert status.upper() in capsys.readouterr().out
