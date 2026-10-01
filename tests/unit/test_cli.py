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
async def test_live_cli_closes_resources_and_marks_verification_failure(monkeypatch):
    from src.graph import builder
    from src.core import config

    settings = config.Settings(
        _env_file=None, DEEPSEEK_API_KEY="test", OPENAI_API_KEY="test", COHERE_API_KEY="test"
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    shutdown_calls = []

    async def run(question, *, thread_id, skip_reflection):
        assert question == "Q"
        assert thread_id.startswith("cli-")
        assert skip_reflection is True
        return {
            "question": question,
            "thread_id": thread_id,
            "answer": "No evidence.",
            "verification": VerificationReport(verdicts=[], passed=False),
        }

    async def shutdown():
        shutdown_calls.append(True)

    monkeypatch.setattr(builder, "run", run)
    monkeypatch.setattr(builder, "shutdown", shutdown)
    report = await cli.run_question(" Q ", skip_reflection=True)
    assert shutdown_calls == [True]
    assert report["verification_passed"] is False


def test_failed_check_has_distinct_exit_code(monkeypatch, capsys):
    async def fake_run(*args, **kwargs):
        return {
            "mode": "live",
            "question": "Q",
            "answer": "Unverified.",
            "verification_passed": False,
            "verification_note": "NO_CITATIONS",
            "citations": [],
            "thread_id": "test",
        }

    monkeypatch.setattr(cli, "run_question", fake_run)
    assert cli.main(["Q"]) == 2
    assert "UNVERIFIED DRAFT" in capsys.readouterr().out
