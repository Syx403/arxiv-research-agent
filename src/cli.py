"""CLI for the same AgentRuntime used by the browser, plus an offline example."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
from uuid import UUID, uuid4

from src.core.citations import claim_text, evidence_key, parse_citations
from src.retrieval.index.types import Hit


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover arXiv papers or read their original evidence with ARA."
    )
    parser.add_argument(
        "question", nargs="?", help="A single research question (live model calls)."
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Show a hand-authored example; no network or database.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the answer and citation evidence as JSON."
    )
    parser.add_argument(
        "--skip-reflection",
        action="store_true",
        help=argparse.SUPPRESS,  # Compatibility only; the old reflection path was removed.
    )
    parser.add_argument("--session", type=UUID, help="Continue a prior CLI conversation UUID.")
    parser.add_argument("--main-effort", choices=("low", "high", "max"), default="low")
    parser.add_argument("--fast-effort", choices=("low", "high", "max"), default="low")
    args = parser.parse_args(argv)
    if args.example and args.question is not None:
        parser.error("Choose either a question or --example.")
    if not args.example and not (args.question or "").strip():
        parser.error("Provide a question, or use --example for an offline preview.")
    return args


def example_report() -> dict:
    fixture = json.loads((Path(__file__).parent / "examples" / "react.json").read_text())
    state = {
        "question": fixture["question"],
        "answer": fixture["answer"],
        "evidence": [Hit(**item) for item in fixture["evidence"]],
        "verification": None,
    }
    report = build_report(state, mode="example")
    report["note"] = fixture["note"]
    return report


def build_report(state: dict, *, mode: str = "live") -> dict:
    answer = state.get("answer") or ""
    verification = state.get("verification")
    evidence = {evidence_key(hit.paper_id, hit.chunk_id): hit for hit in state.get("evidence", [])}
    citations = []
    for citation in parse_citations(answer):
        key = evidence_key(citation.paper_id, citation.chunk_id)
        hit = evidence.get(key)
        verdict = (
            next((item for item in verification.verdicts if item.citation == citation), None)
            if verification
            else None
        )
        arxiv_id = citation.paper_id.removeprefix("arxiv:")
        url = (
            f"https://arxiv.org/abs/{arxiv_id}"
            if (citation.paper_id.startswith("arxiv:") and re.fullmatch(r"[\w.\-/]+", arxiv_id))
            else None
        )
        citations.append(
            {
                "key": key,
                "claim": claim_text(answer, citation),
                "resolved": hit is not None,
                "supports": verdict.supports if verdict else None,
                "rationale": verdict.rationale if verdict else "Not evaluated",
                "paper_url": url,
                "evidence": asdict(hit) if hit else None,
            }
        )
    return {
        "mode": mode,
        "thread_id": state.get("thread_id"),
        "question": state.get("question", ""),
        "answer": answer,
        "verification_passed": verification.passed if verification else None,
        "verification_note": verification.rationale if verification else "Not evaluated",
        "citations": citations,
    }


async def run_question(question: str, *, session_id=None, main_effort="low", fast_effort="low",
                       skip_reflection=False) -> dict:
    from src.agent import AgentRequest, AgentRuntime
    from src.graph import builder
    from src.core.research_policy import ResearchPolicy
    from src.llm.budget import RequestBudget
    from src.llm.registry import ReasoningSettings
    from src.llm.stages import StageSettings

    for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "ARA_EVAL_LANGSMITH_EXPORT"):
        os.environ[key] = "false"
    # Share the browser's existing cumulative allowance. CLI invocations never
    # create a fresh paid allowance or bypass admission by calling builder.run.
    budget = RequestBudget(2.5, path=ROOT / "data/eval_outputs/round1-budget.json",
                           cohere_trial=True, max_requests=3000)
    directory = ROOT / "data/cli"
    try:
        result = await AgentRuntime(directory, budget).query(
            AgentRequest(question=question, session_id=session_id or uuid4(),
                         reasoning=ReasoningSettings(main=main_effort, fast=fast_effort),
                         stages=StageSettings(profile="ui"), policy=ResearchPolicy.configured())
        )
        report = {**result.output, "mode": "live", "question": result.question,
                  "session_id": result.session_id, "turn_id": result.turn_id,
                  "cost": result.cost, "elapsed_s": result.elapsed_s,
                  "trace_path": result.trace_path}
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{result.turn_id}.json").write_text(result.model_dump_json(indent=2))
        return report
    finally:
        await builder.shutdown()


def render_text(report: dict) -> str:
    if report["mode"] == "live" and "status" in report:
        lines = [f"{report['status'].upper()} — {report.get('stop_reason', '')}",
                 "", "Question: " + report["question"], "", report.get("answer", "")]
        for item in report.get("paper_results", []):
            lines += ["", item.get("title", item.get("paper_id", "")), item.get("url", "")]
        for item in report.get("sources", []):
            lines += ["", f"[{item['paper_id']}#{item['chunk_id']}]", item.get("text", ""),
                      item.get("url", "")]
        lines += ["", "Session: " + report["session_id"], "Trace: " + report["trace_path"],
                  f"Time: {report['elapsed_s']:.1f}s; estimated cost: ${report['cost'].get('estimated_usd', 0):.6f}"]
        return "\n".join(lines)
    if report["mode"] == "example":
        status = "OFFLINE EXAMPLE — hand-authored; model verification not run"
    elif report["verification_passed"]:
        status = "CITATION CHECK PASSED — model-assessed support, not a correctness guarantee"
    else:
        status = "UNVERIFIED DRAFT — citation check did not pass"
    lines = [status, "", "Question: " + report["question"], "", report["answer"]]
    if report.get("note"):
        lines += ["", report["note"]]
    for item in report["citations"]:
        support = (
            "supported"
            if item["supports"] is True
            else ("unverified" if item["supports"] is False else "not evaluated")
        )
        lines += ["", f"[{item['key']}] — {support}", "Claim: " + item["claim"]]
        if item["evidence"]:
            lines.append("Evidence: " + item["evidence"]["text"])
        else:
            lines.append("Evidence: identifier not found in this run's retrieved chunks.")
        if item["paper_url"]:
            lines.append("Paper: " + item["paper_url"])
        lines.append("Check: " + item["rationale"])
    if report.get("thread_id"):
        lines += ["", "Stored checkpoint: " + report["thread_id"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = (
            example_report()
            if args.example
            else asyncio.run(run_question(args.question, session_id=args.session,
                                          main_effort=args.main_effort, fast_effort=args.fast_effort))
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Do not dump provider responses, connection strings, or credentials.
        print(
            f"Research run failed ({type(exc).__name__}). Check provider configuration, "
            "start the database with make db-up and run make db-init. See README.md.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_text(report))
    return 0 if args.example or report.get("status") in {"complete", "needs_input"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
