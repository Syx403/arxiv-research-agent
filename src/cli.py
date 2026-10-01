"""Single-question research CLI and a credential-free output example."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
from uuid import uuid4

from src.core.citations import claim_text, evidence_key, parse_citations
from src.retrieval.index.types import Hit


class ConfigurationError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question about the indexed LLM-agent papers."
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
        help="Skip assistant-message storage and semantic reflection (checkpoints still persist).",
    )
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


async def run_question(question: str, *, skip_reflection: bool) -> dict:
    from src.core.config import get_settings
    from src.graph import builder
    from src.llm.registry import REGISTRY

    settings = get_settings()
    fields = {
        "deepseek_native": "deepseek_api_key",
        "openai_native": "openai_api_key",
        "cohere_native": "cohere_api_key",
        "openrouter": "openrouter_api_key",
    }
    required = {fields[route.provider] for route in REGISTRY.values()}
    missing = [
        name.upper()
        for name in sorted(required)
        if not (getattr(settings, name) and getattr(settings, name).get_secret_value())
    ]
    if missing:
        raise ConfigurationError("Missing provider configuration: " + ", ".join(missing))
    try:
        result = await builder.run(
            question.strip(), thread_id=f"cli-{uuid4()}", skip_reflection=skip_reflection
        )
        return build_report(result)
    finally:
        await builder.shutdown()


def render_text(report: dict) -> str:
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
            else asyncio.run(run_question(args.question, skip_reflection=args.skip_reflection))
        )
    except ConfigurationError as exc:
        print(str(exc) + ". See README.md or use --example.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Do not dump provider responses, connection strings, or credentials.
        print(
            f"Research run failed ({type(exc).__name__}). Check provider configuration, "
            "run make db-init, and ingest the corpus. See README.md.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_text(report))
    return 0 if args.example or report["verification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
