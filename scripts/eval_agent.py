"""Evaluate the shared AgentRuntime, without starting the web UI.

Default is a dry run. --execute requires an explicit isolated batch budget.
--rescore reads saved outputs only; --prepare-materials makes no model calls.
"""

from __future__ import annotations
# ruff: noqa: E402 -- standalone imports without requiring an editable install

import argparse
import asyncio
from collections import Counter
import csv
from datetime import date, datetime, UTC
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.runtime import AgentRequest, AgentRuntime
from src.eval.agent_metrics import score_turn, summarize
from src.eval.answer_metrics import METRIC_VERSION, validate_fact_rules
from src.eval.artifacts import code_fingerprint
from src.eval.materials import FrozenMaterials, capture_materials, material_digest
from src.eval.research_acceptance import audit_sources, save_json
from src.llm.budget import RequestBudget
from src.llm.registry import ReasoningSettings
from src.llm.stages import Profile, StageSettings


def load_dataset(path):
    suite = json.loads(path.read_text())
    if suite.get("schema_version") != "ara-gold-review-1":
        raise ValueError("Unsupported dataset schema")
    cases = suite["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if any(ref not in suite["sources"] for ref in case["source_refs"]):
            raise ValueError("Unknown source reference")
        for spec in case["turns"]:
            validate_fact_rules(spec["gold"])
            request = AgentRequest(question=spec["question"], response_fields=spec.get("response_fields", []))
            group = suite.get("evaluation_group")
            if group == "natural" and request.response_fields:
                raise ValueError("Natural tasks cannot contain structured response requirements")
            if group == "structured":
                if not request.response_fields or spec["expected_route"] != "read":
                    raise ValueError("Structured calibration requires explicit reading fields")
                by_name = {f.name: f for f in request.response_fields}
                if set(spec["gold"].get("required_facts", {})) != set(spec["gold"].get("fact_rules", {})):
                    raise ValueError("Every structured fact group requires a scoring rule")
                for rule in spec["gold"]["fact_rules"].values():
                    for clause in rule["all_of"]:
                        if len(clause["names"]) != 1 or clause["names"][0] not in by_name:
                            raise ValueError("Structured rules must use an exact public field name")
                        f = by_name[clause["names"][0]]
                        if f.kind != clause["kind"] or f.paper_ids != (clause["paper_id"],):
                            raise ValueError("Public field and private rule scopes disagree")
                        f.validate_value_domain(clause["expected"])
            if not spec["question"].strip() or spec["expected_route"] not in {
                "read",
                "discover",
                "clarify",
            }:
                raise ValueError("Invalid question/route")
    return suite


def report(directory, records, planned, *, group="legacy"):
    summary = summarize(records, planned)
    summary["evaluation_group"] = group
    save_json(directory / "summary.json", summary)
    # Keep the newly computed per-turn grades accessible after an offline
    # rescore, without rewriting raw execution records or historical outputs.
    save_json(directory / "scores.json", {
        "metric_version": METRIC_VERSION,
        "scorer_code_sha256": code_fingerprint(),
        "turns": [{k: row.get(k) for k in ("case_id", "profile", "repeat", "turn_index", "score")}
                  for row in records],
    })
    with (directory / "summary.csv").open("w", newline="") as f:
        columns = [
            "profile",
            "evaluation_group",
            "executed_turns",
            "planned_turns",
            "program_passes",
            "route_accuracy",
            "reference_coverage",
            "comparison_original_coverage",
            "ndcg_at_3",
            "primary_selection_accuracy",
            "fact_accuracy",
            "fact_grading_coverage",
            "fact_correct",
            "fact_observable",
            "fact_graded",
            "fact_required",
            "content_reviews_pending",
            "runtime_failures",
            "terminal_p50_s",
            "terminal_p95_s",
            "estimated_usd",
            "uncertain_usd",
            "cache_hit_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for profile, row in summary["profiles"].items():
            writer.writerow(
                {
                    "profile": profile,
                    "evaluation_group": group,
                    **{
                        k: row[k]["value"] if isinstance(row[k], dict) else row[k]
                        for k in columns[2:]
                    },
                }
            )
    lines = [
        "# Agent evaluation",
        "",
        f"Evaluation group: **{group}**. Natural and structured tasks have different public inputs; compare model profiles within a group.",
        "",
        "Program checks and content review are separate. Ungraded facts are not correct facts.",
        "",
        "| Profile | Run / planned | Program pass | Route | References | nDCG@3 | Primary | Fact accuracy | Graded / required | Content pending | Cost USD | P50 seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, row in summary["profiles"].items():

        def metric(key):
            item = row[key]
            return f"{item['value']:.3f} (n={item['n']})" if item["value"] is not None else "N/A"

        lines.append(
            f"| {profile} | {row['executed_turns']}/{row['planned_turns']} | {row['program_passes']} | {metric('route_accuracy')} | {metric('reference_coverage')} | {metric('ndcg_at_3')} | {metric('primary_selection_accuracy')} | {metric('fact_accuracy')} | {row['fact_graded']}/{row['fact_required']} | {row['content_reviews_pending']} | {row['estimated_usd']:.6f} + {row['uncertain_usd']:.6f} uncertain | {row['terminal_p50_s']} |"
        )
    lines.extend([
        "", "Fact accuracy here is strict typed-field matching, including missing required fields as misses. "
        "It is not a semantic score for the whole answer. Field aliases are predeclared; unfamiliar equivalent "
        "encodings need review rather than post-hoc relabelling. See scores.json for each missing/mismatching field.",
        "", "| Case | Profile | Turn | Program | Typed facts correct / graded / required | nDCG@3 | Pending review | Failures |",
        "|---|---|---:|---|---:|---:|---|---|",
    ])
    for row in records:
        s = row.get("score")
        if s:
            lines.append(f"| {row['case_id']} | {row['profile']} | {row['turn_index'] + 1} | {s['program_pass']} | "
                         f"{s['fact_correct_count']}/{s['fact_graded_count']}/{s['fact_required_count']} | "
                         f"{s['ndcg_at_3'] if s['ndcg_at_3'] is not None else 'N/A'} | {s['content_review']} | "
                         + "; ".join(s["failures"]).replace("|", "/") + " |")
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return summary


async def execute(args):
    suite = load_dataset(args.dataset)
    group = suite.get("evaluation_group", "legacy")
    if group in {"natural", "structured"} and getattr(args, "typed_observations", False):
        raise ValueError("The split v5 groups do not use experimental free-text extraction")
    cases = [c for c in suite["cases"] if not args.case or c["id"] in args.case]
    if not cases or args.case and set(args.case) - {c["id"] for c in cases}:
        raise ValueError("Unknown or empty case selection")
    if args.phase_case and set(args.phase_case) - {c["id"] for c in cases}:
        raise ValueError("Phase cases must be part of the full experiment")
    if args.phase_profile and set(args.phase_profile) - set(args.profile):
        raise ValueError("Phase profiles must be part of the full experiment")
    if args.prepare_materials:
        if args.materials is None or args.materials.exists():
            raise ValueError("Choose a new --materials path")
        snapshot = await capture_materials(suite, [c["id"] for c in cases])
        save_json(args.materials, snapshot)
        print(
            json.dumps(
                {"material_digest": material_digest(snapshot), "papers": len(snapshot["catalog"])}
            )
        )
        return
    if args.rescore:
        manifest = json.loads((args.output / "manifest.json").read_text())
        if manifest["dataset_sha256"] != hashlib.sha256(args.dataset.read_bytes()).hexdigest():
            raise ValueError("Dataset changed; preserve old grades and use a new dataset/run")
        records = [
            json.loads(p.read_text()) for p in sorted((args.output / "records").glob("*.json"))
        ]
        # No DB/provider calls. Persisted independent source checks remain attached.
        for row in records:
            if row.get("result"):
                if "source_errors" not in row:
                    continue  # A missing source audit cannot be manufactured by offline rescore.
                case = next(c for c in suite["cases"] if c["id"] == row["case_id"])
                row["score"] = score_turn(
                    case["turns"][row["turn_index"]],
                    row["result"],
                    previous=row.get("previous"),
                    source_errors=row.get("source_errors"),
                )
        print(
            json.dumps(
                report(args.output, records, manifest["planned_turns"], group=group),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    planned = {
        profile: sum(len(c["turns"]) for c in cases) * args.repeat for profile in args.profile
    }
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "evaluation_group": group,
                    "public_response_fields": sum(len(t.get("response_fields", [])) for c in cases for t in c["turns"]),
                    "cases": [c["id"] for c in cases],
                    "planned_turns": planned,
                    "concurrency": args.concurrency,
                    "budget_usd": args.budget,
                    "materials": str(args.materials) if args.materials else "live arXiv",
                    "note": "No provider calls. Typed facts use frozen aliases and equality; open prose requires separate source review.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.budget is None:
        raise ValueError("--execute requires --budget; no implicit paid allowance")
    if args.materials is None and any(c.get("fixture_plan") for c in cases):
        raise ValueError("Outage fixtures require isolated frozen --materials")
    snapshot = json.loads(args.materials.read_text()) if args.materials else None
    signature = {
        "code_sha256": code_fingerprint(),
        "evaluation_group": group,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "case_ids": [c["id"] for c in cases],
        "profiles": args.profile,
        "repeat": args.repeat,
        "planned_turns": planned,
        "concurrency": args.concurrency,
        "budget_usd": args.budget,
        "as_of": args.as_of,
        "context_mode": args.context,
        "material_digest": material_digest(snapshot) if snapshot else None,
        "material_scope": "fixed candidate pool and indexed originals; not discovery recall"
        if snapshot
        else "live arXiv; source timing may vary",
        "total_output_cap": args.total_output_cap,
        "typed_observations": getattr(args, "typed_observations", False),
        "application_model_cache": "bypass",
        "ledger_path": str((getattr(args, "ledger", None) or args.output / "budget.json").resolve()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if {k: old.get(k) for k in signature} != signature:
            raise ValueError(
                "Inputs/config/code changed; preserve the experiment and use a new directory"
            )
        manifest = old
    else:
        manifest = {**signature, "run_id": str(uuid4()), "started": datetime.now(UTC).isoformat()}
        save_json(manifest_path, manifest)
    budget = RequestBudget(
        args.budget, path=Path(signature["ledger_path"]), cohere_trial=True, max_requests=1000
    )
    runtime = AgentRuntime(args.output, budget)
    records_dir = args.output / "records"
    records_dir.mkdir(exist_ok=True)
    records = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async def scenario(case, profile, repeat):
        async with semaphore:
            previous = None
            sid = str(uuid4())
            for index, spec in enumerate(case["turns"]):
                path = records_dir / f"{case['id']}-{profile}-{repeat}-{index}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    records.append(row)
                    if row.get("result"):
                        previous = row["result"]
                        sid = previous["session_id"]
                        continue
                    # An interrupted attempt can have a remote charge/checkpoint. Never silently replay it.
                    return
                if code_fingerprint() != signature["code_sha256"]:
                    raise ValueError("Code changed during experiment")
                if budget.snapshot()["committed_usd"] >= args.budget:
                    return
                acquisition = None
                if snapshot:
                    ids = [suite["sources"][r]["paper_id"] for r in case["source_refs"]]
                    frozen = FrozenMaterials(
                        snapshot, paper_ids=ids, unavailable=bool(case.get("fixture_plan"))
                    )
                    await frozen.validate()
                    acquisition = frozen.adapter()
                request = AgentRequest(
                    question=spec["question"],
                    session_id=sid,
                    run_id=manifest["run_id"],
                    reasoning=ReasoningSettings(main="low", fast="low"),
                    stages=StageSettings(profile=profile, total_output_cap=args.total_output_cap),
                    as_of=date.fromisoformat(args.as_of),
                    model_result_cache=False,
                    structured_context=args.context == "structured",
                    include_typed_observations=getattr(args, "typed_observations", False),
                    response_fields=spec.get("response_fields", []),
                )
                row = {
                    "case_id": case["id"],
                    "evaluation_group": group,
                    "profile": profile,
                    "repeat": repeat,
                    "turn_index": index,
                    "request": request.model_dump(mode="json"),
                    "execution_status": "running",
                }
                save_json(path, row)
                result = await runtime.query(request, acquisition=acquisition)
                result = result.model_dump(mode="json")
                row.update(result=result, previous=previous, execution_status="agent_finished")
                save_json(path, row)
                source_errors = await audit_sources(result["output"].get("sources", []))
                score = score_turn(spec, result, previous=previous, source_errors=source_errors)
                row.update(
                    result=result,
                    previous=previous,
                    source_errors=source_errors,
                    score=score,
                    execution_status="finished",
                )
                save_json(path, row)
                records.append(row)
                previous = result
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "profile": profile,
                            "turn": index,
                            "status": result["output"]["status"],
                            "program_pass": score["program_pass"],
                            "failures": score["failures"],
                            "content_review": score["content_review"],
                            "cost": result["cost"]["estimated_usd"],
                            "elapsed_s": result["elapsed_s"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if result["output"]["stop_reason"] in {
                    "runtime_error",
                    "budget_exhausted",
                    "budget_wait_timeout",
                    "turn_timeout",
                }:
                    break

    # Rotate profile order between cases/repeats to reduce systematic warm-cache bias.
    jobs = []
    for repeat in range(args.repeat):
        for i, case in enumerate(cases):
            shift = (i + repeat) % len(args.profile)
            order = args.profile[shift:] + args.profile[:shift]
            jobs.extend(
                scenario(case, profile, repeat)
                for profile in order
                if (not args.phase_case or case["id"] in args.phase_case)
                and (not args.phase_profile or profile in args.phase_profile)
            )
    outcomes = await asyncio.gather(*jobs, return_exceptions=True)
    # Include partial/interrupted records and their costs even if grading failed.
    records = [json.loads(p.read_text()) for p in sorted(records_dir.glob("*.json"))]
    summary = report(args.output, records, planned, group=group)
    ledger = budget.snapshot()
    save_json(
        args.output / "budget_summary.json", {k: v for k, v in ledger.items() if k != "requests"}
    )
    errors = Counter(type(e).__name__ for e in outcomes if isinstance(e, BaseException))
    print(
        json.dumps(
            {
                "profiles": summary["profiles"],
                "batch_errors": dict(errors),
                "cumulative_ledger_committed_usd": ledger["committed_usd"],
                "ledger_path": signature["ledger_path"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if errors:
        raise RuntimeError("Experiment has infrastructure errors; see preserved records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "docs/eval_design/natural_cases_v5.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/eval_outputs/agent-review")
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--phase-case",
        action="append",
        help="Execute a subset while preserving the full frozen experiment",
    )
    parser.add_argument("--phase-profile", nargs="+", choices=Profile.__args__)
    parser.add_argument(
        "--profile", nargs="+", choices=Profile.__args__, default=["legacy", "critical_high"]
    )
    parser.add_argument("--repeat", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--ledger", type=Path, help="Reuse an existing cumulative budget; --budget is the ledger's total ceiling, not a new allowance")
    parser.add_argument("--as-of", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--context", choices=["legacy", "structured"], default="structured")
    parser.add_argument("--total-output-cap", type=int)
    parser.add_argument("--typed-observations", action="store_true",
                        help="Opt into experimental fact/ranking extraction inside the source verifier")
    parser.add_argument("--materials", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--rescore", action="store_true")
    mode.add_argument("--prepare-materials", action="store_true")
    args = parser.parse_args()
    if args.budget is not None and not 0 < args.budget <= 100:
        parser.error("Budget must be explicit and positive, at most $100")
    if len(set(args.profile)) != len(args.profile):
        parser.error("Profiles must be unique")

    async def run_and_close():
        try:
            await execute(args)
        finally:
            from src.graph.builder import shutdown

            await shutdown()

    asyncio.run(run_and_close())


if __name__ == "__main__":
    main()
