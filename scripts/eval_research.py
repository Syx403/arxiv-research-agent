"""Run frozen task cases through the actual local chat API, without a paid judge.

Uses the UI's existing cumulative ledger. No key handling or ledger resets.
--max-spend is an additional conservative batch ceiling, not a new allowance.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import httpx

from src.eval.artifacts import code_fingerprint
from src.eval.research_acceptance import audit_sources, check_turn, load_suite, save_json

ROOT = Path(__file__).resolve().parents[1]


async def run(args):
    suite = load_suite(args.suite)
    cases = [c for c in suite.cases if (args.split == "all" or c.split == args.split)
             and (not args.case or c.id in args.case)]
    if not cases:
        raise ValueError("No selected cases")
    signature = {
        "code_sha256": code_fingerprint(),
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "case_ids": [c.id for c in cases], "reasoning": {"main": "low", "fast": "low"},
        "batch_ceiling_usd": args.max_spend,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    state_path = args.output / "progress.json"
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=15,
                                 headers={"x-ara-client": "local-ui"}) as client:
        async def get(url):
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

        status = await get("/api/status")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["signature"] != signature:
                raise ValueError("Code/cases/settings changed; preserve this run and choose a new directory")
            progress = json.loads(state_path.read_text())
        else:
            manifest = {"signature": signature, "started": datetime.now(UTC).isoformat(),
                        "budget_before": status["budget"], "policy": status["research_policy"],
                        "purpose": "targeted regression" if args.case else args.split,
                        "scope": suite.scope, "label_origin": suite.label_origin,
                        "quality_claim": "Mechanical checks plus separate source review; not a population accuracy estimate"}
            progress = {"cases": {}, "results": []}
            save_json(state_path, progress)
            save_json(manifest_path, manifest)

        async def persist():
            save_json(state_path, progress)
            after = (await get("/api/status"))["budget"]
            save_json(args.output / "summary.json", {
                "completed_turns": len(progress["results"]),
                "mechanical_passes": sum(r["mechanical_pass"] for r in progress["results"]),
                "semantic_review": "pending", "budget_before": manifest["budget_before"],
                "budget_after": after,
                "additional_committed_usd": after["committed_usd"] - manifest["budget_before"]["committed_usd"],
                "results": progress["results"],
            })

        for case in cases:
            saved = progress["cases"].setdefault(case.id, {"session_id": None, "request_ids": []})
            for index, spec in enumerate(case.turns):
                if any(r["case_id"] == case.id and r["turn_index"] == index for r in progress["results"]):
                    continue
                if code_fingerprint() != signature["code_sha256"]:
                    raise ValueError("Source changed during evaluation")
                status = await get("/api/status")
                if status["active_session"] not in {None, saved["session_id"]}:
                    raise RuntimeError("Another research turn is running; resume later")
                if saved["session_id"] is None:
                    response = await client.post("/api/sessions")
                    response.raise_for_status()
                    saved["session_id"] = response.json()["id"]
                    save_json(state_path, progress)
                if len(saved["request_ids"]) <= index:
                    # Even an unexpectedly deep route fits both ceilings. Do not reset or raise either.
                    headroom = max(status["research_policy"]["search_budget_usd"],
                                   status["research_policy"]["reading_budget_usd"])
                    used = status["budget"]["committed_usd"] - manifest["budget_before"]["committed_usd"]
                    if used + headroom > args.max_spend or (
                        status["budget"]["committed_usd"] + headroom > status["budget"]["limit_usd"]
                    ):
                        await persist()
                        print("Budget headroom reached; unfinished cases retained, not marked passed", flush=True)
                        return
                    saved["request_ids"].append(str(uuid4()))
                    save_json(state_path, progress)
                sid, rid = saved["session_id"], saved["request_ids"][index]
                session = await get(f"/api/sessions/{sid}")
                turn = next((t for t in session["turns"] if t["id"] == rid), None)
                if turn is None:
                    # Save ID before transmitting; on ambiguous failure resume with the same ID.
                    response = await client.post(f"/api/sessions/{sid}/messages", json={
                        "question": spec.question, "request_id": rid, "reasoning": signature["reasoning"],
                    })
                    response.raise_for_status()
                    turn = response.json()
                async with asyncio.timeout(330):
                    while turn["status"] == "running":
                        await asyncio.sleep(1)
                        session = await get(f"/api/sessions/{sid}")
                        turn = next(t for t in session["turns"] if t["id"] == rid)
                trace = ROOT / "data/ui/sessions/traces" / sid / f"{rid}.jsonl"
                events = [json.loads(line) for line in trace.read_text().splitlines()]
                previous = session["turns"][index - 1] if index else None
                result = check_turn(spec, turn, events, previous)
                source_failures = await audit_sources(turn.get("sources", []))
                result["failures"].extend(source_failures)
                result["mechanical_pass"] = not result["failures"]
                result.update(case_id=case.id, turn_index=index, session_id=sid, turn_id=rid,
                              split=case.split, trace=str(trace.relative_to(ROOT)))
                save_json(args.output / f"{case.id}-{index}.json", {"turn": turn, "checks": result})
                progress["results"].append(result)
                await persist()
                print(json.dumps({k: result[k] for k in ("case_id", "turn_index", "status", "elapsed_s",
                                                        "cost", "mechanical_pass", "failures")}, ensure_ascii=False), flush=True)
        await persist()
    from src.core import db
    await db.close_pools()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "tests/fixtures/research_acceptance.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["all", "calibration", "acceptance"], default="calibration")
    parser.add_argument("--case", action="append")
    parser.add_argument("--max-spend", type=float, default=.35)
    args = parser.parse_args()
    if not 0 < args.max_spend <= 2:
        parser.error("Use a positive batch ceiling within the existing allowance")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
