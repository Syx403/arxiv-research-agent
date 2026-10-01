"""Local, opt-in operational traces; no credentials or model reasoning text."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter

_active_trace: ContextVar["LocalTrace | None"] = ContextVar("ara_local_trace", default=None)


class LocalTrace:
    def __init__(self, path: Path, *, run_id: str, mode: str,
                 on_event: Callable[[dict], None] | None = None):
        self.path, self.run_id, self.mode = path, run_id, mode
        self.on_event = on_event
        self.started, self.sequence = perf_counter(), 0
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("Choose a new trace path; existing trajectories are preserved.")

    def record(self, event: str, **fields) -> None:
        self.sequence += 1
        row = {
            "run_id": self.run_id, "mode": self.mode, "sequence": self.sequence,
            "time": datetime.now(UTC).isoformat(),
            "elapsed_s": round(perf_counter() - self.started, 3), "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        if self.on_event is not None:
            self.on_event(row)

    @contextmanager
    def activate(self):
        token = _active_trace.set(self)
        try:
            yield self
        finally:
            _active_trace.reset(token)


def record_event(event: str, **fields) -> None:
    trace = _active_trace.get()
    if trace is not None:
        trace.record(event, **fields)


def traced_node(name, operation):
    async def wrapped(state):
        record_event("node_start", node=name)
        started = perf_counter()
        try:
            update = await operation(state)
        except BaseException as exc:
            record_event("node_error", node=name, error_type=type(exc).__name__)
            raise
        combined = {**state, **update}
        report = combined.get("verification")
        record_event(
            "node_end", node=name,
            thread_id=combined.get("thread_id"),
            subquestions=[q.text for q in combined["decomposition"].sub_questions] if combined.get("decomposition") else [],
            evidence=[{"paper_id": h.paper_id, "chunk_id": h.chunk_id} for h in combined.get("evidence", [])],
            subquestion_results=[{
                "index": r.subq_index, "sufficient": r.sufficient,
                "missing_aspects": r.missing_aspects, "stop_reason": r.stop_reason,
            } for r in combined.get("subq_results", {}).values()],
            verification_passed=report.passed if report else None,
            verification_rejected=sum(not v.supports for v in report.verdicts) if report else 0,
            verdicts=[{
                "paper_id": v.citation.paper_id, "chunk_id": v.citation.chunk_id,
                "claim": v.citation.claim_text, "supports": v.supports, "rationale": v.rationale,
                "status": v.status,
            } for v in report.verdicts] if report and name == "self_rag" else [],
            uncited_claim_count=len(report.uncited_claims) if report else 0,
            generation_retries=combined.get("tool_iters", 0),
            status=combined.get("status", "running"), stop_reason=combined.get("stop_reason", ""),
            context_turns=combined.get("context_message_count", len(combined.get("conversation_context", []))),
            research_question=combined.get("research_question", combined.get("question", "")),
            answer=combined.get("answer") if name in {"synthesize", "finalize"} else None,
        )
        timings = {**state.get("node_timings_s", {}), name: round(perf_counter() - started, 3)}
        return {**update, "completed_node": name, "node_timings_s": timings}
    return wrapped
