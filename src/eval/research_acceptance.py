"""Source-backed task checks, separate from semantic answer assessment.

This suite never treats the agent's own verifier as an independent quality score.
Open discovery references are anchors, not an exhaustive relevant-paper set.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TurnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    route: Literal["clarify", "discover", "read"]
    required_ids: list[str] = Field(default_factory=list)
    allowed_ids: list[str] = Field(default_factory=list)
    first_from_previous: bool = False
    previous_paper_position: int | None = Field(default=None, ge=1, le=5)
    only_previous_paper: bool = False
    min_results: int = 0
    max_results: int = 5
    date_from: str | None = None
    date_to: str | None = None
    recent_order: bool = False
    allowed_statuses: list[str] = Field(default_factory=lambda: ["complete"])
    require_claim_audit: bool = False
    require_full_audit: bool = False
    rubric: list[str] = Field(min_length=1)

    @property
    def referenced_position(self) -> int | None:
        return self.previous_paper_position or (1 if self.first_from_previous else None)

    @model_validator(mode="after")
    def previous_reference(self):
        if self.first_from_previous and self.previous_paper_position is not None:
            raise ValueError("Specify only one previous-paper reference")
        if self.only_previous_paper and self.referenced_position is None:
            raise ValueError("An exclusive previous-paper scope requires a position")
        return self


class ResearchCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    split: Literal["calibration", "acceptance"]
    category: Literal["discovery", "reading", "conversation", "boundary"]
    turns: list[TurnSpec] = Field(min_length=1)
    references: list[dict[str, str]] = Field(default_factory=list)


class ResearchSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    checked_on: date
    label_origin: str
    scope: str
    cases: list[ResearchCase]

    @model_validator(mode="after")
    def identities(self):
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Duplicate case identifiers")
        for case in self.cases:
            if case.turns[0].referenced_position is not None:
                raise ValueError("A paper reference requires a previous turn")
            for turn in case.turns:
                if turn.route == "clarify" and turn.required_ids:
                    raise ValueError("Clarification cannot require retrieved papers")
        return self


def load_suite(path: Path) -> ResearchSuite:
    return ResearchSuite.model_validate_json(path.read_text())


def canonical(identifier: str) -> str:
    return identifier.removeprefix("arxiv:")


def same_paper(actual: str, expected: str) -> bool:
    a, b = canonical(actual), canonical(expected)
    return a == b if re.search(r"v\d+$", b) else re.sub(r"v\d+$", "", a) == b


def check_turn(spec: TurnSpec, turn: dict, events: list[dict], previous: dict | None = None):
    """Mechanical checks only. Passing still requires an evidence-based review."""
    failures = []
    nodes = [e["node"] for e in events if e.get("event") == "node_start"]
    papers = turn.get("paper_results", [])
    report = turn.get("paper_search", {})
    sources = turn.get("sources", [])
    selected = [p["paper_id"] for p in papers]
    selected_versions = [p.get("version_id") or p["paper_id"] for p in papers]
    cited = [s["paper_id"] for s in sources]
    expected = list(spec.required_ids)
    previous_pid = None
    if spec.referenced_position is not None:
        prior = (previous or {}).get("paper_results", [])
        if len(prior) < spec.referenced_position:
            failures.append("previous_paper_missing")
        else:
            p = prior[spec.referenced_position - 1]
            previous_pid = p.get("version_id") or p["paper_id"]
            expected.append(previous_pid)
    if turn.get("status") not in spec.allowed_statuses:
        failures.append("unexpected_terminal_status:" + str(turn.get("status")))
    if turn.get("stop_reason") in {"runtime_error", "budget_exhausted", "turn_timeout",
                                  "search_plan_invalid", "candidate_output_invalid"}:
        failures.append("execution_failure:" + turn["stop_reason"])
    if spec.require_claim_audit:
        review = turn.get("claim_review", {})
        if not review.get("targeted_followup") or review.get("status") != "complete" or not review.get("checked_sources"):
            failures.append("claim_audit_incomplete")
        if spec.require_full_audit and not review.get("all_available_material_checked"):
            failures.append("claim_audit_scope_incomplete")
    if spec.route == "clarify":
        if turn.get("stop_reason") != "clarification_required":
            failures.append("missing_clarification")
        if report.get("queries") or "read_papers" in nodes:
            failures.append("acquisition_before_clarification")
    elif spec.route == "discover":
        if not report.get("queries"):
            failures.append("no_external_acquisition")
        if "read_papers" in nodes or "retrieve" in nodes:
            failures.append("unnecessary_full_text")
        direct_count = sum(p.get("fit") != "partial" for p in papers)
        if not spec.min_results <= direct_count <= spec.max_results:
            failures.append("result_count")
    else:
        if "read_papers" not in nodes or "retrieve" not in nodes:
            failures.append("missing_original_reading")
        if not sources and turn.get("status") == "complete":
            failures.append("complete_without_original_evidence")
    target = cited if spec.route == "read" else selected_versions
    for pid in expected:
        if not any(same_paper(got, pid) for got in target):
            failures.append("missing_required_paper:" + pid)
    if spec.allowed_ids and any(
        not any(same_paper(pid, allowed) for allowed in spec.allowed_ids) for pid in target
    ):
        failures.append("outside_requested_papers")
    if spec.only_previous_paper and previous_pid and any(
        not same_paper(pid, previous_pid) for pid in target
    ):
        failures.append("outside_referenced_paper")
    base_ids = [re.sub(r"v\d+$", "", canonical(p)) for p in selected]
    if len(set(base_ids)) != len(base_ids):
        failures.append("duplicate_papers")
    dates = {"direct": [], "partial": []}
    for p in papers:
        published = str(p.get("published_at") or "")[:10]
        if spec.date_from or spec.date_to:
            if not p.get("date_confirmed"):
                failures.append("unconfirmed_required_date:" + p["paper_id"])
            elif (spec.date_from and published < spec.date_from) or (
                spec.date_to and published > spec.date_to
            ):
                failures.append("date_violation:" + p["paper_id"])
        if p.get("date_confirmed"):
            dates["partial" if p.get("fit") == "partial" else "direct"].append(published)
    if spec.recent_order and any(group != sorted(group, reverse=True) for group in dates.values()):
        failures.append("publication_order")
    # Only expose diagnostics; neither internal coverage nor verifier verdicts are gold labels.
    return {
        "mechanical_pass": not failures, "failures": failures,
        "semantic_review": "pending", "rubric": spec.rubric,
        "status": turn.get("status"), "stop_reason": turn.get("stop_reason"),
        "elapsed_s": turn.get("elapsed_s"), "cost": turn.get("cost"),
        "queries": len(report.get("queries", [])), "candidates": report.get("candidate_count", 0),
        "selected_ids": selected, "cited_ids": sorted(set(cited)),
        "selected_version_ids": selected_versions,
        "cache_hits": turn.get("cache_hits", 0), "read_details": report.get("read_details", []),
        "model_format_repairs": sum(e.get("event") == "paper_format" for e in events),
        "decisions": report.get("decisions", []),
    }


async def audit_sources(sources: list[dict]) -> list[str]:
    """Validate delivered source identity and exact spans against persisted originals."""
    if not sources:
        return []
    from src.core import db

    async with db.acquire_app() as conn:
        rows = await conn.fetch("SELECT chunk_id AS id, paper_id, text FROM chunks WHERE chunk_id=ANY($1::bigint[])",
                                list({s["chunk_id"] for s in sources}))
    lookup = {row["id"]: row for row in rows}
    failures = []
    for source in sources:
        row = lookup.get(source["chunk_id"])
        if row is None or row["paper_id"] != source["paper_id"]:
            failures.append(f"source_identity:{source['chunk_id']}")
            continue
        spans = source.get("source_spans", [])
        if any(not (0 <= start < end <= len(row["text"])) for start, end in spans):
            failures.append(f"source_range:{source['chunk_id']}")
            continue
        expected = "\n[…]\n".join(row["text"][start:end] for start, end in spans) if spans else row["text"]
        if source["text"] != expected:
            failures.append(f"source_text:{source['chunk_id']}")
    return failures


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)
