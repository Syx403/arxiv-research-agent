"""One deterministic decision point for acquisition, reading and delivery.

The model supplies semantic judgments and query proposals. This module owns action
admission, progress detection and stops; no extra planning-model call per edge.
"""

from datetime import UTC, datetime
import hashlib
import json

from src.core.research_policy import ResearchPolicy
from src.core.trace import record_event
from src.core.run_context import remaining_turn_s
from src.retrieval.paper_search import Query


def fingerprint(candidates):
    return hashlib.sha256(
        json.dumps(
            sorted(candidates, key=lambda p: p["paper_id"]), sort_keys=True, default=str
        ).encode()
    ).hexdigest()


def search_remaining(state):
    policy = ResearchPolicy.model_validate(state["research_policy"])
    started = datetime.fromisoformat(state["paper_search"]["as_of"])
    return max(0, policy.timeout_s - (datetime.now(UTC) - started).total_seconds())


def reading_remaining(state):
    """Use the runtime's actual deadline; saved timestamps support direct graph clients."""
    remaining = remaining_turn_s()
    if remaining is not None:
        return remaining
    policy = ResearchPolicy.model_validate(state.get("research_policy", {}))
    started = state.get("paper_search", {}).get("as_of")
    elapsed = (datetime.now(UTC) - datetime.fromisoformat(started)).total_seconds() if started else 0
    return max(0.0, policy.reading_timeout_s - elapsed)


def recovery_windows(state):
    """Observed stage times are admission estimates, not latency guarantees."""
    policy = ResearchPolicy.model_validate(state.get("research_policy", {}))
    timings = state.get("node_timings_s", {})
    verify = max(policy.min_optional_recovery_s, timings.get("self_rag", 0) * 1.2 + 2)
    repair = verify + max(5, timings.get("synthesize", 0) * 1.2)
    optional = max(0.0, min(policy.max_evidence_recovery_s, reading_remaining(state) - verify - 2))
    return {"verification_reserve_s": verify, "repair_and_verify_s": repair, "recovery_limit_s": optional}


def recovery_actions(state):
    """Bounded field relaxation, alternate ranking and pagination; no entity list."""
    rows = state["paper_search"].get("queries", [])
    proposed = []
    for row in rows:
        if row.get("status") != "ok" or not row.get("terms"):
            continue
        q = Query(
            terms=row["terms"], scope=row["scope"], sort=row["sort"], offset=row.get("offset", 0), match=row.get("match", "phrase")
        )
        # Relax adjacency, not meaning: dropping the first word could remove
        # the actual task modifier, such as "parallel" or "no".
        if row.get("results", 0) < 10 and q.match == "phrase" and any(len(t.split()) > 1 for t in q.terms):
            proposed.append((q.model_copy(update={"match": "words", "offset": 0}), "compound_phrase_relaxed"))
        if row.get("results") == 0:
            if q.scope == "title":
                proposed.append(
                    (
                        q.model_copy(update={"scope": "title_abstract", "offset": 0}),
                        "empty_title_relaxed",
                    )
                )
            elif len(q.terms) > 1:
                proposed.append(
                    (
                        q.model_copy(
                            update={"terms": q.terms[:-1], "sort": "relevance", "offset": 0}
                        ),
                        "empty_conjunction_broadened",
                    )
                )
    proposed += [
        (Query.model_validate(q), "model_proposed") for q in state.get("pending_paper_queries", [])
    ]
    for row in rows:
        if row.get("status") != "ok" or row.get("results", 0) < 10 or not row.get("terms"):
            continue
        q = Query(
            terms=row["terms"], scope=row["scope"], sort=row["sort"], offset=row.get("offset", 0), match=row.get("match", "phrase")
        )
        proposed.append(
            (
                q.model_copy(
                    update={"sort": "relevance" if q.sort == "recent" else "recent", "offset": 0}
                ),
                "alternate_ranking",
            )
        )
        if q.offset < 40:
            proposed.append((q.model_copy(update={"offset": q.offset + 10}), "next_page"))
    done = {row["key"] for row in rows}
    unique = []
    for query, reason in proposed:
        if query.key() not in done:
            unique.append((query, reason))
            done.add(query.key())
    return unique


async def control_research_node(state):
    policy = ResearchPolicy.model_validate(state["research_policy"])
    report = dict(state.get("paper_search", {}))
    phase = state.get("completed_node", "")
    updates = {}
    action, reason = "present_papers", "search_complete"
    if state.get("forced_stop_reason"):
        action = "finalize" if state.get("result_kind") == "evidence" else "present_papers"
        reason = state["forced_stop_reason"]
    elif state["paper_plan"].get("clarification"):
        reason = "clarification_required"
    elif phase == "plan_papers":
        action, reason = "search_papers", "initial_arxiv_acquisition"
    elif phase == "read_papers":
        if state.get("related_search") and not state.get("reading_added_count"):
            action, reason = "synthesize", "no_new_original_evidence"
        else:
            action = "retrieve" if state.get("retrieval_paper_ids") else "present_papers"
            reason = "read_selected_sources" if action == "retrieve" else "full_text_unavailable"
    elif phase == "retrieve":
        gaps = any(
            not r.sufficient
            and r.stop_reason not in {"relevance_judgment_failed", "sufficiency_judgment_failed"}
            for r in state.get("subq_results", {}).values()
        )
        if (
            gaps
            and state["paper_plan"].get("allow_related")
            and not state.get("related_expansion_attempted")
            and len(state.get("retrieval_paper_ids", [])) < policy.max_papers
            and len(report.get("queries", [])) < policy.max_queries
            and search_remaining(state) > 5
        ):
            action, reason = "expand_reading", "unmet_reading_evidence"
        else:
            action, reason = "synthesize", "deliver_scoped_evidence"
    elif phase == "self_rag":
        from src.graph.edges import route_after_verify

        action = route_after_verify(state)
        remaining, windows = reading_remaining(state), recovery_windows(state)
        if action == "synthesize":
            if not state.get("verification_recovery_attempted") and windows["recovery_limit_s"] >= 5:
                action = "recover_evidence"
            elif remaining < windows["repair_and_verify_s"]:
                action = "finalize"
                updates["forced_stop_reason"] = "verification_time_guard"
                report["optional_recovery_skipped"] = {"remaining_s": round(remaining, 3), **windows}
        reason = "verification_repair" if action == "synthesize" else "verification_finished"
        if action == "recover_evidence":
            reason = "checked_draft_saved_before_optional_recovery"
        elif updates.get("forced_stop_reason"):
            reason = updates["forced_stop_reason"]
    elif phase == "recover_evidence":
        from src.graph.edges import route_after_verify

        action = (("synthesize" if state.get("verification_recovery_requires_rewrite") else "self_rag")
                  if state.get("verification_recovery_changed") else route_after_verify(state))
        remaining, windows = reading_remaining(state), recovery_windows(state)
        required = windows["verification_reserve_s"] if action == "self_rag" else windows["repair_and_verify_s"]
        if action in {"self_rag", "synthesize"} and remaining < required:
            action = "finalize"
            updates["forced_stop_reason"] = "verification_time_guard"
            report["optional_recovery_skipped"] = {"remaining_s": round(remaining, 3), **windows}
        reason = (("reverify_citation_bindings" if not state.get("verification_recovery_attempted")
                   else "reverify_expanded_evidence") if action == "self_rag" else
                  "rebind_new_evidence_before_verification" if state.get("verification_recovery_requires_rewrite")
                  else "no_new_verification_evidence")
        if updates.get("forced_stop_reason"):
            reason = updates["forced_stop_reason"]
    elif phase == "expand_reading":
        action = "search_papers" if state.get("pending_paper_queries") else "synthesize"
        reason = "related_evidence_search" if action == "search_papers" else "no_new_search_action"
    elif (state["paper_plan"].get("named_papers") and state["paper_plan"].get("read_full_text")
          and not state["paper_plan"].get("paper_ids")
          and not state.get("related_search")):
        resolution = report.get("named_resolution", {})
        missing = resolution.get("missing", []) + resolution.get("ambiguous", [])
        actions = recovery_actions(state) if missing else []
        available = policy.max_queries - len(report.get("queries", []))
        if actions and available > 0 and search_remaining(state) > 1:
            selected = actions[:min(2, available)]
            updates["pending_paper_queries"] = [q.model_dump() for q, _ in selected]
            action, reason = "search_papers", "resolve_selected_names"
        elif resolution.get("ids"):
            action, reason = "read_papers", "preserve_selected_comparison_objects"
        else:
            updates["forced_stop_reason"] = "specified_paper_unavailable"
            reason = "selected_names_unresolved"
    elif (
        state["paper_plan"].get("paper_ids")
        and state["paper_plan"].get("read_full_text")
        and not state.get("related_search")
    ):
        action, reason = "read_papers", "user_selected_identity"
    elif (
        phase == "search_papers"
        and state.get("paper_candidates")
        and fingerprint(state["paper_candidates"]) != state.get("assessed_fingerprint")
    ):
        action, reason = "assess_papers", "new_candidate_evidence"
    else:
        results = state.get("paper_results", [])
        all_failed = report.get("queries") and all(
            q.get("status") != "ok" for q in report["queries"]
        )
        enough = (sum(p.get("fit") == "direct" for p in results)
                  >= state["paper_plan"].get("min_results", 1)
                  and report.get("coverage") == "sufficient")
        actions = recovery_actions(state) if not enough else []
        available = policy.max_queries - len(report.get("queries", []))
        can_search = (
            not all_failed
            and available > 0
            and len(state.get("paper_candidates", [])) < policy.max_candidates
            and report.get("assessments", 0) < policy.max_assessments
            and search_remaining(state) > 1
        )
        if actions and can_search:
            # Reserve one slot for deterministic recovery; model proposals cannot starve it.
            program = [a for a in actions if a[1] != "model_proposed"]
            semantic = [a for a in actions if a[1] == "model_proposed"]
            selected = ([program[0]] + semantic[:1]) if program and semantic else actions[:2]
            selected = selected[:available]
            action, reason = "search_papers", ",".join(why for _, why in selected)
            updates["pending_paper_queries"] = [q.model_dump() for q, _ in selected]
        else:
            reason = (
                "external_search_unavailable"
                if all_failed
                else "search_complete"
                if enough
                else "search_limit"
                if actions
                else "search_exhausted"
            )
            updates["pending_paper_queries"] = []
            report["stop_reason"] = reason
            report["coverage"] = "sufficient" if enough else "incomplete"
            if state["paper_plan"].get("read_full_text") and results:
                action = "read_papers"
            elif state.get("related_search") and state.get("retrieval_paper_ids"):
                action = "synthesize"
    decisions = list(report.get("decisions", []))
    decisions.append(
        {
            "from": phase,
            "action": action,
            "reason": reason,
            "queries": len(report.get("queries", [])),
            "candidates": len(state.get("paper_candidates", [])),
        }
    )
    report["decisions"] = decisions
    record_event("research_decision", **decisions[-1])
    return {**updates, "next_node": action, "paper_search": report}
