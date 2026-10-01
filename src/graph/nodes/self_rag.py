from __future__ import annotations

import httpx
import asyncio

from src.core.trace import record_event
from src.core.types import VerificationReport
from src.graph.state import AgentState
from src.llm.budget import budget_stop_reason, BudgetExceeded
from src.llm.errors import EmptyProviderResponseError, LLMError
from src.retrieval.self_rag.verifier import verify_citations
from src.retrieval.self_rag.evidence_recovery import recover_evidence
from src.retrieval.self_rag.citation_repair import rebind_citations
from src.graph.delivery import supported_text, delivery_score
from src.graph.controller import recovery_windows


async def self_rag_node(state: AgentState) -> dict:
    if state.get("synthesis_format_degraded"):
        return {
            "verification": VerificationReport(
                verdicts=[],
                passed=False,
                rationale="synthesis_format_degraded",
            )
        }
    answer = state.get("answer") or ""
    citations = state.get("citations", [])
    lookup = state.get("evidence_lookup", {})
    previous = state.get("verification")
    cached = previous.verdicts if previous and state.get("last_verification_evidence") == lookup else []
    context = {**state.get("external_discovery", {}), "claim_review": state.get("claim_review", {}),
               "answer_requirements": state.get("paper_plan", {}).get("answer_requirements", []),
               "research_question": (state.get("research_question") or state.get("question", ""))
               + "\nOriginal user request: " + state.get("question", "")}
    verification = await verify_citations(answer, citations, lookup, cached_verdicts=cached,
                                          discovery_context=context)
    checked = supported_text(answer, verification)
    best = state.get("best_verified", {})
    question = state.get("research_question") or state.get("question", "")
    if checked and (best.get("question") != question or
                    delivery_score(checked) > delivery_score(best.get("text", ""))):
        best = {"question": question, "draft": answer, "text": checked,
                "verification": verification, "evidence_lookup": lookup,
                "answer_fields": state.get("answer_fields", [])}
    # Commit the initial judgment BEFORE any optional recovery can await a model.
    common = {"best_verified": best, "last_verification_evidence": lookup}
    if verification.passed:
        return {**common, "verification": verification, "verification_notes": []}
    if verification.coverage_status == "error":
        return {**common, "verification": verification,
                "forced_stop_reason": "verification_unavailable"}

    failed = [v for v in verification.verdicts if not v.supports and v.scope_status not in {"unrequested", "duplicate"}]
    # Keep the draft and its citations together until finalization. Filtering
    # only metadata left rejected claims in the answer shown to users.
    return {
        **common,
        "verification": verification,
        "verification_notes": [
            f"Citation {verdict.citation.paper_id}#{verdict.citation.chunk_id} "
            f"failed verification: {verdict.rationale}. Remove or correct its claim."
            for verdict in failed
        ] + [f"Uncited prose must be removed or supported: {claim}" for claim in verification.uncited_claims]
        + [f"The answer still needs this explicitly requested result: {gap}" for gap in verification.missing_aspects]
        + ["Preserve these already supported sentences verbatim; repair only the failed material:\n" + checked],
    }


async def recover_evidence_node(state: AgentState) -> dict:
    """A separately checkpointed, optional evidence expansion; never an inline await."""
    windows = recovery_windows(state)
    if windows["recovery_limit_s"] < 5:
        record_event("verification_evidence_recovery", status="time_guard", **windows)
        return {"verification_recovery_attempted": True, "verification_recovery_changed": False}
    attempted = ({} if state.get("verification_citation_repair_attempted") else
                 {"verification_citation_repair_attempted": True})
    try:
        async with asyncio.timeout(windows["recovery_limit_s"]):
            if attempted:
                rebound = await rebind_citations(state)
                if rebound:
                    # Preserve the old rejection and best checked draft. The
                    # proposed bindings require normal verification first.
                    return {**rebound, **attempted, "verification_recovery_changed": True,
                            "verification_recovery_requires_rewrite": False}
            update = await recover_evidence(state, state["verification"])
    except TimeoutError:
        # Nothing from an unfinished expansion is committed. Keep the checked
        # draft and let the controller admit a repair only if it can be verified.
        record_event("verification_evidence_recovery", status="time_guard", **windows)
        return {**attempted, "verification_recovery_attempted": True, "verification_recovery_changed": False}
    except (BudgetExceeded, LLMError, EmptyProviderResponseError, httpx.HTTPError) as exc:
        record_event("verification_evidence_recovery", status="unavailable", error=type(exc).__name__)
        return {**attempted, "verification_recovery_attempted": True, "verification_recovery_changed": False,
                "forced_stop_reason": budget_stop_reason(exc) if isinstance(exc, BudgetExceeded) else "provider_unavailable"}
    changed = "evidence_lookup" in update
    if changed:
        cited_ids = {v.citation.chunk_id for v in state["verification"].verdicts
                     if not v.supports and v.status == "ok"}
        changed_ids = {cid for cid, value in update["evidence_lookup"].items()
                       if state.get("evidence_lookup", {}).get(cid) != value}
        # New context outside the bound citations cannot change the verdict on
        # an unchanged claim/source pair. Rebind it through synthesis first.
        requires_rewrite = not bool(cited_ids & changed_ids)
        update.update(_preserve_unaffected_facts(state, update["evidence_lookup"]))
        update["verification_recovery_requires_rewrite"] = requires_rewrite
        record_event("verification_recovery_route", changed_chunk_ids=sorted(changed_ids),
                     failed_cited_changed_chunk_ids=sorted(cited_ids & changed_ids),
                     next_action="synthesize" if requires_rewrite else "self_rag")
    return {**update, **attempted, "verification_recovery_attempted": True, "verification_recovery_changed": changed}


def _preserve_unaffected_facts(state, lookup):
    """Changed papers and all inferences need rechecking; independent facts can survive."""
    old = state.get("evidence_lookup", {})
    changed_papers = {entry.paper_id for cid, entry in lookup.items() if old.get(cid) != entry}
    changed_papers.update(entry.paper_id for cid, entry in old.items() if lookup.get(cid) != entry)
    report = state.get("verification")
    groups = {}
    for verdict in report.verdicts if report else []:
        groups.setdefault((verdict.citation.claim_span, verdict.citation.claim_text), []).append(verdict)
    kept = [v for group in groups.values()
            if all(c.supports and c.status == "ok" and c.claim_kind == "paper_fact"
                   and not c.context_claims
                   and c.scope_status in {"requested", "context_needed"}
                   and c.citation.paper_id not in changed_papers for c in group)
            for v in group]
    # Rejected checks remain available for exact-input cache reuse. Expanding an
    # unrelated chunk must not buy another random verdict on unchanged evidence.
    rejected = [v for v in report.verdicts if not v.supports and v.status == "ok"] if report else []
    if not kept and not rejected:
        return {"verification": None, "best_verified": {}, "last_verification_evidence": {}}
    retained = report.model_copy(update={"verdicts": [*kept, *rejected], "passed": False,
        "missing_aspects": [*report.missing_aspects, "Expanded source context still requires verification."]})
    answer = state.get("answer") or ""
    text = supported_text(answer, retained)
    best = {"question": state.get("research_question") or state.get("question", ""),
            "draft": answer, "text": text, "verification": retained, "evidence_lookup": lookup,
            "answer_fields": state.get("answer_fields", [])} if text else {}
    record_event("verification_evidence_invalidation", changed_papers=sorted(changed_papers),
                 preserved_fact_checks=len(kept), invalidated_checks=len(report.verdicts) - len(kept))
    return {"verification": retained, "best_verified": best, "last_verification_evidence": lookup}
