"""Deliver only checked sentences and expose unmet requirements explicitly."""
from __future__ import annotations

from langchain_core.messages import AIMessage
import re

from src.core.types import VerificationReport, SELF_RAG_MAX_RETRIES
from src.graph.nodes.synthesize import _parse_citations
from src.graph.state import AgentState
from src.graph.delivery import supported_text, delivery_score
from src.graph.conversation import delivered_session


async def finalize_node(state: AgentState) -> dict:
    draft = state.get("answer") or ""
    report = state.get("verification")
    verified_answer = "" if state.get("synthesis_format_degraded") else supported_text(draft, report)
    best = state.get("best_verified", {})
    prior = not (report and report.passed) and best.get("question") == (state.get("research_question") or state.get("question", "")) and delivery_score(best.get("text", "")) > delivery_score(verified_answer)
    lookup = state.get("evidence_lookup", {})
    answer_fields = state.get("answer_fields", [])
    selected_report = report
    if prior:
        # Re-extract, never trust a cached display string as independently verified evidence.
        verified_answer = supported_text(best["draft"], best["verification"])
        selected_report, lookup = best["verification"], best["evidence_lookup"]
        answer_fields = best.get("answer_fields", [])
    answer_fields = [entry for entry in answer_fields if entry.get("claim") and entry["claim"] in verified_answer]
    citations = _parse_citations(verified_answer)
    # Rendering changes spans, not the source checks or their verified antecedents.
    # Preserve the audit, rather than manufacture a fresh unconditional pass.
    checked = {(v.citation.claim_text, v.citation.paper_id, v.citation.chunk_id): v
               for v in selected_report.verdicts if v.supports and v.status == "ok"} if selected_report else {}
    delivered_verdicts = [checked[(c.claim_text, c.paper_id, c.chunk_id)].model_copy(update={"citation": c})
                          for c in citations]
    gaps = [
        aspect
        for result in state.get("subq_results", {}).values()
        for aspect in (result.missing_aspects or ([] if result.sufficient else result.next_subqueries or ["Evidence remains insufficient for one subquestion."]))
    ]
    if report:
        gaps.extend("Requested answer aspect missing: " + gap for gap in report.missing_aspects)
    delivered_names = {entry["name"] for entry in answer_fields}
    gaps.extend("Requested structured field missing: " + field["name"]
                for field in state.get("response_fields", []) if field["name"] not in delivered_names)
    judgment_failed = any(r.stop_reason in {"relevance_judgment_failed", "sufficiency_judgment_failed"}
                          for r in state.get("subq_results", {}).values())
    verification_unavailable = bool(report and (report.coverage_status == "error"
                                      or any(v.status == "error" for v in report.verdicts)))
    if judgment_failed:
        gaps = [aspect for r in state.get("subq_results", {}).values() for aspect in r.missing_aspects]
        gaps.append("Some evidence assessments could not be completed; this does not mean the sources are irrelevant or absent.")
    # An empty/unavailable check is not a rejected factual claim.
    rejected = bool(report and (any(not v.supports and v.status == "ok" and v.scope_status not in {"unrequested", "duplicate"}
                                   for v in report.verdicts)
                               or report.uncited_claims))
    discovery = state.get("external_discovery", {})
    current_ids = set(discovery.get("current_report_ids", []))
    current_uncovered = bool(current_ids and not current_ids.intersection(c.paper_id for c in citations))
    if current_uncovered:
        gaps.append("The latest official report was discovered but is not covered by delivered full-text evidence.")
    external_incomplete = discovery.get("attempted") and discovery.get("status") != "complete"
    if external_incomplete:
        gaps.append("External discovery did not fully complete: " + discovery.get("status", "unknown") +
                    ". This does not establish that relevant papers do not exist.")
    if rejected:
        gaps.append("Some drafted claims could not be verified against the supplied sources and were withheld.")
    forced = state.get("forced_stop_reason")
    if forced:
        gaps.append("The run stopped before all required checks completed: " + forced + ".")
    if not citations:
        status = "incomplete"
        reason = (
            "no_evidence" if not state.get("evidence")
            else "synthesis_incomplete" if state.get("synthesis_format_degraded")
            else "verification_retry_limit" if state.get("tool_iters", 0) >= SELF_RAG_MAX_RETRIES
            else "verification_failed"
        )
        gaps.append("No supported conclusion was ready for delivery.")
    elif rejected or gaps:
        status = "partial"
        reason = ("verification_retry_limit" if state.get("tool_iters", 0) >= SELF_RAG_MAX_RETRIES else "verification_failed") if rejected else "evidence_incomplete"
    else:
        status, reason = "complete", "answer_verified"
    if forced:
        status, reason = ("partial" if citations else "incomplete"), forced
    elif judgment_failed:
        status, reason = ("partial" if citations else "incomplete"), "evidence_judgment_failed"
    elif verification_unavailable:
        status, reason = ("partial" if citations else "incomplete"), "verification_unavailable"
    elif external_incomplete and (not citations or not rejected):
        status, reason = ("partial" if citations else "incomplete"), (
            {"planning_failed": "search_planning_failed", "unavailable": "external_search_unavailable",
             "timeout": "external_discovery_timeout", "no_results": "no_matching_papers",
             "no_relevant_candidates": "no_matching_papers", "read_failed": "full_text_unavailable",
             "screening_failed": "candidate_assessment_failed"}.get(discovery.get("status"), "external_discovery_incomplete"))
    gaps = list(dict.fromkeys(gaps))
    answer = verified_answer
    if status != "complete":
        answer = (answer + "\n\n" if answer else "") + "Research incomplete. " + " ".join(gaps)
        answer += (" Next step: retry the failed evidence assessment."
                   if judgment_failed else " Next step: obtain evidence for the remaining questions before drawing further conclusions.")
        if re.search(r"[\u4e00-\u9fff]", state.get("question", "")):
            explanation = (
                "部分核验未能完成，不能据此认定论文不支持相关结论。" if verification_unavailable or judgment_failed
                else "搜索计划未能正确生成；尚未完成外部论文查询。" if reason == "search_planning_failed"
                else _discovery_explanation(discovery, bool(citations)) if external_incomplete and not citations
                else "已保留有原文依据的内容，但尚未覆盖本轮发现的最新官方报告，不能视为最新进展的完整回答。" if current_uncovered
                else "原文获取尚未全部完成，已保留有原文依据的内容。" if external_incomplete
                else "当前证据尚未覆盖全部问题，已保留有原文依据的内容。" if citations and not rejected
                else "部分结论未通过核验，已保留本轮有证据支持的内容。" if citations
                else "本轮未形成可交付的已核验结论。"
            )
            answer = (verified_answer + "\n\n" if verified_answer else "") + explanation + " 具体缺口和执行原因见本轮过程。"
    return {
        "session_context": delivered_session(state, state.get("paper_results", [])),
        "answer": answer,
        "answer_fields": answer_fields,
        "verified_answer": verified_answer,
        "citations": citations,
        "draft_verification": selected_report,
        "evidence_lookup": lookup,
        "delivery_from_prior_draft": bool(prior),
        "verification": VerificationReport(
            verdicts=delivered_verdicts,
            passed=bool(citations),
            rationale="delivered_claims_only; completeness is recorded separately in status",
        ),
        "status": status,
        "stop_reason": reason,
        "missing_aspects": gaps + (["Uncited draft claim withheld: " + claim for claim in report.uncited_claims]
                                    if report else []),
        "forced_stop_reason": forced or "",
        "messages": [AIMessage(content=answer)],
    }


def _discovery_explanation(report: dict, has_citations: bool = False) -> str:
    causes = []
    for error in report.get("errors", []):
        provider = "arXiv" if str(error.get("stage", "")).startswith("arxiv") else "Semantic Scholar" if str(error.get("stage", "")).startswith("semantic") else "外部获取"
        if error.get("http_status") == 429:
            causes.append(provider + "限流")
        elif error.get("error") in {"ReadTimeout", "ConnectTimeout", "deadline_exceeded", "search_deadline_exceeded"}:
            causes.append(provider + "超时")
    detail = "（" + "；".join(dict.fromkeys(causes)) + "）" if causes else ""
    count = len(report.get("candidates", []))
    if report.get("status") in {"no_results", "no_relevant_candidates"}:
        return f"本轮查询{'返回的候选未通过相关性筛选' if count else '未返回匹配候选'}；这不代表没有相关论文。"
    if report.get("status") == "screening_failed":
        return f"已发现 {count} 篇候选，但相关性判断未能完成；不能将其当作不相关论文。"
    if count:
        return f"已发现 {count} 篇候选，但本轮未取得可交付的原文证据{detail}。"
    return f"本轮外部论文搜索未能完成{detail}，尚未取得原文证据；不能据此判断没有相关论文。"
