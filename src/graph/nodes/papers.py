"""arXiv discovery and scoped full-text acquisition, governed by one controller."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
import re
from time import monotonic

import httpx
from langchain_core.messages import AIMessage

from src.core.research_policy import ResearchPolicy, date_window, date_mode, prefers_recent
from src.core.trace import record_event
from src.core.types import Decomposition, SubQuestion
from src.corpus.arxiv_client import GLOBAL_ARXIV_CLIENT, normalize_arxiv_id
from src.corpus.ingestion import ingest_paper_with_result
from src.corpus.access import acquisition_client, ingestion
from src.graph.conversation import conversation_turns, planning_context, planned_session, delivered_session
from src.core.run_context import today
from src.graph.controller import fingerprint, search_remaining
from src.llm.budget import budget_stop_reason, BudgetExceeded, set_turn_budget
from src.llm.errors import EmptyProviderResponseError, LLMError
from src.retrieval.paper_search import (
    PaperOutputError,
    Plan,
    Query,
    assess_candidates,
    metadata_record,
    select_results,
    plan_search,
    results_for,
    resolve_named_papers,
)

RESULTS_PER_QUERY = 10


def error(stage, exc):
    return {
        "stage": stage,
        "error": type(exc).__name__,
        **({"code": exc.code} if isinstance(exc, PaperOutputError) else {}),
        **(
            {"http_status": exc.response.status_code}
            if isinstance(exc, httpx.HTTPStatusError)
            else {}
        ),
    }


def unassessed(candidates):
    return [
        {
            **p,
            "fit": "unassessed",
            "evidence_level": "metadata",
            "summary": "",
            "why_relevant": "",
            "uncertainty": "已取得元数据，筛选未完成。",
            "evidence": [],
        }
        for p in candidates[:5]
    ]


def progress(report, results=None):
    record_event(
        "paper_search", report=report, **({"papers": results} if results is not None else {})
    )


async def plan_papers_node(state):
    policy = ResearchPolicy.model_validate(state["research_policy"])
    history = planning_context(state)
    report = {
        "as_of": datetime.now(UTC).isoformat(),
        "rounds": 0,
        "queries": [],
        "errors": [],
        "network_s": 0.0,
        "candidate_count": 0,
        "assessments": 0,
        "status": "planning",
        "limits": {
            "queries": policy.max_queries,
            "candidates": policy.max_candidates,
            "results": 5,
            "assessments": policy.max_assessments,
            "papers": policy.max_papers,
        },
        "decisions": [],
    }
    try:
        async with asyncio.timeout(policy.timeout_s):
            plan = await plan_search(state["question"], history)
    except (
        ValueError,
        LLMError,
        EmptyProviderResponseError,
        BudgetExceeded,
        httpx.HTTPError,
        TimeoutError,
    ) as exc:
        report["errors"].append(error("planning", exc))
        return {
            "paper_search": report,
            "forced_stop_reason": budget_stop_reason(exc)
            if isinstance(exc, BudgetExceeded)
            else "search_plan_invalid",
            "paper_plan": {"clarification": ""},
        }
    from src.retrieval.paper_search import requested_result_bounds, acquisition_queries

    bounds = requested_result_bounds(state["question"])
    if bounds:
        plan = plan.model_copy(update={"min_results": bounds[0], "max_results": bounds[1]})
    elif plan.min_results > plan.max_results:
        plan = plan.model_copy(update={"min_results": plan.max_results})
    if plan.reading_scope == "verify_claim":
        plan = plan.model_copy(update={"read_full_text": True})
    no_read = re.search(
        r"(?:不要|不用|无需)(?:读取|阅读|读)?(?:原文|全文)|\bdo not read (?:the )?(?:full text|pdf|original)\b",
        state["question"],
        re.I,
    )
    if no_read:
        plan = plan.model_copy(update={"read_full_text": False, "reading_questions": []})
    if max(len(plan.paper_ids), len(plan.named_papers)) > policy.max_papers and plan.read_full_text:
        plan = plan.model_copy(
            update={
                "clarification": f"本轮最多深入阅读 {policy.max_papers} 篇。请先选择这一批要阅读的论文，其余可以接续研究。"
            }
        )
    if plan.read_full_text:
        set_turn_budget(policy.reading_budget_usd)
    lower, upper = date_window(plan.question, today=today())
    report.update(
        date_from=lower,
        date_to=upper,
        date_mode=date_mode(plan.question),
        prefer_recent=prefers_recent(plan.question),
        criteria=plan.criteria,
        hard_constraints=plan.hard_constraints,
        soft_preferences=plan.soft_preferences,
        max_results=plan.max_results,
        status="planned",
    )
    queries = acquisition_queries(plan)
    # Discovery always has a relevance path; a fresh request also samples recent work.
    if queries:
        queries[0] = queries[0].model_copy(update={"sort": "relevance", "offset": 0})
        if report["prefer_recent"] and not any(q.sort == "recent" for q in queries):
            if len(queries) > 1:
                queries[1] = queries[1].model_copy(update={"sort": "recent"})
            else:
                queries.append(queries[0].model_copy(update={"sort": "recent"}))
    progress(report)
    return {
        "paper_plan": plan.model_dump(),
        "session_context": planned_session(state, plan),
        "paper_search": report,
        "pending_paper_queries": [q.model_dump() for q in queries],
        "research_question": plan.question,
        "conversation_context": history,
        "context_message_count": len(conversation_turns(state.get("messages", [])[:-1], max_messages=4)),
        "result_kind": "papers",
    }


async def search_papers_node(state):
    plan = Plan.model_validate(state["paper_plan"])
    policy = ResearchPolicy.model_validate(state["research_policy"])
    report = {
        **state["paper_search"],
        "queries": list(state["paper_search"]["queries"]),
        "errors": list(state["paper_search"]["errors"]),
    }
    report["rounds"] += 1
    report["status"] = "searching"
    found = {p["paper_id"]: p for p in state.get("paper_candidates", [])}
    done = {q["key"] for q in report["queries"]}
    unique = {
        q.key(): q
        for q in map(Query.model_validate, state.get("pending_paper_queries", []))
        if q.key() not in done
    }
    queries = list(unique.values())[: max(0, policy.max_queries - len(done))]
    previous_count = len(found)

    async def obtain(query=None, identifier=None):
        row = {
            "key": query.key() if query else identifier,
            "terms": query.terms if query else [],
            "id": identifier,
            "scope": query.scope if query else "identity",
            "sort": query.sort if query else "identity",
            "match": query.match if query else "phrase",
            "offset": query.offset if query else 0,
            "round": report["rounds"],
        }
        report["queries"].append(row)
        try:
            dates = (
                {"date_from": report["date_from"], "date_to": report["date_to"]}
                if report["date_mode"] == "required"
                else {}
            )
            sources = (
                await acquisition_client(GLOBAL_ARXIV_CLIENT).search(
                    query.api_query(),
                    limit=RESULTS_PER_QUERY,
                    latest=query.sort == "recent",
                    offset=query.offset,
                    fresh=bool(re.search(r"强制刷新|重新联网|忽略缓存|force refresh|bypass cache", state["question"], re.I)),
                    **dates,
                )
                if query
                else [
                    await acquisition_client(GLOBAL_ARXIV_CLIENT).fetch_metadata(
                        identifier, fresh=not bool(re.search(r"v\d+$", identifier))
                    )
                ]
            )
            row.update(status="ok", results=len(sources), paper_ids=[p.paper_id for p in sources])
            row["new_candidates"] = sum(p.paper_id not in found for p in sources)
            for source in sources:
                if source.paper_id in found or len(found) < policy.max_candidates:
                    found[source.paper_id] = metadata_record(source)
            report["candidate_count"] = len(found)
            progress(report, state.get("paper_results") or unassessed(list(found.values())))
        except asyncio.CancelledError:
            row["status"] = "timeout"
            raise
        except (httpx.HTTPError, ValueError, LookupError, TimeoutError) as exc:
            row["status"] = "error"
            report["errors"].append(error("search", exc))

    started = monotonic()
    try:
        async with asyncio.timeout(search_remaining(state)):
            calls = (
                [obtain(identifier=pid) for pid in plan.paper_ids]
                if plan.paper_ids and report["rounds"] == 1
                else [obtain(query=q) for q in queries]
            )
            await asyncio.gather(*calls)
    except TimeoutError as exc:
        report["errors"].append(error("search", exc))
    report["network_s"] = round(report["network_s"] + monotonic() - started, 3)
    report.update(
        candidate_count=len(found),
        new_candidates=len(found) - previous_count,
        status="candidates_ready" if found else "no_candidates",
    )
    if plan.named_papers and plan.read_full_text and not plan.paper_ids:
        report["named_resolution"] = resolve_named_papers(plan.named_papers, list(found.values()))
    progress(report)
    return {
        "paper_candidates": list(found.values()),
        "paper_search": report,
        "pending_paper_queries": [],
    }


async def assess_papers_node(state):
    plan = Plan.model_validate(state["paper_plan"])
    report = {**state["paper_search"], "errors": list(state["paper_search"]["errors"])}
    pool = state["paper_candidates"]
    rejected_cache = dict(state.get("rejected_candidate_fingerprints", {}))
    candidates = [p for p in pool if rejected_cache.get(p["paper_id"]) != fingerprint([p])]
    report["assessed_input_count"] = len(candidates)
    report["reused_exclusions"] = len(pool) - len(candidates)
    if not candidates:
        return {
            "paper_search": report,
            "assessed_fingerprint": fingerprint(pool),
            "pending_paper_queries": [],
        }
    report["assessments"] = report.get("assessments", 0) + 1
    try:
        async with asyncio.timeout(max(0.01, search_remaining(state))):
            assessment = await assess_candidates(
                plan,
                candidates,
                report["queries"],
                fresh=report["prefer_recent"],
                original_question=state["question"],
            )
    except (
        ValueError,
        LLMError,
        EmptyProviderResponseError,
        BudgetExceeded,
        httpx.HTTPError,
        TimeoutError,
    ) as exc:
        report["errors"].append(error("assessment", exc))
        report["status"] = "assessment_failed"
        return {
            "paper_search": report,
            "paper_results": state.get("paper_results") or unassessed(candidates),
            "forced_stop_reason": budget_stop_reason(exc)
            if isinstance(exc, BudgetExceeded)
            else "candidate_output_invalid"
            if isinstance(exc, PaperOutputError)
            else "candidate_assessment_failed",
        }
    results = results_for(assessment, candidates)
    for paper in candidates:
        if assessment.rejected.get(paper["paper_id"]) in {
            "namesake",
            "incidental",
            "constraint_violation",
            "not_topical",
        }:
            rejected_cache[paper["paper_id"]] = fingerprint([paper])
        else:
            rejected_cache.pop(paper["paper_id"], None)

    async def confirm(p):
        if p["date_confirmed"]:
            return
        try:
            current = metadata_record(await acquisition_client(GLOBAL_ARXIV_CLIENT).fetch_metadata(p["paper_id"]))
            p.update(
                {
                    key: current[key]
                    for key in (
                        "published_at",
                        "date_confirmed",
                        "version",
                        "version_id",
                        "updated_at",
                    )
                }
            )
        except (httpx.HTTPError, ValueError, LookupError, TimeoutError) as exc:
            report["errors"].append(error("publication_date", exc))

    try:
        async with asyncio.timeout(max(0.01, min(20, search_remaining(state)))):
            await asyncio.gather(*(confirm(p) for p in results))
    except TimeoutError as exc:
        report["errors"].append(error("publication_date", exc))
    # Metadata can support bibliographic requirements, but provisional search
    # dates cannot satisfy them until the canonical record is confirmed.
    unconfirmed_date_support = False
    for paper in results:
        if (not paper["date_confirmed"] and any(
                "published_at" in check.get("metadata_fields", [])
                for check in paper.get("constraint_checks", []))):
            paper["fit"] = "partial"
            paper["uncertainty"] += " 首次发表日期尚未确认。"
            unconfirmed_date_support = True
    if report["date_mode"] == "required":
        before = len(results)
        results = [
            p
            for p in results
            if not p["date_confirmed"]
            or report["date_from"] <= p["published_at"][:10] <= report["date_to"]
        ]
        report["outside_date_count"] = before - len(results)
        for p in results:
            if not p["date_confirmed"]:
                p["fit"] = "partial"
                if "首次发表日期尚未确认" not in p["uncertainty"]:
                    p["uncertainty"] += " 首次发表日期尚未确认。"
                unconfirmed_date_support = True
    results = select_results(results, max_results=plan.max_results, prefer_recent=report["prefer_recent"])
    direct_count = sum(p["fit"] == "direct" for p in results)
    overview_valid = not unconfirmed_date_support and set(assessment.overview_paper_ids) <= {p["paper_id"] for p in results}
    report.update(
        reason=assessment.reason,
        rejected={
            **{
                pid: reason
                for pid, reason in report.get("rejected", {}).items()
                if pid not in {p["paper_id"] for p in candidates}
            },
            **assessment.rejected,
        },
        selected_count=len(results),
        direct_count=direct_count,
        min_direct_results=plan.min_results,
        status="assessed",
        coverage=assessment.coverage if direct_count >= plan.min_results else "needs_more",
        ordering="confirmed_publication_desc" if report["prefer_recent"] else "relevance",
        overview=assessment.overview if overview_valid else "",
        overview_paper_ids=assessment.overview_paper_ids if overview_valid else [],
    )
    progress(report, results)
    return {
        "paper_results": results,
        "paper_search": report,
        "assessed_fingerprint": fingerprint(pool),
        "rejected_candidate_fingerprints": rejected_cache,
        "pending_paper_queries": [q.model_dump() for q in assessment.next_queries],
    }


async def expand_reading_node(state):
    gaps = [
        gap
        for result in state.get("subq_results", {}).values()
        if not result.sufficient
        for gap in result.missing_aspects
    ]
    history = list(state.get("conversation_context", []))
    history.append(
        "assistant: Evidence gaps to investigate, not new user requirements: " + "; ".join(gaps)
    )
    plan = await plan_search(state["question"], history)
    return {
        "pending_paper_queries": [q.model_dump() for q in plan.queries],
        "related_expansion_attempted": True,
        "related_search": True,
    }


async def read_papers_node(state):
    plan = Plan.model_validate(state["paper_plan"])
    policy = ResearchPolicy.model_validate(state["research_policy"])
    named = state["paper_search"].get("named_resolution", {})
    requested_ids = plan.paper_ids or named.get("ids", [])
    if requested_ids and not state.get("related_search"):
        by_id = {p["paper_id"]: p for p in state.get("paper_candidates", [])}
        # Resume/older checkpoints may already have assessed the selected source.
        by_id.update({p["paper_id"]: p for p in state.get("paper_results", [])})
        results = [
            {
                **by_id[pid],
                "fit": "user_selected",
                "evidence_level": "metadata",
                "summary": "",
                "why_relevant": "",
                "uncertainty": "",
                "evidence": [],
            }
            for requested in requested_ids
            for pid in ["arxiv:" + normalize_arxiv_id(requested)]
            if pid in by_id
        ]
    else:
        results = [dict(p) for p in state["paper_results"]]
    report = {**state["paper_search"], "errors": list(state["paper_search"]["errors"])}
    report.update(
        reading_target="user_selected" if requested_ids else "search_selected", read_details=[]
    )
    if not results:
        report["full_text_status"] = "metadata_unavailable"
        return {
            "paper_results": [],
            "paper_search": report,
            "forced_stop_reason": "specified_paper_unavailable",
        }
    read_ids = list(state.get("retrieval_paper_ids") or [])
    attempted = list(state.get("reading_attempted_ids", []))
    existing = {"arxiv:" + normalize_arxiv_id(pid) for pid in read_ids}
    targets = [
        p for p in results if p["paper_id"] not in attempted and p["paper_id"] not in existing
    ]
    targets = targets[: max(0, policy.max_papers - len(read_ids))]
    # Optional full-text indexing has its own small bound and the same shared paid budget.
    for p in targets:
        attempted.append(p["paper_id"])
        p["read_status"] = "reading"
        progress(report, results)
        try:
            async with asyncio.timeout(75):
                target = p.get("version_id") or p["paper_id"]
                result = await ingestion(ingest_paper_with_result)(target, max_chunks=160)
            p["read_status"] = "cached_full_text" if result.status == "skipped" else "full_text"
            read_ids.append(result.paper_id)
            p["read_paper_id"] = result.paper_id
            report["read_details"].append(
                {
                    "paper_id": result.paper_id,
                    "status": p["read_status"],
                    "chunks_inserted": result.chunks_inserted,
                }
            )
        except (BudgetExceeded, LLMError, httpx.HTTPError, ValueError, TimeoutError) as exc:
            report["errors"].append(error("full_text", exc))
            p["read_status"] = "read_failed"
            report["read_details"].append({"paper_id": p["paper_id"], "status": "read_failed"})
            if isinstance(exc, BudgetExceeded):
                break
    added_count = len(read_ids) - len(state.get("retrieval_paper_ids") or [])
    report["read_paper_ids"] = read_ids
    wanted = {
        "arxiv:" + normalize_arxiv_id(pid)
        for pid in (requested_ids or [p["paper_id"] for p in targets])
    }
    acquired = not (named.get("missing") or named.get("ambiguous")) and bool(read_ids) and wanted.issubset(
        {"arxiv:" + normalize_arxiv_id(pid) for pid in read_ids}
    )
    report["full_text_status"] = (
        "complete" if acquired else "partial" if read_ids else "read_failed"
    )
    report["deferred_paper_ids"] = [
        p["paper_id"] for p in results if p["paper_id"] not in attempted
    ]
    readable = [p for p in results if p.get("read_paper_id") in read_ids]
    questions = reading_questions(plan, readable)
    decomposition = state.get("decomposition") if state.get("related_search") else None
    if decomposition is None:
        separate = len(readable) > 1 and bool(requested_ids) and plan.reading_scope != "overview"
        if separate:
            decomposition = Decomposition(sub_questions=[SubQuestion(
                text=(f"Read ONLY {p['title']} ({p['read_paper_id']}). Establish this paper's own mechanism "
                      "and the objects it operates on. Explain its fit or mismatch to these technical axes: "
                      + json.dumps(plan.criteria, ensure_ascii=False) + ". "
                      "This task produces ONE paper's evidence packet. Other papers and their rankings "
                      "are handled separately in final synthesis, never required here. A different mechanism "
                      "is valid evidence of mismatch, not an evidence gap. Distinguish supported mechanisms "
                      "from features not established in the inspected material; do not require the paper "
                      "to demonstrate an unrelated capability or prove its absence."),
                paper_ids=[p["read_paper_id"]],
                sufficiency_question=(
                    f"For {p['title']} ({p['read_paper_id']}), establish its own core mechanism "
                    "and the objects or operations it acts on from original-method evidence. "
                    "This is one paper's evidence packet, not the final comparison. "
                    "The full user question, suitability judgments, comparisons and ranking are "
                    "checked after synthesis across all selected papers."
                ),
            ) for p in readable])
        else:
            decomposition = Decomposition(sub_questions=[SubQuestion(text=q) for q in questions])
    report["reading_questions"] = [q.text for q in decomposition.sub_questions]
    progress(report, results)
    return {
        "paper_results": results,
        "paper_search": report,
        "reading_attempted_ids": attempted,
        "reading_added_count": added_count,
        "subq_results": {
            i: r
            for i, r in state.get("subq_results", {}).items()
            if r.sufficient or not added_count
        }
        if state.get("related_search")
        else {},
        "retrieval_paper_ids": read_ids,
        "external_discovery": {
            "attempted": True,
            "status": report["full_text_status"],
            "read_paper_ids": read_ids,
            "errors": [e for e in report["errors"] if e["stage"] == "full_text"],
        },
        "decomposition": decomposition,
        "question_type": "factual",
        "result_kind": "evidence" if read_ids else "papers",
        "forced_stop_reason": "" if read_ids else "full_text_unavailable",
    }


def reading_questions(plan, results):
    scope = "; ".join(
        f"{p['title']} ({p.get('read_paper_id') or p['paper_id']})" for p in results[:3]
    )
    # Retrieval suggestions cannot silently become additional acceptance requirements.
    # A focused question remains the complete user-resolved task, even when the
    # planner proposes extra training/inference or benchmark distinctions.
    questions = plan.reading_questions if plan.reading_scope == "overview" else [plan.question]
    if not questions and plan.reading_scope == "overview":
        questions = [
            "What problem does the paper address and how does its core method work?",
            "What experiments, baselines, ablations and results support the method?",
            "What limitations, implementation costs and tradeoffs are stated or evidenced?",
        ]
    questions = questions or [plan.question]
    prefix = "Starting papers" if plan.allow_related else "Read only"
    return [f"{prefix} {scope}. {q}" for q in questions]


async def present_papers_node(state):
    report = dict(state.get("paper_search", {}))
    papers = state.get("paper_results", [])
    forced = state.get("forced_stop_reason", "")
    if forced and not papers:
        papers = unassessed(state.get("paper_candidates", []))
    papers = select_results(papers, max_results=state.get("paper_plan", {}).get("max_results", 5), prefer_recent=report.get("prefer_recent", False))
    assessed = bool(papers) and all(p["fit"] in {"direct", "partial"} for p in papers)
    clarification = state.get("paper_plan", {}).get("clarification", "")
    if clarification:
        return {
            "answer": clarification,
            "paper_intro": clarification,
            "status": "needs_input",
            "stop_reason": "clarification_required",
            "result_kind": "papers",
            "messages": [AIMessage(content=clarification)],
        }
    refined_out = report.get("coverage") != "sufficient"
    complete = (
        assessed
        and not forced
        and not refined_out
        and not any(e["stage"] == "search" for e in report.get("errors", []))
    )
    stop = forced or (
        "papers_found"
        if complete
        else "paper_search_partial"
        if papers
        else report.get("stop_reason", "search_exhausted")
    )
    if papers:
        intro = (
            "已确认你指定的论文，但本轮尚未取得可交付的正文分析。"
            if any(p["fit"] == "user_selected" for p in papers)
            else f"找到 {sum(p['fit'] == 'direct' for p in papers)} 篇直接匹配论文，另有 {sum(p['fit'] == 'partial' for p in papers)} 篇相关参考。以下简析依据 arXiv 摘要。"
            if assessed
            else f"已取得 {len(papers)} 篇候选的 arXiv 元数据，但筛选未完成，暂不能把它们作为推荐。"
        )
    else:
        intro = "本轮尚未取得可推荐的论文。请查看查询过程；这不表示 arXiv 中不存在相关论文。"
    lines = [intro]
    if assessed and report.get("overview"):
        lines.append("基于下列摘要的简析：" + report["overview"])
    if forced:
        lines.append(
            {
                "budget_exhausted": "已达到本轮或累计 API 费用限额，停止新增请求。",
                "candidate_assessment_failed": "模型筛选未完成，已保留取得的论文。",
                "candidate_output_invalid": "筛选模型的输出格式或来源绑定未通过校验；这不代表论文不相关。",
                "specified_paper_unavailable": "未能取得指定论文的元数据；尚未开始正文读取。",
                "full_text_unavailable": "原文读取未完成，已保留摘要和论文链接。",
                "search_plan_invalid": "模型生成的搜索计划未通过格式校验，尚未查询论文。",
                "search_planning_failed": "搜索计划未生成成功，本次尚未完成 arXiv 查询。",
            }.get(forced, "本轮部分步骤未完成。")
        )
    if refined_out:
        lines.append("本轮搜索尚未充分覆盖需求，已保留取得的结果；具体停止原因见查询过程。")
    # Keep result identities in the durable conversation for references like "the second paper".
    intro = "\n\n".join(lines)
    for index, p in enumerate(papers, 1):
        identity = "arxiv:" + p["version_id"] if p.get("version_id") else p["paper_id"]
        lines.append(
            f"{index}. {'[相关参考，不计入直接匹配数量] ' if p['fit'] == 'partial' else ''}{p['title']} ({identity})\n{p['summary']}\n{p['why_relevant']}"
            + (f"\n待确认：{p['uncertainty']}" if p["uncertainty"] else "")
        )
    answer = "\n\n".join(lines)
    report["status"] = "complete" if complete else "partial" if papers else "no_results"
    progress(report, papers)
    return {
        "paper_results": papers,
        "session_context": delivered_session(state, papers),
        "paper_search": report,
        "answer": answer,
        "paper_intro": intro,
        "status": "complete" if complete else "partial" if papers else "incomplete",
        "stop_reason": stop,
        "result_kind": "papers",
        "messages": [AIMessage(content=answer)],
    }
