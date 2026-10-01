"""Deterministic result metrics. Free-text factual judgment remains explicit review."""

from __future__ import annotations

import math
import re
from statistics import mean

from src.eval.research_acceptance import same_paper
from src.eval.answer_metrics import METRIC_VERSION, delivered_observations, grade_facts


def base_id(identifier):
    return re.sub(r"v\d+$", "", identifier.removeprefix("arxiv:"))


def ndcg(ids, grades, k):
    ideal = sorted(grades.values(), reverse=True)[:k]
    denominator = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    if not denominator:
        return None
    seen, gain = set(), 0.0
    for i, pid in enumerate(ids[:k]):
        pid = base_id(pid)
        grade = grades.get(pid, 0) if pid not in seen else 0
        gain += (2**grade - 1) / math.log2(i + 2)
        seen.add(pid)
    return gain / denominator


def score_turn(spec, result, *, previous=None, source_errors=None):
    output, gold = result["output"], spec["gold"]
    papers = output.get("paper_results", [])
    sources = output.get("sources", [])
    selected = [p.get("version_id") or p["paper_id"] for p in papers]
    cited = list(dict.fromkeys(s["paper_id"] for s in sources))
    expected = gold.get("required_version_ids") or gold.get("required_ids") or []
    observed = cited if spec["expected_route"] == "read" else selected
    checks, pending_checks = {}, []
    facts_observed, reading_order, field_errors = delivered_observations(output)
    if spec.get("response_fields") and facts_observed is None:
        # A requested native result that failed completely still belongs in the
        # denominator. Legacy/free prose remains separately ungraded.
        facts_observed, reading_order = [], []
    # Source-card order is never a reading recommendation. Only a separately
    # bound order statement is observable; an empty supported-contract order is
    # a miss, whereas legacy output remains explicitly ungraded.
    rank_ids = selected if result["route"] == "discover" and not any(
        p.get("fit") in {"unassessed", "user_selected"} for p in papers
    ) else reading_order if result["route"] == "read" else None
    ranking_basis = "assessed_discovery_results" if result["route"] == "discover" and rank_ids is not None else (
        "delivered_explicit_recommendation" if rank_ids is not None else "not_programmatically_observable")

    def check(name, passed):
        checks[name] = bool(passed)

    check("route", result["route"] == spec["expected_route"])
    statuses = {spec["expected_status"]}
    if spec["expected_status"] == "partial_or_evidence_unavailable":
        statuses = {"partial", "incomplete"}
    check("terminal_status", output["status"] in statuses)
    check("paper_deduplication", len(selected) == len(set(base_id(p) for p in selected)))
    coverage = (
        sum(any(same_paper(got, pid) for got in observed) for pid in expected) / len(expected)
        if expected
        else None
    )
    if expected:
        check("required_references", coverage == 1)
    if gold.get("required_primary_id"):
        if rank_ids is None:
            pending_checks.append("primary_selection")
        else:
            check("primary_selection", bool(rank_ids) and same_paper(rank_ids[0], gold["required_primary_id"]))
    comparison = gold.get("required_comparison_ids", [])
    comparison_coverage = sum(any(same_paper(got, pid) for got in cited) for pid in comparison) / len(comparison) if comparison else None
    if comparison:
        check("comparison_original_coverage", comparison_coverage == 1)
    if gold.get("primary_result_count") is not None:
        check("result_count", len(papers) == gold["primary_result_count"])
    if gold.get("ordered_ids"):
        check("exact_order", [base_id(p) for p in selected] == gold["ordered_ids"])
    first = gold.get("first_submitted")
    if first:
        dates = [str(p.get("published_at") or "")[:10] for p in papers]
        check("first_submission_date", dates == (first if isinstance(first, list) else [first]))
    direct = [p for p in papers if p.get("fit") != "partial"]
    if gold.get("excluded_from_matches"):
        check(
            "excluded_matches",
            not any(base_id(p["paper_id"]) in gold["excluded_from_matches"] for p in direct),
        )
    constraint = gold.get("constraint")
    if constraint:
        check(
            "date_constraint",
            bool(direct)
            and all(
                p.get("date_confirmed")
                and constraint["start_inclusive"]
                <= str(p.get("published_at") or "")[:10]
                < constraint["end_exclusive"]
                for p in direct
            ),
        )
    if spec["expected_route"] == "clarify":
        check("no_acquisition", not output.get("paper_search", {}).get("queries") and not papers)
    if spec["expected_route"] == "read" and spec["expected_status"] == "complete":
        check("original_citations", bool(sources))
    if source_errors is not None:
        check("source_identity_and_spans", not source_errors)
    if gold.get("stop_class") == "source_unavailable":
        check(
            "source_failure_class",
            output.get("stop_reason") in {"full_text_unavailable", "no_evidence"},
        )
        check("no_false_verified_original", not sources)
    if "actual_display_reference_resolution" in spec["programmatic_checks"] and previous:
        prior = previous["output"].get("paper_results", [])
        check(
            "displayed_reference",
            len(prior) >= 2
            and bool(observed)
            and all(
                same_paper(pid, prior[1].get("version_id") or prior[1]["paper_id"])
                for pid in observed
            ),
        )
    ranking = ndcg(rank_ids, gold["relevance_grades"], 3) if gold.get("relevance_grades") and rank_ids is not None else None
    if gold.get("relevance_grades") and rank_ids is None:
        pending_checks.append("ndcg_at_3")
    # These labels are checked only against a documented structured result, not
    # keyword matches or a second model that would hide an LLM judge in the scorer.
    facts = gold.get("required_facts", {})
    fact_details, ungraded_facts = grade_facts(gold, facts_observed)
    fact_checks = {name: detail["passed"] for name, detail in fact_details.items()}
    if facts_observed is not None and (facts or gold.get("relevance_grades") or gold.get("required_primary_id")):
        check("result_field_bindings", not field_errors)
    checks.update({"fact:" + name: passed for name, passed in fact_checks.items()})
    pending_checks.extend("fact:" + name for name in ungraded_facts)
    review_required = bool(spec.get("review_only") or ungraded_facts or pending_checks)
    return {
        "metric_version": METRIC_VERSION,
        "program_pass": all(checks.values()),
        "checks": checks,
        "pending_checks": pending_checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "reference_coverage": coverage,
        "comparison_original_coverage": comparison_coverage,
        "ndcg_at_3": ranking,
        "ranking_basis": ranking_basis,
        "ranked_ids": rank_ids,
        "ranking_label_provenance": gold.get("grade_provenance"),
        "fact_checks": fact_checks,
        "fact_details": fact_details,
        "field_binding_errors": field_errors if facts_observed is not None else [],
        "fact_observable_count": sum(all(c["observed"] for c in d["clauses"]) for d in fact_details.values()),
        "fact_required_count": len(facts),
        "fact_graded_count": len(fact_checks),
        "fact_correct_count": sum(fact_checks.values()),
        "fact_review_status": "needs_source_review"
        if ungraded_facts
        else "programmatic"
        if facts
        else "not_applicable",
        "content_review": "pending" if review_required else "not_applicable",
        "task_pass": False if not all(checks.values()) else None if review_required else True,
    }


def summarize(records, planned_turns):
    profiles = {}
    for profile in sorted(set(planned_turns) | {r["profile"] for r in records}):
        rows = [r for r in records if r["profile"] == profile]
        completed = [r for r in rows if r.get("result")]
        results = [r["result"] for r in completed]
        scores = [r["score"] for r in completed if r.get("score")]
        elapsed = sorted(r["elapsed_s"] for r in results)

        def average(values):
            defined = [v for v in values if v is not None]
            return {"value": mean(defined) if defined else None, "n": len(defined)}

        def percentile(p):
            if not elapsed:
                return None
            position = (len(elapsed) - 1) * p
            lo, hi = math.floor(position), math.ceil(position)
            return elapsed[lo] + (elapsed[hi] - elapsed[lo]) * (position - lo)

        total_prompt = sum(r["cost"].get("prompt_tokens", 0) for r in results)
        hit_tokens = sum(r["cost"].get("prompt_cache_hit_tokens", 0) for r in results)
        complete_reporting = all(
            r["cost"].get("cache_reporting_requests") == r["cost"].get("chat_settled_requests")
            for r in results
        )
        stage_errors = [r["output"].get("paper_search", {}).get("errors", []) for r in results]
        error_counts = {}
        for errors in stage_errors:
            for error in errors:
                code = str(error.get("stage", "unknown")) + ":" + str(error.get("code") or error.get("error", "unknown"))
                error_counts[code] = error_counts.get(code, 0) + 1
        profiles[profile] = {
            "planned_turns": planned_turns.get(profile, 0),
            "executed_turns": len(results),
            "not_run_or_interrupted": planned_turns.get(profile, 0) - len(results),
            "program_passes": sum(s["program_pass"] for s in scores),
            "scored_turns": len(scores),
            "grading_incomplete": len(results) - len(scores),
            "route_accuracy": average([int(s["checks"]["route"]) for s in scores]),
            "reference_coverage": average([s["reference_coverage"] for s in scores]),
            "comparison_original_coverage": average([s.get("comparison_original_coverage") for s in scores]),
            "ndcg_at_3": average([s["ndcg_at_3"] for s in scores]),
            "fact_required": sum(s["fact_required_count"] for s in scores),
            "fact_graded": sum(s["fact_graded_count"] for s in scores),
            "fact_correct": sum(s["fact_correct_count"] for s in scores),
            "fact_observable": sum(s.get("fact_observable_count", 0) for s in scores),
            "fact_accuracy": {
                "value": sum(s["fact_correct_count"] for s in scores) / sum(s["fact_graded_count"] for s in scores)
                if sum(s["fact_graded_count"] for s in scores) else None,
                "n": sum(s["fact_graded_count"] for s in scores),
            },
            "fact_grading_coverage": {
                "value": sum(s["fact_graded_count"] for s in scores) / sum(s["fact_required_count"] for s in scores)
                if sum(s["fact_required_count"] for s in scores) else None,
                "n": sum(s["fact_required_count"] for s in scores),
            },
            "primary_selection_accuracy": average([int(s["checks"]["primary_selection"]) for s in scores
                                                    if "primary_selection" in s["checks"]]),
            "content_reviews_pending": sum(s["content_review"] == "pending" for s in scores),
            "runtime_failures": sum(
                r["output"].get("stop_reason") in {
                    "runtime_error", "turn_timeout", "budget_exhausted", "budget_wait_timeout",
                    "provider_unavailable", "verification_unavailable", "verification_time_guard"}
                or r["output"].get("status") != "complete" and bool(errors)
                for r, errors in zip(results, stage_errors, strict=True)),
            "turns_with_stage_errors": sum(bool(e) for e in stage_errors),
            "stage_error_counts": error_counts,
            "terminal_p50_s": percentile(0.5),
            "terminal_p95_s": percentile(0.95),
            "latency_n": len(elapsed),
            "estimated_usd": sum(r["cost"]["estimated_usd"] for r in results),
            "uncertain_usd": sum(r["cost"]["uncertain_usd"] for r in results),
            "prompt_tokens": total_prompt,
            "cache_hit_tokens": hit_tokens,
            "cache_hit_rate": hit_tokens / total_prompt
            if total_prompt and complete_reporting
            else None,
        }
    return {
        "metric_version": METRIC_VERSION,
        "profiles": profiles,
        "quality_scope": "Typed equality and explicit order are programmatic; extraction is part of the agent. "
        "Ungraded facts are not passes. Source faithfulness and open-text quality still require separate review.",
    }
