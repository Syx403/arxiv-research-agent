"""Public delivery projection shared by all agent clients."""

import re
import json
from hashlib import sha256

from src.core.answer_contract import CONTRACT_VERSION, RequestedField, bind_fields
from src.core.structured_answer import NATIVE_CONTRACT_VERSION, entry_fields, validate_entry


def _delivered_native_fields(state, sources, requested):
    answer = state.get("answer") or ""
    report = state.get("draft_verification") or state.get("verification")
    definitions = {f.name: f.model_dump(mode="json") for f in requested}
    available = {(s["paper_id"], s["chunk_id"]) for s in sources}
    native, facts, rankings, errors = [], [], [], []
    entries = state.get("answer_fields", [])
    for entry in entries:
        try:
            # Optional public-domain constraints have defaults for pre-v6 checkpoints.
            entry = {**validate_entry(entry).model_dump(mode="json"), **entry}
            name, claim = entry["name"], entry["claim"]
            if (name not in definitions or sum(e.get("name") == name for e in entries) != 1
                    or any(entry.get(k) != v for k, v in definitions[name].items())):
                raise ValueError("native_field_request_mismatch")
            fields = entry_fields(entry)
            citations = [c for c in state.get("citations", []) if c.claim_text == claim]
            identities = {(r["paper_id"], r["chunk_id"]) for r in entry["source_refs"]}
            checks = [v for v in report.verdicts if v.citation.claim_text == claim] if report else []
            if (not citations or identities != {(c.paper_id, c.chunk_id) for c in citations}
                    or not identities <= available or identities != {(v.citation.paper_id, v.citation.chunk_id) for v in checks}
                    or not all(v.supports and v.status == "ok" and v.scope_status in {"requested", "context_needed"}
                               for v in checks)):
                raise ValueError("native_field_not_fully_verified")
            span = citations[0].claim_span
            if not span or answer[span[0]:span[1]].strip() != claim:
                raise ValueError("native_field_not_delivered")
            start = answer.find(claim, span[0], span[1])
            binding = {"answer_span": [start, start + len(claim)], "claim": claim,
                       "source_refs": entry["source_refs"]}
            native.append({**entry, **binding})
            facts.extend({**f, **binding} for f in fields["facts"])
            rankings.extend({**r, **binding} for r in fields["rankings"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc)[:200])
    missing = [f.name for f in requested if f.name not in {e["name"] for e in native}]
    return {"answer_contract": {"version": NATIVE_CONTRACT_VERSION,
                                "status": "complete" if not missing and not errors else "partial" if native else "unavailable",
                                "origin": "native_synthesis; deterministic rendering; agent source verification",
                                "answer_sha256": sha256(answer.encode()).hexdigest(),
                                "requested_fields": list(definitions.values()), "missing_fields": missing,
                                "errors": errors},
            "native_fields": native, "structured_facts": facts, "rankings": rankings,
            "recommendation_order": rankings[0]["paper_ids"] if len(rankings) == 1 else []}


def _delivered_fields(state: dict, sources: list[dict], *, requested: bool | None = None) -> dict:
    """Only observations of retained, fully checked claims reach the public API."""
    answer = state.get("answer") or ""
    if requested is False:
        return {"answer_contract": {"version": CONTRACT_VERSION, "status": "not_requested",
                                    "answer_sha256": sha256(answer.encode()).hexdigest()},
                "structured_facts": [], "rankings": [], "recommendation_order": []}
    report = state.get("draft_verification") or state.get("verification")
    grouped = {}
    for citation in state.get("citations", []):
        grouped.setdefault((citation.claim_span, citation.claim_text), []).append(citation)
    available = {(s["paper_id"], s["chunk_id"]) for s in sources}
    facts, rankings, errors, observed = [], [], [], 0
    for (span, claim), citations in grouped.items():
        if not span or not 0 <= span[0] < span[1] <= len(answer) or answer[span[0]:span[1]].strip() != claim:
            errors.append("delivered_claim_span_mismatch")
            continue
        identities = {(c.paper_id, c.chunk_id) for c in citations}
        checks = [v for v in report.verdicts if v.citation.claim_text == claim
                  and (v.citation.paper_id, v.citation.chunk_id) in identities] if report else []
        if (not identities <= available or identities != {(v.citation.paper_id, v.citation.chunk_id) for v in checks}
                or not all(v.supports and v.status == "ok" and v.scope_status in {"requested", "context_needed"}
                           for v in checks)):
            errors.append("delivered_claim_not_fully_checked")
            continue
        if any(v.fields_status != "complete" or v.result_fields is None for v in checks):
            errors.append("result_fields_invalid" if any(v.fields_status == "invalid" for v in checks)
                          else "result_fields_unavailable")
            continue
        if len({json.dumps(v.result_fields, sort_keys=True) for v in checks}) != 1:
            errors.append("inconsistent_joint_claim_fields")
            continue
        try:
            fields = bind_fields(checks[0].result_fields, claim)
        except ValueError:
            errors.append("result_fields_invalid")
            continue
        observed += 1
        for category, target in (("facts", facts), ("rankings", rankings)):
            for item in fields[category]:
                quote_start = answer.find(item["quote"], span[0], span[1])
                if quote_start < 0:
                    errors.append("result_quote_not_delivered")
                    continue
                paper_ids = {item["paper_id"]} if category == "facts" else set(item["paper_ids"])
                target.append({**item, "answer_span": [quote_start, quote_start + len(item["quote"])],
                               "claim": claim,
                               "source_refs": [{"paper_id": pid, "chunk_id": cid}
                                               for pid, cid in sorted(identities) if pid in paper_ids]})
    orders = {tuple(row["paper_ids"]) for row in rankings}
    if len(orders) > 1:
        errors.append("conflicting_recommendation_orders")
    return {
        "answer_contract": {
            "version": CONTRACT_VERSION,
            "status": "not_applicable" if state.get("result_kind") == "papers"
            else "complete" if grouped and not errors else "partial" if observed else "unavailable",
            "answer_sha256": sha256(answer.encode()).hexdigest(),
            "delivered_statements": len(grouped),
            "observed_statements": observed,
            "errors": errors,
            "origin": "agent_source_verifier; semantic extraction, not independent grading",
        },
        "structured_facts": facts,
        "rankings": rankings,
        "recommendation_order": list(next(iter(orders))) if len(orders) == 1 else [],
    }


def paper_url(paper_id: str, url: str = "") -> str | None:
    if paper_id.startswith("report:") and url:
        from src.corpus.official_reports import canonical_report_url, report_id

        try:
            return canonical_report_url(url) if paper_id == report_id(url) else None
        except ValueError:
            return None
    # Source URLs are constructed from validated identifiers, never model HTML.
    identifier = paper_id.removeprefix("arxiv:")
    if paper_id.startswith("arxiv:") and re.fullmatch(
        r"(?:\d{4}\.\d{4,5}|[a-zA-Z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", identifier
    ):
        return f"https://arxiv.org/abs/{identifier}"
    return None


def present_state(state: dict, titles: dict[str, str] | None = None, *,
                  include_typed_observations: bool | None = None,
                  response_fields: tuple[RequestedField, ...] = ()) -> dict:
    titles = titles or {}
    lookup = state.get("evidence_lookup", {})
    sources: dict[tuple[str, int], dict] = {}
    for citation in state.get("citations", []):
        evidence = lookup.get(citation.chunk_id)
        if evidence is None or evidence.paper_id != citation.paper_id:
            continue
        key = (citation.paper_id, citation.chunk_id)
        source = sources.setdefault(
            key,
            {
                "paper_id": citation.paper_id,
                "chunk_id": citation.chunk_id,
                "title": titles.get(citation.paper_id, citation.paper_id),
                "url": paper_url(citation.paper_id, evidence.url),
                "text": evidence.text,
                "claims": [],
                "source_spans": evidence.source_spans,
                "published_at": evidence.published_at,
            },
        )
        if citation.claim_text not in source["claims"]:
            source["claims"].append(citation.claim_text)
    report = state.get("draft_verification") or state.get("verification")
    return {
        "answer": state.get("paper_intro")
        if state.get("result_kind") == "papers" and state.get("paper_intro")
        else state.get("answer") or "本轮未形成可交付的回答。",
        "result_kind": state.get("result_kind", "evidence"),
        "paper_results": state.get("paper_results", []),
        "paper_search": state.get("paper_search", {}),
        "claim_review": state.get("claim_review", {}),
        "status": state.get("status", "incomplete"),
        "stop_reason": state.get("stop_reason", ""),
        "research_question": state.get("research_question", state.get("question", "")),
        "sources": list(sources.values()),
        "context_messages": state.get(
            "context_message_count", len(state.get("conversation_context", []))
        ),
        "subquestions": [q.text for q in state["decomposition"].sub_questions]
        if state.get("decomposition")
        else [],
        "missing_aspects": state.get("missing_aspects", []),
        "generation_retries": state.get("tool_iters", 0),
        "external_discovery": state.get("external_discovery", {}),
        "delivery_from_prior_draft": state.get("delivery_from_prior_draft", False),
        **(_delivered_native_fields(state, list(sources.values()), response_fields) if response_fields
           else _delivered_fields(state, list(sources.values()), requested=include_typed_observations)),
        "checks": [
            {
                "claim": v.citation.claim_text,
                "supports": v.supports,
                "status": v.status,
                "claim_kind": v.claim_kind,
                "scope_status": v.scope_status,
                "rationale": v.rationale,
                "paper_id": v.citation.paper_id,
                "chunk_id": v.citation.chunk_id,
                "fields_status": v.fields_status,
                "context_claims": v.context_claims,
                "context_verified": v.context_verified,
            }
            for v in report.verdicts
        ]
        if report
        else [],
    }
