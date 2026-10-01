"""Pure result scoring: declared aliases and typed equality, no model or network.

Extraction is part of the agent being evaluated. A correct machine field alone
does not establish that all surrounding prose is faithful to the paper.
"""
from __future__ import annotations

from hashlib import sha256
import re
import unicodedata

from src.core.answer_contract import CONTRACT_VERSION, bind_fields
from src.core.structured_answer import NATIVE_CONTRACT_VERSION, entry_fields
from src.eval.research_acceptance import same_paper


METRIC_VERSION = "ara-answer-metrics-2"


def normal_text(value: str) -> str:
    """Only mechanical spelling normalization; no stemming or semantic matching."""
    return re.sub(r"[\s_-]+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def delivered_observations(output):
    contract = output.get("answer_contract", {})
    native = contract.get("version") == NATIVE_CONTRACT_VERSION
    if contract.get("version") not in {CONTRACT_VERSION, NATIVE_CONTRACT_VERSION}:
        return None, None, ["unsupported_or_missing_answer_contract"]
    if contract.get("status") == "not_requested":
        return None, None, ["typed_observations_not_requested"]
    answer = output.get("answer", "")
    if contract.get("answer_sha256") != sha256(answer.encode()).hexdigest():
        return [], [], ["answer_digest_mismatch"]
    sources = {(s["paper_id"], s["chunk_id"]): s for s in output.get("sources", [])}
    checks = output.get("checks", [])
    valid, orders, errors = [], [], []
    for category, target in (("structured_facts", valid), ("rankings", orders)):
        for item in output.get(category, []):
            try:
                claim, span, quote = item["claim"], item["answer_span"], item["quote"]
                if (len(span) != 2 or any(type(i) is not int for i in span)
                        or not 0 <= span[0] < span[1] <= len(answer)
                        or answer[span[0]:span[1]] != quote or claim not in answer):
                    raise ValueError("field_not_in_delivered_answer")
                refs = {(r["paper_id"], r["chunk_id"]) for r in item["source_refs"]}
                if not refs or not refs <= sources.keys() or not all(claim in sources[r].get("claims", []) for r in refs):
                    raise ValueError("field_source_not_delivered")
                if not all(any(c.get("claim") == claim and (c.get("paper_id"), c.get("chunk_id")) == ref
                               and c.get("supports") is True and c.get("status") == "ok"
                               and c.get("scope_status") in {"requested", "context_needed"}
                               and (native or c.get("fields_status") == "complete") for c in checks) for ref in refs):
                    raise ValueError("field_source_not_checked")
                raw = {k: v for k, v in item.items() if k not in {"claim", "answer_span", "source_refs"}}
                key = "facts" if category == "structured_facts" else "rankings"
                if native:
                    entries = [e for e in output.get("native_fields", []) if e.get("claim") == claim]
                    if len(entries) != 1 or entry_fields(entries[0])[key] != [raw]:
                        raise ValueError("native_value_not_rendered")
                    declared = [f for f in contract.get("requested_fields", []) if f["name"] == entries[0]["name"]]
                    if (len(declared) != 1 or any(entries[0].get(k) != v for k, v in declared[0].items())
                            or refs != {(r["paper_id"], r["chunk_id"]) for r in entries[0]["source_refs"]}):
                        raise ValueError("native_field_contract_mismatch")
                bound = bind_fields({"facts": [raw] if key == "facts" else [],
                                     "rankings": [raw] if key == "rankings" else []}, claim)[key][0]
                paper_ids = {bound["paper_id"]} if key == "facts" else set(bound["paper_ids"])
                if not paper_ids <= {p for p, _ in refs}:
                    raise ValueError("field_source_identity_mismatch")
                target.append(item)
            except (KeyError, TypeError, ValueError):
                errors.append("invalid_" + category + "_binding")
    unique_orders = {tuple(row["paper_ids"]) for row in orders}
    order = list(next(iter(unique_orders))) if len(unique_orders) == 1 else []
    if len(unique_orders) > 1:
        errors.append("conflicting_recommendation_orders")
    if output.get("recommendation_order", []) != order:
        errors.append("recommendation_order_mismatch")
        order = []
    return valid, order, errors


def _equivalent(got, expected, rule):
    aliases = {normal_text(v): normal_text(canonical)
               for canonical, values in rule.get("aliases", {}).items()
               for v in [canonical, *values]}

    def token(value):
        text = normal_text(value)
        return aliases.get(text, text)

    if type(expected) is bool:
        return type(got) is bool and got is expected
    if type(expected) in {int, float}:
        return type(got) in {int, float} and abs(got - expected) <= rule.get("absolute_tolerance", 0)
    if isinstance(expected, str):
        return isinstance(got, str) and token(got) == token(expected)
    if not isinstance(got, list) or len(got) != len(expected):
        return False
    if rule["kind"] == "relations":
        if not all(isinstance(row, list) and all(isinstance(v, str) for v in row) for row in got):
            return False
        def relation(row):
            # A/a and operators may denote different symbols. Do not apply the
            # English-label spelling normalization to graph node identifiers.
            members = [unicodedata.normalize("NFKC", s).strip() for s in row]
            return tuple(sorted(members) if rule.get("unordered_members") else members)
        return sorted(relation(row) for row in got) == sorted(relation(row) for row in expected)
    if not all(isinstance(v, str) for v in got):
        return False
    actual, wanted = [token(v) for v in got], [token(v) for v in expected]
    return sorted(actual) == sorted(wanted) if rule["kind"] == "set" else actual == wanted


def grade_facts(gold, facts):
    """Missing/mismatching observable facts are misses, never removed from n."""
    details, pending = {}, []
    for name in gold.get("required_facts", {}):
        rule = gold.get("fact_rules", {}).get(name)
        if facts is None or rule is None:
            pending.append(name)
            continue
        clauses = []
        for clause in rule["all_of"]:
            names = {normal_text(n) for n in clause["names"]}
            found = [f for f in facts if normal_text(f["name"]) in names
                     and same_paper(f["paper_id"], clause["paper_id"])]
            correct = bool(found) and all(
                f["kind"] == clause["kind"] and _equivalent(f["value"], clause["expected"], clause)
                and (not clause.get("basis") or f["basis"] in clause["basis"])
                and (not clause.get("qualifiers") or normal_text(f.get("qualifiers", "")) in
                     {normal_text(q) for q in clause["qualifiers"]})
                for f in found
            )
            clauses.append({"passed": correct, "observed": [{k: f.get(k) for k in
                            ("name", "kind", "value", "paper_id", "basis", "qualifiers")} for f in found],
                            "reason": "match" if correct else "missing_observation" if not found
                            else "value_type_scope_or_conflict_mismatch"})
        details[name] = {"passed": bool(clauses) and all(c["passed"] for c in clauses), "clauses": clauses}
    return details, pending


def validate_fact_rules(gold):
    """Reject malformed grading contracts before executing paid queries."""
    rules = gold.get("fact_rules", {})
    if set(rules) - set(gold.get("required_facts", {})):
        raise ValueError("Fact rules must reference declared required facts")
    for rule in rules.values():
        if not rule.get("all_of"):
            raise ValueError("Empty fact rule")
        for clause in rule["all_of"]:
            if not clause.get("paper_id") or not clause.get("names") or not all(isinstance(n, str) and n for n in clause["names"]):
                raise ValueError("Fact rule requires a source and explicit field aliases")
            # Schema validation applies equally to gold values and agent values.
            bind_fields({"facts": [{"name": "gold", "kind": clause["kind"], "value": clause["expected"],
                        "paper_id": clause["paper_id"], "quote": "gold", "basis": "paper_fact"}], "rankings": []},
                        "gold [" + clause["paper_id"] + "#1]")
            if clause.get("absolute_tolerance", 0) < 0:
                raise ValueError("Negative fact tolerance")
            seen = {}
            for canonical, variants in clause.get("aliases", {}).items():
                for value in [canonical, *variants]:
                    normalized = normal_text(value)
                    if normalized in seen and seen[normalized] != normal_text(canonical):
                        raise ValueError("Ambiguous fact value alias")
                    seen[normalized] = normal_text(canonical)
