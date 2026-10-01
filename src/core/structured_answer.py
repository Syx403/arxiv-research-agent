"""Public output contracts and deterministic rendering; no evaluation answers.

The model produces the answer once. Its typed values and explanations are
rendered together and go through the ordinary source verifier as whole rows.
"""
from __future__ import annotations

import html
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

from src.core.answer_contract import FactValue, RequestedField, Ranking

NATIVE_CONTRACT_VERSION = "ara-answer-3"
TABLE_HEADER = "| Field / question | Value | Explanation / scope | Basis | Sources |\n|---|---|---|---|---|"


class GeneratedField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value: StrictBool | StrictInt | StrictFloat | StrictStr | list[StrictStr] | list[list[StrictStr]]
    explanation: str = Field(min_length=1, max_length=1200)
    evidence_indices: list[StrictInt] = Field(min_length=1, max_length=8)
    basis: Literal["paper_fact", "derived", "engineering_inference"]
    qualifiers: str = Field(default="", max_length=600)


def request_text(question: str, fields: tuple[RequestedField, ...]) -> str:
    if not fields:
        return question
    return question + (
        "\nPublic response requirements: answer the following named subquestions using the "
        "specified paper versions and value types, with original-source evidence. These are output "
        "requirements, not expected answers.\n"
        + json.dumps([f.model_dump(mode="json") for f in fields], ensure_ascii=False)
    )


def _cell(text: str) -> str:
    # Keep each answer a single table row; never let generated text add citations.
    return (html.escape(" ".join(text.split()), quote=False).replace("|", "&#124;")
            .replace("[", "&#91;").replace("]", "&#93;"))


def render_entry(entry: dict) -> str:
    """Also used to audit that the visible value equals the machine value."""
    value = json.dumps(entry["value"], ensure_ascii=False, allow_nan=False)
    value = value.replace("|", "\\u007c").replace("<", "\\u003c").replace(">", "\\u003e")
    if re.search(r"\[[^\]#\s]+#\d+\]", value):
        raise ValueError("citations_are_program_owned")
    refs = " ".join(f"[{r['paper_id']}#{r['chunk_id']}]" for r in entry["source_refs"])
    explanation = entry["explanation"] + ("; " + entry["qualifiers"] if entry.get("qualifiers") else "")
    return (f"| {_cell(entry['name'] + ': ' + entry['description'])} | {value} | "
            f"{_cell(explanation)} | {entry['basis']} | {refs} |")


def validate_entry(entry: dict) -> RequestedField:
    """Validate stored/public entries without trusting model-created identities."""
    field = RequestedField.model_validate({k: entry[k] for k in RequestedField.model_fields if k in entry})
    refs = entry["source_refs"]
    if (not refs or len(refs) > 8 or any(type(r["chunk_id"]) is not int or r["chunk_id"] < 0 for r in refs)
            or len({(r["paper_id"], r["chunk_id"]) for r in refs}) != len(refs)
            or {r["paper_id"] for r in refs} != set(field.paper_ids)):
        raise ValueError("structured_source_scope_mismatch")
    if not isinstance(entry["explanation"], str) or not entry["explanation"].strip():
        raise ValueError("missing_structured_explanation")
    if entry["basis"] not in {"paper_fact", "derived", "engineering_inference"}:
        raise ValueError("invalid_structured_basis")
    if not isinstance(entry.get("qualifiers", ""), str):
        raise ValueError("invalid_structured_qualifiers")
    if field.kind == "paper_order":
        Ranking(paper_ids=entry["value"], quote="answer")
        if set(entry["value"]) != set(field.paper_ids):
            raise ValueError("ranking_must_order_all_requested_papers")
    else:
        FactValue.model_validate({k: entry[k] for k in ("name", "kind", "value", "basis", "qualifiers")})
        field.validate_value_domain(entry["value"])
    return field


def bind_generated_fields(raw, fields: tuple[RequestedField, ...], evidence) -> tuple[list[dict], list[str]]:
    if not isinstance(raw, dict) or set(raw) != {"fields"} or not isinstance(raw["fields"], list):
        raise ValueError("invalid_structured_answer_envelope")
    if len(raw["fields"]) > 12:
        raise ValueError("too_many_structured_fields")
    requested = {f.name: f for f in fields}
    names = [r.get("name") for r in raw["fields"] if isinstance(r, dict)]
    entries, errors = {}, []
    for row in raw["fields"]:
        try:
            item = GeneratedField.model_validate(row)
            if item.name not in requested or names.count(item.name) != 1:
                raise ValueError("unknown_or_duplicate_field")
            field = requested[item.name]
            if (len(set(item.evidence_indices)) != len(item.evidence_indices)
                    or any(i < 0 or i >= len(evidence) for i in item.evidence_indices)):
                raise ValueError("invalid_evidence_index")
            entry = {**field.model_dump(mode="json"), **item.model_dump(exclude={"evidence_indices"}),
                     "source_refs": [{"paper_id": evidence[i].paper_id, "chunk_id": evidence[i].chunk_id}
                                     for i in item.evidence_indices]}
            validate_entry(entry)
            entry["claim"] = render_entry(entry)
            entries[item.name] = entry
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc)[:300])
    return [entries[f.name] for f in fields if f.name in entries], errors


def entry_fields(entry: dict) -> dict:
    """Project validated native entries into the common scorer contract."""
    field = validate_entry(entry)
    claim = render_entry(entry)
    if entry.get("claim") != claim:
        raise ValueError("structured_render_mismatch")
    if field.kind == "paper_order":
        return {"facts": [], "rankings": [{"paper_ids": entry["value"], "quote": claim}]}
    fact = {k: entry[k] for k in ("name", "kind", "value", "basis", "qualifiers")}
    return {"facts": [{**fact, "paper_id": field.paper_ids[0], "quote": claim}], "rankings": []}
