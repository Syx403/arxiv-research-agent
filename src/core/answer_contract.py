"""Public field types and legacy observations, not independent truth judgments.

Native requests use RequestedField with deterministic rendering in
structured_answer. The opt-in legacy verifier supplies ModelFields; code binds
its quotes and sources. No gold labels or paper-specific rules belong here.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictFloat, StrictBool, StrictStr, model_validator


CONTRACT_VERSION = "ara-answer-2"
_REF = re.compile(r"\[(?P<pid>[^\]#\s]+)#\d+\]")


class FactValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,99}$")
    kind: Literal["boolean", "number", "text", "sequence", "set", "relations"]
    value: StrictBool | StrictInt | StrictFloat | StrictStr | list[StrictStr] | list[list[StrictStr]]
    basis: Literal["paper_fact", "derived", "engineering_inference"]
    # Retained as data, never silently erased to make a value match a gold label.
    qualifiers: str = Field(default="", max_length=600)

    @model_validator(mode="after")
    def check_value(self):
        value = self.value
        valid = {
            "boolean": type(value) is bool,
            "number": type(value) in {int, float},
            "text": isinstance(value, str) and bool(value.strip()),
            "sequence": isinstance(value, list) and bool(value) and all(isinstance(v, str) and v.strip() for v in value),
            "set": isinstance(value, list) and bool(value) and all(isinstance(v, str) and v.strip() for v in value),
            "relations": isinstance(value, list) and bool(value) and all(
                isinstance(v, list) and len(v) >= 2 and all(isinstance(s, str) and s.strip() for s in v)
                for v in value
            ),
        }[self.kind]
        if not valid:
            raise ValueError("fact_value_type_mismatch")
        if self.kind in {"set", "sequence"} and len(value) != len(set(value)):
            raise ValueError("duplicate_fact_items")
        if self.kind == "relations" and len(value) != len({tuple(v) for v in value}):
            raise ValueError("duplicate_fact_relations")
        if self.kind == "number":
            import math
            if not math.isfinite(value):
                raise ValueError("nonfinite_fact_number")
        return self


class Fact(FactValue):
    paper_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=12000)


class ModelFact(FactValue):
    source_index: StrictInt = Field(ge=0)


class ModelRanking(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_indices: list[StrictInt] = Field(min_length=2, max_length=8)


class ModelFields(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    facts: list[ModelFact] = Field(max_length=12)
    rankings: list[ModelRanking] = Field(max_length=1)


class RequestedField(BaseModel):
    """A client's public output request. It must never contain expected answers."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,99}$")
    kind: Literal["boolean", "number", "text", "sequence", "set", "relations", "paper_order"]
    description: str = Field(min_length=1, max_length=600)
    paper_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    allowed_values: tuple[str, ...] = Field(default=(), max_length=32,
        description="Optional public vocabulary: scalar options, list members, or relation node symbols; never identifies the correct choice/order/edges.")

    @model_validator(mode="after")
    def scope(self):
        if self.allowed_values:
            if self.kind not in {"text", "sequence", "set", "relations"}:
                raise ValueError("value_domain_not_applicable_to_kind")
            if (len(self.allowed_values) < 2 or len(set(self.allowed_values)) != len(self.allowed_values)
                    or any(not v.strip() or len(v) > 120 for v in self.allowed_values)):
                raise ValueError("invalid_public_value_domain")
        if len(set(self.paper_ids)) != len(self.paper_ids):
            raise ValueError("duplicate_requested_source")
        if self.kind != "paper_order" and len(self.paper_ids) != 1:
            raise ValueError("a_fact_field_has_one_declared_source_paper")
        if self.kind == "paper_order" and len(self.paper_ids) < 2:
            raise ValueError("an_order_needs_multiple_papers")
        for pid in self.paper_ids:
            if not re.fullmatch(r"arxiv:\d{4}\.\d{4,5}v\d+", pid):
                raise ValueError("requested_fields_require_explicit_arxiv_versions")
        return self

    def validate_value_domain(self, value):
        if not self.allowed_values:
            return
        members = [value] if self.kind == "text" else (
            [symbol for relation in value for symbol in relation] if self.kind == "relations" else value)
        if any(member not in self.allowed_values for member in members):
            raise ValueError("value_outside_public_domain")


class Ranking(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_ids: list[str] = Field(min_length=2, max_length=8)
    quote: str = Field(min_length=1, max_length=12000)

    @model_validator(mode="after")
    def unique_papers(self):
        bases = [re.sub(r"v\d+$", "", pid.removeprefix("arxiv:")) for pid in self.paper_ids]
        if len(bases) != len(set(bases)):
            raise ValueError("duplicate_ranked_paper")
        return self


class ResultFields(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    facts: list[Fact] = Field(max_length=12)
    rankings: list[Ranking] = Field(max_length=1)


def bind_fields(raw, claim: str) -> dict:
    """Require literal answer quotes and exact cited identities; never infer truth."""
    parsed = ResultFields.model_validate(raw)
    cited = set(_REF.findall(claim))

    def canonical(pid):
        if pid in cited:
            return pid
        if "arxiv:" + pid in cited:
            return "arxiv:" + pid
        raise ValueError("field_source_not_cited_in_claim")

    facts, rankings = [], []
    for fact in parsed.facts:
        if fact.quote not in claim or not re.search(r"\w", _REF.sub("", fact.quote)):
            raise ValueError("fact_quote_not_in_claim")
        facts.append(fact.model_copy(update={"paper_id": canonical(fact.paper_id)}).model_dump())
    for ranking in parsed.rankings:
        if ranking.quote not in claim or not re.search(r"\w", _REF.sub("", ranking.quote)):
            raise ValueError("ranking_quote_not_in_claim")
        rankings.append(ranking.model_copy(update={"paper_ids": [canonical(p) for p in ranking.paper_ids]}).model_dump())
    return {"facts": facts, "rankings": rankings}


def claim_papers(claim: str) -> list[str]:
    return list(dict.fromkeys(_REF.findall(claim)))


def bind_model_fields(raw, claim: str) -> dict:
    """The model picks only from numbered sources; code owns quotes and IDs."""
    parsed = ModelFields.model_validate(raw)
    papers = claim_papers(claim)
    def identity(index):
        if not 0 <= index < len(papers):
            raise ValueError("unknown_result_source_index")
        return papers[index]
    facts = [{**f.model_dump(exclude={"source_index"}), "paper_id": identity(f.source_index), "quote": claim}
             for f in parsed.facts]
    rankings = [{"paper_ids": [identity(i) for i in r.source_indices], "quote": claim} for r in parsed.rankings]
    return bind_fields({"facts": facts, "rankings": rankings}, claim)


RESULT_FIELDS_PROMPT = (
    " Also return result_fields:{facts:[],rankings:[]} as a typed, public observation of THIS claim. "
    "These fields serialize the delivered answer, not an independent answer or a grade. "
    "If supports=false return empty arrays. Extract ONLY explicit, unambiguous values in this "
    "claim; never fill a fact from source text that the answer does not state. Omit ambiguous facts. "
    "For facts use {name,kind,value,source_index,basis,qualifiers}. name is a precise English "
    "lower_snake_case attribute (include actor, hierarchy and quantity where needed); value uses "
    "source terminology in English, lower_snake_case for textual concepts, preserving named symbols. "
    "Kinds: boolean (JSON true/false), number (JSON number), text (string), sequence (ordered string "
    "list), set (unordered string list), relations (list of directed [from,to] string pairs or "
    "explicit parallel groups). Preserve numbers, signs, units, hierarchy, scope and conditions. "
    "A multi-step list stated in this one claim is ONE sequence fact. Do not invent missing steps "
    "or split an already explicit complete sequence. Do not merge facts across other claims. "
    "For a labelled single step use a precise name including its position. source_index is an "
    "INTEGER from the numbered paper identities supplied by the program. Do not return a paper_id, "
    "chunk ID, quote or source quotation: the program binds THIS entire answer claim and its sources. "
    "basis is paper_fact, derived (a user "
    "example or arithmetic derived from source mechanics), or engineering_inference. qualifiers "
    "is a STRING retaining necessary conditions/units; use an empty string when none, never [] or {}. "
    "At most 12 facts; skip incidental detail. "
    "Only if THIS claim explicitly orders papers as recommendations, add ONE ranking with "
    "{source_indices:[integers from the numbered paper identities, best first]}. Mention order or "
    "citation order alone is NOT a recommendation. Never invent a ranking from the sources. "
    "Do not supply rankings for unordered comparisons or single-paper answers. No gold labels "
    "are supplied: faithfully serialize the answer even if an evaluator might prefer another value."
)
