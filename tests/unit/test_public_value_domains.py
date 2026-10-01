"""An answer vocabulary is public; its correct selections remain private labels."""

import json
from pathlib import Path

import pytest

from scripts.eval_agent import load_dataset
from src.core.answer_contract import RequestedField
from src.core.structured_answer import bind_generated_fields, entry_fields, request_text
from src.retrieval.index.types import Hit


PID = "arxiv:2601.00001v1"


@pytest.mark.parametrize(
    "kind,good,bad",
    [
        ("text", "queue", "QUEUE"),
        ("sequence", ["queue", "stack"], ["queue", "heap"]),
        ("set", ["stack"], ["stack", "heap"]),
        ("relations", [["queue", "stack"]], [["queue", "heap"]]),
    ],
)
def test_generation_and_saved_entry_audit_enforce_public_domain(kind, good, bad):
    field = RequestedField(
        name="mechanism",
        kind=kind,
        description="Choose mechanisms.",
        paper_ids=(PID,),
        allowed_values=("queue", "stack"),
    )

    def bind(value):
        return bind_generated_fields(
            {
                "fields": [
                    {
                        "name": "mechanism",
                        "value": value,
                        "basis": "paper_fact",
                        "explanation": "Supported mechanism.",
                        "evidence_indices": [0],
                    }
                ]
            },
            (field,),
            [Hit(1, PID, "Method", "Source", 1)],
        )

    entries, errors = bind(good)
    assert entries and not errors
    assert not bind(bad)[0] and bind(bad)[1]
    entries[0]["value"] = bad
    with pytest.raises(ValueError, match="outside_public_domain"):
        entry_fields(entries[0])
    assert "allowed_values" in request_text("Question", (field,))


@pytest.mark.parametrize(
    "kind,domain",
    [("number", ["one", "two"]), ("text", ["only"]), ("text", ["a", "a"]), ("text", ["a", ""])],
)
def test_invalid_domains_rejected_at_public_boundary(kind, domain):
    with pytest.raises(ValueError):
        RequestedField(
            name="x", kind=kind, description="Choose.", paper_ids=(PID,), allowed_values=domain
        )


def test_v6_validates_rules_against_public_domains_without_exposing_gold(tmp_path):
    suite = load_dataset(Path("docs/eval_design/structured_cases_v6.json"))
    spec = next(c for c in suite["cases"] if c["id"].startswith("s05"))["turns"][0]
    assert spec["response_fields"][0]["allowed_values"] == ["A", "B", "C"]
    assert len(spec["gold"]["fact_rules"]["C_start_condition"]["all_of"]) == 3
    assert all(
        "expected" not in f and "aliases" not in f
        for c in suite["cases"]
        for f in c["turns"][0]["response_fields"]
    )
    spec["gold"]["fact_rules"]["dependency_edges"]["all_of"][0]["expected"] = [["A", "D"]]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="outside_public_domain"):
        load_dataset(path)
