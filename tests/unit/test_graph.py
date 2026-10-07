"""The graphs' deterministic parts: reply parsing, evidence collection, routing and finalize. Nodes
that call models are exercised by the live check (DESIGN §12: no fake LLMs)."""

from typing import Any

from langgraph.graph import END

from ara.graph import answer, read
from ara.graph.read import Found
from ara.graph.state import ABSTAIN, Claim, Evidence, Verdict


def evidence(n: int, paragraph: int = 0, text: str = "") -> Evidence:
    return Evidence(
        id=f"E{n}",
        paper_id="qasper:1",
        chunk_id=n,
        paragraph=paragraph,
        heading_path="T",
        text=text or f"sentence {n}",
    )


def test_parse_splits_the_direct_answer_cited_lines_and_uncited_lines() -> None:
    reply = (
        "Answer: BLEU [E1]\nThey report BLEU on WMT14 [E1, E2].\nIt is the best metric.\nSee [E9]."
    )
    claims, uncited = answer.parse(reply, {"E1", "E2"})
    assert [(c.index, c.text, c.citations) for c in claims] == [
        (0, "BLEU", ["E1"]),
        (1, "They report BLEU on WMT14.", ["E1", "E2"]),
    ]
    assert [c.text for c in uncited] == ["It is the best metric.", "See."]


def test_an_abstaining_answer_is_neither_claim_nor_dropped() -> None:
    claims, uncited = answer.parse(f"Answer: {ABSTAIN}", {"E1"})
    assert (claims, uncited) == ([], [])


def test_collect_numbers_distinct_sentences_and_requeries_once() -> None:
    first = Found(round=0, sentences=[evidence(1), evidence(1), evidence(2)], missing=["dataset"])
    state: Any = {"found": [first], "question": "q", "documents": [7]}
    collected: Any = read.collect(state)
    assert [e.id for e in collected["evidence"]] == ["E1", "E2"]
    assert collected["requery"] == ["dataset"]

    after: Any = {**state, **collected}
    sends = read.after_collect(after)
    assert isinstance(sends, list)
    assert [s.arg["query"] for s in sends] == ["dataset"]

    second = Found(round=1, sentences=[], missing=["dataset"])
    later: Any = {**after, "found": [first, second]}
    again: Any = read.collect(later)
    assert again["requery"] == []
    final: Any = {**after, **again}
    assert read.after_collect(final) == END


def claim(index: int, text: str) -> Claim:
    return Claim(index=index, text=text, citations=["E1"])


def test_only_unverified_claims_are_sent_and_one_repair_is_allowed() -> None:
    ok, bad = claim(0, "BLEU"), claim(1, "It always wins.")
    state: Any = {
        "claims": [ok, bad],
        "evidence": [evidence(1)],
        "verdicts": {ok.key: Verdict(supported=True, problem="")},
        "repaired": False,
    }
    sends = answer.to_verify(state)
    assert isinstance(sends, list)
    assert [s.arg["claim"] for s in sends] == [bad]
    state["verdicts"][bad.key] = Verdict(supported=False, problem="too broad")
    assert answer.to_verify(state) == "assemble"
    assert answer.after_assemble(state) == "repair"
    repaired: Any = {**state, "repaired": True}
    assert answer.after_assemble(repaired) == "finalize"


def test_finalize_delivers_only_verified_lines() -> None:
    direct, good, bad = claim(0, "BLEU"), claim(1, "They use BLEU."), claim(2, "It always wins.")
    verdicts = {
        direct.key: Verdict(supported=True, problem=""),
        good.key: Verdict(supported=True, problem=""),
        bad.key: Verdict(supported=False, problem="too broad"),
    }
    state: Any = {
        "question": "q",
        "evidence": [evidence(1), evidence(2)],
        "claims": [direct, good, bad],
        "uncited": [],
        "verdicts": verdicts,
    }
    delivered = answer.finalize(state)["answer"]
    assert (delivered.short, delivered.abstained) == ("BLEU", False)
    assert [c.text for c in delivered.sentences] == ["They use BLEU."]
    assert [c.text for c in delivered.dropped] == ["It always wins."]
    assert [e.id for e in delivered.evidence] == ["E1"]


def test_an_unverified_direct_answer_becomes_an_abstention() -> None:
    direct = claim(0, "ROUGE")
    state: Any = {
        "question": "q",
        "evidence": [evidence(1)],
        "claims": [direct],
        "uncited": [],
        "verdicts": {direct.key: Verdict(supported=False, problem="wrong metric")},
    }
    delivered = answer.finalize(state)["answer"]
    assert (delivered.short, delivered.abstained) == (ABSTAIN, True)


def test_no_evidence_goes_straight_to_an_abstention() -> None:
    empty: Any = {"question": "q", "evidence": []}
    assert answer.start(empty) == "finalize"
    delivered = answer.finalize(empty)["answer"]
    assert delivered.abstained


def test_the_subgraphs_compile() -> None:
    assert {"ingest", "gather", "collect"} <= set(read.build().get_graph().nodes)
    assert {"synthesize", "verify", "repair", "finalize"} <= set(answer.build().get_graph().nodes)


def test_selection_labels_tolerate_appended_sentences() -> None:
    assert read._labels(["S8: All techniques ...", " S10", "none"]) == ["S8", "S10"]
