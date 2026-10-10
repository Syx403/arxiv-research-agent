"""The graphs' deterministic parts: reply parsing, evidence collection, routing and finalize. Nodes
that call models are exercised by the live check (DESIGN §12: no fake LLMs)."""

from typing import Any

from langgraph.graph import END

from ara.graph import answer, read
from ara.graph.read import Found
from ara.graph.state import ABSTAIN, Answer, Claim, Evidence, Verdict


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


def test_an_answer_is_explained_in_paragraphs() -> None:
    """An empty line between explanation lines starts a paragraph; the reply shows each paragraph
    as prose after the direct answer (D37)."""
    reply = "Answer: BLEU [E1]\n\nIt uses WMT14. [E1]\nScores rise. [E2]\n\n\nIt is cheap. [E2]"
    claims, _ = answer.parse(reply, {"E1", "E2"})
    assert [(c.index, c.paragraph) for c in claims] == [(0, 0), (1, 0), (2, 0), (3, 1)]
    delivered = Answer(
        question="q",
        short="BLEU",
        abstained=False,
        sentences=claims[1:],
        dropped=[],
        checked=4,
        rejected=[],
        evidence=[],
    )
    assert delivered.render() == (
        "BLEU\n\nIt uses WMT14. [E1] Scores rise. [E2]\n\nIt is cheap. [E2]"
    )
    assert claims[1].key == claim(1, "It uses WMT14.").key, "paragraphs do not change identity"


def test_an_abstaining_answer_is_neither_claim_nor_dropped() -> None:
    assert answer.parse(f"Answer: {ABSTAIN}", {"E1"}) == ([], [])
    claims, uncited = answer.parse(f"Answer: {ABSTAIN} [E1]\nE1 covers BLEU only. [E1]", {"E1"})
    assert ([c.index for c in claims], uncited) == ([1], [])


def test_only_the_direct_answer_is_verified_with_its_question() -> None:
    direct, line = claim(0, "no"), claim(1, "The weights are not shared.")
    with_question = answer.verify_prompt([evidence(1)], direct, "Are the weights shared?")
    assert "Question: Are the weights shared?" in with_question.item[0].text
    assert "Question" not in answer.verify_prompt([evidence(1)], line, "q").item[0].text
    assert direct.key != claim(1, "no").key


def found(round: int, document: int, query: str, n: list[int], missing: list[str]) -> Found:
    return Found(
        round=round,
        document=document,
        query=query,
        sentences=[evidence(i) for i in n],
        missing=missing,
    )


def test_collect_numbers_sentences_and_requeries_each_document_for_its_own_gaps() -> None:
    first = [found(0, 7, "q", [1, 1, 2], ["dataset"]), found(0, 8, "q", [], ["dataset", "metric"])]
    state: Any = {"found": first, "question": "q", "documents": [7, 8]}
    collected: Any = read.collect(state)
    assert [e.id for e in collected["evidence"]] == ["E1", "E2"]
    assert collected["missing"] == []
    after: Any = {**state, **collected}
    sends = read.after_collect(after)
    assert isinstance(sends, list)
    assert [(s.arg["documents"], s.arg["query"], s.arg["round"]) for s in sends] == [
        ([7], "dataset", 1),
        ([8], "dataset", 1),
        ([8], "metric", 1),
    ]
    assert all(s.node == "search" for s in sends)


def test_the_question_is_searched_once_over_all_papers_and_selected_per_paper() -> None:
    state: Any = {"question": "q", "documents": [7, 8, 7]}
    searches = read.to_search(state)
    assert isinstance(searches, list)
    [search] = searches
    nothing: Any = {**state, "documents": []}
    assert read.to_search(nothing) == "collect"
    assert search.arg["documents"] == [7, 8] and search.arg["round"] == 0
    first = [select_task(7, 0), select_task(8, 0)]
    staged: Any = {"staged": first}
    assert [s.arg["document"] for s in read.to_select(staged)] == [7, 8]
    later: Any = {"staged": [*first, select_task(8, 1)], "requeried": True}
    assert [(s.arg["document"], s.arg["round"]) for s in read.to_select(later)] == [(8, 1)]


def test_after_the_requery_only_aspects_no_search_covered_are_missing() -> None:
    first = [
        found(0, 7, "q", [1], ["metric", "baseline"]),
        found(0, 8, "q", [2], ["dataset", "baseline"]),
    ]
    again = [  # the requery's own `missing` lists are ignored
        found(1, 7, "metric", [], ["everything"]),
        found(1, 8, "dataset", [], ["everything"]),
        found(1, 7, "baseline", [], []),
        found(1, 8, "baseline", [], []),
    ]
    state: Any = {"found": first + again, "question": "q", "requeried": True}
    collected: Any = read.collect(state)
    # document 8 covered "metric" and document 7 covered "dataset" in the first round
    assert collected["missing"] == ["baseline"]
    final: Any = {**state, **collected}
    assert read.after_collect(final) == END

    third: Any = {**state, "found": [*first, *again, found(1, 7, "baseline", [3], [])]}
    assert read.collect(third)["missing"] == []


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
    state["question"] = "q"
    sends = answer.to_verify(state)
    assert isinstance(sends, list)
    assert [(s.arg["claim"], s.arg["question"]) for s in sends] == [(bad, "q")]
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
    assert (delivered.checked, [c.text for c in delivered.rejected]) == (3, ["It always wins."])
    assert [e.id for e in delivered.evidence] == ["E1"]


def test_rejections_of_the_first_draft_are_kept_after_repair() -> None:
    fixed, gone = claim(1, "They use BLEU."), claim(1, "It always wins.")
    state: Any = {
        "question": "q",
        "evidence": [evidence(1)],
        "claims": [claim(0, "BLEU"), fixed],
        "uncited": [],
        "rejected": [gone],
        "verdicts": {
            claim(0, "BLEU").key: Verdict(supported=True, problem=""),
            fixed.key: Verdict(supported=True, problem=""),
            gone.key: Verdict(supported=False, problem="too broad"),
        },
    }
    delivered = answer.finalize(state)["answer"]
    assert [c.text for c in delivered.sentences] == ["They use BLEU."]
    assert (delivered.checked, [c.text for c in delivered.rejected]) == (3, ["It always wins."])


def test_an_unverified_direct_answer_withholds_the_whole_answer() -> None:
    direct, good = claim(0, "ROUGE"), claim(1, "They report ROUGE-L.")
    state: Any = {
        "question": "q",
        "evidence": [evidence(1)],
        "claims": [direct, good],
        "uncited": [],
        "verdicts": {
            direct.key: Verdict(supported=False, problem="wrong metric"),
            good.key: Verdict(supported=True, problem=""),
        },
    }
    delivered = answer.finalize(state)["answer"]
    assert (delivered.short, delivered.abstained, delivered.sentences) == (ABSTAIN, True, [])
    assert [c.text for c in delivered.dropped] == ["ROUGE", "They report ROUGE-L."]
    assert delivered.evidence == []

    uncited: Any = {
        **state,
        "claims": [good],
        "uncited": [Claim(index=0, text="ROUGE", citations=[])],
    }
    assert answer.finalize(uncited)["answer"].sentences == []


def test_a_model_abstention_keeps_its_verified_note() -> None:
    note = claim(1, "The evidence covers BLEU only.")
    state: Any = {
        "question": "q",
        "evidence": [evidence(1)],
        "claims": [note],
        "uncited": [],
        "verdicts": {note.key: Verdict(supported=True, problem="")},
    }
    delivered = answer.finalize(state)["answer"]
    assert (delivered.abstained, delivered.sentences) == (True, [note])


def test_no_evidence_goes_straight_to_an_abstention() -> None:
    empty: Any = {"question": "q", "evidence": []}
    assert answer.start(empty) == "finalize"
    delivered = answer.finalize(empty)["answer"]
    assert delivered.abstained


def test_the_subgraphs_compile() -> None:
    assert {"ingest", "search", "select", "collect"} <= set(read.build().get_graph().nodes)
    assert {"synthesize", "verify", "repair", "finalize"} <= set(answer.build().get_graph().nodes)


def test_selection_labels_tolerate_appended_sentences() -> None:
    assert read._labels(["S8: All techniques ...", " S10", "none"]) == ["S8", "S10"]


def select_task(document: int, round: int) -> Any:
    return {"question": "q", "query": "q", "document": document, "round": round, "passages": []}


def test_a_context_line_is_split_off_and_kept_out_of_verification() -> None:
    reply = "Context: We had read ReAct, which says nothing on tokens.\nAnswer: 5x fewer [E1]"
    context, body = answer.opening(reply)
    assert context == "We had read ReAct, which says nothing on tokens."
    assert [c.text for c in answer.parse(body, {"E1"})[0]] == ["5x fewer"]
    assert answer.opening("Answer: yes [E1]") == ("", "Answer: yes [E1]")


def test_requeries_are_capped_per_turn_and_shared_out_over_the_papers() -> None:
    """Three papers missing three aspects each: three requeries in all, one per paper first,
    since each is a rerank call and rerank calls queue one per 6 s (D39)."""
    first = [found(0, d, "q", [], ["a", "b", "c"]) for d in (7, 8, 9)]
    assert read.requeries(first) == [(7, "a"), (8, "a"), (9, "a")]
    assert read.requeries([found(0, 7, "q", [], ["a", "a", "b"])]) == [(7, "a"), (7, "b")]
