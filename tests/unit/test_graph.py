"""The graphs' deterministic parts: reply parsing, evidence collection, routing and finalize. Nodes
that call models are exercised by the live check (DESIGN §12: no fake LLMs)."""

from typing import Any

from langgraph.graph import END

from ara.graph import answer, read
from ara.graph.read import Found
from ara.graph.state import ABSTAIN, Answer, Claim, Evidence, Verdict
from ara.graph.wording import WORDING, language_name, say


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
        "uncited": [],
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
    # a direct answer that cites nothing is repaired too: it is the one uncited line that matters
    uncited: Any = {**state, "claims": [ok], "uncited": [Claim(index=0, text="BLEU", citations=[])]}
    assert answer.after_assemble(uncited) == "repair"


def test_a_repair_rewrites_only_the_failed_lines_in_their_place() -> None:
    """D40: the repair answers by line number; the other lines stay as they were (and keep their
    verdicts, which are keyed by text), a rewrite cut at its citations takes the failed line's
    place and paragraph, DROP or silence removes a line, and an abstaining direct answer removes
    the direct answer."""
    lines = [
        claim(0, "ROUGE"),
        Claim(index=1, text="Kept.", citations=["E1"], paragraph=0),
        Claim(index=2, text="Too broad.", citations=["E1"], paragraph=1),
        Claim(index=3, text="Gone.", citations=["E2"], paragraph=1),
        Claim(index=4, text="Silent.", citations=["E2"], paragraph=2),
    ]
    reply = "0: Answer: BLEU [E1]\n2: First part [E1]. Second part [E2].\n3: DROP\nnoise"
    claims, uncited = answer.amend(lines, {0, 2, 3, 4}, reply, {"E1", "E2"})
    assert [(c.index, c.text, c.citations, c.paragraph) for c in claims] == [
        (0, "BLEU", ["E1"], 0),
        (2, "Kept.", ["E1"], 0),
        (3, "First part.", ["E1"], 1),
        (4, "Second part.", ["E2"], 1),
    ]
    assert uncited == [] and claims[1].key == lines[1].key
    abstains = answer.amend(lines[:2], {0}, f"0: {ABSTAIN}\n", {"E1"})[0]
    assert [c.text for c in abstains] == ["Kept."] and not abstains[0].direct
    unknown = answer.amend(lines[:2], {1}, "1: Kept again. [E7]", {"E1"})
    assert [c.text for c in unknown[1]] == ["Kept again."], "a rewrite citing nothing known"


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


def test_an_unverified_direct_answer_keeps_its_verified_explanation() -> None:
    """D40 (was D21: the whole answer withheld): the direct answer is not delivered, the reply
    says why, and the verified lines stand; with none, it says no answer could be verified."""
    direct, good = claim(0, "ROUGE"), claim(1, "They report ROUGE-L.")
    state: Any = {
        "question": "q",
        "language": "English",
        "evidence": [evidence(1)],
        "claims": [direct, good],
        "uncited": [],
        "verdicts": {
            direct.key: Verdict(supported=False, problem="wrong metric"),
            good.key: Verdict(supported=True, problem=""),
        },
    }
    delivered = answer.finalize(state)["answer"]
    assert (delivered.abstained, delivered.withheld) == (True, True)
    assert (delivered.short, delivered.sentences) == (say("withheld", "English"), [good])
    assert [c.text for c in delivered.dropped] == ["ROUGE"]
    assert [e.id for e in delivered.evidence] == ["E1"]

    uncited: Any = {
        **state,
        "language": "Chinese",
        "claims": [good],
        "uncited": [Claim(index=0, text="ROUGE", citations=[])],
    }
    kept = answer.finalize(uncited)["answer"]
    assert (kept.short, kept.sentences) == (WORDING["withheld"]["Chinese"], [good])
    rejected = {good.key: Verdict(supported=False, problem="no")}
    nothing: Any = {**state, "verdicts": {**state["verdicts"], **rejected}}
    none = answer.finalize(nothing)["answer"]
    assert (none.short, none.sentences, none.withheld) == (say("unverified", "English"), [], True)


def test_a_model_abstention_keeps_its_verified_note() -> None:
    note = claim(1, "The evidence covers BLEU only.")
    state: Any = {
        "question": "q",
        "language": "English",
        "evidence": [evidence(1)],
        "claims": [note],
        "uncited": [],
        "verdicts": {note.key: Verdict(supported=True, problem="")},
    }
    delivered = answer.finalize(state)["answer"]
    assert (delivered.abstained, delivered.withheld, delivered.sentences) == (True, False, [note])
    assert delivered.short == ABSTAIN


def test_no_evidence_goes_straight_to_an_abstention() -> None:
    empty: Any = {"question": "q", "evidence": [], "language": "Chinese"}
    assert answer.start(empty) == "finalize"
    delivered = answer.finalize(empty)["answer"]
    assert delivered.abstained and delivered.short == WORDING["abstain"]["Chinese"]


def test_every_fixed_sentence_has_english_and_chinese_with_the_same_slots() -> None:
    """D40: the reply's fixed sentences follow the user's language; any other language gets
    English, and understand's name for the language is normalised."""
    import string

    def slots(text: str) -> set[str]:
        return {f for _, f, _, _ in string.Formatter().parse(text) if f}

    for key, wording in WORDING.items():
        assert set(wording) == {"English", "Chinese"}, key
        assert slots(wording["Chinese"]) <= slots(wording["English"]), key  # English fills all
    assert say("max_read", "French", n=3) == "I read at most 3 papers per turn: the first 3 named."
    assert [language_name(n) for n in ("Simplified Chinese", "中文", " English ", "")] == [
        "Chinese",
        "Chinese",
        "English",
        "English",
    ]


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


def test_a_paragraph_on_one_line_is_cut_into_its_cited_sentences() -> None:
    """Each cited sentence is a claim verified alone, even when the model writes a whole paragraph
    on one line; text after the last citation cites nothing (D19, D39)."""
    reply = (
        "Answer: It plans first [E1]\n\n"
        "The Planner writes a blueprint [E1]. Workers fetch evidence[E2, E3] and the Solver"
        " combines them. [E4] It saves tokens."
    )
    claims, uncited = answer.parse(reply, {"E1", "E2", "E3", "E4"})
    assert [(c.index, c.text, c.citations, c.paragraph) for c in claims] == [
        (0, "It plans first", ["E1"], 0),
        (1, "The Planner writes a blueprint.", ["E1"], 0),
        (2, "Workers fetch evidence", ["E2", "E3"], 0),
        (3, "and the Solver combines them.", ["E4"], 0),
    ]
    assert [c.text for c in uncited] == ["It saves tokens."]


def test_chinese_punctuation_closes_a_cited_piece_and_long_drafts_are_capped() -> None:
    """D40: "…token [E1]。其…" is cut after the full stop, with no piece made of punctuation."""
    semi, comma = "\uff1b", "\uff0c"  # full-width
    reply = f"Answer: 一个模型 [E1]\n它有 2.8 万亿参数 [E1]。其架构 [E2]{semi}第二句 [E1]{comma}"
    reply += "第三句。"
    claims, uncited = answer.parse(reply, {"E1", "E2"})
    texts = ["一个模型", "它有 2.8 万亿参数。", f"其架构{semi}", f"第二句{comma}"]
    assert [c.text for c in claims] == texts
    assert [c.text for c in uncited] == ["第三句。"]
    many = [claim(0, "a")] + [claim(n, f"line {n}") for n in range(1, 31)]
    kept, dropped = answer.capped(many, [])
    assert len(kept) == answer.MAX_CLAIMS + 1 and kept[0].direct
    assert [c.text for c in dropped] == [f"line {n}" for n in range(25, 31)]


def test_a_line_that_could_not_be_checked_is_checked_again_not_rewritten() -> None:
    """D41: a failed verify call (the cap, a timeout) judged nothing, so the repair round sends the
    line to verify once more instead of asking the model to rewrite it."""
    ok, lost = claim(0, "BLEU"), claim(1, "They use BLEU.")
    state: Any = {
        "question": "q",
        "claims": [ok, lost],
        "uncited": [],
        "evidence": [evidence(1)],
        "verdicts": {
            ok.key: Verdict(supported=True, problem=""),
            lost.key: Verdict(supported=False, problem=answer.UNCHECKED),
        },
        "repaired": False,
    }
    assert answer._failing(state) == [] and answer.after_assemble(state) == "repair"
    assert answer.to_verify(state) == "assemble", "not before the repair round"
    repaired: Any = {**state, "repaired": True}
    sends = answer.to_verify(repaired)
    assert isinstance(sends, list) and [s.arg["claim"] for s in sends] == [lost]
    assert answer.after_assemble(repaired) == "finalize"
