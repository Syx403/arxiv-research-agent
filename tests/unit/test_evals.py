import json
import re
from typing import Any

import pytest

from ara.graph.state import ResearchRequest, Verdict
from evals.graders.discovery import score as discovery_score
from evals.graders.retrieval import ndcg_at, recall_at, reciprocal_rank, score
from evals.qasper import questions, select
from evals.report import _detection
from evals.stats import mean_ci, paired
from evals.suites import s3, s4, s5, s6

CJK = re.compile("[\u4e00-\u9fff]")  # project data is English (D24)


def test_retrieval_metrics() -> None:
    ranking = [5, 2, 9, 1]
    assert recall_at(ranking, {2, 1}, 2) == 0.5
    assert reciprocal_rank(ranking, {9}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranking, {7}) == 0.0
    assert ndcg_at([2, 5], {2}, 10) == 1.0
    assert ndcg_at([5, 2], {2}, 10) == pytest.approx(1 / 1.5849625)


def test_a_paragraph_split_into_chunks_counts_once_at_its_first_chunk() -> None:
    ranking = [4, 4, 2]  # paragraph 4 was split into two chunks
    assert recall_at(ranking, {2}, 2) == 0.0  # k counts chunks
    assert ndcg_at(ranking, {4}, 10) == 1.0


def test_paired_difference_counts_items() -> None:
    d = paired([1.0, 0.5, 1.0, 0.0], [1.0, 0.0, 0.5, 0.5])
    assert (d.mean, d.better, d.worse) == (0.125, 2, 1)
    assert d.low <= d.mean <= d.high


def test_score_takes_the_best_annotator() -> None:
    assert score([3, 4], [{9}, {3}])["mrr"] == 1.0


def test_bootstrap_interval_contains_the_mean() -> None:
    mean, low, high = mean_ci([0.0, 1.0, 1.0, 0.5])
    assert low <= mean <= high
    assert mean == 0.625


def record(qid: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "1912.01214",
        "title": "T",
        "abstract": "A.",
        "full_text": {"section_name": ["S"], "paragraphs": [["P one.", "P two."]]},
        "qas": {"question_id": [qid], "question": ["Q?"], "answers": [{"answer": answers}]},
    }


def answer(evidence: list[str], unanswerable: bool = False) -> dict[str, Any]:
    return {"unanswerable": unanswerable, "evidence": evidence}


def test_questions_need_every_answering_annotator_to_cite_paragraphs() -> None:
    [kept] = questions(record("q1", [answer(["P two."]), answer(["P one.", "P two."])]))
    assert kept.references == (frozenset({2}), frozenset({1, 2}))  # index 0 is the abstract
    assert not list(questions(record("q2", [answer(["FLOAT SELECTED: Table 1"])])))
    assert not list(questions(record("q3", [answer([], unanswerable=True)])))


def test_selection_is_reproducible() -> None:
    data = {f"p{i}": record(f"q{i}", [answer(["P one."])]) for i in range(5)}
    assert select(data, seed=7, papers=3, per_paper=1) == select(
        data, seed=7, papers=3, per_paper=1
    )


def test_code_perturbations_change_exactly_one_thing() -> None:
    from evals.suites.s5_data import change_number, negate

    assert change_number("The corpus has 26972 sentences in 3 sets.") == (
        "The corpus has 35064 sentences in 3 sets."
    )
    assert change_number("Accuracy rose to 0.85.") == "Accuracy rose to 3.35."
    assert change_number("No numbers here.") is None
    assert negate("The model can handle long inputs.") == "The model can not handle long inputs."
    assert negate("Results improve.") is None


def test_s5_scores_unsupported_as_the_positive_class() -> None:
    item: Any = type("I", (), {"label": "unsupported"})()
    assert s5.score(Verdict(supported=False, problem="wrong number"), item) == {
        "correct": 1.0,
        "flagged": 1.0,
        "unsupported": 1.0,
    }
    item.label = "supported"
    assert s5.score(Verdict(supported=False, problem="x"), item)["correct"] == 0.0


def test_s5_detection_reports_precision_recall_and_kinds() -> None:
    scores = {
        "a:unsupported": {"flagged": 1.0, "unsupported": 1.0},
        "b:unsupported": {"flagged": 0.0, "unsupported": 1.0},
        "a:supported": {"flagged": 1.0, "unsupported": 0.0},
        "b:supported": {"flagged": 0.0, "unsupported": 0.0},
    }
    kinds = {"a:unsupported": "number", "b:unsupported": "entity"}
    [_, line] = _detection({"luna": scores}, kinds)
    assert "precision 0.50 (1/2), recall 0.50 (1/2), F1 0.50" in line
    assert "entity 0/1, number 1/1" in line


def test_s5_data_is_reviewed_and_paired_with_a_split_each() -> None:
    claims = s5._claims()
    assert len(claims) == 58
    assert [c["label"] for c in claims] == ["supported", "unsupported"] * 29
    assert len({c["sentence"] for c in claims}) == 29
    assert set(s5.splits().values()) == {"dev", "test"}


def test_s2_counts_rejections_over_both_drafts() -> None:
    from evals.qasper import Gold
    from evals.suites import s2

    item: Any = type("I", (), {"golds": (Gold("extractive", "BLEU", frozenset({3})),)})()
    result = {
        "short": "BLEU",
        "abstained": False,
        "cited_paragraphs": [3],
        "delivered": 2,
        "dropped": 0,
        "checked": 4,
        "rejected": 1,
    }
    metrics = s2.score(result, item)
    assert (metrics["verified_share"], metrics["rejection_rate"]) == (1.0, 0.25)


def test_s2_repeats_each_unanswerable_item() -> None:
    from evals.suites import s2

    items, _ = s2.load()
    ids = [i.id for i in items]
    unanswerable = [i for i in items if all(g.kind == "unanswerable" for g in i.golds)]
    assert len(items) == 30 + 5 * s2.TRIALS and len(set(ids)) == len(ids)
    assert len(unanswerable) == 5 * s2.TRIALS
    assert set(s2.splits()) == set(ids)


def test_discovery_metrics_grade_the_pool_and_the_list() -> None:
    gold = frozenset({"a", "b", "c", "d"})
    metrics = discovery_score(["a", "b", "x"], ["a", "x"], ["x", "a"], gold)
    assert metrics == {
        "pool_recall": 0.5,
        "shortlist_recall": 0.25,
        "hit_at_5": 1.0,
        "listed": 2.0,
        "gold_precision_at_5": 0.5,
    }
    assert "gold_precision_at_5" not in discovery_score([], [], [], gold)


def test_the_s3_manifest_commits_query_ids_only() -> None:
    manifest = json.loads(s3.MANIFEST.read_text())
    assert {tuple(sorted(e)) for e in manifest["items"]} == {("id", "set", "split")}
    assert [e["split"] for e in manifest["items"]].count("test") == 15


def test_s4_grades_every_field_so_an_added_value_is_an_error() -> None:
    [item] = [i for i in s4.load(reviewed=False) if i.id == "ref-first-third"]
    base: dict[str, Any] = {
        "intent": "read",
        "clarification": None,
        "need": "n",
        "question": "q",
        "paper_ids": [],
        "listed": [3, 1],
        "count": None,
        "constraints": [],
        "priorities": [],
        "titles": [],
        "prefer_recent": False,
        "published_after": None,
        "published_before": None,
    }
    good = s4.score(ResearchRequest(**base), item)
    assert good["clarify_correct"] == good["intent_correct"] == good["fields_correct"] == 1.0
    dated = s4.score(ResearchRequest(**{**base, "published_after": "2024-01-01"}), item)
    assert dated["field_published_after"] == dated["fields_correct"] == 0.0
    assert dated["field_listed"] == 1.0
    asked = s4.score(ResearchRequest(**{**base, "clarification": "Which?"}), item)
    assert asked["clarify_correct"] == 0.0 and asked["clarified"] == 1.0


def test_s4_data_is_english_fully_labelled_and_refers_back_only_to_shown_papers() -> None:
    items = s4.load(reviewed=False)  # also checks that every proceeding item labels every field
    assert len({i.id for i in items}) == len(items)
    for item in items:
        assert not any(CJK.search(m.text) for m in item.messages), item.id
        if item.expected.get("listed"):
            assert max(item.expected["listed"]) <= len(item.shown)


def test_s4_grades_quotes_by_the_words_they_cover() -> None:
    labelled = ["API cost", "waiting time"]
    assert s4._covers(["I care about both API cost and waiting time"], labelled)
    assert s4._covers(["api  cost", "waiting time"], labelled)
    assert not s4._covers(["API cost"], labelled), "a missing phrase fails"
    assert not s4._covers(["API cost", "waiting time", "open source"], labelled)
    assert s4._covers([], []) and not s4._covers(["no fine-tuning"], [])


def test_s6_checks_each_expectation_of_a_turn() -> None:
    from langchain_core.messages import AIMessage

    from ara.graph.state import Constraint, PaperCard
    from ara.memory.store import Fact

    request = ResearchRequest(
        **{
            "intent": "read",
            "clarification": None,
            "need": "n",
            "question": "q",
            "paper_ids": [],
            "listed": [2],
            "count": None,
            "constraints": [Constraint(quote="I never fine-tune models", meaning="m")],
            "priorities": [],
            "titles": [],
            "prefer_recent": False,
            "published_after": None,
            "published_before": None,
        }
    )
    shown = [
        PaperCard(arxiv_id=i, version=1, title="t", abstract="", published="2024-01-01")
        for i in ("2401.00001", "2401.00002")
    ]
    reply = AIMessage("We have not discussed this. Would you like me to search arXiv?")
    state = {
        "request": request,
        "selected": ["arxiv:2401.00002v1"],
        "answer": None,
        "messages": [reply],
    }
    assert s6.check({"cited": ["2401.00002"]}, state, [], shown) == ["cited=['2401.00002']"]
    listed = {**state, "papers": shown}
    assert (
        s6.check({"listed": ["2401.00001", "2401.00002"], "read_at_most": 1}, listed, [], shown)
        == []
    )
    assert s6.check({"read_at_most": 1}, {**state, "selected": []}, [], shown) == ["read_at_most=1"]
    profile = [Fact(key="k", quote="I never fine-tune models", statement="s")]
    passing = {
        "intent": "read",
        "constraint": "never fine-tune",
        "profile_has": "never fine-tune",
        "selected_positions": [2],
        "selected": ["2401.00002"],
        "answered": False,
        "reply_has": "search arXiv",
    }
    assert s6.check(passing, state, profile, shown) == []
    failing = {"intent": "library", "no_constraint": "fine-tune", "profile_lacks": "never"}
    assert len(s6.check(failing, state, profile, shown)) == 3
    assert [s.id for s in s6.load()] == [
        "s6-carry",
        "s6-library",
        "s6-update",
        "s6-forget",
        "s6-refer",
        "s6-episode",
        "s6-mismatch",
        "s6-premise",
        "s6-conflict",
        "s6-library-many",
        "s6-compare-ids",
        "s6-library-fallback",
        "s6-no-ask",
    ]
