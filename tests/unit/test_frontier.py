from __future__ import annotations

from src.core.types import FrontierItem
from src.retrieval.multi_hop.frontier import Frontier


def test_frontier_pops_highest_score_first() -> None:
    frontier = Frontier()
    frontier.add(FrontierItem(paper_id="arxiv:low", score=1.0, depth=1))
    frontier.add(FrontierItem(paper_id="arxiv:high", score=9.0, depth=1))
    frontier.add(FrontierItem(paper_id="arxiv:mid", score=5.0, depth=1))

    assert frontier.pop().paper_id == "arxiv:high"
    assert frontier.pop().paper_id == "arxiv:mid"
    assert frontier.pop().paper_id == "arxiv:low"


def test_frontier_tie_breaks_by_insertion_order() -> None:
    frontier = Frontier()
    frontier.add(FrontierItem(paper_id="arxiv:first", score=5.0, depth=1))
    frontier.add(FrontierItem(paper_id="arxiv:second", score=5.0, depth=1))

    assert frontier.pop().paper_id == "arxiv:first"
    assert frontier.pop().paper_id == "arxiv:second"


def test_frontier_dedups_against_visited_and_enqueued() -> None:
    frontier = Frontier()
    frontier.mark_visited("arxiv:visited")

    assert frontier.add(FrontierItem(paper_id="arxiv:visited", score=10.0, depth=1)) is False
    assert frontier.add(FrontierItem(paper_id="arxiv:new", score=1.0, depth=1)) is True
    assert frontier.add(FrontierItem(paper_id="arxiv:new", score=10.0, depth=1)) is False
    assert len(frontier) == 1


def test_frontier_pop_empty_returns_none() -> None:
    assert Frontier().pop() is None
