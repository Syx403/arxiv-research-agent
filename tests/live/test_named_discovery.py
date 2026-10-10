"""D41/D42 live check: a search for papers about a named model, through the top-level graph as the
simulated user "live-check" (see test_open_questions.py). Approved by Ewan on 2026-10-11: one
turn, about 10 requests, estimated US$0.006, capped at US$0.02.
- "找kimi k3相关论文": understand names "Kimi K3"; prerank screens the papers that name it; the
  calibrated screen grades by object of study; the K3 report leads the list; reasons in Chinese.
Run with `uv run pytest -m live tests/live/test_named_discovery.py -s`.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from ara.graph import app
from tests.live import test_open_questions as base
from tests.live.test_open_questions import CJK, K3, ctx, show

pytestmark = pytest.mark.live
__all__ = ["ctx"]  # the fixture, reused

base.RUN_ID = f"d42-live-{datetime.now(UTC):%Y%m%dT%H%M%S}"
base.RUN_CAP = Decimal("0.02")


async def test_a_search_for_a_named_model_lists_its_report_first(ctx: Any) -> None:
    graph, turn = ctx
    found = await app.send(graph, str(uuid4()), turn(), message="找kimi k3相关论文")
    show(found)
    request, papers = found.state["request"], found.state["papers"]
    print("names:", request.names, "| listed:", [(p.arxiv_id, p.relevance) for p in papers])
    assert [n.casefold() for n in request.names] == ["kimi k3"]
    assert papers and papers[0].arxiv_id == K3
    assert all(CJK.search(p.reason) for p in papers)
