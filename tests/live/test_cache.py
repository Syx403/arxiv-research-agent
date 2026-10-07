"""M0 live check (DESIGN §14): the repeated call to each provider reads its prefix from the cache.

Four billable requests, approved by Ewan on 2026-10-07 and capped at US$0.005 by the ledger:
a Luna prewarm and the call it warms, then a DeepSeek turn and an appended follow-up turn.
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel

from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.llm.prompt import Block, Instructions, Prompt
from ara.llm.stages import FANOUT, Stage
from ara.settings import ROOT

pytestmark = pytest.mark.live

RUN_ID = f"m0-live-{datetime.now(UTC):%Y%m%dT%H%M%S}"
SCOPE = Scope(run_id=RUN_ID, run_cap_usd=Decimal("0.005"))
INSTRUCTIONS = Instructions.load("smoke", Path(__file__).parent)
# DESIGN sections 5-6 as the shared document: real text, above the 1,024-token cache minimum.
DESIGN = (ROOT / "docs/v2/DESIGN.md").read_text(encoding="utf-8")
DOCUMENT = Block("user", DESIGN[DESIGN.index("## 5. RAG design") : DESIGN.index("## 7. Memory")])
LUNA = Stage("smoke_luna", "gpt-6-luna", "low", 1_000, FANOUT)  # breakpoints as in production
FLASH = Stage("smoke_flash", "deepseek-flash", "low", 1_000)


class Answer(BaseModel):
    answer: str


async def test_luna_reads_the_prewarmed_prefix(gateway: Gateway) -> None:
    question = Block("user", "Which rank fusion method does hybrid search use, with which k?")
    prompt = Prompt(INSTRUCTIONS, shared=(DOCUMENT,), item=(question,))

    await gateway.prewarm(LUNA, prompt, Answer, scope=SCOPE)
    answer = await gateway.structured(LUNA, prompt, Answer, scope=SCOPE)

    warm, call = [row for row in await gateway.ledger.calls(RUN_ID) if row["stage"] == LUNA.name]
    assert warm["cache_write_tokens"] >= 1_024
    assert call["cached_tokens"] >= 1_024
    assert "60" in answer.answer


async def test_deepseek_reads_the_earlier_turn_from_its_cache(gateway: Gateway) -> None:
    first = Prompt(
        INSTRUCTIONS,
        shared=(DOCUMENT,),
        item=(Block("user", "Which model writes the answers, and which one verifies them?"),),
    )
    reply = await gateway.text(FLASH, first, scope=SCOPE)
    await asyncio.sleep(8)  # DeepSeek persists its cache units within seconds

    follow_up = Prompt(
        INSTRUCTIONS,
        shared=(DOCUMENT, *first.item, Block("assistant", reply)),
        item=(Block("user", "And which model judges paper relevance offline?"),),
    )
    await gateway.text(FLASH, follow_up, scope=SCOPE)

    _, turn_2 = [row for row in await gateway.ledger.calls(RUN_ID) if row["stage"] == FLASH.name]
    assert turn_2["cached_tokens"] >= 1_024
    assert turn_2["cached_tokens"] <= turn_2["input_tokens"]
