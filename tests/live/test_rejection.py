"""A request the provider rejects releases its reservation (DESIGN §6.5).

One request, rejected by OpenAI with HTTP 400 before any work is done, so it is not billed
(within the CLAUDE.md rule-1 threshold). Run it on its own:
`uv run pytest -m live tests/live/test_rejection.py`.
"""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import openai
import pytest
from pydantic import BaseModel

from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.llm.prompt import Block, Instructions, Prompt
from ara.llm.stages import Stage

pytestmark = pytest.mark.live

RUN_ID = f"m0-reject-{datetime.now(UTC):%Y%m%dT%H%M%S}"
# The Responses API requires max_output_tokens >= 16, so this stage is always rejected.
TOO_SMALL = Stage("smoke_reject", "gpt-6-luna", "low", max_output_tokens=1)


class Answer(BaseModel):
    answer: str


async def test_a_rejected_request_releases_its_reservation(gateway: Gateway) -> None:
    prompt = Prompt(Instructions.load("smoke", Path(__file__).parent), item=(Block("user", "Hi"),))
    scope = Scope(run_id=RUN_ID, run_cap_usd=Decimal("0.001"))

    with pytest.raises(openai.BadRequestError):
        await gateway.structured(TOO_SMALL, prompt, Answer, scope=scope)

    [row] = await gateway.ledger.calls(RUN_ID)
    assert (row["status"], row["charge_usd"]) == ("released", Decimal("0"))
    assert row["error"].startswith("BadRequestError")
