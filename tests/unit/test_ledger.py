import asyncio
from decimal import Decimal

import pytest
from psycopg.rows import DictRow

from ara.db.pool import Pool
from ara.llm.ledger import BudgetExceeded, Ledger, Scope
from ara.llm.pricing import Usage

TURN = Scope(turn_id="turn-1")
USAGE = Usage(
    input_tokens=1_000,
    cached_tokens=600,
    cache_write_tokens=0,
    output_tokens=200,
    reasoning_tokens=50,
)


def make_ledger(pool: Pool, *, total: str = "10", turn: str = "0.05") -> Ledger:
    return Ledger(pool, global_cap_usd=Decimal(total), turn_cap_usd=Decimal(turn))


async def reserve(ledger: Ledger, usd: str, scope: Scope = TURN) -> int:
    return await ledger.reserve(
        stage="verify", model="gpt-6-luna", prompt_version="verify@0", scope=scope, usd=Decimal(usd)
    )


async def rows(pool: Pool) -> list[DictRow]:
    async with pool.connection() as conn:
        return await (await conn.execute("SELECT * FROM llm_calls ORDER BY id")).fetchall()


async def test_a_reservation_counts_until_the_call_is_settled(pool: Pool) -> None:
    ledger = make_ledger(pool)
    call = await reserve(ledger, "0.004")
    [reserved] = await rows(pool)
    assert (reserved["status"], reserved["charge_usd"]) == ("reserved", Decimal("0.004"))

    await ledger.settle(call, USAGE, Decimal("0.0001"), latency_ms=900, response_id="resp_1")
    [settled] = await rows(pool)
    assert (settled["status"], settled["charge_usd"]) == ("settled", Decimal("0.0001"))
    assert (settled["input_tokens"], settled["cached_tokens"], settled["output_tokens"]) == (
        1_000,
        600,
        200,
    )
    assert (settled["latency_ms"], settled["response_id"]) == (900, "resp_1")
    assert settled["settled_at"] is not None


async def test_a_rejected_call_is_released_but_an_unknown_outcome_stays_charged(
    pool: Pool,
) -> None:
    ledger = make_ledger(pool)
    rejected, unknown = await reserve(ledger, "0.004"), await reserve(ledger, "0.004")
    await ledger.fail(rejected, "BadRequestError(400)", released=True)
    await ledger.fail(unknown, "APITimeoutError()", released=False)
    released, still_open = await rows(pool)
    assert (released["status"], released["charge_usd"]) == ("released", Decimal("0"))
    assert released["settled_at"] is not None
    assert (still_open["status"], still_open["charge_usd"]) == ("reserved", Decimal("0.004"))
    assert (still_open["settled_at"], still_open["error"]) == (None, "APITimeoutError()")


async def test_turn_cap_applies_per_turn(pool: Pool) -> None:
    ledger = make_ledger(pool, turn="0.05")
    await reserve(ledger, "0.03")
    with pytest.raises(BudgetExceeded, match="turn cap"):
        await reserve(ledger, "0.03")
    await reserve(ledger, "0.03", Scope(turn_id="turn-2"))


async def test_global_and_run_caps(pool: Pool) -> None:
    ledger = make_ledger(pool, total="0.01")
    with pytest.raises(BudgetExceeded, match="total cap"):
        await reserve(ledger, "0.02", Scope())

    run = Scope(run_id="run-1", run_cap_usd=Decimal("0.005"))
    await reserve(ledger, "0.004", run)
    with pytest.raises(BudgetExceeded, match="run cap"):
        await reserve(ledger, "0.004", run)


async def test_concurrent_reservations_never_overshoot_a_cap(pool: Pool) -> None:
    ledger = make_ledger(pool, turn="0.05")
    results = await asyncio.gather(
        *(reserve(ledger, "0.01") for _ in range(8)), return_exceptions=True
    )
    assert sum(isinstance(result, int) for result in results) == 5
    assert sum(isinstance(result, BudgetExceeded) for result in results) == 3
