"""Budget ledger (DESIGN §6.5): reserve an upper bound before sending, settle with the reported
usage afterwards. A request whose outcome is unknown keeps its reservation."""

from dataclasses import asdict, dataclass
from decimal import Decimal

from psycopg.rows import DictRow

from ara.db.pool import Pool, fetch_one
from ara.llm.pricing import Usage

LOCK_ID = 7_042_002  # pg_advisory_xact_lock key: reservations are checked one at a time


class BudgetExceeded(Exception):
    """A reservation would exceed a cap; the request was not sent."""


@dataclass(frozen=True)
class Scope:
    """The evaluation run and the conversation turn a call belongs to; a run may set its own cap."""

    run_id: str | None = None
    turn_id: str | None = None
    run_cap_usd: Decimal | None = None


SPENT = """
SELECT coalesce(sum(charge_usd), 0) AS total,
       coalesce(sum(charge_usd) FILTER (WHERE run_id = %(run_id)s), 0) AS run,
       coalesce(sum(charge_usd) FILTER (WHERE turn_id = %(turn_id)s), 0) AS turn
FROM llm_calls
"""
RESERVE = """
INSERT INTO llm_calls (stage, model, prompt_version, run_id, turn_id, reserved_usd)
VALUES (%(stage)s, %(model)s, %(prompt_version)s, %(run_id)s, %(turn_id)s, %(usd)s)
RETURNING id
"""
SETTLE = """
UPDATE llm_calls
SET status = 'settled', settled_at = now(), cost_usd = %(cost)s,
    input_tokens = %(input_tokens)s, cached_tokens = %(cached_tokens)s,
    cache_write_tokens = %(cache_write_tokens)s, output_tokens = %(output_tokens)s,
    reasoning_tokens = %(reasoning_tokens)s, latency_ms = %(latency_ms)s,
    response_id = %(response_id)s
WHERE id = %(id)s
"""
# Only a released call is settled (at $0); an unknown outcome stays an open reservation.
FAIL = """
UPDATE llm_calls
SET error = %(error)s,
    status = CASE WHEN %(released)s THEN 'released' ELSE status END,
    settled_at = CASE WHEN %(released)s THEN now() ELSE settled_at END
WHERE id = %(id)s
"""


class Ledger:
    def __init__(self, pool: Pool, *, global_cap_usd: Decimal, turn_cap_usd: Decimal) -> None:
        self.pool = pool
        self.global_cap_usd = global_cap_usd
        self.turn_cap_usd = turn_cap_usd

    async def reserve(
        self, *, stage: str, model: str, prompt_version: str, scope: Scope, usd: Decimal
    ) -> int:
        """Reserve `usd` for one request and return its call id, or raise BudgetExceeded.
        The transaction-scoped advisory lock serialises reservations across processes."""
        params = {
            "stage": stage,
            "model": model,
            "prompt_version": prompt_version,
            "run_id": scope.run_id,
            "turn_id": scope.turn_id,
            "usd": usd,
        }
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(%(lock)s)", {"lock": LOCK_ID})
            spent = await fetch_one(conn, SPENT, params)
            for name, cap in self._caps(scope):
                if spent[name] + usd > cap:
                    raise BudgetExceeded(
                        f"{name} cap ${cap}: ${spent[name]:.6f} spent or reserved,"
                        f" this call needs up to ${usd:.6f}"
                    )
            return int((await fetch_one(conn, RESERVE, params))["id"])

    def _caps(self, scope: Scope) -> list[tuple[str, Decimal]]:
        caps = [("total", self.global_cap_usd)]
        if scope.turn_id is not None:
            caps.append(("turn", self.turn_cap_usd))
        if scope.run_cap_usd is not None:
            caps.append(("run", scope.run_cap_usd))
        return caps

    async def settle(
        self, call_id: int, usage: Usage, cost: Decimal, *, latency_ms: int, response_id: str
    ) -> None:
        params = {
            **asdict(usage),
            "cost": cost,
            "latency_ms": latency_ms,
            "response_id": response_id,
            "id": call_id,
        }
        async with self.pool.connection() as conn:
            await conn.execute(SETTLE, params)

    async def fail(self, call_id: int, error: str, *, released: bool) -> None:
        """Record a failed request. `released` means the provider rejected it, so nothing was
        billed; otherwise the outcome is unknown and the row stays an open, charged reservation."""
        async with self.pool.connection() as conn:
            await conn.execute(FAIL, {"error": error, "released": released, "id": call_id})

    async def calls(self, run_id: str) -> list[DictRow]:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM llm_calls WHERE run_id = %(run_id)s ORDER BY id", {"run_id": run_id}
            )
            return await cursor.fetchall()
