"""A conservative request-admission budget shared by retries and async tasks.

Reservations are persisted BEFORE sending a request. Unknown outcomes retain
their full reservation; provider-reported usage releases unused funds. This is
a local cost estimate, not a substitute for the provider's billing statement.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import fcntl
from pathlib import Path
from threading import Lock
from uuid import uuid4
import asyncio
from time import monotonic

from src.core.trace import record_event
from src.core.run_context import current_run
from src.llm.stages import current_stage

FLASH_PRICE_DATE = "2026-09-14"
_current: ContextVar["RequestBudget | None"] = ContextVar("ara_request_budget", default=None)
_turn: ContextVar["TurnBudget | None"] = ContextVar("ara_turn_budget", default=None)


@dataclass
class TurnBudget:
    identifier: str
    limit_usd: float


@contextmanager
def use_turn_budget(limit_usd: float, *, identifier: str | None = None):
    turn = TurnBudget(identifier or str(uuid4()), limit_usd)
    token = _turn.set(turn)
    try:
        yield turn
    finally:
        _turn.reset(token)


def set_turn_budget(limit_usd: float):
    turn = _turn.get()
    if turn:
        turn.limit_usd = limit_usd


class BudgetExceeded(RuntimeError):
    pass


class BudgetBusy(BudgetExceeded):
    """Outstanding reservations may settle low enough to admit this request."""


class BudgetWaitTimeout(BudgetExceeded):
    pass


def budget_stop_reason(exc):
    return "budget_wait_timeout" if isinstance(exc, BudgetWaitTimeout) else "budget_exhausted"


def flash_cost(raw_usage: dict, *, peak: bool = True) -> float:
    prompt = max(int(raw_usage.get("prompt_tokens") or 0), 0)
    hits = min(prompt, max(int(raw_usage.get("prompt_cache_hit_tokens") or 0), 0))
    completion = max(int(raw_usage.get("completion_tokens") or 0), 0)
    return ((prompt - hits) * .30 + hits * .006 + completion * 1.20) / 1_000_000 * (1 if peak else .5)


def is_peak(now: datetime) -> bool:
    utc = now.astimezone(UTC)
    return utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)


@dataclass
class Reservation:
    budget: "RequestBudget"
    index: int
    finished: bool = False

    def settle(self, raw_usage: dict) -> None:
        with self.budget.transaction():
            row = self.budget.requests[self.index]
            if row["provider"] == "deepseek_native":
                cost = flash_cost(raw_usage, peak=row["peak"])
            elif row["provider"] == "openai_native":
                cost = int(raw_usage.get("prompt_tokens") or raw_usage.get("total_tokens") or 0) * .02 / 1_000_000
            else:
                cost = 0.0  # Only explicitly configured Cohere Trial is admitted.
            row.update(status="settled", estimated_usd=cost, usage=raw_usage)
            self.finished = True
            self.budget._save()
        record_event("request_settled", provider=row["provider"], model=row["model"],
                     estimated_usd=cost, usage=raw_usage)

    def uncertain(self) -> None:
        if self.finished:
            return
        with self.budget.transaction():
            self.budget.requests[self.index]["status"] = "uncertain"
            self.budget._save()
        self.finished = True


class RequestBudget:
    def __init__(self, limit_usd: float, *, path: Path | None = None,
                 cohere_trial: bool = False, max_requests: int = 160):
        if not 0 < limit_usd <= 100:
            raise ValueError("A positive explicit budget is required.")
        self.limit_usd, self.path, self.cohere_trial = limit_usd, path, cohere_trial
        self.max_requests, self.requests, self.lock = max_requests, [], Lock()
        with self.transaction():
            pass

    def _reload(self):
        if self.path and self.path.exists():
            existing = json.loads(self.path.read_text())
            if existing["limit_usd"] != self.limit_usd or existing["cohere_trial"] != self.cohere_trial:
                raise ValueError("Existing budget settings differ; do not silently increase a run's allowance.")
            self.requests = existing["requests"]

    @contextmanager
    def transaction(self):
        """Serialize read/modify/write across UI, CLI processes and async tasks.

        Reservation indices stay stable because the ledger is append-only.
        No awaits or provider calls are performed while this lock is held.
        """
        with self.lock:
            if self.path is None:
                yield
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_suffix(".lock").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    self._reload()
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def activate(self):
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def reserve(self, provider: str, model: str, payload_bytes: int, output_tokens: int) -> Reservation:
        if provider == "deepseek_native" and model == "deepseek-flash":
            # Serialized UTF-8 bytes plus framing overhead conservatively bound
            # input tokens; all output (including thinking) shares max_tokens.
            amount = ((payload_bytes + 1024) * .30 + output_tokens * 1.20) / 1_000_000
        elif provider == "openai_native" and model == "text-embedding-3-small":
            amount = (payload_bytes + 1024) * .02 / 1_000_000
        elif provider == "cohere_native" and self.cohere_trial:
            amount = 0.0
        else:
            raise BudgetExceeded("No approved price/account policy for this provider and model.")
        with self.transaction():
            committed = sum(r["estimated_usd"] for r in self.requests)
            turn = _turn.get()
            if turn:
                used = sum(r['estimated_usd'] for r in self.requests if r.get('turn_id') == turn.identifier)
                if used + amount > turn.limit_usd:
                    pending = sum(r['estimated_usd'] for r in self.requests
                                  if r.get('turn_id') == turn.identifier and r['status'] == 'reserved')
                    if pending and used - pending + amount <= turn.limit_usd:
                        raise BudgetBusy('Waiting for this turn\'s outstanding reservations.')
                    record_event('budget_stop', scope='turn', committed_usd=used,
                                 next_reservation_usd=amount, limit_usd=turn.limit_usd)
                    raise BudgetExceeded('This turn reached its request-admission cost limit.')
            if committed + amount > self.limit_usd or len(self.requests) >= self.max_requests:
                pending = sum(r['estimated_usd'] for r in self.requests if r['status'] == 'reserved')
                if (len(self.requests) < self.max_requests and pending
                        and committed - pending + amount <= self.limit_usd):
                    raise BudgetBusy('Waiting for outstanding batch reservations.')
                record_event("budget_stop", committed_usd=committed, next_reservation_usd=amount,
                             limit_usd=self.limit_usd)
                raise BudgetExceeded("Request budget exhausted; no further provider request was sent.")
            index = len(self.requests)
            self.requests.append({
                "provider": provider, "model": model, "status": "reserved",
                "reserved_usd": amount, "estimated_usd": amount,
                "peak": is_peak(datetime.now(UTC)), "time": datetime.now(UTC).isoformat(),
                "turn_id": turn.identifier if turn else None,
                "run_id": current_run().run_id, "stage": current_stage(),
            })
            self._save()
        record_event("request_reserved", provider=provider, model=model, reserved_usd=amount)
        return Reservation(self, index)

    def snapshot(self) -> dict:
        with self.transaction():
            return {
                "limit_usd": self.limit_usd, "cohere_trial": self.cohere_trial,
                "price_date": FLASH_PRICE_DATE, "requests": [dict(row) for row in self.requests],
                "estimated_settled_usd": sum(r["estimated_usd"] for r in self.requests if r["status"] == "settled"),
                "uncertain_or_reserved_usd": sum(r["estimated_usd"] for r in self.requests if r["status"] != "settled"),
                "committed_usd": sum(r["estimated_usd"] for r in self.requests),
            }

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "limit_usd": self.limit_usd, "cohere_trial": self.cohere_trial,
                "price_date": FLASH_PRICE_DATE, "requests": self.requests,
            }, indent=2))
            temporary.replace(self.path)


def usage_for_turn(snapshot: dict, turn_id: str) -> dict:
    """Attribute by immutable identity, never by an unsafe global list offset."""
    rows = [r for r in snapshot["requests"] if r.get("turn_id") == turn_id]
    settled = [r for r in rows if r["status"] == "settled"]
    chat = [r for r in settled if r["provider"] == "deepseek_native"]
    prompt = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in chat)
    reported = [r for r in chat if "prompt_cache_hit_tokens" in r.get("usage", {})]
    hits = sum(r["usage"]["prompt_cache_hit_tokens"] for r in reported)
    return {
        "requests": len(rows),
        "estimated_usd": sum(r["estimated_usd"] for r in settled),
        "uncertain_usd": sum(r["estimated_usd"] for r in rows if r["status"] != "settled"),
        "prompt_tokens": prompt,
        "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in chat),
        "reasoning_tokens": sum(r.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0) for r in chat),
        "prompt_cache_hit_tokens": hits,
        "cache_reporting_requests": len(reported), "chat_settled_requests": len(chat),
        "prompt_cache_hit_rate": hits / prompt if prompt and len(reported) == len(chat) else None,
        "by_stage": {stage: {"requests": len(group), "committed_usd": sum(r["estimated_usd"] for r in group)}
                     for stage in sorted({r.get("stage") or "other" for r in rows})
                     if (group := [r for r in rows if (r.get("stage") or "other") == stage])},
    }


def reserve_request(provider: str, model: str, payload: object, output_tokens: int = 0) -> Reservation | None:
    budget = _current.get()
    if budget is None:
        return None
    size = len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
    return budget.reserve(provider, model, size, output_tokens)


async def admit_request(provider, model, payload, output_tokens=0, *, max_wait_s=45):
    """Wait only for temporary occupancy; unknown charges never release budget.

    Polling also observes settlements made by another process sharing this ledger.
    No reservation exists for a waiting request; cancellation cannot charge it.
    The enclosing run deadline can always cancel this bounded wait.
    """
    started, waiting = monotonic(), False
    while True:
        try:
            reservation = reserve_request(provider, model, payload, output_tokens)
            if waiting:
                record_event('budget_wait', status='admitted', wait_duration_s=monotonic() - started)
            return reservation
        except BudgetBusy as exc:
            if not waiting:
                record_event('budget_wait', status='waiting', max_wait_s=max_wait_s)
                waiting = True
            remaining = max_wait_s - (monotonic() - started)
            if remaining <= 0:
                record_event('budget_wait', status='timeout', wait_duration_s=monotonic() - started)
                raise BudgetWaitTimeout('Outstanding request reservations did not settle in time.') from exc
            await asyncio.sleep(min(.2, remaining))
