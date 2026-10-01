"""Immutable controls inherited by one run's async child tasks."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from time import perf_counter

from pydantic import BaseModel, ConfigDict
from src.core.answer_contract import RequestedField


class RunContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str = ""
    turn_id: str = ""
    as_of: date | None = None
    model_result_cache: bool = True
    structured_context: bool = True
    include_typed_observations: bool = False
    response_fields: tuple[RequestedField, ...] = ()
    deadline_monotonic: float | None = None


_current: ContextVar[RunContext] = ContextVar("ara_run_context", default=RunContext())


def current_run() -> RunContext:
    return _current.get()


def today() -> date:
    return current_run().as_of or datetime.now(UTC).date()


def remaining_turn_s() -> float | None:
    deadline = current_run().deadline_monotonic
    return max(0.0, deadline - perf_counter()) if deadline is not None else None


@contextmanager
def use_run_context(context: RunContext):
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)
