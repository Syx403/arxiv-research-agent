"""Per-run acquisition injection for reproducible material and outage checks."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Acquisition:
    client: Any
    ingest: Callable


_current: ContextVar[Acquisition | None] = ContextVar("ara_acquisition", default=None)


def acquisition_client(default):
    adapter = _current.get()
    return adapter.client if adapter else default


def ingestion(default):
    adapter = _current.get()
    return adapter.ingest if adapter else default


@contextmanager
def use_acquisition(adapter: Acquisition | None):
    token = _current.set(adapter)
    try:
        yield
    finally:
        _current.reset(token)
