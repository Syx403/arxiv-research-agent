"""Fault hooks (DESIGN §12, M5): `ARA_FAULTS="point=kind[xN],..."` makes a call fail the way a real
dependency fails, so failure paths run against real code without mocks. A point is "arxiv" (every
arXiv request), "fetch" (a paper's full text) or a gateway stage ("synthesize", "verify", "embed",
...). Kinds: an HTTP status (429, 503, ...), timeout, refused (HTTP 400), empty (a model answer
with no text). With "xN"
only the first N calls at that point fail, which is how a retry that recovers is exercised."""

import os
from collections import Counter

import httpx

_tripped: Counter[str] = Counter()


def trip(point: str) -> None:
    """Raise the configured fault for `point`, if any (read at every call, so tests and the S7
    runner can set it per turn)."""
    spec = _faults().get(point)
    if spec is None:
        return
    kind, times = spec
    if times is not None and _tripped[point] >= times:
        return
    _tripped[point] += 1
    raise _error(point, kind)


def reset() -> None:
    _tripped.clear()


def _faults() -> dict[str, tuple[str, int | None]]:
    found: dict[str, tuple[str, int | None]] = {}
    for item in filter(None, (s.strip() for s in os.environ.get("ARA_FAULTS", "").split(","))):
        point, _, kind = item.partition("=")
        kind, _, times = kind.partition("x")
        found[point.strip()] = (kind.strip(), int(times) if times else None)
    return found


def _error(point: str, kind: str) -> Exception:
    request = httpx.Request("GET", f"https://fault.invalid/{point}")
    if kind.isdigit() or kind == "refused":
        status = 400 if kind == "refused" else int(kind)
        response = httpx.Response(status, request=request)
        return httpx.HTTPStatusError(
            f"injected {kind} at {point}", request=request, response=response
        )
    if kind == "timeout":
        return httpx.ReadTimeout(f"injected timeout at {point}", request=request)
    if kind == "empty":
        from ara.llm.gateway import InvalidOutput  # the gateway imports this module

        return InvalidOutput(f"{point}: empty answer (injected)")
    raise ValueError(f"unknown fault kind {kind!r} at {point}")
