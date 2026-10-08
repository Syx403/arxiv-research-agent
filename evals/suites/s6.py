"""S6 multi-turn + memory (DESIGN §11.2, D27): scripted scenarios run through the conversation graph
with the Postgres checkpointer and Store; each scenario has its own user, each session its own
thread. Every turn's expectations are checked by code; a scenario passes when all of them hold."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ara.graph.state import PaperCard, plain
from ara.memory.store import Fact

DATA = Path(__file__).parents[1] / "datasets" / "s6_memory.json"
ARMS = ("product",)
PAIRS: tuple[tuple[str, str], ...] = ()
LABELS: tuple[str, ...] = ()
UNIT_COST_USD = 0.008  # per scenario: up to two reads and a discovery, plus Luna memory calls


@dataclass(frozen=True)
class Scenario:
    id: str
    split: str
    sessions: list[list[dict[str, Any]]]


def load(limit: int | None = None) -> list[Scenario]:
    raw = json.loads(DATA.read_text())["scenarios"]
    return [Scenario(s["id"], s["split"], s["sessions"]) for s in raw[:limit]]


def data_hash() -> str:
    return sha256(DATA.read_bytes()).hexdigest()[:12]


def splits() -> dict[str, str]:
    return {s.id: s.split for s in load()}


def check(
    expect: dict[str, Any], state: dict[str, Any], profile: list[Fact], shown: list[PaperCard]
) -> list[str]:
    """The expectations a turn failed (none if it passed); `shown`: the list before the turn."""
    request, reply = state["request"], state["messages"][-1].text
    quotes = [plain(c.quote) for c in request.constraints]
    facts = [plain(f.quote) for f in profile]
    read = [r.removeprefix("arxiv:").split("v")[0] for r in state["selected"]]
    delivered = state["answer"]
    answered = delivered is not None and not delivered.abstained
    tests: dict[str, Callable[[Any], bool]] = {
        "intent": lambda v: request.intent == v,
        "constraint": lambda v: any(plain(v) in q for q in quotes),
        "no_constraint": lambda v: not any(plain(v) in q for q in quotes),
        "profile_has": lambda v: any(plain(v) in f for f in facts),
        "profile_lacks": lambda v: not any(plain(v) in f for f in facts),
        "selected": lambda v: set(v) <= set(read),
        "selected_positions": lambda v: read == [shown[n - 1].arxiv_id for n in v],
        "answered": lambda v: answered == v,
        "reply_has": lambda v: v in reply,
    }
    return [f"{name}={value!r}" for name, value in expect.items() if not tests[name](value)]
