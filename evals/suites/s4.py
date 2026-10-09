"""S4 understand / clarify (DESIGN §11.2, D22, D24): the product's understand call, with its code
checks, on English messages, each with the conversation, the papers shown before it and the date
it is asked on. Items come from v1's questions (translated) and drafted cases; labels were proposed
by Claude and are reviewed by Ewan before a round runs. Graded by code: clarify or not, intent, and
every field of the request, so a value the model adds where none is expected counts as an error."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from ara.graph import app
from ara.graph.state import PaperCard, ResearchRequest, plain
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.llm.stages import STAGES

DATA = Path(__file__).parents[1] / "datasets" / "s4_understand.json"
ARMS = ("product",)
PAIRS: tuple[tuple[str, str], ...] = ()
LABELS = ("clarify_expected",)  # metrics that record the item's label, not an arm's output
UNIT_COST_USD = 0.0005  # one Luna low call (the live turns measured $0.0001-0.0002)


@dataclass(frozen=True)
class Item:
    id: str
    split: str
    asked_at: date
    messages: list[AnyMessage]
    shown: list[PaperCard]
    expected: dict[str, Any]


def _items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = json.loads(DATA.read_text())["items"]
    return items


def data_hash() -> str:
    return sha256(DATA.read_bytes()).hexdigest()[:12]


def splits() -> dict[str, str]:
    return {i["id"]: i["split"] for i in _items()}


def load(limit: int | None = None, *, reviewed: bool = True) -> list[Item]:
    raw = _items()
    if reviewed and not all(i["reviewed"] for i in raw):
        raise ValueError(f"{DATA} has labels Ewan has not reviewed")
    for i in raw:
        labelled = set(i["expected"]) - {"clarify", "intent"}
        if not i["expected"]["clarify"] and labelled != set(GRADERS):
            raise ValueError(f"{i['id']}: expected labels {sorted(labelled)}, not every field")
    return [
        Item(
            i["id"],
            i["split"],
            date.fromisoformat(i["asked_at"]),
            [*(_message(m) for m in i["context"]), HumanMessage(i["message"])],
            [PaperCard(abstract="", **p) for p in i["shown"]],
            i["expected"],
        )
        for i in raw[:limit]
    ]


def _message(m: dict[str, str]) -> AnyMessage:
    return HumanMessage(m["text"]) if m["role"] == "user" else AIMessage(m["text"])


async def understand(gateway: Gateway, item: Item, scope: Scope) -> ResearchRequest:
    """Exactly what the graph's understand node does: one Luna call, then the code checks."""
    prompt = app.understand_prompt(item.messages, item.shown, item.asked_at)
    raw = await gateway.structured(STAGES["understand"], prompt, ResearchRequest, scope=scope)
    return app.checked(raw, item.messages, item.shown)


def score(request: ResearchRequest, item: Item) -> dict[str, float]:
    """`clarified` and `clarify_expected` give false- and missed-clarify rates in the report;
    intent and each field are graded only on items that should proceed (`field_*` per field,
    `fields_correct` when all of them are right)."""
    expected = item.expected
    clarified = request.clarification is not None
    metrics = {
        "clarify_expected": float(expected["clarify"]),
        "clarified": float(clarified),
        "clarify_correct": float(clarified == expected["clarify"]),
    }
    if expected["clarify"]:
        return metrics
    metrics["intent_correct"] = float(request.intent == expected["intent"])
    fields = {name: grade(request, expected[name]) for name, grade in GRADERS.items()}
    metrics |= {f"field_{name}": float(ok) for name, ok in fields.items()}
    metrics["fields_correct"] = float(all(fields.values()))
    return metrics


def _covers(found: list[str], expected: list[str]) -> bool:
    """The same user words, however the model splits them: every labelled phrase sits inside a
    found quote (or contains it), and every found quote overlaps a labelled phrase, so a missing
    phrase or an invented quote fails (whitespace and case folded)."""
    found, expected = [plain(q) for q in found], [plain(e) for e in expected]

    def near(a: str, b: str) -> bool:
        return a in b or b in a

    return all(any(near(e, f) for f in found) for e in expected) and all(
        any(near(f, e) for e in expected) for f in found
    )


GRADERS: dict[str, Callable[[ResearchRequest, Any], bool]] = {
    "paper_ids": lambda r, e: sorted(r.paper_ids) == sorted(e),
    "listed": lambda r, e: sorted(r.listed) == sorted(e),
    "titles": lambda r, e: _covers(r.titles, e),
    "count": lambda r, e: r.count == e,
    "constraints": lambda r, e: _covers([c.quote for c in r.constraints], e),
    "priorities": lambda r, e: _covers([p.quote for p in r.priorities], e),
    "published_after": lambda r, e: r.published_after == e,
    "published_before": lambda r, e: r.published_before == e,
    "prefer_recent": lambda r, e: r.prefer_recent == e,
    # what a library question refers back to: absent when none is expected, else it covers the
    # labelled key phrase (D33)
    "history": lambda r, e: r.history is None if e is None else plain(e) in plain(r.history or ""),
}
