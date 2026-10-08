"""S4 understand / clarify (DESIGN §11.2, D22): the product's understand call, with its code checks,
on 50 messages: v1's 18 distinct questions and 32 drafted edge cases, each with the conversation
and the papers shown before it. Labels were proposed by Claude and are reviewed by Ewan before a
round runs. Graded by code: clarify or not, intent, and the fields an item specifies."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from ara.graph import app
from ara.graph.state import PaperCard, ResearchRequest
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.llm.stages import STAGES

DATA = Path(__file__).parents[1] / "datasets" / "s4_understand.json"
ARMS = ("product",)
PAIRS: tuple[tuple[str, str], ...] = ()
LABELS = ("clarify_expected",)  # metrics that record the item's label, not an arm's output
UNIT_COST_USD = 0.0005  # one Luna low call (the live turns measured $0.0001-0.0002)
FIELDS = ("paper_ids", "listed", "count", "published_after", "published_before")


@dataclass(frozen=True)
class Item:
    id: str
    split: str
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
    return [
        Item(
            i["id"],
            i["split"],
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
    prompt = app.understand_prompt(item.messages)
    raw = await gateway.structured(STAGES["understand"], prompt, ResearchRequest, scope=scope)
    return app.checked(raw, item.messages, item.shown)


def score(request: ResearchRequest, item: Item) -> dict[str, float]:
    """`clarified` and `clarify_expected` give false- and missed-clarify rates in the report;
    intent and fields are graded only on items that should proceed."""
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
    fields = [f for f in FIELDS if f in expected]
    if "constraints" in expected:
        fields.append("constraints")
    if fields:
        metrics["fields_correct"] = float(all(_matches(request, item, f) for f in fields))
    return metrics


def _matches(request: ResearchRequest, item: Item, field: str) -> bool:
    expected = item.expected[field]
    if field == "constraints":
        return bool(len(request.constraints) == expected)
    value = getattr(request, field)
    if field in ("paper_ids", "listed"):
        return sorted(value) == sorted(expected)
    return bool(value == expected)
