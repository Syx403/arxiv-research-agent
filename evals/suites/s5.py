"""S5 verifier (DESIGN §11.2, D10, D19): the product's verify prompt checks each reviewed claim
against its one evidence sentence, once with Luna (the product stage) and once with DeepSeek.
Graded by code: "unsupported" is the positive class; the report gives precision, recall and F1
on it, recall per perturbation kind, and the paired accuracy difference between the models."""

import json
from dataclasses import dataclass
from typing import Any

from ara.graph.answer import verify_prompt
from ara.graph.state import Claim, Evidence, Verdict
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.llm.stages import FLASH, STAGES, Stage
from ara.rag.sources import qasper_paper
from evals import qasper
from evals.suites import s1, s5_data

DATA = s5_data.OUTPUT
# The DeepSeek arm gets synthesize's output cap: its thinking counts against it (D19).
ARMS = {"luna": STAGES["verify"], "deepseek": Stage("verify", FLASH, "high", 8_000)}
PAIRS = (("luna", "deepseek"),)
LABELS = ("unsupported",)  # metrics that record the item's label, not an arm's output
# Per claim, for the dry-run estimate: the smoke run (2026-10-08, one claim, off-peak) measured
# $0.00008 (Luna) and $0.00015 (DeepSeek); these allow for longer thinking and DeepSeek peak hours.
UNIT_COST_USD = {"luna": 0.0002, "deepseek": 0.0008}


@dataclass(frozen=True)
class Item:
    id: str  # "<qasper question id>:supported" or ":unsupported"
    split: str
    label: str
    kind: str
    claim: str
    evidence: Evidence


def _claims() -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = json.loads(DATA.read_text())["items"]
    if not all(c["reviewed"] for c in claims):
        raise ValueError(f"{DATA} has claims Ewan has not reviewed")
    return claims


def splits() -> dict[str, str]:
    """A claim takes the split of the S1 question its sentence came from."""
    of_question = {e["id"]: e["split"] for e in json.loads(s1.MANIFEST.read_text())["items"]}
    return {c["id"]: of_question[c["id"].split(":")[0]] for c in _claims()}


def kinds() -> dict[str, str]:
    return {c["id"]: c["kind"] for c in _claims()}


def load(limit: int | None = None) -> list[Item]:
    """The first `limit` claims, taken in supported / unsupported pairs."""
    split_of = splits()
    records = qasper.records()
    items = []
    for c in _claims()[: None if limit is None else 2 * limit]:
        paragraph = qasper_paper(records[c["paper"]]).paragraphs[c["paragraph"]]
        evidence = Evidence(
            id="E1",
            paper_id=f"qasper:{c['paper']}",
            chunk_id=0,
            paragraph=c["paragraph"],
            heading_path=paragraph.heading_path,
            text=c["sentence"],
        )
        items.append(Item(c["id"], split_of[c["id"]], c["label"], c["kind"], c["claim"], evidence))
    return items


async def check(gateway: Gateway, arm: str, item: Item, scope: Scope) -> Verdict:
    claim = Claim(index=1, text=item.claim, citations=["E1"])
    prompt = verify_prompt([item.evidence], claim, question="")  # not a direct answer
    return await gateway.structured(ARMS[arm], prompt, Verdict, scope=scope)


def score(verdict: Verdict, item: Item) -> dict[str, float]:
    flagged, unsupported = not verdict.supported, item.label == "unsupported"
    return {
        "correct": float(flagged == unsupported),
        "flagged": float(flagged),
        "unsupported": float(unsupported),
    }
