"""S2 reading QA (DESIGN §11.2, D19): the product's read → answer on one QASPER paper per question.
Items: the 30 S1 questions plus 5 questions every annotator found unanswerable, drawn with a fixed
seed from other validation papers (the S1 papers have none). Graded by code: QASPER token F1 on the
direct answer, abstention, citation precision against gold evidence, and how many lines survived
verification. Free-form answers get a judged score from E1 on (§11.4)."""

import json
import random
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ara.graph.state import Answer
from ara.rag.sources import ParsedPaper, qasper_paper
from evals import qasper
from evals.graders import answers
from evals.qasper import Gold
from evals.suites import s1

MANIFEST = Path(__file__).parents[1] / "datasets" / "s2_qasper.json"
SEED, UNANSWERABLE_ITEMS, DEV_UNANSWERABLE = 20261008, 5, 2
ARMS = ("product",)
# Baselines (closed-book, whole paper in context, naive RAG) and their pairs come with E1.
PAIRS: tuple[tuple[str, str], ...] = ()
UNIT_COST_USD = 0.008  # estimate per item: select, synthesize, verify (DESIGN §11.2), re-measured


@dataclass(frozen=True)
class Item:
    id: str
    paper: str  # arXiv id; the reference is "qasper:<id>"
    split: str
    question: str
    golds: tuple[Gold, ...]


def build_manifest() -> dict[str, Any]:
    s1_manifest = json.loads(s1.MANIFEST.read_text())
    data = qasper.records()
    taken = {e["paper"] for e in s1_manifest["items"]}
    pool = sorted(
        (pid, qid) for pid, r in data.items() if pid not in taken for qid in qasper.unanswerable(r)
    )
    papers = sorted({pid for pid, _ in pool})
    rng = random.Random(SEED)
    chosen = [
        rng.choice([q for p, q in pool if p == pid])
        for pid in rng.sample(papers, UNANSWERABLE_ITEMS)
    ]
    paper_of = dict((q, p) for p, q in pool)
    extra = [
        {"id": qid, "paper": paper_of[qid], "split": "dev" if n < DEV_UNANSWERABLE else "test"}
        for n, qid in enumerate(chosen)
    ]
    return {
        "suite": "s2",
        "dataset": s1_manifest["dataset"],
        "seed": SEED,
        "items": s1_manifest["items"] + extra,
    }


def manifest_hash() -> str:
    return sha256(MANIFEST.read_bytes()).hexdigest()[:12]


def splits() -> dict[str, str]:
    return {e["id"]: e["split"] for e in json.loads(MANIFEST.read_text())["items"]}


def load(limit: int | None = None) -> tuple[list[Item], dict[str, ParsedPaper]]:
    """The first `limit` items in manifest order (all if None) and their parsed papers."""
    manifest = json.loads(MANIFEST.read_text())
    if qasper.download() != manifest["dataset"]["sha256"]:
        raise ValueError(f"{qasper.PATH} differs from the file the S2 manifest was drawn from")
    data = qasper.records()
    entries = manifest["items"][:limit]
    items = []
    for e in entries:
        record = data[e["paper"]]
        qas = record["qas"]
        question = qas["question"][qas["question_id"].index(e["id"])]
        items.append(
            Item(e["id"], e["paper"], e["split"], question, tuple(qasper.golds(record, e["id"])))
        )
    papers = {f"qasper:{e['paper']}": qasper_paper(data[e["paper"]]) for e in entries}
    return items, papers


def output(answer: Answer) -> dict[str, Any]:
    return {
        "short": answer.short,
        "abstained": answer.abstained,
        "lines": [c.text for c in answer.sentences],
        "cited_paragraphs": [e.paragraph for e in answer.evidence],
        "delivered": len(answer.sentences) + int(not answer.abstained),
        "dropped": len(answer.dropped),
    }


def score(result: dict[str, Any], item: Item) -> dict[str, float]:
    metrics = answers.score(
        result["short"], result["abstained"], result["cited_paragraphs"], item.golds
    )
    lines = result["delivered"] + result["dropped"]
    if lines:
        metrics["verified_share"] = result["delivered"] / lines
    return metrics
