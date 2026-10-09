"""S3 discovery (DESIGN §11.2, §11.3, D22): the discover subgraph on PaSa queries, each searched
with its own date cutoff. Dev = 15 AutoScholarQuery queries, held-out = 15 RealScholarQuery
queries, drawn with a fixed seed. The committed manifest holds query ids only; the text stays
local."""

import json
import random
from hashlib import sha256
from pathlib import Path
from typing import Any

from ara.graph.state import MAX_LISTED, ResearchRequest
from evals import pasa
from evals.pasa import Query

MANIFEST = Path(__file__).parents[1] / "datasets" / "s3_pasa.json"
SEED, PER_SPLIT = 20261009, 15
SPLITS = {"dev": "AutoScholarQuery", "test": "RealScholarQuery"}
ARMS = ("product",)
PAIRS: tuple[tuple[str, str], ...] = ()
LABELS: tuple[str, ...] = ()  # metrics that record the item's label, not an arm's output
# Per query: the calibration round s3-20261008T052442 measured $0.0012-0.0031 (2-4 researcher
# steps); the rest is room for a researcher that uses all 8 tool calls.
UNIT_COST_USD = 0.005


def build_manifest() -> dict[str, Any]:
    rng = random.Random(SEED)
    items = []
    for split, name in SPLITS.items():
        qids = sorted(pasa.queries(name))
        items += [{"id": q, "set": name, "split": split} for q in rng.sample(qids, PER_SPLIT)]
    return {
        "suite": "s3",
        "dataset": {n: pasa.digest(pasa.FILES[n]) for n in SPLITS.values()},
        "seed": SEED,
        "items": items,
    }


def manifest_hash() -> str:
    return sha256(MANIFEST.read_bytes()).hexdigest()[:12]


def splits() -> dict[str, str]:
    return {e["id"]: e["split"] for e in json.loads(MANIFEST.read_text())["items"]}


def load(limit: int | None = None) -> list[tuple[str, Query]]:
    """(split, query) for the first `limit` items of each split, from the exact local files."""
    manifest = json.loads(MANIFEST.read_text())
    for name, digest in manifest["dataset"].items():
        if pasa.digest(pasa.FILES[name]) != digest:
            raise ValueError(f"{pasa.FILES[name]} differs from the file S3 was drawn from")
    data = {name: pasa.queries(name) for name in SPLITS.values()}
    chosen = []
    for split in SPLITS:
        entries = [e for e in manifest["items"] if e["split"] == split][:limit]
        chosen += [(split, data[e["set"]][e["id"]]) for e in entries]
    return chosen


def request(query: Query) -> ResearchRequest:
    """The request understand would make of the query; S3 measures discovery alone."""
    return ResearchRequest(
        intent="discover",
        clarification=None,
        need=query.text,
        question=None,
        paper_ids=[],
        listed=[],
        count=MAX_LISTED,
        constraints=[],
        priorities=[],
        titles=[],
        prefer_recent=False,  # S3 grades search quality, not the recency preference (D24)
        published_after=None,
        published_before=query.cutoff.isoformat(),
        history=None,
    )
