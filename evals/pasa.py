"""PaSa query sets (CC BY-NC-SA 4.0, gated): local copies in the git-ignored data/datasets/pasa/,
never committed, uploaded or redistributed (CLAUDE.md rule 10). Only query ids leave this module
for committed files."""

import json
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path

from ara.settings import ROOT

DIR = ROOT / "data/datasets/pasa"
FILES = {
    "AutoScholarQuery": DIR / "AutoScholarQuery/test.jsonl",
    "RealScholarQuery": DIR / "RealScholarQuery/test.jsonl",
}


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    cutoff: date  # the search date limit: papers published after it cannot be gold
    gold: frozenset[str]  # arXiv ids without version


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def queries(name: str) -> dict[str, Query]:
    found = {}
    for line in FILES[name].read_text().splitlines():
        record = json.loads(line)
        cutoff = record["source_meta"]["published_time"]
        found[record["qid"]] = Query(
            id=record["qid"],
            text=record["question"],
            cutoff=date(int(cutoff[:4]), int(cutoff[4:6]), int(cutoff[6:])),
            gold=frozenset(i.split("v")[0] for i in record["answer_arxiv_id"]),
        )
    return found
