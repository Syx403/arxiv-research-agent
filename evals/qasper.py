"""QASPER (CC BY 4.0): the validation split from Hugging Face's Parquet conversion, kept in the
git-ignored data/datasets/qasper/, and the gold evidence of each question as paragraph indices."""

import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq

from ara.rag.sources import ParsedPaper, qasper_paper
from ara.settings import ROOT

URL = (
    "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/"
    "qasper/validation/0000.parquet"
)
PATH = ROOT / "data/datasets/qasper/validation.parquet"

type Record = Mapping[str, Any]


@dataclass(frozen=True)
class Question:
    id: str
    paper_id: str  # arXiv id
    text: str
    # One set of gold paragraph indices per annotator who found an answer; graders take the best.
    references: tuple[frozenset[int], ...]


def download(path: Path = PATH) -> str:
    """Fetch the file if it is missing; return its sha256."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(URL, follow_redirects=True, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return sha256(path.read_bytes()).hexdigest()


def records(path: Path = PATH) -> dict[str, Record]:
    return {record["id"]: record for record in pq.read_table(path).to_pylist()}


def questions(record: Record) -> Iterator[Question]:
    """Questions whose every answering annotator cites evidence found verbatim in the paragraphs.
    Unanswerable questions and evidence from figures or tables are left out (they belong to S2)."""
    paper = qasper_paper(record)
    where = paragraph_index(paper)
    qas = record["qas"]
    for qid, text, answers in zip(qas["question_id"], qas["question"], qas["answers"], strict=True):
        annotations = [a for a in answers["answer"] if not a["unanswerable"]]
        evidence = [[e.strip() for e in a["evidence"]] for a in annotations]
        if annotations and all(ev and all(e in where for e in ev) for ev in evidence):
            references = tuple(frozenset(where[e] for e in ev) for ev in evidence)
            yield Question(qid, record["id"], text, references)


def paragraph_index(paper: ParsedPaper) -> dict[str, int]:
    """Text → index of its first occurrence."""
    index: dict[str, int] = {}
    for i, paragraph in enumerate(paper.paragraphs):
        index.setdefault(paragraph.text, i)
    return index


def select(
    data: Mapping[str, Record], *, seed: int, papers: int, per_paper: int
) -> list[tuple[str, list[str]]]:
    """Papers with at least `per_paper` eligible questions, drawn with a fixed seed: [(paper,
    [question ids])], in draw order."""
    rng = random.Random(seed)
    eligible = {pid: sorted(q.id for q in questions(r)) for pid, r in sorted(data.items())}
    candidates = [pid for pid, ids in eligible.items() if len(ids) >= per_paper]
    return [(pid, rng.sample(eligible[pid], per_paper)) for pid in rng.sample(candidates, papers)]
