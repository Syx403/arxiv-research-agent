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
UNANSWERABLE = "Unanswerable"


@dataclass(frozen=True)
class Gold:
    kind: str  # "extractive", "yes_no", "free_form" or "unanswerable"
    text: str
    evidence: frozenset[int]  # paragraph indices


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


def golds(record: Record, question_id: str) -> list[Gold]:
    """Every annotator's answer to one question, written the way the official scorer writes it:
    extractive spans joined by ", ", "Yes"/"No", the free-form text, or "Unanswerable"."""
    where = paragraph_index(qasper_paper(record))
    qas = record["qas"]
    answers = qas["answers"][qas["question_id"].index(question_id)]["answer"]
    result = []
    for a in answers:
        evidence = frozenset(where[e.strip()] for e in a["evidence"] if e.strip() in where)
        if a["unanswerable"]:
            result.append(Gold("unanswerable", UNANSWERABLE, evidence))
        elif a["yes_no"] is not None:
            result.append(Gold("yes_no", "Yes" if a["yes_no"] else "No", evidence))
        elif a["extractive_spans"]:
            result.append(Gold("extractive", ", ".join(a["extractive_spans"]), evidence))
        else:
            result.append(Gold("free_form", a["free_form_answer"], evidence))
    return result


def unanswerable(record: Record) -> list[str]:
    """Questions every annotator marked unanswerable."""
    qas = record["qas"]
    return [
        qid
        for qid, answers in zip(qas["question_id"], qas["answers"], strict=True)
        if all(a["unanswerable"] for a in answers["answer"])
    ]
