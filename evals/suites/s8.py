"""S8 open questions (D41, D42): the questions real users ask most — research a named model or
method, explain a paper, compare two papers, survey a topic, find the latest work — in English and
Chinese. Messages come from real queries (the Asta Interaction Dataset, SciArena) or from QA
datasets over papers (QASA, SciDQA, ScholarQA-CS2 and others), filtered by an external build
(`docs/v2/eval/s8-dataset-brief.md`); nothing is invented except translations and rubrics drawn
from a reference answer. `uv run python -m evals.suites.s8` validates the file (schema, balance,
every evidence quote found word for word in its paper's arXiv text; free: arXiv only, no model
call). Running and grading the suite come later."""

import asyncio
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from ara.arxiv.client import ArxivClient
from ara.graph.state import plain

DATA = Path(__file__).parents[1] / "datasets" / "s8_open.json"
TYPES = ("survey_named", "explain_paper", "compare_papers", "topic_survey", "latest_topic")
VERSIONED = re.compile(r"^\d{4}\.\d{4,5}v\d+$")
CJK = re.compile(r"[一-鿿]")

type ItemType = Literal[
    "survey_named", "explain_paper", "compare_papers", "topic_survey", "latest_topic"
]


class Evidence(BaseModel):
    paper: str  # versioned arXiv id, one of the item's papers
    section: str  # the heading path as the arXiv HTML shows it, e.g. "Method › Planner"
    quote: str  # one or more whole sentences copied exactly from that paper's text


class RubricItem(BaseModel):
    id: str
    item: str  # one point a good answer covers, as a checkable statement
    kind: Literal[
        "definition", "mechanism", "result", "comparison", "limitation", "example", "other"
    ]
    evidence: list[Evidence] = []  # required for explain_paper and compare_papers


class Source(BaseModel):
    dataset: Literal[
        "Asta",
        "SciArena",
        "QASA",
        "SciDQA",
        "M3SciQA",
        "ScholarQA-CS2",
        "ScholarQA-CS",
        "ResearchQA",
    ]
    id: str  # the source's own id (for Asta, the hashed thread id and the query time)
    license: str
    origin: Literal["real_query", "dataset_question"]  # asked by a user, or written for a dataset
    translated: bool = False  # message translated by the build, from message_en
    # the source's own rubric, points split from its reference answer, points written by the build
    # from the paper itself (each with evidence), or no rubric
    rubric_from: Literal["dataset", "reference_answer", "paper", "none"] = "none"


class Expect(BaseModel):
    intent: Literal["discover", "discover_read", "read"]
    primary: str | None = None  # unused since D43: every discovery item is graded the same way
    recent_days: int | None = None  # latest_topic: listed papers should be this new


class Item(BaseModel):
    id: str = Field(pattern=r"^s8-[a-z0-9-]+$")
    type: ItemType
    split: Literal["dev", "test"]
    language: Literal["English", "Chinese"]
    message: str  # what the user types
    message_en: str  # the same request in English (equal to message for English items)
    source: Source
    papers: list[str] = []  # versioned arXiv ids the message names, or survey_named's primary
    names: list[str] = []  # survey_named: the subject as the user writes it
    rubric: list[RubricItem] = []
    reference_answer: str | None = None  # the dataset's own answer, verbatim, when it has one
    expect: Expect
    reviewed: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def shaped(self) -> "Item":
        problems = []
        if self.language == "Chinese" and not CJK.search(self.message):
            problems.append("a Chinese item's message has no Chinese")
        if self.language == "English" and self.message != self.message_en:
            problems.append("an English item's message_en must equal message")
        if any(not VERSIONED.match(p) for p in self.papers):
            problems.append("papers must be versioned arXiv ids like 2305.18323v1")
        needs = {
            "survey_named": (0, 0, False),  # (papers, rubric items, evidence required)
            "explain_paper": (1, 3, True),
            "compare_papers": (2, 4, True),
            "topic_survey": (0, 4, False),
            "latest_topic": (0, 0, False),
        }[self.type]
        if len(self.papers) != needs[0]:
            problems.append(f"{self.type} names exactly {needs[0]} paper(s)")
        if not needs[1] <= len(self.rubric) <= 8 and needs[1]:
            problems.append(f"{self.type} needs {needs[1]} to 8 rubric items")
        if needs[2] and any(not r.evidence for r in self.rubric):
            problems.append("every rubric item needs evidence quotes")
        if any(e.paper not in self.papers for r in self.rubric for e in r.evidence):
            problems.append("evidence must quote one of the item's papers")
        if self.type == "survey_named" and not self.names:
            problems.append("survey_named needs names")
        if self.type == "latest_topic" and not self.expect.recent_days:
            problems.append("latest_topic needs expect.recent_days")
        if self.rubric and self.source.rubric_from == "none":
            problems.append("say where the rubric comes from (source.rubric_from)")
        if self.source.rubric_from == "paper" and self.type not in (
            "explain_paper",
            "compare_papers",
        ):
            problems.append("only explain and compare rubrics may be written from the paper")
        if self.source.translated and self.language != "Chinese":
            problems.append("only Chinese messages are translated")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class Suite(BaseModel):
    suite: Literal["s8"]
    created: str
    sources: list[dict[str, str]]  # attribution: name, url, license, citation
    items: list[Item]


def load(path: Path = DATA) -> list[Item]:
    return Suite.model_validate_json(path.read_text()).items


def balance(items: list[Item]) -> list[str]:
    """Each type has items in both splits and both languages; ids are unique; no request occurs
    twice, a translation included (it would put one request in both splits)."""
    problems = [f"duplicate id {i}" for i, n in Counter(i.id for i in items).items() if n > 1]
    requests = Counter(plain(i.message_en) for i in items)
    problems += [f"request asked twice: {r[:60]}" for r, n in requests.items() if n > 1]
    for kind in TYPES:
        mine = [i for i in items if i.type == kind]
        for field, values in (("split", ("dev", "test")), ("language", ("English", "Chinese"))):
            missing = {v for v in values} - {getattr(i, field) for i in mine}
            if mine and missing:
                problems.append(f"{kind}: no {field} {sorted(missing)}")
    return problems


async def quotes(items: list[Item]) -> list[str]:
    """Every evidence quote appears in its paper's text as ARA parses it (whitespace and case
    folded), so retrieval can be scored against it."""
    problems: list[str] = []
    texts: dict[str, str] = {}
    async with ArxivClient() as arxiv:
        for paper in sorted({e.paper for i in items for r in i.rubric for e in r.evidence}):
            parsed = await arxiv.paper(paper)
            texts[paper] = plain(" ".join(p.text for p in parsed.paragraphs))
    for item in items:
        for rubric in item.rubric:
            for e in rubric.evidence:
                if plain(e.quote) not in texts[e.paper]:
                    problems.append(f"{item.id}/{rubric.id}: quote not found in {e.paper}")
    return problems


def main() -> None:
    try:
        items = load(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA)
    except ValidationError as error:
        print(error)
        sys.exit(1)
    problems = balance(items) + asyncio.run(quotes(items))
    counts = Counter((i.type, i.split, i.language) for i in items)
    for key, n in sorted(counts.items()):
        print(*key, n)
    print("\n".join(problems) or f"OK: {len(items)} items")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
