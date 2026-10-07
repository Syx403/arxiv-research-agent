"""Domain objects shared by the read and answer subgraphs (DESIGN §4.3), and the run context that
carries their dependencies (LangGraph `context_schema`: no globals)."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel

from ara.db.pool import Pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.rag.sources import ParsedPaper

ABSTAIN = "Not stated in the provided papers."


class Evidence(BaseModel):
    """One sentence from a paper, labelled E1, E2, ... when the evidence is collected."""

    id: str
    paper_id: str
    chunk_id: int
    paragraph: int
    heading_path: str
    text: str


class Claim(BaseModel):
    """One line of an answer: index 0 is the direct answer, the rest are explanation sentences."""

    index: int
    text: str
    citations: list[str]

    @property
    def direct(self) -> bool:
        """The direct answer is verified together with the question it answers (D21)."""
        return self.index == 0

    @property
    def key(self) -> str:
        """Identity for verdict reuse: a line repeated unchanged after repair is not re-verified."""
        return f"{self.direct} {self.text} {sorted(self.citations)}"


class Verdict(BaseModel):
    supported: bool
    problem: str  # empty when supported


class Answer(BaseModel):
    question: str
    short: str  # the direct answer, or ABSTAIN
    abstained: bool
    sentences: list[Claim]  # verified explanation lines
    dropped: list[Claim]  # final-draft lines not delivered: unsupported, uncited, or withheld
    checked: int  # distinct lines the verifier judged, over both drafts
    rejected: list[Claim]  # lines the verifier judged unsupported, over both drafts
    evidence: list[Evidence]  # every evidence sentence the delivered lines cite

    def render(self) -> str:
        lines = [self.short, *(f"{c.text} [{', '.join(c.citations)}]" for c in self.sentences)]
        return "\n".join(lines)


@dataclass(frozen=True)
class Context:
    pool: Pool
    gateway: Gateway
    scope: Scope
    fetch: Callable[[str], Awaitable[ParsedPaper]]  # paper reference → parsed full text
