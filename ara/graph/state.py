"""Domain objects shared by the graphs (DESIGN §4.3), and the run context that carries their
dependencies (LangGraph `context_schema`: no globals)."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from ara.arxiv.client import ArxivClient
from ara.db.pool import Pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.rag.sources import ParsedPaper

ABSTAIN = "Not stated in the provided papers."
MAX_LISTED, MAX_READ = 5, 3  # papers per turn (DESIGN §2)

type Intent = Literal["discover", "discover_read", "read", "library", "other"]


def plain(text: str) -> str:
    """Whitespace and case folded: how a quote the model copied is compared with its source."""
    return " ".join(text.split()).casefold()


class Constraint(BaseModel):
    """A hard requirement on papers: screening drops a paper that breaks it."""

    quote: str = Field(description="The user's own words stating the constraint, copied exactly.")
    meaning: str = Field(description="What a paper must or must not be, to satisfy it.")


class Priority(BaseModel):
    """What the user cares about (a cost, a goal): it steers relevance and the answer's focus, and
    never removes a paper (D24)."""

    quote: str = Field(description="The user's own words stating what matters, copied exactly.")
    meaning: str = Field(description="What the papers and the answer should address because of it.")


class ResearchRequest(BaseModel):
    """What understand makes of the latest message. Code checks it before use (trust boundary)."""

    intent: Intent
    clarification: str | None = Field(
        description="One question to ask first, when the request cannot be acted on as it stands."
    )
    need: str = Field(description="The research need as one self-contained English sentence.")
    question: str | None = Field(
        description="What to answer from the papers' full text (read intents), in English."
    )
    paper_ids: list[str] = Field(description='arXiv ids the user names, e.g. "2305.18323v1".')
    listed: list[int] = Field(description="1-based positions in the papers shown last turn.")
    count: int | None = Field(description="How many papers the user asks for, if stated.")
    constraints: list[Constraint] = Field(description="Hard constraints the user states.")
    priorities: list[Priority] = Field(description="What the user cares about, not a filter.")
    titles: list[str] = Field(description="Titles or names of papers the user names, not ids.")
    prefer_recent: bool = Field(description="Newer papers first among equally relevant ones.")
    published_after: str | None = Field(description="YYYY-MM-DD, only if the user limits dates.")
    published_before: str | None = Field(description="YYYY-MM-DD, only if the user limits dates.")


class PaperCard(BaseModel):
    """A candidate paper: arXiv metadata, then what discovery learned about it."""

    arxiv_id: str
    version: int
    title: str
    abstract: str
    published: str  # YYYY-MM-DD, first version
    similarity: float = 0.0  # prerank: embedding similarity of the abstract to the need
    relevance: int | None = None  # screen: 0-3
    reason: str = ""
    violated: list[str] = []  # quotes of the constraints the paper breaks

    @property
    def reference(self) -> str:
        return f"arxiv:{self.arxiv_id}v{self.version}"


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
    arxiv: ArxivClient
    today: date = field(default_factory=date.today)  # when the user asks; evals pin it (D24)
