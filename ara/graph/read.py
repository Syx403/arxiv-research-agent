"""The read subgraph (DESIGN §4.2, D30): ingest per paper → search (each paper on its own, one
rerank for all) → select evidence per paper → collect, with at most one requery, per paper, for
the aspects its evidence left uncovered.

    START ─(Send ingest per paper)→ ingest → ready ─(Send search)→ search → staged
    staged ─(Send select per paper)→ select → collect
    collect ─(Send search per paper and aspect it missed, once)→ search → staged …  or → END
"""

import operator
import re
from itertools import zip_longest
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import BaseModel, Field

from ara.graph.reliability import LLM_TIMEOUT_S, RETRY, SLOT_BACKSTOP_S, degrade, problem
from ara.graph.state import Context, Evidence
from ara.llm.prompt import Block, Instructions, Prompt
from ara.llm.stages import STAGES
from ara.rag.ingest import document_id, ingest
from ara.rag.retrieve import Passage, retrieve

MAX_REQUERIES = 3  # missing aspects searched again, per turn over all papers (D21, D39)
LABEL = re.compile(r"S\d+")
BARE_ARXIV = re.compile(r"^arxiv:(\d{4}\.\d{4,5})$")
SELECT = Instructions.load("select_evidence")


class EvidenceSelection(BaseModel):
    sentences: list[str] = Field(description='Labels of the chosen sentences only, e.g. "S3".')
    missing: list[str] = Field(description="Uncovered aspects of the question, as search queries.")


class Found(BaseModel):
    round: int
    document: int
    query: str
    sentences: list[Evidence]  # ids are assigned in collect
    missing: list[str]  # aspects of the question these passages leave open


class ReadInput(TypedDict):
    question: str
    papers: list[str]  # references the context's `fetch` understands


class ReadOutput(TypedDict):
    evidence: list[Evidence]
    missing: list[str]
    documents: Annotated[list[int], operator.add]  # one per paper read
    problems: Annotated[list[str], operator.add]  # papers or searches that failed (M5a)


class IngestTask(TypedDict):
    reference: str


class SearchTask(TypedDict):
    question: str
    query: str
    documents: list[int]
    round: int


class SelectTask(TypedDict):
    question: str
    query: str
    document: int
    round: int
    passages: list[Passage]


class ReadState(ReadInput, ReadOutput):
    staged: Annotated[list[SelectTask], operator.add]  # passages awaiting selection, by round
    found: Annotated[list[Found], operator.add]
    requery: list[SearchTask]
    requeried: bool


def _not_read(state: IngestTask, error: BaseException) -> dict[str, Any]:
    """A paper that cannot be fetched or stored is skipped; the others are read (M5a)."""
    return {"documents": [], "problems": [problem(f"reading {state['reference']}", error)]}


@degrade(_not_read)
async def ingest_paper(state: IngestTask, runtime: Runtime[Context]) -> dict[str, Any]:
    """A reference that is a stored paper id ("qasper:…", "arxiv:…v3") is not fetched again. A bare
    "arxiv:2210.03629" means the latest version, which one metadata request names."""
    ctx = runtime.context
    reference = state["reference"]
    if (bare := BARE_ARXIV.match(reference)) is not None:
        reference = f"{reference}v{(await ctx.arxiv.metadata(bare[1])).version}"
    async with ctx.pool.connection() as conn:
        document = await document_id(conn, reference)
    if document is None:
        paper = await ctx.fetch(reference)
        document = await ingest(paper, pool=ctx.pool, gateway=ctx.gateway, scope=ctx.scope)
    return {"documents": [document]}


def ready(state: ReadState) -> dict[str, object]:
    """Joins the ingest branches: the search starts once every paper is stored."""
    return {}


def _no_passages(state: SearchTask, error: BaseException) -> dict[str, Any]:
    """A failed search gives its papers no passages, so they add no evidence (M5a)."""
    return {
        "staged": [_task(state, d, []) for d in state["documents"]],
        "problems": [problem("searching the papers", error)],
    }


def _task(state: SearchTask, document: int, passages: list[Passage]) -> SelectTask:
    return SelectTask(
        question=state["question"],
        query=state["query"],
        document=document,
        round=state["round"],
        passages=passages,
    )


@degrade(_no_passages)
async def search(state: SearchTask, runtime: Runtime[Context]) -> dict[str, Any]:
    """The best passages of each paper for one query (one rerank call for all of them)."""
    ctx = runtime.context
    found = await retrieve(
        state["query"], state["documents"], pool=ctx.pool, gateway=ctx.gateway, scope=ctx.scope
    )
    return {"staged": [_task(state, d, passages) for d, passages in found.items()]}


def staged(state: ReadState) -> dict[str, object]:
    """Joins the search branches: selection starts once every search of the round is done."""
    return {}


def _nothing_selected(state: SelectTask, error: BaseException) -> dict[str, Any]:
    """Selection that failed chooses nothing from this paper's passages (M5a)."""
    return {"found": [_found(state, [], [])], "problems": [problem("selecting evidence", error)]}


@degrade(_nothing_selected)
async def select(state: SelectTask, runtime: Runtime[Context]) -> dict[str, Any]:
    """Let the model pick the sentences of one paper's passages that matter for the question."""
    ctx = runtime.context
    passages = state["passages"]
    if not passages:
        return {"found": [_found(state, [], [])]}
    labelled = labels(passages)
    prompt = Prompt(
        SELECT,
        shared=(Block("user", f"Question: {state['question']}"),),
        item=(
            Block(
                "user", "Passages:\n" + "\n\n".join(_passage_text(p, labelled) for p in passages)
            ),
        ),
    )
    selection = await ctx.gateway.structured(
        STAGES["select_evidence"], prompt, EvidenceSelection, scope=ctx.scope
    )
    chosen = [
        labelled[label]
        for label in dict.fromkeys(_labels(selection.sentences))
        if label in labelled
    ]
    return {"found": [_found(state, chosen, selection.missing)]}


def _found(task: SelectTask, sentences: list[Evidence], missing: list[str]) -> Found:
    return Found(
        round=task["round"],
        document=task["document"],
        query=task["query"],
        sentences=sentences,
        missing=missing,
    )


def _labels(chosen: list[str]) -> list[str]:
    """The label at the start of each entry: models sometimes append the sentence ("S8: ...")."""
    return [m[0] for entry in chosen if (m := LABEL.match(entry.strip()))]


def labels(passages: list[Passage]) -> dict[str, Evidence]:
    """S1, S2, ... for every sentence of every passage, in order (ids are filled in later)."""
    sentences = [
        Evidence(
            id="",
            paper_id=p.paper_id,
            chunk_id=p.chunk_id,
            paragraph=p.paragraph,
            heading_path=p.heading_path,
            text=text,
        )
        for p in passages
        for text in p.sentence_texts()
    ]
    return {f"S{n}": sentence for n, sentence in enumerate(sentences, start=1)}


def _passage_text(passage: Passage, labelled: dict[str, Evidence]) -> str:
    lines = [f"[{passage.heading_path}]"]
    lines += [
        f"{label}: {e.text}" for label, e in labelled.items() if e.chunk_id == passage.chunk_id
    ]
    return "\n".join(lines)


def collect(state: ReadState) -> dict[str, object]:
    """Number the distinct sentences E1, E2, ... After the first round, search each paper once
    more for the aspects it reported missing; after the requery, report what is still uncovered."""
    seen: dict[tuple[int, str], Evidence] = {}
    for found in state.get("found", []):  # none when no paper could be read
        for sentence in found.sentences:
            seen.setdefault((sentence.chunk_id, sentence.text), sentence)
    evidence = [e.model_copy(update={"id": f"E{n}"}) for n, e in enumerate(seen.values(), 1)]
    if state.get("requeried", False):
        return {"evidence": evidence, "missing": uncovered(state.get("found", [])), "requery": []}
    requery = [
        SearchTask(question=state["question"], query=aspect, documents=[document], round=1)
        for document, aspect in requeries(state.get("found", []))
    ]
    return {"evidence": evidence, "missing": [], "requery": requery, "requeried": True}


def requeries(found: list[Found]) -> list[tuple[int, str]]:
    """The aspects to search again, MAX_REQUERIES in all, taken from the papers in turn (each
    paper's first missing aspect, then each one's second, ...): each requery is one rerank call,
    and rerank calls queue one per 6 s across every running turn (D39)."""
    per_paper = [[(f.document, a) for a in dict.fromkeys(f.missing)] for f in found]
    rounds = zip_longest(*per_paper)
    return [pair for row in rounds for pair in row if pair is not None][:MAX_REQUERIES]


def uncovered(found: list[Found]) -> list[str]:
    """Requeried aspects that no search covered: no first-round paper answered them (it chose
    sentences and did not list the aspect) and their own search chose nothing. The `missing` lists
    of the requery are not used: they judge the whole question from one aspect's passages."""
    first = [f for f in found if f.round == 0]
    again = [f for f in found if f.round == 1]

    def covered(aspect: str) -> bool:
        return any(f.sentences and aspect not in f.missing for f in first) or any(
            f.sentences for f in again if f.query == aspect
        )

    return [a for a in dict.fromkeys(f.query for f in again) if not covered(a)]


def to_ingest(state: ReadState) -> list[Send]:
    return [Send("ingest", IngestTask(reference=r)) for r in dict.fromkeys(state["papers"])]


def to_search(state: ReadState) -> list[Send] | str:
    """The question, once, over every paper read: one rerank call, each paper's own passages.
    When no paper could be read, there is nothing to search."""
    if not state["documents"]:
        return "collect"
    task = SearchTask(
        question=state["question"],
        query=state["question"],
        documents=list(dict.fromkeys(state["documents"])),
        round=0,
    )
    return [Send("search", task)]


def to_select(state: ReadState) -> list[Send]:
    current = 1 if state.get("requeried", False) else 0
    return [Send("select", t) for t in state["staged"] if t["round"] == current]


def after_collect(state: ReadState) -> list[Send] | str:
    return [Send("search", task) for task in state["requery"]] or END


def build() -> CompiledStateGraph[ReadState, Context, ReadInput, ReadOutput]:
    graph = StateGraph(
        ReadState, context_schema=Context, input_schema=ReadInput, output_schema=ReadOutput
    )
    graph.add_node(
        "ingest",
        ingest_paper,
        input_schema=IngestTask,
        retry_policy=RETRY,
        timeout=SLOT_BACKSTOP_S,
    )
    graph.add_node("ready", ready)
    graph.add_node(
        "search", search, input_schema=SearchTask, retry_policy=RETRY, timeout=SLOT_BACKSTOP_S
    )
    graph.add_node("staged", staged)
    graph.add_node(
        "select", select, input_schema=SelectTask, retry_policy=RETRY, timeout=LLM_TIMEOUT_S
    )
    graph.add_node("collect", collect)
    graph.add_conditional_edges(START, to_ingest, ["ingest"])
    graph.add_edge("ingest", "ready")
    graph.add_conditional_edges("ready", to_search, ["search", "collect"])
    graph.add_edge("search", "staged")
    graph.add_conditional_edges("staged", to_select, ["select"])
    graph.add_edge("select", "collect")
    graph.add_conditional_edges("collect", after_collect, ["search", END])
    return graph.compile(name="read")
