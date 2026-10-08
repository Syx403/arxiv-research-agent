"""The read subgraph (DESIGN §4.2): ingest per paper → gather per (scope, query) → collect, with
at most one requery for the aspects the evidence left uncovered. A scope is one document, so each
paper is searched on its own, or, for a library question, all documents at once (D29).

    START ─(Send ingest per paper)→ ingest → ready ─(Send gather per scope)→ gather → collect
    collect ─(Send gather per scope and the aspects it missed, once)→ gather …  or → END
"""

import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import BaseModel, Field

from ara.graph.state import Context, Evidence
from ara.llm.prompt import Block, Instructions, Prompt
from ara.llm.stages import STAGES
from ara.rag.ingest import document_id, ingest
from ara.rag.retrieve import Passage, retrieve

MAX_REQUERIES = 3  # missing aspects searched again, per scope (D21)
LABEL = re.compile(r"S\d+")
BARE_ARXIV = re.compile(r"^arxiv:(\d{4}\.\d{4,5})$")
SELECT = Instructions.load("select_evidence")


class EvidenceSelection(BaseModel):
    sentences: list[str] = Field(description='Labels of the chosen sentences only, e.g. "S3".')
    missing: list[str] = Field(description="Uncovered aspects of the question, as search queries.")


class Found(BaseModel):
    round: int
    documents: list[int]  # the scope searched
    query: str
    sentences: list[Evidence]  # ids are assigned in collect
    missing: list[str]  # aspects of the question these passages leave open


class ReadInput(TypedDict):
    question: str
    papers: list[str]  # references the context's `fetch` understands
    merged: bool  # one search over all the papers instead of one per paper (library, D29)


class ReadOutput(TypedDict):
    evidence: list[Evidence]
    missing: list[str]
    documents: Annotated[list[int], operator.add]  # one per paper read


class IngestTask(TypedDict):
    reference: str


class GatherTask(TypedDict):
    question: str
    query: str
    documents: list[int]
    round: int


class ReadState(ReadInput, ReadOutput):
    found: Annotated[list[Found], operator.add]
    requery: list[GatherTask]
    requeried: bool


async def ingest_paper(state: IngestTask, runtime: Runtime[Context]) -> dict[str, list[int]]:
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
    """Joins the ingest branches: the gather fan-out starts once every paper is stored."""
    return {}


async def gather(state: GatherTask, runtime: Runtime[Context]) -> dict[str, list[Found]]:
    """Search one scope for one query, then let the model pick the sentences that matter."""
    ctx = runtime.context
    passages = await retrieve(
        state["query"], state["documents"], pool=ctx.pool, gateway=ctx.gateway, scope=ctx.scope
    )
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


def _found(task: GatherTask, sentences: list[Evidence], missing: list[str]) -> Found:
    return Found(
        round=task["round"],
        documents=task["documents"],
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
    """Number the distinct sentences E1, E2, ... After the first round, search each scope once
    more for the aspects it reported missing; after the requery, report what is still uncovered."""
    seen: dict[tuple[int, str], Evidence] = {}
    for found in state["found"]:
        for sentence in found.sentences:
            seen.setdefault((sentence.chunk_id, sentence.text), sentence)
    evidence = [e.model_copy(update={"id": f"E{n}"}) for n, e in enumerate(seen.values(), 1)]
    if state.get("requeried", False):
        return {"evidence": evidence, "missing": uncovered(state["found"]), "requery": []}
    requery = [
        GatherTask(question=state["question"], query=aspect, documents=f.documents, round=1)
        for f in state["found"]
        for aspect in list(dict.fromkeys(f.missing))[:MAX_REQUERIES]
    ]
    return {"evidence": evidence, "missing": [], "requery": requery, "requeried": True}


def uncovered(found: list[Found]) -> list[str]:
    """Requeried aspects that no search covered: no first-round scope answered them (it chose
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


def to_gather(state: ReadState) -> list[Send]:
    documents = state["documents"]
    scopes = [documents] if state["merged"] else [[d] for d in documents]
    return [
        Send(
            "gather",
            GatherTask(question=state["question"], query=state["question"], documents=s, round=0),
        )
        for s in scopes
    ]


def after_collect(state: ReadState) -> list[Send] | str:
    return [Send("gather", task) for task in state["requery"]] or END


def build() -> CompiledStateGraph[ReadState, Context, ReadInput, ReadOutput]:
    graph = StateGraph(
        ReadState, context_schema=Context, input_schema=ReadInput, output_schema=ReadOutput
    )
    graph.add_node("ingest", ingest_paper, input_schema=IngestTask)
    graph.add_node("ready", ready)
    graph.add_node("gather", gather, input_schema=GatherTask)
    graph.add_node("collect", collect)
    graph.add_conditional_edges(START, to_ingest, ["ingest"])
    graph.add_edge("ingest", "ready")
    graph.add_conditional_edges("ready", to_gather, ["gather"])
    graph.add_edge("gather", "collect")
    graph.add_conditional_edges("collect", after_collect, ["gather", END])
    return graph.compile(name="read")
