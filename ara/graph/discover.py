"""The discover subgraph (DESIGN §4.2): a researcher tool loop on DeepSeek gathers a candidate pool
from arXiv; prerank keeps the abstracts closest to the need; Luna screens them in batches; rank
orders what survived.

    START → researcher ⇄ arxiv_tools (≤ 8 tool calls) → prerank → prewarm
    prewarm ─(Send screen per batch of 8)→ screen → rank → END
"""

import operator
from datetime import date
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import BaseModel, Field, ValidationError

from ara.arxiv.client import ArxivClient, Metadata, SearchError
from ara.graph.state import MAX_LISTED, Context, PaperCard, ResearchRequest
from ara.llm.gateway import Tool, worth_prewarming
from ara.llm.prompt import Block, Instructions, Prompt, ToolCall, data
from ara.llm.stages import STAGES
from ara.rag.embed import embed

MAX_TOOL_CALLS = 8  # per turn (DESIGN §2)
RESULTS = 20  # per search
SNIPPET = 300  # abstract characters shown to the researcher for a new paper
SHORTLIST, BATCH = 24, 8  # prerank keeps three screening batches
RESEARCHER = Instructions.load("researcher")
SCREEN = Instructions.load("screen")


class SearchArgs(BaseModel):
    query: str = Field(description='arXiv query, e.g. abs:"KV cache" AND abs:eviction')


class LookupArgs(BaseModel):
    arxiv_ids: list[str] = Field(description='arXiv ids, e.g. ["2305.18323"]')


TOOLS = (
    Tool("search_arxiv", "Search arXiv; up to 20 results by relevance.", SearchArgs),
    Tool("lookup", "Metadata for papers known by arXiv id.", LookupArgs),
)


class Judgement(BaseModel):
    id: str
    relevance: Literal[0, 1, 2, 3]
    reason: str
    violated: list[str] = Field(description="Quotes of the constraints the paper breaks.")


class Screening(BaseModel):
    papers: list[Judgement]


class DiscoverInput(TypedDict):
    request: ResearchRequest


class DiscoverOutput(TypedDict):
    papers: list[PaperCard]  # ranked, at most MAX_LISTED
    candidates: list[str]  # the pool the researcher found, by arXiv id (S3: search recall)
    shortlisted: list[str]  # what prerank sent to screening


class ScreenTask(TypedDict):
    request: ResearchRequest
    batch: list[PaperCard]


class DiscoverState(DiscoverInput, DiscoverOutput):
    transcript: list[Block]  # researcher turns and tool results, append-only
    found: dict[str, PaperCard]  # the candidate pool, in the order found
    tool_calls: int
    shortlist: list[PaperCard]
    judged: Annotated[list[Judgement], operator.add]


def request_block(request: ResearchRequest) -> Block:
    """The request as the researcher and the screen see it: the shared prefix of their calls."""
    return data(
        "Request",
        {
            "need": request.need,
            "constraints": [c.model_dump() for c in request.constraints],
            "published_after": request.published_after,
            "published_before": request.published_before,
        },
    )


async def researcher(state: DiscoverState, runtime: Runtime[Context]) -> dict[str, object]:
    ctx = runtime.context
    transcript = state.get("transcript", [])
    prompt = Prompt(RESEARCHER, shared=(request_block(state["request"]),), item=tuple(transcript))
    turn = await ctx.gateway.tool_step(STAGES["researcher"], prompt, TOOLS, scope=ctx.scope)
    return {"transcript": [*transcript, turn]}


async def arxiv_tools(state: DiscoverState, runtime: Runtime[Context]) -> dict[str, object]:
    """Run the last turn's tool calls within the budget; every call gets a tool message back."""
    found = dict(state.get("found", {}))
    used = state.get("tool_calls", 0)
    results = []
    for call in state["transcript"][-1].calls:
        if used < MAX_TOOL_CALLS:
            used += 1
            text = await run_tool(call, state["request"], found, runtime.context.arxiv)
        else:
            text = "Not run: the tool-call budget is used up."
        results.append(Block("tool", text, call_id=call.id))
    return {"transcript": [*state["transcript"], *results], "found": found, "tool_calls": used}


async def run_tool(
    call: ToolCall, request: ResearchRequest, found: dict[str, PaperCard], arxiv: ArxivClient
) -> str:
    """One tool call. Bad arguments and rejected queries go back to the model as text; results
    outside the request's date window are dropped here, whatever the query said."""
    after, before = _date(request.published_after), _date(request.published_before)
    try:
        if call.name == "search_arxiv":
            query = SearchArgs.model_validate_json(call.arguments).query
            papers = await arxiv.search(query, after=after, before=before, max_results=RESULTS)
        elif call.name == "lookup":
            ids = LookupArgs.model_validate_json(call.arguments).arxiv_ids[:RESULTS]
            papers = await arxiv.lookup(ids)
        else:
            return f"Unknown tool {call.name}."
    except ValidationError as error:
        return f"Invalid arguments: {error.errors()[0]['msg']}"
    except SearchError as error:
        return str(error)
    papers = [m for m in papers if (not after or m.published >= after)]
    papers = [m for m in papers if (not before or m.published <= before)]
    return describe(papers, found) or "No results."


def describe(papers: list[Metadata], found: dict[str, PaperCard]) -> str:
    """Add new papers to the pool; show the researcher their abstract opening, and only the id
    and title of papers it has seen before, so a repeated hit costs few tokens."""
    lines = []
    for m in papers:
        line = f"{m.arxiv_id} ({m.published.year}) {m.title}"
        if m.arxiv_id in found:
            lines.append(f"{line} [already found]")
            continue
        found[m.arxiv_id] = card(m)
        lines.append(f"{line}\n  {m.abstract[:SNIPPET]}")
    return "\n".join(lines)


def card(m: Metadata) -> PaperCard:
    return PaperCard(
        arxiv_id=m.arxiv_id,
        version=m.version,
        title=m.title,
        abstract=m.abstract,
        published=m.published.isoformat(),
    )


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


async def prerank(state: DiscoverState, runtime: Runtime[Context]) -> dict[str, object]:
    """Embedding similarity of each abstract to the need (deterministic, cached embeddings)."""
    ctx = runtime.context
    pool = list(state.get("found", {}).values())
    if not pool:
        return {"shortlist": [], "candidates": [], "shortlisted": []}
    texts = [state["request"].need, *(f"{p.title}. {p.abstract}" for p in pool)]
    need, *vectors = await embed(texts, pool=ctx.pool, gateway=ctx.gateway, scope=ctx.scope)
    # OpenAI embeddings have unit length, so the dot product is the cosine similarity.
    scored = [
        p.model_copy(update={"similarity": float(v @ need)})
        for p, v in zip(pool, vectors, strict=True)
    ]
    shortlist = sorted(scored, key=lambda p: -p.similarity)[:SHORTLIST]
    return {
        "shortlist": shortlist,
        "candidates": [p.arxiv_id for p in pool],
        "shortlisted": [p.arxiv_id for p in shortlist],
    }


def screen_prompt(request: ResearchRequest, batch: list[PaperCard]) -> Prompt:
    papers = [
        {"id": p.arxiv_id, "title": p.title, "published": p.published, "abstract": p.abstract}
        for p in batch
    ]
    return Prompt(SCREEN, shared=(request_block(request),), item=(data("Papers", papers),))


def batches(shortlist: list[PaperCard]) -> list[list[PaperCard]]:
    return [shortlist[i : i + BATCH] for i in range(0, len(shortlist), BATCH)]


async def prewarm(state: DiscoverState, runtime: Runtime[Context]) -> dict[str, object]:
    ctx = runtime.context
    groups = batches(state["shortlist"])
    if groups:
        prompt = screen_prompt(state["request"], groups[0])  # prewarm drops the item part
        if worth_prewarming(prompt, Screening, len(groups)):
            await ctx.gateway.prewarm(STAGES["screen"], prompt, Screening, scope=ctx.scope)
    return {}


async def screen(state: ScreenTask, runtime: Runtime[Context]) -> dict[str, list[Judgement]]:
    ctx = runtime.context
    prompt = screen_prompt(state["request"], state["batch"])
    result = await ctx.gateway.structured(STAGES["screen"], prompt, Screening, scope=ctx.scope)
    ids = {p.arxiv_id for p in state["batch"]}
    return {"judged": [j for j in result.papers if j.id in ids]}


def rank(state: DiscoverState) -> dict[str, list[PaperCard]]:
    """Keep papers judged relevant (≥ 2) that break no stated constraint; order by relevance, then
    similarity. A violation counts only if it quotes one of the request's constraints."""
    judged = {j.id: j for j in state.get("judged", [])}
    request = state["request"]
    quotes = {c.quote for c in request.constraints}
    cards = [
        p.model_copy(
            update={
                "relevance": j.relevance,
                "reason": j.reason,
                "violated": [q for q in j.violated if q in quotes],
            }
        )
        for p in state["shortlist"]
        if (j := judged.get(p.arxiv_id)) is not None
    ]
    kept = [c for c in cards if (c.relevance or 0) >= 2 and not c.violated]
    kept.sort(key=lambda c: (-(c.relevance or 0), -c.similarity))
    return {"papers": kept[: min(request.count or MAX_LISTED, MAX_LISTED)]}


def after_researcher(state: DiscoverState) -> str:
    calls = state["transcript"][-1].calls
    return "arxiv_tools" if calls and state.get("tool_calls", 0) < MAX_TOOL_CALLS else "prerank"


def after_tools(state: DiscoverState) -> str:
    """Once the budget is spent, asking the researcher again would only cost a call."""
    return "researcher" if state["tool_calls"] < MAX_TOOL_CALLS else "prerank"


def to_screen(state: DiscoverState) -> list[Send] | str:
    groups = batches(state["shortlist"])
    return [Send("screen", ScreenTask(request=state["request"], batch=b)) for b in groups] or "rank"


def build() -> CompiledStateGraph[DiscoverState, Context, DiscoverInput, DiscoverOutput]:
    graph = StateGraph(
        DiscoverState,
        context_schema=Context,
        input_schema=DiscoverInput,
        output_schema=DiscoverOutput,
    )
    graph.add_node("researcher", researcher)
    graph.add_node("arxiv_tools", arxiv_tools)
    graph.add_node("prerank", prerank)
    graph.add_node("prewarm", prewarm)
    graph.add_node("screen", screen, input_schema=ScreenTask)
    graph.add_node("rank", rank)
    graph.add_edge(START, "researcher")
    graph.add_conditional_edges("researcher", after_researcher, ["arxiv_tools", "prerank"])
    graph.add_conditional_edges("arxiv_tools", after_tools, ["researcher", "prerank"])
    graph.add_edge("prerank", "prewarm")
    graph.add_conditional_edges("prewarm", to_screen, ["screen", "rank"])
    graph.add_edge("screen", "rank")
    graph.add_edge("rank", END)
    return graph.compile(name="discover")
