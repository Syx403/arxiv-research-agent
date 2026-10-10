"""The discover subgraph (DESIGN §4.2): a researcher tool loop on DeepSeek gathers a candidate pool
from arXiv; prerank keeps the abstracts closest to the need; Luna screens them in batches; rank
orders what survived.

    START → researcher ⇄ arxiv_tools (≤ 8 tool calls) → prerank → prewarm
    prewarm ─(Send screen per batch of 8)→ screen → rank → END
"""

import operator
import re
from collections.abc import Mapping
from datetime import date
from typing import Annotated, Any, Literal, TypedDict

import httpx
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Send
from pydantic import BaseModel, Field, ValidationError

from ara.arxiv.client import Metadata, SearchError
from ara.graph.reliability import (
    LLM_TIMEOUT_S,
    RETRY,
    SLOT_BACKSTOP_S,
    degrade,
    problem,
)
from ara.graph.state import MAX_LISTED, Context, PaperCard, ResearchRequest, plain
from ara.llm.gateway import Tool, worth_prewarming
from ara.llm.prompt import Block, Instructions, Prompt, ToolCall, data
from ara.llm.stages import STAGES
from ara.rag.embed import embed

MAX_TOOL_CALLS = 8  # per turn (DESIGN §2)
MAX_FAILED_ROUNDS = 2  # researcher rounds in a row with a failed arXiv request (D38)
RESULTS = 20  # per search
SNIPPET = 300  # abstract characters shown to the researcher for a new paper
SHORTLIST, BATCH = 24, 8  # prerank keeps three screening batches
MENTIONS = 40  # papers naming what the request is about that prerank may keep: five batches (D41)
TITLE_RESULTS = 10  # per title search when the user names a paper (D29)
RESEARCHER = Instructions.load("researcher")
SCREEN = Instructions.load("screen")
PAPER_ID = re.compile(r"\d{4}\.\d{4,5}")  # inside "2401.00001v2" or "arXiv:2401.00001"


class SearchArgs(BaseModel):
    query: str = Field(description='arXiv query, e.g. abs:"curriculum learning" AND abs:ordering')
    newest_first: bool = Field(
        default=False, description="Order by submission date, newest first, instead of relevance."
    )


class LookupArgs(BaseModel):
    arxiv_ids: list[str] = Field(description='arXiv ids, e.g. ["2305.18323"]')


TOOLS = (
    Tool(
        "search_arxiv", "Search arXiv; up to 20 results, by relevance or newest first.", SearchArgs
    ),
    Tool("lookup", "Metadata for papers known by arXiv id.", LookupArgs),
)


class Judgement(BaseModel):
    id: str
    relevance: Literal[0, 1, 2, 3]
    reason: str
    violated: list[str] = Field(description="Quotes of the constraints the paper breaks.")
    named: str | None = Field(
        description="The entry of the request's titles this paper is, if it is that paper."
    )


class Screening(BaseModel):
    papers: list[Judgement]


class DiscoverInput(TypedDict):
    request: ResearchRequest
    library: list[PaperCard]  # the user's papers close to the need: candidates too (D30)


class DiscoverOutput(TypedDict):
    papers: list[PaperCard]  # ranked, at most MAX_LISTED
    candidates: list[str]  # the pool the researcher found, by arXiv id (S3: search recall)
    shortlisted: list[str]  # what prerank sent to screening
    unjudged: list[str]  # shortlisted papers the screen returned no judgement for
    problems: Annotated[list[str], operator.add]  # parts that failed after retries (M5a)


class ScreenTask(TypedDict):
    request: ResearchRequest
    batch: list[PaperCard]


class DiscoverState(DiscoverInput, DiscoverOutput):
    transcript: list[Block]  # researcher turns and tool results, append-only
    found: dict[str, PaperCard]  # the candidate pool, in the order found
    tool_calls: int
    failed_rounds: int  # researcher rounds in a row whose arXiv requests failed (D38)
    shortlist: list[PaperCard]
    judged: Annotated[list[Judgement], operator.add]


def request_block(request: ResearchRequest) -> Block:
    """The request as the researcher and the screen see it: the shared prefix of their calls."""
    return data(
        "Request",
        {
            "need": request.need,
            "constraints": [c.model_dump() for c in request.constraints],
            "priorities": [p.model_dump() for p in request.priorities],
            "titles": request.titles,
            "language": request.language,
            "prefer_recent": request.prefer_recent,
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
    """Run the last turn's tool calls within the budget; every call gets a tool message back.
    Once arXiv fails (rate limit, error, timeout), the round's other calls are not sent: they
    would meet the same limit (D38)."""
    found = dict(state.get("found", {}))
    used = state.get("tool_calls", 0)
    results, problems = list[Block](), list[str]()
    for call in state["transcript"][-1].calls:
        if problems:
            text = "Not run: arXiv did not answer the previous request."
        elif used < MAX_TOOL_CALLS:
            used += 1
            try:
                text = await run_tool(call, state["request"], found, runtime.context)
            except httpx.HTTPError as error:  # arXiv busy or unreachable (M5a)
                problems.append(problem("an arXiv request", error))
                text = "arXiv did not answer."
        else:
            text = "Not run: the tool-call budget is used up."
        results.append(Block("tool", text, call_id=call.id))
    return {
        "transcript": [*state["transcript"], *results],
        "found": found,
        "tool_calls": used,
        "failed_rounds": state.get("failed_rounds", 0) + 1 if problems else 0,
        "problems": problems,
    }


async def run_tool(
    call: ToolCall, request: ResearchRequest, found: dict[str, PaperCard], ctx: Context
) -> str:
    """One tool call. Bad arguments and rejected queries go back to the model as text (an arXiv
    that does not answer too, in `arxiv_tools`, which also records the problem); results
    outside the request's date window, which ends today at the latest, are dropped here, whatever
    the query said."""
    after = _date(request.published_after)
    before = _date(request.published_before) or ctx.today
    try:
        if call.name == "search_arxiv":
            args = SearchArgs.model_validate_json(call.arguments)
            papers = await ctx.arxiv.search(
                args.query,
                after=after,
                before=before,
                newest_first=args.newest_first,
                max_results=RESULTS,
            )
        elif call.name == "lookup":
            ids = LookupArgs.model_validate_json(call.arguments).arxiv_ids[:RESULTS]
            papers = await ctx.arxiv.lookup(ids)
        else:
            return f"Unknown tool {call.name}."
    except ValidationError as error:
        return f"Invalid arguments: {error.errors()[0]['msg']}"
    except SearchError as error:
        return str(error)
    papers = [m for m in papers if (not after or m.published >= after)]
    papers = [m for m in papers if m.published <= before]
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


def _newest(card: PaperCard) -> float:
    return -date.fromisoformat(card.published).toordinal()


def _closest(card: PaperCard) -> float:
    return -card.similarity


async def prerank(state: DiscoverState, runtime: Runtime[Context]) -> dict[str, object]:
    """Embedding similarity of each abstract to the need (deterministic, cached embeddings). The
    user's papers close to the need join the arXiv results on equal terms (D30)."""
    ctx = runtime.context
    found = state.get("found", {})
    pool = [*found.values(), *(p for p in state["library"] if p.arxiv_id not in found)]
    if not pool:
        return {"shortlist": [], "candidates": [], "shortlisted": []}
    texts = [state["request"].need, *(f"{p.title}. {p.abstract}" for p in pool)]
    need, *vectors = await embed(texts, pool=ctx.pool, gateway=ctx.gateway, scope=ctx.scope)
    # OpenAI embeddings have unit length, so the dot product is the cosine similarity.
    scored = [
        p.model_copy(update={"similarity": float(v @ need)})
        for p, v in zip(pool, vectors, strict=True)
    ]
    # papers the user named by title always reach screening, closest to the need or not (D28);
    # then papers that name what the request is about: similarity cannot tell "Kimi K3" from
    # "Kimi-VL", and a paper naming it once mid-abstract scores low (D41); then the rest
    request = state["request"]
    pattern = name_pattern(request.names)

    def tier(p: PaperCard) -> int:
        if any(titled(t, p.title) for t in request.titles):
            return 0
        return 1 if pattern and pattern.search(f"{p.title} {p.abstract}") else 2

    ordered = sorted(scored, key=lambda p: (tier(p), -p.similarity))
    first = sum(tier(p) < 2 for p in ordered)
    shortlist = ordered[: max(SHORTLIST, min(first, MENTIONS))]
    return {
        "shortlist": shortlist,
        "candidates": [p.arxiv_id for p in pool],
        "shortlisted": [p.arxiv_id for p in shortlist],
    }


def screen_prompt(
    request: ResearchRequest,
    batch: list[PaperCard],
    passages: Mapping[str, list[str]] | None = None,
) -> Prompt:
    """The batch as the screen sees it; a library paper also brings the full-text passages
    nearest the question, so what only the body says can count (D30)."""
    papers = [
        {"id": p.arxiv_id, "title": p.title, "published": p.published, "abstract": p.abstract}
        | ({"passages": passages[p.reference]} if passages and passages.get(p.reference) else {})
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


def _unscreened(state: ScreenTask, error: BaseException) -> dict[str, Any]:
    """A batch the screen could not judge is reported unjudged, as before (D23, M5a)."""
    return {"judged": [], "problems": [problem("screening a batch", error)]}


@degrade(_unscreened)
async def screen(state: ScreenTask, runtime: Runtime[Context]) -> dict[str, Any]:
    ctx = runtime.context
    prompt = screen_prompt(state["request"], state["batch"])
    result = await ctx.gateway.structured(STAGES["screen"], prompt, Screening, scope=ctx.scope)
    return {"judged": in_batch(result.papers, state["batch"])}


def in_batch(judged: list[Judgement], batch: list[PaperCard]) -> list[Judgement]:
    """Judgements of this batch's papers, by bare arXiv id: the model may write "2401.00001v2" or
    "arXiv:2401.00001" for "2401.00001"; any other id is dropped."""
    ids = {p.arxiv_id for p in batch}
    bare = [j.model_copy(update={"id": m[0]}) for j in judged if (m := PAPER_ID.search(j.id))]
    return [j for j in bare if j.id in ids]


def rank(state: DiscoverState) -> dict[str, object]:
    """The papers the user named come first, one per title, whatever their relevance or
    constraints: the user asked for them (D28). Then papers judged relevant (≥ 2) that break no
    stated constraint, by relevance, then, within a grade, newest first when the request prefers
    recent work and by similarity otherwise (D24); at most MAX_LISTED in all."""
    judged = {j.id: j for j in state.get("judged", [])}
    request = state["request"]
    cards = [
        judged_card(request, p, j)
        for p in state["shortlist"]
        if (j := judged.get(p.arxiv_id)) is not None
    ]
    named = list({c.named: c for c in reversed(cards) if c.named}.values())[::-1]  # best per title
    others = [c for c in cards if c not in named and (c.relevance or 0) >= 2 and not c.violated]
    within = _newest if request.prefer_recent else _closest
    others.sort(key=lambda c: (-(c.relevance or 0), within(c)))
    limit = min(max(request.count or MAX_LISTED, len(named)), MAX_LISTED)
    return {
        "papers": [*named, *others][:limit],
        "unjudged": [p.arxiv_id for p in state["shortlist"] if p.arxiv_id not in judged],
    }


def judged_card(request: ResearchRequest, paper: PaperCard, judgement: Judgement) -> PaperCard:
    """The screen's judgement on the card. A violation counts only if it quotes one of the
    request's constraints, compared as understand compares quotes (whitespace and case folded)."""
    quotes = {plain(c.quote): c.quote for c in request.constraints}
    return paper.model_copy(
        update={
            "relevance": judgement.relevance,
            "reason": judgement.reason,
            "violated": [quotes[plain(q)] for q in judgement.violated if plain(q) in quotes],
            "named": name(request.titles, paper.title, judgement.named),
        }
    )


def name(titles: list[str], title: str, judged: str | None) -> str | None:
    """Which of the user's titles a paper is: by its own title first (code), else by the screen's
    judgement, which knows that "LLMCompiler" is "An LLM Compiler for Parallel Function Calling"."""
    by_code = next((t for t in titles if titled(t, title)), None)
    by_model = next((t for t in titles if judged and plain(judged) == plain(t)), None)
    return by_code or by_model


def name_pattern(names: list[str]) -> re.Pattern[str] | None:
    """Any of the names as a whole word, case-insensitive, its parts joined by any spacing or
    hyphen: "Kimi K3" finds "Kimi-K3" and "kimi k3", not "Kimi K30" (D41)."""
    parts = [re.findall(r"[^\W_]+", n) for n in names]
    alternatives = [r"[\s\-_]*".join(map(re.escape, p)) for p in parts if p]
    if not alternatives:
        return None
    return re.compile(rf"(?<![^\W_])(?:{'|'.join(alternatives)})(?![^\W_])", re.IGNORECASE)


def titled(name: str, title: str) -> bool:
    """The paper's title is the one the user named: equal, or starting with it ("ReWOO: ...")."""
    full, named = plain(title), plain(name)
    return bool(named) and (full == named or full.startswith(named + ":"))


async def find_title(name: str, ctx: Context) -> list[PaperCard]:
    """arXiv papers whose title is `name` by `titled`, most relevant first, from one title search
    (free; no model). A query arXiv rejects finds nothing."""
    query = 'ti:"' + " ".join(name.replace('"', " ").split()) + '"'
    try:
        found = await ctx.arxiv.search(query, before=ctx.today, max_results=TITLE_RESULTS)
    except SearchError:
        return []
    return [card(m) for m in found if titled(name, m.title)]


async def judge(
    request: ResearchRequest,
    papers: list[PaperCard],
    ctx: Context,
    passages: Mapping[str, list[str]] | None = None,
) -> list[PaperCard]:
    """The screen's judgement on papers outside discovery, in one call (≤ one batch): named papers,
    for their constraints (D29); library papers, for their relevance (D30)."""
    prompt = screen_prompt(request, papers, passages)
    result = await ctx.gateway.structured(STAGES["screen"], prompt, Screening, scope=ctx.scope)
    judged = {j.id: j for j in in_batch(result.papers, papers)}
    return [judged_card(request, p, j) if (j := judged.get(p.arxiv_id)) else p for p in papers]


def search_stopped(state: DiscoverState, error: NodeError) -> Command[str]:
    """The researcher or a tool turn failed after retries: rank what was found so far (M5a)."""
    return Command(update={"problems": [problem("the arXiv search", error.error)]}, goto="prerank")


def prerank_failed(state: DiscoverState, error: NodeError) -> Command[str]:
    """Without embeddings, the first candidates found go to screening, in the order found."""
    pool = list(state.get("found", {}).values())
    shortlist = [*state["library"], *pool][:SHORTLIST]
    return Command(
        update={
            "shortlist": shortlist,
            "candidates": [p.arxiv_id for p in pool],
            "shortlisted": [p.arxiv_id for p in shortlist],
            "problems": [problem("ranking the candidates", error.error)],
        },
        goto="prewarm",
    )


def prewarm_failed(state: DiscoverState, error: NodeError) -> Command[Any]:
    """The cache write is an optimisation: screening goes on without it."""
    return Command(update={"problems": []}, goto=to_screen(state))


def after_researcher(state: DiscoverState) -> str:
    calls = state["transcript"][-1].calls
    return "arxiv_tools" if calls and state.get("tool_calls", 0) < MAX_TOOL_CALLS else "prerank"


def after_tools(state: DiscoverState) -> str:
    """Once the budget is spent, or arXiv has failed twice in a row, asking the researcher again
    would only cost a call; one failed round is tried again, since a single refusal often passes
    (S7's f-arxiv-blip)."""
    failing = state.get("failed_rounds", 0) >= MAX_FAILED_ROUNDS
    more = state["tool_calls"] < MAX_TOOL_CALLS and not failing
    return "researcher" if more else "prerank"


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
    graph.add_node(
        "researcher",
        researcher,
        retry_policy=RETRY,
        timeout=LLM_TIMEOUT_S,
        error_handler=search_stopped,
    )
    graph.add_node(
        "arxiv_tools",
        arxiv_tools,
        retry_policy=RETRY,
        timeout=SLOT_BACKSTOP_S,
        error_handler=search_stopped,
    )
    graph.add_node(
        "prerank", prerank, retry_policy=RETRY, timeout=LLM_TIMEOUT_S, error_handler=prerank_failed
    )
    graph.add_node(
        "prewarm", prewarm, retry_policy=RETRY, timeout=LLM_TIMEOUT_S, error_handler=prewarm_failed
    )
    graph.add_node(
        "screen", screen, input_schema=ScreenTask, retry_policy=RETRY, timeout=LLM_TIMEOUT_S
    )
    graph.add_node("rank", rank)
    graph.add_edge(START, "researcher")
    graph.add_conditional_edges("researcher", after_researcher, ["arxiv_tools", "prerank"])
    graph.add_conditional_edges("arxiv_tools", after_tools, ["researcher", "prerank"])
    graph.add_edge("prerank", "prewarm")
    graph.add_conditional_edges("prewarm", to_screen, ["screen", "rank"])
    graph.add_edge("screen", "rank")
    graph.add_edge("rank", END)
    return graph.compile(name="discover")
