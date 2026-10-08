"""The top-level conversation graph (DESIGN §4.1). One thread is one conversation, checkpointed in
Postgres, so a turn that stops for the user (clarify, paper choice) resumes where it stopped.

    START → load_context → understand ─┬─► clarify ⏸ → understand
                                       ├─► discover ⟦sub⟧ ─┬─► choose_papers ⏸? → read …
                                       │                   └─► respond
                                       ├─► resolve → read ⟦sub⟧ → answer ⟦sub⟧ → respond
                                       └─► respond → END

choose_papers stops for the user only when the choice is ambiguous. Memory (remember, the library,
the user profile) arrives in M4; until then a library question is answered with a notice.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from ara.arxiv.client import ARXIV_ID
from ara.db.pool import Pool
from ara.graph import answer, discover, read
from ara.graph.state import (
    MAX_READ,
    Answer,
    Context,
    Evidence,
    PaperCard,
    ResearchRequest,
    plain,
)
from ara.llm.prompt import Block, Instructions, Prompt, data
from ara.llm.stages import STAGES

UNDERSTAND = Instructions.load("understand")
MAX_CLARIFY = 2  # questions per turn; after that the turn proceeds with what it has
RECURSION_LIMIT = 60  # steps per turn, subgraphs included (≤ 8 researcher rounds dominate)
READ, ANSWER, DISCOVER = read.build(), answer.build(), discover.build()
OTHER = (
    "I find arXiv papers and answer questions from their full text. Ask me for papers on a topic,"
    " or to read a paper by its arXiv id."
)
LIBRARY = "Questions about papers from earlier sessions need memory, which is not built yet."

type Status = Literal["running", "needs_input", "complete", "partial", "failed"]


class ConversationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    shown: list[PaperCard]  # the papers listed last, which "the second one" refers to
    request: ResearchRequest | None  # turn-scoped from here on
    clarifications: int
    papers: list[PaperCard]
    selected: list[str]  # paper references to read
    evidence: list[Evidence]
    missing: list[str]
    answer: Answer | None
    status: Status
    stop_reason: str


def load_context(state: ConversationState) -> dict[str, object]:
    """Reset the turn-scoped fields, so nothing from the previous turn leaks into this one."""
    return {
        "request": None,
        "clarifications": 0,
        "papers": [],
        "selected": [],
        "evidence": [],
        "missing": [],
        "answer": None,
        "status": "running",
        "stop_reason": "",
    }


def understand_prompt(messages: list[AnyMessage], shown: list[PaperCard]) -> Prompt:
    """The conversation so far is the shared part; the numbered papers shown last (when there are
    any) and the latest message are the item. Each turn extends the previous turn's cached prefix
    (DESIGN §6.3: understand caches all three parts); after a turn that had papers to show, the
    hit ends before the previous latest message, because the papers block is not repeated."""
    blocks = tuple(_block(m) for m in messages)
    listing = [
        {"n": n, "id": f"{p.arxiv_id}v{p.version}", "title": p.title, "published": p.published}
        for n, p in enumerate(shown, 1)
    ]
    papers = (data("Papers shown last", listing),) if listing else ()
    return Prompt(UNDERSTAND, shared=blocks[:-1], item=(*papers, *blocks[-1:]))


def _block(message: AnyMessage) -> Block:
    return Block("user" if isinstance(message, HumanMessage) else "assistant", message.text)


async def understand(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    ctx = runtime.context
    shown = state.get("shown", [])
    prompt = understand_prompt(state["messages"], shown)
    request = await ctx.gateway.structured(
        STAGES["understand"], prompt, ResearchRequest, scope=ctx.scope
    )
    return {"request": checked(request, state["messages"], shown)}


def checked(
    request: ResearchRequest, messages: list[AnyMessage], shown: list[PaperCard]
) -> ResearchRequest:
    """The trust boundary for understand: a constraint must quote the user literally (v1 rule);
    ids must be arXiv ids written in the conversation, not recalled by the model (v1 rule, DESIGN
    §13); positions must point at a paper shown; dates must parse. A read request that names no
    paper becomes find-then-read."""
    said = plain(" ".join(m.text for m in messages if isinstance(m, HumanMessage)))
    written = " ".join(m.text for m in messages)
    paper_ids = [
        i for i in request.paper_ids if (m := ARXIV_ID.match(i)) and _written(m[1], written)
    ]
    listed = [n for n in request.listed if 1 <= n <= len(shown)]
    intent = request.intent
    if intent == "read" and not (paper_ids or listed):
        intent = "discover_read"
    return request.model_copy(
        update={
            "intent": intent,
            "paper_ids": paper_ids,
            "listed": listed,
            "constraints": [c for c in request.constraints if plain(c.quote) in said],
            "count": request.count if request.count and request.count > 0 else None,
            "published_after": _iso(request.published_after),
            "published_before": _iso(request.published_before),
        }
    )


def _written(arxiv_id: str, text: str) -> bool:
    """The id appears whole: "2305.1832" is not written by "2305.18323"."""
    return re.search(rf"(?<![\d.]){re.escape(arxiv_id)}(?!\d)", text) is not None


def _iso(value: str | None) -> str | None:
    try:
        return date.fromisoformat(value).isoformat() if value else None
    except ValueError:
        return None


def clarify(state: ConversationState) -> dict[str, object]:
    """Ask, and wait. On resume LangGraph re-runs this node from the start, so nothing happens
    before `interrupt`."""
    question = _request(state).clarification or ""
    reply = interrupt({"kind": "clarify", "question": question})
    return {
        "messages": [AIMessage(question), HumanMessage(str(reply))],
        "clarifications": state["clarifications"] + 1,
    }


async def run_discover(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    found = await DISCOVER.ainvoke(
        {"request": _request(state)},
        {"recursion_limit": RECURSION_LIMIT},
        context=runtime.context,
    )
    return {"papers": found["papers"]}


def choose_papers(state: ConversationState) -> dict[str, object]:
    """Deterministic when the user gave a count or one to three papers are direct matches;
    otherwise the user picks (resume value: 1-based positions in the list)."""
    papers, count = state["papers"], _request(state).count
    direct = [p for p in papers if p.relevance == 3]
    if count:
        chosen = papers[: min(count, MAX_READ)]
    elif 1 <= len(direct) <= MAX_READ:
        chosen = direct
    else:
        listing = [{"n": n, "id": p.arxiv_id, "title": p.title} for n, p in enumerate(papers, 1)]
        picked = interrupt({"kind": "choose_papers", "papers": listing, "max": MAX_READ})
        positions = [n for n in _positions(picked) if 1 <= n <= len(papers)][:MAX_READ]
        chosen = [papers[n - 1] for n in positions]
    return {"selected": [p.reference for p in chosen]}


def _positions(value: Any) -> list[int]:
    """The user's choice, from the UI (a list of numbers) or typed text ("1 and 3")."""
    if isinstance(value, list):
        return [int(n) for n in value if isinstance(n, int)]
    return [int(n) for n in re.findall(r"\d+", str(value))]


def resolve(state: ConversationState) -> dict[str, object]:
    """For a read request: the named ids, then the shown papers it points at, one per paper and as
    stored paper ids ("arxiv:2305.18323v1"); a versioned id wins over a bare one, which gets its
    latest version when the read graph ingests it."""
    request, shown = _request(state), state.get("shown", [])
    ids = [
        *request.paper_ids,
        *(f"{shown[n - 1].arxiv_id}v{shown[n - 1].version}" for n in request.listed),
    ]
    by_paper: dict[str, str] = {}
    for arxiv_id in ids:  # checked ids: "2305.18323" or "2305.18323v1"
        paper = arxiv_id.split("v")[0]
        if paper not in by_paper or ("v" in arxiv_id and "v" not in by_paper[paper]):
            by_paper[paper] = arxiv_id
    return {"selected": [f"arxiv:{i}" for i in by_paper.values()][:MAX_READ]}


async def run_read(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    found = await READ.ainvoke(
        {"question": _question(state), "papers": state["selected"]}, context=runtime.context
    )
    return {"evidence": found["evidence"], "missing": found["missing"]}


async def run_answer(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    result = await ANSWER.ainvoke(
        {"question": _question(state), "evidence": state["evidence"], "missing": state["missing"]},
        context=runtime.context,
    )
    return {"answer": result["answer"]}


def respond(state: ConversationState) -> dict[str, object]:
    papers = state["papers"]
    return {
        "messages": [AIMessage(reply(state))],
        "shown": papers or state.get("shown", []),
        "status": "complete",
    }


def reply(state: ConversationState) -> str:
    request = _request(state)
    if request.intent == "other":
        return OTHER
    if request.intent == "library":
        return LIBRARY
    parts = []
    if state["papers"]:
        parts.append("\n".join(_listing(n, p) for n, p in enumerate(state["papers"], 1)))
    elif request.intent in ("discover", "discover_read"):
        parts.append("I found no arXiv papers that match this request.")
    if (delivered := state["answer"]) is not None:
        sources = sorted({e.paper_id for e in delivered.evidence})
        parts.append(delivered.render() + (f"\nSources: {', '.join(sources)}" if sources else ""))
    elif request.intent == "discover_read" and state["papers"] and not state["selected"]:
        parts.append("Nothing was chosen to read.")
    return "\n\n".join(parts)


def _listing(n: int, p: PaperCard) -> str:
    return f"{n}. {p.title} (arXiv {p.arxiv_id}v{p.version}, {p.published}) — {p.reason}"


def _request(state: ConversationState) -> ResearchRequest:
    request = state["request"]
    if request is None:
        raise RuntimeError("understand has not run in this turn")
    return request


def _question(state: ConversationState) -> str:
    request = _request(state)
    return request.question or request.need


def after_understand(state: ConversationState) -> str:
    request = _request(state)
    if request.clarification and state["clarifications"] < MAX_CLARIFY:
        return "clarify"
    return {
        "discover": "discover",
        "discover_read": "discover",
        "read": "resolve",
        "library": "respond",
        "other": "respond",
    }[request.intent]


def after_discover(state: ConversationState) -> str:
    wants_reading = _request(state).intent == "discover_read"
    return "choose_papers" if wants_reading and state["papers"] else "respond"


def after_choice(state: ConversationState) -> str:
    return "read" if state["selected"] else "respond"


def build(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[ConversationState, Context, ConversationState, ConversationState]:
    graph = StateGraph(ConversationState, context_schema=Context)
    graph.add_node("load_context", load_context)
    graph.add_node("understand", understand)
    graph.add_node("clarify", clarify)
    graph.add_node("discover", run_discover)
    graph.add_node("choose_papers", choose_papers)
    graph.add_node("resolve", resolve)
    graph.add_node("read", run_read)
    graph.add_node("answer", run_answer)
    graph.add_node("respond", respond)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "understand")
    graph.add_conditional_edges(
        "understand", after_understand, ["clarify", "discover", "resolve", "respond"]
    )
    graph.add_edge("clarify", "understand")
    graph.add_conditional_edges("discover", after_discover, ["choose_papers", "respond"])
    graph.add_conditional_edges("choose_papers", after_choice, ["read", "respond"])
    graph.add_edge("resolve", "read")
    graph.add_edge("read", "answer")
    graph.add_edge("answer", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer, name="ara")


async def checkpointer(pool: Pool) -> AsyncPostgresSaver:
    """Checkpoints live in the app database; `setup` creates or upgrades LangGraph's own tables."""
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver


@dataclass(frozen=True)
class Turn:
    """What one call to `send` produced: a reply, or a question the graph is waiting on."""

    reply: str | None
    waiting: dict[str, Any] | None
    state: dict[str, Any]


async def send(
    app: CompiledStateGraph[ConversationState, Context, ConversationState, ConversationState],
    thread_id: str,
    context: Context,
    *,
    message: str | None = None,
    resume: Any = None,
) -> Turn:
    """Start a turn with a user message, or resume one that is waiting on the user."""
    config: Any = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    payload: Any = (
        Command(resume=resume) if message is None else {"messages": [HumanMessage(message)]}
    )
    state = await app.ainvoke(payload, config, context=context)
    if waiting := state.get("__interrupt__"):
        return Turn(None, waiting[0].value, state)
    return Turn(state["messages"][-1].text, None, state)
