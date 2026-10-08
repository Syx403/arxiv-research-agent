"""The top-level conversation graph (DESIGN §4.1). One thread is one conversation, checkpointed in
Postgres, so a turn that stops for the user (clarify, paper choice) resumes where it stopped.

    START → load_context → understand ─┬─► clarify ⏸ → understand
                                       ├─► discover ⟦sub⟧ ─┬─► choose_papers ⏸? → read …
                                       │                   └─► respond
                                       ├─► resolve → read ⟦sub⟧ → answer ⟦sub⟧ → respond
                                       ├─► library → read ⟦sub⟧ → answer ⟦sub⟧ → respond
                                       ├─► remember → respond → END          (intent "memory")
                                       └─► respond
    respond → remember → END                                               (every other turn)

choose_papers stops for the user only when the choice is ambiguous. Memory (D27): load_context
reads the user's profile and the past research closest to the message from the Store; remember
keeps what the user says about themselves and records the turn; reading adds to the library.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt

from ara.arxiv.client import ARXIV_ID
from ara.db.pool import Connection, Pool
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
from ara.memory import extract, library, store
from ara.memory.extract import MemoryUpdate
from ara.memory.store import Episode, Fact
from ara.rag.embed import embed

UNDERSTAND = Instructions.load("understand")
MAX_CLARIFY = 2  # questions per turn; after that the turn proceeds with what it has
RECURSION_LIMIT = 60  # steps per turn, subgraphs included (≤ 8 researcher rounds dominate)
READ, ANSWER, DISCOVER = read.build(), answer.build(), discover.build()
OTHER = (
    "I find arXiv papers and answer questions from their full text. Ask me for papers on a topic,"
    " or to read a paper by its arXiv id."
)
NOT_DISCUSSED = (
    "We have not discussed this in any paper we have read."
    " Would you like me to search arXiv for papers on it?"
)
RESEARCH = ("discover", "discover_read", "read", "library")
# Our types kept in checkpoints; LangGraph will refuse to load types it was not told about.
CHECKPOINTED = [
    ("ara.graph.state", name)
    for name in (
        "Answer",
        "Claim",
        "Constraint",
        "Evidence",
        "PaperCard",
        "Priority",
        "ResearchRequest",
        "Verdict",
    )
] + [
    ("ara.memory.store", "Episode"),
    ("ara.memory.store", "Fact"),
    ("ara.memory.extract", "MemoryUpdate"),
]

type Status = Literal["running", "needs_input", "complete", "partial", "failed"]


class ConversationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    shown: list[PaperCard]  # the papers listed last, which "the second one" refers to
    turn_start: int  # index of this turn's first message; turn-scoped from here on
    profile: list[Fact]  # what the user told us in any session
    episodes: list[Episode]  # earlier research closest to the latest message
    memory: MemoryUpdate | None  # what remember changed this turn
    request: ResearchRequest | None
    clarifications: int
    papers: list[PaperCard]  # listed by discovery
    selected: list[str]  # paper references to read
    read: list[PaperCard]  # the papers read, in the order selected
    evidence: list[Evidence]
    missing: list[str]
    answer: Answer | None
    status: Status
    stop_reason: str


def fresh_turn() -> dict[str, object]:
    """The turn-scoped fields, reset so nothing from the previous turn leaks into this one."""
    return {
        "turn_start": 0,
        "profile": [],
        "episodes": [],
        "memory": None,
        "request": None,
        "clarifications": 0,
        "papers": [],
        "selected": [],
        "read": [],
        "evidence": [],
        "missing": [],
        "answer": None,
        "status": "running",
        "stop_reason": "",
    }


async def load_context(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    """Start a turn: fresh turn-scoped fields, and the user's memory from the Store, which is the
    profile and the earlier research closest to the new message (D27)."""
    user, memory = runtime.context.user_id, runtime.store
    if memory is None:  # a graph compiled without a Store (unit tests) has no memory
        return {**fresh_turn(), "turn_start": len(state["messages"]) - 1}
    return {
        **fresh_turn(),
        "turn_start": len(state["messages"]) - 1,
        "profile": await store.profile(memory, user),
        "episodes": await store.recall(memory, user, state["messages"][-1].text),
    }


def understand_prompt(
    messages: list[AnyMessage],
    shown: list[PaperCard],
    today: date,
    profile: list[Fact] = [],  # noqa: B006 (never mutated)
    episodes: list[Episode] = [],  # noqa: B006
) -> Prompt:
    """The user's profile and the conversation so far are the shared part; today's date, earlier
    research related to the message, the numbered papers shown last and the latest message are
    the item. Each turn extends the previous turn's cached prefix, which ends just before the
    previous turn's item (DESIGN §6.3; the date is never placed before the item, D24); the
    profile changes rarely, and only then breaks the prefix (D27)."""
    blocks = tuple(_block(m) for m in messages)
    known = (data("About the user", [f.model_dump(exclude={"key"}) for f in profile]),)
    listing = [
        {"n": n, "id": f"{p.arxiv_id}v{p.version}", "title": p.title, "published": p.published}
        for n, p in enumerate(shown, 1)
    ]
    papers = (data("Papers shown last", listing),) if listing else ()
    earlier = (Block("user", "Earlier research:\n" + "\n".join(e.text() for e in episodes)),)
    when = Block("user", f"Today: {today.isoformat()}")
    return Prompt(
        UNDERSTAND,
        shared=(*(known if profile else ()), *blocks[:-1]),
        item=(when, *(earlier if episodes else ()), *papers, *blocks[-1:]),
    )


def _block(message: AnyMessage) -> Block:
    return Block("user" if isinstance(message, HumanMessage) else "assistant", message.text)


async def understand(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    ctx = runtime.context
    shown, profile, episodes = state.get("shown", []), state["profile"], state["episodes"]
    prompt = understand_prompt(state["messages"], shown, ctx.today, profile, episodes)
    request = await ctx.gateway.structured(
        STAGES["understand"], prompt, ResearchRequest, scope=ctx.scope
    )
    return {"request": checked(request, state["messages"], shown, profile, episodes)}


def checked(
    request: ResearchRequest,
    messages: list[AnyMessage],
    shown: list[PaperCard],
    profile: list[Fact] = [],  # noqa: B006 (never mutated)
    episodes: list[Episode] = [],  # noqa: B006
) -> ResearchRequest:
    """The trust boundary for understand: a constraint or a priority must quote the user literally,
    in this conversation or in a remembered fact (v1 rule, D27); ids must be arXiv ids written in
    the conversation or in the user's earlier research, not recalled by the model (v1 rule, DESIGN
    §13); positions must point at a paper shown; dates must parse; titles are kept once each. A
    read request that names no paper becomes find-then-read."""
    human = [m.text for m in messages if isinstance(m, HumanMessage)]
    said = plain(" ".join([*human, *(f.quote for f in profile)]))
    written = " ".join([*(m.text for m in messages), *(e.text() for e in episodes)])
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
            "priorities": [p for p in request.priorities if plain(p.quote) in said],
            "titles": list(dict.fromkeys(t.strip() for t in request.titles if t.strip())),
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
    ctx = runtime.context
    found = await READ.ainvoke(
        {"question": _question(state), "papers": state["selected"]}, context=ctx
    )
    async with ctx.pool.connection() as conn:
        cards = await read_cards(conn, found["documents"])
        await library.record_read(conn, ctx.user_id, found["documents"])
    order = [r.removeprefix("arxiv:").split("v")[0] for r in state["selected"]]
    cards.sort(key=lambda c: order.index(c.arxiv_id) if c.arxiv_id in order else len(order))
    return {"evidence": found["evidence"], "missing": found["missing"], "read": cards}


async def read_cards(conn: Connection, documents: list[int]) -> list[PaperCard]:
    """The papers behind the documents just read, so a follow-up can point at them by number."""
    cursor = await conn.execute(
        "SELECT p.arxiv_id, p.version, p.title, p.abstract FROM documents d"
        " JOIN papers p ON p.id = d.paper_id WHERE d.id = ANY(%s) AND p.source = 'arxiv'",
        (documents,),
    )
    return [
        PaperCard(
            arxiv_id=row["arxiv_id"],
            version=row["version"],
            title=row["title"],
            abstract=row["abstract"],
            published="",  # not stored with the paper; the listing shows it only after a search
        )
        for row in await cursor.fetchall()
    ]


async def run_answer(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    result = await ANSWER.ainvoke(
        {
            "question": _question(state),
            "evidence": state["evidence"],
            "missing": state["missing"],
            "priorities": [p.meaning for p in _request(state).priorities],
        },
        context=runtime.context,
    )
    return {"answer": result["answer"]}


async def find_in_library(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    """A question about papers read before: the user's papers nearest the question (HNSW over the
    library, D27) are read again, from the stored text, and answered with verified citations."""
    ctx = runtime.context
    [vector] = await embed([_question(state)], pool=ctx.pool, gateway=ctx.gateway, scope=ctx.scope)
    async with ctx.pool.connection() as conn:
        papers = await library.closest_papers(conn, ctx.user_id, vector, MAX_READ)
    return {"selected": papers}


async def remember(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    """Keep what the user said about themselves (one Luna call, only when the turn is about memory
    or stated constraints or priorities), then record a research turn as an episode (by code)."""
    ctx, memory = runtime.context, runtime.store
    if memory is None:
        return {}
    request = _request(state)
    update = MemoryUpdate(facts=[], forget=[])
    if request.intent == "memory" or request.constraints or request.priorities:
        said = [
            m.text for m in state["messages"][state["turn_start"] :] if isinstance(m, HumanMessage)
        ]
        update = await extract.propose(ctx.gateway, state["profile"], said, scope=ctx.scope)
        for fact in update.facts:
            await store.remember(memory, ctx.user_id, fact)
        for key in update.forget:
            await store.forget(memory, ctx.user_id, key)
    found = request.intent != "library" or _answered(state)  # a miss links no paper to the topic
    if request.intent in RESEARCH and found and (papers := state["read"] or state["papers"]):
        await store.record(memory, ctx.user_id, uuid4().hex, episode(state, ctx.today, papers))
    return {"memory": update}


def episode(state: ConversationState, today: date, papers: list[PaperCard]) -> Episode:
    delivered = state["answer"]
    return Episode(
        day=today.isoformat(),
        need=_request(state).need,
        papers=[f"{p.arxiv_id}v{p.version} {p.title}" for p in papers],
        answer=delivered.short if delivered and not delivered.abstained else "",
    )


def respond(state: ConversationState) -> dict[str, object]:
    """The papers a follow-up can point at: those a search listed, else those just read (D24)."""
    shown = state["papers"] or state["read"] or state.get("shown", [])
    return {"messages": [AIMessage(reply(state))], "shown": shown, "status": "complete"}


def reply(state: ConversationState) -> str:
    request = _request(state)
    if request.intent == "other":
        return OTHER
    if request.intent == "memory":
        return memory_reply(state)
    if request.intent == "library" and not _answered(state):
        return NOT_DISCUSSED  # nothing read covers it: offer a search instead (D27, Ewan)
    parts = []
    if request.intent == "library":
        parts.append(
            "From your library:\n" + "\n".join(_title(n, p) for n, p in enumerate(state["read"], 1))
        )
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


def _answered(state: ConversationState) -> bool:
    delivered = state["answer"]
    return delivered is not None and not delivered.abstained


def memory_reply(state: ConversationState) -> str:
    update = state["memory"] or MemoryUpdate(facts=[], forget=[])
    known = {f.key: f.statement for f in state["profile"]}
    lines = [f"Noted: {f.statement}" for f in update.facts]
    lines += [f"Forgotten: {known[k]}" for k in update.forget]
    return "\n".join(lines) or "There was nothing to remember or forget in that message."


def _title(n: int, p: PaperCard) -> str:
    return f"{n}. {p.title} (arXiv {p.arxiv_id}v{p.version})"


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
    # A library question is never clarified: the library search answers it or finds nothing (D27).
    asks = request.clarification and request.intent != "library"
    if asks and state["clarifications"] < MAX_CLARIFY:
        return "clarify"
    return {
        "discover": "discover",
        "discover_read": "discover",
        "read": "resolve",
        "library": "library",
        "memory": "remember",
        "other": "respond",
    }[request.intent]


def after_library(state: ConversationState) -> str:
    return "read" if state["selected"] else "respond"


def after_respond(state: ConversationState) -> str:
    """A memory turn has already remembered; every other turn remembers after replying."""
    return END if _request(state).intent == "memory" else "remember"


def after_remember(state: ConversationState) -> str:
    return "respond" if _request(state).intent == "memory" else END


def after_discover(state: ConversationState) -> str:
    wants_reading = _request(state).intent == "discover_read"
    return "choose_papers" if wants_reading and state["papers"] else "respond"


def after_choice(state: ConversationState) -> str:
    return "read" if state["selected"] else "respond"


def build(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    memory: BaseStore | None = None,
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
    graph.add_node("library", find_in_library)
    graph.add_node("respond", respond)
    graph.add_node("remember", remember)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "understand")
    graph.add_conditional_edges(
        "understand",
        after_understand,
        ["clarify", "discover", "resolve", "library", "remember", "respond"],
    )
    graph.add_edge("clarify", "understand")
    graph.add_conditional_edges("discover", after_discover, ["choose_papers", "respond"])
    graph.add_conditional_edges("choose_papers", after_choice, ["read", "respond"])
    graph.add_edge("resolve", "read")
    graph.add_edge("read", "answer")
    graph.add_conditional_edges("library", after_library, ["read", "respond"])
    graph.add_edge("answer", "respond")
    graph.add_conditional_edges("respond", after_respond, ["remember", END])
    graph.add_conditional_edges("remember", after_remember, ["respond", END])
    return graph.compile(checkpointer=checkpointer, store=memory, name="ara")


async def checkpointer(pool: Pool) -> AsyncPostgresSaver:
    """Checkpoints live in the app database; `setup` creates or upgrades LangGraph's own tables."""
    saver = AsyncPostgresSaver(pool, serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINTED))
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
