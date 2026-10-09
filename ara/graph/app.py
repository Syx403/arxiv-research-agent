"""The top-level conversation graph (DESIGN §4.1). One thread is one conversation, checkpointed in
Postgres, so a turn that stops for the user (clarify) resumes where it stopped.

    START → load_context → understand ─┬─► clarify ⏸ → understand
                                       ├─► discover ⟦sub⟧ ─┬─► choose_papers → read …
                                       │                   └─► respond
                                       ├─► resolve ─┬─► read ⟦sub⟧ → conflicts → answer ⟦sub⟧ …
                                       │            └─► discover (titles not found by title)
                                       ├─► library ─┬─► read …
                                       │            └─► respond (nothing relevant was read)
                                       ├─► remember → respond → END          (intent "memory")
                                       └─► respond
    answer ─┬─► respond
            └─► search_instead → discover … (a library question its papers do not answer, once)
    respond → remember → END                                               (every other turn)

arXiv search is the main path; the user's library is a quick look first (D30): discovery adds the
user's papers close to the need to its candidates, and a question that refers back to papers read
before reads the relevant ones, or searches arXiv when they do not answer it. Choosing papers never
asks the user (D30). Papers the user names (ids, numbers shown, titles) are the papers read,
whatever else the request says (D28, D29): resolve finds them, conflicts tells the user when one
breaks a constraint, and the reply says when one does not discuss the question. Memory (D27):
load_context reads the user's profile and the past research closest to the message from the
Store; remember keeps what the user says about themselves and records the turn; reading adds to
the library.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.config import get_config
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
    MAX_LISTED,
    MAX_READ,
    Answer,
    Context,
    Evidence,
    PaperCard,
    ResearchRequest,
    plain,
    quoted,
)
from ara.llm.prompt import Block, Instructions, Prompt, data
from ara.llm.stages import STAGES
from ara.memory import extract, library, store
from ara.memory.extract import MemoryUpdate
from ara.memory.store import Episode, Fact, Remembered
from ara.rag.chunking import sentence_starts
from ara.rag.embed import embed
from ara.rag.search import rrf

UNDERSTAND = Instructions.load("understand")
MAX_CLARIFY = 2  # questions per turn; after that the turn proceeds with what it has
RECURSION_LIMIT = 60  # steps per turn, subgraphs included (≤ 8 researcher rounds dominate)
LIBRARY_RANKED = 10  # papers ranked by passage distance before fusion with research records
LIBRARY_SCREENED = 8  # library papers a library question screens: one screen batch (D30)
LIBRARY_LOOKED_UP = 5  # library papers discovery adds to its candidates (D30)
PASSAGES_SHOWN = 2  # full-text passages per library paper in its screen (D30)
OPENING = 300  # characters of an abstract quoted when a named paper does not fit the question
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
# Our types kept in checkpoints, subgraph states included: the serializer loads any other type as
# a plain dict. A unit test walks every state schema to keep this list complete (D29).
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
    ("ara.graph.discover", "Judgement"),
    ("ara.graph.read", "Found"),
    ("ara.llm.prompt", "Block"),
    ("ara.llm.prompt", "ToolCall"),
    ("ara.memory.extract", "MemoryUpdate"),
    ("ara.memory.store", "Episode"),
    ("ara.memory.store", "Fact"),
    ("ara.memory.store", "Remembered"),
    ("ara.rag.retrieve", "Passage"),
]

type Status = Literal["running", "needs_input", "complete", "partial", "failed"]


class ConversationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    shown: list[PaperCard]  # the papers listed last, which "the second one" refers to
    turn_start: int  # index of this turn's first message; turn-scoped from here on
    profile: list[Remembered]  # what the user told us in any session
    episodes: list[Episode]  # earlier research closest to the latest message
    memory: MemoryUpdate | None  # what remember changed this turn
    request: ResearchRequest | None
    clarifications: int
    identified: list[str]  # titles resolve found (library or arXiv title search)
    notes: list[str]  # what resolve learned that the reply must say
    earlier: list[PaperCard]  # library papers read that did not answer: arXiv was searched (D30)
    gaps: list[str]  # what evidence selection found those papers do not cover (D30)
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
        "identified": [],
        "notes": [],
        "earlier": [],
        "gaps": [],
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
    profile: Sequence[Fact] = (),
    episodes: Sequence[Episode] = (),
) -> Prompt:
    """The user's profile and the conversation so far are the shared part; today's date, earlier
    research related to the message, the numbered papers shown last and the latest message are
    the item. Each turn extends the previous turn's cached prefix, which ends just before the
    previous turn's item (DESIGN §6.3; the date is never placed before the item, D24); the
    profile changes rarely, and only then breaks the prefix (D27)."""
    blocks = tuple(_block(m) for m in messages)
    facts = [{"quote": f.quote, "statement": f.statement} for f in profile]
    known = (data("About the user", facts),)
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
    profile: Sequence[Fact] = (),
    episodes: Sequence[Episode] = (),
) -> ResearchRequest:
    """The trust boundary for understand: a constraint or a priority must quote the user literally
    (a non-empty quote), in this conversation or in a remembered fact (v1 rule, D27); ids must be
    arXiv ids written in the conversation or in the user's earlier research, not recalled by the
    model (v1 rule, DESIGN §13); positions must point at a paper shown; dates must parse; titles
    are kept once each. A request to read that names papers (ids, numbers shown or titles) reads
    exactly those (D28, D29); one that names none becomes find-then-read."""
    human = [m.text for m in messages if isinstance(m, HumanMessage)]
    said = plain(" ".join([*human, *(f.quote for f in profile)]))
    written = " ".join([*(m.text for m in messages), *(e.text() for e in episodes)])
    paper_ids = [
        i for i in request.paper_ids if (m := ARXIV_ID.match(i)) and _written(m[1], written)
    ]
    listed = [n for n in request.listed if 1 <= n <= len(shown)]
    titles = list(dict.fromkeys(t.strip() for t in request.titles if t.strip()))
    named, intent = bool(paper_ids or listed or titles), request.intent
    if intent == "read" and not named:
        intent = "discover_read"
    elif intent == "discover_read" and named:
        intent = "read"
    return request.model_copy(
        update={
            "intent": intent,
            "paper_ids": paper_ids,
            "listed": listed,
            "constraints": [c for c in request.constraints if quoted(c.quote, said)],
            "priorities": [p for p in request.priorities if quoted(p.quote, said)],
            "titles": titles,
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


def names(request: ResearchRequest) -> bool:
    """The user named the papers (ids, numbers shown, titles): those are the papers read (D29)."""
    return bool(request.paper_ids or request.listed or request.titles)


async def run_discover(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    """Search arXiv; after resolve, only for the titles it could not find by title. The user's
    papers close to the need, within the request's dates, join the candidates on equal terms, and
    a listed paper the user has read is marked (D30). Papers a library question just read without
    an answer are not offered again."""
    ctx, request = runtime.context, _request(state)
    titles = [t for t in request.titles if t not in state["identified"]]
    tried = {p.arxiv_id for p in state["earlier"]}
    nearby, owned = await library_candidates(
        ctx, request.need, state["episodes"], LIBRARY_LOOKED_UP
    )
    found = await DISCOVER.ainvoke(
        {
            "request": request.model_copy(update={"titles": titles}),
            "library": [p for p in nearby if p.arxiv_id not in tried and _within(request, p, ctx)],
        },
        {"recursion_limit": RECURSION_LIMIT},
        context=ctx,
    )
    return {"papers": marked(found["papers"], owned)}


def marked(papers: list[PaperCard], owned: set[str]) -> list[PaperCard]:
    """A listed paper the user has read is marked "read before" (D30)."""
    return [p.model_copy(update={"read_before": p.arxiv_id in owned}) for p in papers]


def _within(request: ResearchRequest, paper: PaperCard, ctx: Context) -> bool:
    """A library paper in the request's date window, which ends today (as arXiv searches do)."""
    if not paper.published:  # stored before migration 0005 and not backfilled: cannot be placed
        return False
    after, before = request.published_after or "", request.published_before or ctx.today.isoformat()
    return after <= paper.published <= before


async def library_candidates(
    ctx: Context, query: str, episodes: Sequence[Episode], limit: int
) -> tuple[list[PaperCard], set[str]]:
    """The user's papers closest to `query`: passage distance (HNSW) and the papers read in the
    research records closest to the message, fused by reciprocal rank (D29, D30); with the bare
    ids of every paper in the library. No request when the library is empty."""
    async with ctx.pool.connection() as conn:
        owned = await library.papers(conn, ctx.user_id)
    if not owned:
        return [], set()
    [vector] = await embed([query], pool=ctx.pool, gateway=ctx.gateway, scope=ctx.scope)
    async with ctx.pool.connection() as conn:
        by_text = await library.closest_papers(conn, ctx.user_id, vector, LIBRARY_RANKED)
    by_reference = {p.reference: p for p in owned}
    by_record = [f"arxiv:{i}" for e in episodes for i in e.read_ids()]
    by_record = [r for r in dict.fromkeys(by_record) if r in by_reference]
    fused = [by_reference[r] for r in rrf([by_text, by_record]) if r in by_reference]
    return fused[:limit], {p.arxiv_id for p in owned}


def choose_papers(state: ConversationState) -> dict[str, object]:
    """Never asks (D30). The papers the user named, when they named any: those resolve found and
    those discovery found by title (D28, D29). Otherwise the first of the list, as many as the
    user asked for; without a count, the direct matches (relevance 3), else the top of the list;
    at most MAX_READ."""
    papers, request = state["papers"], _request(state)
    if names(request):
        named = [*state["selected"], *(p.reference for p in papers if p.named)]
        return {"selected": _distinct(named)[:MAX_READ]}
    direct = [p for p in papers if p.relevance == 3]
    chosen = papers[: request.count] if request.count else direct or papers
    return {"selected": [p.reference for p in chosen[:MAX_READ]]}


async def resolve(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    """The papers the user named, as stored paper ids ("arxiv:2305.18323v1"), whatever else the
    request says (D28, D29): ids, then the shown papers it points at, then titles, matched by code
    against the user's library first and an arXiv title search second (no model). Titles neither
    finds are left to discovery. For a library question, a named paper the user has not read is
    read from arXiv, and the reply says so."""
    ctx, request, shown = runtime.context, _request(state), state.get("shown", [])
    async with ctx.pool.connection() as conn:
        owned = await library.papers(conn, ctx.user_id)
    ids = [
        *request.paper_ids,
        *(f"{shown[n - 1].arxiv_id}v{shown[n - 1].version}" for n in request.listed),
    ]
    titled: dict[str, str] = {}  # bare id → title, for the notes
    identified: list[str] = []
    notes: list[str] = []
    for name in request.titles:
        found = [p for p in owned if discover.titled(name, p.title)]
        found = found or await discover.find_title(name, ctx)
        if not found:
            continue
        paper, identified = found[0], [*identified, name]
        ids.append(f"{paper.arxiv_id}v{paper.version}")
        titled[paper.arxiv_id] = paper.title
        if len(found) > 1:
            notes.append(
                f'I took "{name}" to be "{paper.title}" (arXiv {paper.arxiv_id}v{paper.version});'
                f" {len(found) - 1} other paper(s) have a title starting with it."
            )
    selected = _distinct([f"arxiv:{i}" for i in ids])
    if len(selected) > MAX_READ:
        notes.append(f"I read at most {MAX_READ} papers per turn: the first {MAX_READ} named.")
    if request.intent == "library":
        have = {p.arxiv_id for p in owned}
        notes += [
            f"We had not read {titled.get(b, f'arXiv {b}')} before; I read it from arXiv now."
            for b in map(_bare, selected[:MAX_READ])
            if b not in have
        ]
    return {"selected": selected[:MAX_READ], "identified": identified, "notes": notes}


def _bare(reference: str) -> str:
    """ "arxiv:2305.18323v1" or "2305.18323" → "2305.18323"."""
    return reference.removeprefix("arxiv:").split("v")[0]


def _distinct(references: list[str]) -> list[str]:
    """One reference per paper, in order; a versioned one wins over a bare one, which gets its
    latest version when the read graph ingests it."""
    by_paper: dict[str, str] = {}
    for reference in references:
        paper = _bare(reference)
        if paper not in by_paper or (_version(reference) and not _version(by_paper[paper])):
            by_paper[paper] = reference
    return list(by_paper.values())


def _version(reference: str) -> str:
    return reference.removeprefix("arxiv:").removeprefix(_bare(reference))


async def run_read(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    """Each paper is searched on its own (D30). A card discovery screened keeps what the screen
    said."""
    ctx = runtime.context
    question, papers = _question(state), state["selected"]
    found = await READ.ainvoke({"question": question, "papers": papers}, context=ctx)
    async with ctx.pool.connection() as conn:
        cards = await read_cards(conn, found["documents"])
        await library.record_read(conn, ctx.user_id, found["documents"])
    screened = {p.arxiv_id: p for p in state["papers"]}
    cards = [screened.get(c.arxiv_id, c) for c in cards]
    order = [_bare(r) for r in papers]
    cards.sort(key=lambda c: order.index(c.arxiv_id) if c.arxiv_id in order else len(order))
    return {"evidence": found["evidence"], "missing": found["missing"], "read": cards}


async def conflicts(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    """A paper the user named is read even when it breaks a stated constraint (D28); the screen
    judges the named papers discovery did not screen, so the reply can say so (D29). No call
    without constraints."""
    request = _request(state)
    unscreened = [c for c in state["read"] if c.relevance is None]
    if not (request.constraints and names(request) and unscreened):
        return {}
    judged = await discover.judge(request, unscreened, runtime.context)
    by_id = {c.arxiv_id: c for c in judged}
    return {"read": [by_id.get(c.arxiv_id, c) for c in state["read"]]}


async def read_cards(conn: Connection, documents: list[int]) -> list[PaperCard]:
    """The papers behind the documents just read, so a follow-up can point at them by number."""
    cursor = await conn.execute(
        "SELECT p.arxiv_id, p.version, p.title, p.abstract, p.published FROM documents d"
        " JOIN papers p ON p.id = d.paper_id WHERE d.id = ANY(%s) AND p.source = 'arxiv'",
        (documents,),
    )
    return [
        PaperCard(
            arxiv_id=row["arxiv_id"],
            version=row["version"],
            title=row["title"],
            abstract=row["abstract"],
            published=row["published"].isoformat() if row["published"] else "",
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
            "context": searched_instead_note(state["earlier"], state["gaps"])
            if state["earlier"]
            else "",
        },
        context=runtime.context,
    )
    return {"answer": result["answer"]}


async def find_in_library(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    """A question that refers back to papers read before and names none (D27, D30): the user's
    papers closest to it are screened, each with its abstract and the passages nearest the
    question (one call); the relevant ones are listed, by relevance, and the most relevant read."""
    ctx, question = runtime.context, _question(state)
    candidates, _ = await library_candidates(ctx, question, state["episodes"], LIBRARY_SCREENED)
    if not candidates:
        return {"papers": [], "selected": []}
    [vector] = await embed([question], pool=ctx.pool, gateway=ctx.gateway, scope=ctx.scope)
    async with ctx.pool.connection() as conn:
        nearest = await library.passages_near(
            conn, [p.reference for p in candidates], vector, PASSAGES_SHOWN
        )
    relevant = relevant_first(await discover.judge(_request(state), candidates, ctx, nearest))
    return {
        "papers": relevant[:MAX_LISTED],
        "selected": [p.reference for p in relevant[:MAX_READ]],
    }


def relevant_first(judged: list[PaperCard]) -> list[PaperCard]:
    """The papers the screen judged relevant (≥ 2), by relevance, ties in the candidates' order."""
    return sorted((p for p in judged if (p.relevance or 0) >= 2), key=lambda p: -(p.relevance or 0))


def search_instead(state: ConversationState) -> dict[str, object]:
    """The library papers read did not answer: search arXiv for the same question, once (D30)."""
    return {
        "earlier": state["read"],
        "gaps": state["missing"],
        "papers": [],
        "selected": [],
        "read": [],
        "evidence": [],
        "missing": [],
        "answer": None,
    }


async def remember(state: ConversationState, runtime: Runtime[Context]) -> dict[str, object]:
    """Keep what the user said about themselves, with its source turn (one Luna call, only when
    the turn is about memory or the user stated constraints or priorities in it, not when they
    came from the profile), then record a research turn as an episode (by code)."""
    ctx, memory = runtime.context, runtime.store
    if memory is None:
        return {}
    request = _request(state)
    said = [m.text for m in state["messages"][state["turn_start"] :] if isinstance(m, HumanMessage)]
    text = plain(" ".join(said))
    quotes = [*(c.quote for c in request.constraints), *(p.quote for p in request.priorities)]
    stated = any(quoted(q, text) for q in quotes)
    update = MemoryUpdate(facts=[], forget=[])
    if request.intent == "memory" or stated:
        update = await extract.propose(ctx.gateway, state["profile"], said, scope=ctx.scope)
        source = {
            "thread": get_config()["configurable"]["thread_id"],
            "turn": state["turn_start"],
            "day": ctx.today.isoformat(),
        }
        for fact in update.facts:
            await store.remember(memory, ctx.user_id, Remembered(**fact.model_dump(), **source))
        for key in update.forget:
            await store.forget(memory, ctx.user_id, key)
    if request.intent in RESEARCH and (record := episode(state, ctx.today)) is not None:
        await store.record(memory, ctx.user_id, uuid4().hex, record)
    return {"memory": update}


def episode(state: ConversationState, today: date) -> Episode | None:
    """The turn as research, or None when it links no paper to the need: a library question
    answered from the library keeps only the papers its answer cites (D27, D29)."""
    read = _cited(state) if _from_library(state) else state["read"]
    seen = {p.arxiv_id for p in read}
    listed = [p for p in state["papers"] if p.arxiv_id not in seen]
    if not (read or listed):
        return None
    delivered = state["answer"]
    return Episode(
        day=today.isoformat(),
        need=_request(state).need,
        read=[_entry(p) for p in read],
        listed=[_entry(p) for p in listed],
        answer=delivered.short if delivered and not delivered.abstained else "",
    )


def _entry(p: PaperCard) -> str:
    return f"{p.arxiv_id}v{p.version} {p.title}"


def respond(state: ConversationState) -> dict[str, object]:
    """The papers a follow-up can point at: those the reply lists, else those just read (D24)."""
    shown = state["papers"] or state["read"] or state.get("shown", [])
    return {"messages": [AIMessage(reply(state))], "shown": shown, "status": "complete"}


def reply(state: ConversationState) -> str:
    request = _request(state)
    if request.intent == "other":
        return OTHER
    if request.intent == "memory":
        return memory_reply(state)
    if _by_meaning(state) and not state["papers"] and not state["earlier"]:
        return NOT_DISCUSSED  # no shared history on this: offer a search instead (D27, D30)
    parts = list(state["notes"])
    if state["earlier"]:  # the model's own words when it wrote an answer, else the facts (D31)
        written = state["answer"].context if state["answer"] else ""
        parts.append(written or searched_instead_note(state["earlier"], state["gaps"]))
    parts += [
        _identity(p) for p in state["read"] if p.named and not discover.titled(p.named, p.title)
    ]
    parts += [_conflict(p) for p in state["read"] if p.violated]
    found = {*state["identified"], *(p.named for p in state["papers"])}
    if missing := [t for t in request.titles if t not in found]:
        parts.append(f"I could not find on arXiv: {'; '.join(missing)}.")
    if state["papers"]:
        listing = "\n".join(_listing(n, p) for n, p in enumerate(state["papers"], 1))
        parts.append(("From your library:\n" if _from_library(state) else "") + listing)
    elif request.intent in ("discover", "discover_read") or state["earlier"]:
        parts.append("I found no arXiv papers that match this request.")
    if (delivered := state["answer"]) is not None:
        parts.append("\n".join([delivered.render(), *map(_citation, delivered.evidence)]))
    if names(request):
        selected = {e.paper_id for e in state["evidence"]}
        parts += [_mismatch(p) for p in state["read"] if p.reference not in selected]
    return "\n\n".join(parts)


def searched_instead_note(earlier: list[PaperCard], gaps: list[str]) -> str:
    """Why arXiv was searched: the papers we read, and what evidence selection (the model reading
    their passages) found they leave uncovered. Synthesize rewords it as the reply's opening
    (D31); it is shown as is when no answer was written."""
    titles = " and ".join(f'"{p.title}"' for p in earlier)
    subject = "it does" if len(earlier) == 1 else "they do"
    missing = "; ".join(gaps) or "what you asked"
    return f"We read {titles} before, but {subject} not cover {missing}, so I searched arXiv:"


def _citation(e: Evidence) -> str:
    """One line per cited sentence: where it is in which paper (D30)."""
    source = (
        f"arXiv {e.paper_id.removeprefix('arxiv:')}"
        if e.paper_id.startswith("arxiv:")
        else e.paper_id
    )
    return f"[{e.id}] {e.heading_path} ({source})"


def _identity(p: PaperCard) -> str:
    """The screen, not the title, decided that this is the paper the user named (D28)."""
    return f'I took "{p.named}" to be "{p.title}" (arXiv {p.arxiv_id}v{p.version}).'


def _conflict(p: PaperCard) -> str:
    quotes = ", ".join(f'"{q}"' for q in p.violated)
    return (
        f'Note: "{p.title}" may break what you asked for ({quotes}), judging from its abstract;'
        " I read it because you named it."
    )


def _mismatch(p: PaperCard) -> str:
    """A named paper with no sentence on the question: say so, with the paper's own opening
    sentence from arXiv (stored text, not model text), so the user can see what it is about."""
    starts = sentence_starts(p.abstract)
    opening = p.abstract[: starts[1] if len(starts) > 1 else len(p.abstract)].strip()[:OPENING]
    return (
        f'"{p.title}" (arXiv {p.arxiv_id}v{p.version}) does not seem to discuss this: nothing in'
        f' its full text was found for the question. Its abstract begins: "{opening}" Check the'
        " title or id, or ask me to search arXiv for papers on it."
    )


def _answered(state: ConversationState) -> bool:
    delivered = state["answer"]
    return delivered is not None and not delivered.abstained


def _by_meaning(state: ConversationState) -> bool:
    """A library question that names no paper: the library search chooses what to read."""
    request = _request(state)
    return request.intent == "library" and not names(request)


def _from_library(state: ConversationState) -> bool:
    """A library question answered from the library, without the fallback arXiv search."""
    return _by_meaning(state) and not state["earlier"]


def _cited(state: ConversationState) -> list[PaperCard]:
    """The papers read that the delivered answer cites."""
    delivered = state["answer"]
    cited = {e.paper_id for e in delivered.evidence} if delivered else set()
    return [p for p in state["read"] if p.reference in cited]


def memory_reply(state: ConversationState) -> str:
    update = state["memory"] or MemoryUpdate(facts=[], forget=[])
    known = {f.key: f.statement for f in state["profile"]}
    lines = [f"Noted: {f.statement}" for f in update.facts]
    lines += [f"Forgotten: {known[k]}" for k in update.forget]
    return "\n".join(lines) or "There was nothing to remember or forget in that message."


def _listing(n: int, p: PaperCard) -> str:
    facts = [f"arXiv {p.arxiv_id}v{p.version}", p.published, "read before" if p.read_before else ""]
    return f"{n}. {p.title} ({', '.join(f for f in facts if f)}) — {p.reason}"


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
        "library": "resolve" if names(request) else "library",
        "memory": "remember",
        "other": "respond",
    }[request.intent]


def after_resolve(state: ConversationState) -> str:
    """Titles not found by title go to discovery, which may still find them (D28)."""
    request = _request(state)
    if any(t not in state["identified"] for t in request.titles):
        return "discover"
    return "read" if state["selected"] else "respond"


def after_library(state: ConversationState) -> str:
    return "read" if state["selected"] else "respond"


def after_respond(state: ConversationState) -> str:
    """A memory turn has already remembered; every other turn remembers after replying."""
    return END if _request(state).intent == "memory" else "remember"


def after_remember(state: ConversationState) -> str:
    return "respond" if _request(state).intent == "memory" else END


def after_discover(state: ConversationState) -> str:
    wants_reading = _request(state).intent != "discover"
    return (
        "choose_papers" if wants_reading and (state["papers"] or state["selected"]) else "respond"
    )


def after_choice(state: ConversationState) -> str:
    return "read" if state["selected"] else "respond"


def after_answer(state: ConversationState) -> str:
    """A library question whose papers do not answer it is searched on arXiv, once (D30)."""
    if _by_meaning(state) and not _answered(state) and not state["earlier"]:
        return "search_instead"
    return "respond"


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
    graph.add_node("conflicts", conflicts)
    graph.add_node("answer", run_answer)
    graph.add_node("library", find_in_library)
    graph.add_node("search_instead", search_instead)
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
    graph.add_conditional_edges("resolve", after_resolve, ["discover", "read", "respond"])
    graph.add_edge("read", "conflicts")
    graph.add_edge("conflicts", "answer")
    graph.add_conditional_edges("library", after_library, ["read", "respond"])
    graph.add_conditional_edges("answer", after_answer, ["search_instead", "respond"])
    graph.add_edge("search_instead", "discover")
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
