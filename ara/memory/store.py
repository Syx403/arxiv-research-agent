"""Cross-session memory in the LangGraph Store (DESIGN §7, D27): profile facts under
("users", uid, "profile") and one record per research turn under ("users", uid, "episodes"),
searchable by meaning. Embeddings go through the content-addressed cache and the ledger."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from pydantic import BaseModel, Field

from ara.db.pool import Pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.rag.embed import embed

DIMS = 1536
EPISODES_SHOWN = 3  # records understand sees for the latest message
FACTS_SHOWN = 20  # profile facts understand sees: all of them up to here, then the most relevant
ALL = 10_000  # every item of a namespace, for code that lists it whole
# The turn (or evaluation run) the Store's own embedding requests are charged to (D39): the Store
# calls `vectors` itself, so the scope travels in a context variable.
_charged: ContextVar[Scope | None] = ContextVar("store_scope", default=None)


@contextmanager
def charged_to(scope: Scope) -> Iterator[None]:
    token = _charged.set(scope)
    try:
        yield
    finally:
        _charged.reset(token)


class Fact(BaseModel):
    """Something the user said about themselves, kept across sessions."""

    key: str = Field(description="Short snake_case topic, e.g. model_access; reuse it to update.")
    quote: str = Field(description="The user's own words, copied exactly from this turn.")
    statement: str = Field(description="The fact in one English sentence about the user.")


class Remembered(Fact):
    """A fact as kept: its source turn (DESIGN §7), filled in by code, never by the model."""

    thread: str  # the conversation
    turn: int  # index of the turn's first message in that conversation
    day: str


class Episode(BaseModel):
    """One research turn, written by code: no model summary, so nothing in it is invented."""

    day: str
    need: str
    read: list[str]  # "2210.03629v3 ReAct: ..."; for a library question, the papers cited
    listed: list[str]  # listed by a search but not read
    answer: str  # the direct answer, or "" when nothing was answered

    def text(self) -> str:
        lines = [
            f"{self.day}: {self.need}",
            *(f"  read {p}" for p in self.read),
            *(f"  listed {p}" for p in self.listed),
        ]
        return "\n".join([*lines, f"  answer: {self.answer}"] if self.answer else lines)

    def read_ids(self) -> list[str]:
        """The versioned arXiv ids of the papers read: "2210.03629v3"."""
        return [p.split(" ", 1)[0] for p in self.read]


async def open_store(pool: Pool, gateway: Gateway) -> AsyncPostgresStore:
    """The Store on the app database; `setup` creates or upgrades its tables. Episodes are indexed
    by their need; profile facts are read whole and never searched."""

    async def vectors(texts: Sequence[str]) -> list[list[float]]:
        scope = _charged.get() or Scope()
        found = await embed(list(texts), pool=pool, gateway=gateway, scope=scope)
        return [v.tolist() for v in found]

    store = AsyncPostgresStore(pool, index={"dims": DIMS, "embed": vectors, "fields": ["need"]})
    await store.setup()
    return store


def profile_namespace(user: str) -> tuple[str, ...]:
    return ("users", user, "profile")


def _episodes(user: str) -> tuple[str, ...]:
    return ("users", user, "episodes")


async def profile(store: BaseStore, user: str) -> list[Remembered]:
    """Every fact, by key."""
    items = await store.asearch(profile_namespace(user), limit=ALL)
    return sorted((Remembered(**i.value) for i in items), key=lambda f: f.key)


async def relevant(store: BaseStore, user: str, query: str) -> list[Remembered]:
    """What understand sees of the profile (D39): every fact while there are at most FACTS_SHOWN,
    then the FACTS_SHOWN closest in meaning to `query`, topped up with the newest (a fact kept
    before facts were indexed has no vector)."""
    everything = await store.asearch(profile_namespace(user), limit=ALL)
    if len(everything) <= FACTS_SHOWN:
        chosen = everything
    else:
        close = await store.asearch(profile_namespace(user), query=query, limit=FACTS_SHOWN)
        keys = {i.key for i in close}
        newest = sorted(everything, key=lambda i: i.updated_at, reverse=True)
        chosen = [*close, *(i for i in newest if i.key not in keys)][:FACTS_SHOWN]
    return sorted((Remembered(**i.value) for i in chosen), key=lambda f: f.key)


async def remember(store: BaseStore, user: str, fact: Remembered) -> None:
    """A fact with an existing key replaces the old one (DESIGN §7: newer supersedes); its
    statement is indexed, so a long profile can be searched by meaning (D39)."""
    await store.aput(profile_namespace(user), fact.key, fact.model_dump(), index=["statement"])


async def forget(store: BaseStore, user: str, key: str) -> None:
    await store.adelete(profile_namespace(user), key)


async def record(store: BaseStore, user: str, key: str, episode: Episode) -> None:
    await store.aput(_episodes(user), key, episode.model_dump())


async def recall(store: BaseStore, user: str, query: str) -> list[Episode]:
    """The research turns closest in meaning to `query`, most similar first."""
    items = await store.asearch(_episodes(user), query=query, limit=EPISODES_SHOWN)
    return [Episode(**i.value) for i in items]
