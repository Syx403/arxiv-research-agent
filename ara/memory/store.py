"""Cross-session memory in the LangGraph Store (DESIGN §7, D27): profile facts under
("users", uid, "profile") and one record per research turn under ("users", uid, "episodes"),
searchable by meaning. Embeddings go through the content-addressed cache and the ledger."""

from collections.abc import Sequence

from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from pydantic import BaseModel, Field

from ara.db.pool import Pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.rag.embed import embed

DIMS = 1536
EPISODES_SHOWN = 3  # records understand sees for the latest message


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
        found = await embed(list(texts), pool=pool, gateway=gateway, scope=Scope())
        return [v.tolist() for v in found]

    store = AsyncPostgresStore(pool, index={"dims": DIMS, "embed": vectors, "fields": ["need"]})
    await store.setup()
    return store


def profile_namespace(user: str) -> tuple[str, ...]:
    return ("users", user, "profile")


def _episodes(user: str) -> tuple[str, ...]:
    return ("users", user, "episodes")


async def profile(store: BaseStore, user: str) -> list[Remembered]:
    items = await store.asearch(profile_namespace(user), limit=100)
    return sorted((Remembered(**i.value) for i in items), key=lambda f: f.key)


async def remember(store: BaseStore, user: str, fact: Remembered) -> None:
    """A fact with an existing key replaces the old one (DESIGN §7: newer supersedes)."""
    await store.aput(profile_namespace(user), fact.key, fact.model_dump(), index=False)


async def forget(store: BaseStore, user: str, key: str) -> None:
    await store.adelete(profile_namespace(user), key)


async def record(store: BaseStore, user: str, key: str, episode: Episode) -> None:
    await store.aput(_episodes(user), key, episode.model_dump())


async def recall(store: BaseStore, user: str, query: str) -> list[Episode]:
    """The research turns closest in meaning to `query`, most similar first."""
    items = await store.asearch(_episodes(user), query=query, limit=EPISODES_SHOWN)
    return [Episode(**i.value) for i in items]
