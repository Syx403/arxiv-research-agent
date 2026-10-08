"""What a turn changes in the user's profile (DESIGN §7, D27): one Luna call proposes facts to keep
and keys to forget; code keeps a fact only if its quote is the user's own words from this turn
(v1 rule) and forgets only keys that exist."""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ara.graph.state import plain, quoted
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.llm.prompt import Block, Instructions, Prompt, data
from ara.llm.stages import STAGES
from ara.memory.store import Fact

REMEMBER = Instructions.load("remember")


class MemoryUpdate(BaseModel):
    facts: list[Fact] = Field(description="Facts to keep; a known key replaces that fact.")
    forget: list[str] = Field(description="Keys of remembered facts to delete.")


def remember_prompt(profile: Sequence[Fact], said: Sequence[str]) -> Prompt:
    known = data("Remembered", [{"key": f.key, "statement": f.statement} for f in profile])
    turn = Block("user", "Messages this turn:\n" + "\n".join(said))
    return Prompt(REMEMBER, shared=(known,), item=(turn,))


async def propose(
    gateway: Gateway, profile: Sequence[Fact], said: Sequence[str], *, scope: Scope
) -> MemoryUpdate:
    prompt = remember_prompt(profile, said)
    update = await gateway.structured(STAGES["remember"], prompt, MemoryUpdate, scope=scope)
    return checked(update, profile, said)


def checked(update: MemoryUpdate, profile: Sequence[Fact], said: Sequence[str]) -> MemoryUpdate:
    """The trust boundary: non-empty quotes from this turn's user messages, keys that exist, one
    fact per key (the last one wins), and a key is not both kept and forgotten."""
    text = plain(" ".join(said))
    facts = {f.key: f for f in update.facts if f.key.strip() and quoted(f.quote, text)}
    known = {f.key for f in profile}
    forget = [k for k in dict.fromkeys(update.forget) if k in known and k not in facts]
    return MemoryUpdate(facts=list(facts.values()), forget=forget)
