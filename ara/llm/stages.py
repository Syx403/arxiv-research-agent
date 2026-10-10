"""Stage table (DESIGN §6.2, D10): model, reasoning effort, output cap and cache breakpoints.

Stages built in later milestones start from these values and are re-measured there.
"""

from dataclasses import dataclass
from typing import Literal

from ara.llm.prompt import Part

type Provider = Literal["openai", "deepseek", "cohere", "anthropic"]
type Effort = Literal["none", "low", "medium", "high", "xhigh", "max"]

LUNA, FLASH, HAIKU = "gpt-6-luna", "deepseek-flash", "claude-haiku-5-5"
EMBEDDING_MODEL, RERANK_MODEL = "text-embedding-3-small", "rerank-v4.0-pro"
PROVIDERS: dict[str, Provider] = {
    LUNA: "openai",
    FLASH: "deepseek",
    HAIKU: "anthropic",
    EMBEDDING_MODEL: "openai",
    RERANK_MODEL: "cohere",
}
# Efforts each provider accepts (DESIGN §16); DeepSeek maps any other value to one of these.
EFFORTS: dict[Provider, frozenset[Effort]] = {
    "openai": frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
    "deepseek": frozenset({"low", "high", "max"}),
    "cohere": frozenset(),
    "anthropic": frozenset({"low", "medium", "high", "xhigh", "max"}),
}


@dataclass(frozen=True)
class Stage:
    name: str
    model: str
    effort: Effort
    max_output_tokens: int  # reasoning included; the ledger reserves this much output
    breakpoints: frozenset[Part] = frozenset()  # parts that end with a cache breakpoint (OpenAI,
    # Anthropic; DeepSeek caches prefixes on its own)

    @property
    def provider(self) -> Provider:
        return PROVIDERS[self.model]


FANOUT: frozenset[Part] = frozenset({"static", "shared"})
CONVERSATION: frozenset[Part] = frozenset({"static", "shared", "item"})

STAGES = {
    stage.name: stage
    for stage in (
        Stage("understand", LUNA, "low", 2_000, CONVERSATION),
        Stage("researcher", FLASH, "low", 4_000),
        Stage("screen", LUNA, "medium", 6_000, FANOUT),
        Stage("select_evidence", LUNA, "low", 1_500, FANOUT),  # max seen 384 (D39)
        Stage("synthesize", FLASH, "high", 12_000),  # one draft used all 8K thinking (D39)
        Stage("repair", FLASH, "high", 12_000),
        Stage("verify", LUNA, "medium", 1_500, FANOUT),  # max seen 860 (D39)
        Stage("remember", LUNA, "low", 1_500, frozenset({"static"})),
        Stage("judge_answer", LUNA, "medium", 3_000, FANOUT),
        Stage("judge_relevance", FLASH, "high", 6_000),
    )
}
