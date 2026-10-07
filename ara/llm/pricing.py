"""Prices in US$ per 1M tokens, checked 2026-10-07 (DESIGN §6.1), and the cost of a call."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class Rates:
    input: Decimal
    cached: Decimal
    cache_write: Decimal
    output: Decimal


def _rates(input: str, cached: str, cache_write: str, output: str) -> Rates:
    return Rates(Decimal(input), Decimal(cached), Decimal(cache_write), Decimal(output))


LUNA = _rates(input="0.10", cached="0.01", cache_write="0.125", output="0.50")
# DeepSeek has no cache-write charge: a cache miss is billed at the input rate.
FLASH_PEAK = _rates(input="0.30", cached="0.006", cache_write="0.30", output="1.20")
FLASH_OFF_PEAK = _rates(input="0.15", cached="0.003", cache_write="0.15", output="0.60")
# text-embedding-3-small, checked 2026-10-08 (DESIGN §17); embeddings have no cache.
EMBEDDING = _rates(input="0.02", cached="0.02", cache_write="0.02", output="0")
# Cohere trial keys are free but capped at 1,000 calls a month; the ledger still counts each call.
RERANK_TRIAL = _rates(input="0", cached="0", cache_write="0", output="0")
# DeepSeek peak: 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday. Chinese public holidays are
# off-peak in reality; counting them as peak errs high.
PEAK_HOURS_UTC = frozenset({1, 2, 3, 6, 7, 8, 9})


def rates(model: str, at: datetime) -> Rates:
    """The rates that apply to a request sent at `at` (timezone-aware)."""
    match model:
        case "gpt-6-luna":
            return LUNA
        case "deepseek-flash":
            utc = at.astimezone(UTC)
            peak = utc.weekday() < 5 and utc.hour in PEAK_HOURS_UTC
            return FLASH_PEAK if peak else FLASH_OFF_PEAK
        case "text-embedding-3-small":
            return EMBEDDING
        case "rerank-v4.0-pro":
            return RERANK_TRIAL
    raise KeyError(f"no price for model {model!r}")


@dataclass(frozen=True)
class Usage:
    """Token counts as reported by the provider. `input_tokens` includes the cached and the
    cache-write tokens; `output_tokens` includes the reasoning tokens."""

    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int

    def cost(self, r: Rates) -> Decimal:
        uncached = self.input_tokens - self.cached_tokens - self.cache_write_tokens
        return (
            uncached * r.input
            + self.cached_tokens * r.cached
            + self.cache_write_tokens * r.cache_write
            + self.output_tokens * r.output
        ) / MILLION


def upper_bound(input_tokens: int, output_tokens: int, r: Rates) -> Decimal:
    """The most a request can cost: every input token written to the cache, every output token
    used. This is what the ledger reserves before sending."""
    return (input_tokens * max(r.input, r.cache_write) + output_tokens * r.output) / MILLION
