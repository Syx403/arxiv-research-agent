"""The gateway's deterministic parts: the prewarm rule (D11) and the providers' usage objects,
built by the SDK from the JSON each provider returns."""

from openai.types import CompletionUsage
from openai.types.responses import ResponseUsage
from pydantic import BaseModel

from ara.llm.gateway import (
    AnthropicUsage,
    anthropic_usage,
    deepseek_usage,
    openai_usage,
    strict_schema,
    worth_prewarming,
)
from ara.llm.pricing import Usage
from ara.llm.prompt import Block, Instructions, Prompt

INSTRUCTIONS = Instructions("verify", "Check the claim against the evidence.")
LONG = "evidence " * 1_200  # about 1,200 tokens


class Verdict(BaseModel):
    supported: bool


def test_prewarm_needs_two_calls_and_a_cacheable_shared_prefix() -> None:
    long_shared = Prompt(INSTRUCTIONS, shared=(Block("user", LONG),))
    long_item = Prompt(INSTRUCTIONS, shared=(Block("user", "short"),), item=(Block("user", LONG),))
    assert worth_prewarming(long_shared, Verdict, calls=2)
    assert not worth_prewarming(long_shared, Verdict, calls=1)
    assert not worth_prewarming(long_item, Verdict, calls=8)  # only the shared prefix counts


def test_openai_usage_keeps_cache_reads_and_writes_apart() -> None:
    usage = ResponseUsage.model_validate(
        {
            "input_tokens": 3_000,
            "input_tokens_details": {"cached_tokens": 2_048, "cache_write_tokens": 900},
            "output_tokens": 120,
            "output_tokens_details": {"reasoning_tokens": 64},
            "total_tokens": 3_120,
        }
    )
    assert openai_usage(usage) == Usage(3_000, 2_048, 900, 120, 64)


def test_deepseek_usage_reads_its_own_cache_field() -> None:
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 3_000,
            "completion_tokens": 500,
            "total_tokens": 3_500,
            "completion_tokens_details": {"reasoning_tokens": 300},
            "prompt_cache_hit_tokens": 2_816,
            "prompt_cache_miss_tokens": 184,
        }
    )
    assert deepseek_usage(usage) == Usage(3_000, 2_816, 0, 500, 300)


def test_anthropic_usage_adds_cache_reads_and_writes_to_the_input() -> None:
    """D44: Anthropic's input_tokens is only what follows the last breakpoint."""
    usage = anthropic_usage(
        AnthropicUsage(
            input_tokens=50,
            cache_read_input_tokens=1800,
            cache_creation_input_tokens=248,
            output_tokens=503,
        )
    )
    assert usage == Usage(2098, 1800, 248, 503, 0)


def test_strict_schema_closes_every_object_and_drops_unsupported_bounds() -> None:
    from pydantic import Field

    class Inner(BaseModel):
        n: int = Field(ge=0, le=3)
        note: str | None = None

    class Outer(BaseModel):
        items: list[Inner]

    schema = strict_schema(Outer)
    inner = schema["$defs"]["Inner"]
    assert schema["additionalProperties"] is False and schema["required"] == ["items"]
    assert inner["additionalProperties"] is False and inner["required"] == ["n", "note"]
    assert "minimum" not in inner["properties"]["n"] and "maximum" not in inner["properties"]["n"]
