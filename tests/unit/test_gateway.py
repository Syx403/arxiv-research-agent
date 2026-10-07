"""The gateway's deterministic parts: the prewarm rule (D11) and the providers' usage objects,
built by the SDK from the JSON each provider returns."""

from openai.types import CompletionUsage
from openai.types.responses import ResponseUsage
from pydantic import BaseModel

from ara.llm.gateway import deepseek_usage, openai_usage, worth_prewarming
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
