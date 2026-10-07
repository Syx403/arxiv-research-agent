"""LLM gateway (DESIGN §6.4): renders a prompt for its provider, meters the call in the ledger and
traces it in LangSmith. Retries belong to the graph's RetryPolicy, so SDK retries are off."""

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from functools import cache

import langsmith
import tiktoken
from langsmith.run_trees import RunTree
from langsmith.schemas import ExtractedUsageMetadata
from openai import APIStatusError, AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.responses import ParsedResponse, ResponseUsage
from pydantic import BaseModel

from ara.llm.ledger import Ledger, Scope
from ara.llm.pricing import Rates, Usage, rates, upper_bound
from ara.llm.prompt import Prompt, deepseek_messages, openai_input
from ara.llm.stages import Stage
from ara.settings import Settings

DEEPSEEK_URL = "https://api.deepseek.com"
ESTIMATE_MARGIN = 1.2  # providers tokenise differently from cl100k; reservations err high
CACHE_MINIMUM = 1_024  # OpenAI caches only prefixes of at least this many tokens
MESSAGE_OVERHEAD = 8  # tokens of role and framing per message


class InvalidOutput(Exception):
    """The provider answered, but not with what the stage needs."""


class Gateway:
    def __init__(self, settings: Settings, ledger: Ledger) -> None:
        self.ledger = ledger
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value(), max_retries=0)
        self.deepseek = AsyncOpenAI(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url=DEEPSEEK_URL,
            max_retries=0,
        )

    async def aclose(self) -> None:
        await self.openai.close()
        await self.deepseek.close()

    async def structured[T: BaseModel](
        self, stage: Stage, prompt: Prompt, schema: type[T], *, scope: Scope
    ) -> T:
        """An OpenAI stage with a strict JSON schema (Responses API)."""
        response = await self._respond(stage, prompt, schema, scope, prewarm=False)
        if response.output_parsed is None:
            raise InvalidOutput(f"{stage.name}: no structured output (status {response.status})")
        return response.output_parsed

    async def prewarm(
        self, stage: Stage, prompt: Prompt, schema: type[BaseModel], *, scope: Scope
    ) -> None:
        """Write the static and shared parts to the cache before a fan-out; nothing is generated.
        It sends the model, schema and effort of the calls that follow, so their prefix matches.
        Fan-outs call it only when `worth_prewarming` says so (D11)."""
        await self._respond(stage, replace(prompt, item=()), schema, scope, prewarm=True)

    async def _respond[T: BaseModel](
        self, stage: Stage, prompt: Prompt, schema: type[T], scope: Scope, *, prewarm: bool
    ) -> ParsedResponse[T]:
        messages = openai_input(prompt, stage.breakpoints)
        input_tokens = _estimate(prompt, schema)
        output_tokens = 0 if prewarm else stage.max_output_tokens
        async with self._metered(
            stage, prompt, scope, messages, input_tokens, output_tokens
        ) as call:
            response = await self.openai.responses.parse(
                model=stage.model,
                input=messages,
                text_format=schema,
                reasoning={"effort": stage.effort},
                max_output_tokens=stage.max_output_tokens,
                prompt_cache_options={"mode": "explicit", "prewarm": prewarm},
                store=False,
            )
            await call.settle(openai_usage(response.usage), response.id, response.output_text)
        return response

    async def text(self, stage: Stage, prompt: Prompt, *, scope: Scope) -> str:
        """A DeepSeek stage in thinking mode (Chat Completions); returns the final answer."""
        messages = deepseek_messages(prompt)
        async with self._metered(
            stage, prompt, scope, messages, _estimate(prompt), stage.max_output_tokens
        ) as call:
            response = await self.deepseek.chat.completions.create(
                model=stage.model,
                messages=messages,
                reasoning_effort=stage.effort,
                max_tokens=stage.max_output_tokens,
                extra_body={"thinking": {"type": "enabled"}},
            )
            choice = response.choices[0]
            await call.settle(deepseek_usage(response.usage), response.id, choice.message.content)
        if not choice.message.content:
            raise InvalidOutput(
                f"{stage.name}: empty answer (finish_reason {choice.finish_reason})"
            )
        return choice.message.content

    @asynccontextmanager
    async def _metered(
        self,
        stage: Stage,
        prompt: Prompt,
        scope: Scope,
        messages: object,
        input_tokens: int,
        output_tokens: int,
    ) -> AsyncIterator["Call"]:
        """Reserve before sending. A request the provider rejected (4xx) releases the reservation;
        any other failure keeps it, because the request may have been billed."""
        r = rates(stage.model, datetime.now(UTC))
        call_id = await self.ledger.reserve(
            stage=stage.name,
            model=stage.model,
            prompt_version=prompt.instructions.version,
            scope=scope,
            usd=upper_bound(input_tokens, output_tokens, r),
        )
        metadata = {
            "stage": stage.name,
            "prompt_version": prompt.instructions.version,
            "effort": stage.effort,
            "ls_provider": stage.provider,
            "ls_model_name": stage.model,
            "run_id": scope.run_id,
            "turn_id": scope.turn_id,
        }
        with langsmith.trace(
            stage.name, "llm", inputs={"messages": messages}, metadata=metadata
        ) as run:
            try:
                yield Call(self.ledger, call_id, r, run, time.perf_counter())
            except APIStatusError as error:
                await self.ledger.fail(call_id, repr(error), released=error.status_code < 500)
                raise
            except Exception as error:
                await self.ledger.fail(call_id, repr(error), released=False)
                raise


@dataclass(frozen=True)
class Call:
    """One metered request; the caller settles it once the provider answers."""

    ledger: Ledger
    call_id: int
    rates: Rates
    run: RunTree
    started: float

    async def settle(self, usage: Usage, response_id: str, output: str | None) -> None:
        cost = usage.cost(self.rates)
        latency_ms = round((time.perf_counter() - self.started) * 1000)
        await self.ledger.settle(
            self.call_id, usage, cost, latency_ms=latency_ms, response_id=response_id
        )
        self.run.set(outputs={"output": output}, usage_metadata=_usage_metadata(usage, cost))


def openai_usage(usage: ResponseUsage | None) -> Usage:
    if usage is None:
        raise InvalidOutput("OpenAI response without usage")
    return Usage(
        input_tokens=usage.input_tokens,
        cached_tokens=usage.input_tokens_details.cached_tokens,
        cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
    )


def deepseek_usage(usage: CompletionUsage | None) -> Usage:
    """DeepSeek reports cache hits in its own field, `prompt_cache_hit_tokens`."""
    if usage is None:
        raise InvalidOutput("DeepSeek response without usage")
    details = usage.completion_tokens_details
    return Usage(
        input_tokens=usage.prompt_tokens,
        cached_tokens=(usage.model_extra or {}).get("prompt_cache_hit_tokens", 0),
        cache_write_tokens=0,
        output_tokens=usage.completion_tokens,
        reasoning_tokens=(details.reasoning_tokens or 0) if details else 0,
    )


def _usage_metadata(usage: Usage, cost: Decimal) -> ExtractedUsageMetadata:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "input_token_details": {
            "cache_read": usage.cached_tokens,
            "cache_creation": usage.cache_write_tokens,
        },
        "output_token_details": {"reasoning": usage.reasoning_tokens},
        "total_cost": float(cost),
    }


@cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def worth_prewarming(prompt: Prompt, schema: type[BaseModel], calls: int) -> bool:
    """D11: prewarm only a fan-out of at least two calls whose shared prefix (instructions, shared
    blocks and output schema) reaches the cache minimum; otherwise the write cannot pay off."""
    return calls >= 2 and _tokens(replace(prompt, item=()), schema) >= CACHE_MINIMUM


def _tokens(prompt: Prompt, schema: type[BaseModel] | None = None) -> int:
    """cl100k count of a prompt and its output schema; providers count slightly differently."""
    text = prompt.text() + (json.dumps(schema.model_json_schema()) if schema else "")
    return len(_encoding().encode_ordinary(text))


def _estimate(prompt: Prompt, schema: type[BaseModel] | None = None) -> int:
    """An upper estimate of input tokens, used only for the reservation."""
    messages = sum(len(blocks) for _, blocks in prompt.parts())
    return round(_tokens(prompt, schema) * ESTIMATE_MARGIN) + MESSAGE_OVERHEAD * messages
