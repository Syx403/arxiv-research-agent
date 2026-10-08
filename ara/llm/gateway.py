"""Model gateway (DESIGN §6.4): every billable model call (LLM, embedding, rerank) passes here.
It renders the request for its provider, meters it in the ledger and traces it in LangSmith.
Retries belong to the graph's RetryPolicy, so SDK retries are off."""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
import langsmith
from langsmith.run_trees import RunTree
from langsmith.schemas import ExtractedUsageMetadata
from openai import APIStatusError, AsyncOpenAI, omit
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage, ChatCompletionToolParam
from openai.types.responses import ParsedResponse, ResponseUsage
from pydantic import BaseModel, ValidationError

from ara.llm.ledger import Ledger, Scope
from ara.llm.pricing import Rates, Usage, rates, upper_bound
from ara.llm.prompt import Block, Prompt, ToolCall, deepseek_messages, openai_input
from ara.llm.stages import EMBEDDING_MODEL, PROVIDERS, RERANK_MODEL, Stage
from ara.settings import Settings
from ara.tokens import count_tokens

DEEPSEEK_URL = "https://api.deepseek.com"
COHERE_URL = "https://api.cohere.com"
ESTIMATE_MARGIN = 1.2  # providers tokenise differently from cl100k; reservations err high
CACHE_MINIMUM = 1_024  # OpenAI caches only prefixes of at least this many tokens
MESSAGE_OVERHEAD = 8  # tokens of role and framing per message
RERANK_INTERVAL_S = 6.0  # Cohere trial keys allow 10 rerank calls per minute
# Backstop for one HTTP request to any provider, instead of the SDK's 600 s. It must outlast the
# slowest legitimate call (8K thinking tokens on DeepSeek); callers outside the graph, such as the
# eval runner, rely on it, and graph nodes add their own, tighter timeouts in M5 (§4.4).
REQUEST_TIMEOUT_S = 120.0


class InvalidOutput(Exception):
    """The provider answered, but not with what the stage needs."""


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    id: str
    results: list[RerankResult]


class Gateway:
    def __init__(self, settings: Settings, ledger: Ledger) -> None:
        self.ledger = ledger
        self.openai = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            max_retries=0,
            timeout=REQUEST_TIMEOUT_S,
        )
        self.deepseek = AsyncOpenAI(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url=DEEPSEEK_URL,
            max_retries=0,
            timeout=REQUEST_TIMEOUT_S,
        )
        self.cohere = httpx.AsyncClient(
            base_url=COHERE_URL,
            headers={"Authorization": f"Bearer {settings.cohere_api_key.get_secret_value()}"},
            timeout=REQUEST_TIMEOUT_S,
        )
        self._rerank_slot = asyncio.Lock()
        self._last_rerank = 0.0

    async def aclose(self) -> None:
        await self.openai.close()
        await self.deepseek.close()
        await self.cohere.aclose()

    async def structured[T: BaseModel](
        self, stage: Stage, prompt: Prompt, schema: type[T], *, scope: Scope
    ) -> T:
        """Structured output. OpenAI enforces the strict JSON schema (Responses API); DeepSeek only
        guarantees a JSON object, so its reply is validated here and a mismatch raises InvalidOutput
        (counted by the caller, never repaired; D10)."""
        if stage.provider == "deepseek":
            content = await self._chat(stage, prompt, scope, schema=schema)
            try:
                return schema.model_validate_json(content)
            except ValidationError as error:
                raise InvalidOutput(
                    f"{stage.name}: reply does not match {schema.__name__}"
                ) from error
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
        async with self._metered(
            stage.name,
            stage.model,
            prompt.version,
            scope,
            inputs={"messages": messages},
            input_tokens=_estimate(prompt, schema),
            output_tokens=0 if prewarm else stage.max_output_tokens,
            metadata={"effort": stage.effort, "prewarm": prewarm},
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
        return await self._chat(stage, prompt, scope)

    async def tool_step(
        self, stage: Stage, prompt: Prompt, tools: Sequence["Tool"], *, scope: Scope
    ) -> Block:
        """One step of a DeepSeek tool loop (the researcher): the assistant turn, with its
        reasoning and tool calls, ready to append to the transcript. The graph runs the tools and
        decides when the loop stops (DESIGN §4.2)."""
        message = await self._complete(stage, prompt, scope, tools=tools)
        calls = tuple(
            ToolCall(c.id, c.function.name, c.function.arguments)
            for c in message.tool_calls or ()
            if c.type == "function"
        )
        reasoning = (message.model_extra or {}).get("reasoning_content") or ""
        return Block("assistant", message.content or "", reasoning=reasoning, calls=calls)

    async def _chat(
        self, stage: Stage, prompt: Prompt, scope: Scope, *, schema: type[BaseModel] | None = None
    ) -> str:
        message = await self._complete(stage, prompt, scope, schema=schema)
        if not message.content:
            raise InvalidOutput(f"{stage.name}: empty answer")
        return message.content

    async def _complete(
        self,
        stage: Stage,
        prompt: Prompt,
        scope: Scope,
        *,
        schema: type[BaseModel] | None = None,
        tools: Sequence["Tool"] = (),
    ) -> ChatCompletionMessage:
        """DeepSeek Chat Completions in thinking mode. With a schema, JSON mode is on and the schema
        follows the static instructions, so every call of the stage shares the same prefix."""
        messages = deepseek_messages(prompt)
        if schema is not None:
            contract = (
                f"Reply with one JSON object that matches this JSON Schema:\n{_schema(schema)}"
            )
            messages.insert(1, {"role": "system", "content": contract})
        specs = [tool.spec() for tool in tools]
        async with self._metered(
            stage.name,
            stage.model,
            prompt.version,
            scope,
            inputs={"messages": messages},
            input_tokens=_estimate(prompt, schema) + count_tokens(json.dumps(specs)),
            output_tokens=stage.max_output_tokens,
            metadata={"effort": stage.effort},
        ) as call:
            response = await self.deepseek.chat.completions.create(
                model=stage.model,
                messages=messages,
                reasoning_effort=stage.effort,
                max_tokens=stage.max_output_tokens,
                extra_body={"thinking": {"type": "enabled"}},
                response_format={"type": "json_object"} if schema else omit,
                tools=specs or omit,
            )
            message = response.choices[0].message
            await call.settle(deepseek_usage(response.usage), response.id, message.content)
        return message

    async def embed(self, texts: Sequence[str], *, scope: Scope) -> list[list[float]]:
        """One embeddings request; callers batch and cache (ara.rag.embed)."""
        tokens = sum(count_tokens(text) for text in texts)
        async with self._metered(
            "embed",
            EMBEDDING_MODEL,
            EMBEDDING_MODEL,
            scope,
            inputs={"texts": len(texts), "tokens": tokens},
            input_tokens=round(tokens * ESTIMATE_MARGIN),
            output_tokens=0,
        ) as call:
            response = await self.openai.embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
            usage = Usage(response.usage.prompt_tokens, 0, 0, 0, 0)
            await call.settle(usage, f"embed:{len(texts)}", f"{len(response.data)} vectors")
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int, scope: Scope
    ) -> list[RerankResult]:
        """Cohere rerank, results in descending relevance. Calls are spaced for the trial limit."""
        async with self._rerank_slot:
            await asyncio.sleep(max(0.0, self._last_rerank + RERANK_INTERVAL_S - time.monotonic()))
            try:
                return await self._rerank(query, documents, top_n, scope)
            finally:  # a failed call counts against the rate limit too
                self._last_rerank = time.monotonic()

    async def _rerank(
        self, query: str, documents: Sequence[str], top_n: int, scope: Scope
    ) -> list[RerankResult]:
        async with self._metered(
            "rerank",
            RERANK_MODEL,
            RERANK_MODEL,
            scope,
            inputs={"query": query, "documents": len(documents)},
            input_tokens=0,
            output_tokens=0,
        ) as call:
            response = await self.cohere.post(
                "/v2/rerank",
                json={
                    "model": RERANK_MODEL,
                    "query": query,
                    "documents": list(documents),
                    "top_n": top_n,
                },
            )
            response.raise_for_status()
            body = RerankResponse.model_validate_json(response.content)
            await call.settle(Usage(0, 0, 0, 0, 0), body.id, f"{len(body.results)} results")
        return body.results

    @asynccontextmanager
    async def _metered(
        self,
        name: str,
        model: str,
        version: str,
        scope: Scope,
        *,
        inputs: Mapping[str, Any],
        input_tokens: int,
        output_tokens: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> AsyncIterator["Call"]:
        """Reserve before sending. A request the provider rejected (4xx) releases the reservation;
        any other failure keeps it, because the request may have been billed."""
        r = rates(model, datetime.now(UTC))
        call_id = await self.ledger.reserve(
            stage=name,
            model=model,
            prompt_version=version,
            scope=scope,
            usd=upper_bound(input_tokens, output_tokens, r),
        )
        trace_metadata = {
            "stage": name,
            "prompt_version": version,
            "ls_provider": PROVIDERS[model],
            "ls_model_name": model,
            "run_id": scope.run_id,
            "turn_id": scope.turn_id,
            **(metadata or {}),
        }
        with langsmith.trace(name, "llm", inputs=dict(inputs), metadata=trace_metadata) as run:
            try:
                yield Call(self.ledger, call_id, r, run, time.perf_counter())
            except Exception as error:
                await self.ledger.fail(call_id, repr(error), released=_rejected(error))
                raise


def _rejected(error: Exception) -> bool:
    """A 4xx answer means the provider refused the request before doing (and billing) any work."""
    if isinstance(error, APIStatusError):
        return error.status_code < 500
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code < 500
    return False


@dataclass(frozen=True)
class Tool:
    """A function the model may call; `args` validates what it passes (a trust boundary)."""

    name: str
    description: str
    args: type[BaseModel]

    def spec(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args.model_json_schema(),
            },
        }


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


def worth_prewarming(prompt: Prompt, schema: type[BaseModel], calls: int) -> bool:
    """D11: prewarm only a fan-out of at least two calls whose shared prefix (instructions, shared
    blocks and output schema) reaches the cache minimum; otherwise the write cannot pay off."""
    return calls >= 2 and _tokens(replace(prompt, item=()), schema) >= CACHE_MINIMUM


def _tokens(prompt: Prompt, schema: type[BaseModel] | None = None) -> int:
    """cl100k count of a prompt and its output schema; providers count slightly differently."""
    return count_tokens(prompt.text() + (_schema(schema) if schema else ""))


def _schema(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), sort_keys=True)


def _estimate(prompt: Prompt, schema: type[BaseModel] | None = None) -> int:
    """An upper estimate of input tokens, used only for the reservation."""
    messages = sum(len(blocks) for _, blocks in prompt.parts())
    return round(_tokens(prompt, schema) * ESTIMATE_MARGIN) + MESSAGE_OVERHEAD * messages
