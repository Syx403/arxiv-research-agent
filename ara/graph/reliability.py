"""Reliability (DESIGN §4.4, M5a): one retry layer, node timeouts, and degraded results instead of
exceptions the user would see.

- Retries are only the nodes' `RetryPolicy`; SDK retries are off. `transient` decides what is
  worth another attempt: rate limits (429), server errors (5xx), dropped connections and timeouts,
  and an empty model answer (once more). A refused request (4xx), a budget cap or a malformed
  input fails at once: sending it again costs money and changes nothing.
- Plain nodes that fail after their retries go to an error handler, which records a problem and
  names the next step (`Command(goto=...)`): LangGraph 1.2.14 does not follow the failed node's
  own edges after a handler, and does not run handlers for `Send` tasks at all (both checked).
- `Send` nodes are wrapped by `degrade`: on the last attempt, or for an error not worth retrying,
  the node returns its fallback result (no evidence from that paper, a line left unverified), so
  one item fails and the others carry on.
"""

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

import httpx
import openai
from langgraph.errors import NodeTimeoutError
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from ara.llm.gateway import InvalidOutput

LLM_TIMEOUT_S = 45.0  # one model call per node attempt (DESIGN §4.4)
# Writing an answer thinks longest (35.6 s seen, one draft at its 8K-token cap), and D37 asks for
# longer answers (D39).
SYNTHESIS_TIMEOUT_S = 90.0
# A node that waits for a shared rate-limit slot (arXiv one request per 3 s, rerank one per 6 s,
# shared by every running turn) is not timed by its queue: each request it sends has its own
# timeout, and the node only this backstop (D39).
SLOT_BACKSTOP_S = 300.0
ATTEMPTS = 3  # first try included
EMPTY_ATTEMPTS = 2  # an empty model answer is asked again once

log = logging.getLogger(__name__)


def transient(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return _retryable_status(error.response.status_code)
    if isinstance(error, openai.APIStatusError):
        return _retryable_status(error.status_code)
    return isinstance(
        error, httpx.TransportError | openai.APIConnectionError | NodeTimeoutError | InvalidOutput
    )


def _retryable_status(code: int) -> bool:
    return code == 429 or code >= 500


def attempts(error: BaseException) -> int:
    return EMPTY_ATTEMPTS if isinstance(error, InvalidOutput) else ATTEMPTS


RETRY = (
    RetryPolicy(
        max_attempts=EMPTY_ATTEMPTS,
        initial_interval=1.0,
        retry_on=lambda e: isinstance(e, InvalidOutput),
    ),
    RetryPolicy(
        max_attempts=ATTEMPTS, initial_interval=1.0, backoff_factor=3.0, retry_on=transient
    ),
)


def problem(what: str, error: BaseException) -> str:
    """What the reply tells the user about a part that failed, in words rather than an exception
    name (D38); the exception itself goes to the log."""
    log.warning("%s failed: %r", what, error)
    return f"{what} failed: {why(error)}"


def why(error: BaseException) -> str:
    if isinstance(error, httpx.HTTPStatusError | openai.APIStatusError):
        code = (
            error.response.status_code
            if isinstance(error, httpx.HTTPStatusError)
            else error.status_code
        )
        return "too many requests right now (HTTP 429)" if code == 429 else f"HTTP {code}"
    timeouts = NodeTimeoutError | TimeoutError | httpx.TimeoutException | openai.APITimeoutError
    if isinstance(error, timeouts):
        return "it took too long"
    if isinstance(error, httpx.TransportError | openai.APIConnectionError):
        return "the service could not be reached"
    if isinstance(error, InvalidOutput):
        return "the model returned nothing usable"
    return "an unexpected error"


def degrade[F: Callable[..., Awaitable[dict[str, Any]]]](
    fallback: Callable[[Any, BaseException], dict[str, Any]],
) -> Callable[[F], F]:
    """For a `Send` node: re-raise while a retry may still help, then return `fallback`."""

    def wrap(node: F) -> F:
        @functools.wraps(node)
        async def guarded(state: Any, runtime: Runtime[Any]) -> dict[str, Any]:
            try:
                return await node(state, runtime)
            except Exception as error:
                info = runtime.execution_info
                attempt = info.node_attempt if info is not None else attempts(error)
                if transient(error) and attempt < attempts(error):
                    raise
                return fallback(state, error)

        return cast(F, guarded)

    return wrap
