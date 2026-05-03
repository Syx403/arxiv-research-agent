from __future__ import annotations

import httpx
import pytest

from src.llm.errors import (
    LLMAuthError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMServerError,
)
from src.llm.providers.cohere_rerank import CohereRerankClient
from src.llm.providers.deepseek import DeepSeekChatClient
from src.llm.providers.openai_embed import OpenAIEmbeddingClient
from src.llm.providers.openrouter import OpenRouterChatClient


STATUS_CASES = [
    (400, LLMInvalidRequestError),
    (401, LLMAuthError),
    (403, LLMAuthError),
    (429, LLMRateLimitError),
    (500, LLMServerError),
    (503, LLMServerError),
]


@pytest.mark.parametrize("status,expected", STATUS_CASES)
@pytest.mark.parametrize("client_cls", [OpenRouterChatClient, DeepSeekChatClient])
def test_httpx_chat_provider_status_mapping(status: int, expected: type[Exception], client_cls) -> None:
    client = client_cls(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))),
    )
    response = httpx.Response(status, text="provider error")
    with pytest.raises(expected):
        client._raise_for_status(response)


@pytest.mark.parametrize("status,expected", STATUS_CASES)
@pytest.mark.parametrize("client_cls", [OpenAIEmbeddingClient, CohereRerankClient])
def test_sdk_provider_status_mapping(status: int, expected: type[Exception], client_cls) -> None:
    client = client_cls(api_key="test-key", client=object())
    with pytest.raises(expected):
        client._raise_for_status(status, "provider error")
