from __future__ import annotations

import pytest

from src.core.config import require_secret
from src.llm.errors import LLMAuthError


def test_require_secret_names_missing_env_var() -> None:
    with pytest.raises(LLMAuthError) as exc_info:
        require_secret(None, env_var="DEEPSEEK_API_KEY", provider="deepseek_native")
    assert "DEEPSEEK_API_KEY" in str(exc_info.value)
