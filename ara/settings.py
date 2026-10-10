"""Settings: provider keys come from .env; ARA_* options have local-development defaults."""

import os
from decimal import Decimal
from functools import cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: SecretStr
    deepseek_api_key: SecretStr
    cohere_api_key: SecretStr
    anthropic_api_key: SecretStr | None = None  # the model study only (D44)
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = False

    database_url: str = Field(
        "postgresql://ara:ara@127.0.0.1:5434/ara", validation_alias="ARA_DATABASE_URL"
    )
    langsmith_project: str = Field("ara-v2", validation_alias="ARA_LANGSMITH_PROJECT")
    global_cap_usd: Decimal = Field(Decimal("10"), validation_alias="ARA_GLOBAL_CAP_USD")
    turn_cap_usd: Decimal = Field(Decimal("0.05"), validation_alias="ARA_TURN_CAP_USD")


@cache
def get_settings() -> Settings:
    return Settings()  # required keys are read from the environment and .env


def configure_tracing(settings: Settings) -> None:
    """Export the LangSmith options to the process environment, where langsmith and LangGraph
    read them. Call once at start-up, before the first traced call (langsmith caches them)."""
    key = settings.langsmith_api_key
    if settings.langsmith_tracing and key is not None:
        os.environ.update(
            LANGSMITH_TRACING="true",
            LANGSMITH_API_KEY=key.get_secret_value(),
            LANGSMITH_PROJECT=settings.langsmith_project,
        )
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
