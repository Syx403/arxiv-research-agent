from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.llm.errors import LLMAuthError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    deepseek_api_key: SecretStr | None = Field(default=None, validation_alias="DEEPSEEK_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    cohere_api_key: SecretStr | None = Field(default=None, validation_alias="COHERE_API_KEY")

    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="arxiv-research-agent", validation_alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=True, validation_alias="LANGSMITH_TRACING")

    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="arxiv_agent", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="arxiv_agent", validation_alias="POSTGRES_USER")
    postgres_password: SecretStr = Field(
        default=SecretStr("arxiv_agent_dev"), validation_alias="POSTGRES_PASSWORD"
    )

    semantic_scholar_api_key: SecretStr | None = Field(
        default=None, validation_alias="SEMANTIC_SCHOLAR_API_KEY"
    )
    github_token: SecretStr | None = Field(default=None, validation_alias="GITHUB_TOKEN")

    openrouter_http_referer: str | None = Field(
        default=None, validation_alias="OPENROUTER_HTTP_REFERER"
    )
    openrouter_x_title: str | None = Field(
        default="arxiv-research-agent", validation_alias="OPENROUTER_X_TITLE"
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def require_secret(secret: SecretStr | None, *, env_var: str, provider: str) -> str:
    if secret is None or not secret.get_secret_value():
        raise LLMAuthError(f"Missing required environment variable {env_var}", provider=provider)
    return secret.get_secret_value()
