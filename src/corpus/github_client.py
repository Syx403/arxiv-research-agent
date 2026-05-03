from __future__ import annotations

import httpx

from src.core.config import get_settings


async def fetch_readme(repo_full_name: str) -> str | None:
    settings = get_settings()
    headers = {}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token.get_secret_value()}"
    url = f"https://raw.githubusercontent.com/{repo_full_name}/HEAD/README.md"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url, headers=headers or None)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text
