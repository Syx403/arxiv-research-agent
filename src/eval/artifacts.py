"""Bind evaluation artifacts to the code, configuration and corpus actually used."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

from src.core import db
from src.llm.registry import REGISTRY, get_route

ROOT = Path(__file__).resolve().parents[2]


def code_fingerprint() -> str:
    digest = sha256()
    files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py"), ROOT / "pyproject.toml", ROOT / "uv.lock"])
    for path in files:
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


async def capture_corpus() -> dict:
    snapshots = {}
    async with db.acquire_app() as conn:
        # Fixed local tables only. Includes embeddings and graph data, not any
        # credential, session content, or connection settings.
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            for table in ("papers", "chunks", "citations", "concepts", "concept_aliases", "concept_relations"):
                row = await conn.fetchrow(
                    "SELECT count(*) AS count, md5(COALESCE(string_agg(h, '' ORDER BY h), '')) AS hash "
                    f"FROM (SELECT md5(to_jsonb(t)::text) AS h FROM {table} t) rows"
                )
                snapshots[table] = dict(row)
    return snapshots


def make_manifest(*, cases: list, corpus: dict, mode: str) -> dict:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return {
        "schema_version": 1, "mode": mode, "git_revision": revision,
        "working_code_sha256": code_fingerprint(),
        "profiles": {role: asdict(get_route(role)) for role in REGISTRY},
        "cases": cases, "corpus": corpus,
        "runtime_overrides": {name: os.environ[name] for name in (
            "MULTI_HOP_MAX_ATTEMPTS", "MULTI_HOP_MAX_EXPANSIONS", "MULTI_HOP_WALL_CLOCK_BUDGET_S",
        ) if name in os.environ},
        "cost_basis": "2026-09-14 Flash; role usage uses peak rates; Cohere Trial; request ledger records request-time estimates and uncertain charges",
        "remote_export": os.getenv("ARA_EVAL_LANGSMITH_EXPORT", "false").lower() == "true",
    }


def ensure_manifest(directory: Path, manifest: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Evaluation inputs changed (code/model/corpus/cases). Choose a new output directory.")
    elif (directory / "per_question.csv").exists():
        raise ValueError("Legacy results have no version manifest; choose a new output directory.")
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
