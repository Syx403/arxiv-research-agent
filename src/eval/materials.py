"""Frozen, explicitly bounded paper catalog backed by checked local originals.

This measures agent decisions with fixed materials, NOT open arXiv recall.
No ingestion, embedding rebuild or public cache deletion occurs in replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from src.core import db
from src.core.trace import record_event
from src.corpus.access import Acquisition
from src.corpus.arxiv_client import GLOBAL_ARXIV_CLIENT, PaperMetadata, normalize_arxiv_id
from src.corpus.ingestion import IngestResult


async def index_fingerprints(ids: list[str]) -> dict:
    async with db.acquire_app() as conn:
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            rows = await conn.fetch(
                """
                SELECT p.paper_id, p.ingestion_status, p.raw_metadata->>'index_key' AS index_key,
                       p.raw_metadata->>'material_sha256' AS source_sha256,
                       (SELECT count(*) FROM chunks c WHERE c.paper_id=p.paper_id) AS chunks,
                       (SELECT md5(COALESCE(string_agg(md5(to_jsonb(c)::text), '' ORDER BY c.chunk_id), ''))
                        FROM chunks c WHERE c.paper_id=p.paper_id) AS index_digest
                FROM papers p WHERE p.paper_id=ANY($1::text[])
            """,
                ids,
            )
    return {r["paper_id"]: dict(r) for r in rows}


async def capture_materials(suite: dict, case_ids: list[str]) -> dict:
    refs = {r for c in suite["cases"] if c["id"] in case_ids for r in c["source_refs"]}
    # A contrast version used only in the grader must not be injected as an agent candidate.
    refs.discard("agentless_v2")
    catalog = []
    for ref in sorted(refs):
        source = suite["sources"][ref]
        meta = await GLOBAL_ARXIV_CLIENT.fetch_metadata(source["version_id"], fresh=False)
        row = json.loads(json.dumps(asdict(meta), default=str))
        if row["raw"].get("version_id") != source["version_id"]:
            raise ValueError("Material version differs from the label source")
        catalog.append(row)
    ids = ["arxiv:" + p["raw"]["version_id"] for p in catalog]
    fingerprints = await index_fingerprints(ids)
    missing = [
        pid
        for pid in ids
        if fingerprints.get(pid, {}).get("ingestion_status") != "complete"
        or not fingerprints.get(pid, {}).get("chunks")
    ]
    if missing:
        raise ValueError(
            "Originals must be acquired before the controlled run: " + ", ".join(missing)
        )
    return {
        "schema_version": 1,
        "mode": "fixed_catalog_and_versioned_index",
        "scope": "bounded material quality, not live discovery",
        "catalog": catalog,
        "indexes": fingerprints,
    }


class FrozenMaterials:
    def __init__(self, snapshot: dict, *, paper_ids: list[str] | None = None, unavailable=False):
        self.snapshot = snapshot
        allowed = {normalize_arxiv_id(pid) for pid in paper_ids} if paper_ids else None
        self.catalog = [
            PaperMetadata(**row)
            for row in snapshot["catalog"]
            if allowed is None or normalize_arxiv_id(row["paper_id"]) in allowed
        ]
        self.unavailable = unavailable

    def adapter(self) -> Acquisition:
        return Acquisition(self, self.ingest)

    async def validate(self):
        expected = self.snapshot["indexes"]
        if await index_fingerprints(list(expected)) != expected:
            raise ValueError("Frozen original/index changed; use a new experiment")

    async def fetch_metadata(self, identifier, **kwargs):
        target = normalize_arxiv_id(identifier, keep_version=True)
        for paper in self.catalog:
            if target in {normalize_arxiv_id(paper.paper_id), paper.raw["version_id"]}:
                return paper
        raise ValueError("Requested identity is outside the frozen catalog")

    async def search(
        self, query, *, limit=12, offset=0, date_from=None, date_to=None, latest=False, **kwargs
    ):
        # Deliberately supply the complete declared pool, not a pretend search engine.
        candidates = [
            p
            for p in self.catalog
            if (not date_from or str(p.published_at) >= date_from)
            and (not date_to or str(p.published_at) <= date_to)
        ]
        if latest:
            candidates.sort(key=lambda p: str(p.published_at), reverse=True)
        record_event(
            "fixed_catalog",
            query=query,
            candidates=len(candidates),
            scope="all declared candidates; lexical search quality not measured",
        )
        return candidates[offset : offset + limit]

    async def ingest(self, identifier, **kwargs):
        if self.unavailable:
            raise ValueError("Injected original-unavailable fixture")
        paper = await self.fetch_metadata(identifier)
        pid = "arxiv:" + paper.raw["version_id"]
        expected = self.snapshot["indexes"].get(pid)
        if not expected or expected["ingestion_status"] != "complete":
            raise ValueError("No complete original in the frozen index")
        record_event(
            "frozen_original",
            paper_id=pid,
            **{k: expected[k] for k in ("source_sha256", "index_digest", "index_key")},
        )
        return IngestResult(paper_id=pid, status="skipped")


def material_digest(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
