from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from uuid import uuid4

import pytest

from src.core import db
from src.core.types import (
    Citation,
    CitationVerdict,
    Decomposition,
    ReflectionReport,
    RouteDecision,
    SubQResult,
    SubQuestion,
    VerificationReport,
)
from src.graph import builder
from src.retrieval.index.types import Hit


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="resume integration test requires INTEGRATION_TESTS=1 and running Postgres",
)


@pytest.mark.asyncio
async def test_run_resumes_checkpoint_across_process(monkeypatch) -> None:
    thread_id = f"t-resume-{uuid4()}"
    _patch_builder_nodes(monkeypatch)
    try:
        result = await builder.run("What is ReAct?", thread_id=thread_id)
        expected = result["decomposition"].model_dump()
        assert await _checkpoint_exists(thread_id)

        script = _resume_script(thread_id)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            check=True,
            capture_output=True,
            text=True,
        )
        resumed = json.loads(completed.stdout.strip())
        assert resumed == expected
    finally:
        await _cleanup_thread(thread_id)
        await builder.shutdown()


def _patch_builder_nodes(monkeypatch) -> None:
    async def fake_decompose(state):
        return {
            "decomposition": Decomposition(sub_questions=[SubQuestion(text="What is ReAct?")]),
            "subq_results": {},
            "evidence": [],
            "multi_hop_calls": 0,
            "tool_iters": 0,
        }

    async def fake_retrieve(state):
        hit = _hit()
        return {
            "subq_results": {
                0: SubQResult(
                    subq_index=0,
                    hits=[hit],
                    sufficient=True,
                    route_decision=RouteDecision(),
                    retries_used=0,
                    multi_hop_used=False,
                )
            },
            "evidence": [hit],
            "active_subq_index": None,
        }

    async def fake_synthesize(state):
        return {
            "answer": "ReAct interleaves reasoning and acting [arxiv:2210.03629#1].",
            "citations": [Citation(paper_id="arxiv:2210.03629", chunk_id=1, claim_span=(0, 39))],
            "evidence_lookup": {"arxiv:2210.03629#1": "ReAct interleaves reasoning and acting."},
        }

    async def fake_self_rag(state):
        citation = state["citations"][0]
        return {
            "verification": VerificationReport(
                verdicts=[CitationVerdict(citation=citation, supports=True, rationale="supported")],
                passed=True,
            )
        }

    async def fake_reflect(state):
        return {"reflection": ReflectionReport(concepts_reused=1)}

    monkeypatch.setattr(builder, "_graph", None)
    monkeypatch.setattr(builder, "decompose_node", fake_decompose)
    monkeypatch.setattr(builder, "retrieve_node", fake_retrieve)
    monkeypatch.setattr(builder, "synthesize_node", fake_synthesize)
    monkeypatch.setattr(builder, "self_rag_node", fake_self_rag)
    monkeypatch.setattr(builder, "reflect_node", fake_reflect)


def _resume_script(thread_id: str) -> str:
    return textwrap.dedent(
        f"""
        import asyncio
        import json
        from src.graph import builder

        async def fail_node(state):
            raise AssertionError("resume should not execute graph nodes")

        async def main():
            builder.decompose_node = fail_node
            builder.retrieve_node = fail_node
            builder.synthesize_node = fail_node
            builder.self_rag_node = fail_node
            builder.reflect_node = fail_node
            result = await builder.run("", thread_id={thread_id!r})
            print(json.dumps(result["decomposition"].model_dump(), sort_keys=True))
            await builder.shutdown()

        asyncio.run(main())
        """
    )


async def _cleanup_thread(thread_id: str) -> None:
    async with db.acquire_app() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM sessions WHERE session_id = $1", thread_id)
            await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = $1", thread_id)
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = $1", thread_id)
            await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = $1", thread_id)


async def _checkpoint_exists(thread_id: str) -> bool:
    async with db.acquire_app() as conn:
        return bool(
            await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM checkpoints WHERE thread_id = $1)",
                thread_id,
            )
        )


def _hit() -> Hit:
    return Hit(
        chunk_id=1,
        paper_id="arxiv:2210.03629",
        section="Introduction",
        text="ReAct interleaves reasoning and acting.",
        score=1.0,
    )
