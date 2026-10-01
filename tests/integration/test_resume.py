from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from uuid import uuid4

import pytest

from src.core import db
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
        assert result["status"] == "complete"
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


@pytest.mark.asyncio
async def test_native_fields_resume_across_process_and_new_turn_clears(monkeypatch):
    from src.core.run_context import RunContext, use_run_context
    from tests.unit.test_native_structured_answer import FIELD, fixture_state
    thread_id = f"t-native-resume-{uuid4()}"
    _patch_builder_nodes(monkeypatch)

    async def structured_fixture(state):
        saved = fixture_state()
        return {k: saved[k] for k in ("answer", "answer_fields", "citations", "evidence_lookup")}

    monkeypatch.setattr(builder, "synthesize_node", structured_fixture)
    try:
        with use_run_context(RunContext(response_fields=(FIELD,))):
            result = await builder.run("Count sensors with original-source evidence.", thread_id=thread_id)
        assert result["status"] == "complete" and len(result["answer_fields"]) == 1
        expected = {k: result[k] for k in ("answer_fields", "response_fields")}
        script = _resume_script(thread_id).replace(
            'result["decomposition"].model_dump()',
            '{k: result[k] for k in ("answer_fields", "response_fields")}')
        completed = subprocess.run([sys.executable, "-c", script], cwd=os.getcwd(), check=True,
                                   capture_output=True, text=True)
        assert json.loads(completed.stdout.strip()) == expected
        # Reset node patches to a natural response and use the same DB thread.
        _patch_builder_nodes(monkeypatch)
        from src.graph.nodes.synthesize import synthesize_node
        monkeypatch.setattr(builder, "synthesize_node", synthesize_node)
        following = await builder.run("What is ReAct?", thread_id=thread_id)
        assert following["answer_fields"] == following["response_fields"] == []
    finally:
        await _cleanup_thread(thread_id)
        await builder.shutdown()


@pytest.mark.asyncio
async def test_verified_claim_context_survives_checkpoint_resume(monkeypatch):
    from src.core.types import CitationVerdict, VerificationReport
    from src.graph.nodes.synthesize import _parse_citations
    from tests.unit.test_claim_context import ANSWER, LOOKUP
    thread_id = f"t-claim-context-{uuid4()}"
    _patch_builder_nodes(monkeypatch)
    citations = _parse_citations(ANSWER)
    claims = list(dict.fromkeys(c.claim_text for c in citations))
    report = VerificationReport(passed=True, verdicts=[CitationVerdict(
        citation=c, supports=True, rationale="fixture", claim_kind="paper_fact",
        context_claims=claims[:claims.index(c.claim_text)], context_verified=True) for c in citations])

    async def generation(state):
        return {"answer": ANSWER, "citations": citations, "evidence_lookup": LOOKUP}

    async def verification(state):
        return {"verification": report}

    monkeypatch.setattr(builder, "synthesize_node", generation)
    monkeypatch.setattr(builder, "self_rag_node", verification)
    try:
        result = await builder.run("Compare mechanisms.", thread_id=thread_id)
        expected = result["verification"].model_dump(mode="json")
        assert result["status"] == "complete" and expected["verdicts"][-1]["context_claims"]
        script = _resume_script(thread_id).replace(
            'result["decomposition"].model_dump()', 'result["verification"].model_dump(mode="json")')
        completed = subprocess.run([sys.executable, "-c", script], cwd=os.getcwd(),
                                   check=True, capture_output=True, text=True)
        assert json.loads(completed.stdout.strip()) == expected
    finally:
        await _cleanup_thread(thread_id)
        await builder.shutdown()


def _patch_builder_nodes(monkeypatch) -> None:
    from tests.unit.test_agent_delivery import _wire
    _wire(monkeypatch)
    monkeypatch.setattr(builder, "_graph", None)


def _resume_script(thread_id: str) -> str:
    return textwrap.dedent(
        f"""
        import asyncio
        import json
        from src.graph import builder

        async def fail_node(state):
            raise AssertionError("resume should not execute graph nodes")

        async def main():
            builder.plan_papers_node = fail_node
            builder.retrieve_node = fail_node
            builder.synthesize_node = fail_node
            builder.self_rag_node = fail_node
            builder.control_research_node = fail_node
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
