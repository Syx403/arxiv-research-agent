"""The audit reads only active indexes of the explicitly requested versions."""
import os
from uuid import uuid4

import pytest

from src.core import db
from src.retrieval.self_rag.claim_audit import load_scope

pytestmark = pytest.mark.skipif(os.getenv("INTEGRATION_TESTS") != "1", reason="Requires local Postgres")


async def test_audit_excludes_stale_indexes_and_other_versions():
    base = "test-claim-audit-" + str(uuid4())
    identifiers = [base + "v1", base + "v2"]
    try:
        async with db.acquire_app() as conn:
            for pid in identifiers:
                await conn.execute("""INSERT INTO papers
                    (paper_id,source,title,authors,ingestion_status,raw_metadata)
                    VALUES ($1,'arxiv','Audit scope fixture','[]','complete','{"index_key":"active"}')""", pid)
                for index in ("old", "active"):
                    await conn.execute("""INSERT INTO chunks
                        (paper_id,section,ord,text,token_count,embedding,index_key)
                        VALUES ($1,'Appendix',0,$2,8,$3,$4)""",
                        pid, index + " text for " + pid, [0.0] * 1536, index)
        hits = await load_scope([identifiers[0]])
        assert len(hits) == 1
        assert hits[0].paper_id == identifiers[0]
        assert hits[0].text == "active text for " + identifiers[0]
        assert hits[0].section == "Appendix"
    finally:
        async with db.acquire_app() as conn:
            await conn.execute("DELETE FROM papers WHERE paper_id=ANY($1::text[])", identifiers)
        await db.close_pools()
