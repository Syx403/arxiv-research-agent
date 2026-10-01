"""Real PostgreSQL, synthetic isolated corpus; no external API calls."""
import os
from uuid import uuid4

import pytest

from src.core import db
from src.retrieval.index.hybrid_search import hybrid_search

pytestmark = pytest.mark.skipif(os.getenv('INTEGRATION_TESTS') != '1', reason='Requires local Postgres')


@pytest.mark.asyncio
async def test_hybrid_recall_keeps_original_paper_with_verbose_query_and_metadata():
    pid='test-evidence-'+str(uuid4())
    vector=[1.0]+[0.0]*1535
    try:
        async with db.acquire_app() as conn:
            await conn.execute("INSERT INTO papers (paper_id,source,title,authors,ingestion_status) VALUES ($1,'arxiv','FixtureAgent: Stable Memory','[]','complete')",pid)
            await conn.execute("INSERT INTO chunks (paper_id,section,ord,text,token_count,embedding) VALUES ($1,'Method',0,'FixtureAgent writes memory after every trial.',9,$2)",pid,vector)
        hits=await hybrid_search('FixtureAgent unfamiliar words mechanism workflow nonexistentterm',vector,
                                 filter_paper_ids=[pid],original_query='Explain FixtureAgent memory management')
        assert len(hits)==1
        assert hits[0].paper_id==pid
        assert hits[0].title=='FixtureAgent: Stable Memory'
        assert hits[0].section=='Method'
    finally:
        async with db.acquire_app() as conn:
            await conn.execute('DELETE FROM chunks WHERE paper_id=$1',pid)
            await conn.execute('DELETE FROM papers WHERE paper_id=$1',pid)
        await db.close_pools()
