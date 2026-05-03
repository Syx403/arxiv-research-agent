CREATE TABLE IF NOT EXISTS papers (
  paper_id      TEXT PRIMARY KEY,        -- e.g. arxiv:2210.03629
  source        TEXT NOT NULL,           -- 'arxiv' | 'semantic_scholar' | 'github'
  title         TEXT NOT NULL,
  authors       JSONB NOT NULL DEFAULT '[]',
  abstract      TEXT,
  published_at  DATE,
  url           TEXT,
  pdf_path      TEXT,                    -- local cache path
  ingestion_status TEXT NOT NULL DEFAULT 'complete', -- 'complete' | 'in_progress' | 'failed'
  raw_metadata  JSONB NOT NULL DEFAULT '{}',
  ingested_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS papers_source_idx ON papers(source);
CREATE INDEX IF NOT EXISTS papers_published_idx ON papers(published_at DESC);
