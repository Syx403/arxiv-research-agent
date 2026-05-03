CREATE TABLE IF NOT EXISTS chunks (
  chunk_id     BIGSERIAL PRIMARY KEY,
  paper_id     TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  section      TEXT,
  ord          INT NOT NULL,
  text         TEXT NOT NULL,
  token_count  INT NOT NULL,
  embedding    vector(1536) NOT NULL,
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (paper_id, ord)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_paper_idx ON chunks(paper_id);
