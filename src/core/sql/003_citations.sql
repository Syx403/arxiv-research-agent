CREATE TABLE IF NOT EXISTS citations (
  citing_paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  cited_paper_id  TEXT NOT NULL,         -- not FK; cited paper may not be in our DB yet
  cited_title     TEXT,
  evidence_chunk  BIGINT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
  PRIMARY KEY (citing_paper_id, cited_paper_id)
);
CREATE INDEX IF NOT EXISTS citations_cited_idx ON citations(cited_paper_id);
