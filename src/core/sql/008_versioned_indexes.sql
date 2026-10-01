-- Keep historical chunk IDs immutable when a parser or embedding model changes.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS index_key TEXT NOT NULL DEFAULT 'legacy';
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_paper_id_ord_key;
CREATE UNIQUE INDEX IF NOT EXISTS chunks_edition_index_ord ON chunks(paper_id, index_key, ord);
