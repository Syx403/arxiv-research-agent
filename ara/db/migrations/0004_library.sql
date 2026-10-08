-- 0004: the user's library (DESIGN §7, §8; M4, D27). A row per paper a user has read; reading it
-- again only moves `last_read_at`. Single user for now: user_id is "local" in the product.
CREATE TABLE library_items (
    user_id       text NOT NULL,
    paper_id      text NOT NULL REFERENCES papers (id),
    status        text NOT NULL DEFAULT 'read' CHECK (status IN ('read', 'saved')),
    first_read_at timestamptz NOT NULL DEFAULT now(),
    last_read_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, paper_id)
);

-- Library-wide dense search (DESIGN §5): HNSW with pgvector iterative scans, so a filter on the
-- user's documents still returns enough rows. Scoped reads of up to 3 papers keep the exact scan.
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
