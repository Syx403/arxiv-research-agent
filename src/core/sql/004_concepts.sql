CREATE TABLE IF NOT EXISTS concepts (
  concept_id   BIGSERIAL PRIMARY KEY,
  canonical    TEXT NOT NULL UNIQUE,
  display      TEXT NOT NULL,
  definition   TEXT,
  embedding    vector(1536),
  evidence_chunks JSONB NOT NULL DEFAULT '[]',
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS concept_aliases (
  alias        TEXT PRIMARY KEY,
  concept_id   BIGINT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS concept_aliases_concept_idx ON concept_aliases(concept_id);

CREATE TABLE IF NOT EXISTS concept_relations (
  src_id       BIGINT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
  dst_id       BIGINT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
  rel_type     TEXT NOT NULL,            -- 'extends' | 'compares' | 'uses' | 'cites' | 'is-a'
  evidence_chunks JSONB NOT NULL DEFAULT '[]',
  weight       REAL NOT NULL DEFAULT 1.0,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (src_id, dst_id, rel_type)
);
CREATE INDEX IF NOT EXISTS concept_relations_dst_idx ON concept_relations(dst_id);
