-- 0002: papers, their parsed documents and searchable chunks (DESIGN §5, §8).

-- One row per paper edition we can read: an arXiv version ("arxiv:2210.03629v3") or a dataset's
-- own full text ("qasper:1912.01214").
CREATE TABLE papers (
    id         text PRIMARY KEY,
    source     text NOT NULL CHECK (source IN ('arxiv', 'qasper')),
    arxiv_id   text NOT NULL,
    version    integer,
    title      text NOT NULL,
    abstract   text NOT NULL DEFAULT '',
    metadata   jsonb NOT NULL DEFAULT '{}'
);

-- One parsed edition per (paper, PIPELINE_VERSION), written with its chunks in one transaction,
-- so a row exists only for a complete document.
CREATE TABLE documents (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    paper_id         text NOT NULL REFERENCES papers (id),
    pipeline_version text NOT NULL,
    format           text NOT NULL CHECK (format IN ('html', 'pdf', 'qasper')),
    created_at       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (paper_id, pipeline_version)
);

CREATE TABLE chunks (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ord          integer NOT NULL,
    paragraph    integer NOT NULL,          -- index of the source paragraph in the document
    heading_path text NOT NULL,             -- "Title › Section › Subsection"
    text         text NOT NULL,             -- shown to models and users
    search_text  text NOT NULL,             -- heading path + text, for BM25 and embeddings
    sentences    integer[] NOT NULL,        -- start offset of each sentence in `text`
    embedding    vector(1536) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    UNIQUE (document_id, ord)
);

CREATE INDEX chunks_document_id ON chunks (document_id);
CREATE INDEX chunks_tsv ON chunks USING gin (tsv);
-- BM25 with two tokenizers on the same text, compared in S1 (D15): plain words, and English
-- stemming under the alias `search_text_stemmed`. document_id is indexed for filter pushdown.
CREATE INDEX chunks_bm25 ON chunks USING paradedb (
    id,
    document_id,
    (search_text::pdb.unicode_words),
    (search_text::pdb.unicode_words('stemmer=english', 'alias=search_text_stemmed'))
);

-- Content-addressed embeddings: sha256 of model + text. Re-ingesting unchanged text is free.
CREATE TABLE embedding_cache (
    key        bytea PRIMARY KEY,
    model      text NOT NULL,
    embedding  vector(1536) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
