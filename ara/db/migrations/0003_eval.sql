-- 0003: evaluation runs and per-item results (DESIGN §11.5). A run's id is also its ledger run_id.
CREATE TABLE eval_runs (
    id                   text PRIMARY KEY,
    suite                text NOT NULL,
    status               text NOT NULL DEFAULT 'running'
                         CHECK (status IN ('running', 'complete', 'partial', 'failed')),
    config               jsonb NOT NULL,       -- arms, pipeline version, manifest hash, caps
    langsmith_experiment text,
    created_at           timestamptz NOT NULL DEFAULT now(),
    finished_at          timestamptz
);

CREATE TABLE eval_results (
    run_id  text NOT NULL REFERENCES eval_runs (id),
    item_id text NOT NULL,
    arm     text NOT NULL,
    metrics jsonb NOT NULL,                    -- e.g. {"recall@8": 1.0, "mrr": 0.5}
    output  jsonb NOT NULL,                    -- what the arm returned, e.g. ranked paragraphs
    error   text,
    PRIMARY KEY (run_id, item_id, arm)
);
