-- 0001: extensions used from M1 on, and the LLM cost ledger (DESIGN §6.5).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

-- One row per billable request. It is inserted as 'reserved' (an upper bound) before the request
-- is sent, then becomes 'settled' (reported usage and cost) or 'released' (the provider rejected
-- the request, so nothing was billed). A request whose outcome is unknown stays 'reserved'.
CREATE TABLE llm_calls (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at         timestamptz NOT NULL DEFAULT now(),
    settled_at         timestamptz,
    status             text NOT NULL DEFAULT 'reserved'
                       CHECK (status IN ('reserved', 'settled', 'released')),
    stage              text NOT NULL,
    model              text NOT NULL,
    prompt_version     text NOT NULL,
    run_id             text,
    turn_id            text,
    reserved_usd       numeric(12, 8) NOT NULL CHECK (reserved_usd >= 0),
    cost_usd           numeric(12, 8) CHECK (cost_usd >= 0),
    -- What the call counts against the budget caps.
    charge_usd         numeric(12, 8) GENERATED ALWAYS AS (
                           CASE status WHEN 'settled' THEN cost_usd
                                       WHEN 'released' THEN 0
                                       ELSE reserved_usd END
                       ) STORED,
    input_tokens       integer,
    cached_tokens      integer,
    cache_write_tokens integer,
    output_tokens      integer,
    reasoning_tokens   integer,
    latency_ms         integer,
    response_id        text,
    error              text
);

CREATE INDEX llm_calls_run_id ON llm_calls (run_id);
CREATE INDEX llm_calls_turn_id ON llm_calls (turn_id);
