CREATE TABLE IF NOT EXISTS session_summaries (
  session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  up_to_message_id    BIGINT NOT NULL,
  summary             TEXT NOT NULL,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (session_id, up_to_message_id)
);
