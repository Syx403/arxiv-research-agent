-- 0007: the app's conversations and their turns (D36). A conversation is a LangGraph thread (same
-- id); a turn row is written by the API when a turn (or its first half, up to a clarifying
-- question) ends, with what the UI shows again later: the reply, the answer and its evidence, the
-- papers, the problems, and the nodes the turn went through. Model calls stay in llm_calls,
-- joined on turn_id. Deleting a conversation deletes its turns (and the API deletes its thread).
CREATE TABLE conversations (
    id         text PRIMARY KEY,
    user_id    text NOT NULL,
    title      text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX conversations_recent ON conversations (user_id, updated_at DESC);

CREATE TABLE turns (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    turn_id         text NOT NULL UNIQUE,
    message         text NOT NULL,
    reply           text NOT NULL,
    status          text NOT NULL,
    intent          text,
    answer          jsonb,
    papers          jsonb NOT NULL,
    read            jsonb NOT NULL,
    problems        jsonb NOT NULL,
    trace           jsonb NOT NULL,
    waiting         jsonb,
    started_at      timestamptz NOT NULL,
    finished_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX turns_by_conversation ON turns (conversation_id, id);
