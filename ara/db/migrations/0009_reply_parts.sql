-- 0009: a recorded turn keeps its reply in parts (D39), so the UI renders the listing and the
-- answer from the turn's data instead of parsing the reply text. Turns recorded before have none.
ALTER TABLE turns ADD COLUMN parts jsonb NOT NULL DEFAULT '[]';
