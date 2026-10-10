-- 0008: running spend totals (D39). A reservation used to sum the whole ledger under the global
-- advisory lock, once per model call; now it reads three rows. A trigger keeps them equal to the
-- sums of charge_usd ("total", "run:<id>", "turn:<id>"), so no code path can forget to update
-- them; the ledger stays the record and the totals can be rebuilt from it.
CREATE TABLE spend_totals (
    key text PRIMARY KEY,
    usd numeric(16, 8) NOT NULL
);

INSERT INTO spend_totals (key, usd)
SELECT 'total', coalesce(sum(charge_usd), 0) FROM llm_calls
UNION ALL
SELECT 'run:' || run_id, sum(charge_usd) FROM llm_calls WHERE run_id IS NOT NULL GROUP BY run_id
UNION ALL
SELECT 'turn:' || turn_id, sum(charge_usd) FROM llm_calls WHERE turn_id IS NOT NULL
GROUP BY turn_id;

CREATE FUNCTION spend_add(run text, turn text, usd numeric) RETURNS void LANGUAGE sql AS $$
    INSERT INTO spend_totals (key, usd)
    SELECT k, usd FROM unnest(ARRAY[
        'total',
        CASE WHEN run IS NOT NULL THEN 'run:' || run END,
        CASE WHEN turn IS NOT NULL THEN 'turn:' || turn END
    ]) AS k WHERE k IS NOT NULL
    ON CONFLICT (key) DO UPDATE SET usd = spend_totals.usd + EXCLUDED.usd;
$$;

CREATE FUNCTION spend_follow() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP IN ('UPDATE', 'DELETE') THEN
        PERFORM spend_add(OLD.run_id, OLD.turn_id, -OLD.charge_usd);
    END IF;
    IF TG_OP IN ('INSERT', 'UPDATE') THEN
        PERFORM spend_add(NEW.run_id, NEW.turn_id, NEW.charge_usd);
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER llm_calls_spend AFTER INSERT OR UPDATE OR DELETE ON llm_calls
FOR EACH ROW EXECUTE FUNCTION spend_follow();
