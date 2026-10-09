-- 0006: papers written for evaluation (M5a, S7): the prompt-injection document lives only in the
-- local database, with its own source, so it is never mistaken for an arXiv or QASPER paper.
ALTER TABLE papers DROP CONSTRAINT papers_source_check;
ALTER TABLE papers ADD CONSTRAINT papers_source_check
    CHECK (source IN ('arxiv', 'qasper', 'synthetic'));
