-- 0005: a paper's first-submission date (D30). Library candidates join discovery, which filters
-- and orders by date, and read cards show it. Written by ingestion from arXiv metadata; rows
-- stored before 0005 are filled once by `ara db backfill` (one free arXiv lookup). QASPER papers
-- have no date.
ALTER TABLE papers ADD COLUMN published date;
