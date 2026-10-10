# S8 open-questions dataset: brief for the building agent

You are building one evaluation dataset for ARA, a research agent that finds arXiv papers and answers
questions from their full text, for English- and Chinese-speaking researchers. The suite is S8: the
open questions real users ask most. Your job is to read large existing datasets, measure what real
users ask, and select and convert real questions. You do not invent questions.

Work in the repository at `/Users/richsion/Desktop/arxiv-research-agent` (branch `v2`). Today is
2026-10-11. Finish every phase, then stop and report. Do not ask questions mid-way: when something
is unclear, choose the conservative option and record it in the selection log.

## Hard rules

1. Write only these paths:
   - `evals/datasets/s8_open.json` (the dataset)
   - `docs/v2/eval/s8-selection.md` (your selection log and census)
   - `data/datasets/s8_sources/<source>/` (raw downloads, one folder per source; git-ignored)
   Do not edit any other file. Do not run `git add`, `git commit` or `git push`.
2. Do not run `ara` commands, the app, live tests, or anything in this repository that calls an
   LLM, embedding or rerank API. Do not open or print `.env`. Do not touch `data/datasets/pasa/`.
3. Network: arXiv (export API `http://export.arxiv.org/api/query` and `https://arxiv.org/html/…`;
   at most one request every 3 seconds), Hugging Face and GitHub only. No Semantic Scholar, no
   OpenAlex, no search engines.
4. Download only the files named below. Every source folder gets a `README.md`: URL, revision or
   commit, licence as the source states it, date fetched, sha256 of each file kept.
5. Every fact you write (queries, arXiv ids, versions, dates, quotes, answers) must come from a file
   or page you fetched in this session. If you cannot verify something, drop the candidate and log
   why.
6. Use only sources whose licence allows redistribution with attribution (ODC-BY, CC BY, MIT,
   Apache-2.0). If a source is non-commercial or share-alike (for example PeerQA, CC BY-NC-SA) or
   states no licence for its data, do not use it; log that you skipped it.

## The five types

| type | the user wants | ARA path | what grades it |
|---|---|---|---|
| `survey_named` | papers about one named model, method, system or benchmark ("papers on Kimi K3", "work building on FlashAttention-3") | find | the subject's own paper listed first; the listed papers' relevance |
| `explain_paper` | an in-depth explanation of one named paper's method, architecture, results | read | rubric points with evidence quotes |
| `compare_papers` | two named papers compared on an aspect | read | rubric points with evidence quotes from both |
| `topic_survey` | an overview of approaches to a research question | find, then read | rubric |
| `latest_topic` | the newest work on a topic | find | recency and relevance of what is listed |

## Phase 1: census of real queries (Asta Interaction Dataset)

Source: Hugging Face `allenai/asta-user-interactions`, subset `optin_queries` only (about 259k rows;
fields `query`, `thread_id`, `query_ts`, `tool` = `pf` for paper finding or `sqa` for question
answering). Licence on the card: ODC-BY (the paper, arXiv 2602.23335, says CC BY 4.0; record both).
Do not download the other subsets.

1. Keep the first query of each thread, drop exact and near-exact duplicates, queries under three
   words, and anything that looks like personal information.
2. Count languages (by script: Latin, CJK, other). Report how many Chinese queries there are.
3. Draw a random sample of 3,000 (seed 20261011; say how you sampled) and classify each into one
   of the five types or `other` (factoid lookup, a single known paper by title, non-research,
   outside computer science, too vague). Record your classification rules first, then apply them
   consistently. Report counts and shares per type, separately for `pf` and `sqa`, and 3 example
   queries per type.
4. From these shares, set the number of S8 items per type: 40 items in all, proportional to the
   computer-science share of each type, with at least 4 and at most 12 per type, even numbers.
   Write the arithmetic in the log.

## Phase 2: candidates per type

For each type collect about three times as many candidates as items needed, then select.

**Real queries (Asta; also SciArena).** `survey_named`, `latest_topic`, `topic_survey` and, where
they exist, `compare_papers` and `explain_paper` items come from real queries first. SciArena
(Hugging Face `yale-nlp/SciArena`, MIT; field `question`, plus `subject`) is a second source of real
questions for `topic_survey` and `compare_papers`. A real query is kept verbatim as `message_en`
(fix nothing, not even typos). A query is suitable when:
- it is about computer science (machine learning, NLP, IR, systems for ML, agents, vision);
- it can be served from arXiv papers alone;
- for `survey_named`: it names one subject that has its own arXiv paper or report. Find the
  primary paper with the arXiv API (record the query you used) and check that at least eight other
  arXiv papers name the subject in title or abstract (`search_query=abs:"<name>"`; record the count
  and the date);
- for `explain_paper` / `compare_papers`: the paper(s) it names exist on arXiv with an HTML
  version (`https://arxiv.org/html/<id>v<n>` loads);
- for `latest_topic`: the topic has many arXiv papers in the last 180 days (check with the arXiv
  API sorted by submission date; record what you saw).

**Questions written for datasets (when real queries do not give enough items of a type).**
- `explain_paper`: QASA (GitHub `lgresearch/QASA`; check the licence covers the data) and SciDQA
  (Hugging Face `yale-nlp/SciDQA`, ODC-BY). Prefer "how" and "why" questions about a paper's
  method, design or results that the text answers without figures or tables, on papers that have
  an arXiv HTML version. Keep the dataset's answer verbatim in `reference_answer`.
- `topic_survey`: ScholarQA-CS2 (in the AstaBench release, `github.com/allenai/asta-bench`; find
  it and check its licence), else ScholarQA-CS (`github.com/AkariAsai/ScholarQABench`,
  `data/scholarqa_cs/`, ODC-BY), else ResearchQA (`realliyifei/ResearchQA`, MIT, `test` split
  only). Keep their rubric items verbatim.
- `compare_papers`: M3SciQA only if its licence allows redistribution and the question needs
  text only (not figures).

## Phase 3: rubrics and evidence

- `topic_survey`: the source's rubric verbatim (4–8 items; if more than eight, keep the eight most
  central and say which you dropped in `notes`); `source.rubric_from` = `dataset`. No evidence.
- `explain_paper` (3–8 items) and `compare_papers` (4–8 items, covering both papers and at least
  one difference): if the source has a reference answer, split it into its checkable points
  (`rubric_from` = `reference_answer`). For a real query without an answer, write the points a good
  answer must cover from the paper itself (`rubric_from` = `paper`), each a single checkable
  statement that its evidence quotes support. Each point gets one to three evidence quotes:
  `paper` (versioned id), `section` (the heading path as the HTML shows it, joined with " › ",
  without numbers), `quote` (one or more whole sentences copied exactly from that version's HTML;
  avoid mathematical notation, footnote marks and bracketed citations; never a figure or table
  caption).
- `survey_named` and `latest_topic`: no rubric (`rubric_from` = `none`).

## Phase 4: languages and splits

- Half of each type's items are Chinese. Use real Chinese queries first (from the census). If
  there are not enough, translate selected real English queries into the Chinese a researcher would
  type (English technical terms kept), set `source.translated` = true, and keep the English
  original in `message_en`. Never translate a Chinese query into a different request.
- Half of each type's items are `dev`, half `test`, with both languages in each split.
- These were used while tuning ARA, so they go to `dev` only: Kimi K3 (2607.24653), ReWOO
  (2305.18323), LLMCompiler (2312.04511), ReAct (2210.03629), and the topics KV-cache eviction,
  tool-call latency in agents, speculative decoding, LLM serving cost, agentic RAG, backtest
  engines, diffusion models for image generation.

## Phase 5: the file

`evals/datasets/s8_open.json`, UTF-8, two-space indentation:

```json
{
  "suite": "s8",
  "created": "2026-10-11",
  "sources": [
    {"name": "Asta Interaction Dataset", "url": "https://huggingface.co/datasets/allenai/asta-user-interactions",
     "license": "ODC-BY", "citation": "Ai2, arXiv:2602.23335"}
  ],
  "items": [
    {
      "id": "s8-explain-qasa-123",
      "type": "explain_paper",
      "split": "test",
      "language": "Chinese",
      "message": "<the Chinese request>",
      "message_en": "<the source question, verbatim>",
      "source": {"dataset": "QASA", "id": "123", "license": "MIT", "origin": "dataset_question",
                 "translated": true, "rubric_from": "reference_answer"},
      "papers": ["2106.09685v2"],
      "names": [],
      "rubric": [
        {"id": "r1", "item": "<one checkable point of the reference answer>", "kind": "mechanism",
         "evidence": [{"paper": "2106.09685v2", "section": "<heading path>", "quote": "<exact sentences>"}]}
      ],
      "reference_answer": "<the dataset's answer, verbatim>",
      "expect": {"intent": "read", "primary": null, "recent_days": null},
      "reviewed": false,
      "notes": ""
    }
  ]
}
```

(Placeholders in angle brackets; the paper id is only a format example.) Field rules:
- `id`: `s8-<type word>-<source>-<source id or slug>`, unique, lowercase.
- `source.origin`: `real_query` for Asta and SciArena, `dataset_question` for the others.
- `source.id`: the source's own id; for Asta, the hashed `thread_id` and `query_ts`.
- `papers`: the versioned arXiv ids the request names; for `survey_named`, the primary paper.
- `names`: `survey_named` only, the subject as the query writes it.
- `kind` of a rubric point: definition, mechanism, result, comparison, limitation, example, other.
- `expect.intent`: `discover` (survey_named, latest_topic), `discover_read` (topic_survey),
  `read` (explain_paper, compare_papers); `expect.primary` (survey_named: bare arXiv id);
  `expect.recent_days` (latest_topic: 180).
- `reviewed` is always `false`. English items: `message_en` equals `message`.
- `sources` lists every source used, with its licence and citation.

Validate with the repository's checker until it prints `OK: 40 items` (it checks the schema,
the balance of splits and languages per type, and fetches every quoted paper from arXiv to confirm
each quote word for word):

```bash
uv run python -m evals.suites.s8
```

## Phase 6: the log

`docs/v2/eval/s8-selection.md`, in English:
1. Sources: URL, revision, licence, files and sha256; sources skipped and why.
2. The census: filtering counts, language counts, the classification rules, counts and shares per
   type for `pf` and `sqa`, examples, and the arithmetic for items per type.
3. Per type: the candidates considered (one line each: kept or dropped, and why).
4. Per kept item: what you verified (arXiv API queries and counts with dates, HTML pages quoted).
5. Uncertain choices and what you decided.
6. The checker's final output.

## When you finish

Reply with a short summary: items per type with split and language, how many are real queries,
translated or dataset questions, the census shares, anything skipped or uncertain, and the
checker's last line. Nothing else.
