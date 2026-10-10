# S8 open-questions dataset: brief for the building agent

You are building one evaluation dataset for ARA, a research agent that finds arXiv papers and answers
questions from their full text (English and Chinese users). The suite is called S8. It covers the
open questions real users ask most, which our other suites do not: research a named model or
method, explain a paper, compare two papers, survey a topic, find the latest work on a topic.

Work in the repository at `/Users/richsion/Desktop/arxiv-research-agent` (branch `v2`). Today is
2026-10-10. Finish every phase below, then stop and report. Do not ask questions mid-way: when
something is unclear, choose the conservative option and record it in the selection log.

## Hard rules

1. Write only these paths:
   - `evals/datasets/s8_open.json` (the dataset; committed later by the maintainer)
   - `docs/v2/eval/s8-selection.md` (your selection log)
   - `data/datasets/scholarqa_cs/` and `data/datasets/researchqa/` (raw downloads; git-ignored)
   Do not edit any other file. Do not run `git add`, `git commit` or `git push`.
2. Do not run `ara` commands, the app, tests marked live, or anything that calls an LLM,
   embedding or rerank API from this repository. Do not open or print `.env`. Do not touch
   `data/datasets/pasa/` (gated data).
3. Network: arXiv (export API `http://export.arxiv.org/api/query` and `https://arxiv.org/html/…`,
   at most one request every 3 seconds), Hugging Face, and GitHub only. No Semantic Scholar, no
   Google Scholar, no other search engine.
4. Keep downloads small: only the files named below, never the full ResearchQA split.
5. Every fact you write (arXiv ids, versions, dates, quotes) must come from a page or file you
   fetched in this session. Never fill a field from memory. If you cannot verify something, drop
   the candidate and log why.

## Phase 1: external sources

Download and record each source in a `README.md` inside its folder: source URL, git commit or
dataset revision, licence, date fetched, and the sha256 of every file kept.

- ScholarQA-CS (Ai2, part of ScholarQABench; data licence ODC-BY): repository
  `https://github.com/AkariAsai/ScholarQABench`, folder `data/scholarqa_cs/`. Keep the file
  that holds the questions with their rubrics (the README calls it `test_configs_snippets.json`)
  in `data/datasets/scholarqa_cs/`. Inspect its structure and write in the README which field
  holds the question, which holds the rubric ("ingredients"; the README of the source describes
  most-important and nice-to-have items), and how many questions it has.
- ResearchQA (licence MIT): Hugging Face dataset `realliyifei/ResearchQA`, split `test` only
  (Parquet), in `data/datasets/researchqa/`. Fields include `id`, `general_domain`, `subdomain`,
  `field`, `query`, `date`, `rubric` (4–8 items with `rubric_item`, `type`,
  `citation_metadata`). Record the domain and field values you find and how many test rows are
  computer science.

## Phase 2: choose the items

30 items, six per type. Within each type: three `dev` and three `test`, and three `English` and
three `Chinese`, spread so that each split has both languages. A Chinese item is a request a
Chinese-speaking researcher would actually type (mixing in English technical terms is normal, e.g.
"详细介绍一下 Mamba 的 selective scan 是怎么实现的"), not a word-for-word translation; its
`message_en` says the same request in English.

Topics and papers the maintainers already tuned on go to `dev` only, never `test`: Kimi K3
(2607.24653), ReWOO (2305.18323), LLMCompiler (2312.04511), ReAct (2210.03629), and the topics
KV-cache eviction, tool-call latency in agents, speculative decoding, LLM serving cost, agentic
RAG, backtest engines, diffusion models for image generation. Prefer computer-science topics in
machine learning, NLP, information retrieval, systems for ML, agents and computer vision.

### topic_survey (6): from the external sources

Three from ScholarQA-CS, three from ResearchQA `test` (computer-science fields only). Choose
questions that:
- ask for an overview, comparison or synthesis across several papers ("What approaches exist
  for …", "How do methods for … differ", "What are the open problems in …");
- can be answered from arXiv papers alone (machine learning, NLP, systems; not clinical,
  legal, or questions that need non-paper data);
- are specific enough to search (a researcher could name a few likely papers), and at most about
  40 words;
- have rubric items that are concrete and checkable.

Keep the question text verbatim as `message_en`. For the Chinese ones, write the Chinese `message`
yourself. Rubric: 4–8 items; from ScholarQA-CS use the most-important ingredients (if more than
eight, keep the eight most central and say which you dropped in `notes`); from ResearchQA use its
`rubric_item` texts verbatim. Map each item's kind to one of: definition, mechanism, result,
comparison, limitation, example, other (ResearchQA's `citation` type maps to `other`). Topic-survey
rubric items need no evidence quotes. `source` gives the dataset, its own id and the licence
(`ODC-BY-1.0` or `MIT`). `papers` is empty. `expect.intent` is `discover_read`.

### survey_named (6): authored

A user asks for papers about one named model, method, system or benchmark ("找 Kimi K3 相关论文",
"Find papers that build on FlashAttention-3"). Choose subjects that:
- have their own original arXiv paper or technical report (the primary paper);
- are named, in title or abstract, by at least eight other arXiv papers, which you check with the
  arXiv API (`search_query=abs:"<name>"`; record the count you saw and the date in `notes`);
- mix kinds: at least two models, two methods, one benchmark or dataset.

Fill `names` with the subject as the user writes it, `papers` with the primary paper's versioned
id (the latest version, from the arXiv API), `expect.primary` with its bare id, `expect.intent`
`discover`. No rubric. Kimi K3 may be one of the dev items.

### explain_paper (6): authored

A user asks to explain one named paper in depth ("Explain how LLMCompiler's planner and executor
work", "详细介绍一下 X 的架构"). Choose papers that have an arXiv HTML version
(`https://arxiv.org/html/<id>v<n>` loads), with a mix of lengths: at least two long technical
reports (many sections) and at least two short conference papers. Write 5–8 rubric items: the
points a good answer must cover (how it works, the key components, the main results and their
conditions), each a single checkable statement.

Every rubric item has one to three `evidence` quotes:
- `paper`: the versioned arXiv id the item names;
- `section`: the heading path as the HTML shows it, joined with " › " and without section
  numbers (e.g. "Method › Planner");
- `quote`: one or more whole sentences copied exactly from that version's HTML text. Do not
  change a character. Choose sentences without mathematical notation, footnote marks or
  citations in brackets where possible; never quote figure or table captions.

`papers` lists the one paper; `expect.intent` is `read`.

### compare_papers (6): authored

A user asks to compare two named papers on one aspect ("Compare how ReWOO and LLMCompiler avoid
calling the LLM after every tool call"). Choose pairs that really address the same problem. Rubric
5–8 items, covering both papers and at least one item that states a difference; evidence as for
explain_paper, quoting the paper each point is about. `papers` lists both versioned ids;
`expect.intent` is `read`.

### latest_topic (6): authored

A user asks for the newest work on a topic ("最近半年有哪些关于 LLM agent memory 的新论文？").
Choose active topics with many arXiv papers in the last six months (check with the arXiv API,
sorted by submission date; record what you saw in `notes`). No rubric, no papers;
`expect.intent` `discover`, `expect.recent_days` 180.

## Phase 3: the file

`evals/datasets/s8_open.json`, UTF-8, two-space indentation, in this shape:

```json
{
  "suite": "s8",
  "created": "2026-10-10",
  "sources": [
    {"name": "ScholarQA-CS (ScholarQABench)", "url": "https://github.com/AkariAsai/ScholarQABench",
     "license": "ODC-BY-1.0", "citation": "Asai et al., OpenScholar, arXiv:2411.14199"},
    {"name": "ResearchQA", "url": "https://huggingface.co/datasets/realliyifei/ResearchQA",
     "license": "MIT", "citation": "Li et al., ResearchQA, arXiv:2509.00496"},
    {"name": "authored", "url": "", "license": "same as this repository", "citation": ""}
  ],
  "items": [
    {
      "id": "s8-explain-rewoo-workers",
      "type": "explain_paper",
      "split": "dev",
      "language": "Chinese",
      "message": "详细讲讲 ReWOO 的 planner 和 worker 是怎么配合的",
      "message_en": "Explain in detail how ReWOO's planner and workers work together.",
      "source": {"dataset": "authored", "id": null, "license": "same as this repository"},
      "papers": ["2305.18323v1"],
      "names": [],
      "rubric": [
        {
          "id": "r1",
          "item": "Workers are called with the instructions in the Planner's blueprint and fill its evidence slots with real observations.",
          "kind": "mechanism",
          "evidence": [
            {
              "paper": "2305.18323v1",
              "section": "Methodology › ReWOO with Plan-Work-Solve Paradigm",
              "quote": "Worker enables ReWOO to interact with the environment through tool-calls."
            }
          ]
        }
      ],
      "expect": {"intent": "read", "primary": null, "recent_days": null},
      "reviewed": false,
      "notes": ""
    }
  ]
}
```

(The example shows one rubric item; real explain items have five to eight.) Rules for the file:
- `id`: `s8-<type word>-<short slug>`, unique, lowercase, e.g. `s8-survey-kimi-k3`,
  `s8-topic-sqa-<source id>`, `s8-compare-rewoo-llmcompiler`, `s8-latest-agent-memory`.
- `reviewed` is always `false`: the maintainer reviews every item.
- English items: `message_en` equals `message`.

Validate with:

```bash
uv run python -m evals.suites.s8
```

It checks the schema, the split and language balance, and fetches every quoted paper from arXiv
to confirm each quote appears word for word (whitespace and case folded) in the text ARA parses.
Fix what it reports and run it again until it prints `OK: 30 items`. Mathematical notation is
turned into LaTeX by the parser, which is why quotes should avoid it.

## Phase 4: the selection log

`docs/v2/eval/s8-selection.md`, in English:
- the sources (URL, revision, licence, files and their sha256, as in the READMEs);
- for each type, the candidates you considered (about twice as many as you kept), with one line
  each on why kept or dropped;
- for each kept item: the evidence you relied on (arXiv API counts with their dates, HTML pages
  quoted);
- anything you were unsure about and the choice you made;
- the checker's final output.

## When you finish

Reply with a short summary: the 30 ids by type, split and language; anything dropped or uncertain;
and the checker's last line. Nothing else.
