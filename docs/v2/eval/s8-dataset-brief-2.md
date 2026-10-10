# S8 revision: brief for the building agent

You built `evals/datasets/s8_open.json` from `docs/v2/eval/s8-dataset-brief.md`. Its review found
problems that come from general rules, so the fixes below are general rules too. Apply them to the
whole file, not to single items. Everything in the first brief still holds (hard rules, sources,
schema, the checker) unless this brief changes it. Today is 2026-10-11.

## Changes

1. **Every discovery item is graded the same way.** `survey_named` items no longer carry a primary
   paper: `papers` is empty and `expect.primary` is null. Keep `names` (the subject as the query
   writes it). A query qualifies whether it asks about the subject itself ("papers on LoRA") or
   about papers that use or evaluate on it ("papers that use HotPotQA as a benchmark"); record which
   in `notes`, in those words ("about the subject" / "uses the subject"). Keep the arXiv count of
   papers naming the subject in `notes`.
2. **Fixed counts instead of shares.** 8 items per type, 40 in all: per type, 2 English and 2
   Chinese in `dev`, the same in `test`. The census stays in the log as information about what users
   ask; it no longer sets the counts. If a type cannot reach 8 under the rules, stop at 6 and log why.
3. **No request twice.** A request and its translation are the same request: at most one of them
   is in the file. The checker now rejects two items whose `message_en` match (whitespace and case
   folded).
4. **Length and language by characters.** The census filter "under three words" dropped Chinese
   queries, which have no spaces. Redo the filter: a query is too short when it has under three
   whitespace words and under six CJK characters. Recount the Chinese queries and report the new
   count.
5. **Real Chinese queries first, for every type.** Use real Chinese queries wherever one qualifies;
   translate a real English query only when no real Chinese query of that type qualifies, and log,
   per type, how many real Chinese candidates you checked and why each was dropped. `topic_survey`
   keeps rubric-bearing dataset questions, so its Chinese items stay translations; say so in the log.
6. **Licences must be explicit for data.** A source whose licence grant does not name the data (for
   example an MIT grant for "software and associated documentation files") is not used. Replace any
   item from such a source.
7. **Explain and compare from real queries.** Fill `explain_paper` and `compare_papers` from real
   queries (Asta across the whole kept corpus, SciArena) under the first brief's rules; use dataset
   questions only from explicitly licensed sources, and only if real queries run out.

## Output

Overwrite `evals/datasets/s8_open.json` and update `docs/v2/eval/s8-selection.md`: add a section
"Revision 2" that lists every item added, removed or changed and the rule that caused it, and the
new census numbers. Run `uv run python -m evals.suites.s8` until it prints `OK: 40 items` (or the
reduced count you logged). Then reply with a short summary as before.
