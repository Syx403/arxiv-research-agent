"""What each suite tests, for the evaluation page (D38): the part of the system it covers, its data,
how it is graded, and what every metric it records means. Kept beside the suites and served by
the API, so the page says what DESIGN §11.2 says. Item counts come from the suites' own splits."""

from dataclasses import asdict, dataclass, field
from typing import Any

from ara.llm.prompt import versions as prompt_versions
from evals.report import SUITES as MODULES


@dataclass(frozen=True)
class Suite:
    id: str
    name: str
    covers: str  # the part of the system under test
    question: str  # what the suite answers, in one sentence
    data: str
    grading: str
    headline: str  # the metric shown on the suite's card
    arm: str  # the arm the headline is read from
    metrics: dict[str, str] = field(default_factory=dict)  # name → meaning


SUITES = [
    Suite(
        "s1",
        "Retrieval",
        "read subgraph · search",
        "Do the passages we retrieve from a paper contain the sentences that answer the question?",
        "QASPER validation: 10 papers, 3 questions each, full text from the dataset.",
        "Code, against QASPER's gold evidence paragraphs. Six retrievers are compared on the"
        " same items; the product uses RRF + rerank.",
        "recall@8",
        "rrf_rerank",
        {
            "recall@8": "Share of gold evidence paragraphs among the top 8 passages (what the"
            " product reads).",
            "recall@20": "The same within the top 20.",
            "mrr": "1 / rank of the first gold paragraph.",
            "ndcg@10": "Ranking quality of the top 10, gold paragraphs weighted by position.",
        },
    ),
    Suite(
        "s2",
        "Reading QA",
        "read + answer subgraphs",
        "Given a paper and a question, is the verified answer right, and does it abstain when the"
        " paper has no answer?",
        "The 30 S1 questions, plus 5 unanswerable ones from other QASPER papers run 3 times each.",
        "Code: QASPER token F1 on the direct answer, abstention against the gold label, citations"
        " against gold evidence.",
        "answer_f1",
        "product",
        {
            "answer_f1": "Token F1 of the direct answer against the closest gold answer.",
            "citation_precision": "Share of cited paragraphs that are gold evidence.",
            "abstained": "Share of answers that abstained.",
            "abstain_correct": "Abstained exactly when the question is unanswerable.",
            "unanswerable": "Label: share of items with no answer in the paper.",
            "verified_share": "Share of drafted lines that passed the verifier and were delivered.",
            "rejection_rate": "Share of checked lines the verifier rejected, over both drafts.",
        },
    ),
    Suite(
        "s3",
        "Discovery",
        "discover subgraph",
        "Does the arXiv search find the papers an expert would list, and are the ones we show"
        " relevant?",
        "PaSa: 15 AutoScholarQuery (dev) and 15 RealScholarQuery (held-out) queries, each run as"
        " of its own date.",
        "Code, against PaSa's gold paper sets (pooled, so precision is a lower bound).",
        "pool_recall",
        "product",
        {
            "pool_recall": "Share of gold papers anywhere in the candidate pool the search built.",
            "shortlist_recall": "Share of gold papers in the shortlist sent to screening.",
            "gold_precision_at_5": "Share of the listed papers (at most 5) that are gold.",
            "hit_at_5": "At least one gold paper is listed.",
            "listed": "Papers listed.",
            "unjudged": "Shortlisted papers the screen returned no judgement for.",
        },
    ),
    Suite(
        "s4",
        "Understanding",
        "understand node",
        "Is each message turned into the right request: intent, papers, constraints, dates, and"
        " a question back only when one is needed?",
        "62 messages: 12 from v1 and 50 drafted, labels reviewed by Ewan.",
        "Code, field by field against the reviewed labels.",
        "intent_correct",
        "product",
        {
            "intent_correct": "The intent (discover, read, library, memory, ...) is right.",
            "fields_correct": "Every labelled field is right.",
            "clarify_correct": "Asked a question back exactly when the label says to.",
            "clarified": "Share of messages answered with a question.",
            "clarify_expected": "Label: share of messages that need a question.",
            "field_paper_ids": "arXiv ids named in the message.",
            "field_listed": 'Positions in an earlier list ("the second one").',
            "field_titles": "Paper titles named in the message.",
            "field_count": "How many papers were asked for.",
            "field_constraints": "Requirements the papers must meet, quoted from the user.",
            "field_priorities": "What the user cares most about, quoted.",
            "field_published_after": "Earliest publication date asked for.",
            "field_published_before": "Latest publication date asked for.",
            "field_prefer_recent": "Newer papers preferred.",
            "field_history": "The part of earlier conversation the request depends on.",
        },
    ),
    Suite(
        "s5",
        "Verifier",
        "answer subgraph · verify",
        "Does the verifier reject a claim its evidence does not support, and accept one it does?",
        "58 QASPER evidence sentences: paraphrases (supported) and perturbations of a number,"
        " a negation, an entity or a generalisation (unsupported), reviewed by Ewan.",
        "Code, against the labels; two verifier models compared on the same items.",
        "correct",
        "luna",
        {
            "correct": "The verdict matches the label.",
            "flagged": "Share of claims judged unsupported.",
            "unsupported": "Label: share of claims that are unsupported.",
        },
    ),
    Suite(
        "s6",
        "Multi-turn + memory",
        "the whole conversation graph",
        "Across several turns, does it keep references, constraints and memory straight, answer"
        " from what we read before, and handle papers the user names?",
        "19 scripted scenarios (37 turns), drafted by Claude and reviewed by Ewan; the held-out"
        " ones are frozen until the E rounds.",
        "Code: each turn has checks on intent, papers read and listed, citations, memory and"
        " reply text.",
        "passed",
        "product",
        {
            "passed": "Every check of the scenario passed.",
            "checks_passed": "Share of the scenario's checks that passed.",
        },
    ),
    Suite(
        "s7",
        "Robustness",
        "retries, timeouts, degraded replies",
        "When a dependency fails, does the user still get a reply that says what failed; can text"
        " inside a paper take over the agent?",
        "6 turns with injected faults (ARA_FAULTS) and 3 turns over a synthetic paper carrying a"
        " prompt injection, stored locally.",
        "Code: a fault turn must end in a reply, not an exception; the injection's marker must"
        " never appear in a delivered answer.",
        "graceful",
        "product",
        {
            "graceful": "The fault turn ended in a reply that names what failed.",
            "injected": "The injected instruction reached the answer (must be 0).",
        },
    ),
]

PROTOCOL = [
    "Each suite has dev items, used to make choices, and held-out items, used only to report."
    " S6's held-out scenarios are frozen until the E rounds (D33).",
    "Every metric is a mean over items with a 95% bootstrap interval; at n of 15 to 30 the interval"
    " is wide, so differences are directional, not accuracy claims.",
    "Graded by code against labelled data; no model grades a model yet (the calibrated judge"
    " comes in E1).",
    "Real model calls only, no mocks or recordings; a round costs at most US$1 and every run is"
    " approved before it starts.",
    "A round is measured on the code and prompts of its day: one whose prompts differ from"
    " today's is marked, so an old number is not read as the current system's (D39).",
]


def catalog() -> dict[str, Any]:
    """The suites with their item counts by split, and the protocol they share."""
    suites = []
    for suite in SUITES:
        splits = MODULES[suite.id].splits()
        counts = {s: sum(1 for v in splits.values() if v == s) for s in ("dev", "test")}
        suites.append({**asdict(suite), "items": counts})
    return {"suites": suites, "protocol": PROTOCOL, "prompts": prompt_versions()}
