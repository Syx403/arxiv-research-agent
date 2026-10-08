"""M2 review live check (D21): the graph paths the first live check never reached.

About 10 requests and under US$0.005 of spend, approved by Ewan on 2026-10-08. The run cap is
US$0.01 because one repair call alone reserves about US$0.0052 (8K output tokens at the DeepSeek
peak price) before it settles near US$0.001.
- a true phrase that does not answer the question fails verification (1 Luna request);
- prewarm, then two verifications that read it from the cache (3 Luna requests);
- one repair of a rejected line (1 DeepSeek request; the verdicts are given);
- the read graph's requery on a question with an aspect the paper does not cover (Luna
  select_evidence and query embeddings; reranks are free).
Run one test with `uv run pytest -m live tests/live/test_answer_paths.py -k <name>`.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from langgraph.runtime import Runtime

from ara.arxiv.client import ArxivClient
from ara.graph import answer
from ara.graph.qa import READ
from ara.graph.state import Claim, Context, Evidence, Verdict
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.llm.stages import STAGES
from ara.rag.chunking import sentence_starts
from ara.rag.sources import ParsedPaper, qasper_paper
from ara.tokens import count_tokens
from evals import qasper

pytestmark = pytest.mark.live

RUN_ID = f"m2-review-{datetime.now(UTC):%Y%m%dT%H%M%S}"
SCOPE = Scope(run_id=RUN_ID, run_cap_usd=Decimal("0.01"))
PAPER = "1806.00738"  # an S2 held-out paper; its yes/no question's gold answer is "No"
QUESTION = "Do the decoder LSTMs all have the same weights?"
DECODERS = "Our model contains five independent decoders, one for each image in the sequence."


def paper() -> ParsedPaper:
    return qasper_paper(qasper.records()[PAPER])


def pack(min_tokens: int) -> list[Evidence]:
    """The paper's sentences in order, labelled E1, E2, ..., until the pack is long enough."""
    evidence: list[Evidence] = []
    for n, p in enumerate(paper().paragraphs):
        starts = sentence_starts(p.text)
        for a, b in zip(starts, (*starts[1:], len(p.text)), strict=True):
            e = Evidence(
                id=f"E{len(evidence) + 1}",
                paper_id=f"qasper:{PAPER}",
                chunk_id=0,
                paragraph=n,
                heading_path=p.heading_path,
                text=p.text[a:b].strip(),
            )
            evidence.append(e)
        if count_tokens(answer.pack(evidence).text) >= min_tokens and any(
            e.text == DECODERS for e in evidence
        ):
            return evidence
    raise AssertionError("the paper is shorter than the pack needs")


def cite(evidence: list[Evidence], text: str) -> str:
    return next(e.id for e in evidence if e.text == text)


def draft(evidence: list[Evidence]) -> str:
    """A reply whose second line the decoder sentence does not support."""
    e = cite(evidence, DECODERS)
    return f"Answer: no [{e}]\nAll five decoders share one set of weights. [{e}]"


def context(gateway: Gateway) -> Context:
    async def fetch(reference: str) -> ParsedPaper:
        return paper()

    return Context(gateway.ledger.pool, gateway, SCOPE, fetch, ArxivClient())  # sends nothing


async def test_a_true_phrase_that_does_not_answer_the_question_is_rejected(
    gateway: Gateway,
) -> None:
    evidence = [e for e in pack(0) if e.text == DECODERS]
    off_question = Claim(index=0, text="five independent decoders", citations=[evidence[0].id])
    prompt = answer.verify_prompt(evidence, off_question, "How many images are in each sequence?")
    verdict = await gateway.structured(STAGES["verify"], prompt, Verdict, scope=SCOPE)
    assert not verdict.supported  # the words are in the evidence, but they do not answer this


async def test_verification_reads_the_prewarmed_pack(gateway: Gateway) -> None:
    evidence = pack(1_100)
    claims, _ = answer.parse(draft(evidence), {x.id for x in evidence})
    state: Any = {"question": QUESTION, "evidence": evidence, "claims": claims, "verdicts": {}}
    runtime = Runtime(context=context(gateway))
    await answer.prewarm(state, runtime)
    verdicts: dict[str, Verdict] = {}
    for c in claims:
        task = answer.VerifyTask(question=QUESTION, claim=c, evidence=evidence)
        verdicts |= (await answer.verify(task, runtime))["verdicts"]
    assert not verdicts[claims[1].key].supported
    print("verdict on the direct answer 'no':", verdicts[claims[0].key])

    rows = [r for r in await gateway.ledger.calls(RUN_ID) if r["stage"] == "verify"]
    [warm] = [r for r in rows if r["output_tokens"] == 0]
    assert (warm["cache_write_tokens"] or 0) + (warm["cached_tokens"] or 0) >= 1_024
    assert any((r["cached_tokens"] or 0) >= 1_024 for r in rows if r["output_tokens"])


async def test_one_repair_rewrites_the_rejected_line(gateway: Gateway) -> None:
    """The verdicts are given, so this sends only the repair request."""
    evidence = pack(0)
    text = draft(evidence)
    claims, uncited = answer.parse(text, {x.id for x in evidence})
    problem = "The evidence says the decoders are independent, not that they share weights."
    state: Any = {
        "question": QUESTION,
        "evidence": evidence,
        "missing": [],
        "draft": text,
        "claims": claims,
        "uncited": uncited,
        "verdicts": {
            claims[0].key: Verdict(supported=True, problem=""),
            claims[1].key: Verdict(supported=False, problem=problem),
        },
    }
    repaired: Any = await answer.repair(state, Runtime(context=context(gateway)))
    assert "share one set of weights" not in repaired["draft"]
    assert [c.text for c in repaired["rejected"]] == [claims[1].text]
    print("repaired draft:", repaired["draft"])
    [fix] = [r for r in await gateway.ledger.calls(RUN_ID) if r["stage"] == "repair"]
    assert "+repair@" in fix["prompt_version"]


async def test_the_read_graph_requeries_an_uncovered_aspect(gateway: Gateway) -> None:
    question = f"{QUESTION[:-1]}, and how many GPUs were used to train them?"
    result = await READ.ainvoke(
        {"question": question, "papers": [f"qasper:{PAPER}"]}, context=context(gateway)
    )
    selections = [r for r in await gateway.ledger.calls(RUN_ID) if r["stage"] == "select_evidence"]
    assert len(selections) >= 2  # the first round and at least one requery
    assert result["evidence"]
    print("missing after the requery:", result["missing"])
