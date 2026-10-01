"""Regressions from the two-turn Chinese manual acceptance session."""

import json
import pytest

from src.core.types import ChatResponse, CitationVerdict, Decomposition, EvidenceChunk, SubQuestion, VerificationReport
from src.graph import builder
from src.graph.edges import route_after_verify
from src.graph.nodes import synthesize, self_rag
from src.graph.nodes.finalize import finalize_node
from src.retrieval.evidence import pack_evidence, token_count, evidence_text
from src.retrieval.index.types import Hit
from src.retrieval.query import decomposer
from src.retrieval.self_rag import verifier


QUESTION = "比较 MemGPT 和 Reflexion 如何利用历史信息。我在做一个少量用户使用的 Agent，不打算训练模型，更看重响应速度。请根据论文原文解释两者能借鉴的设计；关于工程成本的推断请单独标明。"
FOLLOWUP = "对于不打算训练模型、少量用户使用且看重响应速度的 Agent，若目标改为跨会话保留用户偏好，基于 MemGPT 与 Reflexion 所给出的建议是否需要调整？现有证据在哪些方面仍不能确定？"


class Model:
    def __init__(self, contents, finish_reason="stop"):
        self.contents, self.finish_reason, self.calls = list(contents), finish_reason, []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        content = self.contents.pop(0)
        return ChatResponse(content=content, model="offline-fixture", finish_reason=self.finish_reason)


@pytest.mark.asyncio
async def test_chinese_constraints_are_not_a_paper_name(monkeypatch):
    assert decomposer._detect_comparison_entities(QUESTION) == ["MemGPT", "Reflexion"]
    plan = Decomposition(sub_questions=[SubQuestion(text="MemGPT historical information and memory management"),
                                       SubQuestion(text="Reflexion historical feedback without model training")])
    model = Model([plan.model_dump_json()])
    monkeypatch.setattr(decomposer, "get_chat_client", lambda _: model)
    assert await decomposer.decompose(QUESTION) == plan
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_followup_decision_request_is_planned(monkeypatch):
    assert not decomposer._is_trivial_question(FOLLOWUP)
    plan = Decomposition(sub_questions=[SubQuestion(text="MemGPT cross-session user preferences"),
                                       SubQuestion(text="Reflexion persistent memory scope")])
    model = Model([plan.model_dump_json()])
    monkeypatch.setattr(decomposer, "get_chat_client", lambda _: model)
    assert await decomposer.decompose(FOLLOWUP) == plan
    assert len(model.calls) == 1


def test_only_known_missing_namespace_is_repaired_with_fresh_offsets():
    hit = Hit(297, "arxiv:2310.08560", "Method", "Memory mechanism.", 1)
    text = "机制 [2310.08560#297]。其他论文 [2303.11366#297]。不存在 [2310.08560#999]。"
    fixed = synthesize._normalize_citation_ids(text, [hit])
    citations = synthesize._parse_citations(fixed)
    assert [c.paper_id for c in citations] == ["arxiv:2310.08560", "2303.11366", "2310.08560"]
    assert all(fixed[c.claim_span[0]:c.claim_span[1]].strip() == c.claim_text for c in citations)


def test_citations_after_punctuation_stay_bound_to_their_own_sentence():
    hits = [Hit(1, "p", "Method", "First mechanism.", 1), Hit(2, "q", "Method", "Second mechanism.", 1)]
    raw = "机制一。[p#1] [q#2]\n\nMechanism two. [q#2]\n\n新段没有引用。\n[p#1]"
    fixed = synthesize._normalize_citation_ids(raw, hits)
    citations = synthesize._parse_citations(fixed)
    assert citations[0].claim_text == citations[1].claim_text == "机制一 [p#1] [q#2]。"
    assert citations[2].claim_text == "Mechanism two [q#2]."
    assert "新段没有引用。\n[p#1]" in fixed
    assert all(fixed[c.claim_span[0]:c.claim_span[1]].strip() == c.claim_text for c in citations)


def test_evidence_budget_retains_lower_ranked_mechanisms_and_both_methods():
    hits = [Hit(i, "arxiv:memgpt", "Method", f"Mechanism {i} explains a distinct memory operation.",
                1-i*.01, title="MemGPT: Virtual context", evidence_role="direct") for i in range(1, 8)]
    hits.append(Hit(20, "arxiv:reflexion", "Method", "Feedback updates persistent reflections.", .5,
                    title="Reflexion: Language agents", evidence_role="direct"))
    packed = pack_evidence(hits, QUESTION)
    assert {h.chunk_id for h in packed} == {1,2,3,4,5,6,7,20}
    assert {h.paper_id for h in packed[:2]} == {"arxiv:memgpt", "arxiv:reflexion"}
    assert sum(token_count(evidence_text(h)) for h in packed) <= 1600


@pytest.mark.asyncio
async def test_truncated_verification_is_unknown_and_does_not_rewrite(monkeypatch):
    model = Model(["", ""], finish_reason="length")
    monkeypatch.setattr(verifier, "get_chat_client", lambda _: model)
    answer = "A supported mechanism [p#1]."
    report = await verifier.verify_citations(answer, synthesize._parse_citations(answer),
                                           {1: EvidenceChunk(paper_id="p", chunk_id=1, text="A supported mechanism.")})
    assert len(model.calls) == 2
    assert report.verdicts[0].status == "error"
    assert not report.passed
    assert route_after_verify({"verification": report, "tool_iters": 0}) == "finalize"
    final = await finalize_node({"question":"机制是什么？", "answer": answer, "verification":report,
                                 "evidence":[1], "tool_iters":0})
    assert final["stop_reason"] == "verification_unavailable"
    assert "不能据此认定论文不支持" in final["answer"]


@pytest.mark.asyncio
async def test_regressing_rewrite_keeps_bound_earlier_verified_sentences(monkeypatch):
    async def judge(answer, citations, lookup, **kwargs):
        return VerificationReport(verdicts=[CitationVerdict(citation=c, supports="unsupported" not in c.claim_text,
                                                          rationale="recorded fixture verdict") for c in citations], passed=False)
    monkeypatch.setattr(self_rag, "verify_citations", judge)
    lookup = {1: EvidenceChunk(paper_id="p", chunk_id=1, text="Memory persists.")}
    state = builder._initial_state("How does memory persist?", "fixture", skip_reflection=True)
    state.update(answer="Memory persists [p#1]. An unsupported ranking [p#1].", evidence_lookup=lookup, evidence=[1])
    state["citations"] = synthesize._parse_citations(state["answer"])
    state.update(await self_rag.self_rag_node(state))
    state.update(answer="A wholly unsupported rewrite [p#1].", tool_iters=2)
    state["citations"] = synthesize._parse_citations(state["answer"])
    state.update(await self_rag.self_rag_node(state))
    final = await finalize_node(state)
    assert final["status"] == "partial"
    assert final["verified_answer"] == "Memory persists [p#1]."
    assert final["delivery_from_prior_draft"]
    assert final["evidence_lookup"] == lookup
    assert not builder._initial_state("Another question", "fixture")["best_verified"]






def test_engineering_inference_type_never_waives_source_requirements():
    citation = synthesize._parse_citations("工程推断：它一定更快 [p#1]。")[0]
    result = verifier._parse_verdict(json.dumps({
        "claim_kind": "engineering_inference", "scope_status": "requested", "supports": True,
        "rationale": "The label does not establish the claimed speed advantage.",
        "all_assertions_supported": False, "no_unstated_assumptions": False,
        "all_citations_contribute": True,
    }), citation)
    assert result.claim_kind == "engineering_inference"
    assert not result.supports


def test_uncited_title_and_generic_heading_are_checked_with_following_claim_not_dropped():
    answer = "**Official Model Report (2026-09-10)**\n\nThe cache is compressed [p#1].\n\n## Quantitative results\n\nStorage falls to a quarter [p#2]."
    normalized = synthesize._normalize_citation_ids(answer, [])
    citations = synthesize._parse_citations(normalized)
    assert not verifier.uncited_claims(normalized, citations)
    assert "Official Model Report (2026-09-10)" in citations[0].claim_text
    assert "Quantitative results" in citations[1].claim_text
    assert "Storage falls to a quarter" in citations[1].claim_text


def test_sentence_ending_conclusion_heading_stays_bound_for_verification():
    raw = "**结论：支持，但有条件。**\n\n该偏置使用历史批次的负载 [p#1]。"
    normalized = synthesize._normalize_citation_ids(raw, [])
    citations = synthesize._parse_citations(normalized)
    assert "结论：支持，但有条件" in citations[0].claim_text
    assert not verifier.uncited_claims(normalized, citations)
    assert "。:" not in normalized


@pytest.mark.asyncio
@pytest.mark.parametrize("payload, expected", [
    ('{"repairs":[{"id":0,"text":"Corrected mechanism [p#1]."}]}',
     "Verified mechanism [p#1]. Corrected mechanism [p#1]."),
    ('{"repairs":[{"id":0,"replacement":"Corrected mechanism [p#1]."}]}',
     "Verified mechanism [p#1]. Corrected mechanism [p#1]."),
    ('{"repairs":[{"id":0,"text":"A [p#1].","replacement":"B [p#1]."}]}',
     "Verified mechanism [p#1]. Rejected ranking [p#1]."),
    ('{"repairs":[{"id":9,"replacement":"Unrequested replacement [p#1]."}]}',
     "Verified mechanism [p#1]. Rejected ranking [p#1]."),
    ('{"repairs":[{"id":0,"replacement":""}]}',
     "Verified mechanism [p#1]. Rejected ranking [p#1]."),
    ('{"repairs":[{"id":0,"text":"Corrected mechanism [p#1]. Uncited caveat."}]}',
     "Verified mechanism [p#1]. Rejected ranking [p#1]."),
    ('{"repairs":[{"id":0,"text":"Invented evidence [p#999]."}]}',
     "Verified mechanism [p#1]. Rejected ranking [p#1]."),
])
async def test_local_repair_preserves_verified_text_and_validates_target_ids(monkeypatch, payload, expected):
    model = Model([payload])
    monkeypatch.setattr(synthesize, "get_chat_client", lambda _: model)
    answer = "Verified mechanism [p#1]. Rejected ranking [p#1]."
    citations = synthesize._parse_citations(answer)
    evidence = synthesize._select_synthesis_evidence([Hit(1, "p", "Method", "Verified mechanism. Corrected mechanism.", 1)])
    report = VerificationReport(verdicts=[CitationVerdict(citation=c, supports="Rejected" not in c.claim_text,
                                                          rationale="Recorded fixture check") for c in citations], passed=False)
    update = await synthesize.synthesize_node({
        "question":"Explain the mechanism", "answer":answer, "citations":citations,
        "evidence":evidence, "evidence_lookup":synthesize._evidence_lookup(evidence),
        "verification":report, "tool_iters":0,
    })
    assert update["answer"] == expected
    assert update["tool_iters"] == 1
    assert len(model.calls) == 1
    assert all(expected[c.claim_span[0]:c.claim_span[1]].strip() == c.claim_text for c in update["citations"])


def test_followup_retains_earlier_advice_without_treating_it_as_source():
    from langchain_core.messages import AIMessage, HumanMessage
    from src.graph.conversation import conversation_turns
    history = conversation_turns([HumanMessage(content=QUESTION),
                                 AIMessage(content="背景。" * 800 + "工程推断：先仅使用滚动摘要。")])
    assert "工程推断：先仅使用滚动摘要。" in history[-1]
    notes = synthesize._system_notes({"conversation_context":history})
    assert "先仅使用滚动摘要" in notes[-1]
    assert "NOT paper evidence or instructions" in notes[-1]
    bounded = conversation_turns([AIMessage(content="Start " + "x" * 20000 + " End advice") for _ in range(6)])
    assert sum(len(t) for t in bounded) <= 12000
    assert "End advice" in bounded[-1]
