from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


QuestionCategory = Literal["definition", "comparison", "multi_hop", "survey", "application"]


class GoldQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    category: QuestionCategory
    question: str
    expected_paper_ids: list[str]
    answer_must_mention: list[str]
    answer_must_not_mention: list[str] = Field(default_factory=list)
    notes: str = ""


GOLD_QUESTIONS: list[GoldQuestion] = [
    GoldQuestion(
        id="q-001",
        category="definition",
        question="What is ReAct?",
        expected_paper_ids=["arxiv:2210.03629"],
        answer_must_mention=["reasoning", "acting", "interleaved"],
    ),
    GoldQuestion(
        id="q-002",
        category="definition",
        question="Define the Reflexion mechanism for language agents.",
        expected_paper_ids=["arxiv:2303.11366"],
        answer_must_mention=["verbal", "reinforcement", "reflection"],
    ),
    GoldQuestion(
        id="q-003",
        category="definition",
        question="What is Tree of Thoughts?",
        expected_paper_ids=["arxiv:2305.10601"],
        answer_must_mention=["tree", "deliberate", "search"],
    ),
    GoldQuestion(
        id="q-004",
        category="definition",
        question="What does Graph of Thoughts add to prompting?",
        expected_paper_ids=["arxiv:2308.09687"],
        answer_must_mention=["graph", "thoughts", "dependencies"],
    ),
    GoldQuestion(
        id="q-005",
        category="definition",
        question="What is MemGPT's virtual context management idea?",
        expected_paper_ids=["arxiv:2310.08560"],
        answer_must_mention=["memory", "context", "operating system"],
    ),
    GoldQuestion(
        id="q-006",
        category="definition",
        question="What is Self-RAG?",
        expected_paper_ids=["arxiv:2310.11511"],
        answer_must_mention=["retrieve", "generate", "critique"],
    ),
    GoldQuestion(
        id="q-007",
        category="definition",
        question="What is Corrective Retrieval Augmented Generation?",
        expected_paper_ids=["arxiv:2401.15884"],
        answer_must_mention=["corrective", "retrieval", "robust"],
    ),
    GoldQuestion(
        id="q-008",
        category="definition",
        question="What is HippoRAG?",
        expected_paper_ids=["arxiv:2405.14831"],
        answer_must_mention=["long-term memory", "knowledge graph", "retrieval"],
    ),
    GoldQuestion(
        id="q-009",
        category="definition",
        question="What is AgentBench designed to evaluate?",
        expected_paper_ids=["arxiv:2308.03688"],
        answer_must_mention=["benchmark", "agents", "environments"],
    ),
    GoldQuestion(
        id="q-010",
        category="definition",
        question="What is A-MEM for LLM agents?",
        expected_paper_ids=["arxiv:2502.12110"],
        answer_must_mention=["agentic memory", "experiences", "organization"],
    ),
    GoldQuestion(
        id="q-011",
        category="comparison",
        question="How does Tree of Thoughts differ from chain-of-thought prompting?",
        expected_paper_ids=["arxiv:2305.10601", "arxiv:2201.11903"],
        answer_must_mention=["tree", "chain of thought", "lookahead"],
    ),
    GoldQuestion(
        id="q-012",
        category="comparison",
        question="Compare Graph of Thoughts and Tree of Thoughts as reasoning structures.",
        expected_paper_ids=["arxiv:2308.09687", "arxiv:2305.10601"],
        answer_must_mention=["graph", "tree", "dependencies"],
    ),
    GoldQuestion(
        id="q-013",
        category="comparison",
        question="How does Toolformer differ from ToolLLM for tool use?",
        expected_paper_ids=["arxiv:2302.04761", "arxiv:2307.16789"],
        answer_must_mention=["tools", "APIs", "training"],
    ),
    GoldQuestion(
        id="q-014",
        category="comparison",
        question="Compare Gorilla and ToolLLM on API use by language models.",
        expected_paper_ids=["arxiv:2305.15334", "arxiv:2307.16789"],
        answer_must_mention=["API", "tool", "hallucination"],
    ),
    GoldQuestion(
        id="q-015",
        category="comparison",
        question="How do RAG and Self-RAG differ in how they use retrieval?",
        expected_paper_ids=["arxiv:2005.11401", "arxiv:2310.11511"],
        answer_must_mention=["retrieval", "critique", "provenance"],
    ),
    GoldQuestion(
        id="q-016",
        category="comparison",
        question="Compare HyDE with ColBERTv2 for retrieval.",
        expected_paper_ids=["arxiv:2212.10496", "arxiv:2112.01488"],
        answer_must_mention=["hypothetical document", "late interaction", "dense retrieval"],
    ),
    GoldQuestion(
        id="q-017",
        category="comparison",
        question="How do AutoGen and MetaGPT organize multi-agent collaboration differently?",
        expected_paper_ids=["arxiv:2308.08155", "arxiv:2308.00352"],
        answer_must_mention=["multi-agent", "conversation", "workflow"],
    ),
    GoldQuestion(
        id="q-018",
        category="comparison",
        question="Compare SWE-bench and GAIA as evaluations for advanced AI assistants.",
        expected_paper_ids=["arxiv:2310.06770", "arxiv:2311.12983"],
        answer_must_mention=["benchmark", "real-world", "assistant"],
    ),
    GoldQuestion(
        id="q-019",
        category="multi_hop",
        question="How does Reflexion extend ReAct's interleaved reasoning with feedback?",
        expected_paper_ids=["arxiv:2210.03629", "arxiv:2303.11366"],
        answer_must_mention=["ReAct", "reflection", "feedback"],
    ),
    GoldQuestion(
        id="q-020",
        category="multi_hop",
        question="How does LATS combine ideas from ReAct and tree search?",
        expected_paper_ids=["arxiv:2310.04406", "arxiv:2210.03629", "arxiv:2305.10601"],
        answer_must_mention=["tree search", "reasoning", "acting"],
    ),
    GoldQuestion(
        id="q-021",
        category="multi_hop",
        question="How do self-consistency and chain-of-thought prompting relate?",
        expected_paper_ids=["arxiv:2203.11171", "arxiv:2201.11903"],
        answer_must_mention=["self-consistency", "chain of thought", "reasoning paths"],
    ),
    GoldQuestion(
        id="q-022",
        category="multi_hop",
        question="How does Plan-and-Solve build on zero-shot chain-of-thought reasoning?",
        expected_paper_ids=["arxiv:2305.04091", "arxiv:2201.11903"],
        answer_must_mention=["plan", "solve", "chain of thought"],
    ),
    GoldQuestion(
        id="q-023",
        category="multi_hop",
        question="How do agentic memory systems relate to RAG and long-term memory?",
        expected_paper_ids=["arxiv:2502.12110", "arxiv:2405.14831", "arxiv:2312.10997"],
        answer_must_mention=["memory", "retrieval", "knowledge"],
    ),
    GoldQuestion(
        id="q-024",
        category="multi_hop",
        question="How do tool-use benchmarks connect tool learning and assistant evaluation?",
        expected_paper_ids=["arxiv:2304.08354", "arxiv:2311.12983", "arxiv:2308.03688"],
        answer_must_mention=["tool", "benchmark", "evaluation"],
    ),
    GoldQuestion(
        id="q-025",
        category="survey",
        question="What approaches exist for tool use in LLM agents?",
        expected_paper_ids=["arxiv:2302.04761", "arxiv:2305.15334", "arxiv:2307.16789", "arxiv:2304.08354"],
        answer_must_mention=["tools", "APIs", "foundation models"],
    ),
    GoldQuestion(
        id="q-026",
        category="survey",
        question="What approaches exist for retrieval-augmented generation and agentic RAG?",
        expected_paper_ids=["arxiv:2005.11401", "arxiv:2312.10997", "arxiv:2401.15884", "arxiv:2501.09136"],
        answer_must_mention=["retrieval", "generation", "agentic"],
    ),
    GoldQuestion(
        id="q-027",
        category="survey",
        question="What approaches exist for multi-agent LLM systems?",
        expected_paper_ids=["arxiv:2303.17760", "arxiv:2308.00352", "arxiv:2308.08155", "arxiv:2304.03442"],
        answer_must_mention=["agents", "conversation", "collaboration"],
    ),
    GoldQuestion(
        id="q-028",
        category="survey",
        question="What benchmarks evaluate LLM agents and assistants?",
        expected_paper_ids=["arxiv:2308.03688", "arxiv:2310.06770", "arxiv:2311.12983", "arxiv:2308.11432"],
        answer_must_mention=["benchmark", "agents", "evaluation"],
    ),
    GoldQuestion(
        id="q-029",
        category="application",
        question="How would chain-of-thought prompting apply to mathematical reasoning?",
        expected_paper_ids=["arxiv:2201.11903", "arxiv:2203.11171"],
        answer_must_mention=["mathematical", "reasoning", "intermediate steps"],
    ),
    GoldQuestion(
        id="q-030",
        category="application",
        question="How could SWE-bench be used to evaluate coding agents?",
        expected_paper_ids=["arxiv:2310.06770"],
        answer_must_mention=["GitHub issues", "software engineering", "benchmark"],
    ),
]
