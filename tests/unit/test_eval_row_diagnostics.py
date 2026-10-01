from __future__ import annotations

from src.core.types import Citation, CitationVerdict, JudgeVerdict, VerificationReport
from src.eval import runner


def test_eval_row_diagnostics_from_verified_result() -> None:
    answer = "A [arxiv:1#1]. B [arxiv:1#2]. C [arxiv:1#3]. D [arxiv:1#4]. E [arxiv:1#5]."
    result = {
        "answer": answer,
        "citations": [_citation(1), _citation(2)],
        "verification": VerificationReport(
            verdicts=[
                CitationVerdict(citation=_citation(1), supports=True, rationale="ok"),
                CitationVerdict(citation=_citation(2), supports=False, rationale="PARSE_FAILURE: bad"),
            ],
            passed=False,
        ),
        "synthesis_finish_reason": "length",
        "multi_hop_attempts": 2,
        "multi_hop_expansions": 1,
    }

    diagnostics = runner._diagnostics_from_result(result, answer)

    assert diagnostics["synthesis_finish_reason"] == "length"
    assert diagnostics["citation_parse_count"] == 5
    assert diagnostics["verifier_rejected_count"] == 1
    assert diagnostics["parse_failure_count"] == 1
    assert diagnostics["multi_hop_attempts"] == 2
    assert diagnostics["multi_hop_expansions"] == 1


def test_failure_classification_distinguishes_hard_and_warning_classes() -> None:
    judge = JudgeVerdict(correctness=5, groundedness=5, completeness=5, rationale="ok")

    assert _classify({"synthesis_format_degraded": True}, "", ["p"], [], judge, 0) == "synthesis_empty"
    assert _classify({"synthesis_format_degraded": True}, "foo", ["p"], [], judge, 0) == "citation_parse_empty"
    assert _classify({}, "foo", [], [], judge, 0) == "retrieval_empty"
    assert _classify({}, "A [arxiv:1#1].", ["p"], [], judge, 1) == "verifier_all_rejected"
    assert _classify({}, "ok", ["p"], ["p"], judge, 1) == ""

    invalid_judge = JudgeVerdict(
        correctness=1,
        groundedness=1,
        completeness=1,
        rationale="Judge returned invalid JSON twice; scored conservatively.",
    )
    assert _classify({}, "ok", ["p"], ["p"], invalid_judge, 1) == "judge_invalid"


def _classify(result, answer, retrieved, cited, judge, parse_count):
    return runner._classify_result(
        result=result,
        answer=answer,
        retrieved_paper_ids=retrieved,
        cited_paper_ids=cited,
        judge=judge,
        citation_parse_count=parse_count,
    )


def _citation(chunk_id: int) -> Citation:
    return Citation(paper_id="arxiv:1", chunk_id=chunk_id, claim_text=f"claim {chunk_id}")
