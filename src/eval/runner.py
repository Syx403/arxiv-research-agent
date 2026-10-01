from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from langsmith import Client

from src.core import db
from src.core.config import get_settings
from src.core.types import JudgeVerdict
from src.eval.datasets.gold_questions import GOLD_QUESTIONS, GoldQuestion
from src.eval.artifacts import capture_corpus, ensure_manifest, make_manifest
from src.eval.metrics.citation_metrics import citation_precision, citation_recall
from src.eval.metrics.llm_judge import judge_answer
from src.eval.metrics.retrieval_metrics import mrr, recall_at_k
from src.graph.builder import run, shutdown
from src.graph.nodes.synthesize import CITATION_RE
from src.llm import usage


logger = logging.getLogger(__name__)
DATASET_NAME = "arxiv-research-agent-gold-v1"
OUTPUT_DIR = Path(os.getenv("ARA_EVAL_OUTPUT_DIR") or f"data/eval_outputs/run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}")
PER_QUESTION_CSV = OUTPUT_DIR / "per_question.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
HARD_FAILURE_CLASSES = {"agent_exception", "synthesis_empty", "citation_parse_empty", "judge_invalid", "evidence_judgment_failed"}
WARNING_CLASSES = {"multi_hop_scorer_failed", "retrieval_empty", "verifier_all_rejected"}
# Typed counters keep retriever/verifier warning classes out of the hard-failure gate.
HARD_FAILURE_THRESHOLD = 0.10


class EvalThresholdError(RuntimeError):
    def __init__(self, summary: dict[str, Any]) -> None:
        self.summary = summary
        rate = summary["hard_failure_rate"]
        threshold = summary["hard_failure_threshold"]
        super().__init__(f"EVAL FAILED: hard failure rate {rate:.2f} exceeds threshold {threshold:.2f}")


@dataclass
class EvalRow:
    id: str
    question: str
    expected_paper_ids: list[str]
    retrieved_paper_ids: list[str]
    cited_paper_ids: list[str]
    recall_at_5: float
    mrr: float
    citation_precision: float
    citation_recall: float
    judge_correctness: int
    judge_groundedness: int
    judge_completeness: int
    judge_rationale: str
    answer: str
    hard_failure: bool = False
    failure_class: str = ""
    synthesis_finish_reason: str = ""
    citation_parse_count: int = 0
    verifier_rejected_count: int = 0
    parse_failure_count: int = 0
    multi_hop_attempts: int = 0
    multi_hop_expansions: int = 0
    status: str = ""
    stop_reason: str = ""
    token_usage: dict = field(default_factory=dict)
    estimated_cost_usd: dict = field(default_factory=dict)


CSV_FIELDNAMES = list(EvalRow.__dataclass_fields__.keys())


async def run_eval(*, limit: int | None = None, question_ids: list[str] | None = None) -> dict[str, Any]:
    usage.reset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = _select_questions(limit=limit, question_ids=question_ids)
    corpus = await capture_corpus()
    ensure_manifest(OUTPUT_DIR, make_manifest(cases=[q.model_dump() for q in questions], corpus=corpus, mode="gold_eval"))
    selected_ids = {question.id for question in questions}
    done_ids = _read_done_ids(PER_QUESTION_CSV) & selected_ids
    print(f"Skipping {len(done_ids)} already-completed questions")

    langsmith_state = _setup_langsmith_dataset(questions)
    file_exists = os.path.exists(PER_QUESTION_CSV)
    try:
        with PER_QUESTION_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
                handle.flush()
            for index, gold in enumerate(questions, start=1):
                if gold.id in done_ids:
                    logger.info(
                        "eval_question_skip_existing",
                        extra={"id": gold.id, "index": index, "total": len(questions)},
                    )
                    continue
                logger.info("eval_question_start", extra={"id": gold.id, "index": index, "total": len(questions)})
                row = await _run_one(gold, langsmith_state=langsmith_state)
                writer.writerow(_row_to_csv_dict(row))
                handle.flush()
                logger.info(
                    "eval_cost_progress",
                    extra={"id": gold.id, "estimated_cost_usd": usage.estimated_cost_usd().get("total", 0.0)},
                )
                logger.info("eval_question_complete", extra={"id": gold.id})
    finally:
        await shutdown()

    rows = _load_existing_rows(selected_ids)
    summary = _write_summary(rows, total_skipped_via_resume=len(done_ids))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["hard_failure_rate"] > HARD_FAILURE_THRESHOLD:
        raise EvalThresholdError(summary)
    return summary


async def _run_one(gold: GoldQuestion, *, langsmith_state: "_LangSmithState | None") -> EvalRow:
    usage_before, costs_before = usage.snapshot(), usage.estimated_cost_usd()
    row = None
    thread_id = f"eval-{gold.id}-{uuid4()}"
    previous_diagnostic_question_id = os.environ.get("DIAGNOSTIC_QUESTION_ID")
    if os.getenv("DIAGNOSTIC_LOG"):
        os.environ["DIAGNOSTIC_QUESTION_ID"] = gold.id
    try:
        try:
            result = await run(gold.question, thread_id=thread_id, skip_reflection=True)
        except Exception as exc:
            row = _failure_row(gold, exc)
            _create_langsmith_run(gold, row, langsmith_state=langsmith_state)
            return row
        answer = result.get("answer") or ""
        retrieved_paper_ids = _dedupe_paper_ids([hit.paper_id for hit in result.get("evidence", [])])
        cited_paper_ids = _dedupe_paper_ids([citation.paper_id for citation in result.get("citations", [])])
        judge = await judge_answer(gold.question, answer, gold)
        diagnostics = _diagnostics_from_result(result, answer)
        failure_class = _classify_result(
            result=result,
            answer=answer,
            retrieved_paper_ids=retrieved_paper_ids,
            cited_paper_ids=cited_paper_ids,
            judge=judge,
            citation_parse_count=diagnostics["citation_parse_count"],
        )
        row = _row_from_result(
            gold,
            answer,
            retrieved_paper_ids,
            cited_paper_ids,
            judge,
            failure_class=failure_class,
            synthesis_finish_reason=diagnostics["synthesis_finish_reason"],
            citation_parse_count=diagnostics["citation_parse_count"],
            verifier_rejected_count=diagnostics["verifier_rejected_count"],
            parse_failure_count=diagnostics["parse_failure_count"],
            multi_hop_attempts=diagnostics["multi_hop_attempts"],
            multi_hop_expansions=diagnostics["multi_hop_expansions"],
        )
        row.status = result.get("status", "")
        row.stop_reason = result.get("stop_reason", "")
        _create_langsmith_run(gold, row, langsmith_state=langsmith_state)
        return row
    finally:
        if row is not None:
            row.token_usage = {
                role: {key: value - usage_before.get(role, {}).get(key, 0)
                       for key, value in data.items() if isinstance(value, (int, float))}
                for role, data in usage.snapshot().items()
            }
            row.estimated_cost_usd = {role: value - costs_before.get(role, 0.0)
                                      for role, value in usage.estimated_cost_usd().items()}
        if os.getenv("DIAGNOSTIC_LOG"):
            if previous_diagnostic_question_id is None:
                os.environ.pop("DIAGNOSTIC_QUESTION_ID", None)
            else:
                os.environ["DIAGNOSTIC_QUESTION_ID"] = previous_diagnostic_question_id
        await _cleanup_eval_run(thread_id)


def _row_from_result(
    gold: GoldQuestion,
    answer: str,
    retrieved_paper_ids: list[str],
    cited_paper_ids: list[str],
    judge: JudgeVerdict,
    failure_class: str = "",
    synthesis_finish_reason: str = "",
    citation_parse_count: int = 0,
    verifier_rejected_count: int = 0,
    parse_failure_count: int = 0,
    multi_hop_attempts: int = 0,
    multi_hop_expansions: int = 0,
) -> EvalRow:
    return EvalRow(
        id=gold.id,
        question=gold.question,
        expected_paper_ids=gold.expected_paper_ids,
        retrieved_paper_ids=retrieved_paper_ids,
        cited_paper_ids=cited_paper_ids,
        recall_at_5=recall_at_k(retrieved_paper_ids, gold.expected_paper_ids, 5),
        mrr=mrr(retrieved_paper_ids, gold.expected_paper_ids),
        citation_precision=citation_precision(cited_paper_ids, gold.expected_paper_ids),
        citation_recall=citation_recall(cited_paper_ids, gold.expected_paper_ids),
        judge_correctness=judge.correctness,
        judge_groundedness=judge.groundedness,
        judge_completeness=judge.completeness,
        judge_rationale=judge.rationale,
        answer=answer,
        hard_failure=failure_class in HARD_FAILURE_CLASSES,
        failure_class=failure_class,
        synthesis_finish_reason=synthesis_finish_reason,
        citation_parse_count=citation_parse_count,
        verifier_rejected_count=verifier_rejected_count,
        parse_failure_count=parse_failure_count,
        multi_hop_attempts=multi_hop_attempts,
        multi_hop_expansions=multi_hop_expansions,
    )


def _failure_row(gold: GoldQuestion, exc: Exception) -> EvalRow:
    message = f"AGENT_RUN_FAILED: {type(exc).__name__}: {exc}"
    return EvalRow(
        id=gold.id,
        question=gold.question,
        expected_paper_ids=gold.expected_paper_ids,
        retrieved_paper_ids=[],
        cited_paper_ids=[],
        recall_at_5=0.0,
        mrr=0.0,
        citation_precision=0.0,
        citation_recall=0.0,
        judge_correctness=1,
        judge_groundedness=1,
        judge_completeness=1,
        judge_rationale=message,
        answer=message,
        hard_failure=True,
        failure_class="agent_exception",
    )


def _select_questions(*, limit: int | None, question_ids: list[str] | None) -> list[GoldQuestion]:
    questions = GOLD_QUESTIONS
    if question_ids:
        wanted = set(question_ids)
        questions = [question for question in questions if question.id in wanted]
    if limit is not None:
        questions = questions[:limit]
    return questions


def _read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["id"] for row in reader if row.get("id")}


def _load_existing_rows(selected_ids: set[str]) -> list[EvalRow]:
    if not PER_QUESTION_CSV.exists():
        return []
    rows: list[EvalRow] = []
    with PER_QUESTION_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if raw["id"] not in selected_ids:
                continue
            rows.append(_row_from_csv_dict(raw))
    return sorted(rows, key=lambda row: row.id)


def _row_to_csv_dict(row: EvalRow) -> dict[str, Any]:
    data = asdict(row)
    data["expected_paper_ids"] = json.dumps(row.expected_paper_ids)
    data["retrieved_paper_ids"] = json.dumps(row.retrieved_paper_ids)
    data["cited_paper_ids"] = json.dumps(row.cited_paper_ids)
    data["token_usage"] = json.dumps(row.token_usage)
    data["estimated_cost_usd"] = json.dumps(row.estimated_cost_usd)
    return data


def _row_from_csv_dict(raw: dict[str, str]) -> EvalRow:
    return EvalRow(
        id=raw["id"],
        question=raw["question"],
        expected_paper_ids=json.loads(raw["expected_paper_ids"]),
        retrieved_paper_ids=json.loads(raw["retrieved_paper_ids"]),
        cited_paper_ids=json.loads(raw["cited_paper_ids"]),
        recall_at_5=float(raw["recall_at_5"]),
        mrr=float(raw["mrr"]),
        citation_precision=float(raw["citation_precision"]),
        citation_recall=float(raw["citation_recall"]),
        judge_correctness=int(raw["judge_correctness"]),
        judge_groundedness=int(raw["judge_groundedness"]),
        judge_completeness=int(raw["judge_completeness"]),
        judge_rationale=raw["judge_rationale"],
        answer=raw["answer"],
        hard_failure=(raw.get("hard_failure") or "").lower() == "true",
        failure_class=raw.get("failure_class", ""),
        synthesis_finish_reason=raw.get("synthesis_finish_reason", ""),
        citation_parse_count=int(raw.get("citation_parse_count") or 0),
        verifier_rejected_count=int(raw.get("verifier_rejected_count") or 0),
        parse_failure_count=int(raw.get("parse_failure_count") or 0),
        multi_hop_attempts=int(raw.get("multi_hop_attempts") or 0),
        multi_hop_expansions=int(raw.get("multi_hop_expansions") or 0),
        status=raw.get("status", ""), stop_reason=raw.get("stop_reason", ""),
        token_usage=json.loads(raw.get("token_usage") or "{}"),
        estimated_cost_usd=json.loads(raw.get("estimated_cost_usd") or "{}"),
    )


def _write_summary(rows: list[EvalRow], *, total_skipped_via_resume: int) -> dict[str, Any]:
    if not rows:
        summary: dict[str, Any] = {
            "total_questions": 0,
            "total_skipped_via_resume": total_skipped_via_resume,
            "avg_recall_at_5": 0.0,
            "avg_mrr": 0.0,
            "avg_citation_precision": 0.0,
            "avg_citation_recall": 0.0,
            "avg_correctness": 0.0,
            "avg_groundedness": 0.0,
            "avg_completeness": 0.0,
            "hard_failure_count": 0,
            "hard_failure_rate": 0.0,
            "hard_failure_threshold": HARD_FAILURE_THRESHOLD,
            "failure_classes": {},
            "multi_hop_attempts_total": 0,
            "multi_hop_expansions_total": 0,
        }
    else:
        failure_classes = _failure_class_counts(rows)
        hard_failure_count = sum(1 for row in rows if row.hard_failure)
        summary = {
            "total_questions": len(rows),
            "total_skipped_via_resume": total_skipped_via_resume,
            "avg_recall_at_5": mean(row.recall_at_5 for row in rows),
            "avg_mrr": mean(row.mrr for row in rows),
            "avg_citation_precision": mean(row.citation_precision for row in rows),
            "avg_citation_recall": mean(row.citation_recall for row in rows),
            "avg_correctness": mean(row.judge_correctness for row in rows),
            "avg_groundedness": mean(row.judge_groundedness for row in rows),
            "avg_completeness": mean(row.judge_completeness for row in rows),
            "hard_failure_count": hard_failure_count,
            "hard_failure_rate": hard_failure_count / len(rows),
            "hard_failure_threshold": HARD_FAILURE_THRESHOLD,
            "failure_classes": failure_classes,
            "multi_hop_attempts_total": sum(row.multi_hop_attempts for row in rows),
            "multi_hop_expansions_total": sum(row.multi_hop_expansions for row in rows),
            "estimated_run_level_calls": len(rows),
            "estimated_judge_calls": len(rows),
            "cost_note": "Token usage is recorded when provider responses expose usage; rerank cost is estimated per call.",
        }
    accumulated_usage, accumulated_cost = {}, {}
    for row in rows:
        for role, data in row.token_usage.items():
            counters = accumulated_usage.setdefault(role, {})
            for key, value in data.items():
                counters[key] = counters.get(key, 0) + value
        for role, value in row.estimated_cost_usd.items():
            accumulated_cost[role] = accumulated_cost.get(role, 0.0) + value
    summary["token_usage"] = accumulated_usage
    summary["estimated_cost_usd"] = accumulated_cost
    summary["delivery_status_counts"] = {status: sum(r.status == status for r in rows)
                                         for status in ("complete", "partial", "incomplete", "")}
    summary["cost_note"] = "Per-question usage and peak-rate estimates include resumed rows; uncertain request charges require the separate request budget ledger. Cohere is Trial."
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _diagnostics_from_result(result: dict[str, Any], answer: str) -> dict[str, int | str]:
    citation_parse_count = len(list(CITATION_RE.finditer(answer)))
    cited_count = len(result.get("citations", []))
    draft_report = result.get("draft_verification") or result.get("verification")
    return {
        "synthesis_finish_reason": str(result.get("synthesis_finish_reason") or ""),
        "citation_parse_count": citation_parse_count,
        "verifier_rejected_count": (sum(not v.supports for v in draft_report.verdicts)
                                     if draft_report is not None else max(citation_parse_count - cited_count, 0)),
        "parse_failure_count": _parse_failure_count(draft_report),
        "multi_hop_attempts": int(result.get("multi_hop_attempts", 0) or 0),
        "multi_hop_expansions": int(result.get("multi_hop_expansions", 0) or 0),
    }


def _classify_result(
    *,
    result: dict[str, Any],
    answer: str,
    retrieved_paper_ids: list[str],
    cited_paper_ids: list[str],
    judge: JudgeVerdict,
    citation_parse_count: int,
) -> str:
    if _judge_invalid(judge):
        return "judge_invalid"
    if result.get("stop_reason") == "evidence_judgment_failed":
        return "evidence_judgment_failed"
    if result.get("synthesis_format_degraded"):
        if not answer.strip():
            return "synthesis_empty"
        if citation_parse_count == 0:
            return "citation_parse_empty"
    if not retrieved_paper_ids:
        return "retrieval_empty"
    if citation_parse_count > 0 and not cited_paper_ids:
        return "verifier_all_rejected"
    return ""


def _judge_invalid(judge: JudgeVerdict) -> bool:
    return judge.rationale.startswith("Judge returned invalid JSON twice")


def _parse_failure_count(verification) -> int:
    if verification is None:
        verdicts = []
    elif isinstance(verification, dict):
        verdicts = verification.get("verdicts", [])
    else:
        verdicts = getattr(verification, "verdicts", [])
    return sum(
        1
        for verdict in verdicts
        if str(verdict.get("rationale", "") if isinstance(verdict, dict) else getattr(verdict, "rationale", "")).startswith(
            "PARSE_FAILURE:"
        )
    )


def _failure_class_counts(rows: list[EvalRow]) -> dict[str, int]:
    classes = ["", *sorted(HARD_FAILURE_CLASSES | WARNING_CLASSES)]
    counts = {failure_class: 0 for failure_class in classes}
    for row in rows:
        counts[row.failure_class] = counts.get(row.failure_class, 0) + 1
    return counts


def _setup_langsmith_dataset(questions: list[GoldQuestion]) -> "_LangSmithState | None":
    # Dataset writes are a separate opt-in from passive trace configuration.
    if os.getenv("ARA_EVAL_LANGSMITH_EXPORT", "").lower() != "true":
        return None
    try:
        settings = get_settings()
        api_key = settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else None
        client = Client(api_key=api_key)
        dataset = _get_or_create_dataset(client)
        existing_example_ids = _existing_example_ids(client, dataset.id)
        example_ids: dict[str, str] = {}
        for gold in questions:
            if gold.id in existing_example_ids:
                example_ids[gold.id] = existing_example_ids[gold.id]
                continue
            example_id = str(uuid5(NAMESPACE_URL, f"{DATASET_NAME}:{gold.id}"))
            try:
                client.create_example(
                    example_id=example_id,
                    dataset_id=dataset.id,
                    inputs={"question": gold.question},
                    outputs={
                        "expected_paper_ids": gold.expected_paper_ids,
                        "answer_must_mention": gold.answer_must_mention,
                    },
                    metadata={"id": gold.id, "category": gold.category},
                )
            except Exception:
                client.update_example(
                    example_id,
                    inputs={"question": gold.question},
                    outputs={
                        "expected_paper_ids": gold.expected_paper_ids,
                        "answer_must_mention": gold.answer_must_mention,
                    },
                    metadata={"id": gold.id, "category": gold.category},
                )
            example_ids[gold.id] = example_id
        return _LangSmithState(client=client, example_ids=example_ids)
    except Exception as exc:
        logger.warning("langsmith_dataset_setup_failed", extra={"error": str(exc)})
        return None


def _get_or_create_dataset(client: Client):
    try:
        return client.create_dataset(
            DATASET_NAME,
            description="Gold evaluation questions for arxiv-research-agent.",
        )
    except Exception:
        for dataset in client.list_datasets(dataset_name=DATASET_NAME, limit=1):
            return dataset
        raise


def _existing_example_ids(client: Client, dataset_id) -> dict[str, str]:
    examples: dict[str, str] = {}
    try:
        for example in client.list_examples(dataset_id=dataset_id):
            metadata = getattr(example, "metadata", None) or {}
            gold_id = metadata.get("id")
            if gold_id:
                examples[gold_id] = str(example.id)
    except Exception as exc:
        logger.warning("langsmith_example_list_failed", extra={"error": str(exc)})
    return examples


def _create_langsmith_run(
    gold: GoldQuestion,
    row: EvalRow,
    *,
    langsmith_state: "_LangSmithState | None",
) -> None:
    if langsmith_state is None:
        return
    try:
        settings = get_settings()
        langsmith_state.client.create_run(
            name=f"eval-{gold.id}",
            run_type="chain",
            project_name=settings.langsmith_project,
            inputs={"question": gold.question},
            outputs=asdict(row),
            reference_example_id=langsmith_state.example_ids.get(gold.id),
        )
    except Exception as exc:
        logger.warning("langsmith_run_create_failed", extra={"id": gold.id, "error": str(exc)})


async def _cleanup_eval_run(thread_id: str) -> None:
    async with db.acquire_app() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM sessions WHERE session_id = $1", thread_id)
            await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = $1", thread_id)
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = $1", thread_id)
            await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = $1", thread_id)


def _dedupe_paper_ids(paper_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for paper_id in paper_ids:
        if paper_id in seen:
            continue
        seen.add(paper_id)
        deduped.append(paper_id)
    return deduped


@dataclass
class _LangSmithState:
    client: Client
    example_ids: dict[str, str]
