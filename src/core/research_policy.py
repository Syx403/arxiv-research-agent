"""A per-turn policy, snapshotted into graph state for deterministic resume."""

import calendar
from datetime import date, timedelta
import re

from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import get_settings
from src.core.run_context import today as run_today


class ResearchPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timeout_s: float = Field(default=90, gt=0, le=180)
    reading_timeout_s: float = Field(default=300, gt=0, le=300)
    min_optional_recovery_s: float = Field(default=30, ge=0, le=180)
    max_evidence_recovery_s: float = Field(default=30, gt=0, le=90)
    max_papers: int = Field(default=3, ge=1, le=3)
    max_candidates: int = Field(default=60, ge=1, le=80)
    max_queries: int = Field(default=8, ge=1, le=12)
    max_assessments: int = Field(default=3, ge=1, le=4)
    search_budget_usd: float = Field(default=0.05, gt=0, le=0.05)
    reading_budget_usd: float = Field(default=0.15, gt=0, le=0.15)

    @classmethod
    def configured(cls):
        settings = get_settings()
        return cls(timeout_s=settings.discovery_timeout_s, max_papers=settings.discovery_max_papers)


# Tests replace the acquisition/model adapters; production has one arXiv workflow.
_policy: ContextVar[ResearchPolicy] = ContextVar("research_policy", default=ResearchPolicy())


def current_policy() -> ResearchPolicy:
    return _policy.get()


@contextmanager
def use_research_policy(policy: ResearchPolicy):
    token = _policy.set(policy)
    try:
        yield policy
    finally:
        _policy.reset(token)


def date_window(question: str, today: date | None = None) -> tuple[str | None, str]:
    today = today or run_today()
    years = [int(y) for y in re.findall(r"(?<![\d.])(20[0-9]{2})(?![\d.])", question)]
    fresh = bool(
        re.search(
            r"最新|最近|近期|今年|去年|过去.*(?:月|年)|latest|recent|newest|current|this year|last year|past.*(?:month|year)",
            question,
            re.I,
        )
    )
    dates = re.findall(r"(?<!\d)20\d{2}-\d{2}-\d{2}(?!\d)", question)
    relative = re.search(
        r"(?:过去|最近|近|past|last)\s*(\d+|一|两|二|三|one|two|three|a)\s*(个月|月|年|months?|years?)",
        question,
        re.I,
    )
    if len(dates) >= 2:
        try:
            parsed_dates = [date.fromisoformat(value).isoformat() for value in dates]
            return min(parsed_dates), min(max(parsed_dates), today.isoformat())
        except ValueError:
            pass
    if years:
        start = date(min(years), 1, 1)
        end = min(today, date(max(years), 12, 31))
    elif "去年" in question or "last year" in question.lower():
        start, end = date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)
    elif "今年" in question or "this year" in question.lower():
        start, end = date(today.year, 1, 1), today
    elif relative:
        value, unit = relative.groups()
        count = (
            int(value)
            if value.isdigit()
            else {"一": 1, "两": 2, "二": 2, "三": 3, "one": 1, "two": 2, "three": 3, "a": 1}[
                value.lower()
            ]
        )
        months = min(1200, max(1, count) * (12 if unit.lower() in {"年", "year", "years"} else 1))
        year, month = divmod(today.year * 12 + today.month - 1 - months, 12)
        start, end = (
            date(year, month + 1, min(today.day, calendar.monthrange(year, month + 1)[1])),
            today,
        )
    elif fresh:
        start, end = today - timedelta(days=365), today
    else:
        return None, today.isoformat()
    return start.isoformat(), end.isoformat()


def requested_ids(question: str) -> list[str]:
    matches = re.findall(r"(?<![\d.])\d{4}\.\d{4,5}(?:v\d+)?(?!\d)", question)
    return list(dict.fromkeys(re.sub(r"v\d+$", "", match) for match in matches))[:3]


def date_mode(question: str) -> str:
    """A freshness preference is not permission to exclude foundational work."""
    lower, _ = date_window(question)
    if not lower:
        return "none"
    explicit = re.search(
        r"(?<![\d.])20\d{2}(?![\d.])|今年|去年|this year|last year|"
        r"(?:过去|最近|近)\s*(?:\d+|[一二两三四五六七八九十]+)\s*(?:个月|月|年)|"
        r"(?:past|last)\s+(?:\d+|one|two|three|a)\s+(?:months?|years?)",
        question,
        re.I,
    )
    return "required" if explicit else "preferred"


def prefers_recent(question: str) -> bool:
    """A date filter alone does not mean sort by newest rather than relevance."""
    return bool(
        re.search(r"最新|最近|近期|今年|latest|recent|newest|current|this year", question, re.I)
    )
