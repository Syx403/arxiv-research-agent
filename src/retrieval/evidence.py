"""Select original sentences once; carry the same excerpts through every check."""
from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache

import tiktoken

from src.retrieval.index.types import Hit, SourceSpan


EVIDENCE_TOTAL_TOKENS = 4800
EVIDENCE_SUBQUESTION_TOKENS = 1600
EVIDENCE_CHUNK_TOKENS = 400
EVIDENCE_SUBQUESTION_CHUNKS = 8
_SENTENCE_END = re.compile(r'(?<=[.!?。！？])\s+(?=[A-Z0-9“"(])|\n\s*\n')
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "Fig.", "Eq.", "Sec.", "vs.")
_RUNNING_HEADER = re.compile(r"^\d+\s+(?:Published as (?:a )?conference paper|Proceedings of|arXiv:)", re.I)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


@lru_cache(maxsize=1)
def _encoding():
    return tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    # Treat document strings as text, including strings resembling special tokens.
    return len(_encoding().encode(text, disallowed_special=()))


def sentence_spans(text: str) -> list[SourceSpan]:
    """Stable offsets; split oversized PDF sentences at whitespace, never rewrite them."""
    boundaries = [0]
    for match in _SENTENCE_END.finditer(text):
        # PDF footnotes and running headers can interrupt a single sentence.
        # Keep the preceding source unit with its continuation, preserving every
        # original character and the condition before the page break.
        if not text[:match.start()].endswith(_ABBREVIATIONS) and not _RUNNING_HEADER.match(text[match.end():]):
            boundaries.append(match.end())
    boundaries.append(len(text))
    result = []
    for left, right in zip(boundaries, boundaries[1:]):
        start = left
        for word in re.finditer(r"\S+", text[left:right]):
            end = left + word.end()
            if token_count(text[start:end]) > 200 and left + word.start() > start:
                result.append(_trim_span(text, start, left + word.start()))
                start = left + word.start()
        if text[start:right].strip():
            result.append(_trim_span(text, start, right))
    return result


def _trim_span(text: str, start: int, end: int) -> SourceSpan:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return SourceSpan(start, end)


def merge_spans(spans) -> tuple[SourceSpan, ...]:
    result: list[SourceSpan] = []
    for span in sorted(set(spans)):
        if result and span.start <= result[-1].end:
            result[-1] = SourceSpan(result[-1].start, max(result[-1].end, span.end))
        else:
            result.append(span)
    return tuple(result)


def evidence_text(hit: Hit) -> str:
    if not hit.evidence_spans:
        return hit.text
    spans = merge_spans(hit.evidence_spans)
    if any(not 0 <= s.start < s.end <= len(hit.text) for s in spans):
        raise ValueError("Evidence span does not belong to its source chunk")
    return "\n[…]\n".join(hit.text[s.start:s.end] for s in spans)


def select_sentences(hit: Hit, ids: list[int], *, role: str) -> Hit:
    spans = sentence_spans(hit.text)
    if not ids or len(ids) != len(set(ids)) or any(i < 0 or i >= len(spans) for i in ids):
        raise ValueError("Missing, duplicate, or out-of-range source sentence IDs")
    chosen = []
    for i in ids:  # Model priority decides which complete sentences fit.
        candidate = replace(hit, evidence_spans=merge_spans([*chosen, spans[i]]))
        if token_count(evidence_text(candidate)) <= EVIDENCE_CHUNK_TOKENS:
            chosen.append(spans[i])
    if not chosen:
        raise ValueError("No complete selected sentence fits the evidence budget")
    return replace(hit, evidence_spans=merge_spans(chosen), evidence_role=role)


def is_primary_source(question: str, title: str) -> bool:
    """Only explicit full titles or distinctive colon-prefix method names qualify."""
    query = " " + " ".join(_TOKEN_RE.findall(question.lower())) + " "
    name = title.split(":", 1)[0].strip()
    terms = _TOKEN_RE.findall(name.lower())
    if not terms or len(name) < 3:
        return False
    if len(terms) == 1 and ":" not in title:
        return False
    return " " + " ".join(terms) + " " in query


def is_reference_list(hit: Hit, question: str = "") -> bool:
    # Bibliographies remain searchable for explicit citation/bibliography questions.
    if re.search(r"bibliograph|reference list|references cited|参考文献列表|引用列表", question, re.I):
        return False
    section = re.sub(r"[^a-z]", "", (hit.section or "").lower())
    # Mixed sections must go to semantic judging rather than a hard exclusion.
    return section in {"references", "bibliography", "acknowledgments", "acknowledgements"}


def pack_evidence(hits: list[Hit], question: str, *, token_budget: int = EVIDENCE_SUBQUESTION_TOKENS,
                  limit: int = EVIDENCE_SUBQUESTION_CHUNKS) -> list[Hit]:
    ranked = sorted(hits, key=lambda h: (
        h.evidence_role != "background", is_primary_source(question, h.title), h.score
    ), reverse=True)
    # Give explicitly requested original papers a slot before taking additional
    # chunks from one method. The token budget, not a four-chunk cutoff, bounds context.
    first, rest, paper_seen = [], [], set()
    for hit in ranked:
        if is_primary_source(question, hit.title) and hit.paper_id not in paper_seen:
            first.append(hit)
            paper_seen.add(hit.paper_id)
        else:
            rest.append(hit)
    ranked = first + rest
    primary_direct = any(h.evidence_role == "direct" and is_primary_source(question, h.title) for h in hits)
    needs_external_context = bool(re.search(
        r"compar|versus|\bvs\b|critic|subsequent|follow.up|later|external|survey|评价|后续|批评|比较|对比|综述",
        question, re.I,
    ))
    selected, seen, used = [], set(), 0
    for hit in ranked:
        if hit.chunk_id in seen or is_reference_list(hit, question):
            continue
        if (primary_direct and not needs_external_context and hit.evidence_role == "background"
                and not is_primary_source(question, hit.title)):
            # Drop secondary mentions, not complementary evidence from the selected original.
            # A source's introduction/conclusion can answer another requested aspect.
            continue
        # Compatibility for callers supplying old/unselected hits: choose whole
        # query-matching sentences, not a character prefix. Live retrieval supplies IDs.
        if not hit.evidence_spans:
            words = set(_TOKEN_RE.findall(question.lower()))
            spans = sentence_spans(hit.text)
            order = sorted(range(len(spans)), key=lambda i: len(
                words & set(_TOKEN_RE.findall(hit.text[spans[i].start:spans[i].end].lower()))
            ), reverse=True)
            try:
                hit = select_sentences(hit, order, role=hit.evidence_role)
            except ValueError:
                continue
        cost = token_count(evidence_text(hit))
        if used + cost > token_budget:
            continue
        selected.append(hit)
        seen.add(hit.chunk_id)
        used += cost
        if len(selected) >= limit:
            break
    return selected


def format_evidence(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{h.paper_id}#{h.chunk_id}] Title: {h.title or h.paper_id}\n"
        f"Publication metadata (arXiv): {h.published_at or 'unknown'}\n"
        f"Section: {h.section or 'Unknown'}; role: {h.evidence_role}\n{evidence_text(h)}"
        for h in hits
    )
