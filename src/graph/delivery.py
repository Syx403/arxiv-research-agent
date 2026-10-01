"""Extract fully checked sentences without detaching prose from its sources."""
from src.graph.nodes.synthesize import _parse_citations
from src.core.answer_structure import retained_table_structure, table_context, tables


def supported_text(draft, report):
    grouped = {}
    for verdict in report.verdicts if report else []:
        span = verdict.citation.claim_span
        if span and 0 <= span[0] < span[1] <= len(draft):
            if draft[span[0]:span[1]].strip() == verdict.citation.claim_text:
                grouped.setdefault(span, []).append(verdict)
    kept, seen, retained_claims = [], set(), set()
    for (start, end), checks in sorted(grouped.items()):
        sentence = draft[start:end].strip()
        expected = _parse_citations(sentence)
        checked = {(v.citation.paper_id, v.citation.chunk_id) for v in checks}
        if expected and all(v.supports and v.status == "ok" and v.context_verified
                            and set(v.context_claims) <= retained_claims
                            and v.scope_status not in {"unrequested", "duplicate"}
                            for v in checks) and all(
            (c.paper_id, c.chunk_id) in checked for c in expected
        ):
            key = (sentence, table_context(draft, (start, end)))
            if key not in seen:
                kept.append((start, end))
                seen.add(key)
                retained_claims.add(checks[0].citation.claim_text)
    kept.extend(retained_table_structure(draft, list(kept)))
    spans = []
    for start, end in sorted(set(kept)):
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
        else:
            spans.append((start, end))
    table_ranges = [(header[0], body[-1][1]) for header, _, body in tables(draft)]
    result, previous_owner = "", None
    for start, end in spans:
        text = draft[start:end].strip()
        owner = next((i for i, (a, b) in enumerate(table_ranges) if a <= start < b), None)
        separator = "\n" if owner is not None and owner == previous_owner else "\n\n"
        result += (separator if result else "") + text
        previous_owner = owner
    return result


def delivery_score(text):
    return (len({c.claim_text for c in _parse_citations(text)}), len(text))
