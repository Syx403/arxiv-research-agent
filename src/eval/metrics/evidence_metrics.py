"""Source-span diagnostics; operational unknowns are separate from relevance."""
from __future__ import annotations

from src.retrieval.evidence import evidence_text


def evidence_metrics(case, selected, verdicts=None):
    relevant=set(case['relevant_ids'])
    selected_ids={h.chunk_id for h in selected}
    by_id={h.chunk_id:evidence_text(h) for h in selected}
    spans=case.get('required_spans',[])
    covered=sum(s['text'] in by_id.get(s['chunk_id'],'') for s in spans)
    errors=sum(v.status=='error' for v in (verdicts or []))
    return {
        'selected_count':len(selected),
        'irrelevant_selected':len(selected_ids-relevant),
        'relevant_selected':len(selected_ids & relevant),
        'required_spans':len(spans),
        'covered_spans':covered,
        'judgment_unknowns':errors,
        'abstained':not selected,
    }
