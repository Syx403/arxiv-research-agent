"""One bounded pass over already retrieved originals when selected excerpts lack context."""
from dataclasses import replace
import re

from src.core.trace import record_event
from src.core.types import EvidenceChunk
from src.retrieval.evidence import (
    EVIDENCE_CHUNK_TOKENS, EVIDENCE_SUBQUESTION_TOKENS, EVIDENCE_TOTAL_TOKENS,
    evidence_text, merge_spans, select_sentences, sentence_spans, token_count,
)
from src.retrieval.index.types import Hit
from src.retrieval.self_rag.relevance_judge import judge_batch


async def recover_evidence(state, report):
    if state.get("verification_recovery_attempted"):
        return {}
    originals = {h.chunk_id: h for h in state.get("evidence", []) if isinstance(h, Hit)}
    lookup = state.get("evidence_lookup", {})
    claims, cited_ids, paper_ids = [], set(), set()
    for verdict in report.verdicts:
        citation = verdict.citation
        hit, bound = originals.get(citation.chunk_id), lookup.get(citation.chunk_id)
        if (verdict.supports or verdict.status != "ok" or verdict.scope_status in {"unrequested", "duplicate"}
                or hit is None or not isinstance(bound, EvidenceChunk)):
            continue
        if citation.paper_id != hit.paper_id or citation.paper_id != bound.paper_id or not hit.evidence_spans:
            continue
        if verdict.rationale.startswith(("source_", "claim_span_", "empty_cited_", "joint_claim_")):
            continue
        cited_ids.add(citation.chunk_id)
        paper_ids.add(citation.paper_id)
        item = citation.claim_text + "\nMissing/contested support: " + verdict.rationale
        if item not in claims:
            claims.append(item)
    if not claims:
        return {}
    # The draft may cite the wrong fragment of the correct paper. Include other
    # already bound chunks of that paper, prioritizing missing lexical context.
    words = set(re.findall(r"[a-z]{4,}", " ".join(claims).lower())) - {
        "claim", "evidence", "paper", "source", "cited", "excerpt", "excerpts",
        "support", "supports", "supported", "directly", "establish", "states", "that", "this",
    }
    candidates = [h for h in originals.values() if h.paper_id in paper_ids and h.chunk_id in lookup
                  and isinstance(lookup[h.chunk_id], EvidenceChunk)
                  and lookup[h.chunk_id].paper_id == h.paper_id and lookup[h.chunk_id].chunk_id == h.chunk_id
                  and h.evidence_spans and any(
                      not any(old.start <= span.start and old.end >= span.end for old in h.evidence_spans)
                      for span in sentence_spans(h.text))]
    def priority(hit):
        raw = set(re.findall(r"[a-z]{4,}", hit.text.lower()))
        selected = set(re.findall(r"[a-z]{4,}", evidence_text(hit).lower()))
        return len((raw - selected) & words), hit.chunk_id in cited_ids
    targets = {h.chunk_id: h for h in sorted(candidates, key=priority, reverse=True)[:3]}
    if not targets:
        return {}
    question = (
        "Check these contested claims against the ORIGINAL passages. Select missing mechanism details, "
        "qualifying conditions, exceptions OR counterevidence; do not presume the claims are true. "
        "Previously selected excerpts may have omitted context.\n" + "\n\n".join(claims)[:6000]
    )
    decisions = await judge_batch(question, list(targets.values()))
    replacements = {}
    groups = state.get("subq_results", {})
    for hit, decision in zip(targets.values(), decisions):
        if decision.status != "ok" or decision.relevant is not True:
            continue
        selected = select_sentences(hit, decision.sentence_ids, role=hit.evidence_role)
        for span in selected.evidence_spans:
            current = replacements.get(hit.chunk_id, hit)
            candidate = replace(current, evidence_spans=merge_spans([*current.evidence_spans, span]))
            proposal = {**replacements, hit.chunk_id: candidate}
            if token_count(evidence_text(candidate)) > EVIDENCE_CHUNK_TOKENS:
                continue
            if sum(token_count(evidence_text(proposal.get(h.chunk_id, h))) for h in originals.values()) > EVIDENCE_TOTAL_TOKENS:
                continue
            if any(sum(token_count(evidence_text(proposal.get(h.chunk_id, h))) for h in group.hits)
                   > EVIDENCE_SUBQUESTION_TOKENS for group in groups.values()):
                continue
            replacements[hit.chunk_id] = candidate
    replacements = {cid: h for cid, h in replacements.items() if h.evidence_spans != originals[cid].evidence_spans}
    record_event("verification_evidence_recovery", attempted_chunk_ids=list(targets),
                 expanded_chunk_ids=list(replacements),
                 judgment_errors=sum(d.status == "error" for d in decisions))
    if not replacements:
        return {"verification_recovery_attempted": True}
    new_lookup = dict(lookup)
    for cid, hit in replacements.items():
        new_lookup[cid] = EvidenceChunk(
            paper_id=hit.paper_id, chunk_id=cid, text=evidence_text(hit), title=hit.title,
            published_at=hit.published_at, url=hit.url,
            source_spans=[(s.start, s.end) for s in hit.evidence_spans],
        )
    return {
        "verification_recovery_attempted": True,
        "evidence": [replacements.get(h.chunk_id, h) for h in state["evidence"]],
        "evidence_lookup": new_lookup,
        "subq_results": {i: group.model_copy(update={"hits": [replacements.get(h.chunk_id, h) for h in group.hits]})
                         for i, group in groups.items()},
    }
