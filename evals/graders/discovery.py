"""Discovery metrics (DESIGN §11.3). The product lists at most 5 papers, while PaSa gold sets are
larger, so search is graded by gold recall in the candidate pool and selection by the precision of
the final list. Gold is pooled, not exhaustive: gold precision is a lower bound until a judge
adjudicates the non-gold papers (E1)."""

from collections.abc import Sequence


def score(
    candidates: Sequence[str],
    shortlisted: Sequence[str],
    listed: Sequence[str],
    gold: frozenset[str],
) -> dict[str, float]:
    metrics = {
        "pool_recall": len(gold & set(candidates)) / len(gold),
        "shortlist_recall": len(gold & set(shortlisted)) / len(gold),
        "hit_at_5": float(bool(gold & set(listed[:5]))),
        "listed": float(len(listed)),
    }
    if listed:  # undefined for an empty list, which `listed` shows
        metrics["gold_precision_at_5"] = len(gold & set(listed[:5])) / len(listed[:5])
    return metrics
