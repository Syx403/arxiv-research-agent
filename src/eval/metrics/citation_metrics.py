from __future__ import annotations


def citation_precision(cited_paper_ids: list[str], expected: list[str]) -> float:
    cited = set(cited_paper_ids)
    if not cited:
        return 0.0
    return len(cited & set(expected)) / len(cited)


def citation_recall(cited_paper_ids: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    return len(set(cited_paper_ids) & expected_set) / len(expected_set)
