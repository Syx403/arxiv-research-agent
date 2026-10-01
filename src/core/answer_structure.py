"""Table headers are context for checked rows, not an allow-list of label words."""
import re


def table_cells(line):
    value = line.strip()
    if not value.startswith("|") or value.count("|") < 2:
        return []
    return [c.strip() for c in re.split(r"(?<!\\)\|", value.strip("|"))]


def tables(text):
    """Yield exact header, delimiter and contiguous body-row spans."""
    rows, offset = [], 0
    for line in text.splitlines(keepends=True):
        rows.append((offset, offset + len(line.rstrip("\r\n")), line))
        offset += len(line)
    for i in range(1, len(rows)):
        header, divider = rows[i-1], rows[i]
        cells, labels = table_cells(divider[2]), table_cells(header[2])
        if (len(cells) < 2 or len(cells) != len(labels)
                or not all(re.fullmatch(r":?-{3,}:?", c) for c in cells)):
            continue
        body = []
        for row in rows[i+1:]:
            if len(table_cells(row[2])) != len(labels):
                break
            body.append(row[:2])
        if body:
            yield header[:2], divider[:2], body


def table_context(text, span):
    if span:
        for header, _, body in tables(text):
            if any(a <= span[0] < b for a, b in body):
                return text[header[0]:header[1]]
    return ""


def substantive_text(text, start=0, end=None):
    # Headers are checked with each cited row by the source verifier. Removing
    # them from standalone uncited prose does not exempt their factual content.
    spans = [span for header, divider, _ in tables(text) for span in (header, divider)]
    for left, right in spans:
        text = text[:left] + " " * (right-left) + text[right:]
    return text[start:end].strip()


def retained_table_structure(draft, kept):
    for header, divider, body in tables(draft):
        if any(start <= a < end for start, end in body for a, _ in kept):
            yield header
            yield divider
