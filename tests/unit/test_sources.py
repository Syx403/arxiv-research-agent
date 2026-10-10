from typing import Any

from ara.rag.sources import _pdf_paragraphs, arxiv_html_paper, qasper_paper

HTML = """
<html><body><article class="ltx_document">
<h1 class="ltx_title ltx_title_document">KV Cache Eviction</h1>
<div class="ltx_abstract"><h6>Abstract</h6><p class="ltx_p">We evict keys.</p></div>
<section class="ltx_section"><h2 class="ltx_title"><span class="ltx_tag">1</span>Method</h2>
  <div class="ltx_para"><p class="ltx_p">The score is <math alttext="s_i=q\\cdot k_i">s</math>
  per key.<span class="ltx_note">A footnote.</span></p></div>
  <section class="ltx_subsection"><h3 class="ltx_title">Budget</h3>
    <div class="ltx_para"><p class="ltx_p">We keep 20% of keys.</p>
      <div class="ltx_para"><p class="ltx_p">Nested text is part of its parent.</p></div></div>
    <figure class="ltx_table"><figcaption>Table 1: Results.</figcaption></figure>
  </section>
</section>
<section class="ltx_bibliography"><div class="ltx_para"><p>[1] A reference.</p></div></section>
</article></body></html>
"""


def test_arxiv_html_keeps_math_source_and_heading_paths() -> None:
    paper = arxiv_html_paper(HTML, "2401.00001", 2)
    assert (paper.id, paper.title, paper.abstract) == (
        "arxiv:2401.00001v2",
        "KV Cache Eviction",
        "We evict keys.",
    )
    assert [(p.heading_path, p.text) for p in paper.paragraphs] == [
        ("KV Cache Eviction › Abstract", "We evict keys."),
        ("KV Cache Eviction › Method", "The score is $s_i=q\\cdot k_i$ per key."),
        (
            "KV Cache Eviction › Method › Budget",
            "We keep 20% of keys. Nested text is part of its parent.",
        ),
        ("KV Cache Eviction › Method › Budget › Figure or table caption", "Table 1: Results."),
    ]


RECORD: dict[str, Any] = {
    "id": "1912.01214",
    "title": "Cross-lingual Transfer",
    "abstract": "We transfer.",
    "full_text": {
        "section_name": ["Introduction", "Approach ::: Pre-training", None],
        "paragraphs": [["First paragraph."], ["Second.", "  "], ["Orphan."]],
    },
}


def test_qasper_keeps_paragraphs_verbatim_with_subsection_paths() -> None:
    paper = qasper_paper(RECORD)
    assert (paper.id, paper.version, paper.format) == ("qasper:1912.01214", None, "qasper")
    assert [(p.heading_path, p.text) for p in paper.paragraphs] == [
        ("Cross-lingual Transfer › Abstract", "We transfer."),
        ("Cross-lingual Transfer › Introduction", "First paragraph."),
        ("Cross-lingual Transfer › Approach › Pre-training", "Second."),
        ("Cross-lingual Transfer", "Orphan."),
    ]


def test_pdf_lines_regroup_into_paragraphs() -> None:
    header, full = "Published at ICLR", "x" * 60
    pages = [
        [header, f"{full} chain-of-", "thought is", f"{full} end.", "Short last line.", "1"],
        [header, "\x14\x03garbled figure text", f"{full} goes on", "and ends here.", "2"],
        [header, "Final words.", "3"],
    ]
    assert _pdf_paragraphs(pages) == [
        f"{full} chain-of-thought is {full} end. Short last line.",
        f"{full} goes on and ends here.",
        "Final words.",
    ]
