import pytest

from src.corpus.html_loader import load_html
from src.corpus.chunking import chunk_sections


def document(body, identifier="2408.15664v1"):
    return ('<html><body><nav>Search navigation</nav><div>arXiv:' + identifier
            + '</div><article class="ltx_document"><h1>A paper</h1><h2>3 Method</h2>'
            + body + '<p>' + 'Independent contextual source sentence. ' * 10 + '</p></article></body></html>')


def test_arxiv_math_retains_overbar_sign_and_algorithm_context_after_chunking():
    html = document(r'<p>Calculate the load error <math alttext="e_i=\overline{c_i}-c_i"><mi>e</mi></math>.</p>'
                    r'<p>Update the bias <math alttext="b_i=b_i+u\operatorname{sign}(e_i)"><mi>b</mi></math>.</p>')
    sections = load_html(html, expected_id="2408.15664v1")
    text = ' '.join(c.text for c in chunk_sections(sections))
    assert r'e_i=\overline{c_i}-c_i' in text
    assert r'b_i=b_i+u\operatorname{sign}(e_i)' in text
    assert 'Search navigation' not in text
    assert sections[0].name == '3 Method'


def test_version_mismatch_and_error_page_cannot_become_original_text():
    with pytest.raises(ValueError, match='version identity'):
        load_html(document('<p>Body</p>', identifier='2408.15664v2'), expected_id='2408.15664v1')
    with pytest.raises(ValueError, match='article'):
        load_html('<html><body>Service unavailable</body></html>')


def test_math_without_a_text_representation_is_explicitly_unavailable():
    sections = load_html(document('<p>Formula: <math><mi>e</mi><mi>c</mi><mi>c</mi></math>.</p>'))
    assert 'Mathematical notation unavailable' in sections[0].text
    assert 'ecc' not in sections[0].text
