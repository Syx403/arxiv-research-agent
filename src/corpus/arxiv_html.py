"""Small parsers for official arXiv search/abstract pages used during API outages.

Keep this adapter separate: HTML is less stable than Atom. Unexpected markup is
an adapter error, never an empty search result.
"""
from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
import re


class Element:
    def __init__(self, tag="", attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def text(self):
        return "".join(child if isinstance(child, str) else child.text() for child in self.children)

    def find(self, tag=None, css_class=None):
        for child in self.children:
            if isinstance(child, Element):
                if (tag is None or child.tag == tag) and (css_class is None or css_class in child.attrs.get("class", "").split()):
                    yield child
                yield from child.find(tag, css_class)


class Document(HTMLParser):
    def __init__(self, content):
        super().__init__(convert_charrefs=True)
        self.root = Element()
        self.stack = [self.root]
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        element = Element(tag, attrs)
        self.stack[-1].children.append(element)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append(element)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def parse_search(content: str) -> list[dict]:
    document = Document(content)
    entries = list(document.root.find("li", "arxiv-result"))
    if not entries and not re.search(r"(no results|0 results|did not produce any results)", document.root.text(), re.I):
        raise ValueError("Unrecognized arXiv search response")
    results = []
    for entry in entries:
        links = [link.attrs.get("href", "") for link in entry.find("a")]
        url = next((link for link in links if re.fullmatch(r"https://arxiv\.org/abs/[^?#]+", link)), None)
        titles = list(entry.find("p", "title"))
        abstracts = list(entry.find("span", "abstract-full"))
        if not url or not titles or not abstracts:
            raise ValueError("Incomplete arXiv search entry")
        submitted = re.search(r"Submitted\s+(\d+\s+[A-Za-z]+,\s+\d{4})", entry.text())
        published = datetime.strptime(submitted.group(1), "%d %B, %Y").date() if submitted else None
        results.append({"url": url, "title": " ".join(titles[0].text().split()),
                        "abstract": re.sub(r"\s*△?\s*Less\s*$", "", abstracts[0].text()).strip(),
                        "authors": [" ".join(author.text().split()) for group in entry.find("p", "authors") for author in group.find("a")],
                        "published_at": published})
    return results


def parse_abstract(content: str) -> dict:
    document = Document(content)
    metadata = {}
    for element in document.root.find("meta"):
        name = element.attrs.get("name") or element.attrs.get("property")
        if name:
            metadata.setdefault(name, []).append(element.attrs.get("content", ""))
    try:
        title = metadata["citation_title"][0]
        identifier = metadata["citation_arxiv_id"][0]
        published = datetime.strptime(metadata["citation_date"][0], "%Y/%m/%d").date()
        abstract = next(document.root.find("blockquote", "abstract")).text()
    except (KeyError, IndexError, StopIteration) as exc:
        raise ValueError("Incomplete arXiv abstract page") from exc
    base = re.sub(r'v\d+$', '', identifier)
    version = re.search(re.escape(base) + r'v(\d+)', identifier + ' ' + document.root.text())
    updated = metadata.get('citation_online_date', [''])[0]
    return {"arxiv_id": identifier, "title": title,
            "version": int(version[1]) if version else None,
            "version_id": base + 'v' + version[1] if version else None,
            "updated": updated.replace('/', '-') or None,
            "published_at": published, "authors": metadata.get("citation_author", []),
            "abstract": re.sub(r"^\s*Abstract:\s*", "", abstract).strip()}
