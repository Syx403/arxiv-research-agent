# ruff: noqa: RUF001  (Chinese wording uses full-width punctuation)
"""The fixed sentences of a reply, in the language the user wrote in (D40). The model writes the
rest of the reply in that language; a language without its own wording here gets English. What
failed (a problem's own words, M5a) stays in English."""

from ara.graph.state import ABSTAIN

ENGLISH = "English"

WORDING: dict[str, dict[str, str]] = {
    "abstain": {ENGLISH: ABSTAIN, "Chinese": "所读论文中没有讲到这一点。"},
    "withheld": {
        ENGLISH: "I could not verify a one-line answer, so this is only what the papers' text"
        " supports, sentence by sentence:",
        "Chinese": "我没能核实出一句话的结论，下面只给出逐句经论文原文核实的内容：",
    },
    "unverified": {
        ENGLISH: "I could not verify an answer against the papers' text.",
        "Chinese": "我没能用论文原文核实出一个答案。",
    },
    "other": {
        ENGLISH: "I find arXiv papers and answer questions from their full text. Ask me for papers"
        " on a topic, or to read a paper by its arXiv id.",
        "Chinese": "我负责查找 arXiv 论文，并根据论文全文回答问题。你可以让我找某个主题的论文，"
        "或者给出 arXiv 编号让我读某篇论文。",
    },
    "not_discussed": {
        ENGLISH: "We have not discussed this in any paper we have read."
        " Would you like me to search arXiv for papers on it?",
        "Chinese": "我们读过的论文里没有讨论过这个。要我在 arXiv 上搜索相关论文吗？",
    },
    "unfinished": {
        ENGLISH: "I could not finish this request, so there is nothing to show yet."
        " Please try again.",
        "Chinese": "这个请求没能完成，暂时没有可显示的内容，请再试一次。",
    },
    "no_papers": {
        ENGLISH: "I found no arXiv papers that match this request.",
        "Chinese": "我在 arXiv 上没有找到符合这个请求的论文。",
    },
    "not_found": {
        ENGLISH: "I could not find on arXiv: {titles}.",
        "Chinese": "我在 arXiv 上没有找到：{titles}。",
    },
    "not_searched": {
        ENGLISH: "I could not search arXiv for: {titles}.",
        "Chinese": "我没能在 arXiv 上搜索：{titles}。",
    },
    "identity": {
        ENGLISH: 'I took "{named}" to be "{title}" (arXiv {paper}).',
        "Chinese": "我把“{named}”理解为《{title}》（arXiv {paper}）。",
    },
    "several": {
        ENGLISH: 'I took "{named}" to be "{title}" (arXiv {paper}); {others} other paper(s) have a'
        " title starting with it.",
        "Chinese": "我把“{named}”理解为《{title}》（arXiv {paper}）；"
        "另有 {others} 篇论文的标题也以它开头。",
    },
    "max_read": {
        ENGLISH: "I read at most {n} papers per turn: the first {n} named.",
        "Chinese": "每轮最多读 {n} 篇论文：我读了你提到的前 {n} 篇。",
    },
    "not_read_before": {
        ENGLISH: "We had not read {paper} before; I read it from arXiv now.",
        "Chinese": "我们之前没有读过 {paper}，这次从 arXiv 上读了它。",
    },
    "conflict": {
        ENGLISH: 'Note: "{title}" may break what you asked for ({quotes}), judging from its'
        " abstract; I read it because you named it.",
        "Chinese": "注意：从摘要看，《{title}》可能不符合你的要求（{quotes}）；"
        "因为是你点名的，我还是读了它。",
    },
    "mismatch": {
        ENGLISH: '"{title}" (arXiv {paper}) does not seem to discuss this: nothing in its full text'
        ' was found for the question. Its abstract begins: "{opening}" Check the title or id, or'
        " ask me to search arXiv for papers on it.",
        "Chinese": "《{title}》（arXiv {paper}）似乎没有讨论这个问题：全文中没有找到相关内容。"
        "它的摘要开头是：“{opening}” 请核对标题或编号，或者让我在 arXiv 上搜索相关论文。",
    },
    "searched_instead": {
        ENGLISH: "We read {titles} before, but {subject} not cover {missing}, so I searched arXiv:",
        "Chinese": "我们之前读过 {titles}，但没有讲到{missing}，所以我在 arXiv 上重新搜索了：",
    },
    "what_you_asked": {ENGLISH: "what you asked", "Chinese": "你问的内容"},
    "read_before": {ENGLISH: "read before", "Chinese": "读过"},
    "from_library": {ENGLISH: "From your library:", "Chinese": "来自你的文献库："},
    "noted": {ENGLISH: "Noted: {fact}", "Chinese": "已记下：{fact}"},
    "forgotten": {ENGLISH: "Forgotten: {fact}", "Chinese": "已忘记：{fact}"},
    "nothing_to_remember": {
        ENGLISH: "There was nothing to remember or forget in that message.",
        "Chinese": "这条消息里没有需要记住或忘记的内容。",
    },
    "note": {ENGLISH: "Note: {problems}.", "Chinese": "注：{problems}。"},
    "unaffected": {
        ENGLISH: "The rest of this reply is unaffected.",
        "Chinese": "回复的其余部分不受影响。",
    },
}


def say(key: str, language: str, **values: object) -> str:
    """One fixed sentence in `language`, or in English when it has no wording here."""
    templates = WORDING[key]
    return templates.get(language, templates[ENGLISH]).format(**values)


def language_name(stated: str) -> str:
    """The language understand named, as this table names it: any Chinese is "Chinese"."""
    name = " ".join(stated.split())
    if "chinese" in name.casefold() or "中文" in name:
        return "Chinese"
    return name or ENGLISH
