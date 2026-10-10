"""The answer subgraph (DESIGN §4.2): synthesize → prewarm → verify per claim → assemble → at most
one repair of the failed lines → verify the rewrites → finalize. Only lines whose citations passed
are delivered; a direct answer that failed no longer withholds the verified explanation (D40).

    START → synthesize → prewarm ─(Send verify per unverified claim)→ verify → assemble
    assemble → repair ─(Send verify per new claim)→ verify …  or → finalize → END
    (no evidence: START → finalize, which abstains)
"""

import operator
import re
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Send

from ara.graph.reliability import LLM_TIMEOUT_S, RETRY, SYNTHESIS_TIMEOUT_S, degrade, problem
from ara.graph.state import ABSTAIN, Answer, Claim, Context, Evidence, Verdict
from ara.graph.wording import say
from ara.llm.gateway import worth_prewarming
from ara.llm.prompt import Block, Instructions, Prompt
from ara.llm.stages import STAGES

SYNTHESIZE = Instructions.load("synthesize")
REPAIR = Instructions.load("repair")
VERIFY = Instructions.load("verify")
CITATION = re.compile(r"\[\s*(E\d+(?:\s*,\s*E\d+)*)\s*\]")
CLOSING = ".,;:!?。，；：！？、"  # noqa: RUF001  ends a cited piece, Chinese included (D40)
MAX_CLAIMS = 24  # explanation pieces verified per draft; the rest are dropped unverified (D40)
VERIFY_CONCURRENCY = 8  # verify calls in flight: each reserves its worst case under the turn cap
CONFIG: RunnableConfig = {"max_concurrency": VERIFY_CONCURRENCY}  # for whoever invokes the graph
REWRITE = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)  # "3: <line> [E2]" (D40)
DROP = "DROP"
UNCHECKED = "could not be verified"  # a verify call that failed: the line is checked again (D41)


def merge(a: dict[str, Verdict], b: dict[str, Verdict]) -> dict[str, Verdict]:
    return {**a, **b}


class AnswerInput(TypedDict):
    question: str
    evidence: list[Evidence]
    missing: list[str]
    priorities: list[str]  # what the user cares about: the explanation's focus (D24)
    context: str  # facts of this turn the reply opens with, in the model's words; "" for none (D31)
    language: str  # the user's: the reply is written in it (D40)


class AnswerOutput(TypedDict):
    answer: Answer
    problems: Annotated[list[str], operator.add]  # parts that failed after retries (M5a)


class AnswerState(AnswerInput, AnswerOutput):
    draft: str
    opening: str  # the draft's "Context:" line, delivered unverified (D31)
    claims: list[Claim]  # lines that cite known evidence: these are verified
    uncited: list[Claim]  # lines citing nothing (or unknown labels), or past MAX_CLAIMS: dropped
    # unverified, except a direct answer that cites nothing, which the repair is asked to cite
    verdicts: Annotated[dict[str, Verdict], merge]  # by Claim.key, across both rounds
    rejected: Annotated[list[Claim], operator.add]  # first-draft lines the verifier rejected
    repaired: bool
    unchecked: int  # draft lines past MAX_CLAIMS (D41)


class VerifyTask(TypedDict):
    question: str
    claim: Claim
    evidence: list[Evidence]


def opening(reply: str) -> tuple[str, str]:
    """A leading "Context: ..." line, and the reply without it. It says how the answer was found,
    not what the papers say, so it is delivered without verification (D31)."""
    first, _, rest = reply.strip().partition("\n")
    if first.startswith("Context:"):
        return first.removeprefix("Context:").strip(), rest
    return "", reply


def parse(reply: str, known: set[str]) -> tuple[list[Claim], list[Claim]]:
    """Split a reply into claims (lines citing known evidence) and uncited lines. Line 0 is the
    direct answer, written "Answer: ..."; an abstaining answer is neither."""
    found = [_claim(i, paragraph, line) for i, paragraph, line in _claims(_lines(reply))]
    return _sorted([c for c in found if c is not None], known)


def _claim(index: int, paragraph: int, line: str) -> Claim | None:
    """One line as a claim; None for a direct answer that abstains (cited or not)."""
    if index == 0:
        line = line.removeprefix("Answer:").strip()
    ids = list(dict.fromkeys(i.strip() for g in CITATION.findall(line) for i in g.split(",")))
    text = re.sub(rf"\s+([{CLOSING}])", r"\1", " ".join(CITATION.sub(" ", line).split()))
    if index == 0 and text.startswith(ABSTAIN.rstrip(".")):
        return None
    return Claim(index=index, text=text, citations=ids, paragraph=paragraph)


def _sorted(found: list[Claim], known: set[str]) -> tuple[list[Claim], list[Claim]]:
    """Claims citing only known evidence, and the rest."""
    claims = [c for c in found if c.citations and set(c.citations) <= known]
    return claims, [c for c in found if c not in claims]


def amend(
    lines: list[Claim], failed: set[int], reply: str, known: set[str]
) -> tuple[list[Claim], list[Claim]]:
    """The draft's lines with the failed ones (by index) replaced by the repair's rewrites
    ("3: <line> [E2]") in their place and paragraph, or removed ("3: DROP", or not mentioned).
    A rewrite is cut at its citations like any explanation line; the direct answer's rewrite stays
    the direct answer, and one that abstains removes it (D40)."""
    rewrites = {int(m[1]): m[2] for m in REWRITE.finditer(reply)}
    amended: list[Claim] = []
    for line in lines:
        if line.index not in failed:
            amended.append(line)
            continue
        rewrite = rewrites.get(line.index, DROP)
        if rewrite.strip(" .") == DROP:
            continue
        pieces = [rewrite] if line.direct else _cut(rewrite)
        index = 0 if line.direct else 1
        amended += [c for p in pieces if (c := _claim(index, line.paragraph, p)) is not None]
    numbered = [
        c.model_copy(update={"index": 0 if c.direct else n}) for n, c in enumerate(amended, 1)
    ]
    return _sorted(numbered, known)


def _lines(reply: str) -> list[tuple[int, int, str]]:
    """Each non-empty line with its index and its paragraph: an empty line between explanation
    lines starts a new paragraph (D37)."""
    found: list[tuple[int, int, str]] = []
    paragraph, gap = 0, False
    for raw in reply.splitlines():
        line = raw.strip()
        if not line:
            gap = True
            continue
        if gap and len(found) > 1:
            paragraph += 1
        found.append((len(found), paragraph, line))
        gap = False
    return found


def _claims(lines: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Every claim: the direct answer's line whole; an explanation line cut after each of its
    citations, so a model that writes a paragraph on one line still has each cited sentence
    verified on its own (D19; seen in D39's live check). Text after the last citation is a claim
    without one. Claims are numbered in order; 0 stays the direct answer."""
    found: list[tuple[int, int, str]] = []
    for index, paragraph, line in lines:
        for piece in [line] if index == 0 else _cut(line):
            found.append((len(found), paragraph, piece))
    return found


def _cut(line: str) -> list[str]:
    """The line cut after each citation (and the punctuation that follows it)."""
    pieces, start = [], 0
    for match in CITATION.finditer(line):
        end = match.end()
        while end < len(line) and line[end] in CLOSING:
            end += 1
        pieces.append(line[start:end])
        start = end
    pieces.append(line[start:])
    return [p.strip() for p in pieces if p.strip()]


def capped(claims: list[Claim], uncited: list[Claim]) -> tuple[list[Claim], list[Claim]]:
    """At most MAX_CLAIMS explanation pieces are verified; the rest join the lines dropped
    unverified, so a long draft cannot fan out past the turn's cap (D40)."""
    kept = [c for c in claims if c.direct] + [c for c in claims if not c.direct][:MAX_CLAIMS]
    return kept, [*uncited, *(c for c in claims if c not in kept)]


def pack(evidence: list[Evidence]) -> Block:
    """The evidence pack: the shared prefix of synthesize, repair and every verify call."""
    lines = [f"{e.id} ({e.heading_path}): {e.text}" for e in evidence]
    return Block("user", "Evidence:\n" + "\n".join(lines))


def synthesis_prompt(state: AnswerInput) -> Prompt:
    question = f"Question: {state['question']}"
    if state["priorities"]:
        question += "\nThe user cares about: " + "; ".join(state["priorities"])
    if state["missing"]:
        question += "\nThe evidence search found nothing on: " + "; ".join(state["missing"])
    if state["context"]:
        question += "\nHow this evidence was found: " + state["context"]
    question += f"\nLanguage: {state['language']}"
    return Prompt(SYNTHESIZE, shared=(pack(state["evidence"]),), item=(Block("user", question),))


def verify_prompt(evidence: list[Evidence], claim: Claim, question: str) -> Prompt:
    """The direct answer is a short phrase ("no", "BLEU"), so it is checked as the answer to the
    question; explanation lines are checked on their own (D21)."""
    item = f"Claim: {claim.text}\nCites: {', '.join(claim.citations)}"
    if claim.direct:
        item = f"Question: {question}\nDirect answer to the question.\n{item}"
    return Prompt(VERIFY, shared=(pack(evidence),), item=(Block("user", item),))


async def synthesize(state: AnswerState, runtime: Runtime[Context]) -> dict[str, object]:
    ctx = runtime.context
    draft = await ctx.gateway.text(STAGES["synthesize"], synthesis_prompt(state), scope=ctx.scope)
    context, body = opening(draft) if state["context"] else ("", draft)
    found, uncited = parse(body, {e.id for e in state["evidence"]})
    claims, uncited = capped(found, uncited)
    return {
        "unchecked": len(found) - len(claims),
        "draft": draft,
        "opening": context,
        "claims": claims,
        "uncited": uncited,
        "repaired": False,
    }


async def prewarm(state: AnswerState, runtime: Runtime[Context]) -> dict[str, object]:
    """Write the evidence pack to the cache once, before the claims are verified in parallel."""
    ctx = runtime.context
    pending = _unverified(state)
    if not pending:
        return {}
    # prewarm sends only the static and shared parts, so any pending claim gives the same prefix
    prompt = verify_prompt(state["evidence"], pending[0], state["question"])
    if worth_prewarming(prompt, Verdict, len(pending)):
        await ctx.gateway.prewarm(STAGES["verify"], prompt, Verdict, scope=ctx.scope)
    return {}


def _unverified_line(state: VerifyTask, error: BaseException) -> dict[str, Any]:
    """A line the verifier could not check is not delivered (M5a)."""
    verdict = Verdict(supported=False, problem=UNCHECKED)
    return {"verdicts": {state["claim"].key: verdict}, "problems": [problem("verification", error)]}


@degrade(_unverified_line)
async def verify(state: VerifyTask, runtime: Runtime[Context]) -> dict[str, Any]:
    ctx = runtime.context
    prompt = verify_prompt(state["evidence"], state["claim"], state["question"])
    verdict = await ctx.gateway.structured(STAGES["verify"], prompt, Verdict, scope=ctx.scope)
    return {"verdicts": {state["claim"].key: verdict}}


def assemble(state: AnswerState) -> dict[str, object]:
    """Joins the verify branches."""
    return {}


async def repair(state: AnswerState, runtime: Runtime[Context]) -> dict[str, object]:
    """One rewrite of the failed lines only, each by its number (D40); the lines that passed stay
    as they are. It continues the synthesize conversation (append-only), so DeepSeek serves the
    evidence pack and question from its cache."""
    ctx = runtime.context
    failed = _failing(state)
    if not failed:  # only lines that could not be checked: they are verified again, not rewritten
        return {"repaired": True}
    first = synthesis_prompt(state)
    failures = "\n".join(
        f"{c.index}: {c.text} [{', '.join(c.citations)}] — {why}" for c, why in failed
    )
    prompt = Prompt(
        SYNTHESIZE,
        shared=(*first.shared, *first.item, Block("assistant", state["draft"])),
        item=(Block("user", f"{REPAIR.text}\n{failures}"),),
        follow_up=REPAIR,
    )
    reply = await ctx.gateway.text(STAGES["repair"], prompt, scope=ctx.scope)
    lines = [*state["claims"], *(c for c in state["uncited"] if c.direct)]
    known = {e.id for e in state["evidence"]}
    claims, uncited = amend(lines, {c.index for c, _ in failed}, reply, known)
    return {
        "claims": claims,
        "uncited": [*(c for c in state["uncited"] if not c.direct), *uncited],
        "rejected": [c for c, _ in failed if c in state["claims"]],
        "repaired": True,
    }


def _failing(state: AnswerState) -> list[tuple[Claim, str]]:
    """The lines a repair rewrites, with why: those the verifier rejected, and a direct answer
    that cites no known evidence (an uncited explanation line is only dropped, D19). A line whose
    verify call failed was not judged, so it is not rewritten (D41)."""
    verdicts = state.get("verdicts", {})
    rejected = [
        (c, verdicts[c.key].problem)
        for c in state["claims"]
        if not verdicts[c.key].supported and verdicts[c.key].problem != UNCHECKED
    ]
    return rejected + [(c, "it cites no evidence") for c in state["uncited"] if c.direct]


def _unchecked(state: AnswerState) -> list[Claim]:
    verdicts = state.get("verdicts", {})
    return [
        c for c in state["claims"] if c.key in verdicts and verdicts[c.key].problem == UNCHECKED
    ]


def finalize(state: AnswerState) -> dict[str, Answer]:
    """Deliver the verified lines. A direct answer that failed (unsupported or uncited) is not
    delivered, and the reply says so before the verified explanation; with no verified line, it
    says no answer could be verified (D40, was: the whole answer withheld, D21). A model that
    abstained keeps its verified lines on what the evidence does cover."""
    verdicts = state.get("verdicts", {})
    claims = state.get("claims", [])
    uncited = state.get("uncited", [])
    supported = [c for c in claims if verdicts[c.key].supported]
    direct = next((c for c in supported if c.direct), None)
    withheld = direct is None and any(c.direct for c in claims + uncited)
    sentences = [c for c in supported if not c.direct]
    delivered = ([direct] if direct else []) + sentences
    rejected = state.get("rejected", []) + [c for c in claims if c not in supported]
    cited = {i for c in delivered for i in c.citations}
    instead = ("withheld" if sentences else "unverified") if withheld else "abstain"
    answer = Answer(
        question=state["question"],
        short=direct.text if direct else say(instead, state["language"]),
        abstained=direct is None,
        withheld=withheld,
        unchecked=state.get("unchecked", 0),
        sentences=sentences,
        dropped=[c for c in claims if c not in delivered] + uncited,
        checked=len(verdicts),
        rejected=list({c.key: c for c in rejected}.values()),
        evidence=[e for e in state["evidence"] if e.id in cited],
        context=state.get("opening", ""),
    )
    return {"answer": answer}


def synthesis_failed(state: AnswerState, error: NodeError) -> Command[str]:
    """No draft: the answer abstains, and the reply says why (M5a)."""
    update = {"claims": [], "uncited": [], "problems": [problem("writing the answer", error.error)]}
    return Command(update=update, goto="finalize")


def repair_failed(state: AnswerState, error: NodeError) -> Command[str]:
    """The first draft's verified lines stand; its rejected lines stay undelivered (M5a)."""
    return Command(update={"problems": [problem("repair", error.error)]}, goto="finalize")


def prewarm_failed(state: AnswerState, error: NodeError) -> Command[Any]:
    """The cache write is an optimisation: verification goes on without it."""
    return Command(update={"problems": []}, goto=to_verify(state))


def _unverified(state: AnswerState) -> list[Claim]:
    """Lines with no verdict yet; after the repair, also those whose verify call failed (once)."""
    verdicts = state.get("verdicts", {})
    again = _unchecked(state) if state.get("repaired") else []
    return [c for c in state["claims"] if c.key not in verdicts] + again


def start(state: AnswerState) -> str:
    return "synthesize" if state["evidence"] else "finalize"


def to_verify(state: AnswerState) -> list[Send] | str:
    pending = _unverified(state)
    if not pending:
        return "assemble"
    return [
        Send("verify", VerifyTask(question=state["question"], claim=c, evidence=state["evidence"]))
        for c in pending
    ]


def after_assemble(state: AnswerState) -> str:
    """One repair round: rewrites of the rejected lines, and a second check of the unchecked."""
    retry = _failing(state) or _unchecked(state)
    return "repair" if retry and not state["repaired"] else "finalize"


def build() -> CompiledStateGraph[AnswerState, Context, AnswerInput, AnswerOutput]:
    graph = StateGraph(
        AnswerState, context_schema=Context, input_schema=AnswerInput, output_schema=AnswerOutput
    )
    graph.add_node(
        "synthesize",
        synthesize,
        retry_policy=RETRY,
        timeout=SYNTHESIS_TIMEOUT_S,
        error_handler=synthesis_failed,
    )
    graph.add_node(
        "prewarm", prewarm, retry_policy=RETRY, timeout=LLM_TIMEOUT_S, error_handler=prewarm_failed
    )
    graph.add_node(
        "verify",
        verify,
        input_schema=VerifyTask,
        retry_policy=RETRY,
        timeout=LLM_TIMEOUT_S,
    )
    graph.add_node("assemble", assemble)
    graph.add_node(
        "repair",
        repair,
        retry_policy=RETRY,
        timeout=SYNTHESIS_TIMEOUT_S,
        error_handler=repair_failed,
    )
    graph.add_node("finalize", finalize)
    graph.add_conditional_edges(START, start, ["synthesize", "finalize"])
    graph.add_edge("synthesize", "prewarm")
    graph.add_conditional_edges("prewarm", to_verify, ["verify", "assemble"])
    graph.add_edge("verify", "assemble")
    graph.add_conditional_edges("assemble", after_assemble, ["repair", "finalize"])
    graph.add_conditional_edges("repair", to_verify, ["verify", "assemble"])
    graph.add_edge("finalize", END)
    return graph.compile(name="answer")
