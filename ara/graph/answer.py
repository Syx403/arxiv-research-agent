"""The answer subgraph (DESIGN §4.2): synthesize → prewarm → verify per claim → assemble → at most
one repair → verify the changed lines → finalize. Only lines whose citations passed are delivered.

    START → synthesize → prewarm ─(Send verify per unverified claim)→ verify → assemble
    assemble → repair ─(Send verify per new claim)→ verify …  or → finalize → END
    (no evidence: START → finalize, which abstains)
"""

import operator
import re
from typing import Annotated, Any, TypedDict

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Send

from ara.graph.reliability import LLM_TIMEOUT_S, RETRY, SYNTHESIS_TIMEOUT_S, degrade, problem
from ara.graph.state import ABSTAIN, Answer, Claim, Context, Evidence, Verdict
from ara.llm.gateway import worth_prewarming
from ara.llm.prompt import Block, Instructions, Prompt
from ara.llm.stages import STAGES

SYNTHESIZE = Instructions.load("synthesize")
REPAIR = Instructions.load("repair")
VERIFY = Instructions.load("verify")
CITATION = re.compile(r"\[\s*(E\d+(?:\s*,\s*E\d+)*)\s*\]")


def merge(a: dict[str, Verdict], b: dict[str, Verdict]) -> dict[str, Verdict]:
    return {**a, **b}


class AnswerInput(TypedDict):
    question: str
    evidence: list[Evidence]
    missing: list[str]
    priorities: list[str]  # what the user cares about: the explanation's focus (D24)
    context: str  # facts of this turn the reply opens with, in the model's words; "" for none (D31)


class AnswerOutput(TypedDict):
    answer: Answer
    problems: Annotated[list[str], operator.add]  # parts that failed after retries (M5a)


class AnswerState(AnswerInput, AnswerOutput):
    draft: str
    opening: str  # the draft's "Context:" line, delivered unverified (D31)
    claims: list[Claim]  # lines that cite known evidence: these are verified
    uncited: list[Claim]  # lines citing nothing (or unknown labels): dropped unverified
    verdicts: Annotated[dict[str, Verdict], merge]  # by Claim.key, across both rounds
    rejected: Annotated[list[Claim], operator.add]  # first-draft lines the verifier rejected
    repaired: bool


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
    claims: list[Claim] = []
    uncited: list[Claim] = []
    for index, paragraph, line in _lines(reply):
        if index == 0:
            line = line.removeprefix("Answer:").strip()
        ids = list(dict.fromkeys(i.strip() for g in CITATION.findall(line) for i in g.split(",")))
        text = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(CITATION.sub(" ", line).split()))
        claim = Claim(index=index, text=text, citations=ids, paragraph=paragraph)
        if index == 0 and text.startswith(ABSTAIN.rstrip(".")):
            continue  # the model abstained, with or without a citation
        (claims if ids and set(ids) <= known else uncited).append(claim)
    return claims, uncited


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
    claims, uncited = parse(body, {e.id for e in state["evidence"]})
    return {
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
    verdict = Verdict(supported=False, problem="could not be verified")
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
    """One rewrite of the failed lines. It continues the synthesize conversation (append-only), so
    DeepSeek serves the evidence pack and question from its cache."""
    ctx = runtime.context
    first = synthesis_prompt(state)
    failed = [c for c in state["claims"] if not state["verdicts"][c.key].supported]
    failures = "\n".join(
        f"- {c.text} [{', '.join(c.citations)}]: {state['verdicts'][c.key].problem}" for c in failed
    )
    prompt = Prompt(
        SYNTHESIZE,
        shared=(*first.shared, *first.item, Block("assistant", state["draft"])),
        item=(Block("user", f"{REPAIR.text}\n{failures}"),),
        follow_up=REPAIR,
    )
    draft = await ctx.gateway.text(STAGES["repair"], prompt, scope=ctx.scope)
    claims, uncited = parse(opening(draft)[1], {e.id for e in state["evidence"]})
    return {
        "draft": draft,
        "claims": claims,
        "uncited": uncited,
        "rejected": failed,
        "repaired": True,
    }


def finalize(state: AnswerState) -> dict[str, Answer]:
    """Deliver the verified lines. A direct answer that failed (unsupported or uncited) withholds
    the whole answer (D19, D21); a model that abstained keeps its verified note on what the
    evidence does cover."""
    verdicts = state.get("verdicts", {})
    claims = state.get("claims", [])
    uncited = state.get("uncited", [])
    supported = [c for c in claims if verdicts[c.key].supported]
    direct = next((c for c in supported if c.direct), None)
    failed = direct is None and any(c.direct for c in claims + uncited)
    sentences = [] if failed else [c for c in supported if not c.direct]
    delivered = ([direct] if direct else []) + sentences
    rejected = state.get("rejected", []) + [c for c in claims if c not in supported]
    cited = {i for c in delivered for i in c.citations}
    answer = Answer(
        question=state["question"],
        short=direct.text if direct else ABSTAIN,
        abstained=direct is None,
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
    verdicts = state.get("verdicts", {})
    return [c for c in state["claims"] if c.key not in verdicts]


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
    failed = any(not state["verdicts"][c.key].supported for c in state["claims"])
    return "repair" if failed and not state["repaired"] else "finalize"


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
