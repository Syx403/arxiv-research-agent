"""The answer subgraph (DESIGN §4.2): synthesize → prewarm → verify per claim → assemble → at most
one repair → verify the changed lines → finalize. Only lines whose citations passed are delivered.

    START → synthesize → prewarm ─(Send verify per unverified claim)→ verify → assemble
    assemble → repair ─(Send verify per new claim)→ verify …  or → finalize → END
    (no evidence: START → finalize, which abstains)
"""

import re
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

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


class AnswerOutput(TypedDict):
    answer: Answer


class AnswerState(AnswerInput, AnswerOutput):
    draft: str
    claims: list[Claim]  # lines that cite known evidence: these are verified
    uncited: list[Claim]  # lines citing nothing (or unknown labels): dropped unverified
    verdicts: Annotated[dict[str, Verdict], merge]  # by Claim.key, across both rounds
    repaired: bool


class VerifyTask(TypedDict):
    claim: Claim
    evidence: list[Evidence]


def parse(reply: str, known: set[str]) -> tuple[list[Claim], list[Claim]]:
    """Split a reply into claims (lines citing known evidence) and uncited lines. Line 0 is the
    direct answer, written "Answer: ..."; an abstaining answer is neither."""
    claims: list[Claim] = []
    uncited: list[Claim] = []
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if index == 0:
            line = line.removeprefix("Answer:").strip()
        ids = list(dict.fromkeys(i.strip() for g in CITATION.findall(line) for i in g.split(",")))
        text = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(CITATION.sub(" ", line).split()))
        claim = Claim(index=index, text=text, citations=ids)
        if ids and set(ids) <= known:
            claims.append(claim)
        elif not (index == 0 and text.startswith(ABSTAIN.rstrip("."))):
            uncited.append(claim)
    return claims, uncited


def pack(evidence: list[Evidence]) -> Block:
    """The evidence pack: the shared prefix of synthesize, repair and every verify call."""
    lines = [f"{e.id} ({e.heading_path}): {e.text}" for e in evidence]
    return Block("user", "Evidence:\n" + "\n".join(lines))


def synthesis_prompt(state: AnswerInput) -> Prompt:
    question = f"Question: {state['question']}"
    if state["missing"]:
        question += "\nThe evidence search found nothing on: " + "; ".join(state["missing"])
    return Prompt(SYNTHESIZE, shared=(pack(state["evidence"]),), item=(Block("user", question),))


def verify_prompt(evidence: list[Evidence], claim: Claim) -> Prompt:
    item = f"Claim: {claim.text}\nCites: {', '.join(claim.citations)}"
    return Prompt(VERIFY, shared=(pack(evidence),), item=(Block("user", item),))


async def synthesize(state: AnswerState, runtime: Runtime[Context]) -> dict[str, object]:
    ctx = runtime.context
    draft = await ctx.gateway.text(STAGES["synthesize"], synthesis_prompt(state), scope=ctx.scope)
    claims, uncited = parse(draft, {e.id for e in state["evidence"]})
    return {"draft": draft, "claims": claims, "uncited": uncited, "repaired": False}


async def prewarm(state: AnswerState, runtime: Runtime[Context]) -> dict[str, object]:
    """Write the evidence pack to the cache once, before the claims are verified in parallel."""
    ctx = runtime.context
    pending = _unverified(state)
    if not pending:
        return {}
    prompt = verify_prompt(state["evidence"], pending[0])  # prewarm drops the item part
    if worth_prewarming(prompt, Verdict, len(pending)):
        await ctx.gateway.prewarm(STAGES["verify"], prompt, Verdict, scope=ctx.scope)
    return {}


async def verify(state: VerifyTask, runtime: Runtime[Context]) -> dict[str, dict[str, Verdict]]:
    ctx = runtime.context
    prompt = verify_prompt(state["evidence"], state["claim"])
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
    failures = "\n".join(
        f"- {c.text} [{', '.join(c.citations)}]: {state['verdicts'][c.key].problem}"
        for c in state["claims"]
        if not state["verdicts"][c.key].supported
    )
    prompt = Prompt(
        SYNTHESIZE,
        shared=(*first.shared, *first.item, Block("assistant", state["draft"])),
        item=(Block("user", f"{REPAIR.text}\n{failures}"),),
    )
    draft = await ctx.gateway.text(STAGES["repair"], prompt, scope=ctx.scope)
    claims, uncited = parse(draft, {e.id for e in state["evidence"]})
    return {"draft": draft, "claims": claims, "uncited": uncited, "repaired": True}


def finalize(state: AnswerState) -> dict[str, Answer]:
    verdicts = state.get("verdicts", {})
    claims = state.get("claims", [])
    supported = [c for c in claims if verdicts[c.key].supported]
    direct = next((c for c in supported if c.index == 0), None)
    sentences = [c for c in supported if c.index > 0]
    cited = {i for c in supported for i in c.citations}
    answer = Answer(
        question=state["question"],
        short=direct.text if direct else ABSTAIN,
        abstained=direct is None,
        sentences=sentences,
        dropped=[c for c in claims if c not in supported] + state.get("uncited", []),
        evidence=[e for e in state["evidence"] if e.id in cited],
    )
    return {"answer": answer}


def _unverified(state: AnswerState) -> list[Claim]:
    verdicts = state.get("verdicts", {})
    return [c for c in state["claims"] if c.key not in verdicts]


def start(state: AnswerState) -> str:
    return "synthesize" if state["evidence"] else "finalize"


def to_verify(state: AnswerState) -> list[Send] | str:
    pending = _unverified(state)
    if not pending:
        return "assemble"
    return [Send("verify", VerifyTask(claim=c, evidence=state["evidence"])) for c in pending]


def after_assemble(state: AnswerState) -> str:
    failed = any(not state["verdicts"][c.key].supported for c in state["claims"])
    return "repair" if failed and not state["repaired"] else "finalize"


def build() -> CompiledStateGraph[AnswerState, Context, AnswerInput, AnswerOutput]:
    graph = StateGraph(
        AnswerState, context_schema=Context, input_schema=AnswerInput, output_schema=AnswerOutput
    )
    graph.add_node("synthesize", synthesize)
    graph.add_node("prewarm", prewarm)
    graph.add_node("verify", verify, input_schema=VerifyTask)
    graph.add_node("assemble", assemble)
    graph.add_node("repair", repair)
    graph.add_node("finalize", finalize)
    graph.add_conditional_edges(START, start, ["synthesize", "finalize"])
    graph.add_edge("synthesize", "prewarm")
    graph.add_conditional_edges("prewarm", to_verify, ["verify", "assemble"])
    graph.add_edge("verify", "assemble")
    graph.add_conditional_edges("assemble", after_assemble, ["repair", "finalize"])
    graph.add_conditional_edges("repair", to_verify, ["verify", "assemble"])
    graph.add_edge("finalize", END)
    return graph.compile(name="answer")
