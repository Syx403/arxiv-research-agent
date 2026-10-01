from src.core.types import SELF_RAG_MAX_RETRIES
from src.graph.state import AgentState


def route_after_verify(state: AgentState) -> str:
    if state.get("forced_stop_reason") or state.get("synthesis_format_degraded") or state.get("verification_stalled"):
        return "finalize"
    verification = state.get("verification")
    if verification and any(v.status == "error" for v in verification.verdicts):
        return "finalize"  # A failed assessment is not a reason to rewrite facts.
    if (
        verification is not None
        and not verification.passed
        and state.get("tool_iters", 0) < SELF_RAG_MAX_RETRIES
    ):
        return "synthesize"
    return "finalize"
