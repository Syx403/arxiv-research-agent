import json

from src.core.run_context import current_run


def conversation_turns(messages, *, max_messages=6, max_chars=12000):
    """Keep recent advice as well as introductions, with a bounded history budget."""
    selected, remaining = [], max_chars
    for message in reversed([m for m in messages if m.type in {"human", "ai"}][-max_messages:]):
        text = str(message.content)
        cap = min(4000, remaining - 40)
        if cap < 100:
            break
        if len(text) > cap:
            half = (cap - 40) // 2
            text = text[:half] + "\n[Earlier middle section omitted]\n" + text[-half:]
        item = f"{message.type}: {text}"
        selected.append(item)
        remaining -= len(item)
    return list(reversed(selected))


def planning_context(state):
    """Stable structured state plus bounded recent messages, not generated memory."""
    if not current_run().structured_context:
        return conversation_turns(state.get("messages", [])[:-1])
    memory = state.get("session_context") or {}
    recent = conversation_turns(state.get("messages", [])[:-1], max_messages=4, max_chars=7000)
    if not memory:
        return recent
    # Only literal, previously validated human quotes may become hard requirements.
    quotes = ["human: " + item["quote"] for item in memory.get("requirements", [])]
    return ["session_state (context, not source evidence): " + json.dumps(memory, ensure_ascii=False, sort_keys=True),
            *quotes, *recent]


def planned_session(state, plan):
    previous = state.get("session_context") or {}
    question = state["question"]
    requirements = []
    for kind, quotes in (("hard", plan.hard_constraints), ("preference", plan.soft_preferences)):
        for quote in quotes:
            old = next((r for r in previous.get("requirements", []) if r["quote"] == quote), None)
            if quote in question:
                origin = current_run().turn_id or state.get("turn_id", "")
            elif old:
                origin = old["source_turn"]
            else:
                message = next((m for m in reversed(state.get("messages", []))
                                if m.type == "human" and quote in str(m.content)), None)
                if message is None:
                    continue
                origin = message.id or "prior_human_message"
            requirements.append({"quote": quote, "kind": kind, "source_turn": origin})
    return {
        "version": 1, "goal": plan.question, "requirements": requirements,
        "selected_papers": plan.paper_ids, "pending_clarification": plan.clarification,
        "displayed_papers": previous.get("displayed_papers", []),
        "updated_turn": current_run().turn_id or state.get("turn_id", ""),
    }


def delivered_session(state, papers=None):
    memory = dict(state.get("session_context") or {})
    if papers is not None:
        memory["displayed_papers"] = [
            {"position": i, "paper_id": p["paper_id"], "version_id": p.get("version_id"),
             "title": p["title"][:300]}
            for i, p in enumerate(papers[:5], 1)
        ]
    return memory
