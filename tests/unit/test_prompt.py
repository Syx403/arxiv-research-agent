import json
from typing import Any

from ara.llm.prompt import (
    Block,
    Instructions,
    Part,
    Prompt,
    anthropic_request,
    data,
    deepseek_messages,
    openai_input,
)
from ara.llm.stages import CONVERSATION, FANOUT

INSTRUCTIONS = Instructions("verify", "Check each claim against the evidence.")
EVIDENCE = (data("evidence", {"E2": "second", "E1": "first"}),)
TRANSCRIPT = (Block("user", "Q1"), Block("assistant", "A1"))


def claim(text: str) -> Prompt:
    return Prompt(INSTRUCTIONS, shared=EVIDENCE, item=(Block("user", text),))


def render(prompt: Prompt, breakpoints: frozenset[Part]) -> list[Any]:
    """The request body as the provider receives it."""
    rendered: list[Any] = json.loads(json.dumps(openai_input(prompt, breakpoints)))
    return rendered


def breakpoint_flags(messages: list[Any]) -> list[bool]:
    return [
        isinstance(m["content"], list) and "prompt_cache_breakpoint" in m["content"][-1]
        for m in messages
    ]


def test_rendering_is_byte_stable() -> None:
    assert json.dumps(render(claim("A"), FANOUT)) == json.dumps(render(claim("A"), FANOUT))


def test_fan_out_calls_share_the_static_and_shared_prefix() -> None:
    a, b = render(claim("A"), FANOUT), render(claim("B"), FANOUT)
    assert a[:2] == b[:2]
    assert a[2] != b[2]


def test_fan_out_breakpoints_end_static_and_shared_only() -> None:
    assert breakpoint_flags(render(claim("A"), FANOUT)) == [True, True, False]


def test_a_breakpoint_skips_assistant_text() -> None:
    prompt = Prompt(INSTRUCTIONS, shared=TRANSCRIPT, item=(Block("user", "Q2"),))
    # developer, Q1 (last input of shared), A1 (assistant: cannot carry one), Q2 (item)
    assert breakpoint_flags(render(prompt, CONVERSATION)) == [True, True, False, True]


def test_json_blocks_sort_keys() -> None:
    assert EVIDENCE[0].text == 'evidence:\n{"E1": "first", "E2": "second"}'


def test_the_version_follows_the_text() -> None:
    edited = Instructions("verify", INSTRUCTIONS.text + " Be strict.")
    assert INSTRUCTIONS.version.startswith("verify@")
    assert edited.version != INSTRUCTIONS.version


def test_deepseek_messages_map_developer_to_system() -> None:
    prompt = Prompt(INSTRUCTIONS, shared=TRANSCRIPT, item=(Block("user", "Q2"),))
    assert [m["role"] for m in deepseek_messages(prompt)] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_a_follow_up_file_is_part_of_the_version() -> None:
    repair = Instructions("repair", "Fix the listed lines.")
    prompt = Prompt(INSTRUCTIONS, item=(Block("user", repair.text),), follow_up=repair)
    assert prompt.version == f"{INSTRUCTIONS.version}+{repair.version}"
    assert claim("A").version == INSTRUCTIONS.version


def test_anthropic_requests_put_instructions_in_system_and_cache_each_breakpoint_part() -> None:
    """D44: the system text and the last block of each breakpoint part carry cache_control;
    consecutive blocks of one role become one turn, because the API alternates roles."""
    prompt = Prompt(INSTRUCTIONS, shared=(*EVIDENCE, Block("user", "more")), item=TRANSCRIPT)
    system, messages = anthropic_request(prompt, FANOUT)
    assert system == [
        {"type": "text", "text": INSTRUCTIONS.text, "cache_control": {"type": "ephemeral"}}
    ]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    shared, q1 = messages[0]["content"][:2], messages[0]["content"][2]
    assert ["cache_control" in c for c in shared] == [False, True] and "cache_control" not in q1
    assert "cache_control" not in messages[1]["content"][-1]
    _, conversation = anthropic_request(prompt, CONVERSATION)
    assert "cache_control" in conversation[-1]["content"][-1], "the item part ends with one too"
