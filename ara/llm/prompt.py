"""Cache-aware prompts (DESIGN §6.3): static → shared → item, rendered byte-stably per provider."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputItemParam,
    ResponseInputTextParam,
)

type Role = Literal["developer", "user", "assistant"]
type Part = Literal["static", "shared", "item"]

PROMPTS = Path(__file__).with_name("prompts")


@dataclass(frozen=True)
class Instructions:
    """A prompt file. Its version is the file name plus a short hash of its text (D14)."""

    name: str
    text: str

    @property
    def version(self) -> str:
        return f"{self.name}@{sha256(self.text.encode()).hexdigest()[:8]}"

    @classmethod
    def load(cls, name: str, directory: Path = PROMPTS) -> "Instructions":
        return cls(name, (directory / f"{name}.md").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Block:
    """One message."""

    role: Role
    text: str


def data(label: str, value: Any) -> Block:
    """A user block holding JSON with sorted keys, so equal data always renders to equal bytes."""
    return Block("user", f"{label}:\n{json.dumps(value, sort_keys=True, ensure_ascii=False)}")


@dataclass(frozen=True)
class Prompt:
    """Static instructions, context shared by a group of calls, and the per-call part."""

    instructions: Instructions
    shared: tuple[Block, ...] = ()
    item: tuple[Block, ...] = ()

    def parts(self) -> tuple[tuple[Part, tuple[Block, ...]], ...]:
        static = (Block("developer", self.instructions.text),)
        return (("static", static), ("shared", self.shared), ("item", self.item))

    def text(self) -> str:
        return "\n".join(block.text for _, blocks in self.parts() for block in blocks)


def openai_input(prompt: Prompt, breakpoints: frozenset[Part]) -> list[ResponseInputItemParam]:
    """Responses API input. Each part named in `breakpoints` ends with an explicit cache
    breakpoint on its last non-assistant block (assistant text cannot carry one)."""
    messages: list[ResponseInputItemParam] = []
    for part, blocks in prompt.parts():
        inputs = [i for i, block in enumerate(blocks) if block.role != "assistant"]
        mark = inputs[-1] if part in breakpoints and inputs else None
        messages += [_openai_message(block, i == mark) for i, block in enumerate(blocks)]
    return messages


def _openai_message(block: Block, breakpoint: bool) -> EasyInputMessageParam:
    if block.role == "assistant":
        return {"role": "assistant", "content": block.text}
    text: ResponseInputTextParam = {"type": "input_text", "text": block.text}
    if breakpoint:
        text["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return {"role": block.role, "content": [text]}


def deepseek_messages(prompt: Prompt) -> list[ChatCompletionMessageParam]:
    """Chat Completions messages. DeepSeek caches prefixes on its own; there are no breakpoints."""
    return [_deepseek_message(block) for _, blocks in prompt.parts() for block in blocks]


def _deepseek_message(block: Block) -> ChatCompletionMessageParam:
    match block.role:
        case "developer":
            return {"role": "system", "content": block.text}
        case "user":
            return {"role": "user", "content": block.text}
        case "assistant":
            return {"role": "assistant", "content": block.text}
