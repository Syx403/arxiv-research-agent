# Read one answer back to its evidence

This is a **hand-authored fixture**, not a saved live-model run. Its purpose is to show the output contract and make the project understandable before configuring databases or API keys.

## Question

> What is ReAct?

## Illustrative answer

> ReAct alternates language-model reasoning with actions that obtain information from an external environment [arxiv:2210.03629#1].

## What a reader can inspect

| Field | Example | Meaning |
| --- | --- | --- |
| Claim | The sentence before the bracketed reference | The actual assertion to check |
| Paper | `arxiv:2210.03629` | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), Yao et al. |
| Chunk | `1` | An illustrative local ID in this fixture; real IDs come from the indexed database |
| Evidence | “ReAct combines reasoning and action steps. Reasoning updates the plan, while actions consult external information or interact with an environment.” | A short paraphrase of the paper's abstract, not an extracted quotation |
| Resolved | `true` | The paper/chunk pair exists in this fixture |
| Supports | `null` / not evaluated | No model verification has been performed |

The distinction between **resolved** and **supported** matters. Looking up a source successfully does not establish that it proves a claim. Live runs add the model's support verdict and rationale; failures remain visible.

## Run it

```bash
uv run python -m src.cli --example
uv run python -m src.cli --example --json
```

The command reads [the checked-in fixture](../src/examples/react.json), applies the runtime's citation parser and an example formatter, and resolves the reference against its supplied evidence. Retrieval, model inference, and benchmark evaluation are not executed. Live CLI output is projected by the same AgentRuntime as the browser; it only exposes delivered findings.

For an actual model-generated answer, follow [the live quickstart](../README.md#start) or [CLI instructions](../README.md#command-line-entry). Runtime traces stay in gitignored `data/`; this illustrative example does not replace live evaluation records.
