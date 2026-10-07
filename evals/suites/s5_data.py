"""S5 verifier data (DESIGN §11.2, D19): 30 supported and 30 unsupported claims, each checked
against one QASPER evidence sentence. The supported claim is a DeepSeek paraphrase of the
sentence; the unsupported one changes one thing: a number or a negation (by code), an entity or
the scope (by DeepSeek). Ewan reviews every item before S5 uses it (`reviewed`)."""

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.llm.prompt import Instructions, Prompt, data
from ara.llm.stages import FLASH, Stage
from ara.rag.chunking import sentence_starts
from ara.rag.sources import qasper_paper
from evals import qasper
from evals.suites import s1

OUTPUT = Path(__file__).parents[1] / "datasets" / "s5_claims.json"
PROGRESS = Path(__file__).parents[2] / "data" / "s5_rewrites.json"
PERTURB = Instructions.load("s5_perturb", Path(__file__).parents[1] / "prompts")
STAGE = Stage("s5_perturb", FLASH, "high", 6_000)
QUOTA = {"number": 8, "negation": 7, "entity": 8, "overgeneralisation": 7}
BATCH = 5
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
AUXILIARY = re.compile(
    r"\b(is|are|was|were|can|could|does|do|did|has|have|had|will|would|should|may|might)\b"
)


class Rewrite(BaseModel):
    id: str
    paraphrase: str
    perturbed: str


class Rewrites(BaseModel):
    items: list[Rewrite]


def sources() -> list[dict[str, Any]]:
    """One evidence sentence per S1 item (the first annotator's first gold paragraph), preferring a
    sentence with a number, then the longest; and the perturbation it will get."""
    items, _ = s1.load()
    records = qasper.records()
    counts = dict.fromkeys(QUOTA, 0)
    chosen = []
    for item in items:
        paper = qasper_paper(records[item.paper])
        paragraph = min(item.references[0])
        text = paper.paragraphs[paragraph].text
        starts = sentence_starts(text)
        sentences = [
            text[a:b].strip() for a, b in zip(starts, (*starts[1:], len(text)), strict=True)
        ]
        sentence = max(sentences, key=lambda s: (bool(NUMBER.search(s)), len(s.split())))
        kind = _kind(sentence, counts)
        counts[kind] += 1
        chosen.append(
            {
                "item": item.id,
                "paper": item.paper,
                "paragraph": paragraph,
                "sentence": sentence,
                "kind": kind,
            }
        )
    return chosen


def _kind(sentence: str, counts: dict[str, int]) -> str:
    """Code perturbations where the sentence allows them, the LLM kinds otherwise."""
    if NUMBER.search(sentence) and counts["number"] < QUOTA["number"]:
        return "number"
    if AUXILIARY.search(sentence) and counts["negation"] < QUOTA["negation"]:
        return "negation"
    return "entity" if counts["entity"] < QUOTA["entity"] else "overgeneralisation"


def change_number(claim: str) -> str | None:
    """Alter the first number: small integers double, larger ones grow by about a third."""
    match = NUMBER.search(claim)
    if match is None:
        return None
    value = match[0]
    if "." in value:
        new = f"{float(value) + 2.5:.{len(value.split('.')[1])}f}"
    else:
        n = int(value)
        new = str(n * 2 if n < 10 else round(n * 1.3))
    return claim[: match.start()] + new + claim[match.end() :]


def negate(claim: str) -> str | None:
    match = AUXILIARY.search(claim)
    if match is None:
        return None
    return claim[: match.end()] + " not" + claim[match.end() :]


async def generate(gateway: Gateway, scope: Scope) -> list[dict[str, Any]]:
    """Batches already in PROGRESS are skipped, and each new batch is saved as it arrives, so a
    stopped run resumes without paying again."""
    picks = sources()
    saved = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    rewrites = {key: Rewrite.model_validate(value) for key, value in saved.items()}
    for start in range(0, len(picks), BATCH):
        batch = picks[start : start + BATCH]
        if all(p["item"] in rewrites for p in batch):
            continue
        tasks = [
            {
                "id": p["item"],
                "sentence": p["sentence"],
                "task": p["kind"] if p["kind"] in ("entity", "overgeneralisation") else "none",
            }
            for p in batch
        ]
        result = await gateway.structured(
            STAGE, Prompt(PERTURB, item=(data("Sentences", tasks),)), Rewrites, scope=scope
        )
        rewrites |= {r.id: r for r in result.items}
        PROGRESS.write_text(
            json.dumps(
                {k: r.model_dump() for k, r in rewrites.items()}, indent=2, ensure_ascii=False
            )
        )
    claims = []
    for p in picks:
        rewrite = rewrites[p["item"]]
        evidence = {"paper": p["paper"], "paragraph": p["paragraph"], "sentence": p["sentence"]}
        claims.append(
            {
                "id": f"{p['item']}:supported",
                "label": "supported",
                "kind": "paraphrase",
                "made_by": "deepseek",
                "claim": rewrite.paraphrase,
                **evidence,
            }
        )
        if p["kind"] == "number":
            perturbed, made_by = (
                change_number(rewrite.paraphrase) or change_number(p["sentence"]),
                "code",
            )
        elif p["kind"] == "negation":
            perturbed, made_by = negate(rewrite.paraphrase) or negate(p["sentence"]), "code"
        else:
            perturbed, made_by = rewrite.perturbed, "deepseek"
        claims.append(
            {
                "id": f"{p['item']}:unsupported",
                "label": "unsupported",
                "kind": p["kind"],
                "made_by": made_by,
                "claim": perturbed,
                **evidence,
            }
        )
    return [{**c, "reviewed": False} for c in claims]


def write(claims: list[dict[str, Any]]) -> None:
    OUTPUT.write_text(
        json.dumps(
            {"source": "QASPER validation via the S1 manifest", "items": claims},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
