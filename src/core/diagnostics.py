from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any


_LOCK = Lock()
_DIAGNOSTIC_PATH = Path("data/eval_outputs/latest/diagnostic.jsonl")


def log_raw_response(role: str, raw_content: str | None, **fields: Any) -> None:
    if not os.getenv("DIAGNOSTIC_LOG"):
        return
    content = raw_content or ""
    row = {
        "time": datetime.now(UTC).isoformat(),
        "role": role,
        "question_id": os.getenv("DIAGNOSTIC_QUESTION_ID", ""),
        "raw_content_preview": content[:400],
        "content_length": len(content),
        **fields,
    }
    _DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with _DIAGNOSTIC_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
