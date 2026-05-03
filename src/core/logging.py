from __future__ import annotations

import logging
import re
from typing import Any


_REDACTION = "[REDACTED]"
_SECRET_PATTERNS = [
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)([^\s,;]+)"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)((?:api[_-]?key|token)\s*[:=]\s*)([A-Za-z0-9._\-]{16,})"),
    re.compile(r"\b(sk-[A-Za-z0-9._\-]{8,})\b"),
    re.compile(r"\b(sk-ant-[A-Za-z0-9._\-]{8,})\b"),
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]{16,})"),
]


def redact_secrets(value: Any) -> str:
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(lambda match: f"{match.group(1)}{_REDACTION}", text)
        else:
            text = pattern.sub(_REDACTION, text)
    return text


class SecretMaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)
        record.msg = redact_secrets(rendered)
        record.args = ()
        return True


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    root = logging.getLogger()
    if not any(isinstance(flt, SecretMaskingFilter) for flt in root.filters):
        root.addFilter(SecretMaskingFilter())
    for handler in root.handlers:
        if not any(isinstance(flt, SecretMaskingFilter) for flt in handler.filters):
            handler.addFilter(SecretMaskingFilter())
