from __future__ import annotations

import logging

from src.core.logging import SecretMaskingFilter, redact_secrets


def test_redact_secrets_masks_authorization_bearer() -> None:
    text = "Authorization: Bearer abc123def456ghi789"
    redacted = redact_secrets(text)
    assert "abc123def456ghi789" not in redacted
    assert "[REDACTED]" in redacted


def test_logging_filter_masks_record_args() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x-api-key=%s",
        args=("key_live_abcdefghijklmnopqrstuvwxyz",),
        exc_info=None,
    )
    assert SecretMaskingFilter().filter(record)
    rendered = record.getMessage()
    assert "key_live_abcdefghijklmnopqrstuvwxyz" not in rendered
    assert "[REDACTED]" in rendered
