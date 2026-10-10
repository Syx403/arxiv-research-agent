from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ara.llm.pricing import FLASH_OFF_PEAK, FLASH_PEAK, LUNA, Rates, Usage, rates, upper_bound

SINGAPORE = timezone(timedelta(hours=8))


def test_each_token_is_billed_at_exactly_one_rate() -> None:
    usage = Usage(
        input_tokens=10_000,
        cached_tokens=6_000,
        cache_write_tokens=2_000,
        output_tokens=1_000,
        reasoning_tokens=400,
    )
    # 2,000 uncached x 0.10 + 6,000 x 0.01 + 2,000 x 0.125 + 1,000 x 0.50 = 1,010 per million
    assert usage.cost(LUNA) == Decimal("0.00101")


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        (datetime(2026, 10, 7, 2, 30, tzinfo=UTC), FLASH_PEAK),  # Wednesday, first window
        (datetime(2026, 10, 7, 9, 59, tzinfo=UTC), FLASH_PEAK),  # Wednesday, second window
        (datetime(2026, 10, 7, 10, 0, tzinfo=UTC), FLASH_OFF_PEAK),  # windows end exclusive
        (datetime(2026, 10, 7, 4, 30, tzinfo=UTC), FLASH_OFF_PEAK),  # between the windows
        (datetime(2026, 10, 10, 2, 30, tzinfo=UTC), FLASH_OFF_PEAK),  # Saturday
        (datetime(2026, 10, 7, 10, 30, tzinfo=SINGAPORE), FLASH_PEAK),  # = 02:30 UTC
    ],
)
def test_deepseek_peak_hours(at: datetime, expected: Rates) -> None:
    assert rates("deepseek-flash", at) == expected


def test_a_reservation_assumes_every_input_token_is_written_to_the_cache() -> None:
    assert upper_bound(1_000_000, 0, LUNA) == Decimal("0.125")
    assert upper_bound(0, 1_000_000, LUNA) == Decimal("0.5")


def test_claude_haiku_has_its_own_rates() -> None:
    """D44: Haiku 5.5, prompts up to 100K tokens, 5-minute cache writes."""
    r = rates("claude-haiku-5-5", datetime(2026, 10, 11, tzinfo=UTC))
    assert (r.input, r.cached, r.cache_write, r.output) == (
        Decimal("0.10"),
        Decimal("0.01"),
        Decimal("0.125"),
        Decimal("0.50"),
    )
