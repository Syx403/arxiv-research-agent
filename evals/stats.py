"""Small-sample statistics (DESIGN §11.1): means with 95% bootstrap confidence intervals, and paired
differences between two arms graded on the same items."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def mean_ci(
    values: Sequence[float], *, resamples: int = 2_000, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, low, high): the 2.5th and 97.5th percentiles of resampled means."""
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(resamples, len(data)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(data.mean()), float(low), float(high)


@dataclass(frozen=True)
class Paired:
    mean: float  # mean of a - b over the shared items
    low: float
    high: float
    better: int  # items where a > b
    worse: int  # items where a < b


def paired(a: Sequence[float], b: Sequence[float]) -> Paired:
    """Bootstrap the per-item differences a - b: arms on the same items are compared item by item,
    which is tighter (and correct) where comparing two separate intervals is not."""
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, low, high = mean_ci(diffs.tolist())
    return Paired(mean, low, high, int((diffs > 0).sum()), int((diffs < 0).sum()))
