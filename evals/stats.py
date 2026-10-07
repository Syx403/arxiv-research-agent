"""Small-sample statistics (DESIGN §11.1): means with 95% bootstrap confidence intervals."""

from collections.abc import Sequence

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
