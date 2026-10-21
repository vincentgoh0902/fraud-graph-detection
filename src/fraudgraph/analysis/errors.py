"""False-negative analysis: which illicit transactions does the model miss, and why? (D-033)"""

import numpy as np
import pandas as pd


def missed_illicit(y: np.ndarray, score: np.ndarray, threshold: float) -> np.ndarray:
    """Illicit rows scored below the threshold (flagged means score >= threshold)."""
    return (y == 1) & (score < threshold)


def standardised_difference(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> pd.Series:
    """(mean a - mean b) / pooled standard deviation per column, largest |difference| first.

    Unlike a raw difference it is comparable across features on different scales; unlike a
    t-test it does not grow with the sample size. NaN where neither group varies.
    """
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na - 1) * a[cols].var(ddof=1) + (nb - 1) * b[cols].var(ddof=1)) / (na + nb - 2)
    )
    d = (a[cols].mean() - b[cols].mean()) / pooled.where(pooled > 0)
    return d.reindex(d.abs().sort_values(ascending=False, na_position="last").index)


def seed_agreement(missed: list[np.ndarray]) -> pd.Series:
    """How many illicit rows are missed by 0, 1, ... n seeds. Masks align row by row."""
    counts = np.sum(missed, axis=0)
    return pd.Series(counts).value_counts().reindex(range(len(missed) + 1), fill_value=0)
