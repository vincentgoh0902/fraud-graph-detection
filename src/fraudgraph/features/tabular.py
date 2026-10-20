"""Tabular feature sets (a) and (b) from the brief.

(a) "local": the 93 per-transaction features.
(b) "all":   local + the 72 aggregated features. These are already one-hop neighbourhood
             statistics, so (a) -> (b) is itself a test of graph information (D-003).
Feature set (c), local + our graph features, arrives in Phase 3.
"""

import numpy as np

from fraudgraph.data.load import EllipticData

FEATURE_SETS = ("local", "all")

# Columns that must never be model inputs: identifiers, the target, and time (D-013).
NON_FEATURES = {"tx_id", "label", "time_step"}


def feature_columns(data: EllipticData, feature_set: str) -> list[str]:
    if feature_set == "local":
        cols = data.local_cols
    elif feature_set == "all":
        cols = data.local_cols + data.agg_cols
    else:
        raise ValueError(f"Unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")
    leaked = NON_FEATURES & set(cols)
    if leaked:
        raise ValueError(f"Non-feature columns in feature set: {sorted(leaked)}")
    return list(cols)


def feature_matrix(data: EllipticData, feature_set: str) -> np.ndarray:
    """All nodes (labelled or not) as a float32 matrix, rows aligned with data.nodes."""
    return data.nodes[feature_columns(data, feature_set)].to_numpy(dtype=np.float32)
