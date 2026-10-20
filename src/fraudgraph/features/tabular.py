"""Feature sets (a), (b) and (c) from the brief, plus the Phase 3 variants.

(a) "local":           the 93 per-transaction features.
(b) "all":             local + the 72 aggregated features. These are already one-hop
                       neighbourhood statistics, so (a) -> (b) is itself a test of graph
                       information (D-003).
(c) "local+graph":     local + our graph features, with neighbour risk from a local-only
                       stage-1 model (D-018).
    "all+graph":       (b) + our graph features, stage-1 on all features. Graph features
                       only "help" if this beats (b) (D-003).
    "local+structure": local + graph shape only, nothing built from predicted risk (D-021).
Graph columns come from data/processed/graph_features.parquet (features.graph).
"""

import numpy as np

from fraudgraph.data.load import EllipticData
from fraudgraph.features.graph import STRUCTURE_FEATURES, risk_columns

FEATURE_SETS = ("local", "all", "local+graph", "all+graph", "local+structure")

# Columns that must never be model inputs: identifiers, the target, time (D-013), and
# community labels, which are arbitrary integers (D-002).
NON_FEATURES = {"tx_id", "label", "time_step", "community_id"}


def feature_columns(data: EllipticData, feature_set: str) -> list[str]:
    if feature_set == "local":
        cols = data.local_cols
    elif feature_set == "all":
        cols = data.local_cols + data.agg_cols
    elif feature_set == "local+graph":
        cols = data.local_cols + STRUCTURE_FEATURES + risk_columns("local")
    elif feature_set == "all+graph":
        cols = data.local_cols + data.agg_cols + STRUCTURE_FEATURES + risk_columns("all")
    elif feature_set == "local+structure":
        cols = data.local_cols + STRUCTURE_FEATURES
    else:
        raise ValueError(f"Unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")
    leaked = NON_FEATURES & set(cols)
    if leaked:
        raise ValueError(f"Non-feature columns in feature set: {sorted(leaked)}")
    missing = [c for c in cols if c not in data.nodes]
    if missing:
        raise ValueError(
            f"Columns {missing[:3]}... missing for {feature_set!r}; build them with "
            "scripts/build_graph_features.py and attach with features.graph."
        )
    return list(cols)


def feature_matrix(data: EllipticData, feature_set: str) -> np.ndarray:
    """All nodes (labelled or not) as a float32 matrix, rows aligned with data.nodes.

    Graph features may be NaN (a node with no neighbours); XGBoost handles missing values.
    """
    return data.nodes[feature_columns(data, feature_set)].to_numpy(dtype=np.float32)
