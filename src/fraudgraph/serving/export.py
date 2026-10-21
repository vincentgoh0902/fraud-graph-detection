"""Building the serving tables from model outputs (D-038), and the Space subset (D-040).

Only NumPy and pandas here; scripts/export_serving.py does the scoring with XGBoost.
"""

import numpy as np
import pandas as pd

from fraudgraph.serving.graph import bfs_hops


def top_contributions(
    tx_ids: np.ndarray, contribs: np.ndarray, features: list[str], groups: list[str], n: int
) -> pd.DataFrame:
    """The n largest SHAP values by size per transaction, sign kept, long format.

    A stable sort keeps feature order among equal sizes, so the output is reproducible.
    """
    order = np.argsort(-np.abs(contribs), axis=1, kind="stable")[:, :n]
    k = order.shape[1]
    rows = np.repeat(np.arange(len(tx_ids)), k)  # row index of each output line
    ranks = np.tile(np.arange(1, k + 1), len(tx_ids))
    cols = order.ravel()
    return pd.DataFrame(
        {
            "tx_id": np.asarray(tx_ids)[rows],
            "rank": ranks,
            "feature": np.asarray(features)[cols],
            "group": np.asarray(groups)[cols],
            "shap": contribs[rows, cols].astype(np.float32),
        }
    )


def transactions_table(tx_ids, steps, labels, scores, threshold: float, community) -> pd.DataFrame:
    """One row per served transaction; flagged when score >= threshold, as in the protocol."""
    return pd.DataFrame(
        {
            "tx_id": tx_ids,
            "time_step": steps,
            "label": labels,
            "risk_score": np.asarray(scores, dtype=np.float32),
            "predicted_label": (np.asarray(scores) >= threshold).astype(np.int8),
            "community_id": community,
        }
    )


def top_alerts(tx: pd.DataFrame, k: int) -> pd.DataFrame:
    """The k highest-risk transactions per step; equal scores ordered by tx_id."""
    t = tx.sort_values(["time_step", "risk_score", "tx_id"], ascending=[True, False, True])
    return t.groupby("time_step").head(k).reset_index(drop=True)


def hop_neighbourhood(edges: pd.DataFrame, seeds, hops: int) -> set[int]:
    """Every node within `hops` of any seed, following edges in both directions."""
    adj: dict[int, set[int]] = {}
    for a, b in zip(edges["src"], edges["dst"], strict=True):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def pairs(frontier):
        return [(a, b) for a in frontier for b in adj.get(a, ())]

    out: set[int] = set()
    for s in seeds:
        out |= set(bfs_hops(pairs, s, hops))
    return out


def demo_subset(
    tx: pd.DataFrame,
    shap: pd.DataFrame,
    edges: pd.DataFrame,
    alerts_per_step: int,
    hops: int,
    include_labels: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    """What the public Space may ship (D-012, D-040).

    Scores and SHAP for every served transaction (model outputs), but edges only inside the
    `hops`-neighbourhoods of each step's top alerts, so no full edge list leaves the repo.
    Returns the tables and the alert ids whose neighbourhoods are complete.
    """
    roots = [int(t) for t in top_alerts(tx, alerts_per_step)["tx_id"]]
    keep = hop_neighbourhood(edges, roots, hops)
    sub_edges = edges[edges["src"].isin(keep) & edges["dst"].isin(keep)].reset_index(drop=True)
    sub_tx = tx if include_labels else tx.drop(columns="label")
    return sub_tx.reset_index(drop=True), shap, sub_edges, roots
