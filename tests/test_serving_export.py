"""Phase 6 export (D-038, D-040): the tables the API and the Space read."""

import numpy as np
import pandas as pd
import pytest

from fraudgraph.serving.artifacts import load_artifacts, write_artifacts
from fraudgraph.serving.export import (
    demo_subset,
    hop_neighbourhood,
    top_alerts,
    top_contributions,
    transactions_table,
)

from .conftest import SERVING_EDGES, serving_tables


def test_top_contributions_ranks_by_size_and_keeps_sign():
    contribs = np.array([[0.1, -2.0, 0.5], [0.0, 0.3, -0.3]], dtype=np.float32)
    out = top_contributions(np.array([10, 20]), contribs, ["a", "b", "c"], ["g1", "g2", "g3"], 2)
    first = out[out["tx_id"] == 10]
    assert list(first["feature"]) == ["b", "c"]
    assert list(first["rank"]) == [1, 2]
    assert first["shap"].iloc[0] == pytest.approx(-2.0)  # sign kept: pushes towards licit
    assert list(first["group"]) == ["g2", "g3"]
    # Equal sizes keep feature order, so the output does not depend on sort internals.
    assert list(out[out["tx_id"] == 20]["feature"]) == ["b", "c"]


def test_transactions_table_flags_at_or_above_the_threshold():
    t = transactions_table(
        tx_ids=np.array([1, 2, 3]),
        steps=np.array([35, 35, 36]),
        labels=np.array([1, 0, -1]),
        scores=np.array([0.5, 0.49, 0.9], dtype=np.float32),
        threshold=0.5,
        community=np.array([7, 7, 8]),
    )
    assert list(t["predicted_label"]) == [1, 0, 1]  # same >= rule as the protocol
    assert list(t.columns) == [
        "tx_id", "time_step", "label", "risk_score", "predicted_label", "community_id"
    ]  # fmt: skip


def test_top_alerts_per_step_ties_broken_by_id():
    tx = pd.DataFrame(
        {"tx_id": [5, 3, 4, 9], "time_step": [35, 35, 35, 36],
         "risk_score": [0.9, 0.9, 0.1, 0.2]}
    )  # fmt: skip
    out = top_alerts(tx, k=2)
    assert list(out["tx_id"]) == [3, 5, 9]


def test_hop_neighbourhood_is_undirected_and_bounded():
    edges = pd.DataFrame(SERVING_EDGES, columns=["src", "dst"])
    assert hop_neighbourhood(edges, [4], hops=1) == {3, 4, 5}
    assert hop_neighbourhood(edges, [4], hops=2) == {2, 3, 4, 5}
    assert hop_neighbourhood(edges, [101], hops=5) == {101, 102, 103}  # never leaves its step


def test_demo_subset_keeps_alert_neighbourhoods_only():
    tx, shap, edges, _, _ = serving_tables()
    sub_tx, sub_shap, sub_edges, roots = demo_subset(
        tx, shap, edges, alerts_per_step=1, hops=1, include_labels=False
    )
    assert roots == [8, 102]  # the top alert of each step
    kept = set(sub_edges["src"]) | set(sub_edges["dst"])
    assert kept == {8, 2, 6, 7, 101, 102, 103}
    # Only edges with both ends inside a kept neighbourhood ship.
    assert (7, 1) not in set(zip(sub_edges["src"], sub_edges["dst"], strict=True))
    # Scores and SHAP ship for every served transaction; labels do not, unless allowed.
    assert len(sub_tx) == len(tx) and len(sub_shap) == len(shap)
    assert "label" not in sub_tx.columns


def test_demo_subset_keeps_labels_when_allowed():
    tx, shap, edges, _, _ = serving_tables()
    sub_tx, *_ = demo_subset(tx, shap, edges, alerts_per_step=1, hops=1, include_labels=True)
    assert "label" in sub_tx.columns


def test_artifacts_round_trip(tmp_path):
    tables = serving_tables()
    write_artifacts(tmp_path, *tables)
    data = load_artifacts(tmp_path)
    assert data.meta["model_version"] == "xgboost-test-seed0"
    assert data.transactions.index.name == "tx_id"
    assert data.transactions.loc[8, "risk_score"] == pytest.approx(0.95)
    assert len(data.edges) == len(SERVING_EDGES)
    assert list(data.per_step["time_step"]) == [35, 36]
