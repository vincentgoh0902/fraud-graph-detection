"""A tiny synthetic Elliptic dataset in the exact raw format, so tests never need
the real 700 MB download (and CI can run them)."""

import os

import numpy as np
import pandas as pd
import pytest

from fraudgraph.data.load import read_raw
from fraudgraph.data.split import SplitConfig

N_LOCAL, N_AGG = 93, 72
N_STEPS, PER_STEP = 6, 6


@pytest.fixture
def raw_files(tmp_path):
    """Write synthetic features/classes/edges CSVs; return their paths."""
    rng = np.random.default_rng(0)
    rows, classes, edges = [], [], []
    raw_labels = ["1", "2", "2", "unknown", "unknown", "unknown"]  # per step
    for step in range(1, N_STEPS + 1):
        ids = [step * 100 + i for i in range(PER_STEP)]
        for tx, raw_label in zip(ids, raw_labels, strict=True):
            rows.append([tx, step, *rng.normal(size=N_LOCAL + N_AGG)])
            classes.append((tx, raw_label))
        # A chain within the step: every edge stays inside one time step, like the real data.
        edges += [(a, b) for a, b in zip(ids, ids[1:], strict=False)]

    features = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(features, header=False, index=False)
    classes_p = tmp_path / "classes.csv"
    pd.DataFrame(classes, columns=["txId", "class"]).to_csv(classes_p, index=False)
    edges_p = tmp_path / "edges.csv"
    pd.DataFrame(edges, columns=["txId1", "txId2"]).to_csv(edges_p, index=False)
    return features, classes_p, edges_p


@pytest.fixture
def data(raw_files):
    return read_raw(*raw_files, n_local=N_LOCAL, n_agg=N_AGG)


@pytest.fixture
def split_cfg():
    # Same shape as configs/split.yaml, scaled down to six steps.
    return SplitConfig(train_steps=(1, 3), val_steps=(4, 4), test_steps=(5, 6))


# torch and XGBoost each bring their own OpenMP runtime, and on macOS a process that uses
# both crashes (D-030). So tests that import torch live in tests/torch/ and are skipped in
# the main pytest process; tests/test_torch_suite.py runs them in a separate process.
TORCH_TESTS_ENV = "FRAUDGRAPH_TORCH_TESTS"
if not os.environ.get(TORCH_TESTS_ENV):
    collect_ignore_glob = ["torch/*"]


# --- Phase 6 serving: a tiny export in the exact artefact format (no model, no Neo4j). ---
# Step 35: 1-2-3-4-5 chain plus 1->6, 7->1, and 2, 6, 7 all paying 8 (a fan-in).
# Step 36: 101->102->103 and an isolated 104. Flagged (score >= 0.5): 2, 6, 7, 8, 102.
SERVING_EDGES = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 6), (7, 1), (2, 8), (6, 8), (7, 8),
                 (101, 102), (102, 103)]  # fmt: skip
SERVING_RISK = {1: 0.30, 2: 0.90, 3: 0.20, 4: 0.10, 5: 0.05, 6: 0.70, 7: 0.80, 8: 0.95,
                101: 0.40, 102: 0.60, 103: 0.10, 104: 0.01}  # fmt: skip


def serving_tables():
    """(transactions, shap, edges, meta, per_step) shaped like scripts/export_serving.py output."""
    ids = list(SERVING_RISK)
    steps = [35 if t < 100 else 36 for t in ids]
    labels = [1 if t in (2, 8, 3) else 0 if t in (1, 4, 6, 101, 102) else -1 for t in ids]
    community = [0 if t <= 4 else 1 if t < 100 else 2 for t in ids]
    risk = [SERVING_RISK[t] for t in ids]
    tx = pd.DataFrame(
        {
            "tx_id": ids,
            "time_step": steps,
            "label": labels,
            "risk_score": np.array(risk, dtype=np.float32),
            "predicted_label": [int(r >= 0.5) for r in risk],
            "community_id": community,
        }
    )
    shap = pd.DataFrame(
        [
            {"tx_id": t, "rank": r + 1, "feature": f"local_{r + 1}", "group": "local",
             "shap": (1.0 - 0.1 * r) * (1 if r % 2 == 0 else -1)}
            for t in ids
            for r in range(3)
        ]
    )  # fmt: skip
    edges = pd.DataFrame(SERVING_EDGES, columns=["src", "dst"])
    meta = {
        "model_version": "xgboost-test-seed0",
        "threshold": 0.5,
        "base_value": -2.0,
        "steps": [35, 36],
        "demo": False,
        "has_full_graph": True,
        "labels_included": True,
        "max_hops": 3,
        "neighbourhood_roots": None,
    }
    per_step = pd.DataFrame({"time_step": [35, 36], "f1": [0.8, 0.1], "n_illicit": [3, 0]})
    return tx, shap, edges, meta, per_step


@pytest.fixture
def serving_dir(tmp_path):
    from fraudgraph.serving.artifacts import write_artifacts

    write_artifacts(tmp_path / "serving", *serving_tables())
    return tmp_path / "serving"


@pytest.fixture
def service(serving_dir):
    from fraudgraph.serving.artifacts import load_artifacts
    from fraudgraph.serving.graph import InMemoryStore
    from fraudgraph.serving.service import Limits, ServingService

    data = load_artifacts(serving_dir)
    limits = Limits(max_hops=3, max_k=20, max_nodes=500, max_path_len=6, max_paths=20)
    return ServingService(data, InMemoryStore(data.edges, data.transactions), limits)
