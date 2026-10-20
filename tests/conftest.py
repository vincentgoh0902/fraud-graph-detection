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
