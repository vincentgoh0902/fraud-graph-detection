import pandas as pd
import pytest

from fraudgraph.data.load import (
    ILLICIT,
    LICIT,
    UNKNOWN,
    DataValidationError,
    read_raw,
    validate,
)
from tests.conftest import N_AGG, N_LOCAL, N_STEPS, PER_STEP


def test_columns_and_shapes(data):
    assert list(data.nodes.columns[:3]) == ["tx_id", "time_step", "label"]
    assert len(data.local_cols) == N_LOCAL and len(data.agg_cols) == N_AGG
    assert data.local_cols[0] == "local_1" and data.agg_cols[-1] == "agg_72"
    assert len(data.nodes) == N_STEPS * PER_STEP
    assert list(data.edges.columns) == ["src", "dst"]


def test_time_step_is_not_a_feature(data):
    assert "time_step" not in data.feature_cols


def test_label_mapping(data):
    # In the fixture, the first tx of each step is raw "1" and the second raw "2".
    label = data.nodes.set_index("tx_id")["label"]
    assert label[100] == ILLICIT  # raw "1"
    assert label[101] == LICIT  # raw "2"
    assert label[103] == UNKNOWN  # raw "unknown"
    assert set(data.nodes["label"]) == {ILLICIT, LICIT, UNKNOWN}


def test_clean_data_validates(data):
    validate(data, {"n_nodes": N_STEPS * PER_STEP, "n_illicit": N_STEPS})


def test_unexpected_label_rejected(raw_files):
    features, classes, edges = raw_files
    df = pd.read_csv(classes, dtype=str)
    df.loc[0, "class"] = "3"
    df.to_csv(classes, index=False)
    with pytest.raises(DataValidationError, match="Unexpected label"):
        read_raw(features, classes, edges, N_LOCAL, N_AGG)


def test_duplicate_tx_rejected(raw_files):
    features, classes, edges = raw_files
    df = pd.read_csv(classes, dtype=str)
    pd.concat([df, df.iloc[[0]]]).to_csv(classes, index=False)
    with pytest.raises(DataValidationError, match="Duplicate"):
        read_raw(features, classes, edges, N_LOCAL, N_AGG)


def test_dangling_edge_rejected(data):
    bad = pd.concat([data.edges, pd.DataFrame({"src": [100], "dst": [999_999]})])
    with pytest.raises(DataValidationError, match="unknown transactions"):
        validate(type(data)(data.nodes, bad, data.local_cols, data.agg_cols))


def test_cross_step_edge_rejected(data):
    # tx 100 is in step 1, tx 600 in step 6. The real data never has this;
    # if it ever did, the disjoint-graph assumption (D-001) would be broken.
    bad = pd.concat([data.edges, pd.DataFrame({"src": [100], "dst": [600]})])
    with pytest.raises(DataValidationError, match="different time steps"):
        validate(type(data)(data.nodes, bad, data.local_cols, data.agg_cols))


def test_wrong_counts_rejected(data):
    with pytest.raises(DataValidationError, match="published figures"):
        validate(data, {"n_nodes": 203769})
