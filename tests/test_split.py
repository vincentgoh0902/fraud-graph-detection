import numpy as np
import pandas as pd
import pytest

from fraudgraph.data.load import UNKNOWN
from fraudgraph.data.split import (
    LeakageError,
    SplitConfig,
    assert_no_cross_split_edges,
    load_split_config,
    temporal_split,
)


def steps_of(data, mask):
    return set(data.nodes.loc[mask, "time_step"])


def test_masks_cover_the_right_steps(data, split_cfg):
    s = temporal_split(data.nodes, split_cfg)
    assert steps_of(data, s.train) == {1, 2, 3}
    assert steps_of(data, s.val) == {4}
    assert steps_of(data, s.test) == {5, 6}


def test_masks_do_not_overlap(data, split_cfg):
    s = temporal_split(data.nodes, split_cfg)
    assert not (s.train & s.val).any()
    assert not (s.train & s.test).any()
    assert not (s.val & s.test).any()


def test_unknown_labels_excluded_but_kept_in_nodes(data, split_cfg):
    s = temporal_split(data.nodes, split_cfg)
    unknown = (data.nodes["label"] == UNKNOWN).to_numpy()
    assert not (unknown & (s.train | s.val | s.test)).any()
    # Unknown nodes are still in the table, so GNNs can pass messages through them.
    assert unknown.sum() > 0 and len(s.train) == len(data.nodes)


def test_every_labelled_node_in_exactly_one_split(data, split_cfg):
    s = temporal_split(data.nodes, split_cfg)
    labelled = (data.nodes["label"] != UNKNOWN).to_numpy()
    count = s.train.astype(int) + s.val.astype(int) + s.test.astype(int)
    assert np.array_equal(count[labelled], np.ones(labelled.sum(), dtype=int))


def test_project_config_loads():
    cfg = load_split_config()
    assert cfg == SplitConfig(train_steps=(1, 29), val_steps=(30, 34), test_steps=(35, 49))


@pytest.mark.parametrize(
    "train, val, test",
    [
        ((1, 3), (3, 4), (5, 6)),  # train and val overlap at step 3
        ((4, 5), (1, 3), (6, 6)),  # val before train: tuning on the past, training on the future
        ((1, 3), (5, 6), (4, 4)),  # test before val
        ((3, 1), (4, 4), (5, 6)),  # start after end
    ],
)
def test_bad_config_rejected(train, val, test):
    with pytest.raises(ValueError):
        SplitConfig(train_steps=train, val_steps=val, test_steps=test)


def test_steps_outside_data_rejected(data):
    cfg = SplitConfig(train_steps=(1, 3), val_steps=(4, 4), test_steps=(5, 7))  # data stops at 6
    with pytest.raises(ValueError, match="outside"):
        temporal_split(data.nodes, cfg)


def test_clean_graph_has_no_cross_split_edges(data, split_cfg):
    assert_no_cross_split_edges(data.nodes, data.edges, split_cfg)


def test_injected_cross_split_edge_is_caught(data, split_cfg):
    # Prove the leakage check can fail: a step-3 (train) -> step-5 (test) edge.
    leaky = pd.concat([data.edges, pd.DataFrame({"src": [300], "dst": [500]})])
    with pytest.raises(LeakageError):
        assert_no_cross_split_edges(data.nodes, leaky, split_cfg)
