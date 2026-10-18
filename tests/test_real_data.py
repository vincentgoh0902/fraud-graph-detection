"""Checks on the real dataset. Skipped when it hasn't been downloaded (e.g. in CI);
run locally after `scripts/download_data.py`."""

import pytest

from fraudgraph.config import load_config, resolve
from fraudgraph.data.load import load_elliptic
from fraudgraph.data.split import assert_no_cross_split_edges, load_split_config, temporal_split

cfg = load_config("data")
raw = resolve(cfg["raw_dir"]) / cfg["files"]["features"]
pytestmark = pytest.mark.skipif(not raw.exists(), reason="Elliptic data not downloaded")


@pytest.fixture(scope="module")
def real():
    # load_elliptic validates counts against the published figures and checks that
    # no edge crosses time steps; it raises if either fails.
    return load_elliptic()


def test_real_split_has_no_leakage(real):
    split_cfg = load_split_config()
    assert_no_cross_split_edges(real.nodes, real.edges, split_cfg)
    s = temporal_split(real.nodes, split_cfg)
    assert s.train.sum() > 0 and s.val.sum() > 0 and s.test.sum() > 0
