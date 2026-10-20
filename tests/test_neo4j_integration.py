"""GDS against the Python implementation on the real graph (D-019, D-020).

Local only: skipped unless Neo4j is reachable and the data has been downloaded and loaded
(`docker compose up -d neo4j`, then `scripts/load_neo4j.py`). CI has neither.
"""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from fraudgraph.config import load_config, resolve
from fraudgraph.data.load import load_elliptic
from fraudgraph.graph.structural import structural_features

data_cfg = load_config("data")
raw = resolve(data_cfg["raw_dir"]) / data_cfg["files"]["features"]


def _driver_or_none():
    try:
        from fraudgraph.graph.neo4j_client import connect

        return connect()
    except Exception:
        return None


pytestmark = pytest.mark.skipif(
    not raw.exists() or (drv := _driver_or_none()) is None,
    reason="needs the Elliptic data and a running Neo4j",
)


@pytest.fixture(scope="module")
def features():
    from fraudgraph.graph.neo4j_client import structural_gds

    cfg = load_config("graph")
    data = load_elliptic()
    with drv:
        runs = [
            structural_gds(drv, cfg["neo4j"]["database"], cfg["pagerank"], cfg["louvain"])
            for _ in range(2)
        ]
    gds = [r.set_index("tx_id").loc[data.nodes["tx_id"]] for r in runs]
    py = structural_features(data.nodes, data.edges, cfg["pagerank"], cfg["louvain"])
    return data, gds, py.set_index("tx_id")


def test_degree_and_clustering_match_exactly(features):
    _, (gds, _), py = features
    for col in ("in_degree", "out_degree"):
        assert (gds[col].to_numpy() == py[col].to_numpy()).all()
    np.testing.assert_allclose(gds["clustering"], py["clustering"], atol=1e-9)


def test_pagerank_matches(features):
    _, (gds, _), py = features
    np.testing.assert_allclose(gds["pagerank"], py["pagerank"], atol=1e-6)


def test_gds_louvain_is_reproducible(features):
    _, (first, second), _ = features
    assert adjusted_rand_score(first["community_id"], second["community_id"]) == 1.0


def test_no_gds_community_spans_two_steps(features):
    data, (gds, _), _ = features
    step = data.nodes.set_index("tx_id")["time_step"]
    assert (gds.assign(step=step).groupby("community_id")["step"].nunique() == 1).all()


def test_louvain_roughly_agrees_with_networkx(features):
    # Different Louvain implementations: similar partitions, not identical (D-020).
    _, (gds, _), py = features
    assert adjusted_rand_score(gds["community_id"], py["community_id"]) > 0.8
