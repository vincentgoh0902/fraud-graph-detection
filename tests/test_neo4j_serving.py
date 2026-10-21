"""Neo4j and the in-memory store give the same answers on the real graph (D-008, D-040).

Local only: skipped unless artifacts/serving exists and Neo4j holds the graph and the
predictions (scripts/load_neo4j.py, export_serving.py, write_predictions.py). CI has neither.
"""

import pandas as pd
import pytest

from fraudgraph.config import load_config, resolve

serving_dir = resolve(load_config("serving")["output_dir"])


def _ready():
    if not (serving_dir / "meta.json").exists():
        return None
    try:
        from fraudgraph.graph.neo4j_client import connect
        from fraudgraph.serving.artifacts import load_artifacts

        data = load_artifacts(serving_dir)
        driver = connect()
        n = driver.execute_query(
            "MATCH (t:Transaction {model_version: $v}) RETURN count(t) AS n",
            {"v": data.meta["model_version"]},
        ).records[0]["n"]
        return (driver, data) if n == len(data.transactions) else None
    except Exception:
        return None


pytestmark = pytest.mark.skipif(
    (ready := _ready()) is None, reason="needs artifacts/serving and Neo4j with predictions"
)


@pytest.fixture(scope="module")
def stores():
    from fraudgraph.serving.graph import InMemoryStore, Neo4jStore

    driver, data = ready
    yield Neo4jStore(driver), InMemoryStore(data.edges, data.transactions), data
    driver.close()


def _sample(data, per_step=2):
    """Top alerts plus the highest-degree transaction of a few steps (the hard cases)."""
    tx = data.transactions.reset_index()
    tx = tx[tx["time_step"].isin([35, 40, 43, 49])]
    top = tx.sort_values("risk_score", ascending=False).groupby("time_step").head(per_step)
    deg = pd.concat([data.edges["src"], data.edges["dst"]]).value_counts()
    return [int(t) for t in top["tx_id"]] + [int(deg.idxmax())]


def test_hop_distances_agree(stores):
    neo, mem, data = stores
    for t in _sample(data):
        for hops in (1, 2, 3):
            assert neo.hop_distances(t, hops) == mem.hop_distances(t, hops), (t, hops)


def test_edges_among_agree(stores):
    neo, mem, data = stores
    for t in _sample(data):
        ids = list(mem.hop_distances(t, 2))
        a = neo.edges_among(ids).sort_values(["src", "dst"], ignore_index=True)
        b = mem.edges_among(ids).sort_values(["src", "dst"], ignore_index=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_paths_agree(stores):
    neo, mem, data = stores
    ids = _sample(data, per_step=3)
    for a, b in zip(ids, ids[1:], strict=False):
        assert neo.shortest_paths(a, b, 6, 20) == mem.shortest_paths(a, b, 6, 20), (a, b)


@pytest.mark.parametrize("step", [35, 40, 43, 49])
def test_fan_in_and_communities_agree(stores, step):
    neo, mem, _ = stores
    pd.testing.assert_frame_equal(neo.fan_in(step), mem.fan_in(step), check_dtype=False)
    a = neo.community_summary(step).sort_values("community_id", ignore_index=True)
    b = mem.community_summary(step).sort_values("community_id", ignore_index=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-6)
