"""Investigation queries on the in-memory graph store, and the shared rules both stores use.

The same queries run in Neo4j in the full stack; tests/test_neo4j_serving.py checks the two
agree on the real graph (D-008, D-040).
"""

import pandas as pd
import pytest

from fraudgraph.serving.graph import InMemoryStore, bfs_hops, truncate

from .conftest import serving_tables


@pytest.fixture
def store():
    tx, _, edges, _, _ = serving_tables()
    return InMemoryStore(edges, tx.set_index("tx_id"))


def test_hop_distances_follow_edges_both_ways(store):
    assert store.hop_distances(1, 1) == {1: 0, 2: 1, 6: 1, 7: 1}
    assert store.hop_distances(1, 2) == {1: 0, 2: 1, 6: 1, 7: 1, 3: 2, 8: 2}


def test_hop_distances_never_leave_the_step(store):
    assert store.hop_distances(101, 3) == {101: 0, 102: 1, 103: 2}
    assert store.hop_distances(104, 2) == {104: 0}  # isolated: itself only


def test_bfs_keeps_the_shortest_distance():
    pairs = {1: [2, 3], 2: [1, 3], 3: [1, 2]}
    assert bfs_hops(lambda frontier: [(a, b) for a in frontier for b in pairs[a]], 1, 2) == {
        1: 0, 2: 1, 3: 1
    }  # fmt: skip


def test_truncate_keeps_near_hops_then_high_risk():
    hops = {1: 0, 2: 1, 6: 1, 7: 1, 3: 2}
    risk = pd.Series({1: 0.3, 2: 0.9, 6: 0.7, 7: 0.8, 3: 0.2})
    kept, cut = truncate(hops, risk, max_nodes=3)
    assert kept == [1, 2, 7] and cut
    kept, cut = truncate(hops, risk, max_nodes=10)
    assert kept == [1, 2, 7, 6, 3] and not cut


def test_edges_among_returns_directed_edges_inside_the_set(store):
    e = store.edges_among([1, 2, 7, 8])
    assert set(zip(e["src"], e["dst"], strict=True)) == {(1, 2), (7, 1), (2, 8), (7, 8)}


def test_shortest_paths(store):
    assert store.shortest_paths(1, 4, max_len=6, max_paths=20) == [[1, 2, 3, 4]]
    # Three equally short routes, returned sorted so both stores agree.
    assert store.shortest_paths(1, 8, max_len=6, max_paths=20) == [
        [1, 2, 8], [1, 6, 8], [1, 7, 8]
    ]  # fmt: skip
    assert store.shortest_paths(1, 8, max_len=6, max_paths=2) == [[1, 2, 8], [1, 6, 8]]
    assert store.shortest_paths(1, 5, max_len=3, max_paths=20) == []  # longer than allowed
    assert store.shortest_paths(1, 101, max_len=6, max_paths=20) == []  # different steps


def test_fan_in_counts_distinct_flagged_payers(store):
    f = store.fan_in(35).set_index("tx_id")["n_flagged_payers"]
    # 8 is paid by 2, 6 and 7 (all flagged); 1 by 7; 3 by 2. Direction matters: 2 is
    # paid only by 1, which is not flagged, so 2 is absent.
    assert f.to_dict() == {8: 3, 1: 1, 3: 1}


def test_community_summary(store):
    c = store.community_summary(35).set_index("community_id")
    assert c.loc[0, "size"] == 4 and c.loc[0, "n_flagged"] == 1
    assert c.loc[1, "n_flagged"] == 3
    assert c.loc[1, "mean_risk"] == pytest.approx((0.05 + 0.70 + 0.80 + 0.95) / 4)
