"""Graph features on hand-worked tiny graphs, and the assembled table on the synthetic
dataset. Uses the NetworkX/NumPy implementation, so no Neo4j is needed (D-019)."""

import numpy as np
import pandas as pd
import pytest

from fraudgraph.features.graph import (
    STRUCTURE_FEATURES,
    attach_graph_features,
    build_graph_features,
    community_features,
    neighbour_risk,
    risk_columns,
)
from fraudgraph.features.tabular import NON_FEATURES, feature_columns, feature_matrix
from fraudgraph.graph.structural import STRUCTURAL_COLUMNS, pagerank, structural_features

PR_CFG = {"damping": 0.85, "max_iterations": 100, "tolerance": 1e-12}
LOUVAIN_CFG = {"seed": 0}


def test_pagerank_chain_by_hand():
    # a -> b -> c. PR(a) = 0.15; PR(b) = 0.15 + 0.85 * 0.15; PR(c) = 0.15 + 0.85 * PR(b).
    # c has no outgoing edge and passes its rank to nobody (GDS convention).
    pr = pagerank(3, np.array([0, 1]), np.array([1, 2]), **PR_CFG)
    assert pr == pytest.approx([0.15, 0.2775, 0.385875])


def test_pagerank_of_one_component_ignores_the_others():
    # Two disjoint components, like two time steps. Scores computed together must equal
    # scores computed separately: no rank leaks between steps.
    src, dst = np.array([0, 1, 2, 3, 4]), np.array([1, 2, 0, 4, 3])
    together = pagerank(5, src, dst, **PR_CFG)
    first = pagerank(3, src[:3], dst[:3], **PR_CFG)
    second = pagerank(2, src[3:] - 3, dst[3:] - 3, **PR_CFG)
    assert together == pytest.approx(np.r_[first, second])


@pytest.fixture
def triangle_plus_tail():
    # Triangle 1-2-3, plus 4 hanging off 1, plus 5 with no edges at all.
    nodes = pd.DataFrame({"tx_id": [1, 2, 3, 4, 5], "time_step": [1] * 5})
    edges = pd.DataFrame({"src": [1, 2, 3, 4], "dst": [2, 3, 1, 1]})
    return nodes, edges


def test_degrees_and_clustering_by_hand(triangle_plus_tail):
    nodes, edges = triangle_plus_tail
    s = structural_features(nodes, edges, PR_CFG, LOUVAIN_CFG).set_index("tx_id")
    assert list(s.columns) == STRUCTURAL_COLUMNS
    assert s["out_degree"].tolist() == [1, 1, 1, 1, 0]
    assert s["in_degree"].tolist() == [2, 1, 1, 0, 0]
    # Node 1 has neighbours 2, 3, 4: one of the three pairs (2-3) is connected.
    assert s["clustering"].tolist() == pytest.approx([1 / 3, 1, 1, 0, 0])


def test_neighbour_risk_by_hand(triangle_plus_tail):
    nodes, edges = triangle_plus_tail
    risk = np.array([0.9, 0.1, 0.2, 0.5, 0.7])
    nbr = neighbour_risk(nodes["tx_id"].to_numpy(), edges, risk)
    # Node 1's neighbours (either direction): 2, 3, 4.
    assert nbr.loc[1, "nbr_risk_mean"] == pytest.approx((0.1 + 0.2 + 0.5) / 3)
    assert nbr.loc[1, "nbr_risk_max"] == pytest.approx(0.5)
    assert nbr.loc[4, "nbr_risk_mean"] == pytest.approx(0.9)
    # A node with no neighbours has no neighbour risk, not zero risk.
    assert np.isnan(nbr.loc[5, "nbr_risk_mean"])


def test_community_density_by_hand(triangle_plus_tail):
    nodes, edges = triangle_plus_tail
    community = np.array([0, 0, 0, 0, 1])  # {1, 2, 3, 4} and {5}
    c = community_features(nodes["tx_id"].to_numpy(), community, edges)
    # 4 internal edges out of 4 * 3 / 2 = 6 possible pairs.
    assert c["community_size"].tolist() == [4, 4, 4, 4, 1]
    assert c["community_density"].iloc[0] == pytest.approx(4 / 6)
    assert np.isnan(c["community_density"].iloc[4])  # one node: no pairs


def test_no_community_spans_two_time_steps(data):
    s = structural_features(data.nodes, data.edges, PR_CFG, LOUVAIN_CFG)
    step = data.nodes.set_index("tx_id")["time_step"]
    steps_per_community = s.assign(step=s["tx_id"].map(step)).groupby("community_id")["step"]
    assert (steps_per_community.nunique() == 1).all()


@pytest.fixture
def with_graph(data):
    s = structural_features(data.nodes, data.edges, PR_CFG, LOUVAIN_CFG)
    rng = np.random.default_rng(0)
    risks = {base: rng.uniform(size=len(data.nodes)) for base in ("local", "all")}
    table = build_graph_features(data.nodes, data.edges, s, risks)
    return attach_graph_features(data, table), table


def test_table_aligned_and_complete(data, with_graph):
    _, table = with_graph
    assert table["tx_id"].tolist() == data.nodes["tx_id"].tolist()
    assert not table[STRUCTURE_FEATURES[:4]].isna().any().any()
    for base in ("local", "all"):
        assert set(risk_columns(base)) <= set(table.columns)


@pytest.mark.parametrize("fs", ["local", "all", "local+graph", "all+graph", "local+structure"])
def test_no_identifier_label_time_or_community_id_in_features(with_graph, fs):
    attached, _ = with_graph
    assert not NON_FEATURES & set(feature_columns(attached, fs))


def test_feature_set_contents(with_graph):
    attached, _ = with_graph
    local_graph = set(feature_columns(attached, "local+graph"))
    assert set(risk_columns("local")) <= local_graph
    assert not set(risk_columns("all")) & local_graph  # D-018: no aggregated-based risk
    assert not any(c.startswith("agg_") for c in local_graph)
    structure = set(feature_columns(attached, "local+structure"))
    assert not any("risk" in c for c in structure)  # D-021: shape only
    assert set(risk_columns("all")) <= set(feature_columns(attached, "all+graph"))


def test_matrix_shape(with_graph):
    attached, _ = with_graph
    X = feature_matrix(attached, "all+graph")
    assert X.shape == (len(attached.nodes), 93 + 72 + len(STRUCTURE_FEATURES) + 3)
