"""Structural graph features computed in Python (NetworkX + NumPy), no Neo4j needed.

Neo4j GDS computes the features used in experiments (neo4j_client.structural_gds);
this module computes the same ones so CI can test the pipeline without a database,
and so the integration test can check GDS against an independent implementation (D-019).

Output columns, one row per transaction:
    tx_id, in_degree, out_degree, pagerank   (directed graph: payments have a direction)
    clustering, community_id                 (undirected graph)
community_id is an arbitrary label, never a feature (D-002); it is only used to group
nodes for community-level features.
"""

import networkx as nx
import numpy as np
import pandas as pd

STRUCTURAL_COLUMNS = ["in_degree", "out_degree", "pagerank", "clustering", "community_id"]


def pagerank(
    n: int, src: np.ndarray, dst: np.ndarray, damping: float, max_iterations: int, tolerance: float
) -> np.ndarray:
    """PageRank in the same unnormalised form as Neo4j GDS (D-019).

    PR(v) = (1 - d) + d * sum over in-neighbours u of PR(u) / outdeg(u)

    Nodes without outgoing edges pass their rank to nobody. NetworkX instead spreads it
    over every node in the graph, which would let one time step's scores depend on
    another's. Here each step's scores depend only on that step.
    src and dst are integer node positions 0..n-1.
    """
    out_degree = np.bincount(src, minlength=n)
    rank = np.full(n, 1 - damping)
    # Safe divide: nodes with no outgoing edges give nothing.
    inv_out = np.divide(1.0, out_degree, out=np.zeros(n), where=out_degree > 0)
    for _ in range(max_iterations):
        new = (1 - damping) + damping * np.bincount(dst, weights=(rank * inv_out)[src], minlength=n)
        # GDS stops when every node changes by less than the tolerance.
        done = np.abs(new - rank).max() < tolerance
        rank = new
        if done:
            break
    return rank


def structural_features(
    nodes: pd.DataFrame, edges: pd.DataFrame, pagerank_cfg: dict, louvain_cfg: dict
) -> pd.DataFrame:
    tx_ids = nodes["tx_id"].to_numpy()
    pos = pd.Series(np.arange(len(tx_ids)), index=tx_ids)
    src, dst = pos[edges["src"]].to_numpy(), pos[edges["dst"]].to_numpy()
    n = len(tx_ids)

    out = pd.DataFrame({"tx_id": tx_ids})
    out["in_degree"] = np.bincount(dst, minlength=n)
    out["out_degree"] = np.bincount(src, minlength=n)
    out["pagerank"] = pagerank(
        n,
        src,
        dst,
        pagerank_cfg["damping"],
        pagerank_cfg["max_iterations"],
        pagerank_cfg["tolerance"],
    )

    # Undirected, one edge per pair: matches the GDS projection (UNDIRECTED, SINGLE).
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(zip(src.tolist(), dst.tolist(), strict=True))
    g.remove_edges_from(nx.selfloop_edges(g))
    clustering = nx.clustering(g)
    out["clustering"] = [clustering[i] for i in range(n)]

    community = np.empty(n, dtype=np.int64)
    for cid, members in enumerate(nx.community.louvain_communities(g, seed=louvain_cfg["seed"])):
        community[list(members)] = cid
    out["community_id"] = community
    return out
