"""Graph feature table: structure, community and neighbour-risk features per transaction.

Inputs are the structural columns (from GDS or graph.structural) and stage-1 predicted
risk for each comparison base (D-001, D-018). Output: one row per transaction, saved to
data/processed/graph_features.parquet and joined onto the nodes for experiments.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from fraudgraph.data.load import EllipticData

# Pure graph shape: no stage-1 predictions involved (D-021).
STRUCTURE_FEATURES = [
    "in_degree",
    "out_degree",
    "pagerank",
    "clustering",
    "community_size",
    "community_density",
]
# Built from stage-1 predicted risk; one copy per base feature set (D-018).
RISK_FEATURES = ["community_risk_mean", "nbr_risk_mean", "nbr_risk_max"]


def risk_columns(base: str) -> list[str]:
    return [f"{c}_{base}" for c in RISK_FEATURES]


def undirected_pairs(edges: pd.DataFrame) -> pd.DataFrame:
    """Each edge in both directions, once per ordered pair, without self-loops.

    Risk spreads both ways: a transaction paying a risky one and one paid by a risky one
    are both suspicious, so neighbours are taken regardless of direction.
    """
    both = pd.concat(
        [edges[["src", "dst"]], edges.rename(columns={"src": "dst", "dst": "src"})[["src", "dst"]]]
    )
    both = both[both["src"] != both["dst"]]
    return both.drop_duplicates(ignore_index=True)


def neighbour_risk(tx_ids: np.ndarray, edges: pd.DataFrame, risk: np.ndarray) -> pd.DataFrame:
    """Mean and max stage-1 risk over each node's neighbours. NaN for a node with none."""
    risk_by_tx = pd.Series(risk, index=tx_ids)
    pairs = undirected_pairs(edges)
    pairs["nbr_risk"] = pairs["dst"].map(risk_by_tx).to_numpy()
    agg = pairs.groupby("src")["nbr_risk"].agg(nbr_risk_mean="mean", nbr_risk_max="max")
    return agg.reindex(tx_ids)


def community_features(
    tx_ids: np.ndarray, community: np.ndarray, edges: pd.DataFrame
) -> pd.DataFrame:
    """Size and internal edge density of each node's community (D-002).

    Density = edges inside the community / possible pairs, size * (size - 1) / 2.
    NaN for a single-node community, which has no possible pairs.
    """
    comm = pd.Series(community, index=tx_ids)
    size = comm.value_counts()
    pairs = undirected_pairs(edges)
    pairs = pairs[pairs["src"] < pairs["dst"]]  # each undirected edge once
    cs, cd = pairs["src"].map(comm), pairs["dst"].map(comm)
    internal = cs[cs.to_numpy() == cd.to_numpy()].value_counts().reindex(size.index, fill_value=0)
    possible = size * (size - 1) / 2
    density = (internal / possible.where(possible > 0)).astype(float)
    return pd.DataFrame(
        {
            "community_size": comm.map(size).to_numpy(),
            "community_density": comm.map(density).to_numpy(),
        },
        index=tx_ids,
    )


def build_graph_features(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    structural: pd.DataFrame,
    risks: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Assemble the feature table, rows aligned with `nodes`.

    risks maps a base feature set name ('local', 'all') to stage-1 risk per node.
    community_id is kept in the table (Phase 6 queries suspicious communities) but is on
    the never-a-feature list in features.tabular (D-002).
    """
    tx_ids = nodes["tx_id"].to_numpy()
    s = structural.set_index("tx_id").reindex(tx_ids)
    if s.isna().any().any():
        raise ValueError("Structural features are missing for some transactions")

    out = pd.DataFrame({"tx_id": tx_ids})
    for col in ("in_degree", "out_degree", "pagerank", "clustering", "community_id"):
        out[col] = s[col].to_numpy()
    comm = community_features(tx_ids, out["community_id"].to_numpy(), edges)
    out["community_size"] = comm["community_size"].to_numpy()
    out["community_density"] = comm["community_density"].to_numpy()

    for base, risk in risks.items():
        community_risk = pd.Series(risk).groupby(out["community_id"]).transform("mean")
        nbr = neighbour_risk(tx_ids, edges, risk)
        out[f"community_risk_mean_{base}"] = community_risk.to_numpy()
        out[f"nbr_risk_mean_{base}"] = nbr["nbr_risk_mean"].to_numpy()
        out[f"nbr_risk_max_{base}"] = nbr["nbr_risk_max"].to_numpy()
    return out


def attach_graph_features(data: EllipticData, table: pd.DataFrame) -> EllipticData:
    """Join the graph feature table onto the nodes, keeping row order."""
    extra = table.drop(columns=[c for c in ("time_step", "label") if c in table])
    nodes = data.nodes.merge(extra, on="tx_id", how="left", validate="one_to_one")
    if len(nodes) != len(data.nodes) or nodes["in_degree"].isna().any():
        raise ValueError("Graph feature table does not cover every transaction")
    return replace(data, nodes=nodes)
