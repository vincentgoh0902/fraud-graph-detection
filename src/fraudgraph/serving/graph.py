"""Investigation queries behind one interface, with two stores (D-008, D-040).

- Neo4jStore runs them as Cypher against the full graph (docker compose).
- InMemoryStore runs them with NetworkX and pandas on the exported files (the public demo, CI).

The rules that decide an answer (breadth-first hop distances, truncation order, path
sorting) live here once, outside both stores, so the two cannot disagree on them.
"""

from collections.abc import Callable, Iterable
from itertools import islice
from typing import Protocol

import networkx as nx
import pandas as pd

# Enumerating every shortest path can explode around a hub; both stores stop here.
PATH_ENUMERATION_CAP = 1000

Pairs = Iterable[tuple[int, int]]


def bfs_hops(neighbours: Callable[[list[int]], Pairs], start: int, hops: int) -> dict[int, int]:
    """Hop distance from start to every node within `hops`, breadth first.

    neighbours(frontier) returns (node, neighbour) pairs for the frontier. One call per hop,
    which for Neo4j means one query per hop instead of enumerating every path.
    """
    dist = {start: 0}
    frontier = [start]
    for hop in range(1, hops + 1):
        nxt = sorted({b for _, b in neighbours(frontier) if b not in dist})
        for b in nxt:
            dist[b] = hop
        if not nxt:
            break
        frontier = nxt
    return dist


def truncate(hops: dict[int, int], risk: pd.Series, max_nodes: int) -> tuple[list[int], bool]:
    """Keep at most max_nodes: nearer hops first, then higher risk, then lower tx_id."""
    order = sorted(hops, key=lambda t: (hops[t], -float(risk[t]), t))
    return order[:max_nodes], len(order) > max_nodes


def sort_paths(paths: Iterable[list[int]], max_paths: int) -> list[list[int]]:
    return sorted(list(p) for p in islice(paths, PATH_ENUMERATION_CAP))[:max_paths]


class GraphStore(Protocol):
    name: str

    def hop_distances(self, tx_id: int, hops: int) -> dict[int, int]: ...
    def edges_among(self, tx_ids: list[int]) -> pd.DataFrame: ...
    def shortest_paths(
        self, source: int, target: int, max_len: int, max_paths: int
    ) -> list[list[int]]: ...
    def fan_in(self, time_step: int) -> pd.DataFrame: ...
    def community_summary(self, time_step: int) -> pd.DataFrame: ...


class InMemoryStore:
    """Edges and served transactions in memory. Payments are directed; reach is not (D-026)."""

    name = "memory"

    def __init__(self, edges: pd.DataFrame, transactions: pd.DataFrame):
        self.edges = edges[["src", "dst"]].drop_duplicates(ignore_index=True)
        self.tx = transactions
        self.graph = nx.Graph()
        self.graph.add_nodes_from(transactions.index)
        self.graph.add_edges_from(zip(self.edges["src"], self.edges["dst"], strict=True))

    def _neighbour_pairs(self, frontier: list[int]) -> Pairs:
        return [(a, b) for a in frontier for b in self.graph.adj[a]]

    def hop_distances(self, tx_id: int, hops: int) -> dict[int, int]:
        return bfs_hops(self._neighbour_pairs, tx_id, hops)

    def edges_among(self, tx_ids: list[int]) -> pd.DataFrame:
        keep = set(tx_ids)
        e = self.edges
        return e[e["src"].isin(keep) & e["dst"].isin(keep)].reset_index(drop=True)

    def shortest_paths(
        self, source: int, target: int, max_len: int, max_paths: int
    ) -> list[list[int]]:
        g = self.graph
        if source not in g or target not in g or not nx.has_path(g, source, target):
            return []
        if nx.shortest_path_length(g, source, target) > max_len:
            return []
        return sort_paths(nx.all_shortest_paths(g, source, target), max_paths)

    def fan_in(self, time_step: int) -> pd.DataFrame:
        flagged = self.tx.index[
            (self.tx["time_step"] == time_step) & (self.tx["predicted_label"] == 1)
        ]
        e = self.edges[self.edges["src"].isin(flagged)]
        out = e.groupby("dst")["src"].nunique().rename("n_flagged_payers")
        return out.rename_axis("tx_id").reset_index().sort_values("tx_id", ignore_index=True)

    def community_summary(self, time_step: int) -> pd.DataFrame:
        t = self.tx[self.tx["time_step"] == time_step]
        g = t.groupby("community_id").agg(
            size=("risk_score", "size"),
            n_flagged=("predicted_label", "sum"),
            mean_risk=("risk_score", "mean"),
        )
        return g.reset_index()


class Neo4jStore:
    """The same queries in Cypher. Needs scripts/write_predictions.py to have run."""

    name = "neo4j"

    def __init__(self, driver, database: str = "neo4j"):
        self.driver, self.database = driver, database

    def _records(self, query: str, **params) -> list:
        return self.driver.execute_query(query, params, database_=self.database).records

    def _neighbour_pairs(self, frontier: list[int]) -> Pairs:
        rows = self._records(
            "MATCH (a:Transaction)-[:PAYS]-(b:Transaction) WHERE a.tx_id IN $ids "
            "RETURN a.tx_id AS a, b.tx_id AS b",
            ids=frontier,
        )
        return [(r["a"], r["b"]) for r in rows]

    def hop_distances(self, tx_id: int, hops: int) -> dict[int, int]:
        return bfs_hops(self._neighbour_pairs, tx_id, hops)

    def edges_among(self, tx_ids: list[int]) -> pd.DataFrame:
        rows = self._records(
            "MATCH (a:Transaction)-[:PAYS]->(b:Transaction) "
            "WHERE a.tx_id IN $ids AND b.tx_id IN $ids "
            "RETURN DISTINCT a.tx_id AS src, b.tx_id AS dst",
            ids=list(tx_ids),
        )
        return pd.DataFrame([dict(r) for r in rows], columns=["src", "dst"])

    def shortest_paths(
        self, source: int, target: int, max_len: int, max_paths: int
    ) -> list[list[int]]:
        # A variable-length bound cannot be a query parameter; int() keeps it a number.
        rows = self._records(
            "MATCH (a:Transaction {tx_id: $a}), (b:Transaction {tx_id: $b}) "
            f"MATCH p = allShortestPaths((a)-[:PAYS*..{int(max_len)}]-(b)) "
            "RETURN [n IN nodes(p) | n.tx_id] AS path LIMIT $cap",
            a=source,
            b=target,
            cap=PATH_ENUMERATION_CAP,
        )
        return sort_paths((r["path"] for r in rows), max_paths)

    def fan_in(self, time_step: int) -> pd.DataFrame:
        rows = self._records(
            "MATCH (f:Transaction {time_step: $s, predicted_label: 1})-[:PAYS]->(t:Transaction) "
            "RETURN t.tx_id AS tx_id, count(DISTINCT f) AS n_flagged_payers",
            s=time_step,
        )
        out = pd.DataFrame([dict(r) for r in rows], columns=["tx_id", "n_flagged_payers"])
        return out.sort_values("tx_id", ignore_index=True)

    def community_summary(self, time_step: int) -> pd.DataFrame:
        rows = self._records(
            "MATCH (t:Transaction {time_step: $s}) WHERE t.risk_score IS NOT NULL "
            "RETURN t.community_id AS community_id, count(*) AS size, "
            "sum(t.predicted_label) AS n_flagged, avg(t.risk_score) AS mean_risk",
            s=time_step,
        )
        cols = ["community_id", "size", "n_flagged", "mean_risk"]
        return pd.DataFrame([dict(r) for r in rows], columns=cols)
