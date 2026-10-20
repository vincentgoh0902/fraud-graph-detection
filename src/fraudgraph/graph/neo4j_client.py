"""Neo4j: load the transaction graph and compute structural features with GDS.

Only tx_id, time_step and edges are loaded. No labels go into Neo4j, so no graph
algorithm can see one (D-019).
"""

import os

import pandas as pd
from neo4j import Driver, GraphDatabase, Result

from fraudgraph.config import REPO_ROOT

DIRECTED, UNDIRECTED = "fg_directed", "fg_undirected"


def _env(name: str) -> str:
    """Read a setting from the environment, falling back to the repo's .env file."""
    if name in os.environ:
        return os.environ[name]
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                return value.strip()
    raise KeyError(f"{name} is not set (see .env.example)")


def connect() -> Driver:
    driver = GraphDatabase.driver(
        _env("NEO4J_URI"), auth=(_env("NEO4J_USER"), _env("NEO4J_PASSWORD"))
    )
    driver.verify_connectivity()
    return driver


def _batches(rows: list[dict], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def load_graph(
    driver: Driver, nodes: pd.DataFrame, edges: pd.DataFrame, database: str, batch_size: int
) -> None:
    """Replace whatever is in the database with the Elliptic graph, then check counts."""
    with driver.session(database=database) as session:
        # IN TRANSACTIONS deletes in chunks, so 200k nodes never sit in one transaction.
        # It needs an auto-commit query, which session.run is.
        session.run(
            "MATCH (t:Transaction) CALL (t) { DETACH DELETE t } IN TRANSACTIONS OF 10000 ROWS"
        ).consume()
        # The constraint also creates the index the edge load uses to find endpoints.
        session.run(
            "CREATE CONSTRAINT tx_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE"
        ).consume()

        # int() turns numpy integers into Python ones, which the driver can send.
        node_rows = [
            {"tx_id": int(t), "time_step": int(s)}
            for t, s in zip(nodes["tx_id"], nodes["time_step"], strict=True)
        ]
        for batch in _batches(node_rows, batch_size):
            session.run(
                "UNWIND $rows AS r CREATE (:Transaction {tx_id: r.tx_id, time_step: r.time_step})",
                rows=batch,
            ).consume()

        edge_rows = [
            {"src": int(a), "dst": int(b)} for a, b in zip(edges["src"], edges["dst"], strict=True)
        ]
        for batch in _batches(edge_rows, batch_size):
            session.run(
                "UNWIND $rows AS r "
                "MATCH (a:Transaction {tx_id: r.src}) MATCH (b:Transaction {tx_id: r.dst}) "
                "CREATE (a)-[:PAYS]->(b)",
                rows=batch,
            ).consume()

        n_nodes = session.run("MATCH (t:Transaction) RETURN count(t) AS n").single()["n"]
        n_edges = session.run("MATCH ()-[p:PAYS]->() RETURN count(p) AS n").single()["n"]
    if (n_nodes, n_edges) != (len(nodes), len(edges)):
        raise RuntimeError(
            f"Neo4j has {n_nodes} nodes / {n_edges} edges, expected {len(nodes)} / {len(edges)}"
        )


def _stream(driver: Driver, database: str, call: str, value: str, params: dict) -> pd.DataFrame:
    """Run a GDS stream procedure and return (tx_id, value) for every node."""
    query = f"{call} YIELD nodeId, {value} RETURN gds.util.asNode(nodeId).tx_id AS tx_id, {value}"
    return driver.execute_query(query, params, database_=database, result_transformer_=Result.to_df)


def structural_gds(
    driver: Driver, database: str, pagerank_cfg: dict, louvain_cfg: dict
) -> pd.DataFrame:
    """Same columns as graph.structural.structural_features, computed by GDS."""

    def drop_projections():
        for name in (DIRECTED, UNDIRECTED):
            # failIfMissing = false: dropping a projection that isn't there is fine.
            # YIELD only graphName: the default return includes a deprecated field.
            driver.execute_query(
                "CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName",
                g=name,
                database_=database,
            )

    drop_projections()
    # readConcurrency 1: a parallel load assigns internal node ids in a different order
    # each time, and Louvain's result depends on that order (D-020).
    driver.execute_query(
        "CALL gds.graph.project($g, 'Transaction', {PAYS: {orientation: 'NATURAL'}}, "
        "{readConcurrency: 1})",
        g=DIRECTED,
        database_=database,
    )
    # SINGLE keeps one relationship per node pair, so a pair is never counted twice.
    driver.execute_query(
        "CALL gds.graph.project($g, 'Transaction', "
        "{PAYS: {orientation: 'UNDIRECTED', aggregation: 'SINGLE'}}, {readConcurrency: 1})",
        g=UNDIRECTED,
        database_=database,
    )
    try:
        out_deg = _stream(
            driver,
            database,
            "CALL gds.degree.stream($g, {orientation: 'NATURAL'})",
            "score",
            {"g": DIRECTED},
        ).rename(columns={"score": "out_degree"})
        in_deg = _stream(
            driver,
            database,
            "CALL gds.degree.stream($g, {orientation: 'REVERSE'})",
            "score",
            {"g": DIRECTED},
        ).rename(columns={"score": "in_degree"})
        pr = _stream(
            driver,
            database,
            "CALL gds.pageRank.stream($g, "
            "{dampingFactor: $d, maxIterations: $it, tolerance: $tol})",
            "score",
            {
                "g": DIRECTED,
                "d": pagerank_cfg["damping"],
                "it": pagerank_cfg["max_iterations"],
                "tol": pagerank_cfg["tolerance"],
            },
        ).rename(columns={"score": "pagerank"})
        cc = _stream(
            driver,
            database,
            "CALL gds.localClusteringCoefficient.stream($g)",
            "localClusteringCoefficient",
            {"g": UNDIRECTED},
        ).rename(columns={"localClusteringCoefficient": "clustering"})
        louvain = _stream(
            driver,
            database,
            "CALL gds.louvain.stream($g, {concurrency: $c})",  # single-threaded (D-020)
            "communityId",
            {"g": UNDIRECTED, "c": louvain_cfg["concurrency"]},
        ).rename(columns={"communityId": "community_id"})
    finally:
        drop_projections()

    out = in_deg
    for part in (out_deg, pr, cc, louvain):
        out = out.merge(part, on="tx_id", validate="one_to_one")
    for col in ("in_degree", "out_degree", "community_id"):
        out[col] = out[col].astype("int64")
    return out
