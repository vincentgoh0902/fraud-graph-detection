"""Load the Elliptic graph into Neo4j (transactions and payment edges, no labels).

Usage:
    docker compose up -d neo4j
    uv run python scripts/load_neo4j.py

Replaces any Transaction nodes already in the database, then checks the counts.
"""

import sys
import time

from fraudgraph.config import load_config
from fraudgraph.data.load import load_elliptic
from fraudgraph.graph.neo4j_client import connect, load_graph


def main() -> int:
    cfg = load_config("graph")["neo4j"]
    data = load_elliptic()
    start = time.perf_counter()
    with connect() as driver:
        load_graph(driver, data.nodes, data.edges, cfg["database"], cfg["batch_size"])
    elapsed = time.perf_counter() - start
    print(f"Loaded {len(data.nodes)} transactions and {len(data.edges)} edges in {elapsed:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
