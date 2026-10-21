"""Write the served predictions into Neo4j as node properties (brief section 5).

Usage (after scripts/load_neo4j.py and scripts/export_serving.py):
    docker compose up -d neo4j
    uv run python scripts/write_predictions.py

Reads artifacts/serving/, so Neo4j holds exactly what the API serves (D-038).
"""

import sys
import time

from fraudgraph.config import load_config, resolve
from fraudgraph.graph.neo4j_client import connect, write_predictions
from fraudgraph.serving.artifacts import load_artifacts


def main() -> int:
    neo = load_config("graph")["neo4j"]
    data = load_artifacts(resolve(load_config("serving")["output_dir"]))
    start = time.perf_counter()
    with connect() as driver:
        write_predictions(
            driver,
            data.transactions.reset_index(),
            data.meta["model_version"],
            neo["database"],
            neo["batch_size"],
        )
    print(
        f"Wrote {len(data.transactions)} predictions ({data.meta['model_version']}) "
        f"in {time.perf_counter() - start:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
