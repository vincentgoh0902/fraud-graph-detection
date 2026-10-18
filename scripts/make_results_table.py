"""Write the results tables into README.md from the logged MLflow runs.

Usage:
    uv run python scripts/make_results_table.py

Replaces the text between <!-- results:start --> and <!-- results:end -->. Every
number comes from MLflow; nothing in that block is edited by hand.
"""

import sys

import mlflow

from fraudgraph.config import REPO_ROOT, load_config
from fraudgraph.evaluation.report import replace_between_markers, results_tables
from fraudgraph.tracking import setup_tracking


def main() -> int:
    cfg = load_config("baselines")
    setup_tracking(cfg["experiment"])
    runs = mlflow.search_runs(experiment_names=[cfg["experiment"]])
    if runs.empty or "tags.role" not in runs:
        print("No runs logged yet; run scripts/run_experiment.py first.", file=sys.stderr)
        return 1

    block = results_tables(runs, k=cfg["evaluation"]["k"])
    readme = REPO_ROOT / "README.md"
    readme.write_text(replace_between_markers(readme.read_text(), block, "results"))
    print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
