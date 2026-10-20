"""MLflow setup: a local SQLite store under mlruns/ (git-ignored)."""

import os
from typing import Any

import mlflow
import yaml

from fraudgraph.config import load_config, resolve


def setup_tracking(experiment: str) -> None:
    cfg = load_config("tracking")
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        db = resolve(cfg["db"])
        db.parent.mkdir(parents=True, exist_ok=True)
        # Absolute path, so runs land in the same store whatever the working directory.
        uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        artifacts = resolve(cfg["artifact_root"]) / experiment
        mlflow.create_experiment(experiment, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(experiment)


def winning_params(experiment: str, model: str, feature_set: str) -> dict[str, Any]:
    """Settings of the model the protocol selected, from the latest final runs in MLflow.

    MLflow stores params as strings. yaml.safe_load turns "600" back into 600 and "0.05"
    into 0.05; Python's None is stored as "None", which YAML would keep as a string.
    """
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=(
            f"tags.role = 'final' and tags.model = '{model}' and tags.feature_set = '{feature_set}'"
        ),
    )
    if runs.empty:
        raise LookupError(f"No final {model}/{feature_set} runs in experiment {experiment!r}")
    latest = runs[runs["tags.run_group"] == runs["tags.run_group"].max()].iloc[0]
    params = {
        c.removeprefix("params."): None if v == "None" else yaml.safe_load(v)
        for c, v in latest.items()
        if c.startswith("params.") and v is not None
    }
    params.pop("seed", None)  # the caller chooses the seed
    return params
