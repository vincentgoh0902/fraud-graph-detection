"""MLflow setup: a local SQLite store under mlruns/ (git-ignored)."""

import os

import mlflow

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
