"""Loading YAML configs from configs/."""

from pathlib import Path
from typing import Any

import yaml

# src/fraudgraph/config.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml as a dict."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def load_experiment_config(name: str) -> dict[str, Any]:
    """Load an experiment config, filling it in from its `base` config if it names one.

    Top-level keys override the base. `models` may be a list of model names, which picks
    those models (with their grids) from the base, so a later phase reuses a grid
    instead of copying it (D-014).
    """
    cfg = load_config(name)
    if "base" not in cfg:
        return cfg
    base = load_experiment_config(cfg["base"])
    merged = base | {k: v for k, v in cfg.items() if k != "base"}
    if isinstance(merged["models"], list):
        merged["models"] = {m: base["models"][m] for m in merged["models"]}
    return merged


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root, so scripts work from any cwd."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
