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


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root, so scripts work from any cwd."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
