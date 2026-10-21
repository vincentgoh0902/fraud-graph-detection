"""Assemble the Hugging Face Space folder (D-008, D-040).

Usage (after scripts/export_serving.py):
    uv run python scripts/build_space.py
    docker build -t fraud-graph-space build/space
    docker run --rm -p 127.0.0.1:7860:7860 fraud-graph-space

build/space/ is exactly what would be pushed to the Space: the dashboard, the serving code
and the demo subset in artifacts/space. Publishing is a separate, manual step.
"""

import shutil
import sys

from fraudgraph.config import REPO_ROOT, load_config, resolve

COPY = ["pyproject.toml", "uv.lock", "src", "dashboard", ".streamlit"]


def main() -> int:
    out = REPO_ROOT / "build" / "space"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in COPY:
        src = REPO_ROOT / name
        if src.is_dir():
            shutil.copytree(src, out / name, ignore=ignore)
        else:
            shutil.copy2(src, out / name)
    (out / "configs").mkdir()
    for cfg in ("serving.yaml", "graph.yaml"):
        shutil.copy2(REPO_ROOT / "configs" / cfg, out / "configs" / cfg)
    demo = resolve(load_config("serving")["demo"]["output_dir"])
    if not (demo / "meta.json").exists():
        raise SystemExit(f"{demo} is missing: run scripts/export_serving.py first")
    shutil.copytree(demo, out / "artifacts" / "space")
    shutil.copy2(REPO_ROOT / "space" / "Dockerfile", out / "Dockerfile")
    shutil.copy2(REPO_ROOT / "space" / "README.md", out / "README.md")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Space folder: {out} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
