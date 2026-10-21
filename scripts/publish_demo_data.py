"""Publish the demo subset to a public Hugging Face dataset repo (D-042).

Usage (after scripts/export_serving.py, logged in with `hf auth login`):
    uv run python scripts/publish_demo_data.py --repo <user>/fraud-graph-detection-demo --dry-run
    uv run python scripts/publish_demo_data.py --repo <user>/fraud-graph-detection-demo

Assembles build/demo-dataset/ (the demo files from artifacts/space plus the dataset card
hub/README.md with the licence and attribution) and uploads it. Prints the commit hash to
pin as DEMO_DATASET_REVISION in the Streamlit app's secrets.
"""

import argparse
import shutil
import sys

from fraudgraph.config import REPO_ROOT, load_config, resolve
from fraudgraph.serving.artifacts import FILES, load_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="<user>/<name> on Hugging Face")
    parser.add_argument("--dry-run", action="store_true", help="assemble only, upload nothing")
    args = parser.parse_args()

    src = resolve(load_config("serving")["demo"]["output_dir"])
    data = load_artifacts(src)  # fails early if a file is missing or unreadable
    meta = data.meta
    # The public subset must be the demo export: no labels, partial graph (D-012, D-040).
    if not meta["demo"] or meta["labels_included"] or "label" in data.transactions.columns:
        raise SystemExit(f"{src} is not a label-free demo export (D-040)")

    out = REPO_ROOT / "build" / "demo-dataset"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in FILES.values():
        shutil.copy2(src / name, out / name)
    shutil.copy2(REPO_ROOT / "hub" / "README.md", out / "README.md")
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"Assembled {out} ({size:.1f} MB, model {meta['model_version']})")
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=out,
        commit_message=f"Demo subset for {meta['model_version']}",
    )
    print(f"Uploaded: https://huggingface.co/datasets/{args.repo}")
    print(f'Pin this revision in the app\'s secrets: DEMO_DATASET_REVISION = "{commit.oid}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
