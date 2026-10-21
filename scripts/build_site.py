"""Build the static demo site into build/site/ (Phase 7, D-043 to D-045).

Usage:
    uv run python scripts/build_site.py                      # download the pinned revision
    uv run python scripts/build_site.py --data artifacts/space   # a local demo export

Downloads the demo subset at the revision pinned in configs/serving.yaml, generates the
JSON with the serving layer (the dashboard's own code), checks it (no labels, no features,
no edge outside the subset), and copies the static front end from site/. Preview with:
    python -m http.server -d build/site 8000
"""

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

from fraudgraph.config import REPO_ROOT, load_config, resolve
from fraudgraph.site.build import build_site_data, demo_service
from fraudgraph.site.schema import check_site

FIRST_LOAD = ["index.html", "style.css", "app.js", "charts.js", "data.js", "format.js",
              "data/index.json"]  # fmt: skip


def compact(payload) -> bytes:
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="a local demo export instead of the pinned download")
    args = parser.parse_args()
    cfg = load_config("serving")
    demo, site_cfg = cfg["demo"], cfg["site"]

    if args.data:
        folder = Path(args.data)
    else:
        from fraudgraph.serving.hub import download_demo

        folder = download_demo(demo["dataset_repo"], demo["dataset_revision"])
    service = demo_service(folder)
    payloads = build_site_data(
        service,
        dataset={"repo": demo["dataset_repo"], "revision": demo["dataset_revision"]},
        alerts_k=site_cfg["alerts_k"],
        shards_per_step=site_cfg["shards_per_step"],
        layout_seed=site_cfg["layout_seed"],
    )
    check_site(payloads, service.data.edges)  # fails the build on any leak (D-044)
    print(f"Checked {len(payloads)} data files: no labels, no features, no stray edges")

    out = resolve(site_cfg["output_dir"])
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(REPO_ROOT / "site", out)
    (out / ".nojekyll").touch()  # serve files as they are, no Jekyll processing
    for path, payload in payloads.items():
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(compact(payload))

    raw = sum((out / f).stat().st_size for f in FIRST_LOAD)
    gz = sum(len(gzip.compress((out / f).read_bytes())) for f in FIRST_LOAD)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Wrote {out}: {total / 1e6:.1f} MB in all")
    print(f"First load: {raw / 1e3:.0f} KB, {gz / 1e3:.0f} KB gzipped ({', '.join(FIRST_LOAD)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
