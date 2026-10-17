"""Download the Elliptic dataset from Kaggle and verify its checksums.

Usage:
    uv run python scripts/download_data.py                     # download + verify
    uv run python scripts/download_data.py --record-checksums  # first run only

Needs Kaggle credentials (~/.kaggle/kaggle.json or ~/.kaggle/access_token).
The data is CC BY-NC-ND 4.0, so it is downloaded locally and never committed.
"""

import argparse
import hashlib
import subprocess
import sys

import yaml

from fraudgraph.config import CONFIG_DIR, load_config, resolve


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in 1 MB chunks so the 690 MB features file never sits in memory at once.
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-checksums",
        action="store_true",
        help="write the downloaded files' checksums to configs/data_checksums.yaml",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    cfg = load_config("data")
    raw_dir = resolve(cfg["raw_dir"])
    paths = {name: raw_dir / rel for name, rel in cfg["files"].items()}

    if args.force or not all(p.exists() for p in paths.values()):
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {cfg['kaggle_dataset']} to {raw_dir} ...")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                cfg["kaggle_dataset"],
                "-p",
                str(raw_dir),
                "--unzip",
            ],
            check=True,
        )
    else:
        print("Files already present; skipping download (use --force to re-download).")

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print(f"Missing after download: {missing}", file=sys.stderr)
        return 1

    actual = {name: sha256(p) for name, p in paths.items()}

    checksum_file = CONFIG_DIR / "data_checksums.yaml"
    if args.record_checksums:
        # A separate file, so rewriting it never strips the comments in data.yaml.
        with open(checksum_file, "w") as f:
            yaml.safe_dump(actual, f)
        print(f"Recorded checksums in {checksum_file.name}. Commit that file.")
        return 0

    expected = yaml.safe_load(checksum_file.read_text()) if checksum_file.exists() else {}
    if not expected:
        print("No checksums recorded yet; run once with --record-checksums.", file=sys.stderr)
        return 1
    bad = [name for name in paths if expected.get(name) != actual[name]]
    if bad:
        print(f"Checksum mismatch for {bad}: the files differ from the recorded version.")
        return 1
    print("All checksums match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
