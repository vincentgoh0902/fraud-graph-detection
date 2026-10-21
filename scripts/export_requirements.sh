#!/bin/sh
# Pinned requirements for Streamlit Community Cloud, generated from uv.lock (D-042).
# Cloud reads dashboard/requirements.txt (next to the entrypoint) before any root file.
set -e
cd "$(dirname "$0")/.."
{
  echo "# Generated from uv.lock by scripts/export_requirements.sh; do not edit by hand."
  echo "# The dashboard group only: no XGBoost, no torch (D-041)."
  uv export --no-default-groups --group dashboard --no-hashes --no-emit-project \
    --no-header --format requirements-txt --quiet
} > dashboard/requirements.txt
