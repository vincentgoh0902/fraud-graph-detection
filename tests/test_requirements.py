"""dashboard/requirements.txt (Streamlit Community Cloud) matches uv.lock (D-042)."""

import shutil
import subprocess

import pytest

from fraudgraph.config import REPO_ROOT


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_cloud_requirements_match_the_lock_file():
    out = subprocess.run(
        ["uv", "export", "--no-default-groups", "--group", "dashboard", "--no-hashes",
         "--no-emit-project", "--no-header", "--format", "requirements-txt", "--quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    committed = (REPO_ROOT / "dashboard" / "requirements.txt").read_text()
    lines = committed.splitlines(keepends=True)
    body = "".join(line for line in lines if not line.startswith("# "))
    assert body == out, "run scripts/export_requirements.sh"
