"""Runs the torch tests (tests/torch/) in their own process.

torch and XGBoost cannot share a process on macOS: each loads its own OpenMP runtime and
the second one to run crashes (D-030). The main pytest process imports XGBoost, so the
torch tests are skipped there (tests/conftest.py) and run here instead, in a child
pytest process. A failure shows the child's full output.
"""

import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import TORCH_TESTS_ENV

TORCH_DIR = Path(__file__).parent / "torch"


def test_torch_suite_passes():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(TORCH_DIR), "-q", "-p", "no:cacheprovider"],
        env=os.environ | {TORCH_TESTS_ENV: "1"},
        capture_output=True,
        text=True,
        cwd=TORCH_DIR.parents[1],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert " passed" in proc.stdout  # the suite really ran (not "no tests collected")
