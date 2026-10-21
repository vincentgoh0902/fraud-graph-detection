"""The page formats numbers exactly as Python does (site/format.js, D-044)."""

import json
import random
import shutil
import subprocess

import pytest

from fraudgraph.config import REPO_ROOT

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

rng = random.Random(0)
EDGE_CASES = [0.625, 0.125, -0.625, 2.5, 3.5, 0.03125, -0.03125, 0.0, -0.0, 1e-5,
              0.949999988079071, 0.9998513460159302, 0.7968711853027344, 1.17, -0.31,
              12345.6789, 0.5, 1.5]  # fmt: skip
# Random values, plus multiples of 1/64: exact binary fractions, many of them exact ties.
VALUES = (
    EDGE_CASES
    + [rng.uniform(-5, 5) for _ in range(300)]
    + [rng.randint(0, 64) / 64 for _ in range(100)]
)


def format_in_node(function: str, places: int) -> list[str]:
    return run_node(
        f"console.log(JSON.stringify({json.dumps(VALUES)}.map(v => {function}(v, {places}))))"
    )


def run_node(code: str) -> list[str]:
    module = (REPO_ROOT / "site" / "format.js").as_uri()
    script = f"import {{ fixed, signed }} from '{module}';\n{code}"
    out = subprocess.run(["node", "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True)  # fmt: skip
    return json.loads(out.stdout)


@pytest.mark.parametrize("places", [0, 2, 3, 4])
def test_fixed_matches_python(places):
    got = format_in_node("fixed", places)
    # Python's json.dumps writes -0.0, which node parses back to -0, so the sign survives.
    assert got == [format(v, f".{places}f") for v in VALUES]


@pytest.mark.parametrize("places", [2, 3])
def test_signed_matches_python(places):
    got = format_in_node("signed", places)
    assert got == [format(v, f"+.{places}f") for v in VALUES]
