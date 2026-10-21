"""The serving code must not load the training stack (D-041).

Checked in a fresh process: in this one, other tests have already imported XGBoost.
"""

import subprocess
import sys

TRAINING = ["xgboost", "sklearn", "mlflow", "torch", "matplotlib", "scipy"]


def test_serving_imports_no_training_library():
    code = (
        "import sys\n"
        "import fraudgraph.api.app, fraudgraph.serving.client, fraudgraph.serving.graph\n"
        f"print([m for m in {TRAINING!r} if m in sys.modules])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
