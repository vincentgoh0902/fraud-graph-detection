"""The public demo's data, from a Hugging Face dataset repo (D-042).

The demo app (Streamlit Community Cloud) downloads the exported demo subset at start-up
instead of shipping it in git: the repo holds code, the dataset repo holds the data with
its licence and attribution.
"""

from pathlib import Path

from fraudgraph.serving.artifacts import FILES


def download_demo(repo_id: str, revision: str | None = None) -> Path:
    """Download the demo files (only those) and return the local folder.

    Pass a commit hash as revision so the app keeps serving the files it was checked
    against, even if the dataset repo changes later.
    """
    from huggingface_hub import snapshot_download  # only the dashboard needs it

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=list(FILES.values()),
    )
    missing = [f for f in FILES.values() if not (Path(path) / f).exists()]
    if missing:
        raise FileNotFoundError(f"{repo_id} is missing {missing}")
    return Path(path)
