"""Loading the demo subset from a Hugging Face dataset repo (D-042), without the network."""

import pytest

from fraudgraph.serving.artifacts import FILES, load_artifacts
from fraudgraph.serving.hub import download_demo


def test_download_demo_asks_for_the_dataset_files_at_a_revision(serving_dir, monkeypatch):
    seen = {}

    def fake_snapshot(**kwargs):
        seen.update(kwargs)
        return str(serving_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    path = download_demo("someone/demo", revision="abc123")
    assert seen["repo_type"] == "dataset" and seen["revision"] == "abc123"
    assert sorted(seen["allow_patterns"]) == sorted(FILES.values())
    assert load_artifacts(path).meta["model_version"] == "xgboost-test-seed0"


def test_download_demo_refuses_an_incomplete_repo(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text("{}")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: str(tmp_path))
    with pytest.raises(FileNotFoundError):
        download_demo("someone/demo")
