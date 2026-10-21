"""The dashboard renders every view without errors, in-process (the Space's mode)."""

import sys

import pytest

from fraudgraph.config import REPO_ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "dashboard"))
import views  # noqa: E402

APP = str(REPO_ROOT / "dashboard" / "app.py")


def test_risk_band_edges():
    assert [views.risk_band(s) for s in (0.0, 0.19, 0.2, 0.79, 0.8, 1.0)] == [0, 0, 1, 3, 4, 4]


def test_network_html_marks_focal_and_flagged(service):
    nb = service.neighbourhood(1, 1)
    html = views.network_html(nb)
    assert '"shape": "star"' in html and '"shape": "diamond"' in html


@pytest.fixture
def app(serving_dir, monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("ARTIFACTS_DIR", str(serving_dir))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_start_page(app):
    assert not app.exception
    assert app.title[0].value.startswith("Investigating")


@pytest.mark.parametrize("page", ["Alert queue", "Transaction", "Patterns"])
def test_every_view_renders(app, page):
    app.session_state["step"] = 35
    app.session_state["tx"] = 1
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, app.exception
