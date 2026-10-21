"""The build-time leak check (D-012, D-040, D-044): no labels, no features, no stray edges."""

import copy

import pandas as pd
import pytest

from fraudgraph.site.build import build_site_data
from fraudgraph.site.schema import SiteCheckError, check_site

from .conftest import SERVING_EDGES


@pytest.fixture
def site(service):
    service.data.transactions = service.data.transactions.drop(columns="label")
    service.data.meta.update(demo=True, has_full_graph=False, labels_included=False,
                             max_hops=2, neighbourhood_roots=[8, 102])  # fmt: skip
    return build_site_data(service, dataset={"repo": "r", "revision": "v"}, alerts_k=5,
                           shards_per_step=3, layout_seed=0)  # fmt: skip


EDGES = pd.DataFrame(SERVING_EDGES, columns=["src", "dst"])


def test_clean_build_passes(site):
    check_site(site, EDGES)


def test_a_label_value_fails(site):
    bad = copy.deepcopy(site)
    bad["data/steps/35.json"]["alerts"][0]["label"] = "illicit"
    with pytest.raises(SiteCheckError, match="label"):
        check_site(bad, EDGES)


def test_a_feature_column_fails(site):
    bad = copy.deepcopy(site)
    bad["data/shap/35-0.json"]["transactions"][0]["local_1"] = 0.3
    with pytest.raises(SiteCheckError, match="local_1"):
        check_site(bad, EDGES)


def test_an_unknown_key_fails(site):
    bad = copy.deepcopy(site)
    bad["data/index.json"]["meta"]["note"] = "x"
    with pytest.raises(SiteCheckError, match="note"):
        check_site(bad, EDGES)


def test_an_edge_outside_the_shipped_subset_fails(site):
    bad = copy.deepcopy(site)
    bad["data/steps/35.json"]["neighbourhoods"]["8"]["2"]["edges"].append({"src": 4, "dst": 5})
    with pytest.raises(SiteCheckError, match="edge"):
        check_site(bad, EDGES[~((EDGES["src"] == 4) & (EDGES["dst"] == 5))])


def test_an_unexpected_file_fails(site):
    bad = dict(site, **{"data/features.json": {}})
    with pytest.raises(SiteCheckError, match="features.json"):
        check_site(bad, EDGES)
