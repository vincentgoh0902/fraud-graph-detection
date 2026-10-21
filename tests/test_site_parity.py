"""The static site shows exactly what the dashboard shows (D-044).

Every generated file is compared with fresh ServingService output: on the tiny fixture
always, and on the real demo subset when SITE_DATA_DIR points at it (CI downloads the
pinned revision; locally artifacts/space works too).
"""

import json
import os
from pathlib import Path

import pytest

from fraudgraph.config import load_config
from fraudgraph.site.build import build_site_data, expand_transaction


def _check_parity(service, site: dict, alerts_k: int) -> None:
    # JSON round trip: the browser sees exactly these values.
    site = json.loads(json.dumps(site))
    index = site["data/index.json"]
    meta = index["meta"]
    assert {k: meta[k] for k in service.meta()} == service.meta()
    assert index["steps"] == service.steps()["steps"]
    first, last = meta["steps"]
    for t in range(first, last + 1):
        step = site[f"data/steps/{t}.json"]
        shipped = step["alerts"]
        # The page shows the first k for any k <= alerts_k; that must equal alerts(t, k).
        for k in (1, 2, 5, 20, alerts_k):
            if k <= alerts_k:
                assert shipped[:k] == service.alerts(t, k)["alerts"], (t, k)
        assert step["communities"] == service.communities(t, 5, 10)
        for root, by_hops in step["neighbourhoods"].items():
            for hops, nb in by_hops.items():
                expected = service.neighbourhood(int(root), int(hops))
                assert {k: v for k, v in nb.items() if k != "positions"} == expected
    n = 0
    for path, shard in site.items():
        if path.startswith("data/shap/"):
            for entry in shard["transactions"]:
                assert expand_transaction(entry, shard, meta) == service.transaction(entry["tx_id"])
                n += 1
    assert n == len(service.data.transactions)


def test_parity_on_the_fixture(service):
    service.data.transactions = service.data.transactions.drop(columns="label")
    service.data.meta.update(demo=True, has_full_graph=False, labels_included=False,
                             max_hops=2, neighbourhood_roots=[8, 102])  # fmt: skip
    site = build_site_data(service, dataset={"repo": "r", "revision": "v"}, alerts_k=5,
                           shards_per_step=3, layout_seed=0)  # fmt: skip
    _check_parity(service, site, alerts_k=5)


REAL = os.environ.get("SITE_DATA_DIR")


@pytest.mark.skipif(not REAL or not Path(REAL, "meta.json").exists(),
                    reason="set SITE_DATA_DIR to the demo subset")  # fmt: skip
def test_parity_on_the_real_demo_subset():
    from fraudgraph.site.build import demo_service

    cfg = load_config("serving")
    service = demo_service(Path(REAL))
    site = build_site_data(service, dataset={"repo": "r", "revision": "v"},
                           alerts_k=cfg["site"]["alerts_k"],
                           shards_per_step=cfg["site"]["shards_per_step"],
                           layout_seed=cfg["site"]["layout_seed"])  # fmt: skip
    _check_parity(service, site, alerts_k=cfg["site"]["alerts_k"])
