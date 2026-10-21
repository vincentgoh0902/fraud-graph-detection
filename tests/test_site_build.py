"""Static site data (Phase 7, D-044): what the build writes and how it is split up."""

import pytest

from fraudgraph.site.build import build_site_data, expand_transaction, layout_positions

K, SHARDS = 5, 3


@pytest.fixture
def demo_service(service):
    """The tiny serving fixture, shaped like the public demo export (D-040)."""
    service.data.transactions = service.data.transactions.drop(columns="label")
    service.data.meta.update(
        demo=True, has_full_graph=False, labels_included=False, max_hops=2,
        neighbourhood_roots=[8, 102],
    )  # fmt: skip
    return service


@pytest.fixture
def site(demo_service):
    return build_site_data(
        demo_service, dataset={"repo": "someone/demo", "revision": "abc"},
        alerts_k=K, shards_per_step=SHARDS, layout_seed=0,
    )  # fmt: skip


def test_files_written(site):
    assert set(site) == {
        "data/index.json", "data/lookup.json", "data/steps/35.json", "data/steps/36.json",
        *(f"data/shap/{t}-{s}.json" for t in (35, 36) for s in range(SHARDS)),
    }  # fmt: skip


def test_index_holds_what_the_first_screen_needs(site):
    index = site["data/index.json"]
    assert index["meta"]["dataset_revision"] == "abc"
    assert [s["time_step"] for s in index["steps"]] == [35, 36]
    # The tour's steps 40 and 43 are kept inside the served range, as in the dashboard.
    assert index["tour"]["before"] == 36 and index["tour"]["after"] == 36
    assert index["tour"]["alert"]["tx_id"] == 102


def test_every_transaction_in_exactly_one_shard(site, demo_service):
    seen = {}
    for path, payload in site.items():
        if path.startswith("data/shap/"):
            step, shard = map(int, path.removeprefix("data/shap/").removesuffix(".json").split("-"))
            assert payload["time_step"] == step
            for entry in payload["transactions"]:
                assert entry["tx_id"] % SHARDS == shard
                assert entry["tx_id"] not in seen
                seen[entry["tx_id"]] = step
    assert seen == demo_service.data.transactions["time_step"].to_dict()


def test_lookup_maps_every_id_to_its_step(site, demo_service):
    lookup = site["data/lookup.json"]
    flat = {t: int(step) for step, ids in lookup.items() for t in ids}
    assert flat == demo_service.data.transactions["time_step"].to_dict()


def test_step_file_has_neighbourhoods_for_its_roots_only(site):
    step35 = site["data/steps/35.json"]
    assert set(step35["neighbourhoods"]) == {"8"}
    assert set(step35["neighbourhoods"]["8"]) == {"1", "2"}
    assert len(step35["alerts"]) == K


def test_expand_transaction_rebuilds_the_service_shape(site, demo_service):
    index = site["data/index.json"]
    shard = site[f"data/shap/35-{8 % SHARDS}.json"]
    entry = next(e for e in shard["transactions"] if e["tx_id"] == 8)
    assert expand_transaction(entry, shard, index["meta"]) == demo_service.transaction(8)


def test_layout_is_seeded_and_covers_every_node(site, demo_service):
    nb = demo_service.neighbourhood(8, 2)
    a, b = layout_positions(nb, seed=0), layout_positions(nb, seed=0)
    assert a == b and len(a) == len(nb["nodes"])
    assert all(0 <= x <= 1000 and 0 <= y <= 1000 for x, y in a)
