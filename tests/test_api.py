"""The API contract (brief section 6), on the in-memory store and a tiny export."""

import pytest
from fastapi.testclient import TestClient

from fraudgraph.api.app import create_app
from fraudgraph.serving.client import ApiClient, LocalClient


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_version"] == "xgboost-test-seed0"


def test_transaction_detail(client):
    body = client.get("/transactions/8").json()
    assert body["time_step"] == 35
    assert body["risk_score"] == pytest.approx(0.95)
    assert body["flagged"] is True
    assert body["label"] == "illicit"
    assert [f["feature"] for f in body["top_features"]] == ["local_1", "local_2", "local_3"]
    assert body["base_value"] == pytest.approx(-2.0)


def test_unknown_label_and_unknown_transaction(client):
    assert client.get("/transactions/5").json()["label"] == "unknown"
    assert client.get("/transactions/999").status_code == 404


def test_alerts_ranked_by_risk(client):
    body = client.get("/alerts", params={"time_step": 35, "k": 3}).json()
    assert [a["tx_id"] for a in body["alerts"]] == [8, 2, 7]
    assert body["threshold"] == pytest.approx(0.5)


def test_alerts_limits(client):
    assert client.get("/alerts", params={"time_step": 35, "k": 21}).status_code == 422
    assert client.get("/alerts", params={"time_step": 35, "k": 0}).status_code == 422
    assert client.get("/alerts", params={"time_step": 12, "k": 3}).status_code == 404


def test_neighbourhood(client):
    body = client.get("/transactions/1/neighbourhood", params={"hops": 1}).json()
    nodes = {n["tx_id"]: n for n in body["nodes"]}
    assert set(nodes) == {1, 2, 6, 7}
    assert nodes[1]["hop"] == 0 and nodes[2]["hop"] == 1
    assert nodes[2]["flagged"] is True
    assert {(e["src"], e["dst"]) for e in body["edges"]} == {(1, 2), (1, 6), (7, 1)}
    assert body["truncated"] is False
    assert client.get("/transactions/1/neighbourhood", params={"hops": 4}).status_code == 422


def test_paths_fan_in_communities(client):
    assert client.get("/paths", params={"source": 1, "target": 4}).json()["paths"] == [
        [1, 2, 3, 4]
    ]  # fmt: skip
    fan = client.get("/fan-in", params={"time_step": 35, "min_flagged": 2}).json()["items"]
    assert [(f["tx_id"], f["n_flagged_payers"]) for f in fan] == [(8, 3)]
    comm = client.get("/communities", params={"time_step": 35, "min_size": 2}).json()["items"]
    assert [c["community_id"] for c in comm] == [1, 0]  # higher flagged share first


def test_demo_mode_hides_queries_that_need_the_whole_graph(service):
    service.data.meta.update(has_full_graph=False, neighbourhood_roots=[8], max_hops=1)
    client = TestClient(create_app(service))
    assert client.get("/paths", params={"source": 1, "target": 4}).status_code == 501
    assert client.get("/fan-in", params={"time_step": 35}).status_code == 501
    assert client.get("/transactions/8/neighbourhood", params={"hops": 1}).status_code == 200
    # Outside the shipped neighbourhoods the graph is incomplete, so it is refused.
    assert client.get("/transactions/1/neighbourhood", params={"hops": 1}).status_code == 501
    assert client.get("/transactions/8/neighbourhood", params={"hops": 2}).status_code == 422


def test_demo_without_labels(service):
    service.data.transactions = service.data.transactions.drop(columns="label")
    service.data.meta["labels_included"] = False
    body = TestClient(create_app(service)).get("/transactions/8").json()
    assert body["label"] is None


def test_api_and_local_clients_agree(service):
    # The dashboard talks to the API in docker compose and to the service directly in the
    # public demo; both must show the same thing.
    http = TestClient(create_app(service))
    api, local = ApiClient("", get=http.get), LocalClient(service)
    for name, args in [
        ("meta", ()),
        ("steps", ()),
        ("transaction", (8,)),
        ("alerts", (35, 3)),
        ("neighbourhood", (1, 2)),
        ("communities", (35, 2, 10)),
        ("fan_in", (35, 1, 10)),
        ("paths", (1, 8)),
    ]:
        assert getattr(api, name)(*args) == getattr(local, name)(*args), name
