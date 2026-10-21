"""Response models: the API's contract, shown in its /docs page.

They mirror the dicts ServingService returns; tests/test_api.py checks the API and the
in-process client give identical answers, so a field missing here would fail it.
"""

from pydantic import BaseModel


class Meta(BaseModel):
    model_version: str
    threshold: float
    steps: list[int]
    backend: str
    demo: bool
    has_full_graph: bool
    labels_included: bool
    neighbourhood_roots: list[int] | None
    max_hops: int
    max_k: int
    max_nodes: int


class StepSummary(BaseModel):
    time_step: int
    n_transactions: int
    n_flagged: int
    f1: float | None = None
    pr_auc: float | None = None
    recall: float | None = None
    n_illicit: float | None = None


class Steps(BaseModel):
    steps: list[StepSummary]


class Node(BaseModel):
    tx_id: int
    time_step: int
    risk_score: float
    flagged: bool
    label: str | None  # None when the export ships no labels (D-040)


class Contribution(BaseModel):
    feature: str
    group: str
    shap: float  # log-odds; positive pushes towards illicit (D-034)


class Transaction(Node):
    threshold: float
    base_value: float
    top_features: list[Contribution]


class Alerts(BaseModel):
    time_step: int
    threshold: float
    alerts: list[Node]


class HopNode(Node):
    hop: int


class Edge(BaseModel):
    src: int
    dst: int


class Neighbourhood(BaseModel):
    tx_id: int
    hops: int
    truncated: bool
    nodes: list[HopNode]
    edges: list[Edge]


class Paths(BaseModel):
    source: int
    target: int
    paths: list[list[int]]


class Community(BaseModel):
    community_id: int
    size: int
    n_flagged: int
    share_flagged: float
    mean_risk: float


class Communities(BaseModel):
    time_step: int
    items: list[Community]


class FanInNode(Node):
    n_flagged_payers: int


class FanIn(BaseModel):
    time_step: int
    items: list[FanInNode]
