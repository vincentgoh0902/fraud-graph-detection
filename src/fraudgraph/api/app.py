"""FastAPI app (brief section 6). Everything is loaded once at startup; nothing is scored
per request (D-038).

Run locally:
    uv run uvicorn fraudgraph.api.app:app --port 8000

Settings (environment):
    ARTIFACTS_DIR   exported serving files (default artifacts/serving)
    GRAPH_BACKEND   neo4j | memory (default memory); neo4j also reads NEO4J_URI/USER/PASSWORD
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fraudgraph.api import schemas
from fraudgraph.config import load_config, resolve
from fraudgraph.serving.artifacts import load_artifacts
from fraudgraph.serving.graph import InMemoryStore, Neo4jStore
from fraudgraph.serving.service import (
    InvalidRequest,
    Limits,
    NotFound,
    ServingService,
    Unavailable,
)


def service_from_env() -> ServingService:
    cfg = load_config("serving")
    data = load_artifacts(resolve(os.environ.get("ARTIFACTS_DIR", cfg["output_dir"])))
    backend = os.environ.get("GRAPH_BACKEND", "memory")
    if backend == "neo4j":
        from fraudgraph.graph.neo4j_client import connect

        store = Neo4jStore(connect(), load_config("graph")["neo4j"]["database"])
    elif backend == "memory":
        store = InMemoryStore(data.edges, data.transactions)
    else:
        raise ValueError(f"GRAPH_BACKEND must be neo4j or memory, got {backend!r}")
    return ServingService(data, store, Limits(**cfg["limits"]))


def create_app(service: ServingService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # A service passed in (tests) is set below; otherwise load once, at startup.
        if not hasattr(app.state, "service"):
            app.state.service = service_from_env()
        yield
        driver = getattr(app.state.service.store, "driver", None)
        if driver is not None:
            driver.close()

    app = FastAPI(
        title="fraud-graph-detection",
        description="Risk scores, SHAP explanations and graph queries for Elliptic steps 35-49.",
        lifespan=lifespan,
    )
    if service is not None:
        app.state.service = service

    # Service errors map to HTTP codes in one place; the dashboard's ApiClient maps them back.
    for exc, code in ((NotFound, 404), (InvalidRequest, 422), (Unavailable, 501)):

        def handler(request: Request, err: Exception, code: int = code) -> JSONResponse:
            return JSONResponse(status_code=code, content={"detail": str(err)})

        app.add_exception_handler(exc, handler)

    def svc(request: Request) -> ServingService:
        return request.app.state.service

    @app.get("/health")
    def health(request: Request) -> dict:
        m = svc(request).meta()
        return {"status": "ok", "model_version": m["model_version"], "backend": m["backend"]}

    @app.get("/meta", response_model=schemas.Meta)
    def meta(request: Request):
        return svc(request).meta()

    @app.get("/steps", response_model=schemas.Steps)
    def steps(request: Request):
        return svc(request).steps()

    @app.get("/transactions/{tx_id}", response_model=schemas.Transaction)
    def transaction(tx_id: int, request: Request):
        return svc(request).transaction(tx_id)

    @app.get("/transactions/{tx_id}/neighbourhood", response_model=schemas.Neighbourhood)
    def neighbourhood(tx_id: int, request: Request, hops: int = 2):
        return svc(request).neighbourhood(tx_id, hops)

    @app.get("/alerts", response_model=schemas.Alerts)
    def alerts(time_step: int, request: Request, k: int = 20):
        return svc(request).alerts(time_step, k)

    @app.get("/paths", response_model=schemas.Paths)
    def paths(source: int, target: int, request: Request):
        return svc(request).paths(source, target)

    @app.get("/communities", response_model=schemas.Communities)
    def communities(time_step: int, request: Request, min_size: int = 5, top: int = 10):
        return svc(request).communities(time_step, min_size, top)

    @app.get("/fan-in", response_model=schemas.FanIn)
    def fan_in(time_step: int, request: Request, min_flagged: int = 2, top: int = 20):
        return svc(request).fan_in(time_step, min_flagged, top)

    return app


app = create_app()
