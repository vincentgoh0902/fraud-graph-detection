# FastAPI service (Phase 6). Only the serving dependencies (D-041): no XGBoost, no torch.
# Build from the repo root: docker build -f docker/api.Dockerfile .
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
# Dependencies first, so code changes don't re-download them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --group api --no-install-project
COPY src ./src
RUN uv sync --locked --no-default-groups --group api

FROM python:3.12-slim
WORKDIR /app
COPY --from=build /app /app
COPY configs/serving.yaml configs/graph.yaml ./configs/
RUN useradd --create-home app
USER app
ENV PATH="/app/.venv/bin:$PATH" PORT=8000 ARTIFACTS_DIR=/artifacts GRAPH_BACKEND=memory
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=12 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn fraudgraph.api.app:app --host 0.0.0.0 --port ${PORT}"]
