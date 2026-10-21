# Streamlit dashboard (Phase 6). Talks to the API when API_URL is set; otherwise reads
# exported files from ARTIFACTS_DIR. No XGBoost, no torch (D-041).
# Build from the repo root: docker build -f docker/dashboard.Dockerfile .
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --group dashboard --no-install-project
COPY src ./src
RUN uv sync --locked --no-default-groups --group dashboard

FROM python:3.12-slim
WORKDIR /app
COPY --from=build /app /app
COPY configs/serving.yaml configs/graph.yaml ./configs/
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
RUN useradd --create-home app
USER app
ENV PATH="/app/.venv/bin:$PATH" PORT=8501 ARTIFACTS_DIR=/artifacts
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --retries=12 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health')"
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port ${PORT} --server.address 0.0.0.0"]
