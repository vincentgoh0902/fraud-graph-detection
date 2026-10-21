# Hugging Face Space (Docker SDK): the dashboard in-process on the demo subset (D-040).
# Built from the folder scripts/build_space.py assembles, which is what gets pushed to the
# Space. Spaces route traffic to port 7860.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --group dashboard --no-install-project
COPY src ./src
RUN uv sync --locked --no-default-groups --group dashboard

FROM python:3.12-slim
# Spaces run the container as user 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build /app /app
COPY configs ./configs
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
COPY artifacts ./artifacts
USER app
ENV PATH="/app/.venv/bin:$PATH" PORT=7860 ARTIFACTS_DIR=/app/artifacts/space
EXPOSE 7860
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port ${PORT} --server.address 0.0.0.0"]
