FROM python:3.14-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /bin/uv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
RUN useradd --create-home --uid 10001 morgott

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra azure --extra cascade

COPY src ./src
COPY --chown=morgott:morgott model-artifacts.json ./
COPY --chown=morgott:morgott reports/retrieval-lineage-hybrid-parity-relaxed-20260820.json \
    ./reports/retrieval-lineage-hybrid-parity-relaxed-20260820.json
COPY --chown=morgott:morgott artifacts/models/mmbert-lora-full-ctx1024-u17000-s42/serving \
    ./artifacts/models/mmbert-lora-full-ctx1024-u17000-s42/serving
RUN uv sync --frozen --no-dev --extra azure --extra cascade

USER 10001
EXPOSE 8000
CMD ["uvicorn", "--factory", "morgott.azure_app:create_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
