FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock .python-version ./
# Large wheels (transformers, torch) can outlast uv's 30 s default on a slow link. The cache mount
# keeps what was already downloaded, so a retried build resumes instead of starting over.
ARG UV_HTTP_TIMEOUT=300
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY app ./app
COPY start.sh ./

RUN --mount=type=cache,target=/root/.cache/uv \
    chmod +x start.sh \
    && uv sync --frozen

EXPOSE 8895 8897

CMD ["./start.sh"]
