# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# cad-mcp — MCP tool layer.
#
# Holds no CAD logic: it calls cad-api over HTTP, so the drawing engine exists
# once and the agent and the browser cannot drift apart on the numbers.
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=builder /install /usr/local
RUN useradd --create-home --uid 10001 cad
WORKDIR /app
COPY --chown=cad:cad server.py ./server.py
USER cad
EXPOSE 8000
CMD ["python", "server.py"]
