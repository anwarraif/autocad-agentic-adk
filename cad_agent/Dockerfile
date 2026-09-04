# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# cad-agent — ADK agent exposed over HTTP.
#
# `adk api_server` is run from the PARENT of the agent package: ADK scans the
# given directory for subdirectories that look like agent packages.
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
COPY --chown=cad:cad cad_agent ./cad_agent
USER cad
EXPOSE 8000
CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8000", "."]
