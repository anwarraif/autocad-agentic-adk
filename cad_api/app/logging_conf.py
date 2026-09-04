"""Structured JSON logging to stdout.

Cloud Run reads stdout and parses JSON into structured log fields; locally
`docker compose logs` shows the same lines. One format everywhere means a
log line that is useful in production is also the one seen during development.
"""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import json as jsonlogger

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Install a JSON formatter on the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"levelname": "severity", "asctime": "timestamp"},
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # pymongo's heartbeat chatter is noise at DEBUG and can echo connection
    # details; keep it at WARNING regardless of our level.
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    _CONFIGURED = True
