# ADK scans for this exact import. Without it the agent does not appear in
# `adk web` / `adk api_server` and there is no error message explaining why.
from . import agent  # noqa: F401
