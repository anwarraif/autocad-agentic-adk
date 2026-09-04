"""Configuration, entirely from the environment.

Nothing here has a hard-coded production value. The two exceptions are
deliberate and live in `mongo.py`: the database name and the collection
prefix are pinned in code precisely so that a stray environment variable
cannot redirect writes into a collection belonging to another team.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for cad-api."""

    model_config = SettingsConfigDict(env_prefix="CAD_", extra="ignore")

    #: Directory holding the .dxf files available for ingest (read-only mount).
    dxf_dir: Path = Path("/data/dxf")

    #: Directory holding the original .dwg files (read-only mount). Separate
    #: from dxf_dir, and `--all` used to scan only the latter -- so five of the
    #: eighteen drawings in the store, including the reference drawing the whole
    #: campaign measures itself against, were silently never re-ingested. The
    #: report said "13 ok, 0 failed" and was telling the truth about the wrong
    #: thing.
    dwg_dir: Path = Path("/data/dwg")

    #: Directory where rendered SVGs are cached, gzip-compressed.
    svg_dir: Path = Path("/data/svg")

    #: Where documents embedded INSIDE a drawing are cached once extracted.
    #:
    #: Under the same writable mount as the SVG cache, and for the same
    #: reason: it is derived from the drawing, cheap to rebuild, and nothing
    #: is lost if it is deleted. Extraction is lazy -- the first request for a
    #: drawing pays for it -- because most drawings embed nothing and re-
    #: reading a 136 MB DXF on every ingest to discover that would be a cost
    #: paid by everyone for the benefit of a few.
    embedded_dir: Path = Path("/data/svg/embedded")

    #: Cap on entities rendered in a single SVG. A drawing above this renders
    #: partially and says so in the response, rather than timing out the
    #: request. Janadriyah's modelspace is ~20k, so the default lets it
    #: through whole; lower it if the browser struggles.
    render_max_entities: int = 60_000

    #: Rows returned by a single list endpoint. Mirrors the MCP tool cap so
    #: the agent and the UI cannot disagree about what "a page" means.
    max_rows: int = 100

    #: Above this many matches, list endpoints return an aggregate breakdown
    #: instead of rows, so a broad query degrades into a summary rather than
    #: a wall of data.
    too_many_threshold: int = 5_000

    log_level: str = "INFO"

    #: Comma-separated origins allowed to call this API from a browser.
    cors_origins: str = "http://localhost:4310"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
