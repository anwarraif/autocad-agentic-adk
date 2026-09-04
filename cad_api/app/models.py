"""Request and response models for cad-api."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CommentIn(BaseModel):
    """A comment to anchor onto one entity."""

    drawing_id: str = Field(..., description="Content-hash id of the drawing.")
    handle: str = Field(..., description="DWG handle of the entity being commented on.")
    body: str = Field(..., min_length=1, max_length=5000, description="Comment text.")
    author: str = Field(
        default="user:local",
        description="Who wrote it, e.g. 'user:kurnia' or 'agent:cad-agent'.",
    )
    anchor: list[float] | None = Field(
        default=None,
        description="Optional [x, y] display position, in drawing units.",
    )
    client_request_id: str | None = Field(
        default=None,
        description=(
            "Idempotency key. Re-sending the same value returns the original "
            "comment instead of creating a duplicate."
        ),
    )
    provenance: dict[str, Any] | None = Field(
        default=None,
        description="Where this comment came from: tool, session, model.",
    )


class CommentOut(BaseModel):
    """A stored comment."""

    id: str = Field(default="", alias="_id")
    drawing_id: str
    entity_handle: str
    body: str
    author: str
    created_at: datetime
    status: str
    anchor: dict[str, Any] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    client_request_id: str | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}


class RegionIn(BaseModel):
    """A region drawn over the drawing, in **drawing coordinates**.

    Not screen pixels, and the distinction is the whole feature. A region
    stored in pixels stops meaning anything the moment the user zooms; stored
    in drawing coordinates it stays fixed to the drawing for ever, which is
    what lets the same selection survive pan, zoom and a page reload.
    """

    points: list[list[float]] = Field(
        ...,
        min_length=2,
        max_length=200,
        description=(
            "Vertices in drawing units. Two opposite corners for kind='rect', "
            "three or more vertices for kind='polygon', and [centre, rim] for "
            "kind='circle' — the radius is the distance between them."
        ),
    )
    kind: str = Field(
        default="rect",
        pattern="^(rect|polygon|circle)$",
        description="'rect', 'polygon' or 'circle'.",
    )
    mode: str = Field(
        default="crossing",
        pattern="^(window|crossing)$",
        description=(
            "AutoCAD's two selection modes. 'window' takes only entities "
            "wholly inside the region (drag left-to-right); 'crossing' takes "
            "anything it touches (drag right-to-left)."
        ),
    )
    layout: str | None = Field(
        default=None,
        description=(
            "Layout the region was drawn on. Almost always required: without "
            "it, a region drawn on a paper sheet also matches modelspace "
            "geometry sharing those coordinates."
        ),
    )
    max_handles: int = Field(
        default=25000,
        ge=1,
        le=25000,
        description=(
            "Cap on how many handles are NAMED. The counts are exact "
            "regardless of it: a region matching 40,000 objects reports "
            "40,000 whatever this is set to."
        ),
    )


class DescribeSelectionIn(BaseModel):
    """What to describe: a stored selection, or an explicit handle list.

    `selection_id` is the preferred form and the reason the store exists —
    the caller passes a reference and the server reads the whole selection,
    however large. `handles` remains for callers that hold a list already;
    exactly one of the two must be given.
    """

    selection_id: str | None = Field(
        default=None,
        max_length=64,
        description="Id from POST /drawings/{id}/selections.",
    )
    handles: list[str] | None = Field(
        default=None,
        max_length=25000,
        description="Entity handles, when no stored selection is being used.",
    )
    sample: int = Field(
        default=5,
        ge=0,
        le=25,
        description="Example rows per (layer, type) group.",
    )


class RegionRowsIn(RegionIn):
    """One (layer, type) group of a region, requested as rows.

    Inherits the region itself so the server re-evaluates it rather than
    trusting a handle list the client may have truncated.
    """

    layer: str = Field(..., description="Exact layer name from the summary.")
    type: str = Field(..., description="Exact DXF type from the summary.")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=200, ge=1, le=500)


class NameCount(BaseModel):
    """One row of an exact breakdown the viewer already computed."""

    name: str = Field(..., max_length=255)
    count: int = Field(..., ge=0)


class SaveSelectionIn(BaseModel):
    """A selection to store and refer to by id afterwards."""

    handles: list[str] = Field(..., min_length=1, max_length=25000)
    layout: str | None = None
    mode: str | None = Field(default=None, pattern="^(window|crossing)$")
    kind: str | None = Field(default=None, pattern="^(rect|polygon|circle|click)$")
    summary: str | None = Field(
        default=None,
        max_length=2000,
        description="Human-readable one-liner the viewer already computed.",
    )
    total: int | None = Field(
        default=None,
        ge=0,
        description=(
            "How many objects the selection really holds. Sent separately "
            "from `handles` because a selection may match more than the "
            "enumeration ceiling can name, and the count is exact either way."
        ),
    )
    by_layer: list[NameCount] | None = Field(default=None, max_length=500)
    by_type: list[NameCount] | None = Field(default=None, max_length=500)
