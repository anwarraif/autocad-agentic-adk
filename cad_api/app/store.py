"""Persistence for drawings, entities, comments and rendered SVG.

Every database call in this module goes through `mongo.coll()`, which refuses
any collection name outside the `autocad_` prefix. See `mongo.py`.

Where the SVG lives, and why not in MongoDB
-------------------------------------------
A rendered Janadriyah modelspace is ~17.5 MB of SVG; a BSON document is capped
at 16 MB, so the obvious "just store it alongside the drawing" does not
actually work. It compresses ~95:1 (measured), so it is written to disk as
`.svg.gz` and served with `Content-Encoding: gzip`; MongoDB keeps only the
metadata needed to find and trust it. On Cloud Run the same code writes to a
GCS bucket instead — only `_svg_path()` changes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Any, Final, Iterable, Sequence

from pymongo import ReplaceOne, UpdateOne
from pymongo.errors import DuplicateKeyError

from . import embedded, mongo, region, shx
from .config import get_settings
from .mongo import SELECTION_TTL_SECONDS
from .extract import BLOCK_LAYOUT_PREFIX, INGEST_VERSION, DrawingDoc, EntityDoc
from .mongo import (
    COLL_COMMENTS,
    COLL_DRAWINGS,
    COLL_ENTITIES,
    COLL_RENDERS,
    COLL_SELECTIONS,
    coll,
)

log = logging.getLogger(__name__)

# Batch size for bulk entity writes. 1000 keeps each command comfortably under
# the 48 MB command limit even for entities carrying long text.
_BULK_BATCH = 1000

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")

#: Entries listed in a "too many matches" breakdown. The long tail of
#: near-empty layers is noise in a summary; the cap is reported rather than
#: applied in silence.
_BREAKDOWN_LIMIT = 25


class CommentIdConflict(RuntimeError):
    """A `client_request_id` was reused for a different entity.

    An idempotency key means "this is the same write again". Reusing one for a
    different target is a caller bug, and must be reported rather than
    silently resolved in either direction.
    """


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Drawings
# ---------------------------------------------------------------------------


def upsert_drawing(drawing: DrawingDoc) -> None:
    """Insert or replace one drawing document.

    Idempotent by construction: `_id` is the content hash, so re-ingesting the
    same file overwrites its own document and creates nothing new.
    """
    doc = drawing.to_mongo()
    doc["ingested_at"] = _utcnow()
    coll(COLL_DRAWINGS).replace_one({"_id": drawing._id}, doc, upsert=True)


def get_drawing(drawing_id: str) -> dict[str, Any] | None:
    """One drawing document, or None if the id is unknown."""
    return coll(COLL_DRAWINGS).find_one({"_id": drawing_id})


#: The directory whose drawings exist to be TESTED against, not to be read.
#:
#: `scripts/dossier_diff.py` writes a synthetic before/after pair here so that
#: `revision_diff` has something to compare when no vendor revision exists. They
#: are real ingested drawings and must stay reachable by id -- the revision
#: tests open them directly -- but they are not this project's drawings, and a
#: picker that offers them invites someone to review a file nobody drew.
#:
#: A directory name, not a drawing name: nothing here knows what is inside
#: those files (G1).
FIXTURE_DIR: Final[str] = "revision_pair"


def _is_fixture(row: Mapping[str, Any]) -> bool:
    """Was this drawing ingested from the fixture directory?

    Compared as a path SEGMENT rather than a substring, so a real drawing that
    happens to have those letters in its filename is not hidden by accident.
    Both separators are handled because the stored path comes from whichever
    machine ingested it.
    """
    source = str(row.get("source_path") or "")
    if not source:
        return False
    parts = source.replace("\\", "/").split("/")
    return FIXTURE_DIR in parts[:-1]



def mark_ingest_incomplete(drawing_id: str) -> None:
    """Hide a drawing from the picker until its entities are stored.

    Set the moment the drawing document is written and cleared when the
    entities are safely down. A crash between the two leaves the flag set,
    which is the honest outcome: the drawing is not usable and saying so
    beats offering it.
    """
    coll(COLL_DRAWINGS).update_one(
        {"_id": drawing_id}, {"$set": {"ingest_incomplete": True}}
    )


def clear_ingest_incomplete(drawing_id: str) -> None:
    """The entities are stored; the drawing is real. Unset, not set-false, so
    the field leaves no trace on a healthy drawing."""
    coll(COLL_DRAWINGS).update_one(
        {"_id": drawing_id}, {"$unset": {"ingest_incomplete": ""}}
    )

def list_drawings(*, include_fixtures: bool = False) -> list[dict[str, Any]]:
    """All ingested drawings, lightest fields only, newest first.

    The per-layer and per-type count maps are excluded: on Janadriyah they are
    292 and 40 keys respectively, which is far more than a drawing picker
    needs and would dominate the payload.

    Test fixtures are left out by default and counted rather than dropped
    silently -- the caller is told how many were withheld and how to ask for
    them, because a list that quietly omits things is how someone concludes a
    drawing was never ingested.
    """
    cursor = (
        coll(COLL_DRAWINGS)
        .find(
            # A drawing whose entities never landed is not offered. The
            # drawing document is written BEFORE `replace_entities`, so an
            # ingest that dies in between leaves a picker entry announcing
            # hundreds of thousands of entities with nothing behind it, which
            # to a user is indistinguishable from a broken viewer. Sedra did
            # exactly that when its store stage failed.
            #
            # The flag is NEGATIVE on purpose: absent means complete, so every
            # drawing ingested before it existed stays visible.
            {"ingest_incomplete": {"$ne": True}},
            {
                "counts_by_layer": 0,
                "counts_by_type": 0,
                "layers": 0,
                "blocks": 0,
            },
        )
        .sort("ingested_at", -1)
    )
    rows = list(cursor)
    if include_fixtures:
        return rows
    return [row for row in rows if not _is_fixture(row)]


def count_fixtures() -> int:
    """How many ingested drawings are test fixtures."""
    cursor = coll(COLL_DRAWINGS).find({}, {"source_path": 1})
    return sum(1 for row in cursor if _is_fixture(row))


def version_facts_of(drawing_id: str) -> tuple[int | None, Any]:
    """`(ingest_version, ingested_at)` in ONE round trip.

    They were two projections and are always wanted together, and against
    Atlas a round trip is about 0.2 s -- which is real money in a request
    whose whole remaining fixed cost is under two seconds and is now made of
    round trips rather than of any one slow query.
    """
    row = coll(COLL_DRAWINGS).find_one(
        {"_id": drawing_id}, {"ingest_version": 1, "ingested_at": 1}
    ) or {}
    return row.get("ingest_version"), row.get("ingested_at")


def ingested_at_of(drawing_id: str) -> Any:
    """When this drawing was last ingested. Projected, for the same reason."""
    return version_facts_of(drawing_id)[1]


def ingest_version_of(drawing_id: str) -> int | None:
    """Just the version, without dragging the whole drawing document over.

    `get_drawing` returns everything, and on Sedra everything is 441 KB --
    705 layers, 705 `counts_by_layer` entries and 493 block names. Measured on
    Atlas that document costs 2.74 s to fetch and decode, against 0.22 s for
    this projection. It matters because the ETag's preconditions want ONE
    field from it and are computed on every request, twice.
    """
    row = coll(COLL_DRAWINGS).find_one({"_id": drawing_id}, {"ingest_version": 1})
    return (row or {}).get("ingest_version")


def drawing_is_current(drawing_id: str, svg_dir: Path) -> bool:
    """True if this drawing is fully ingested at the current version.

    "Fully" means stored, rendered, **and the rendered files still on disk**.

    Three states have to be told apart, and each was a real failure:

    1. The drawing document is written *before* rendering, so checking the
       ingest version alone treated a run killed mid-render as complete: every
       later `--all` reported SKIP while `GET /svg` returned RENDER_NOT_FOUND
       for ever.
    2. Render *records* live in MongoDB but the SVG files live on a Docker
       volume, and the two can be destroyed independently. After a
       `docker compose down -v` the records survive and the files do not, so
       trusting the records alone skipped twelve drawings whose SVGs no longer
       existed -- the viewer would have opened to an error on every one.
    3. A drawing with no renderable layout at all (content only in block
       definitions) is legitimately current. Demanding a render there would
       make it re-ingest for ever.
    """
    doc = coll(COLL_DRAWINGS).find_one(
        {"_id": drawing_id},
        {"ingest_version": 1, "layouts": 1},
    )
    if not doc or doc.get("ingest_version") != INGEST_VERSION:
        return False

    renderable = [l for l in doc.get("layouts", []) if l.get("entity_count", 0) > 0]
    records = list(coll(COLL_RENDERS).find({"drawing_id": drawing_id}, {"layout": 1}))

    if not renderable:
        return True
    if not records:
        return False

    # Every recorded render must still have its file. Checked against the
    # current svg_dir rather than the stored `path`, which was written by a
    # possibly differently-mounted container.
    return all(
        _svg_path(svg_dir, drawing_id, record["layout"]).exists()
        for record in records
    )


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


def replace_entities(drawing_id: str, entities: Sequence[EntityDoc]) -> int:
    """Write all entities for one drawing, replacing whatever was there.

    Upsert-by-`_id` rather than delete-then-insert: a re-ingest never leaves
    the collection empty in the window between the two operations, so a
    concurrent read cannot observe a drawing with no entities.

    Stale entities from a previous ingest of the *same* `drawing_id` cannot
    exist -- the id is a content hash, so different content means a different
    drawing_id -- but they are cleaned up anyway if the extractor changed.

    Replace, not `$set`. A `$set` adds fields and never removes them, so a
    field dropped from `EntityDoc` survives in the store as a ghost that reads
    exactly like live data. That is not hypothetical: renaming `centroid` to
    `bbox_centre` left both keys sitting on the same documents, which is the
    two-names-for-one-thing state the rename existed to end.
    """
    written = 0
    fresh_ids: list[str] = []
    batch: list[ReplaceOne] = []

    for entity in entities:
        doc = entity.to_mongo()
        fresh_ids.append(entity._id)
        batch.append(ReplaceOne({"_id": entity._id}, doc, upsert=True))
        if len(batch) >= _BULK_BATCH:
            coll(COLL_ENTITIES).bulk_write(batch, ordered=False)
            written += len(batch)
            batch = []

    if batch:
        coll(COLL_ENTITIES).bulk_write(batch, ordered=False)
        written += len(batch)

    removed = _sweep_stale(drawing_id, fresh_ids)
    if removed:
        log.info(
            "removed stale entities",
            extra={"drawing_id": drawing_id, "removed": removed},
        )
    return written


def _sweep_stale(drawing_id: str, fresh_ids: Sequence[str]) -> int:
    """Delete this drawing's entities that this ingest did not write.

    The obvious form of this is one `delete_many` carrying
    `{"_id": {"$nin": fresh_ids}}`, and that is what it used to be. A `$nin`
    holding every fresh id is a single BSON command document, so its size
    grows with the drawing, and MongoDB caps a command at 16 MB. Measured:
    Sedra extracts 990,790 entities and the sweep failed outright with
    `DocumentTooLarge: 'delete' command document too large`, after a 24 minute
    extract had already succeeded. The drawing was unstorable for a reason
    that had nothing to do with the drawing.

    So the difference is taken here instead. The stored ids for this drawing
    are read back, the ones this ingest wrote are subtracted, and only what is
    genuinely stale is named in the delete, in batches. Every command is
    bounded no matter how large the drawing is, and the common case sends no
    delete at all because there is nothing stale to remove.
    """
    fresh = set(fresh_ids)
    stale = [
        row["_id"]
        for row in coll(COLL_ENTITIES).find({"drawing_id": drawing_id}, {"_id": 1})
        if row["_id"] not in fresh
    ]
    if not stale:
        return 0

    removed = 0
    for start in range(0, len(stale), _BULK_BATCH):
        removed += coll(COLL_ENTITIES).delete_many(
            {"_id": {"$in": stale[start : start + _BULK_BATCH]}}
        ).deleted_count
    return removed


def get_entity(drawing_id: str, handle: str) -> dict[str, Any] | None:
    """One entity by its `(drawing_id, handle)` key, and what contains it.

    The ring is deliberately withheld. A single polygon can carry thousands of
    vertices, and this response goes straight into an agent tool call; one
    entity would spend the whole context budget. `ring_vertex_count` says how
    many there are, and the geometry endpoints return the ring when something
    actually needs it.

    `h3_cells` is withheld for exactly the same reason and was measured doing
    exactly the same damage: the site-boundary polygon on the reference
    drawing covers 20,000 cells and its entity response came to 392 KB, from
    15 KB. `h3_cell` names the one cell that stands for the entity and
    `h3_cell_count` says how many there are in total, which is what a reader
    of one entity wants; the covering itself belongs to the geo endpoints.

    `labels_inside` is attached here rather than left as a separate call, and
    that is the point of it. Asked for the area of a plot number on 23 August,
    the agent answered about the TEXT entity -- which has no area at all. The
    failure was not a wrong number; nothing in the response pointed at the
    parcel the number labels. Now it does.
    """
    doc = coll(COLL_ENTITIES).find_one(
        {"_id": f"{drawing_id}:{handle}"},
        {
            "ring": 0,
            "ring_diagnostic": 0,
            "edge_lengths": 0,
            "h3_cells": 0,
            # An index accelerator, not a fact about the entity. `h3_coarse`
            # is the resolution-8 parent of `h3_cell`, written so a viewport
            # can match an index instead of scanning; a reader who wants the
            # cell has `h3_cell`, and a reader who depends on this one would
            # be depending on how the query happens to be optimised today.
            "h3_coarse": 0,
        },
    )
    if doc is None:
        return None

    # Two directions, and each is attached to the entity it actually answers
    # for. This used to attach `labels_inside` -- the parcel->labels direction
    # -- to anything that had an insertion point, which is precisely the set of
    # entities that never have a ring, so it returned "this is not a polygon"
    # every single time. The pairing below is the fix, and it is the same
    # mix-up the `/containing` route carried for a day.
    if doc.get("ring_status") == "complete":
        try:
            doc["labels_inside"] = store_spatial.labels_inside(drawing_id, handle)
        except Exception as exc:  # containment is an extra, never the answer
            log.debug("labels_inside failed for %s: %s", handle, exc)
    if doc.get("anchor_point") is not None or doc.get("polygon_centroid"):
        try:
            doc["contained_by"] = store_spatial.parcels_containing(drawing_id, handle)
        except Exception as exc:
            log.debug("parcels_containing failed for %s: %s", handle, exc)

    reading = _reading_for(drawing_id, doc)
    if reading is not None:
        doc.update(reading)
    return doc


def _reading_for(drawing_id: str, doc: dict[str, Any]) -> dict[str, Any] | None:
    """The Arabic reading for one text entity, if one can be read at all.

    The text style is resolved through this drawing's `text_styles` table, not
    guessed from the letters. That is the difference between reading `مسجد محلي`
    out of `ls{] lpgD` and inventing an Arabic word from an English word that
    happens to be typed on the same keys.
    """
    text = doc.get("text")
    if not text:
        return None
    style_name = doc.get("text_style")
    font = bigfont = None
    style_known = False
    if style_name:
        drawing = coll(COLL_DRAWINGS).find_one(
            {"_id": drawing_id}, {"text_styles": 1, "_id": 0}
        )
        for style in (drawing or {}).get("text_styles") or []:
            if str(style.get("name")) == str(style_name):
                font = style.get("font")
                bigfont = style.get("bigfont")
                style_known = True
                break
    out = shx.describe(text, font=font, bigfont=bigfont, style_known=style_known)
    out.pop("text", None)  # `text` is already on the document and must not change
    out["text_style_font"] = font
    out["text_style_bigfont"] = bigfont
    return out


def _embedded_cache(drawing_id: str) -> Path:
    return get_settings().embedded_dir / drawing_id


def _drawing_source(drawing: Mapping[str, Any]) -> Path | None:
    """The DXF this drawing was read from, if it is still on disk.

    Five of the eighteen drawings here came from DWG and were converted, so
    the path on the document may point at the original. The converted DXF is
    the one that holds the text this reads.
    """
    for key in ("dxf_path", "converted_path", "source_path"):
        value = drawing.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.suffix.lower() == ".dxf" and path.is_file():
            return path
    name = drawing.get("original_filename") or ""
    if name:
        candidate = get_settings().svg_dir / "converted" / str(name)
        if candidate.is_file():
            return candidate
        candidate = get_settings().dxf_dir / str(name)
        if candidate.is_file():
            return candidate
    return None


def embedded_index(
    drawing_id: str, drawing: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """What this drawing carries inside it, extracting on first ask.

    Lazy on purpose. Most drawings embed nothing, and re-reading a 136 MB DXF
    during every ingest to find that out is a cost paid by everyone for the
    benefit of a few. The count comes from the entity documents, which are
    already stored, so "are there any" is answered without touching the file.
    """
    drawing = drawing or get_drawing(drawing_id) or {}
    declared = coll(COLL_ENTITIES).count_documents(
        {"drawing_id": drawing_id, "type": {"$in": sorted(embedded.PAYLOAD_TYPES)}}
    )
    cache = _embedded_cache(drawing_id)
    marker = cache / "index.json"

    if declared == 0:
        return {
            "drawing_id": drawing_id,
            "declared": 0,
            "payloads": [],
            "note": "this drawing carries no embedded documents",
        }

    if marker.is_file():
        try:
            rows = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows = None
        if isinstance(rows, list):
            return {
                "drawing_id": drawing_id,
                "declared": declared,
                "payloads": rows,
                "note": None,
            }

    source = _drawing_source(drawing)
    if source is None:
        return {
            "drawing_id": drawing_id,
            "declared": declared,
            "payloads": [],
            "not_measured": (
                f"{declared} embedded documents are in this drawing, but its "
                "DXF file is no longer on disk, so their contents cannot be "
                "extracted. The count comes from the stored entities"
            ),
        }

    rows = embedded.write_all(source, cache)
    try:
        marker.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    except OSError:  # cache is an optimisation, never the answer
        log.debug("could not write embedded index for %s", drawing_id)
    return {
        "drawing_id": drawing_id,
        "declared": declared,
        "payloads": rows,
        "note": (
            "extracted from the file on request; the contents are drawn by no "
            "renderer at all and can only be read by opening them"
        ),
    }


def embedded_file(
    drawing_id: str, handle: str, drawing: Mapping[str, Any] | None = None
) -> tuple[Path, str] | None:
    """One extracted payload and its media type, or `None`."""
    index = embedded_index(drawing_id, drawing)
    for row in index.get("payloads") or []:
        if str(row.get("handle")) == handle:
            path = _embedded_cache(drawing_id) / str(row.get("file"))
            if path.is_file():
                return path, str(row.get("media_type") or "application/octet-stream")
    return None


def entity_handles(
    drawing_id: str,
    *,
    layout_name: str | None = None,
    layer: str | None = None,
    dxftype: str | None = None,
    block_name: str | None = None,
    limit: int = 25_000,
) -> dict[str, Any]:
    """Handles only, for marking objects on screen.

    Deliberately not a variant of `query_entities`. That function returns rows
    that a model reads, so every field it carries is a field competing for
    context, and its 100-row cap exists to protect exactly that. This one
    returns seven characters per object and is read by a browser, where the
    same cap meant a request per hundred objects: marking the 9,867 dimensions
    of the reference drawing cost 99 round trips.

    Same filters, same index, same meaning of `total` -- so a count from here
    and a count from `query_entities` can never disagree.
    """
    query: dict[str, Any] = {"drawing_id": drawing_id}
    if layout_name:
        query["layout"] = layout_name
    if layer:
        query["layer"] = layer
    if dxftype:
        query["type"] = dxftype.upper()
    if block_name:
        query["block_name"] = block_name

    collection = coll(COLL_ENTITIES)
    total = collection.count_documents(query)
    rows = collection.find(query, {"handle": 1, "_id": 0}).limit(limit)
    handles = [r["handle"] for r in rows if r.get("handle")]
    return {
        "drawing_id": drawing_id,
        "total_matches": total,
        "returned": len(handles),
        "truncated": total > len(handles),
        "handles": handles,
    }


def query_entities(
    drawing_id: str,
    *,
    layer: str | None = None,
    dxftype: str | None = None,
    block_name: str | None = None,
    layout_name: str | None = None,
    text_contains: str | None = None,
    handles: Iterable[str] | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Structural filter over entities, always paginated.

    Returns a dict carrying `total_matches` and `truncated` alongside the
    rows. Those two fields are not decoration: without them a caller cannot
    tell "there are 12 results" from "there are 40,000 and you are looking at
    the first 100", and that difference changes the answer a user gets.
    """
    _reject_empty_filters(
        layer=layer, type=dxftype, block_name=block_name,
        layout=layout_name, text_contains=text_contains,
    )
    query: dict[str, Any] = {"drawing_id": drawing_id}
    if layer:
        query["layer"] = layer
    if dxftype:
        query["type"] = dxftype.upper()
    if block_name:
        query["block_name"] = block_name
    if layout_name:
        query["layout"] = layout_name
    if text_contains:
        # Anchored escape: a user string must not be able to inject regex.
        query["text"] = {"$regex": re.escape(text_contains), "$options": "i"}
    if handles is not None:
        query["handle"] = {"$in": list(handles)}

    collection = coll(COLL_ENTITIES)
    total = collection.count_documents(query)

    # The same question asked of the whole drawing.
    #
    # A layout-filtered count answers "what is on this sheet", which is the
    # right default and the wrong answer to an audit. Asked whether anything
    # sat on layer 0 -- a drafting-standards question -- the agent correctly
    # reported 10 for the sheet on screen and never mentioned the 639 in the
    # drawing, sixty-four times more. It labelled its scope, so it did not
    # mislead; it simply could not offer the figure the question was reaching
    # for, because it had only been given one.
    #
    # The remedy is the same one D-058 established: the fix belongs in the
    # data handed over, not in a firmer instruction. An agent holding both
    # numbers reports both. One extra indexed count, only when a layout
    # filter is actually narrowing something.
    total_in_drawing: int | None = None
    if layout_name:
        wider = {k: v for k, v in query.items() if k != "layout"}
        total_in_drawing = collection.count_documents(wider)

    # The same blindness, reported the same way. `total_in_drawing` counts every
    # layout INCLUDING the block-definition pseudo-layouts, so the two figures
    # can differ for a reason that has nothing to do with sheets, and a reader
    # comparing them deserves to know which.
    shadow = _block_definition_shadow(query)

    projection = {
        "handle": 1,
        "type": 1,
        "layer": 1,
        "layout": 1,
        "block_name": 1,
        "text": 1,
        "bbox_centre": 1,
        "bbox": 1,
    }
    rows = list(
        # Sorted so pagination is stable. MongoDB gives no order guarantee
        # without one, so `offset` could repeat or skip rows between pages.
        collection.find(query, projection)
        .sort([("layer", 1), ("handle", 1)])
        .skip(offset)
        .limit(limit)
    )
    returned = len(rows)
    truncated = total > offset + returned
    return {
        "drawing_id": drawing_id,
        "total_matches": total,
        "returned": returned,
        "offset": offset,
        "truncated": truncated,
        "next_offset": offset + returned if truncated else None,
        # Present only when a layout filter was applied, so its absence is
        # never ambiguous: it means the count already spans the drawing.
        **({"total_in_drawing": total_in_drawing} if layout_name else {}),
        **({"block_definition_shadow": shadow} if shadow else {}),
        **(
            {
                "scope_note": (
                    str(shadow["entities"]) + " further entities match this "
                    "filter inside block definitions and are not counted here. "
                    "They are included in total_in_drawing."
                )
            }
            if shadow
            else {}
        ),
        "entities": rows,
        # The exact Mongo query that produced `total_matches`. A caller
        # wanting a breakdown of these results must aggregate over *this*,
        # not over a query it reconstructs from the original parameters --
        # rebuilding it drops filters and misses normalisation (`type` is
        # upper-cased here), which yields a breakdown describing a different
        # population than the count it sits next to.
        "_query": query,
    }


def entity_breakdown(query: dict[str, Any]) -> dict[str, Any]:
    """Counts per layer and per type for a query that matched too much.

    Given to a caller instead of rows when a result set is too large: an
    aggregate is a way forward, an error is a wall.

    Args:
        query: the *same* Mongo query that produced the count being explained,
            as returned in `query_entities()["_query"]`. Passing a rebuilt
            query makes the breakdown describe a different set of documents
            than the total it accompanies.
    """
    pipeline = [
        {"$match": query},
        {
            "$facet": {
                "by_layer": [
                    {"$group": {"_id": "$layer", "n": {"$sum": 1}}},
                    {"$sort": {"n": -1}},
                    {"$limit": _BREAKDOWN_LIMIT},
                ],
                "by_type": [
                    {"$group": {"_id": "$type", "n": {"$sum": 1}}},
                    {"$sort": {"n": -1}},
                    {"$limit": _BREAKDOWN_LIMIT},
                ],
            }
        },
    ]
    result = list(coll(COLL_ENTITIES).aggregate(pipeline))
    if not result:
        return {"by_layer": {}, "by_type": {}}
    facets = result[0]
    by_layer = facets.get("by_layer", [])
    by_type = facets.get("by_type", [])
    breakdown: dict[str, Any] = {
        "by_layer": {r["_id"]: r["n"] for r in by_layer},
        "by_type": {r["_id"]: r["n"] for r in by_type},
    }
    # Said out loud. The lists are capped at 25 and Janadriyah's model space
    # has 73 layers holding entities, so a caller reading this as the whole
    # picture is reading a quarter of it. The cap stays -- the long tail is
    # noise in a summary -- but silence about it does not.
    if len(by_layer) >= _BREAKDOWN_LIMIT or len(by_type) >= _BREAKDOWN_LIMIT:
        breakdown["truncated"] = True
        breakdown["truncated_note"] = (
            f"Only the top {_BREAKDOWN_LIMIT} are listed. For the complete "
            "picture use the distinct endpoint/tool: "
            "distinct_values(field='layer'|'type', layout=..., top=...)."
        )
    return breakdown


def spatial_query(
    drawing_id: str,
    *,
    bbox: Sequence[float],
    layout_name: str | None = None,
    layer: str | None = None,
    dxftype: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Entities whose bounding box overlaps `bbox = [minx, miny, maxx, maxy]`.

    Plain rectangle overlap on stored numbers, not a `2dsphere` index: these
    are local CAD coordinates (Janadriyah spans hundreds of metres, the
    samples are in inches), and `2dsphere` would interpret them as
    longitude/latitude and reject anything beyond ±180. See DECISIONS-LOG D-004.
    """
    _reject_empty_filters(layer=layer, type=dxftype, layout=layout_name)
    minx, miny, maxx, maxy = bbox
    query: dict[str, Any] = {
        "drawing_id": drawing_id,
        "bbox.min.0": {"$lte": maxx},
        "bbox.max.0": {"$gte": minx},
        "bbox.min.1": {"$lte": maxy},
        "bbox.max.1": {"$gte": miny},
    }
    # Scoping, added after a measured failure: "what is within 100 m of this
    # label" returned 437 across model space, three paper sheets and the block
    # definitions at once, where the answer meant for model space was 238.
    # Layouts do not share a coordinate frame in any meaningful sense, so
    # unioning them is not a broader answer, it is a wrong one.
    if layout_name:
        query["layout"] = layout_name
    if layer:
        query["layer"] = layer
    if dxftype:
        query["type"] = dxftype.upper()
    collection = coll(COLL_ENTITIES)
    total = collection.count_documents(query)
    rows = list(
        collection.find(
            query,
            {
                "handle": 1,
                "type": 1,
                "layer": 1,
                "text": 1,
                "bbox_centre": 1,
                "bbox": 1,
            },
        ).sort([("layer", 1), ("handle", 1)]).limit(limit)
    )
    result = {
        "drawing_id": drawing_id,
        "bbox": list(bbox),
        "scope": {
            k: v
            for k, v in (
                ("layout", layout_name),
                ("layer", layer),
                ("type", dxftype.upper() if dxftype else None),
            )
            if v
        }
        or "whole drawing: model space, every sheet and every block definition",
        "total_matches": total,
        "returned": len(rows),
        "truncated": total > len(rows),
        "entities": rows,
    }
    if total == 0 and (layout_name or layer or dxftype):
        result.update(
            diagnose_empty_query(
                drawing_id,
                layer=layer,
                dxftype=dxftype,
                layout_name=layout_name,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Explaining an empty result
# ---------------------------------------------------------------------------

#: DXF types people ask for by a name the file does not use.
#:
#: The one that cost a session: an agent was asked for the polylines on layer
#: `fram` and answered "there are none". It was telling the truth — `fram`
#: holds 1,100 LWPOLYLINE and exactly 0 POLYLINE — and it was useless, because
#: a drafter saying "polyline" means either. AutoCAD itself blurs them: the
#: PLINE command has produced LWPOLYLINE since R14, while the LISP name and
#: every conversation still say "polyline".
#:
#: Matching is NOT quietly widened, because "how many POLYLINE" has an exact
#: answer and changing it would break the questions that meant it literally.
#: The near-miss is reported instead, so the caller can ask again knowing why.
TYPE_NEAR_MISSES: dict[str, tuple[str, ...]] = {
    "POLYLINE": ("LWPOLYLINE", "POLYLINE2D", "POLYLINE3D"),
    "LWPOLYLINE": ("POLYLINE",),
    "TEXT": ("MTEXT", "ATTRIB", "ATTDEF"),
    "MTEXT": ("TEXT", "MULTILEADER"),
    "DIMENSION": ("ARC_DIMENSION", "LARGE_RADIAL_DIMENSION"),
    "BLOCK": ("INSERT",),
    "BLOCKREF": ("INSERT",),
    "LINE": ("LWPOLYLINE", "POLYLINE"),
    "RECTANGLE": ("LWPOLYLINE",),
    "CIRCLE": ("ARC", "ELLIPSE"),
    "LEADER": ("MULTILEADER",),
}


def diagnose_empty_query(
    drawing_id: str,
    *,
    layer: str | None = None,
    dxftype: str | None = None,
    layout_name: str | None = None,
    block_name: str | None = None,
) -> dict[str, Any]:
    """Explain a filter that matched nothing, instead of returning silence.

    Zero matches has at least four causes and they are indistinguishable from
    each other in the response: the filter value does not exist; it exists but
    is spelled with different capitals; it exists but not in the layout that
    was asked for; or it exists and the *type* asked for is a near-miss of the
    one the file uses. Every one of those looks exactly like "there is nothing
    there", which is the single most misleading answer this API can give — a
    caller believes it and reports absence as fact.

    So an empty result is explained. Everything here is read from the drawing
    document and from indexed aggregations over one drawing; it runs only when
    a filtered query already returned zero, so it costs nothing on the path
    that found something.
    """
    drawing = coll(COLL_DRAWINGS).find_one(
        {"_id": drawing_id},
        {"layers": 1, "layouts": 1, "counts_by_type": 1, "counts_by_layer": 1, "blocks": 1},
    )
    if not drawing:
        return {}

    notes: list[str] = []
    facts: dict[str, Any] = {}

    layer_names = [l.get("name", "") for l in drawing.get("layers", [])]
    layout_names = [l.get("name", "") for l in drawing.get("layouts", [])]
    counts_by_layer = drawing.get("counts_by_layer", {}) or {}
    counts_by_type = drawing.get("counts_by_type", {}) or {}
    block_names = [
        b.get("name") if isinstance(b, dict) else b
        for b in (drawing.get("blocks") or [])
    ]

    # --- the layer ---------------------------------------------------------
    if layer is not None:
        if layer in layer_names:
            total_on_layer = counts_by_layer.get(layer, 0)
            facts["layer_exists"] = True
            facts["entities_on_layer_all_layouts"] = total_on_layer
            if total_on_layer == 0:
                notes.append(
                    f"Layer {layer!r} is declared in the file but holds no entities."
                )
        else:
            facts["layer_exists"] = False
            close = [n for n in layer_names if n.lower() == layer.lower()]
            if close:
                notes.append(
                    f"No layer named {layer!r}, but {close[0]!r} exists — layer "
                    "matching is case-sensitive."
                )
                facts["layer_did_you_mean"] = close
            else:
                partial = [n for n in layer_names if layer.lower() in n.lower()][:5]
                if partial:
                    # The names go IN the sentence. They were being put in
                    # `why_empty_facts` while the note said only "no layer named
                    # 'Green'" -- so a drafter who wrote "hatch Green" was told
                    # the layer did not exist, and its 113 entities were never
                    # reported. Same lesson as D-068, unlearned in this branch.
                    facts["layer_did_you_mean"] = partial
                    notes.append(
                        f"This drawing has no layer named {layer!r}, but "
                        + ", ".join(repr(n) for n in partial)
                        + " contain(s) it. Layer names here often carry a "
                        "prefix or suffix the question leaves out."
                    )
                else:
                    notes.append(f"This drawing has no layer named {layer!r}.")

    # --- the layout --------------------------------------------------------
    if layout_name is not None and layout_name not in layout_names:
        close = [n for n in layout_names if n.lower() == layout_name.lower()]
        notes.append(
            f"This drawing has no layout named {layout_name!r}."
            + (f" Did you mean {close[0]!r}?" if close else "")
        )
        # Real layouts only, and say how many there are. This list is handed
        # over at the exact moment a caller is most likely to pick a name off
        # it -- straight after a typo -- and it was 78 block definitions
        # labelled "layouts" (D-070, fixed at one call site of three).
        real = [n for n in layout_names if not n.startswith(BLOCK_LAYOUT_PREFIX)]
        facts["layouts"] = real
        facts["layouts_note"] = (
            str(len(real)) + " layout(s). "
            + str(len(layout_names) - len(real))
            + " block definitions are stored alongside them and are not "
            "layouts; query them with layout='" + BLOCK_LAYOUT_PREFIX + "<name>'."
        )

    # --- the type ----------------------------------------------------------
    if dxftype is not None:
        wanted = dxftype.upper()
        facts["type_exists_in_drawing"] = wanted in counts_by_type
        near = [t for t in TYPE_NEAR_MISSES.get(wanted, ()) if t in counts_by_type]
        if not facts["type_exists_in_drawing"]:
            notes.append(f"No entity anywhere in this drawing has type {wanted!r}.")
        if near:
            facts["type_near_misses"] = {t: counts_by_type[t] for t in near}
            notes.append(
                f"The drawing does contain {', '.join(f'{t} ({counts_by_type[t]})' for t in near)}"
                f" — a different DXF type from {wanted!r}, but often what is meant by it."
            )

    # --- the block ---------------------------------------------------------
    # This branch was missing, and its absence was the one gap that mattered:
    # a misspelled block name returned zero with `why_empty` empty, so the tool
    # said nothing at all about why. "How many trees are there" answered with a
    # bare 0 reads as "there are no trees", which on this drawing is false --
    # the block is called TREE, Tree 6, Tree-Deciduous or Tree-Evergreen, and
    # which one you asked for decides the answer.
    block_exists = block_name is not None and block_name in block_names
    if block_name is not None and not block_exists:
        facts["block_exists"] = False
        close = [n for n in block_names if str(n).lower() == block_name.lower()]
        if close:
            notes.append(
                f"No block named {block_name!r}, but {close[0]!r} exists — block "
                "names are matched exactly, including case."
            )
            facts["block_did_you_mean"] = close
        else:
            partial = [
                n for n in block_names if block_name.lower() in str(n).lower()
            ][:5]
            if partial:
                facts["block_did_you_mean"] = partial
                notes.append(
                    f"This drawing has no block named {block_name!r}. It does have "
                    + ", ".join(repr(n) for n in partial)
                    + "."
                )
            else:
                notes.append(
                    f"This drawing has no block named {block_name!r}, and no block "
                    "name contains it."
                )
    elif block_exists:
        # The half D-069 left open, and the one that actually fires in practice.
        # A misspelled block is the rare case; a real block with no instance in
        # the layout you asked about is the ordinary one, and it returned zero
        # with nothing said. "How many trees are there" answered "none" on a
        # drawing holding four tree blocks.
        placed = _layouts_of_block(drawing_id, block_name)
        facts["block_exists"] = True
        facts["block_instances_by_layout"] = placed
        total_placed = sum(placed.values())
        if not placed:
            notes.append(
                f"Block {block_name!r} is defined in this drawing but never "
                "placed — there are no insertions of it anywhere, in any layout."
            )
        else:
            where = ", ".join(f"{name} ({n})" for name, n in placed.items())
            scope_text = (
                f" but none in {layout_name!r}" if layout_name else ""
            )
            notes.append(
                f"Block {block_name!r} exists and is placed {total_placed} "
                f"time(s){scope_text}. Its insertions are in: {where}."
            )
            siblings = [
                n for n in block_names
                if n != block_name
                and str(n).lower().startswith(str(block_name).lower()[:4])
            ][:5]
            if siblings:
                facts["related_blocks"] = siblings
                notes.append(
                    "Similarly named blocks exist and are counted separately: "
                    + ", ".join(repr(n) for n in siblings)
                    + ". A question about a kind of object usually spans all of "
                    "them."
                )

    # --- what IS there, where it was asked for -----------------------------
    # The most useful single fact for a caller that got nothing: the shape of
    # whatever *does* live at the place they looked.
    scope: dict[str, Any] = {"drawing_id": drawing_id}
    if layer:
        scope["layer"] = layer
    if layout_name:
        scope["layout"] = layout_name
    if block_name and block_exists:
        # Only when it exists. Leaving a nonexistent block in the scope makes
        # the "what IS there" lookup match nothing either, and the caller gets
        # silence from the very branch meant to break the silence.
        scope["block_name"] = block_name
    if len(scope) > 1:
        present = _types_and_layouts(scope)
        if present["by_type"]:
            facts["present_at_that_scope"] = present
            notes.append(
                "Dropping the filter that matched nothing, that scope holds: "
                + ", ".join(f"{t} ({n})" for t, n in present["by_type"].items())
            )
        elif layer and facts.get("layer_exists"):
            # The layer exists and has entities, but not in this layout.
            elsewhere = _layouts_of_layer(drawing_id, layer)
            if elsewhere:
                facts["layer_present_in_layouts"] = elsewhere
                notes.append(
                    f"Layer {layer!r} has no entities in this scope, but does in: "
                    + ", ".join(f"{k} ({v})" for k, v in elsewhere.items())
                )

    if not notes:
        return {}
    # One quotable sentence, for the same reason `measure` has `statement`
    # (D-066, D-068): a list of notes gets summarised, and the summary is where
    # the meaning is lost. Asked how many trees there were, an agent holding
    # four correct notes still opened with "there are no blocks named ...",
    # which is what the reader acts on. The first note is the finding; the rest
    # are context, and joining them puts the finding first.
    return {
        "why_empty": notes,
        "why_empty_statement": " ".join(notes),
        "why_empty_facts": facts,
    }


def _types_and_layouts(query: dict[str, Any]) -> dict[str, Any]:
    """Types and layouts present for a query, capped."""
    pipeline = [
        {"$match": query},
        {
            "$facet": {
                "by_type": [
                    {"$group": {"_id": "$type", "n": {"$sum": 1}}},
                    {"$sort": {"n": -1}},
                    {"$limit": 15},
                ],
                "by_layout": [
                    {"$group": {"_id": "$layout", "n": {"$sum": 1}}},
                    {"$sort": {"n": -1}},
                    {"$limit": 15},
                ],
            }
        },
    ]
    facets = next(iter(coll(COLL_ENTITIES).aggregate(pipeline)), {})
    return {
        "by_type": {r["_id"]: r["n"] for r in facets.get("by_type", [])},
        "by_layout": {r["_id"]: r["n"] for r in facets.get("by_layout", [])},
    }


def _layouts_of_layer(drawing_id: str, layer: str) -> dict[str, int]:
    pipeline = [
        {"$match": {"drawing_id": drawing_id, "layer": layer}},
        {"$group": {"_id": "$layout", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
        {"$limit": 15},
    ]
    return {r["_id"]: r["n"] for r in coll(COLL_ENTITIES).aggregate(pipeline)}


def _layouts_of_block(drawing_id: str, block_name: str) -> dict[str, int]:
    """Where a block is actually placed, and how often.

    The sibling of `_layouts_of_layer`, and it exists for the same reason: a
    filter that matched nothing is only explained once you can say where the
    thing it named does live.
    """
    pipeline = [
        {"$match": {"drawing_id": drawing_id, "block_name": block_name}},
        {"$group": {"_id": "$layout", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
        {"$limit": 15},
    ]
    return {r["_id"]: r["n"] for r in coll(COLL_ENTITIES).aggregate(pipeline)}


#: Fields whose DEFINED set is larger than the set that appears in the data.
#: A drawing declares layers and block definitions up front; how many of them
#: anything is actually drawn on is a separate question, and the gap between the
#: two is where several wrong answers have come from.
#:
#: `type` and `layout` are absent deliberately: a DXF type exists because an
#: entity has it, and a layout with nothing on it is not declared anywhere we
#: can read. There is no second half for those.
DECLARED_SET_FIELDS = frozenset({"block_name", "layer"})

#: How many absent values to name before the list truncates.
MAX_ABSENT_VALUES = 60


def _declared_but_absent(
    drawing_id: str,
    drawing: dict[str, Any],
    field: str,
    present: set[str],
) -> dict[str, Any] | None:
    """The half of a set that this query cannot show: defined, and not here.

    `distinct_values` groups the entities it can see, so it can only ever return
    values that appear. Everything a drawing declares and does not use is
    invisible to it -- and invisible in a way that reads as non-existent, because
    a complete-looking list came back.

    That gap has produced two separate wrong answers already. Asked how many
    objects of a kind a drawing held, an agent listed the ones with insertions
    and reported the rest as not existing; they were defined and never placed,
    which is a different fact and the one being asked about. Asked which layers
    were empty, another answer used the layers present in one layout and called
    the remainder empty across the drawing.

    So both halves travel together. D-058 again: an agent holding one number
    reports one number, and an agent holding two reports both. The fix is the
    shape of the response, not a firmer instruction -- there is no wording that
    makes a model mention a value it was never given.

    Deliberately general. `block_name` is what surfaced it; `layer` has the same
    shape and has already failed once. Any field whose declared set is knowable
    belongs in `DECLARED_SET_FIELDS` and needs no further code.
    """
    if field not in DECLARED_SET_FIELDS:
        return None

    if field == "block_name":
        declared = [
            b.get("name") if isinstance(b, dict) else b
            for b in (drawing.get("blocks") or [])
        ]
        # Where a block is inserted, and whether that insertion is itself drawn.
        # A definition placed only inside another definition that is never
        # placed is not on the drawing, and saying "it is used" would be false.
        rows = coll(COLL_ENTITIES).aggregate([
            {"$match": {"drawing_id": drawing_id, "type": "INSERT"}},
            {"$group": {"_id": "$block_name", "layouts": {"$addToSet": "$layout"},
                        "n": {"$sum": 1}}},
        ])
        placements = {r["_id"]: r for r in rows if r.get("_id")}
        reachable = _reachable_blocks(drawing_id)

        def elsewhere(name: str) -> tuple[bool, str | None]:
            row = placements.get(name)
            if not row:
                return False, "never inserted anywhere in this drawing"
            where = ", ".join(sorted(row["layouts"])[:3])
            if name in reachable:
                return True, f"inserted {row['n']} time(s), in {where}"
            return False, (
                f"inserted {row['n']} time(s) in {where}, but that block "
                "definition is itself never placed, so it is not drawn"
            )
    else:  # layer
        layers = drawing.get("layers") or []
        declared = [l.get("name") for l in layers if isinstance(l, dict)]
        counts = {
            l.get("name"): int(l.get("entity_count") or 0)
            for l in layers
            if isinstance(l, dict)
        }

        def elsewhere(name: str) -> tuple[bool, str | None]:
            n = counts.get(name, 0)
            if n:
                return True, f"holds {n} entities elsewhere in this drawing"
            return False, "declared in the file and holds nothing anywhere"

    absent = [n for n in declared if n and n not in present]
    if not absent:
        return {
            "count": 0,
            "values": [],
            "truncated": False,
            "note": (
                "Every " + field + " this drawing declares also appears in this "
                "scope, so the list above is the whole set."
            ),
        }

    rows_out = []
    for name in sorted(absent)[:MAX_ABSENT_VALUES]:
        appears, where = elsewhere(name)
        rows_out.append(
            {"value": name, "appears_elsewhere": appears, "where": where}
        )
    drawn = sum(1 for r in rows_out if r["appears_elsewhere"])

    return {
        "count": len(absent),
        "values": rows_out,
        "truncated": len(absent) > MAX_ABSENT_VALUES,
        "note": (
            str(len(absent)) + " " + field + " value(s) are DEFINED by this "
            "drawing and do not appear in this scope. They are not absent from "
            "the drawing -- " + str(drawn) + " of the "
            + str(len(rows_out)) + " listed are drawn somewhere else, and the "
            "rest are declared and never used. An answer about what this "
            "drawing contains needs both halves; the list above alone is the "
            "used half."
        ),
    }


def distinct_values(
    drawing_id: str,
    field: str,
    *,
    layer: str | None = None,
    layout_name: str | None = None,
    dxftype: str | None = None,
    top: int = 25,
) -> dict[str, Any]:
    """How many distinct values a field takes, and the commonest ones.

    Exists because the alternative was 56 paged calls. "How many different
    text strings are in model space" is a normal question about a drawing and
    there was no way to answer it except by fetching every row and counting in
    the caller — around 1.1 million characters of tool output to produce one
    integer.

    Args:
        drawing_id: which drawing.
        field: `text`, `layer`, `type`, `layout` or `block_name`.
        layer, layout_name, dxftype: narrow the population first.
        top: how many of the commonest values to list alongside the count.
    """
    _reject_empty_filters(
        field=field, layer=layer, layout=layout_name, type=dxftype,
    )
    if field in STRUCTURAL_FIELDS:
        top = MAX_STRUCTURAL_VALUES
    if field not in DISTINCTABLE_FIELDS:
        raise ValueError(
            f"{field!r} is not a countable field; expected one of "
            f"{sorted(DISTINCTABLE_FIELDS)}"
        )
    query: dict[str, Any] = {"drawing_id": drawing_id}
    if layer:
        query["layer"] = layer
    if layout_name:
        query["layout"] = layout_name
    if dxftype:
        query["type"] = dxftype.upper()

    pipeline = [
        {"$match": query},
        # Null and empty are not values. Counting them as one inflates every
        # answer by exactly one and nobody notices.
        {"$match": {field: {"$nin": [None, ""]}}},
        {"$group": {"_id": f"${field}", "n": {"$sum": 1}}},
        {
            "$facet": {
                "distinct": [{"$count": "n"}],
                "total": [{"$group": {"_id": None, "n": {"$sum": "$n"}}}],
                "top": [{"$sort": {"n": -1}}, {"$limit": max(top, 0)}],
            }
        },
    ]
    facets = next(iter(coll(COLL_ENTITIES).aggregate(pipeline)), {})
    distinct = (facets.get("distinct") or [{"n": 0}])[0]["n"]
    total = (facets.get("total") or [{"n": 0}])[0]["n"]

    # The other half of the set, in the same response. See
    # `_declared_but_absent`: grouping entities can only ever return values that
    # appear, and the ones a drawing declares and never uses are invisible in a
    # way that reads as non-existent.
    absent = _declared_but_absent(
        drawing_id,
        get_drawing(drawing_id) or {},
        field,
        {r["_id"] for r in facets.get("top", []) if r.get("_id")},
    )

    return {
        "drawing_id": drawing_id,
        "field": field,
        "scope": {k: v for k, v in query.items() if k != "drawing_id"},
        "distinct_values": distinct,
        "entities_with_a_value": total,
        "most_common": [
            {"value": _clip(r["_id"], 120), "count": r["n"]}
            for r in facets.get("top", [])
        ],
        # A ranked list that does not say it is ranked, or that it stops, gets
        # read as the whole population. Asked to audit 292 layer names against
        # a naming convention, an answer audited the 20 it was given and drew a
        # conclusion about the drawing.
        # "genuinely absent" is true of the SCOPE and was read as true of the
        # drawing. With layout="Model", `00_Internal Road` is absent from this
        # list and holds 2,557 entities inside a block definition. The claim now
        # carries the scope it is a claim about.
        "most_common_list_is": (
            (
                "every value in this scope ("
                + (
                    ", ".join(
                        k + "=" + str(v)
                        for k, v in query.items()
                        if k != "drawing_id"
                    )
                    or "the whole drawing"
                )
                + ") -- complete, so a value missing from it is absent FROM "
                "THIS SCOPE. A layout-scoped list does not see values that "
                "exist only inside block definitions; widen the scope before "
                "calling anything absent from the drawing."
            )
            if field in STRUCTURAL_FIELDS
            and distinct <= len(facets.get("top", []))
            else "the commonest values only; absence from this list proves "
            "nothing"
        ),
        **({"defined_but_not_present": absent} if absent else {}),
        "most_common_order": (
            "by count, descending"
            + (
                ", truncated to the top " + str(len(facets.get("top", [])))
                + " of " + str(distinct) + " distinct values -- the rest are "
                "not shown and must not be described as absent"
                if distinct > len(facets.get("top", []))
                else ", complete"
            )
        ),
    }


#: Fields worth counting distinct values of. Restricted rather than open so a
#: caller cannot make the database group by an unindexed nested structure.
DISTINCTABLE_FIELDS = frozenset({"text", "layer", "type", "layout", "block_name"})

#: Fields whose values are the drawing's own vocabulary rather than its content.
#: These are returned COMPLETE, because a truncated vocabulary gets read as the
#: whole of it. Asked for school plot areas, an agent called
#: distinct_values(field="layer"), did not find `Primary School` in the top 25
#: of 82, and reported that the layer did not exist. It holds two entities. The
#: response carried a line saying the rest "must not be described as absent" and
#: that was not enough -- so the truncation is removed instead of explained.
#:
#: `text` keeps its cap: 2,861 distinct strings on this drawing is a different
#: kind of list, and nobody asks "does this exact string exist" of it.
STRUCTURAL_FIELDS = frozenset({"layer", "type", "layout", "block_name"})

#: Ceiling on a complete structural listing, so a pathological file cannot
#: return tens of thousands of names. Above this the list truncates and says so.
MAX_STRUCTURAL_VALUES = 500


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------

#: How many handles one selection may carry back. A selection is a *pointer*
#: to work, not the work itself: the summary is what a human reads and what
#: the agent reasons over, and the handle list only exists so the viewer can
#: highlight and so `describe_selection` can be asked for more. 5,000 keeps
#: the JSON around 60 KB and matches the cap on the MCP tool, so the browser
#: and the agent can never disagree about what a selection contains.
#: Enumeration ceiling: how many handles a region query will NAME.
#:
#: It says nothing about how many it COUNTS. `total`, `by_layer` and
#: `by_type` come from `count_documents` and aggregation, and are exact at any
#: size -- a region matching 40,000 objects reports 40,000.
#:
#: This used to be 5,000, sized when the handle list was pasted into the
#: model's prompt and every handle cost context. Since D-046 only a
#: selection_id travels, so the cap now bounds one HTTP response and one Mongo
#: document, and there is no reason for it to differ from what a selection can
#: hold. Raised to match, so the ceiling is one number rather than three that
#: quietly disagree.
MAX_SELECTION_HANDLES = 25_000

#: Ceiling on the rows a polygon selection may pull out of MongoDB before the
#: exact test runs in Python. The indexed bounding-box pre-filter usually
#: leaves tens or hundreds; a polygon drawn around a whole site plan can leave
#: tens of thousands, and refining those in a request thread is how an API
#: stops answering. Above this the caller is told to narrow the region rather
#: than being made to wait for an answer nobody wants.
MAX_POLYGON_CANDIDATES = 60_000

#: Distinct (layer, type) pairs listed in a selection summary. Janadriyah's
#: whole modelspace has 218; a cap keeps a pathological drawing from turning
#: the summary into the wall of data the summary exists to prevent.
MAX_SELECTION_GROUPS = 300

#: Names reported per category when a text search finds nothing. Enough to
#: show the pattern -- five school layers read as "the schools are on layers"
#: -- without turning an empty result into a wall of names.
_NEAR_MISS_LIMIT = 12


class TooManyCandidates(RuntimeError):
    """A polygon region matched more entities than can be refined in-request."""

    def __init__(self, candidates: int, limit: int) -> None:
        super().__init__(
            f"{candidates} entities fall in the polygon's bounding box, "
            f"above the {limit} that can be tested exactly in one request"
        )
        self.candidates = candidates
        self.limit = limit


def _rect_query(
    drawing_id: str,
    box: region.Box,
    *,
    mode: str,
    layout: str | None,
) -> dict[str, Any]:
    """The Mongo filter for a rectangular region.

    Both modes are expressible as plain range comparisons on the stored
    bounding box, so a rectangle never leaves the database — no candidate set
    is materialised and no Python loop runs. Only the polygon path needs that.

    Note the asymmetry between the two, which is the whole point of the
    distinction: *crossing* compares each side of the entity against the
    opposite side of the region (they merely have to reach each other),
    *window* compares like against like (the entity has to stay inside).
    """
    minx, miny, maxx, maxy = box
    query: dict[str, Any] = {"drawing_id": drawing_id}
    if layout:
        query["layout"] = layout
    if mode == "window":
        query.update(
            {
                "bbox.min.0": {"$gte": minx},
                "bbox.min.1": {"$gte": miny},
                "bbox.max.0": {"$lte": maxx},
                "bbox.max.1": {"$lte": maxy},
            }
        )
    else:
        query.update(
            {
                "bbox.min.0": {"$lte": maxx},
                "bbox.max.0": {"$gte": minx},
                "bbox.min.1": {"$lte": maxy},
                "bbox.max.1": {"$gte": miny},
            }
        )
    return query


def _summarise(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counts per layer, per type and per (layer, type) from in-memory rows.

    Used by the polygon path, where the surviving set is only known after the
    exact test. The rectangle path gets the same shape out of an aggregation
    instead, so both return identical structures and the API has one contract.
    """
    by_layer: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_group: Counter[tuple[str, str]] = Counter()
    for row in rows:
        layer = row.get("layer") or "(none)"
        dxftype = row.get("type") or "(unknown)"
        by_layer[layer] += 1
        by_type[dxftype] += 1
        by_group[(layer, dxftype)] += 1
    return _shape_summary(by_layer, by_type, by_group)


def _shape_summary(
    by_layer: Counter[str],
    by_type: Counter[str],
    by_group: Counter[tuple[str, str]],
) -> dict[str, Any]:
    groups = [
        {"layer": layer, "type": dxftype, "count": count}
        for (layer, dxftype), count in by_group.most_common(MAX_SELECTION_GROUPS)
    ]
    return {
        "by_layer": [
            {"name": name, "count": count} for name, count in by_layer.most_common()
        ],
        "by_type": [
            {"name": name, "count": count} for name, count in by_type.most_common()
        ],
        "groups": groups,
        "groups_truncated": len(by_group) > len(groups),
    }


def _extents_of(boxes: Iterable[region.Box]) -> dict[str, list[float]] | None:
    lows_x, lows_y, highs_x, highs_y = [], [], [], []
    for minx, miny, maxx, maxy in boxes:
        lows_x.append(minx)
        lows_y.append(miny)
        highs_x.append(maxx)
        highs_y.append(maxy)
    if not lows_x:
        return None
    return {
        "min": [min(lows_x), min(lows_y)],
        "max": [max(highs_x), max(highs_y)],
    }


def select_region(
    drawing_id: str,
    *,
    points: Sequence[region.Point],
    kind: str,
    mode: str,
    layout: str | None = None,
    max_handles: int = MAX_SELECTION_HANDLES,
) -> dict[str, Any]:
    """Entities inside (or touching) a region, summarised before listed.

    The response leads with counts and only then carries handles, because a
    region drawn over a site plan routinely matches thousands of objects and
    a raw list of those is unusable by a person and unaffordable for a model.

    Args:
        drawing_id: which drawing.
        points: region vertices **in drawing coordinates**. Two opposite
            corners for `kind="rect"`, three or more vertices for
            `kind="polygon"`.
        kind: `"rect"` or `"polygon"`.
        mode: `"window"` (entity wholly inside) or `"crossing"` (entity
            touches), matching AutoCAD's two selection modes.
        layout: restrict to one layout. The viewer always passes the layout
            it is showing — without it a region drawn on a paper sheet also
            matches modelspace geometry that happens to share those
            coordinates, and the count is nonsense.
        max_handles: cap on the returned handle list.

    Raises:
        TooManyCandidates: a polygon or circle whose bounding box holds more
            entities than can be refined in one request.
    """
    box = region.shape_bbox(kind, points)
    collection = coll(COLL_ENTITIES)

    if kind == "rect":
        query = _rect_query(drawing_id, box, mode=mode, layout=layout)
        total = collection.count_documents(query)
        summary = _aggregate_summary(collection, query)
        rows = list(
            collection.find(query, {"handle": 1})
            .sort([("handle", 1)])
            .limit(max_handles)
        )
        handles = [r["handle"] for r in rows]
        extents = summary.pop("_extents", None)
        no_bbox = 0
    else:
        # Indexed pre-filter first: only entities whose bounding box reaches
        # the polygon's bounding box can possibly qualify, and that test is
        # the one MongoDB can do with an index.
        prefilter = _rect_query(drawing_id, box, mode="crossing", layout=layout)
        candidates = collection.count_documents(prefilter)
        if candidates > MAX_POLYGON_CANDIDATES:
            raise TooManyCandidates(candidates, MAX_POLYGON_CANDIDATES)

        inside = region.box_in_shape(kind, mode, points)
        kept: list[dict[str, Any]] = []
        boxes: list[region.Box] = []
        no_bbox = 0
        for row in collection.find(
            prefilter, {"handle": 1, "layer": 1, "type": 1, "bbox": 1}
        ):
            entity = region.entity_box(row.get("bbox"))
            if entity is None:
                no_bbox += 1
                continue
            if inside(entity):
                kept.append(row)
                boxes.append(entity)
        total = len(kept)
        summary = _summarise(kept)
        handles = [r["handle"] for r in kept[:max_handles]]
        extents = _extents_of(boxes)

    result: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "kind": kind,
        "mode": mode,
        "region": [list(p) for p in points],
        "region_bbox": list(box),
        "total": total,
        "handles": handles,
        "handles_truncated": total > len(handles),
        "extents": extents,
        # Said out loud rather than implied: selection compares stored
        # bounding boxes, not geometry. See region.py.
        "basis": "bounding-box",
    }
    result.update(summary)
    if no_bbox:
        result["without_bbox"] = no_bbox
    return result


def select_region_rows(
    drawing_id: str,
    *,
    points: Sequence[region.Point],
    kind: str,
    mode: str,
    layer: str,
    dxftype: str,
    layout: str | None = None,
    offset: int = 0,
    limit: int = 200,
) -> dict[str, Any]:
    """One (layer, type) group of a region, as rows, paginated.

    The other half of "summary first": the summary says a region holds 1,284
    objects across 12 layers, and this is what runs when a person opens one of
    those groups. Nothing here is fetched until they do.

    The region is re-evaluated rather than filtered from a previously returned
    handle list. That list is capped, so filtering it would quietly return a
    short group and call it complete.
    """
    box = region.shape_bbox(kind, points)
    collection = coll(COLL_ENTITIES)
    projection = {
        "handle": 1,
        "type": 1,
        "layer": 1,
        "text": 1,
        "bbox_centre": 1,
    }

    if kind == "rect":
        query = _rect_query(drawing_id, box, mode=mode, layout=layout)
        query["layer"] = layer
        query["type"] = dxftype
        total = collection.count_documents(query)
        rows = list(
            collection.find(query, projection)
            .sort([("layer", 1), ("handle", 1)])
            .skip(offset)
            .limit(limit)
        )
    else:
        prefilter = _rect_query(drawing_id, box, mode="crossing", layout=layout)
        prefilter["layer"] = layer
        prefilter["type"] = dxftype
        candidates = collection.count_documents(prefilter)
        if candidates > MAX_POLYGON_CANDIDATES:
            raise TooManyCandidates(candidates, MAX_POLYGON_CANDIDATES)
        inside = region.box_in_shape(kind, mode, points)
        kept = [
            row
            for row in collection.find(prefilter, {**projection, "bbox": 1})
            if (entity := region.entity_box(row.get("bbox"))) is not None
            and inside(entity)
        ]
        total = len(kept)
        rows = [
            {k: v for k, v in row.items() if k != "bbox"}
            for row in kept[offset : offset + limit]
        ]

    for row in rows:
        row.pop("_id", None)
        if row.get("text"):
            row["text"] = _clip(row["text"])
    return {
        "drawing_id": drawing_id,
        "layer": layer,
        "type": dxftype,
        "total": total,
        "returned": len(rows),
        "offset": offset,
        "truncated": total > offset + len(rows),
        "entities": rows,
    }


def _aggregate_summary(collection, query: dict[str, Any]) -> dict[str, Any]:
    """Per-layer / per-type / per-group counts and extents, computed in Mongo.

    One aggregation rather than four, and never a second pass over the rows:
    the rectangle path can match tens of thousands of entities, and pulling
    those into Python only to count them is the difference between a
    selection that answers instantly and one that does not.
    """
    pipeline = [
        {"$match": query},
        {
            "$facet": {
                "groups": [
                    {
                        "$group": {
                            "_id": {"layer": "$layer", "type": "$type"},
                            "n": {"$sum": 1},
                        }
                    },
                    {"$sort": {"n": -1}},
                    # One more than the cap, so "was it truncated?" is
                    # answerable without a second count.
                    {"$limit": MAX_SELECTION_GROUPS + 1},
                ],
                "layers": [
                    {"$group": {"_id": "$layer", "n": {"$sum": 1}}},
                    {"$sort": {"n": -1}},
                ],
                "types": [
                    {"$group": {"_id": "$type", "n": {"$sum": 1}}},
                    {"$sort": {"n": -1}},
                ],
                "extents": [
                    {
                        "$group": {
                            "_id": None,
                            "minx": {"$min": {"$arrayElemAt": ["$bbox.min", 0]}},
                            "miny": {"$min": {"$arrayElemAt": ["$bbox.min", 1]}},
                            "maxx": {"$max": {"$arrayElemAt": ["$bbox.max", 0]}},
                            "maxy": {"$max": {"$arrayElemAt": ["$bbox.max", 1]}},
                        }
                    }
                ],
            }
        },
    ]
    facets = next(iter(collection.aggregate(pipeline)), {})

    raw_groups = facets.get("groups", [])
    groups = [
        {
            "layer": g["_id"].get("layer") or "(none)",
            "type": g["_id"].get("type") or "(unknown)",
            "count": g["n"],
        }
        for g in raw_groups[:MAX_SELECTION_GROUPS]
    ]
    summary: dict[str, Any] = {
        "by_layer": [
            {"name": r["_id"] or "(none)", "count": r["n"]}
            for r in facets.get("layers", [])
        ],
        "by_type": [
            {"name": r["_id"] or "(unknown)", "count": r["n"]}
            for r in facets.get("types", [])
        ],
        "groups": groups,
        "groups_truncated": len(raw_groups) > MAX_SELECTION_GROUPS,
    }

    extent_rows = facets.get("extents", [])
    if extent_rows and extent_rows[0].get("minx") is not None:
        row = extent_rows[0]
        summary["_extents"] = {
            "min": [row["minx"], row["miny"]],
            "max": [row["maxx"], row["maxy"]],
        }
    else:
        summary["_extents"] = None
    return summary


#: Handles one stored selection may hold.
#:
#: Far above the 5,000 that used to be inlined into a prompt, and that is the
#: point of storing them at all: the agent can now describe a selection of any
#: realistic size instead of apologising for holding a subset of it. 25,000
#: covers a window over Janadriyah's whole modelspace (20,325) with room to
#: spare, and the document stays a couple of hundred kilobytes.
#: Handles one stored selection may hold. Equal to MAX_SELECTION_HANDLES by
#: construction -- a region that can name 25,000 must be storable whole.
MAX_STORED_SELECTION = 25_000


def save_selection(
    drawing_id: str,
    handles: Sequence[str],
    *,
    layout: str | None,
    mode: str | None,
    kind: str | None,
    summary: str | None,
    counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store a selection and return a reference to it.

    Why this exists, stated plainly because it reverses an earlier design:
    the selection used to travel to the agent as a literal list of handles
    inside the prompt -- about 16 KB re-sent with every single message of a
    conversation, and capped at 2,000, so any larger selection was answered
    with "I am working with 2000 handles out of your selection of 3,573".
    That contradicted this project's own rule that *the model receives access,
    not data*. A selection is now written once and passed by id; the tools
    read it server-side, whole.

    The document is deliberately small and derived: handles plus the summary
    the viewer already computed. No entity data is copied -- that lives in
    `autocad_entities` and would go stale here.
    """
    kept = list(dict.fromkeys(h for h in handles if h))[:MAX_STORED_SELECTION]
    doc = {
        "_id": uuid.uuid4().hex[:16],
        "drawing_id": drawing_id,
        "layout": layout,
        "mode": mode,
        "kind": kind,
        "summary": summary,
        "handles": kept,
        "total": len(kept),
        # The EXACT figures, as the region reported them, independent of how
        # many handles fit. A selection larger than the enumeration ceiling
        # can still answer "how many, on which layers, of which types"
        # exactly; only "name every one of them" has a limit, and no answer
        # was ever going to name 40,000 objects anyway.
        "counts": counts,
        "created_at": _utcnow(),
    }
    coll(COLL_SELECTIONS).insert_one(doc)
    exact_total = (counts or {}).get("total")
    return {
        "selection_id": doc["_id"],
        "drawing_id": drawing_id,
        # `.get(key, default)` returns a stored None rather than the default,
        # and `counts` carries total=None whenever the caller sent a breakdown
        # without one. Tested for a real int instead.
        "total": exact_total if isinstance(exact_total, int) else doc["total"],
        "enumerated": doc["total"],
        "truncated": len(handles) > len(kept),
        "expires_in_seconds": SELECTION_TTL_SECONDS,
    }


def get_selection(drawing_id: str, selection_id: str) -> dict[str, Any] | None:
    """One stored selection, scoped to its drawing.

    Scoped on purpose: a selection id from another drawing must not resolve
    here, or an agent could describe one drawing while naming another.
    """
    return coll(COLL_SELECTIONS).find_one(
        {"_id": selection_id, "drawing_id": drawing_id}
    )


def describe_handles(
    drawing_id: str,
    handles: Sequence[str],
    *,
    sample: int = 10,
    exact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate description of an explicit set of handles.

    This is what a model is given instead of the entities themselves. The
    principle the whole agent design rests on: *the model does not receive the
    data, it receives access*. A summary plus the ability to call `get_entity`
    on anything interesting beats any amount of pre-digested rows, and it
    keeps every claim traceable to a handle.

    Args:
        drawing_id: which drawing.
        handles: the handles to describe. Order is irrelevant; duplicates are
            collapsed.
        sample: how many example rows to include per (layer, type) group —
            enough for the model to see what the group *is* without receiving
            the group.

    Returns counts per layer and per type, total length and area where the
    entities carry them, the combined extents, a small sample of rows, and the
    handles that were asked for but do not exist in this drawing.
    """
    wanted = list(dict.fromkeys(handles))
    collection = coll(COLL_ENTITIES)
    rows = list(
        collection.find(
            # Queried by _id, which is the primary key: this is the one
            # lookup shape that stays fast whatever the drawing's size, and
            # it needs no index of its own.
            {"_id": {"$in": [f"{drawing_id}:{h}" for h in wanted]}},
            {
                "handle": 1,
                "type": 1,
                "layer": 1,
                "layout": 1,
                "block_name": 1,
                "text": 1,
                "bbox": 1,
                "bbox_centre": 1,
                "length": 1,
                "area": 1,
            },
        )
    )

    found = {row["handle"] for row in rows}
    by_layer: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_group: Counter[tuple[str, str]] = Counter()
    by_layout: Counter[str] = Counter()
    boxes: list[region.Box] = []
    total_length = 0.0
    total_area = 0.0
    with_length = 0
    with_area = 0
    # Per layer, the same three figures the selection already reports for the
    # whole set.
    #
    # Measured: asked how much residential area a region holds, the agent got
    # the count right -- 2,070 plots -- and then had to say that a total for
    # only those plots "cannot be directly computed with the currently
    # available tools", because the selection published one area for everything
    # in it. That is the question a reviewer actually asks about a region, and
    # the numbers were already being summed one row at a time to produce the
    # total. Splitting the accumulator by layer costs one dictionary.
    area_by_layer: dict[str, float] = {}
    length_by_layer: dict[str, float] = {}
    measured_by_layer: dict[str, int] = {}
    samples: dict[tuple[str, str], list[dict[str, Any]]] = {}
    texts: list[str] = []

    for row in rows:
        layer = row.get("layer") or "(none)"
        dxftype = row.get("type") or "(unknown)"
        by_layer[layer] += 1
        by_type[dxftype] += 1
        by_group[(layer, dxftype)] += 1
        by_layout[row.get("layout") or "(none)"] += 1

        box = region.entity_box(row.get("bbox"))
        if box is not None:
            boxes.append(box)
        if isinstance(row.get("length"), (int, float)):
            total_length += float(row["length"])
            with_length += 1
            length_by_layer[layer] = length_by_layer.get(layer, 0.0) + float(
                row["length"]
            )
        if isinstance(row.get("area"), (int, float)):
            total_area += float(row["area"])
            with_area += 1
            area_by_layer[layer] = area_by_layer.get(layer, 0.0) + float(row["area"])
            measured_by_layer[layer] = measured_by_layer.get(layer, 0) + 1

        bucket = samples.setdefault((layer, dxftype), [])
        if len(bucket) < sample:
            bucket.append(
                {
                    "handle": row.get("handle"),
                    "type": dxftype,
                    "layer": layer,
                    "text": _clip(row.get("text")),
                    "bbox_centre": [round(c, 3) for c in row["bbox_centre"]]
                    if row.get("bbox_centre")
                    else None,
                }
            )
        if row.get("text"):
            texts.append(str(row["text"]))

    # Comments on the selected objects — the one relation in this database
    # that turns a selection from geometry into review state.
    #
    # `autocad_comments` is keyed by (drawing_id, entity_handle), the same
    # compound index the viewer uses, so this is one indexed lookup rather
    # than a scan. It matters because the question this project exists to
    # answer is "has this comment been dealt with?", and an agent that can
    # describe a selection's geometry but not its review history can only
    # ever answer half of it.
    commented: dict[str, list[str]] = {}
    if found:
        for note in coll(COLL_COMMENTS).find(
            {"drawing_id": drawing_id, "entity_handle": {"$in": list(found)}},
            {"entity_handle": 1, "body": 1, "author": 1},
        ):
            commented.setdefault(note["entity_handle"], []).append(
                f"{note.get('author', '?')}: {_clip(note.get('body'), 120)}"
            )

    summary = _shape_summary(by_layer, by_type, by_group)
    # Units travel WITH the measurements, not in a separate call.
    #
    # Without this the tool reported "total length 5,965.078" and a model,
    # told to state units and given none, said the drawing does not declare
    # any. It declares metres. A measurement whose units the reader has to
    # guess is not a measurement, and the guess was wrong in the one direction
    # that sounds careful.
    drawing = coll(COLL_DRAWINGS).find_one(
        {"_id": drawing_id}, {"units_name": 1, "units_code": 1}
    ) or {}

    # Units are read off the rows' own layouts, not off the drawing header.
    # The header describes model space; a region dragged on a sheet is page
    # geometry and carries no real-world unit at all.
    selection_units = _unit_names_for_layouts(drawing, list(by_layout))

    result: dict[str, Any] = {
        "drawing_id": drawing_id,
        "units": selection_units["name"],
        "units_declared_in_file": selection_units["declared_in_file"],
        "unit_detail": selection_units,
        "requested": len(wanted),
        "found": len(rows),
        "missing_handles": [h for h in wanted if h not in found][:20],
        "by_layout": [
            {"name": name, "count": count} for name, count in by_layout.most_common()
        ],
        "extents": _extents_of(boxes),
        "total_length": round(total_length, 3) if with_length else None,
        "entities_with_length": with_length,
        "total_area": round(total_area, 3) if with_area else None,
        "entities_with_area": with_area,
        "by_layer_measured": [
            {
                "layer": layer,
                "entities_with_area": measured_by_layer.get(layer, 0),
                "total_area": round(area, 3),
                "total_length": (
                    round(length_by_layer[layer], 3)
                    if layer in length_by_layer
                    else None
                ),
            }
            for layer, area in sorted(
                area_by_layer.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ],
        "by_layer_measured_note": (
            "the same three figures as above, split by layer, so a question "
            "about one land use inside this selection can be answered without "
            "fetching the entities. `entities_with_area` is how many rows on "
            "that layer carried a measurable area -- it is NOT the number of "
            "objects on the layer, which is in `by_layer`, and the two differ "
            "wherever a ring could not be measured"
        ),
        "by_layer_length_only": [
            {"layer": layer, "total_length": round(length, 3)}
            for layer, length in sorted(
                length_by_layer.items(), key=lambda kv: (-kv[1], kv[0])
            )
            if layer not in area_by_layer
        ],
        "totals_warning": (
            None
            if selection_units["space"] == "model"
            else selection_units.get("why_no_unit")
        ),
        "sample_by_group": [
            {"layer": layer, "type": dxftype, "rows": rows_}
            for (layer, dxftype), rows_ in list(samples.items())[:MAX_SELECTION_GROUPS]
        ],
        "distinct_text_count": len(set(texts)),
        "objects_with_comments": len(commented),
        "comments": [
            {"handle": handle, "notes": notes}
            for handle, notes in list(commented.items())[:20]
        ],
    }
    result.update(summary)

    # `exact` is what the region actually matched, which can exceed what the
    # handle list can name. Where it is given it wins, because the figures
    # computed above are only as complete as the handles they were computed
    # from -- and an answer that says 5,000 when the truth is 8,122 is the
    # failure this project exists to prevent.
    if exact:
        enumerated = result["found"]
        total = exact.get("total")
        if isinstance(total, int) and total >= 0:
            result["selection_total"] = total
            result["enumerated"] = enumerated
            result["complete"] = enumerated >= total
        for key in ("by_layer", "by_type"):
            rows = exact.get(key)
            if rows:
                result[key] = rows
    return result


def _clip(text: Any, limit: int = 80) -> str | None:
    if not text:
        return None
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def search_text(drawing_id: str, query: str, limit: int = 20) -> dict[str, Any]:
    """Case-insensitive substring search over entity text.

    A regex scan rather than the `$text` index: `$text` matches whole words
    only, and CAD annotations are full of fragments like `A-101` or `FFL+2.40`
    where a word-boundary match returns nothing and looks like absence of data.
    The scan is bounded by `drawing_id`, which is indexed.
    """
    _reject_empty_filters(query=query)
    mongo_query = {
        "drawing_id": drawing_id,
        "text": {"$regex": re.escape(query), "$options": "i"},
    }
    collection = coll(COLL_ENTITIES)
    total = collection.count_documents(mongo_query)
    # Sorted, and the order named. An unordered `.find().limit(n)` hands back
    # rows in whatever order the storage engine chooses, so "3 of 4,086" answers
    # a question nobody asked and does not reproduce between calls. Layer then
    # handle is arbitrary but stable, which is the property that matters.
    rows = list(
        collection.find(
            mongo_query,
            {"handle": 1, "type": 1, "layer": 1, "text": 1, "bbox_centre": 1},
        ).sort([("layer", 1), ("handle", 1)]).limit(limit)
    )
    result: dict[str, Any] = {
        "drawing_id": drawing_id,
        "query": query,
        "total_matches": total,
        "returned": len(rows),
        "truncated": total > len(rows),
        "entities": rows,
        "entities_order": (
            "by layer, then handle -- stable between calls, and NOT by relevance,"
            " position or size. With " + str(total) + " matches and "
            + str(len(rows)) + " returned, these are not 'the most important'"
            " ones; narrow the query to choose which you see."
        ),
    }
    result["matched_on"] = "the stored text"
    if any(ch >= "؀" and ch <= "ۿ" for ch in query):
        result.update(_search_arabic(drawing_id, query, limit, result))
    if result["total_matches"] == 0:
        result.update(_text_near_misses(drawing_id, query))
        result.update(_query_is_a_handle(drawing_id, query))
    return result


def _query_is_a_handle(drawing_id: str, query: str) -> dict[str, Any]:
    """Was the caller searching for an entity HANDLE as though it were text?

    Measured: asked which facilities lie within 300 m of plot `205D12B`, the
    agent ran `search_text` for that string, found nothing, tried it as a plot
    NUMBER -- `20512`, the digits with the letters dropped -- found nothing
    again, and reported that the plot could not be located. The entity is in
    the drawing. It is the one every combination question on this drawing
    returns, and `get_entity` fetches it in one call.

    A handle is the object's identity in the DXF, not a word written on the
    sheet, so a text search can never find one. Anybody who reads DXF will
    paste handles, which makes this a question the tool should expect rather
    than a mistake it should punish.

    Checked against the store, never inferred from the shape of the string: the
    hint is published only when an entity with that handle really is there, so
    it can be acted on without being verified first.
    """
    candidate = query.strip().upper()
    if not (2 <= len(candidate) <= 12):
        return {}
    if not all(ch in "0123456789ABCDEF" for ch in candidate):
        return {}
    row = coll(COLL_ENTITIES).find_one(
        {"drawing_id": drawing_id, "handle": candidate},
        {"handle": 1, "type": 1, "layer": 1, "layout": 1},
    )
    if not row:
        return {}
    return {
        "looks_like_a_handle": {
            "handle": candidate,
            "found": True,
            "type": row.get("type"),
            "layer": row.get("layer"),
            "layout": row.get("layout"),
            "why_the_text_search_missed_it": (
                "a handle is the entity's identity inside the DXF, not text "
                "written on the sheet, so a text search cannot match it however "
                "the drawing is annotated"
            ),
            "ask_instead": (
                "get_entity with this handle returns the object itself; "
                "distance_matrix and proximity_count both take handles "
                "directly, so a question about what lies near it needs no text "
                "search at all"
            ),
        }
    }


def _search_arabic(
    drawing_id: str, query: str, limit: int, so_far: dict[str, Any]
) -> dict[str, Any]:
    """Find an Arabic word through its SHX shape — UPLIFT-04.

    A drawing typed in an Arabic SHX font contains not one Arabic letter.
    Searching `مسجد` in the stored text returns zero, and that zero is right
    about the bytes while being entirely wrong about the drawing: the mosque
    is there, and it is written `ls{]`.

    **The grade of evidence differs and is stated.** "Matched on the stored
    text" and "matched on a conjectured reading" are two different claims, and
    equating them means reporting a guess in the tone of a reading.
    """
    forms = shx.shx_forms(query)
    if not forms:
        return {
            "arabic_search": {
                "attempted": True,
                "forms_tried": 0,
                "not_measured": (
                    "this word contains letters that are not in any recognised "
                    "SHX keyboard layout, so no text in any drawing could "
                    "spell it"
                ),
            }
        }

    collection = coll(COLL_ENTITIES)
    pattern = "|".join(re.escape(f) for f in forms)
    q = {"drawing_id": drawing_id, "text": {"$regex": pattern}}
    total = collection.count_documents(q)
    room = max(0, limit - len(so_far.get("entities") or []))
    rows = list(
        collection.find(
            q, {"handle": 1, "type": 1, "layer": 1, "text": 1, "bbox_centre": 1}
        ).sort([("layer", 1), ("handle", 1)]).limit(room)
    ) if room else []
    for row in rows:
        row["text_reading"] = shx.read(row.get("text")).reading

    merged = list(so_far.get("entities") or [])
    seen = {r.get("handle") for r in merged}
    merged += [r for r in rows if r.get("handle") not in seen]
    return {
        "entities": merged,
        "returned": len(merged),
        "total_matches": so_far.get("total_matches", 0) + total,
        "truncated": (so_far.get("total_matches", 0) + total) > len(merged),
        "matched_on": (
            "the stored text AND its SHX shape"
            if so_far.get("total_matches")
            else "its SHX shape, not the stored text"
        ),
        "arabic_search": {
            "attempted": True,
            "forms_tried": len(forms),
            "forms_sample": forms[:8],
            "matches": total,
            "note": shx.READING_NOTE,
            "evidence_note": (
                "the matches here were found through the keyboard shape, not "
                "through stored Arabic letters; this drawing contains not one "
                "Arabic letter"
            ),
        },
    }


def _text_near_misses(drawing_id: str, query: str) -> dict[str, Any]:
    """Where else the searched word appears, when no entity text carries it.

    `query_entities` has explained its empty results since D-037: a layer that
    exists under different capitals, or lives on another layout, is named
    rather than left as a bare zero, because "0 matches" and "you asked the
    wrong question" are otherwise the same answer. `search_text` never got the
    same treatment, and it showed.

    Measured: asked to find SCHOOL, the agent answered that no text contains
    it. True -- no entity carries that text. But five layers are named
    `Primary School`, `Secondary School`, `Private School`,
    `Intermediate School` and `SchoolHatch`, and anyone asking where the
    schools are wants exactly that. The answer was correct and useless.

    Names are the drawing's other vocabulary. A drafter searching for a word
    does not care whether the author put it in an annotation, a layer name or
    a block name -- and in a site plan, hatched land uses are named on layers
    precisely because there is no text to label them.

    Returns nothing at all when there is nothing to say, so a genuinely
    absent word still reads as absent.
    """
    drawing = get_drawing(drawing_id)
    if not drawing:
        return {}
    needle = query.casefold().strip()
    if not needle:
        return {}

    def hits(names: Iterable[Any]) -> list[str]:
        seen: list[str] = []
        for name in names:
            if isinstance(name, str) and needle in name.casefold():
                if name not in seen:
                    seen.append(name)
        return seen[:_NEAR_MISS_LIMIT]

    layers = hits(
        l.get("name") if isinstance(l, dict) else l
        for l in (drawing.get("layers") or [])
    )
    blocks = hits(
        b.get("name") if isinstance(b, dict) else b
        for b in (drawing.get("blocks") or [])
    )
    # `layouts` on the drawing record is not a list of layouts. It carries the
    # block definitions too, tagged "[block] ...": 81 entries on Janadriyah for
    # 3 real layouts. Reporting those as layouts is a defect this project has
    # already fixed once, in `list_drawings`, where it made the drawing look
    # like it had 81 sheets. Repeating it here would tell a reader that TREE is
    # "a layout name" when it is a block -- a different object, found a
    # different way -- and the block list beside this one already names it.
    layouts = hits(
        name
        for name in (
            l.get("name") if isinstance(l, dict) else l
            for l in (drawing.get("layouts") or [])
        )
        if not str(name).startswith("[block]")
    )
    if not (layers or blocks or layouts):
        return {}

    # The names go IN the sentence, not beside it. An earlier version counted
    # them -- "appears in 5 layer name(s)" -- and the agent faithfully quoted
    # the count, which leaves the reader asking which five. The most quotable
    # string in a response should carry the answer, not a summary of it.
    where: list[str] = []
    if layers:
        where.append("layer names (" + ", ".join(layers) + ")")
    if blocks:
        where.append("block names (" + ", ".join(blocks) + ")")
    if layouts:
        where.append("layout names (" + ", ".join(layouts) + ")")

    found: dict[str, Any] = {}
    if layers:
        found["layers"] = layers
    if blocks:
        found["blocks"] = blocks
    if layouts:
        found["layouts"] = layouts

    return {
        "why_empty": [
            f"No entity text contains {query!r}, but the name does appear in "
            + " and ".join(where)
            + ". In a drawing, land uses and components are often named on "
            "layers or blocks rather than written as annotation.",
        ],
        "near_misses": found,
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

#: Groups listed by a grouped measurement. The totals always cover everything.
MAX_MEASURE_GROUPS = 200


class MeasureRefused(Exception):
    """A measurement that could only have been answered misleadingly.

    Carries the code, the sentence explaining it, and -- the part that matters
    -- the corrected call. A refusal the caller cannot act on sends an agent
    back to fetching entities one at a time, which is the failure this tool
    exists to end.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


#: The one layout whose coordinates the drawing header's units describe.
MODEL_SPACE = "Model"

#: How many skipped-type rows a measurement lists before it truncates.
MAX_SKIPPED_TYPES = 10


def _reject_empty_filters(**filters: Any) -> None:
    """Refuse a filter that was supplied but says nothing.

    ``if layer:`` treats ``""`` as "no filter given", so a caller that passed an
    empty layer received the whole layout and had every reason to believe it was
    one layer's worth. On model space that is 20,334 entities returned under a
    scope label reading ``layout=Model`` -- the shape of the worst answer this
    project has produced, arriving from the API rather than from the model.

    Whitespace counts as empty. ``layer=" "`` cannot match a real layer name
    here and is far more likely to be a formatting accident than an intent.

    Raised rather than ignored because the two readings -- "no filter" and "a
    filter I could not fill in" -- lead to answers that differ by four orders of
    magnitude, and nothing downstream can tell them apart.
    """
    for name, value in filters.items():
        if value is None:
            continue
        # Anything that is not a non-empty string is a filter that says nothing.
        # The first version tested `not isinstance(value, str) or value.strip()`
        # and so waved through 0, False and [] -- values that a truthiness check
        # drops exactly as it drops "". HTTP and MCP coerce to str today, which
        # made it unreachable rather than harmless.
        if isinstance(value, str) and value.strip():
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            continue
        raise MeasureRefused(
            "EMPTY_FILTER_VALUE",
            repr(name) + " was given as " + repr(value) + ", which filters "
            "nothing. An empty value is silently dropped by a truthiness check, "
            "so the answer would cover the whole scope while reading as though "
            "it covered one " + name + ".",
            "Pass a real value for " + name + ", or omit " + name
            + " entirely if you meant no filter.",
        )


def _space_of(layout_name: str | None) -> str:
    """Which coordinate space a layout's numbers live in.

    Three answers, and they are not interchangeable:

    - ``model``  - real-world coordinates. $INSUNITS describes these.
    - ``paper``  - a sheet. Its coordinates are page geometry: a border drawn
      on an A0 sheet is about 1189 units long because the sheet is 1189 mm
      wide, not because anything in the world is.
    - ``block``  - inside a block definition. The geometry is in the block's
      own space and each INSERT scales it, so one stored number can stand for
      many different real lengths at once.
    """
    if layout_name is None:
        return "drawing"
    if layout_name == MODEL_SPACE:
        return "model"
    if layout_name.startswith(BLOCK_LAYOUT_PREFIX):
        return "block"
    return "paper"


def _unit_names(
    drawing: dict[str, Any], layout_name: str | None = None
) -> dict[str, Any]:
    """What one unit of the stored numbers is called, and whether to trust it.

    Eleven of the eighteen drawings here are in inches, one in millimetres and
    three declare nothing at all. A right number carrying the wrong unit is a
    wrong number, so the unit travels with every figure and the undeclared case
    is named rather than filled in.

    `layout_name` is not optional in spirit. $INSUNITS is a *drawing* header
    field and it describes model space only; DXF has no per-layout unit
    declaration to read. Stamping it on a sheet produced the worst number this
    project has shipped: 1,867 border lines on a paper layout reported as
    "2992.231252 m", complete, with `warnings: []` and `declared_in_file: true`.
    That sheet is an A0 page, so the figure is out by roughly a thousandfold and
    every signal built to mark a number untrustworthy said it was fine.

    The temptation is to notice the sheet is about 1189 x 841 and call it
    millimetres. That is a guess that happens to be right often, which is the
    most dangerous kind of number there is - it would be indistinguishable from
    a measurement, and wrong on the first sheet drawn in inches. So paper and
    block space get `None` and a reason, and the caller is expected to say so.
    """
    space = _space_of(layout_name)
    if space in ("paper", "block"):
        return {
            "name": (
                "sheet coordinates" if space == "paper"
                else "block definition coordinates"
            ),
            "declared_in_file": False,
            "length_unit": None,
            "area_unit": None,
            "space": space,
            "why_no_unit": (
                "This drawing declares "
                + repr(drawing.get("units_name") or "no units")
                + ", but that is a drawing-header field describing MODEL space. "
                + (
                    "These numbers are page geometry on a sheet: a border is "
                    "long because the paper is large, not because anything in "
                    "the world is."
                    if space == "paper"
                    else "These numbers are inside a block definition and each "
                    "insertion scales them, so one stored length stands for "
                    "several real ones."
                )
                + " DXF carries no per-layout unit, so none is claimed here."
            ),
        }

    name = drawing.get("units_name")
    declared = bool(drawing.get("units_code"))
    if not name or name == "unitless":
        return {
            "name": "unitless",
            "declared_in_file": False,
            "length_unit": None,
            "area_unit": None,
            "space": space,
        }
    return {
        "name": name,
        "declared_in_file": declared,
        "length_unit": name,
        "area_unit": name + "2",
        "space": space,
    }


def _unit_names_for_layouts(
    drawing: dict[str, Any], layout_names: Sequence[str]
) -> dict[str, Any]:
    """Units for a set of rows whose layouts are known only after the query.

    A selection is not asked for by layout -- it is a rectangle the user dragged
    -- so the space it lives in has to be read back off the rows. One layout
    gives a definite answer. More than one does not: a total that adds sheet
    coordinates to model coordinates is not a quantity in any unit, and saying
    so is the only honest option.
    """
    distinct = sorted({l for l in layout_names if l})
    if len(distinct) == 1:
        return _unit_names(drawing, distinct[0])
    if not distinct:
        return _unit_names(drawing, None)
    spaces = {_space_of(l) for l in distinct}
    if spaces == {"model"}:
        return _unit_names(drawing, MODEL_SPACE)
    return {
        "name": "mixed spaces",
        "declared_in_file": False,
        "length_unit": None,
        "area_unit": None,
        "space": "mixed",
        "why_no_unit": (
            "These rows span " + str(len(distinct)) + " layouts across "
            + ", ".join(sorted(spaces)) + " space. Adding a sheet coordinate to "
            "a model coordinate does not produce a length in any unit, so none "
            "is claimed. Layouts: " + ", ".join(distinct[:6])
            + ("..." if len(distinct) > 6 else "")
        ),
    }


def _reachable_blocks(drawing_id: str) -> set[str]:
    """Block definitions whose geometry actually reaches a real layout.

    A block is reachable if it is inserted on a real layout, or inserted inside
    a block that is itself reachable. Anything else is defined and drawn
    nowhere.

    This is not a corner case. On the reference drawing **61 of 78** block
    definitions holding geometry are never inserted at all, and they account for
    11,938 of the 23,439 entities that a naive "inside block definitions" count
    picks up -- more than half. On `title_block-iso.dxf` there are zero INSERT
    entities in the whole file, so every definition is unreachable and model
    space is genuinely empty.

    Without this, the shadow told a reader "this is not an absence" over
    geometry that is not drawn, in the sentence D-077 exists to make quotable.
    Denying a true absence is the same defect as asserting a false one, and it
    arrived through the fix for it.
    """
    rows = list(coll(COLL_ENTITIES).aggregate([
        {"$match": {"drawing_id": drawing_id, "type": "INSERT"}},
        {"$group": {"_id": {"block": "$block_name", "where": "$layout"}}},
    ]))

    #: block -> the layouts it is inserted on (real layouts and pseudo ones)
    inserted_in: dict[str, set[str]] = {}
    for row in rows:
        key = row["_id"] or {}
        block, where = key.get("block"), key.get("where")
        if block and where:
            inserted_in.setdefault(block, set()).add(where)

    reachable: set[str] = set()
    frontier = {
        b for b, wheres in inserted_in.items()
        if any(not w.startswith(BLOCK_LAYOUT_PREFIX) for w in wheres)
    }
    # Walk up: a block inserted inside a reachable definition is reachable too.
    while frontier:
        reachable |= frontier
        nxt = set()
        for block, wheres in inserted_in.items():
            if block in reachable:
                continue
            for where in wheres:
                if (
                    where.startswith(BLOCK_LAYOUT_PREFIX)
                    and where[len(BLOCK_LAYOUT_PREFIX):] in reachable
                ):
                    nxt.add(block)
                    break
        frontier = nxt
    return reachable


def _block_definition_shadow(
    query: dict[str, Any], measure: str | None = None
) -> dict[str, Any] | None:
    """How much of the same filter lives inside block definitions.

    Every layout-scoped count in this API is blind in the same way, and the
    blindness is invisible in the result. Entities inside a block *definition*
    are stored under a pseudo-layout ``[block] <name>`` (see
    `extract.BLOCK_LAYOUT_PREFIX`), so ``layout="Model"`` never sees them --
    and the answer comes back looking complete.

    Measured on the reference drawing, this is not a corner case:

    ======================  ===============  =========================
    layer                   in model space   inside block definitions
    ======================  ===============  =========================
    00_Internal Road        0                2,557 (24,302 length)
    00_Prop - Road - CL_    1,233            411   (27,987 length)
    VL2                     133              0
    ======================  ===============  =========================

    A drafter asking for internal road length is told nothing matched, while
    24 km of it sits one level down. On the road centreline layer the reported
    total is a third short.

    Returned as a warning rather than added to the total, and that is the whole
    design: block geometry is drawn once per INSERT and each insert scales and
    rotates it, so the stored length is not a real-world quantity and cannot be
    summed into one. What the caller needs is to know the gap exists and how
    big it might be -- not a number that silently averages over it.

    `None` when the query is not layout-scoped, or when nothing is hidden.
    """
    layout = query.get("layout")
    if not isinstance(layout, str) or layout.startswith(BLOCK_LAYOUT_PREFIX):
        return None

    hidden = {k: v for k, v in query.items() if k != "layout"}
    hidden["layout"] = {"$regex": "^" + re.escape(BLOCK_LAYOUT_PREFIX)}

    group: dict[str, Any] = {"_id": None, "entities": {"$sum": 1},
                             "blocks": {"$addToSet": "$layout"}}
    if measure:
        field = "$" + measure
        is_number = {"$isNumber": field}
        group["measurable"] = {"$sum": {"$cond": [is_number, 1, 0]}}
        group["stored_total"] = {"$sum": {"$cond": [is_number, field, 0]}}

    rows = list(coll(COLL_ENTITIES).aggregate([
        {"$match": hidden},
        {"$group": group},
    ]))
    if not rows or not rows[0].get("entities"):
        return None

    row = rows[0]
    names = sorted(
        b[len(BLOCK_LAYOUT_PREFIX):] for b in (row.get("blocks") or [])
    )

    # Split by whether the definition is drawn at all. Reporting the two
    # together denies a true absence: on one drawing every definition is
    # unreachable and model space really is empty.
    reachable = _reachable_blocks(query["drawing_id"])
    placed_names = [n for n in names if n in reachable]
    unplaced_names = [n for n in names if n not in reachable]

    per_block = {
        (r["_id"] or "")[len(BLOCK_LAYOUT_PREFIX):]: r
        for r in coll(COLL_ENTITIES).aggregate([
            {"$match": hidden},
            {"$group": {
                "_id": "$layout",
                "n": {"$sum": 1},
                **(
                    {"m": {"$sum": {"$cond": [{"$isNumber": "$" + measure}, 1, 0]}},
                     "t": {"$sum": {"$cond": [{"$isNumber": "$" + measure},
                                              "$" + measure, 0]}}}
                    if measure else {}
                ),
            }},
        ])
    }

    def tally(chosen: list[str]) -> dict[str, Any]:
        got = {"entities": sum(int(per_block[n]["n"]) for n in chosen if n in per_block)}
        if measure:
            got["measurable"] = sum(
                int(per_block[n].get("m") or 0) for n in chosen if n in per_block
            )
            got["stored_" + measure] = round(
                sum(float(per_block[n].get("t") or 0.0) for n in chosen if n in per_block),
                6,
            )
        return got

    placed = tally(placed_names)
    unplaced = tally(unplaced_names)

    out: dict[str, Any] = {
        "entities": int(row["entities"]),
        "block_count": len(names),
        "blocks": names[:10],
        "blocks_truncated": len(names) > 10,
        # The half that is actually drawn somewhere.
        "in_placed_blocks": {**placed, "blocks": placed_names[:10],
                             "block_count": len(placed_names)},
        # Defined and inserted nowhere: not drawn, and not an absence to deny.
        "in_never_placed_blocks": {**unplaced, "blocks": unplaced_names[:10],
                                   "block_count": len(unplaced_names)},
    }
    if measure:
        out["measurable"] = int(row.get("measurable") or 0)
        out["stored_" + measure] = round(float(row.get("stored_total") or 0.0), 6)
    return out


# ---------------------------------------------------------------------------
# Duplicate geometry behind a measured total
#
# `measure(length, layer='<a road centreline layer>', layout=Model)` on the
# reference drawing returns 83,962.565 -- three spatially detached copies of
# one 27,987.522 estate, summed into a single confident figure. The arithmetic
# is correct and the question it answers is the wrong one, and nothing in the
# response said so. An earlier hand analysis published the tripled number for
# exactly this reason.
#
# Phase 1's Dossier already records those clusters. This reads that record and
# attaches it to the total it qualifies -- additively, and never at the cost of
# the tool working.
# ---------------------------------------------------------------------------

#: Cluster-count spellings a Dossier duplicate record may use. The reader
#: module owns that shape and may change it; reading a few spellings keeps this
#: block carrying NUMBERS rather than degrading into a warning with none in it.
_DUP_CLUSTER_KEYS = ("clusters", "cluster_count", "copies", "members",
                     "members_total", "duplicate_clusters")
_DUP_PER_CLUSTER_ENTITY_KEYS = ("entities_per_cluster", "entities_each",
                                "entities_per_copy")
_DUP_INVOLVED_KEYS = ("entities_involved", "entities_total")

#: Keys whose explicit `False` means "checked, and there is nothing here".
_DUP_CLEAN_FLAGS = ("duplicate_clusters", "has_duplicates", "duplicates")

#: Fields of the Dossier record named back to the reader, capped and counted.
_DUP_FIELDS_LISTED = 25


def _dossier_reader():
    """The Dossier reader module, or None when it cannot be imported.

    `measure` is one of the most-used tools in this stack, and an enrichment
    module is not allowed to be able to break it. The import is deferred to
    call time so that every failure -- module absent, module raising on import,
    an older deployment without it -- lands in the same place: the answer
    becomes "not computed" and the response is otherwise exactly what it was.
    """
    try:
        from . import dossier_read
    except Exception:  # pragma: no cover - defensive by design
        log.debug("dossier_read is not importable", exc_info=True)
        return None
    return dossier_read


def _first_number(record: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    """The first of `keys` carrying a real number. Booleans are not numbers."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
    return None


def _measure_unit(measure: str, units: Mapping[str, Any]) -> tuple[Any, str, str]:
    """The unit a figure of this measure carries -- and it may not have one (G2).

    11 of the drawings in this store are in inches and 3 declare nothing at
    all, so a length is never labelled "m" because the file currently open
    happens to be. Returns `(unit, label, basis)`; `unit` is None whenever
    nothing may honestly be claimed, and `label` is what goes inside prose.
    """
    unit = units.get("length_unit") if measure == "length" else units.get("area_unit")
    if unit:
        return unit, str(unit), (
            "the unit this drawing's header declares for model space"
        )
    space = units.get("space", "model")
    if space == "paper":
        return None, "sheet units", (
            "these are page coordinates, not a real-world measurement, so no "
            "unit is claimed for them"
        )
    if space == "block":
        return None, "block definition units", (
            "block geometry is scaled by each insert, so one stored figure is "
            "many real lengths and no unit is claimed for it"
        )
    return None, "drawing units", (
        "this file declares no units, so every figure here is in its own "
        "drawing units and must not be read as metres"
    )


def _duplicate_warning(
    drawing_id: str,
    *,
    layer: str,
    layout: str,
    measure: str,
    units: Mapping[str, Any],
    sum_measured: float | None,
) -> dict[str, Any]:
    """What the Dossier knows about this layer being drawn more than once.

    Three outcomes, and the third is the one this exists for:

    * `duplicate_clusters` -- the Dossier records copies, and the total above
      sums all of them;
    * `no_duplicates_recorded` -- a Dossier exists and records none here;
    * `not_computed` -- no Dossier, no reader, or the lookup failed.

    `not_computed` is NEVER collapsed into `no_duplicates_recorded`. An absence
    of checking that reads as a clean bill of health is the precise failure
    this whole campaign exists to end.
    """
    unit, unit_label, unit_basis = _measure_unit(measure, units)
    scope = "layer " + repr(layer) + " in layout " + repr(layout)
    total_text = (
        str(sum_measured) + " " + unit_label if sum_measured is not None else "total"
    )

    block: dict[str, Any] = {
        "layer": layer,
        "layout": layout,
        "measure": measure,
        "unit": unit,
        "unit_label": unit_label,
        "unit_basis": unit_basis,
        "total_reported": sum_measured,
        "total_reported_is": (
            "sum_measured_only from this response, in " + unit_label
            + " -- the figure this warning qualifies"
        ),
        "status": "not_computed",
        "checked": False,
        "reason": None,
        "clusters": None,
        "entities_per_cluster": None,
        "entities_involved": None,
        "per_copy": None,
        "per_copy_is": "",
        "dossier_fields": [],
        "dossier_fields_total": 0,
        "message": "",
        "hint": "",
    }

    def not_computed(reason: str) -> dict[str, Any]:
        return {
            **block,
            "status": "not_computed",
            "checked": False,
            "reason": reason,
            "per_copy_is": "No per-copy figure: nothing has been computed here.",
            "message": (
                "NOT COMPUTED, which is not the same answer as 'no duplicates': "
                + reason + " Nothing here has checked whether " + scope
                + " holds the same geometry drawn more than once, so the "
                + total_text + " above is UNKNOWN in that respect rather than "
                "clean. A layer in this store sums three copies of one estate "
                "into a single confident figure, and a published analysis "
                "quoted it."
            ),
            "hint": (
                "Build this drawing's Dossier (scripts/dossier_backfill.py) and "
                "measure again, or compare the layer's detached clusters by "
                "hand, before quoting this total as a real-world quantity."
            ),
        }

    reader = _dossier_reader()
    if reader is None:
        return not_computed("the Dossier reader is not available in this build.")

    try:
        record = reader.duplicate_warning(drawing_id, layer=layer, layout=layout)
    except Exception:  # pragma: no cover - defensive by design
        log.debug("duplicate_warning lookup failed", exc_info=True)
        return not_computed("the Dossier lookup failed for this drawing.")

    if record is not None and not isinstance(record, Mapping):
        return not_computed(
            "the Dossier returned a duplicate record this tool cannot read."
        )

    def clean(basis: str) -> dict[str, Any]:
        return {
            **block,
            "status": "no_duplicates_recorded",
            "checked": True,
            "reason": basis,
            "per_copy_is": (
                "No per-copy figure, and none is needed: this total is not a "
                "sum of copies."
            ),
            "message": (
                "Checked: " + basis + ", so the " + total_text + " above is not "
                "a sum of copies of the same geometry. That is a statement "
                "about " + scope + " only -- it says nothing about other "
                "layers, other layouts, or geometry inside block definitions."
            ),
            "hint": "",
        }

    if record is None:
        # None is ambiguous on its own, so ask whether a Dossier exists at all.
        # Missing Dossier and checked-and-clean are different answers and must
        # not collapse into one.
        try:
            built = reader.dossier_for(drawing_id)
        except Exception:  # pragma: no cover - defensive by design
            log.debug("dossier_for lookup failed", exc_info=True)
            built = None
        if not built:
            return not_computed("no Dossier has been built for this drawing.")
        return clean(
            "this drawing's Dossier records no duplicate clusters on " + scope
        )

    # The reader nests one entry per duplicate GROUP, because a layer can hold
    # more than one set of copies. The figures this block quotes -- how many
    # copies, how many entities each, how much one copy measures -- live on
    # that entry, so the largest group is flattened into the search space
    # alongside the record's own keys.
    #
    # Without this the warning still fires, and that is exactly why it is worth
    # the lines: it fired on the reference road layer reading "several
    # spatially detached clusters" with `clusters: null` and `per_copy: null`.
    # A warning that cannot say three, or say 27,987.52, leaves the reader with
    # the 83,962.565 total and a vague misgiving -- which is close to no
    # warning at all.
    groups = record.get("groups")
    largest = {}
    if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
        entries = [g for g in groups if isinstance(g, Mapping)]
        if entries:
            largest = max(
                entries,
                key=lambda g: _first_number(g, _DUP_CLUSTER_KEYS) or 0,
            )
    per_copy_block = largest.get("per_copy")
    record = {
        **{k: v for k, v in largest.items() if k != "per_copy"},
        **(per_copy_block if isinstance(per_copy_block, Mapping) else {}),
        **record,
    }
    # The reader also writes the whole sentence, with every figure already in
    # it. Prefer it over anything reassembled here: one wording, from the
    # module that did the measuring.
    reader_statement = record.get("statement")

    fields = sorted(str(k) for k in record)
    block["dossier_fields"] = fields[:_DUP_FIELDS_LISTED]
    block["dossier_fields_total"] = len(fields)

    raw_clusters = _first_number(record, _DUP_CLUSTER_KEYS)
    clusters = int(raw_clusters) if raw_clusters is not None else None
    if any(record.get(flag) is False for flag in _DUP_CLEAN_FLAGS) or (
        clusters is not None and clusters < 2
    ):
        return clean(
            "this drawing's Dossier records no duplicate clusters on " + scope
        )

    raw_each = _first_number(record, _DUP_PER_CLUSTER_ENTITY_KEYS)
    entities_each = int(raw_each) if raw_each is not None else None
    raw_involved = _first_number(record, _DUP_INVOLVED_KEYS)
    involved = int(raw_involved) if raw_involved is not None else None

    recorded_per_copy = _first_number(record, (
        measure + "_total_each",
        measure + "_per_cluster",
        "per_copy_" + measure,
    ))
    if recorded_per_copy is not None:
        per_copy: float | None = round(float(recorded_per_copy), 6)
        per_copy_basis = (
            "the per-cluster " + measure + " the Dossier recorded, read "
            "independently of the sum above rather than divided out of it"
        )
    elif clusters and sum_measured is not None:
        per_copy = round(sum_measured / clusters, 6)
        per_copy_basis = (
            "the reported sum divided by " + str(clusters) + " clusters. That "
            "division holds only if the copies really are identical, which the "
            "cluster match argues from bulk shape and does not prove"
        )
    else:
        per_copy = None
        per_copy_basis = (
            "no per-copy figure is derivable: "
            + ("the cluster count is not recorded"
               if not clusters else "no total was measured to divide")
        )

    copies = str(clusters) if clusters else "several"
    each = (
        str(entities_each) + " entities each"
        if entities_each is not None
        else "the same entity count each"
    )
    message = (
        "The Dossier records " + copies + " spatially detached clusters holding "
        + each + " on " + scope + " -- the same geometry drawn " + copies
        + " times. The " + total_text + " above sums all of them. The "
        "arithmetic is not wrong; the question it answers is 'how much is drawn "
        "on this layer', not 'how much of it is there'"
        + ("" if per_copy is None
           else ". One copy measures " + str(per_copy) + " " + unit_label)
        + ". Do not quote this total as a real-world quantity without saying "
        "which copy it is, or that it counts " + copies + " of them."
    )
    # The reader's own sentence leads when it has one: it was written where the
    # figures were measured, so it cannot drift from them the way a sentence
    # reassembled here can. The instruction stays appended, because that is
    # what a reader is supposed to DO about it.
    if isinstance(reader_statement, str) and reader_statement.strip():
        message = (
            reader_statement.strip()
            + " Do not quote the total as a real-world quantity without saying "
            "which copy it is, or that it counts " + copies + " of them."
        )

    return {
        **block,
        "status": "duplicate_clusters",
        "checked": True,
        "reason": None,
        "clusters": clusters,
        "entities_per_cluster": entities_each,
        "entities_involved": involved,
        "per_copy": per_copy,
        "per_copy_is": (
            "No per-copy figure: " + per_copy_basis + "."
            if per_copy is None
            else str(per_copy) + " " + unit_label + " of " + measure
            + " for ONE copy -- " + per_copy_basis + ". It is one copy's "
            "figure, not this layer's total."
        ),
        "message": message,
        "hint": (
            "Measure one cluster instead of the layer -- the Dossier records "
            "each cluster's bounding box and example handles, and "
            "describe_drawing carries the same finding. Dividing by " + copies
            + " assumes the copies are identical."
        ),
    }


def measure_entities(
    drawing_id: str,
    *,
    measure: str,
    layout_name: str,
    layer: str | None = None,
    dxftype: str | None = None,
    block_name: str | None = None,
    group_by: str | None = None,
    top: int = 25,
) -> dict[str, Any]:
    """Total the length or area of everything matching a filter, in one pass.

    The tool quantity take-off needed and did not have. Asked for one layer's
    total length, an agent previously paged `query_entities` and then called
    `get_entity` per handle: 136 calls, an error, and no total. The same figure
    is one aggregation -- 133 objects, 10084.024105 m, in about 270 ms.

    **The denominator is not decoration.** In one model space only 4,205 of
    20,334 entities carry a length at all, so a sum over a filter answers a
    narrower question than the filter suggests and a bare number hides which
    one. Three counts always travel with the sum:

    * `measured_entities` -- carried a numeric value and are in the sum;
    * `unmeasurable_entities` -- stored `null`, meaning the geometry cannot be
      measured, NOT that it measured zero;
    * `zero_valued_entities` -- genuinely measured as zero, and included.

    **`total_for_all_matched` exists only when nothing was skipped.** Where any
    matched entity is unmeasurable it is `null`, and the partial figure is
    reachable solely as `sum_measured_only` -- a name that cannot be quoted as
    a total by accident. That asymmetry is the design: this project has already
    shipped a partial figure narrated as a total, with a percentage over the
    wrong denominator.
    """
    _reject_empty_filters(layer=layer, type=dxftype, block_name=block_name,
                          layout=layout_name, group_by=group_by)
    if measure not in ("length", "area"):
        raise MeasureRefused(
            "INVALID_MEASURE",
            "measure must be 'length' or 'area', got " + repr(measure) + ".",
            "Pass measure='length' for run lengths, 'area' for enclosed areas.",
        )

    drawing = get_drawing(drawing_id)
    if not drawing:
        return {}

    # The 812-versus-818 guard, and the cheapest high-value check here. `DIM`
    # is a layer in this drawing and `DIMENSION` is a DXF type; an agent handed
    # one for the other answers fluently and wrongly. Layers and types are
    # different axes, so a value belonging to the other axis is a question
    # asked wrongly rather than a query that legitimately found nothing.
    counts_by_type = drawing.get("counts_by_type") or {}
    if layer:
        layer_names = {
            (l.get("name") if isinstance(l, dict) else l)
            for l in (drawing.get("layers") or [])
        }
        if layer not in layer_names and layer.upper() in counts_by_type:
            raise MeasureRefused(
                "AMBIGUOUS_FILTER_VALUE",
                repr(layer) + " is not a layer in this drawing, but "
                + repr(layer.upper()) + " is a DXF type ("
                + str(counts_by_type[layer.upper()])
                + " entities). Layers and types are different axes.",
                "Retry with type=" + repr(layer.upper())
                + " instead of layer=" + repr(layer) + ".",
            )

    query: dict[str, Any] = {"drawing_id": drawing_id, "layout": layout_name}
    if layer:
        query["layer"] = layer
    if dxftype:
        query["type"] = dxftype.upper()
    if block_name:
        query["block_name"] = block_name

    field = "$" + measure
    is_number = {"$isNumber": field}
    collection = coll(COLL_ENTITIES)

    facets: dict[str, Any] = {
        # Totals are computed over the WHOLE match, in their own branch, so a
        # truncated group list can never quietly change the grand total.
        "totals": [
            {
                "$group": {
                    "_id": None,
                    "matched": {"$sum": 1},
                    "measured": {"$sum": {"$cond": [is_number, 1, 0]}},
                    "zeros": {
                        "$sum": {
                            "$cond": [
                                {"$and": [is_number, {"$eq": [field, 0]}]},
                                1,
                                0,
                            ]
                        }
                    },
                    "total": {"$sum": {"$cond": [is_number, field, 0]}},
                }
            }
        ],
        # What was skipped, by type. A reader told "summed from 40 of 210"
        # immediately asks which 170, and the answer decides whether the number
        # is usable or the drawing needs fixing first.
        "skipped": [
            {"$match": {measure: {"$not": {"$type": "number"}}}},
            {"$group": {"_id": "$type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            # 11 rather than 10: the extra row is never returned, it only tells
            # us whether the list was truncated. A capped list that does not
            # admit it is the D-074 defect in miniature.
            {"$limit": MAX_SKIPPED_TYPES + 1},
        ],
        # Which types are present at all, so a mixed sum can be shown for what
        # it is rather than refused outright.
        "types": [
            {"$group": {"_id": "$type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 50},
        ],
    }

    grouped_field = None
    if group_by:
        if group_by not in ("layer", "type"):
            raise MeasureRefused(
                "INVALID_FIELD",
                "group_by must be 'layer' or 'type', got " + repr(group_by) + ".",
                "Use group_by='layer' for a per-layer recap, 'type' to see "
                "what actually matched.",
            )
        grouped_field = group_by
        facets["groups"] = [
            {
                "$group": {
                    "_id": "$" + group_by,
                    "matched": {"$sum": 1},
                    "measured": {"$sum": {"$cond": [is_number, 1, 0]}},
                    "total": {"$sum": {"$cond": [is_number, field, 0]}},
                }
            },
            {"$sort": {"total": -1, "matched": -1}},
            {"$limit": min(top, MAX_MEASURE_GROUPS)},
        ]
        facets["group_count"] = [
            {"$group": {"_id": "$" + group_by}},
            {"$count": "n"},
        ]

    raw = list(collection.aggregate([{"$match": query}, {"$facet": facets}]))
    facet = raw[0] if raw else {}
    totals_rows = facet.get("totals") or []
    totals = (
        totals_rows[0]
        if totals_rows
        else {"matched": 0, "measured": 0, "zeros": 0, "total": 0.0}
    )

    matched = int(totals.get("matched") or 0)
    measured = int(totals.get("measured") or 0)
    unmeasurable = matched - measured
    units = _unit_names(drawing, layout_name)

    scope = {
        key: value
        for key, value in (
            ("layout", layout_name),
            ("layer", layer),
            ("type", dxftype.upper() if dxftype else None),
            ("block_name", block_name),
        )
        if value
    }
    scope_label = " AND ".join(k + "=" + str(v) for k, v in scope.items())

    sum_measured = round(float(totals.get("total") or 0.0), 6) if measured else None
    complete = matched > 0 and unmeasurable == 0

    result: dict[str, Any] = {
        "drawing_id": drawing_id,
        "measure": measure,
        "scope": scope,
        "scope_label": scope_label,
        "units": units,
        "total_matches": matched,
        "measured_entities": measured,
        "unmeasurable_entities": unmeasurable,
        "zero_valued_entities": int(totals.get("zeros") or 0),
        "measured_fraction": round(measured / matched, 4) if matched else None,
        "sum_measured_only": sum_measured,
        "total_for_all_matched": sum_measured if complete else None,
        "total_for_all_matched_withheld": None
        if complete
        else (
            str(unmeasurable) + " of " + str(matched) + " matched entities carry "
            "no " + measure + "; no complete total exists for this filter."
            if matched
            else "Nothing matched this filter."
        ),
        "unmeasurable_by_type": [
            {"type": row["_id"], "count": row["count"]}
            for row in (facet.get("skipped") or [])[:MAX_SKIPPED_TYPES]
            if row.get("_id")
        ],
        "unmeasurable_by_type_order": (
            "by count, descending"
            + (
                ", truncated to the " + str(MAX_SKIPPED_TYPES) + " commonest; "
                "more types were skipped than are listed here"
                if len(facet.get("skipped") or []) > MAX_SKIPPED_TYPES
                else ", complete"
            )
        ),
        "warnings": [],
    }

    present_types = [r["_id"] for r in (facet.get("types") or []) if r.get("_id")]

    # A run-length total that silently includes circumferences is the mixed-unit
    # trap in miniature. Warned about rather than refused: a layer holding LINE,
    # ARC and LWPOLYLINE together is ordinary take-off, and refusing that would
    # send the caller back to counting by hand -- the failure being fixed.
    if measure == "length" and "CIRCLE" in present_types:
        result["warnings"].append(
            {
                "code": "CIRCUMFERENCE_IN_LENGTH_TOTAL",
                "severity": "high",
                "message": (
                    "CIRCLE entities matched. Their stored length is a "
                    "circumference, not a run length, and it is inside this sum."
                ),
                "hint": "Add type='LWPOLYLINE', or group_by='type' to separate them.",
            }
        )
    if matched and measured == 0:
        result["warnings"].append(
            {
                "code": "NOTHING_MEASURABLE",
                "severity": "high",
                "message": (
                    str(matched) + " entities matched but none carry a "
                    + measure + ". Only LINE, ARC, CIRCLE and LWPOLYLINE carry "
                    "a length; only CIRCLE and closed LWPOLYLINE carry an area; "
                    "an LWPOLYLINE with any curved segment carries neither, and "
                    "HATCH carries no area at all."
                ),
                "hint": "Call with group_by='type' to see what actually matched.",
            }
        )
    shadow = _block_definition_shadow(query, measure)
    if shadow:
        result["block_definition_shadow"] = shadow
        result["warnings"].append(
            {
                "code": "GEOMETRY_INSIDE_BLOCK_DEFINITIONS",
                "severity": "high",
                "message": (
                    str(shadow["entities"]) + " more entities match this filter "
                    "inside block definitions and are NOT in this total ("
                    + str(shadow.get("measurable", 0)) + " of them carry a "
                    + measure + ", storing " + str(shadow.get("stored_" + measure, 0))
                    + " between them). They are drawn " + str(shadow["block_count"])
                    + " block definition(s) deep: "
                    + ", ".join(shadow["blocks"])
                    + ("..." if shadow["blocks_truncated"] else "")
                    + ". Each INSERT scales and rotates that geometry, so the "
                    "stored figure is not a real-world quantity and cannot "
                    "simply be added -- and on this data some of it is "
                    "coincident duplicates of entities already counted above, "
                    "so the gap may be smaller than the count suggests, or "
                    "zero."
                ),
                "hint": (
                    "To see them, measure with layout='"
                    + BLOCK_LAYOUT_PREFIX + "<name>'. To judge the real "
                    "contribution, count the INSERTs of those blocks in this "
                    "layout and check their scale."
                ),
            }
        )
    if units.get("space") in ("paper", "block"):
        result["warnings"].append(
            {
                "code": "NOT_MODEL_SPACE_UNITS",
                "severity": "high",
                "message": (
                    "These figures are "
                    + ("sheet" if units["space"] == "paper" else "block definition")
                    + " coordinates, not model-space measurements, and no unit "
                    "is claimed for them. "
                    + units.get("why_no_unit", "")
                ),
                "hint": (
                    "For a real-world quantity, measure with layout='"
                    + MODEL_SPACE + "'."
                ),
            }
        )
    elif units["name"] == "unitless":
        result["warnings"].append(
            {
                "code": "UNITS_NOT_DECLARED",
                "severity": "high",
                "message": (
                    "This drawing declares no units. These figures are drawing "
                    "units, not metres."
                ),
                "hint": "Confirm the scale against the title block before pricing.",
            }
        )

    # A measured total for ONE named layer is the only place this question
    # makes sense: the Dossier records duplicates per layer x layout, and a
    # sum with no layer filter has no single layer to be three copies of.
    # Attached whenever such a total exists -- including when the answer is
    # "not computed", because silence there reads as a clean bill of health.
    if layer and sum_measured is not None:
        duplicates = _duplicate_warning(
            drawing_id,
            layer=layer,
            layout=layout_name,
            measure=measure,
            units=units,
            sum_measured=sum_measured,
        )
        result["duplicate_warning"] = duplicates
        if duplicates["status"] == "duplicate_clusters":
            # High severity, so `_warning_note` carries it into `statement`.
            # The instruction tells the agent to quote that sentence; a caveat
            # sitting outside it is optional, and this one is not.
            result["warnings"].append(
                {
                    "code": "DUPLICATE_GEOMETRY_IN_TOTAL",
                    "severity": "high",
                    "message": duplicates["message"],
                    "hint": duplicates["hint"],
                }
            )

    if grouped_field:
        rows = facet.get("groups") or []
        total_groups = 0
        group_count = facet.get("group_count") or []
        if group_count:
            total_groups = int(group_count[0].get("n") or 0)
        result["by_" + grouped_field] = [
            {
                "name": row["_id"] if row.get("_id") is not None else "(none)",
                "count": int(row.get("matched") or 0),
                "measured_entities": int(row.get("measured") or 0),
                "unmeasurable_entities": int(row.get("matched") or 0)
                - int(row.get("measured") or 0),
                "sum_" + measure + "_measured_only": round(
                    float(row.get("total") or 0.0), 6
                )
                if row.get("measured")
                else None,
            }
            for row in rows
        ]
        result["groups_listed"] = len(rows)
        result["total_groups"] = total_groups
        # Say what the ordering means. These rows are ranked by the measure --
        # by summed length or area -- and NOT by how many entities each group
        # holds. Without this line, "top 20 layers by entity count" was answered
        # from this list: it came back led by a road layer of 1,233 while DIM,
        # with 9,738 entities and no length at all, sat near the bottom. Every
        # figure in that answer was real and the ranking was for another
        # question. A count ranking is distinct_values(field=..., layout=...),
        # whose `most_common` is ordered by count.
        result["groups_ordered_by"] = (
            "sum_" + measure + "_measured_only, descending — NOT entity count. "
            "For a ranking by number of entities use distinct_values("
            "field='" + grouped_field + "', layout=...) and read `most_common`."
        )
        result["grouping_complete"] = total_groups <= len(rows)
        if not result["grouping_complete"]:
            result["truncated_note"] = (
                str(total_groups) + " groups exist; the " + str(len(rows))
                + " largest are listed. The counts and sums above are computed "
                "over the WHOLE match, not over the listed groups, so they do "
                "not shrink with this cap."
                + (
                    " (total_for_all_matched is null here for the separate "
                    "reason that some matched entities are unmeasurable.)"
                    if result.get("total_for_all_matched") is None
                    else ""
                )
            )

    # `measured_fraction` rounded 1/35,841 to 0.0 and sat beside a real sum, so
    # a reader taking the fraction at face value saw "nothing was measured"
    # while a number was present. Rounding is fine in the middle of the range
    # and destroys the meaning at either end.
    frac = result.get("measured_fraction")
    if isinstance(frac, float):
        if frac == 0.0 and result.get("measured_entities"):
            result["measured_fraction"] = None
            result["measured_fraction_note"] = (
                "Under 0.005% of matched entities were measured -- too small to "
                "express at this precision without reading as zero. The counts "
                "are exact: see measured_entities and total_matches."
            )
        elif frac == 1.0 and result.get("unmeasurable_entities"):
            result["measured_fraction"] = None
            result["measured_fraction_note"] = (
                "Over 99.995% but not all matched entities were measured. "
                "Reported as null rather than 1.0, which would read as "
                "complete; total_for_all_matched is withheld for the same "
                "reason."
            )

    result["statement"] = _measure_statement(result)

    if matched == 0:
        result.update(
            diagnose_empty_query(
                drawing_id,
                layer=layer,
                dxftype=dxftype,
                layout_name=layout_name,
                block_name=block_name,
            )
        )
    return result


#: Types that carry neither a length nor an area by their nature.
_NEITHER = {"DIMENSION", "TEXT", "MTEXT", "HATCH", "INSERT", "ATTRIB", "ATTDEF",
            "LEADER", "MULTILEADER", "POINT", "VIEWPORT", "OLE2FRAME", "IMAGE",
            "WIPEOUT", "PDFREFERENCE", "SOLID", "3DSOLID", "ACAD_TABLE",
            "3DFACE", "LIGHT", "RAY", "XLINE", "SHAPE", "TOLERANCE"}

#: Open curves. They have a length; they enclose nothing, so an area question
#: is not something they failed to answer -- it is not a question about them.
_OPEN_CURVES = {"LINE", "ARC", "SPLINE", "POLYLINE", "ELLIPSE", "HELIX"}


def _carries_no(measure: str) -> set[str]:
    """Types with nothing to contribute to *this* measure, by their nature.

    One list served both measures, and for `area` it counted LINE and ARC as
    geometry that "could have carried an area and did not". On model space that
    turned a 547-entity gap into a claimed 1,617 -- a 66% overstatement, in the
    sentence written to stop overstatement. The tool's own NOTHING_MEASURABLE
    warning already said "only CIRCLE and closed LWPOLYLINE carry an area".
    """
    return _NEITHER | (_OPEN_CURVES if measure == "area" else set())


def _warning_note(result: dict[str, Any]) -> str:
    """High-severity warnings, inside the sentence that gets quoted.

    The instruction tells the agent to quote `statement` and says nothing about
    carrying `warnings`. So a CIRCLE length total -- a sum of circumferences --
    arrived as a complete, denominator-bearing, entirely quotable sentence with
    its `severity: high` warning sitting outside it, optional.
    """
    high = [
        w for w in (result.get("warnings") or [])
        if w.get("severity") == "high"
        and w.get("code") != "GEOMETRY_INSIDE_BLOCK_DEFINITIONS"
    ]
    if not high:
        return ""
    return " CAVEAT: " + " ".join(w["message"] for w in high)


def _missing_length_note(result: dict[str, Any]) -> str:
    """How much of the unmeasured population is a real gap.

    The sentence used to end "so the true length is unknown and larger" in every
    partial case. On model space that reads as a warning about 16,129 entities,
    when 15,829 of them are DIMENSION, TEXT, HATCH and INSERT -- objects with no
    length to have. Only 300 bulged polylines are length that exists and was not
    computed.

    Telling a reader their 417 km total is meaningfully short, when it is short
    by 300 polylines, spends the credibility that a real warning needs.
    """
    skipped = result.get("unmeasurable_by_type") or []
    if not skipped:
        return ""
    measure = result["measure"]
    blind = _carries_no(measure)
    by_nature = sum(r["count"] for r in skipped if r["type"] in blind)
    real_gap = sum(r["count"] for r in skipped if r["type"] not in blind)

    # R#6: the list is capped, the count is not. Saying "X of them are this and
    # Y are that" over a truncated list loses the remainder inside a sentence
    # that claims to partition the population exhaustively -- 564 + 2082 = 2646
    # against a stated 2648.
    listed = by_nature + real_gap
    total_skipped = int(result.get("unmeasurable_entities") or listed)
    unlisted = max(0, total_skipped - listed)
    tail = (
        " A further " + str(unlisted) + " are of types beyond the "
        + str(len(skipped)) + " listed and are not classified here."
        if unlisted else ""
    )
    if real_gap == 0:
        return (
            " All " + str(by_nature) + " of them are types that carry no "
            + measure + " at all (" + ", ".join(
                r["type"] for r in skipped[:4]
            ) + "), so the total is not missing anything they could have "
            "contributed." + tail
        )
    if by_nature == 0:
        return (
            " All " + str(real_gap) + " could have carried a " + measure
            + " and did not, so the true " + measure + " is larger by an "
            "unknown amount." + tail
        )
    return (
        " Of those, " + str(by_nature) + " are types that carry no " + measure
        + " at all; " + str(real_gap) + " could have and did not, so the true "
        + measure + " is larger by that much and no more." + tail
    )


def _measure_statement(result: dict[str, Any]) -> str:
    """One sentence carrying the number, its unit, its scope and its denominator.

    Written here rather than left to the caller because both wrong numbers this
    project has shipped were composed in prose around a correct figure. Making
    the honest sentence the cheapest thing to quote is a steadier control than
    instructing a model not to paraphrase.
    """
    measure = result["measure"]
    unit = (
        result["units"]["length_unit"]
        if measure == "length"
        else result["units"]["area_unit"]
    )
    space = result["units"].get("space", "model")
    if unit:
        unit_text = " " + unit
    elif space == "paper":
        # Not "drawing units": that phrasing invites the reader to look up the
        # drawing's units and apply them, which is the mistake being prevented.
        unit_text = " sheet units (page geometry, NOT a real-world length)"
    elif space == "block":
        unit_text = " block definition units (scaled differently by each insert)"
    else:
        unit_text = " drawing units (this file declares none)"
    where = result["scope_label"] or "this drawing"
    matched = result["total_matches"]
    measured = result["measured_entities"]

    # The shadow belongs IN the sentence, not only beside it. "Nothing matched"
    # is the most quotable string in this response and it is the one a reader
    # acts on; on `00_Internal Road` it was literally true of model space and
    # completely wrong as an answer, because 2,557 entities of that layer live
    # inside a block definition one level down.
    shadow = result.get("block_definition_shadow")
    shadow_text = ""
    if shadow:
        placed = shadow.get("in_placed_blocks") or {}
        unplaced = shadow.get("in_never_placed_blocks") or {}
        parts = []
        if placed.get("entities"):
            parts.append(
                str(placed["entities"]) + " further entities match this filter "
                "inside block definition(s) that are placed in this drawing and "
                "are NOT included; their stored " + measure + " is "
                + str(placed.get("stored_" + measure, "unknown"))
                + ", which each insertion scales, so it cannot simply be added, "
                "and some of it may be coincident duplicates of what is already "
                "counted."
            )
        if unplaced.get("entities"):
            parts.append(
                "A further " + str(unplaced["entities"]) + " sit in block "
                "definitions that are never inserted anywhere, so they are not "
                "drawn and are not missing from this total."
            )
        shadow_text = (" NOTE: " + " ".join(parts)) if parts else ""

    if matched == 0:
        placed = (shadow or {}).get("in_placed_blocks") or {}
        unplaced = (shadow or {}).get("in_never_placed_blocks") or {}
        if placed.get("entities"):
            return (
                "Nothing matched " + where + " -- but this is not an absence: "
                + str(placed["entities"]) + " entities matching this filter live "
                "inside block definition(s) that ARE placed in this drawing ("
                + ", ".join(placed.get("blocks") or [])
                + "), where a layout-scoped query cannot see them. Their stored "
                + measure + " is " + str(placed.get("stored_" + measure, "unknown"))
                + ", scaled by each insertion, so it is not a quantity you can "
                "quote directly."
            )
        if unplaced.get("entities"):
            # Denying a true absence is the same defect as asserting a false
            # one. On one drawing there are no INSERT entities at all, so every
            # definition is unreachable and model space really is empty.
            return (
                "Nothing matched " + where + ", and this IS an absence. "
                + str(unplaced["entities"]) + " entities matching this filter "
                "exist inside block definition(s) (" + ", ".join(
                    unplaced.get("blocks") or []
                ) + "), but those definitions are never inserted anywhere in this "
                "drawing, so none of that geometry is drawn."
            )
        return "Nothing matched " + where + ", so there is no " + measure + " to report."
    if measured == 0:
        return (
            str(matched) + " entities matched " + where + ", but none of them "
            "carry a " + measure + ", so no total can be given."
            + _warning_note(result) + shadow_text
        )
    if result["total_for_all_matched"] is not None:
        return (
            "Total " + measure + " of " + str(matched) + " entities where "
            + where + ": " + str(result["total_for_all_matched"]) + unit_text
            + ", summed from all " + str(measured) + " of " + str(matched)
            + " matched entities." + _warning_note(result) + shadow_text
        )
    return (
        "PARTIAL: of " + str(matched) + " entities where " + where + ", only "
        + str(measured) + " carry a " + measure + ". Those " + str(measured)
        + " sum to " + str(result["sum_measured_only"]) + unit_text + ". The other "
        + str(result["unmeasurable_entities"]) + " are not measurable and were "
        "not counted as zero." + _missing_length_note(result)
        + _warning_note(result) + shadow_text
    )


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


def add_comment(
    drawing_id: str,
    handle: str,
    body: str,
    *,
    author: str,
    anchor: Sequence[float] | None = None,
    client_request_id: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store a comment anchored to one entity.

    `client_request_id` makes the write idempotent. An agent loop retries; a
    retry must not leave two identical comments on the same door.
    """
    doc: dict[str, Any] = {
        "drawing_id": drawing_id,
        "entity_handle": handle,
        "body": body,
        "author": author,
        "created_at": _utcnow(),
        "status": "open",
        "anchor": {"point": list(anchor)} if anchor else None,
        "provenance": provenance or {},
    }
    if client_request_id:
        doc["client_request_id"] = client_request_id

    try:
        result = coll(COLL_COMMENTS).insert_one(doc)
    except DuplicateKeyError:
        # Same client_request_id already stored. Returning it is right only if
        # it is genuinely the same write retried: the unique index is global,
        # so a caller reusing one id across two entities would otherwise get a
        # cheerful 201 carrying somebody else's comment while nothing was
        # saved for the entity they asked about.
        existing = coll(COLL_COMMENTS).find_one(
            {"client_request_id": client_request_id}
        )
        if existing is None:  # pragma: no cover - only on a racing delete
            raise
        if (
            existing.get("drawing_id") != drawing_id
            or existing.get("entity_handle") != handle
        ):
            raise CommentIdConflict(
                f"client_request_id {client_request_id!r} was already used for "
                f"{existing.get('drawing_id')}:{existing.get('entity_handle')}, "
                f"not for {drawing_id}:{handle}"
            )
        log.info(
            "comment retry collapsed",
            extra={"drawing_id": drawing_id, "handle": handle},
        )
        existing["_id"] = str(existing["_id"])
        return existing

    doc["_id"] = str(result.inserted_id)
    return doc


def list_comments(
    drawing_id: str, handle: str | None = None
) -> list[dict[str, Any]]:
    """Comments for a drawing, optionally narrowed to one entity."""
    query: dict[str, Any] = {"drawing_id": drawing_id}
    if handle:
        query["entity_handle"] = handle
    rows = list(coll(COLL_COMMENTS).find(query).sort("created_at", 1))
    for row in rows:
        row["_id"] = str(row["_id"])
    return rows


def comment_counts(drawing_id: str) -> dict[str, int]:
    """handle -> number of comments, for painting markers in the viewer."""
    pipeline = [
        {"$match": {"drawing_id": drawing_id}},
        {"$group": {"_id": "$entity_handle", "n": {"$sum": 1}}},
    ]
    return {r["_id"]: r["n"] for r in coll(COLL_COMMENTS).aggregate(pipeline)}


# ---------------------------------------------------------------------------
# Rendered SVG
# ---------------------------------------------------------------------------


def _svg_path(svg_dir: Path, drawing_id: str, layout_name: str) -> Path:
    """Cache path for one rendered layout.

    Layout names come from the file and contain spaces and punctuation
    (`"DMP Layout1"`, `"SECTIONS AND DETAILS"`), so they are sanitised before
    reaching the filesystem.

    Sanitising alone is not injective -- `"Layout 1"`, `"Layout+1"` and
    `"Layout/1"` all collapse to `Layout_1`, which would make two layouts
    share one file and let the second render silently overwrite the first,
    while MongoDB (keyed on the raw name) still reported two distinct
    records. A short digest of the *unsanitised* name is appended so the
    mapping is one-to-one; the readable part is kept so the cache directory
    stays browsable.
    """
    safe_layout = _UNSAFE_FILENAME.sub("_", layout_name) or "layout"
    digest = hashlib.sha256(layout_name.encode("utf-8")).hexdigest()[:8]
    return svg_dir / f"{drawing_id}__{safe_layout}-{digest}.svg.gz"


def save_svg(
    svg_dir: Path,
    drawing_id: str,
    layout_name: str,
    svg: str,
    meta: dict[str, Any],
) -> Path:
    """Write the gzipped SVG to disk and record its metadata in MongoDB.

    Written to a temporary file and renamed, so a reader never sees a
    half-written SVG if the process dies mid-render.
    """
    svg_dir.mkdir(parents=True, exist_ok=True)
    target = _svg_path(svg_dir, drawing_id, layout_name)
    tmp = target.with_suffix(".tmp")

    raw = svg.encode("utf-8")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(raw)
    tmp.replace(target)

    record = {
        "_id": f"{drawing_id}:{layout_name}",
        "drawing_id": drawing_id,
        "layout": layout_name,
        "path": str(target),
        "bytes_raw": len(raw),
        "bytes_gzip": target.stat().st_size,
        "rendered_at": _utcnow(),
        **meta,
    }
    coll(COLL_RENDERS).replace_one({"_id": record["_id"]}, record, upsert=True)
    log.info(
        "cached svg",
        extra={
            "drawing_id": drawing_id,
            "layout": layout_name,
            "bytes_raw": len(raw),
            "bytes_gzip": record["bytes_gzip"],
        },
    )
    return target


def load_svg_gzip(
    svg_dir: Path, drawing_id: str, layout_name: str
) -> bytes | None:
    """The cached gzip bytes for one layout, ready to serve as-is, or None.

    Returned still compressed: the API sets `Content-Encoding: gzip` and never
    decompresses a 36 MB document just to have the browser recompress it.
    """
    path = _svg_path(svg_dir, drawing_id, layout_name)
    if not path.exists():
        return None
    return path.read_bytes()


def get_render_meta(drawing_id: str, layout_name: str) -> dict[str, Any] | None:
    """Stored metadata for one cached render."""
    return coll(COLL_RENDERS).find_one({"_id": f"{drawing_id}:{layout_name}"})


def list_render_meta(drawing_id: str) -> list[dict[str, Any]]:
    """Every cached render for one drawing."""
    return list(coll(COLL_RENDERS).find({"drawing_id": drawing_id}))


# --- One module per UPLIFT spec ----------------------------------------------
#
# Every Wave 2 subagent owns one of the files below. The split exists so that
# eight specs worked on at the same time never touch the same line, and so that
# the merge can be done one at a time in priority order.
#
# The imports are written out IN FULL here once, by the coordinator, precisely
# so that no subagent needs to add a line to this file. New functions are
# reached as `store.store_spatial.join_labels(...)` and so on — not by adding a
# re-export here.
#
# Placed at the bottom, not the top: those modules take `coll` from `.mongo`
# directly, and an import at the head of this file would be circular.
from . import (  # noqa: E402,F401
    store_geometry,   # E — UPLIFT-07
    store_landuse,    # A — UPLIFT-02, then UPLIFT-14
    store_schedule,   # the drawn schedule table vs the measured area
    store_recipes,    # H — UPLIFT-13, then UPLIFT-09
    store_spatial,    # C — UPLIFT-03, then UPLIFT-05
    store_stats,      # D — UPLIFT-06
    store_tables,     # F — UPLIFT-11, then UPLIFT-04
)
