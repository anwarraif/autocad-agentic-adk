"""Writing H3 cells onto a drawing's entities.

One function, called from two places — `ingest` after a drawing is stored, and
`scripts/backfill_h3.py` by hand. Two implementations would be two answers to
"which cells is this entity in", and the one nobody runs would be the one that
drifts.

Three properties it is built for, and each is a rule rather than a nicety:

**Per drawing.** Every query starts with `drawing_id`; the cluster runs
`notablescan` and a sweep across all drawings would be a collection scan on
the largest collection in a shared database.

**Batched.** Reads stream through a cursor and writes go out in bulk batches,
so memory is bounded by the batch and not by the drawing. A file ten times the
size is a longer run, not a redesign.

**Idempotent.** Writes are `$set` on `h3_*` keys only, addressed by `_id`.
Running it twice leaves the same state as running it once, and running it over
a drawing whose config changed resolution rewrites every row in place rather
than leaving two generations of cells mixed together.

Nothing is dropped in silence. Every entity that ends with no cell is counted
AND named, grouped by the reason, and the reasons are sentences rather than
codes because the report is read by whoever is asking why a count looks low.

One interaction worth knowing: `store.replace_entities` writes with
`ReplaceOne`, so re-ingesting a drawing removes these fields. That is correct
— the geometry may have changed — and it is why ingest runs this immediately
afterwards rather than assuming the old cells survived.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Mapping

from pymongo import UpdateOne

from . import geo_h3, landuse
from .mongo import COLL_DRAWINGS, COLL_ENTITIES, coll

log = logging.getLogger(__name__)

#: Documents updated per round trip. The same figure `store.replace_entities`
#: uses; there is no reason for the two to differ and every reason for them to
#: be tuned together.
BATCH = 1_000

#: Fields read off each entity. Named rather than taking the whole document:
#: a ring can carry 4,096 vertices and the cursor holds a batch of them at a
#: time, so the projection is the difference between tens of megabytes and a
#: few.
_PROJECTION = {
    "handle": 1,
    "layer": 1,
    "ring": 1,
    "ring_status": 1,
    "path_points": 1,
    "ends": 1,
    "anchor_point": 1,
    "polygon_centroid": 1,
    "bbox_centre": 1,
    # Read only to notice an object with real extent that resolves to one
    # cell. See `geo_h3._coverage_gap`.
    "length": 1,
    "type": 1,
    # Decides whether these coordinates are a position on the earth at all.
    "layout": 1,
}

#: Excluded handles printed by the CLI before it stops listing them. The
#: report itself always carries every one — this is a display limit, and the
#: caller is told the full count beside the sample (G7).
SAMPLE_HANDLES = 20


def _crs_and_resolution(drawing_id: str) -> tuple[Any, int, str | None]:
    """The CRS to index against and the resolution to index at.

    Imported here rather than at module scope: `store_landuse` reaches into
    the store, and this module is imported by `ingest`, which the store also
    reaches. A deferred import closes no cycle because by the time this runs,
    every module involved has finished executing.
    """
    from . import store_landuse

    try:
        config = landuse.for_drawing(drawing_id)
    except landuse.ConfigError as exc:
        return None, landuse.DEFAULT_H3_RESOLUTION, (
            f"this drawing's config cannot be read, so it cannot be indexed: "
            f"{exc.message}"
        )

    choice = store_landuse.crs_for_drawing(drawing_id, config=config)
    resolution = config.h3_resolution if config else landuse.DEFAULT_H3_RESOLUTION
    if choice.crs is None:
        return None, resolution, (
            choice.note
            or (
                "this drawing is not georeferenced: it declares no coordinate "
                "system and has no CRS config, so its coordinates cannot be "
                "turned into cells. Nothing was written"
            )
        )
    return choice.crs, resolution, None



def drawing_facts(drawing_id: str) -> tuple[Any, dict[str, Any]]:
    """The two drawing-level facts cell assignment needs, fetched together.

    Kept as one function because they are read together and were once fetched
    apart: the projection asked for `extents` only and the next line read
    `block_placement` off the result, so the placements were always empty and
    every block-resident entity was refused. Nothing raised and nothing was
    logged; a 990,790 entity drawing simply came out with 19 cells, which
    looks exactly like a drawing that has little in it.

    `extents` bounds where a point may be; `block_placement` says which block
    definitions model space places with the identity transform, so their own
    coordinates are already world coordinates (see
    `geo_h3._block_is_world_placed`). A drawing ingested before
    INGEST_VERSION 7 carries no placements and gets `{}`, which the rule reads
    as "no measurement", not as "no".
    """
    drawing = coll(COLL_DRAWINGS).find_one(
        {"_id": drawing_id}, {"extents": 1, "block_placement": 1}
    ) or {}
    return drawing.get("extents"), drawing.get("block_placement") or {}

def assign_cells(
    drawing_id: str,
    *,
    dry_run: bool = False,
    batch: int = BATCH,
) -> dict[str, Any]:
    """Index one drawing's entities into H3 cells.

    Returns a report and never raises for an ordinary refusal. A drawing that
    cannot be indexed is a normal outcome — seventeen of the eighteen in the
    store are in that position — and it comes back as `ok: false` with a
    reason, not as an exception and not as a run that says it wrote zero rows
    and looks successful.
    """
    started = time.perf_counter()
    report: dict[str, Any] = {
        "drawing_id": drawing_id,
        "ok": False,
        "dry_run": dry_run,
        "reason": None,
        "resolution": None,
        "epsg": None,
        "crs_source_layer": None,
        "entities": 0,
        "assigned": 0,
        "world_placed": 0,
        "excluded": 0,
        "excluded_by_reason": {},
        "by_strategy": {},
        "truncated": [],
        "partial_coverage": [],
        "distinct_cells": 0,
        "elapsed_ms": 0,
    }

    if not geo_h3.available():
        report["reason"] = "the h3 package is not installed in this image"
        return report

    from . import store_landuse

    crs, resolution, refusal = _crs_and_resolution(drawing_id)
    report["resolution"] = resolution
    if crs is None:
        report["reason"] = refusal
        report["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return report

    report["epsg"] = crs.epsg
    report["crs_source_layer"] = store_landuse.crs_for_drawing(
        drawing_id
    ).layer

    # The drawing's own bounding box, read once. Every candidate point is
    # checked against it, because a model-space point outside the drawing is
    # not a position in the drawing — see `geo_h3._inside_extents`.
    extents, placements = drawing_facts(drawing_id)

    excluded: dict[str, list[str]] = {}
    strategies: dict[str, int] = {}
    distinct: set[str] = set()
    pending: list[UpdateOne] = []
    assigned = 0
    world_placed = 0
    total = 0

    cursor = coll(COLL_ENTITIES).find(
        # drawing_id first. Every index on this collection is prefixed with
        # it, and the cluster refuses a query that cannot use one.
        {"drawing_id": drawing_id},
        _PROJECTION,
    ).batch_size(batch)

    for row in cursor:
        total += 1
        handle = str(row.get("handle") or row.get("_id"))
        assignment = geo_h3.cells_for_entity(crs, row, resolution, extents, placements)

        if assignment.cells:
            assigned += 1
            if assignment.world_placed:
                # Stored under a block layout, but its own coordinates put it
                # inside the drawing. Counted so the figure is visible rather
                # than folded silently into the total.
                world_placed += 1
            strategies[assignment.strategy or "unknown"] = (
                strategies.get(assignment.strategy or "unknown", 0) + 1
            )
            distinct.update(assignment.cells)
            if assignment.coverage_note:
                # Indexed, but not indexed WHOLE. Counted apart from an
                # exclusion because the row WILL answer vicinity questions,
                # and answer them short — which is the harder failure to
                # notice of the two.
                report["partial_coverage"].append(
                    {
                        "handle": handle,
                        "layer": row.get("layer"),
                        "note": assignment.coverage_note,
                    }
                )
            if assignment.truncated:
                report["truncated"].append(
                    {
                        "handle": handle,
                        "layer": row.get("layer"),
                        "kept": len(assignment.cells),
                        "total": assignment.total,
                    }
                )
        else:
            reason = assignment.note or "no reason recorded"
            excluded.setdefault(reason, []).append(handle)

        pending.append(
            UpdateOne(
                {"_id": row["_id"]},
                # `$set` on h3_* only. A ReplaceOne here would drop every
                # field this module does not know about, which is all of them.
                {"$set": geo_h3.fields_for_storage(assignment, resolution)},
            )
        )
        if len(pending) >= batch:
            if not dry_run:
                coll(COLL_ENTITIES).bulk_write(pending, ordered=False)
            pending = []

    if pending and not dry_run:
        coll(COLL_ENTITIES).bulk_write(pending, ordered=False)

    report.update(
        ok=True,
        entities=total,
        assigned=assigned,
        excluded=sum(len(v) for v in excluded.values()),
        excluded_by_reason={
            reason: {"count": len(handles), "handles": handles}
            for reason, handles in sorted(excluded.items())
        },
        by_strategy=dict(sorted(strategies.items())),
        world_placed=world_placed,
        distinct_cells=len(distinct),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    log.info(
        "h3 cells assigned",
        extra={
            "drawing_id": drawing_id,
            "assigned": assigned,
            "world_placed": world_placed,
            "excluded": report["excluded"],
            "resolution": resolution,
        },
    )
    return report


def format_report(report: Mapping[str, Any], *, sample: int = SAMPLE_HANDLES) -> str:
    """The report as a human reads it.

    Exclusions are printed by reason with a count and a sample of handles, and
    the sample says how many it is a sample OF. A list that silently stops at
    twenty reads exactly like a list of twenty.
    """
    lines: list[str] = []
    head = f"drawing {report['drawing_id']}"
    if report.get("dry_run"):
        head += "  (dry run — nothing written)"
    lines.append(head)

    if not report.get("ok"):
        lines.append(f"  NOT INDEXED: {report.get('reason')}")
        return "\n".join(lines)

    lines.append(
        f"  resolution {report['resolution']} · EPSG:{report['epsg']} "
        f"· CRS from {report['crs_source_layer']}"
    )
    lines.append(
        f"  entities {report['entities']:,}  assigned {report['assigned']:,}  "
        f"excluded {report['excluded']:,}  distinct cells "
        f"{report['distinct_cells']:,}  ({report['elapsed_ms']:,} ms)"
    )
    if report.get("by_strategy"):
        by = "  ".join(f"{k} {v:,}" for k, v in report["by_strategy"].items())
        lines.append(f"  by strategy: {by}")

    for reason, block in (report.get("excluded_by_reason") or {}).items():
        handles = block["handles"]
        shown = handles[:sample]
        lines.append(f"  EXCLUDED {block['count']:,}: {reason}")
        lines.append(
            f"    handles: {', '.join(shown)}"
            + (
                f"  … and {block['count'] - len(shown):,} more"
                if block["count"] > len(shown)
                else ""
            )
        )

    for row in report.get("truncated") or []:
        lines.append(
            f"  TRUNCATED {row['handle']} on layer {row['layer']!r}: kept "
            f"{row['kept']:,} of {row['total']:,} cells"
        )

    partial = report.get("partial_coverage") or []
    if partial:
        lines.append(
            f"  PARTIAL COVERAGE {len(partial):,}: indexed, but the cells do "
            "not cover the whole object"
        )
        for row in partial[:sample]:
            lines.append(f"    {row['handle']} on layer {row['layer']!r}")
        if len(partial) > sample:
            lines.append(f"    … and {len(partial) - sample:,} more")
        lines.append(f"    reason: {partial[0]['note']}")

    return "\n".join(lines)


def excluded_handles(report: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    """(handle, reason) for every entity that got no cell, all of them."""
    for reason, block in (report.get("excluded_by_reason") or {}).items():
        for handle in block["handles"]:
            yield handle, reason
