"""The drawing's own schedule, and what happens when you believe it.

`drawn_tables.py` reads a drawn table as a grid. This turns a grid into an answer:
which column identifies a thing, which column measures it, and -- the part
that makes it worth having -- whether the measurement agrees with the geometry
the same drawing carries.

Two sources for one quantity is the whole point. Until now every area in this
system was measured from an outline. The reference drawing also *states* its
areas, in sixty tables holding 2,449 rows, written by the engineer who drew the
plots. Where the two agree, an answer can be given with unusual confidence.
Where they disagree, that is not a nuisance to be smoothed over: it is a defect
in the drawing, and finding those is the job.

Measured on the reference drawing: 2,443 of 2,449 scheduled plots sit inside
exactly one outline, 2,383 of those agree within 2 %, and **60 do not**. The
six with no outline and the sixty that disagree are reported as themselves,
never averaged away.

Nothing here is specific to plots. The identifier column is found from its
values, the layer carrying the labels is found by seeing which layer's text
actually matches those values, and the meaning of the measured column is
confirmed by agreement rather than assumed. A drawing whose tables are a door
schedule reads the same way and simply finds nothing to agree with.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import evidence as ev
from . import store_spatial as sp
from . import drawn_tables as tbl
from .config import get_settings
from .mongo import COLL_ENTITIES, coll

log = logging.getLogger(__name__)

#: Default agreement band between a stated and a measured quantity.
#:
#: Two percent, and not tighter, because the two numbers are not produced the
#: same way: a schedule is rounded to whole square metres by the person who
#: typed it, and a shoelace area is exact to the vertex coordinates. On a
#: 300 m2 plot a rounded value is already 0.17 % away before anything is
#: wrong. Tighter than this reports arithmetic as a defect.
DEFAULT_TOLERANCE = 0.02

#: How many rows a comparison returns by default. The counts always cover
#: everything; only the listing is cut, and it says so when it is.
DEFAULT_LIMIT = 200

#: Types that can carry a label. Mirrors the spatial join so the two cannot
#: drift apart.
LABEL_TYPES = ("TEXT", "MTEXT")


class ScheduleError(Exception):
    """No schedule could be read, with the reason a reader can act on."""

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


# --- reading the schedule ----------------------------------------------------


def _cache_path(drawing_id: str) -> Path:
    return get_settings().embedded_dir / drawing_id / "schedule.json"


def _source_dxf(drawing: Mapping[str, Any]) -> Path | None:
    """The DXF this drawing was read from, if it is still on disk."""
    for key in ("dxf_path", "converted_path", "source_path"):
        value = drawing.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.suffix.lower() == ".dxf" and path.is_file():
            return path
    name = drawing.get("original_filename") or ""
    if name:
        for candidate in (
            get_settings().svg_dir / "converted" / str(name),
            get_settings().dxf_dir / str(name),
        ):
            if candidate.is_file():
                return candidate
    return None


def read_schedule(
    drawing_id: str, drawing: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Every drawn table in the drawing, merged and interpreted.

    Cached on disk after the first read. Parsing sixty tables out of a 136 MB
    file takes about ten seconds, which is fine to pay once and rude to pay on
    every question.
    """
    from . import store  # local import: `store` imports this module

    drawing = drawing or store.get_drawing(drawing_id) or {}
    cache = _cache_path(drawing_id)
    if cache.is_file():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.debug("unreadable schedule cache for %s", drawing_id)

    source = _source_dxf(drawing)
    if source is None:
        raise ScheduleError(
            "SCHEDULE_NO_SOURCE",
            f"the DXF file for drawing {drawing_id!r} is not on disk, so its "
            "tables cannot be read.",
            "Tables are read from the file itself, not from the stored "
            "entities. Put the file back in `dxf_dir` and try again.",
        )

    found = tbl.read(source)
    grid = tbl.merge_grids(found)
    columns = tbl.describe_columns(grid)
    identifiers = [c for c in columns if c.kind == "identifier"]
    measures = [c for c in columns if c.kind == "measure"]

    rows: list[dict[str, Any]] = []
    if identifiers and measures:
        key_at, value_at = identifiers[0].index, measures[0].index
        for row in grid:
            if len(row) <= max(key_at, value_at):
                continue
            key = tbl.as_number(row[key_at])
            value = tbl.as_number(row[value_at])
            if key is None or value is None:
                continue
            rows.append({"key": int(key), "value": value})

    out = {
        "drawing_id": drawing_id,
        "source": "ACAD_TABLE",
        "tables": [t.as_dict() for t in found],
        "table_count": len(found),
        "row_count": len(rows),
        "columns": [
            {
                "index": c.index,
                "role": c.kind,
                "reason": c.reason,
                "filled": c.filled,
                "numeric": c.numeric,
                "distinct": c.unique,
                "min": c.minimum,
                "max": c.maximum,
            }
            for c in columns
        ],
        "key_column": identifiers[0].index if identifiers else None,
        "value_column": measures[0].index if measures else None,
        "rows": rows,
        "note": _schedule_note(found, identifiers, measures, rows),
    }
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except OSError:  # a cache is an optimisation, never the answer
        log.debug("could not cache schedule for %s", drawing_id)
    return out


def _schedule_note(
    found: Sequence[tbl.Table],
    identifiers: Sequence[tbl.Column],
    measures: Sequence[tbl.Column],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    if not found:
        return "this drawing holds no drawn table (ACAD_TABLE)"
    if not identifiers:
        return (
            f"{len(found)} tables were read, but no column behaves like a "
            "numbering, so their rows cannot be turned into a keyed schedule"
        )
    if not measures:
        return (
            f"{len(found)} tables were read with a numbering column, but there "
            "is no other numeric column to take as its value"
        )
    return (
        f"{len(rows)} rows from {len(found)} tables; the key column and the "
        "value column are decided from the spread of their values, not from a "
        "column heading -- this schedule has no header row at all"
    )


# --- which layer carries the labels ------------------------------------------


def _label_layer_for(
    drawing_id: str, layout: str, keys: set[int]
) -> tuple[str | None, list[dict[str, Any]]]:
    """The layer whose text actually matches the schedule's own keys.

    Not a name, and not a guess. Every layer that carries text in this layout
    is scored by how much of the schedule it accounts for, and the winner is
    reported alongside the runners-up so the choice can be argued with.
    """
    cursor = coll(COLL_ENTITIES).find(
        {
            "drawing_id": drawing_id,
            "layout": layout,
            "type": {"$in": list(LABEL_TYPES)},
            "text": {"$ne": None},
        },
        {"layer": 1, "text": 1},
    )
    tally: dict[str, set[int]] = {}
    total: dict[str, int] = {}
    for doc in cursor:
        layer = str(doc.get("layer") or "")
        total[layer] = total.get(layer, 0) + 1
        value = tbl.as_number(str(doc.get("text") or ""))
        if value is None or not float(value).is_integer():
            continue
        if int(value) in keys:
            tally.setdefault(layer, set()).add(int(value))

    scored = sorted(
        (
            {
                "layer": layer,
                "matched": len(matched),
                "texts": total.get(layer, 0),
                "coverage": len(matched) / len(keys) if keys else 0.0,
            }
            for layer, matched in tally.items()
        ),
        key=lambda row: (-row["matched"], row["layer"]),
    )
    best = scored[0]["layer"] if scored and scored[0]["matched"] > 0 else None
    return best, scored[:5]


# --- believing it, and checking it -------------------------------------------


def check_areas(
    drawing_id: str,
    *,
    layout: str,
    label_layer: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    limit: int = DEFAULT_LIMIT,
    only_mismatches: bool = True,
) -> dict[str, Any]:
    """What the schedule says against what the geometry measures.

    The rule for deciding which outline belongs to an identifier comes from
    the data and not from a layer name: the smallest polygon that contains the
    label **and contains no other identifier**. A polygon holding several
    numbers is a block or a site boundary -- it is not the thing being
    numbered -- and on the reference drawing that single rule moves the
    agreement from a meaningless 0 % to 97.5 %.
    """
    from . import store  # local import: `store` imports this module

    drawing = store.get_drawing(drawing_id) or {}
    schedule = read_schedule(drawing_id, drawing)
    stated = {int(row["key"]): float(row["value"]) for row in schedule["rows"]}
    if not stated:
        raise ScheduleError(
            "SCHEDULE_EMPTY",
            f"no keyed schedule can be read from drawing {drawing_id!r}.",
            schedule.get("note") or "Check /schedule to see what was read.",
        )

    chosen, candidates = (
        (label_layer, [])
        if label_layer
        else _label_layer_for(drawing_id, layout, set(stated))
    )
    if chosen is None:
        raise ScheduleError(
            "SCHEDULE_NO_LABELS",
            f"no text layer in layout {layout!r} carries values that match the "
            "schedule's keys.",
            "The schedule was read, but its numbers do not appear as text in "
            "this layout. Try another layout, or name a `label_layer`.",
        )

    layers = sorted(
        {
            str(doc["layer"])
            for doc in coll(COLL_ENTITIES).find(
                {
                    "drawing_id": drawing_id,
                    "layout": layout,
                    "ring_status": "complete",
                },
                {"layer": 1},
            )
            if doc.get("layer")
        }
    )
    if not layers:
        raise ScheduleError(
            "SCHEDULE_NO_RINGS",
            f"there is no closed polygon in layout {layout!r} to compare with.",
            "An area can only be measured from a closed ring; look at "
            "`ring_status` on the entities.",
        )

    joined = sp.join_labels(
        drawing_id,
        layout=layout,
        target_layer=layers,
        label_layer=chosen,
        limit=1_000_000,
    )

    measured: dict[int, tuple[str, str, float]] = {}
    for row in joined.get("rows") or []:
        labels = row.get("labels") or []
        if len(labels) != 1:
            continue  # a container holding several numbers is not the thing
        area = (row.get("target_area") or {}).get("value")
        if area is None:
            continue
        value = tbl.as_number(str(labels[0].get("text") or ""))
        if value is None or not float(value).is_integer():
            continue
        key = int(value)
        best = measured.get(key)
        if best is None or float(area) < best[2]:
            measured[key] = (
                str(row.get("target_handle") or ""),
                str(row.get("target_layer") or ""),
                float(area),
            )

    scope = sp._scope_of(
        drawing_id,
        layout,
        f"the drawing's table schedule compared with ring areas in layout "
        f"{layout!r}",
    )

    comparisons: list[dict[str, Any]] = []
    agree = 0
    for key in sorted(set(stated) & set(measured)):
        handle, layer, area = measured[key]
        said = stated[key]
        gap = abs(area - said)
        relative = gap / said if said else None
        ok = relative is not None and relative <= tolerance
        agree += 1 if ok else 0
        # Both forms, and both named for what they are.
        #
        # A bare `relative` is a figure without a unit, and a reader who has
        # to guess will guess wrong: 43.21 printed under a column headed "%"
        # is how a plot that is off by 4,321 % came to be reported as off by
        # 43 %. Seen on screen, 24 August 2026. The percent field exists so
        # nothing has to multiply, and the ratio keeps its own name.
        row = {
            "key": key,
            "stated": said,
            "measured": area,
            "difference": area - said,
            "difference_ratio": relative,
            "difference_percent": None if relative is None else relative * 100,
            "agrees": ok,
            "handle": handle,
            "layer": layer,
        }
        row["evidence"] = _row_evidence(
            row,
            drawing_id=drawing_id,
            scope=scope,
            layout=layout,
            tolerance=tolerance,
        ).as_dict()
        comparisons.append(row)

    missing = sorted(set(stated) - set(measured))
    unscheduled = sorted(set(measured) - set(stated))
    mismatches = [row for row in comparisons if not row["agrees"]]
    shown = mismatches if only_mismatches else comparisons
    shown = sorted(shown, key=lambda row: -(row["difference_ratio"] or 0))

    return {
        "drawing_id": drawing_id,
        "layout": layout,
        "label_layer": chosen,
        "label_layer_candidates": candidates,
        "tolerance": tolerance,
        "scheduled": len(stated),
        "compared": len(comparisons),
        "agree": agree,
        "disagree": len(mismatches),
        "no_outline": missing,
        "unscheduled": unscheduled[:50],
        "unscheduled_count": len(unscheduled),
        "rows": shown[:limit],
        "rows_truncated": max(0, len(shown) - limit),
        "showing": "mismatches only" if only_mismatches else "everything",
        "method": (
            "the smallest polygon that contains the label AND contains no other "
            "number; the area from a shoelace over the closed ring, compared "
            "with the value written in the drawing's table"
        ),
        "note": _check_note(len(stated), len(comparisons), agree, missing),
        "scope": scope.as_dict(),
    }


def _check_note(
    scheduled: int, compared: int, agree: int, missing: Sequence[int]
) -> str:
    if not compared:
        return (
            "no scheduled number falls inside a singly-labelled polygon, so "
            "there is nothing to compare"
        )
    share = 100 * agree / compared
    parts = [
        f"{compared} of {scheduled} schedule rows have a polygon of their own; "
        f"{agree} agree ({share:.1f}%), {compared - agree} do not"
    ]
    if missing:
        parts.append(
            f"{len(missing)} numbers have no polygon that contains them alone "
            f"({', '.join(str(m) for m in missing[:8])}"
            f"{'...' if len(missing) > 8 else ''})"
        )
    return "; ".join(parts)


def _provenance() -> ev.Provenance:
    """No configuration decides anything here.

    A schedule is read off the drawing and an area is measured from it. There
    is no mapping to verify, so the layer is `NO_CONFIG` and `verified` is
    false -- not because the comparison is doubtful, but because there is
    nothing configured to stand behind it.
    """
    return ev.Provenance(
        config_layer=ev.ConfigLayer.NO_CONFIG,
        note=(
            "the value is read from the drawn table and the area is computed "
            "from coordinates; no config is read"
        ),
    )


def _row_evidence(
    row: Mapping[str, Any],
    *,
    drawing_id: str,
    scope: ev.Scope,
    layout: str,
    tolerance: float,
) -> ev.Evidence:
    """One comparison, with both sides shown.

    The two sides fail for different reasons, which is what makes this worth
    doing. A typed cell can be stale, mistyped, or copied from the row above;
    a shoelace area can only be wrong if the outline is. Neither can quietly
    drag the other along, so when they agree the agreement means something --
    and the contract, not this function, decides how much.
    """
    key = row["key"]
    observations = [
        ev.Observation(
            origin=ev.Origin.ENTITY_TEXT,
            detail=(
                f"the drawing's table records {row['stated']:,.0f} for number "
                f"{key}"
            ),
            observed=f"{row['stated']:.0f}",
            locator=f"ACAD_TABLE:{key}",
            layout=layout,
        ),
        ev.Observation(
            origin=ev.Origin.GEOMETRY,
            detail=(
                f"closed ring {row['handle']} on layer {row['layer']!r} "
                f"measures {row['measured']:,.2f}"
            ),
            observed=f"{row['measured']:.2f}",
            locator=str(row["handle"]),
            layout=layout,
        ),
    ]
    if row["agrees"]:
        # The token is the figure itself. Without it nothing in the claim can
        # be spoken for, and a value the drawing literally prints would be
        # graded `inferred` -- which is the contract working correctly on a
        # claim that forgot to say what it was claiming.
        #
        # It stops at `stated`, never `corroborated`, and that is right:
        # coordinates never speak. The geometry is a check on the figure, not
        # a second voice saying it.
        return ev.Evidence.of(
            ev.Claim(
                value=f"the area of number {key} is {row['stated']:,.0f}",
                tokens=(f"{row['stated']:.0f}", f"{row['stated']}"),
            ),
            drawing_id=drawing_id,
            scope=scope,
            provenance=_provenance(),
            observations=observations,
            not_established=(
                "which unit the table means. The agreement between the figures "
                "is strong, but that table writes no unit anywhere"
            ),
            how_to_verify=(
                "open the table in the drawing, or compare it with the area "
                "calculation sheet this drawing refers to"
            ),
        )
    return ev.Evidence.of(
        ev.Claim(
            value=(
                f"the schedule and the geometry do NOT agree about number "
                f"{key}: {row['stated']:,.0f} against {row['measured']:,.2f}"
            )
        ),
        drawing_id=drawing_id,
        scope=scope,
        provenance=_provenance(),
        observations=observations,
        not_established=(
            "which of the two is right. What is measured is that they differ "
            f"by more than {tolerance:.0%}, not that the outline is wrong"
        ),
        how_to_verify=(
            "open this plot in the drawing and compare its outline with its "
            "row in the table; a gap this size usually means the table was not "
            "updated when the outline was changed"
        ),
    )


def area_of(
    drawing_id: str,
    key: int,
    *,
    layout: str,
    label_layer: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """One identifier's area from both sources, and whether they agree.

    The answer to "what is the area of plot 2043" that a drawing can actually
    support: 300 m2 because the schedule says so, and 299.9999992 m2 because
    the outline measures so.
    """
    full = check_areas(
        drawing_id,
        layout=layout,
        label_layer=label_layer,
        tolerance=tolerance,
        limit=1_000_000,
        only_mismatches=False,
    )
    for row in full["rows"]:
        if row["key"] == key:
            return {
                **{k: v for k, v in full.items() if k != "rows"},
                "row": row,
                "answer": (
                    f"{row['stated']:,.0f} (recorded in the drawing's table) "
                    f"against {row['measured']:,.1f} (measured from its outline)"
                ),
            }
    reason = (
        "this number is in the schedule but has no polygon that contains it "
        "alone"
        if key in full["no_outline"]
        else "this number is not in the drawing's schedule"
    )
    return {
        **{k: v for k, v in full.items() if k != "rows"},
        "row": None,
        "answer": None,
        "not_measured": reason,
    }
