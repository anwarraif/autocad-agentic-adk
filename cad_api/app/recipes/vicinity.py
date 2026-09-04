"""What is near something, counted over H3 cells.

The question this exists for was relayed from Dr. Adel through Prasang: *how
many houses or villas are within 100 metres of the school?* It is already
answerable exactly — `proximity_count` measures a real distance over the real
geometry — and this recipe does not replace it. It answers the same question
a different way, cheaply, over cells that are already stored, and it says
plainly how the two differ.

**What a hexagonal neighbourhood is.** `grid_disk(cell, k)` returns the cells
within k steps. At resolution 11 the step between neighbouring cell centres is
49.6 m, so k=2 reaches 100.2 m centre to centre — measured, not assumed. But
the disk is a hexagon, not a circle: its corners reach 126.8 m. A parcel 120 m
from the school is counted if it lies towards a corner and not if it lies
towards a flat. That is the approximation, it is stated in every response, and
it is the reason `exact_check` names the call that settles it.

**It measures from the footprint, not from a centre point.** The rings start
from every cell the subject occupies, so "within 100 m of the school" means
from the school, not from the middle of it. That is what a person asking the
question means, and it is why `exact_check` points at `proximity_count` in
`edge` mode: on this drawing, centroid mode at the same radius answers 45
where edge mode answers 111, and the difference is the school's own 74 m.

**It is not a walking distance.** Nothing here follows a road. Two plots on
either side of a wall are neighbours to this recipe and a long walk apart in
the world.

Zero new tools: this rides `run_analysis` like every other recipe, and the
agent chooses its name and fills its parameters without writing any logic.
"""

from __future__ import annotations

import math
from typing import Any, Final, Mapping

from .. import evidence as ev
from .. import geo_h3
from .. import landuse
from .. import store_landuse as landuse_store
from ..extract import RING_TYPES
from ..mongo import COLL_ENTITIES, coll
from .registry import (
    MAX_ROWS_RETURNED,
    Param,
    Recipe,
    RecipeRefused,
    register,
)

try:  # pragma: no cover - the dependency is pinned
    import h3
except ImportError:  # pragma: no cover
    h3 = None  # type: ignore[assignment]


#: The resolution the rings are counted at, when nobody names one.
#:
#: Not the resolution the cells are STORED at. Storage is fine-grained so that
#: a 200 m2 plot is resolvable at all; a neighbourhood ring wants cells about
#: the size of the step being taken, and at resolution 13 a 100 m reach would
#: be k=24 and 1,801 cells per subject cell to reach the same ground.
DEFAULT_RES: Final[int] = 11

#: Rings, when nobody names a number. Two, because that is 100 m at the
#: default resolution — which is the question that was asked.
DEFAULT_K: Final[int] = 2

#: The most rings one call may take. `grid_disk` grows as 3k²+3k+1, so k=25 is
#: 1,951 cells per subject cell before any union; past that this stops being a
#: neighbourhood question and becomes a whole-drawing one, which
#: `parcel_inventory` already answers (G7).
MAX_K: Final[int] = 25

#: Subjects one call may answer for. Nine schools is the question; fifty is
#: already a report rather than an answer, and 2,380 houses would be a
#: cross-product nobody reads.
MAX_SUBJECTS: Final[int] = 50


def _require_h3() -> None:
    if h3 is None:  # pragma: no cover
        raise RecipeRefused(
            "RECIPE_NO_CONFIG",
            "the h3 package is not installed in this image.",
            "This recipe reads the H3 cells written at ingest. Without the "
            "package there are no cells to read.",
        )


def _reach_metres(res: int, k: int) -> dict[str, float]:
    """How far k rings actually reach at this resolution, both ways.

    Two numbers because a hexagon has two answers and quoting one of them is
    how an approximation turns into a wrong figure. The centre-to-centre reach
    is what "about 100 m" means; the vertex reach is the furthest ground the
    disk actually covers.
    """
    edge = h3.average_hexagon_edge_length(res, unit="m")
    step = math.sqrt(3) * edge
    return {
        "cell_edge_m": round(edge, 3),
        "step_between_cells_m": round(step, 3),
        "reach_centre_to_centre_m": round(k * step, 1),
        "reach_to_disk_corner_m": round(k * step + edge, 1),
    }


def _subjects(
    drawing_id: str, layout: str, subject: str, classified: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """The entities the rings are drawn around, and how they were chosen.

    A handle first, then a land use. That order matters: `education` is not a
    handle and `205D153` is not a land use, so there is no ambiguity to
    resolve — but checking the handle first means a drawing that ever names a
    layer `205D153` cannot silently change what the parameter means.
    """
    wanted = (subject or "").strip()
    if not wanted:
        raise RecipeRefused(
            "RECIPE_PARAM_REQUIRED",
            "`subject` is required.",
            "Give a handle — the objects the rings are drawn around — or a "
            "land use such as `education` to take every parcel of that kind.",
        )

    row = coll(COLL_ENTITIES).find_one(
        {"_id": f"{drawing_id}:{wanted.upper()}"},
        {"handle": 1, "layer": 1, "layout": 1, "h3_cell": 1, "h3_cells": 1},
    )
    if row:
        return [row], f"handle {wanted.upper()}"

    layers = sorted(
        name for name, use in classified.items() if use.use == wanted.lower()
    )
    if not layers:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"`subject` {wanted!r} is neither a handle in this drawing nor a "
            "land use it classifies.",
            "Land uses this drawing knows: "
            + (", ".join(sorted({u.use for u in classified.values()})) or "none")
            + ". A handle comes from any answer that names one.",
        )
    rows = list(
        coll(COLL_ENTITIES)
        .find(
            {
                "drawing_id": drawing_id,
                "layout": layout,
                "layer": {"$in": layers},
                "type": {"$in": sorted(RING_TYPES)},
            },
            {"handle": 1, "layer": 1, "layout": 1, "h3_cell": 1, "h3_cells": 1},
        )
        .limit(MAX_SUBJECTS + 1)
    )
    if len(rows) > MAX_SUBJECTS:
        raise RecipeRefused(
            "RECIPE_INPUT_TOO_LARGE",
            f"land use {wanted!r} has more than {MAX_SUBJECTS} parcels in this "
            "layout.",
            "Draw the rings around a smaller set: name one handle, or ask "
            "about a land use with fewer parcels. A ring around every house is "
            "a cross-product, not an answer.",
        )
    return rows, f"every {wanted.lower()} parcel"


def _index_by_cell(
    drawing_id: str,
    scope_layout: str,
    classified: Mapping[str, Any],
    resolution: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, set[str]],
    dict[str, dict[str, Any]],
    int,
]:
    """Every classified object in this layout, indexed by cell.

    Parcels are indexed by their REPRESENTATIVE cell and networks by their
    whole covering, and the split is not a convenience. An area is where its
    centre is; counting a parcel once keeps the totals independent of the
    resolution, exactly as `/geo` does. A road is not anywhere in
    particular — it runs — so a road that crosses the neighbourhood belongs to
    it even when both its ends are far outside.

    Shared by `h3_vicinity` and `catchment_coverage` rather than written
    twice: the two recipes ask different questions of the SAME index, and two
    copies of this loop would answer them from two populations that drift
    apart at the first edit.
    """
    parcels_by_cell: dict[str, list[dict[str, Any]]] = {}
    network_by_cell: dict[str, set[str]] = {}
    network_rows: dict[str, dict[str, Any]] = {}
    counted_parcels = 0

    for row in coll(COLL_ENTITIES).find(
        {"drawing_id": drawing_id, "layout": scope_layout, "h3_cell": {"$ne": None}},
        {
            "handle": 1,
            "layer": 1,
            "type": 1,
            "area": 1,
            "length": 1,
            "h3_cell": 1,
            "h3_cells": 1,
        },
    ):
        land_use = classified.get(str(row.get("layer") or ""))
        if land_use is None:
            continue
        if land_use.is_parcel:
            if str(row.get("type") or "") not in RING_TYPES:
                continue
            parent = h3.cell_to_parent(row["h3_cell"], resolution)
            parcels_by_cell.setdefault(parent, []).append(
                {
                    "handle": row.get("handle"),
                    "use": land_use.use,
                    "area": row.get("area"),
                    "layer": row.get("layer"),
                }
            )
            counted_parcels += 1
        elif land_use.is_network:
            handle = str(row.get("handle") or "")
            network_rows[handle] = {
                "layer": row.get("layer"),
                "use": land_use.use,
                "length": row.get("length"),
            }
            for cell in row.get("h3_cells") or [row["h3_cell"]]:
                network_by_cell.setdefault(
                    h3.cell_to_parent(cell, resolution), set()
                ).add(handle)

    return parcels_by_cell, network_by_cell, network_rows, counted_parcels


def _run_h3_vicinity(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    subject: str,
    k: float | None,
    res: float | None,
    bands: bool | None,
    outline: bool | None,
) -> dict[str, Any]:
    _require_h3()
    scope_layout = (layout or "").strip() or "Model"

    config = landuse.for_drawing(drawing_id)
    choice = landuse_store.crs_for_drawing(drawing_id, config=config)
    stored_res = (
        config.h3_resolution if config else landuse.DEFAULT_H3_RESOLUTION
    )
    if choice.crs is None:
        raise RecipeRefused(
            "RECIPE_NO_CONFIG",
            "this drawing is not georeferenced, so it has no cells to ring.",
            "A drawing gets cells once it has a coordinate system — either a "
            "GEODATA object in the file or a `crs:` block in its config. "
            "`proximity_count` measures a radius in drawing coordinates and "
            "needs neither.",
        )

    rings = DEFAULT_K if k is None else int(k)
    resolution = DEFAULT_RES if res is None else int(res)
    if not 1 <= rings <= MAX_K:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"`k` is {rings}; it counts rings and must be between 1 and {MAX_K}.",
            f"At resolution {resolution} one ring is about "
            f"{_reach_metres(resolution, 1)['reach_centre_to_centre_m']} m.",
        )
    if not 0 <= resolution <= stored_res:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"`res` is {resolution}; this drawing's cells are stored at "
            f"{stored_res}.",
            f"Ask for {stored_res} or coarser. A finer ring would be drawn "
            "around cells that were never computed.",
        )

    pattern_set = landuse.patterns()
    classified: dict[str, landuse.LandUse] = {}
    for name in coll(COLL_ENTITIES).distinct("layer", {"drawing_id": drawing_id}):
        text = str(name or "")
        if not text:
            continue
        hit = landuse.classify(
            drawing_id, text, config=config, pattern_set=pattern_set
        )
        if hit is not None:
            classified[text] = hit

    subjects, chosen_by = _subjects(drawing_id, scope_layout, subject, classified)

    (
        parcels_by_cell,
        network_by_cell,
        network_rows,
        counted_parcels,
    ) = _index_by_cell(drawing_id, scope_layout, classified, resolution)

    # --- one row per subject ------------------------------------------------
    reach = _reach_metres(resolution, rings)
    area_unit = units.get("area_unit")
    length_unit = units.get("length_unit")
    rows: list[dict[str, Any]] = []
    without_cells: list[str] = []

    for item in subjects:
        cells = item.get("h3_cells") or ([item["h3_cell"]] if item.get("h3_cell") else [])
        if not cells:
            without_cells.append(str(item.get("handle") or ""))
            continue
        own = {h3.cell_to_parent(c, resolution) for c in cells}
        disk: set[str] = set()
        for cell in own:
            disk.update(h3.grid_disk(cell, rings))

        # Which ring each cell sits on, measured from the subject's whole
        # footprint rather than from one point: a cell is in band r when r is
        # the FEWEST steps from any of the subject's own cells. Built by
        # growing the disk one ring at a time and claiming what is new, which
        # is the same walk the disk itself was built by — so a band can never
        # contain a cell the disk does not.
        band_of: dict[str, int] = {}
        if bands:
            grown: set[str] = set()
            for ring in range(rings + 1):
                reached: set[str] = set()
                for cell in own:
                    reached.update(h3.grid_disk(cell, ring))
                for cell in reached - grown:
                    band_of[cell] = ring
                grown = reached

        counts: dict[str, int] = {}
        areas: dict[str, float] = {}
        by_band: dict[int, dict[str, int]] = {}
        neighbours: list[str] = []
        self_handle = str(item.get("handle") or "")
        seen: set[str] = set()
        for cell in disk:
            for parcel in parcels_by_cell.get(cell, ()):
                handle = str(parcel["handle"])
                if handle == self_handle or handle in seen:
                    continue
                seen.add(handle)
                counts[parcel["use"]] = counts.get(parcel["use"], 0) + 1
                if bands:
                    ring = band_of.get(cell, rings)
                    slot = by_band.setdefault(ring, {})
                    slot[parcel["use"]] = slot.get(parcel["use"], 0) + 1
                if area_unit and isinstance(parcel["area"], (int, float)):
                    areas[parcel["use"]] = round(
                        areas.get(parcel["use"], 0.0) + float(parcel["area"]), 3
                    )
                neighbours.append(handle)

        network: dict[str, dict[str, Any]] = {}
        for cell in disk:
            for handle in network_by_cell.get(cell, ()):
                entry = network_rows[handle]
                bucket = network.setdefault(
                    str(entry["layer"]),
                    {"objects": 0, "length": 0.0, "length_unit": length_unit,
                     "handles_not_listed": True},
                )
                if handle in bucket.setdefault("_seen", set()):
                    continue
                bucket["_seen"].add(handle)
                bucket["objects"] += 1
                if isinstance(entry["length"], (int, float)):
                    bucket["length"] = round(bucket["length"] + float(entry["length"]), 3)
        for bucket in network.values():
            bucket.pop("_seen", None)

        rows.append(
            {
                "handle": self_handle,
                "layer": item.get("layer"),
                "land_use": (classified.get(str(item.get("layer") or "")) or None)
                and classified[str(item.get("layer"))].use,
                "own_cells": len(own),
                "disk_cells": len(disk),
                "counts": dict(sorted(counts.items())),
                "area": dict(sorted(areas.items())) if area_unit else None,
                "area_unit": area_unit,
                "network": network,
                "neighbours_sample": sorted(neighbours)[:20],
                "neighbours_total": len(neighbours),
                "bands": (
                    None
                    if not bands
                    else [
                        {
                            "ring": ring,
                            "from_m": round(max(0, ring - 1) * reach["step_between_cells_m"], 1)
                            if ring
                            else 0.0,
                            "to_m": round(ring * reach["step_between_cells_m"], 1),
                            "counts": dict(sorted(by_band.get(ring, {}).items())),
                            "total": sum(by_band.get(ring, {}).values()),
                        }
                        for ring in range(rings + 1)
                    ]
                ),
                "outline": geo_h3.outline_of(disk) if outline else None,
            }
        )

    # Which layers this drawing calls a network, and what it means when none
    # do. An empty `network` block reads as "no roads near the school", and on
    # this drawing that would be false: the road centrelines are there, on a
    # layer the config has never been given a `role: network` entry for. The
    # absence of a configured role is not the absence of roads, and the two
    # must not look the same (G3).
    network_layers = sorted(n for n, use in classified.items() if use.is_network)
    if network_layers:
        network_note = (
            "lengths are Σ over the network objects whose cells reach the "
            "disk, per layer. An object is counted whole even when only part "
            "of it is inside"
        )
    else:
        network_note = (
            "this drawing classifies NO layer as a network, so `network` is "
            "empty for every subject. That is a gap in its land use config, "
            "not a statement that there are no roads nearby — the road "
            "centrelines are in the drawing and are simply unclassified. "
            "Give the layer a `role: network` entry in "
            f"cad_api/app/landuse/{landuse.config_path(drawing_id).name} and "
            "the lengths appear here"
        )

    summary = landuse_store.land_use_summary(drawing_id, scope_layout)
    body: dict[str, Any] = {
        "subject": subject,
        "chosen_by": chosen_by,
        "subjects": rows,
        "subjects_without_cells": without_cells,
        "method": {
            "resolution": resolution,
            "stored_resolution": stored_res,
            "rings": rings,
            **reach,
            "how": (
                f"every cell within {rings} ring(s) of the subject's own cells "
                f"at resolution {resolution}, then the parcels whose "
                "representative cell falls in that set. Coarser cells are "
                "derived from the stored ones with cell_to_parent"
            ),
        },
        "bands_note": (
            None
            if not bands
            else (
                "one row per ring, measured in steps from the subject's own "
                "cells rather than from a point. `from_m`/`to_m` are that "
                f"step in metres — {reach['step_between_cells_m']} m at "
                f"resolution {resolution} — and carry the same approximation "
                "as the total: a band is made of whole cells, so its edge is "
                "a hexagon boundary and not a circle. Ring 0 is the subject's "
                "own footprint; the bands sum to the counts above"
            )
        ),
        "outline_note": (
            None
            if not outline
            else (
                "each subject's `outline` is the boundary of its whole disk, "
                "dissolved into one shape: the shared edges between "
                "neighbouring cells are gone. It is still the hexagons' own "
                "edges, so it is exactly as approximate as the counts are"
            )
        ),
        "network_layers": network_layers,
        "network_note": network_note,
        "counted_from": (
            "the rings start from EVERY cell the subject occupies, so the "
            "reach is measured outward from its footprint rather than from a "
            "centre point — 'within 100 m of the school' rather than 'within "
            "100 m of the middle of the school'. A parcel is counted once, at "
            "its representative cell; a network object is counted when ANY of "
            "its cells is in the disk, because a road that crosses the "
            "neighbourhood belongs to it even when both its ends are outside"
        ),
        "caveat": (
            f"this is a hexagonal neighbourhood, not a circle. At resolution "
            f"{resolution}, {rings} ring(s) reach "
            f"{reach['reach_centre_to_centre_m']} m from cell centre to cell "
            f"centre and {reach['reach_to_disk_corner_m']} m to the disk's "
            "corners, so an object near a corner is included at a distance an "
            "object near a flat side would not be. It is also a straight-line "
            "neighbourhood: nothing here follows a road, and two plots either "
            "side of a wall are neighbours to this recipe"
        ),
        "exact_check": {
            # `tool`, not `recipe`. proximity_count is a tool of its own and
            # has never been a recipe, so the block used to name a call that
            # cannot be made: `run_analysis` with recipe `proximity_count`
            # returns RECIPE_UNKNOWN and the catalogue. An approximation whose
            # own instructions for checking it fail is worse than one that
            # names no check, because the failure looks like the check.
            "tool": "proximity_count",
            "why": (
                "it measures a real distance in drawing units, with no hexagon "
                "in between. Use it to settle any figure from here that matters"
            ),
            "call": {
                "tool": "proximity_count",
                "params": {
                    "drawing_id": drawing_id,
                    "layout": scope_layout,
                    # Comma-separated, which is what the tool and the route
                    # both take. A JSON list here would be one more thing for
                    # the caller to get right on a call it was handed.
                    "around_layers": ",".join(
                        sorted({str(r.get("layer")) for r in subjects if r.get("layer")})
                    ),
                    "count_layers": "<the layers of the land use being counted, comma-separated>",
                    "radius": reach["reach_centre_to_centre_m"],
                    # `edge`, not `centroid`, and this is the whole of making
                    # the two comparable. The rings here start from EVERY cell
                    # the subject occupies, so they reach outward from its
                    # boundary; `edge` measures from the boundary too.
                    # `centroid` measures centre to centre, which for a school
                    # 74 m across is a different question and answers it with
                    # roughly a third of the number — measured on this drawing:
                    # 45 against 111 for the same school at the same radius.
                    "measure_from": "edge",
                },
            },
            "bases_differ": (
                "the two are not expected to be identical. This recipe rings "
                "outward from every cell the subject occupies, so it measures "
                "from the subject's footprint — which is why the call above "
                "uses `edge`. Against that basis it USUALLY runs high, "
                "because a cell is counted whole: the disk reaches "
                f"{reach['reach_centre_to_centre_m']} m centre to centre but "
                f"{reach['reach_to_disk_corner_m']} m into its corners, and "
                "everything in a counted cell is counted. But the difference "
                "is NOT one-sided, and a claim that it is would be wrong: a "
                "parcel that lies inside the radius but beyond a flat side of "
                "the disk falls in no counted cell and is missed, so a subject "
                "can come out slightly below. Measured over one reference "
                "drawing's nine schools the spread ran from a little under to "
                "about a quarter over. Say which of the two a quoted figure "
                "came from"
            ),
            "do_not_compare_with": (
                "proximity_count in `centroid` mode at the same radius. That "
                "measures centre to centre and will be far lower for any "
                "subject larger than a cell — not a contradiction, a "
                "different question"
            ),
        },
        "population": {
            "parcels_indexed": counted_parcels,
            "network_objects_indexed": len(network_rows),
            "note": (
                "the population these counts are drawn from: every classified "
                "parcel and network object in this layout that has a cell. An "
                "object with no cell is in `subjects_without_cells` when it is "
                "a subject, and is simply absent when it is not — "
                "GET /drawings/{id}/geo reports how many those are"
            ),
        },
        "not_measured": (
            "the true distance to anything. Nothing here is a measurement of "
            "a gap: an object is in or out according to which cell it falls "
            "in, and the cell is up to "
            f"{reach['reach_to_disk_corner_m'] - reach['reach_centre_to_centre_m']:.1f} m "
            "wider than the reach quoted. Walking distance is not measured "
            "either, and neither is whether anything is actually reachable"
        ),
        "scope_note": (
            f"layout {scope_layout!r}; {len(rows)} subject(s) chosen by "
            f"{chosen_by}; counts over {counted_parcels} classified parcels; "
            f"areas in {area_unit or 'no unit the drawing states'}"
        ),
        "evidence": summary.get("evidence") if summary else None,
        "land_use_basis": (
            "the labels on these counts are this drawing's own land use "
            "classification, unchanged. `evidence` above is the one "
            "land_use_summary publishes for it; this recipe establishes no "
            "classification of its own"
        ),
    }
    return body


register(
    Recipe(
        name="h3_vicinity",
        answers=(
            "how many parcels of each land use, and how much network, lie in "
            "the cell neighbourhood around one object or around every parcel "
            "of one land use"
        ),
        when_to_use=(
            "'how many houses are within 100 m of each school', 'what is "
            "around this mosque'. It is fast and approximate — a hexagon, not "
            "a circle. When the figure has to be exact, or when the drawing "
            "has no coordinate system, use proximity_count instead."
        ),
        params=(
            Param(
                "subject",
                "text",
                "what the rings are drawn around: a handle, or a land use "
                "such as `education` to take every parcel of that kind.",
                required=True,
            ),
            Param(
                "k",
                "number",
                f"how many rings. Default {DEFAULT_K}, which is about 100 m at "
                f"the default resolution. The response says what k really "
                f"reaches, both to cell centres and to the disk's corners.",
                default=DEFAULT_K,
            ),
            Param(
                "bands",
                "flag",
                "break the counts down by ring, so 'how many in the first "
                "50 m' is answerable without asking again. The bands sum to "
                "the total for the same k.",
                default=False,
            ),
            Param(
                "outline",
                "flag",
                "return each subject's neighbourhood as ONE dissolved "
                "MultiPolygon boundary, for drawing a catchment. Without it a "
                "map has to draw every hexagon and every shared edge twice.",
                default=False,
            ),
            Param(
                "res",
                "number",
                f"the resolution the rings are counted at. Default "
                f"{DEFAULT_RES}; coarser rings cover more ground per step and "
                "cannot be finer than the resolution the cells are stored at.",
                default=DEFAULT_RES,
            ),
        ),
        returns=(
            "subjects[].counts",
            "subjects[].bands",
            "subjects[].outline",
            "subjects[].network",
            "method",
            "caveat",
            "exact_check",
            "not_measured",
        ),
        built_on=(
            "geo_h3.outline_of",
            "store_landuse.crs_for_drawing",
            "store_landuse.land_use_summary",
            "landuse.classify",
            "h3.grid_disk",
            "h3.cell_to_parent",
        ),
        limits={
            "rings": MAX_K,
            "subjects": MAX_SUBJECTS,
            "neighbours_listed_per_subject": 20,
        },
        carries_meaning=True,
        run=_run_h3_vicinity,
    )
)


# =============================================================================
# catchment_coverage — the PDF's 400/800 m service catchment, answered radially
# =============================================================================
#
# The RCDC poster names "catchment 400/800 m" as a Spatial Analysis element.
# This is that question, and it is deliberately the SAME machinery as
# `h3_vicinity` asking a different one: vicinity reports what surrounds each
# subject, catchment reports the UNION of every subject's reach and then names
# what falls OUTSIDE it. The complement is the whole point. A coverage figure
# that cannot name the parcels it left out is a number nobody can act on.

#: Radii, in metres, when nobody names any. Straight from the poster.
DEFAULT_RADII_M: Final[tuple[float, ...]] = (400.0, 800.0)

#: The land use whose coverage is measured, when nobody names one.
DEFAULT_SERVED_USE: Final[str] = "residential"

#: The most radii one call may take. Each one is a full disk union over every
#: amenity, and a list of ten is a report rather than an answer.
MAX_RADII: Final[int] = 5

#: The furthest a radius may reach. Past this the disk stops being a
#: neighbourhood and `parcel_inventory` answers the whole-drawing question.
MAX_RADIUS_M: Final[float] = 5000.0


def _rings_for_radius(res: int, radius_m: float) -> int:
    """How many rings approximate `radius_m` at this resolution.

    The inverse of `_reach_metres`, and it rounds rather than truncating: at
    resolution 11 one step is about 50 m, so truncating a 400 m ask would
    quietly deliver 350 m. Each band states the reach actually used, so the
    approximation is visible rather than implied.
    """
    edge = h3.average_hexagon_edge_length(res, unit="m")
    step = math.sqrt(3) * edge
    return max(1, int(round(radius_m / step)))


def _parse_radii(raw: Any) -> list[float]:
    """`radii` as metres. Accepts one number or a comma-separated list."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return list(DEFAULT_RADII_M)
    text = str(raw).replace(" ", "")
    out: list[float] = []
    for piece in text.split(","):
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            raise RecipeRefused(
                "RECIPE_PARAM_INVALID",
                f"`radii` contains {piece!r}, which is not a number of metres.",
                "Give one radius or several, in metres, like `400` or "
                "`400,800`.",
            ) from None
        if not 0 < value <= MAX_RADIUS_M:
            raise RecipeRefused(
                "RECIPE_PARAM_INVALID",
                f"`radii` contains {value:g} m; each radius must be above 0 "
                f"and at most {MAX_RADIUS_M:g} m.",
                "A catchment larger than that is a whole-drawing question, "
                "which `parcel_inventory` already answers.",
            )
        out.append(value)
    if not out:
        return list(DEFAULT_RADII_M)
    if len(out) > MAX_RADII:
        raise RecipeRefused(
            "RECIPE_INPUT_TOO_LARGE",
            f"`radii` names {len(out)} distances; the limit is {MAX_RADII}.",
            "Each radius is a full disk union over every amenity. Ask for the "
            "few that matter, such as `400,800`.",
        )
    return sorted(dict.fromkeys(out))


def _run_catchment_coverage(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    amenity: str,
    radii: Any,
    served: str | None,
    res: float | None,
) -> dict[str, Any]:
    _require_h3()
    scope_layout = (layout or "").strip() or "Model"

    config = landuse.for_drawing(drawing_id)
    choice = landuse_store.crs_for_drawing(drawing_id, config=config)
    stored_res = config.h3_resolution if config else landuse.DEFAULT_H3_RESOLUTION
    if choice.crs is None:
        raise RecipeRefused(
            "RECIPE_NO_CONFIG",
            "this drawing is not georeferenced, so it has no cells to draw a "
            "catchment from.",
            "A drawing gets cells once it has a coordinate system, either a "
            "GEODATA object in the file or a `crs:` block in its config. "
            "`proximity_count` measures a radius in drawing coordinates and "
            "needs neither.",
        )

    resolution = DEFAULT_RES if res is None else int(res)
    if not 0 <= resolution <= stored_res:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"`res` is {resolution}; this drawing's cells are stored at "
            f"{stored_res}.",
            f"Ask for {stored_res} or coarser. A finer catchment would be "
            "drawn around cells that were never computed.",
        )

    wanted_radii = _parse_radii(radii)
    served_use = (served or DEFAULT_SERVED_USE).strip().lower()

    pattern_set = landuse.patterns()
    classified: dict[str, landuse.LandUse] = {}
    for name in coll(COLL_ENTITIES).distinct("layer", {"drawing_id": drawing_id}):
        text = str(name or "")
        if not text:
            continue
        hit = landuse.classify(
            drawing_id, text, config=config, pattern_set=pattern_set
        )
        if hit is not None:
            classified[text] = hit

    amenities, chosen_by = _subjects(drawing_id, scope_layout, amenity, classified)
    parcels_by_cell, _network_by_cell, _network_rows, counted_parcels = _index_by_cell(
        drawing_id, scope_layout, classified, resolution
    )

    # The population whose coverage is measured. Held as one flat list so that
    # served is a set-membership test and unserved is its complement, rather
    # than two independent counts that can disagree with each other.
    population: list[dict[str, Any]] = []
    for cell, rows in parcels_by_cell.items():
        for row in rows:
            if row.get("use") == served_use:
                population.append({**row, "cell": cell})

    amenity_cells: set[str] = set()
    without_cells: list[str] = []
    for item in amenities:
        cells = item.get("h3_cells") or (
            [item["h3_cell"]] if item.get("h3_cell") else []
        )
        if not cells:
            without_cells.append(str(item.get("handle") or ""))
            continue
        amenity_cells.update(h3.cell_to_parent(c, resolution) for c in cells)

    bands: list[dict[str, Any]] = []
    for radius in wanted_radii:
        rings = _rings_for_radius(resolution, radius)
        reach = _reach_metres(resolution, rings)
        disk: set[str] = set()
        for cell in amenity_cells:
            disk.update(h3.grid_disk(cell, rings))

        served_rows = [row for row in population if row["cell"] in disk]
        unserved_rows = [row for row in population if row["cell"] not in disk]
        total = len(population)
        listed = unserved_rows[:MAX_ROWS_RETURNED]

        bands.append(
            {
                "radius_m": radius,
                "rings": rings,
                "reach": reach,
                "cells_in_catchment": len(disk),
                "served": len(served_rows),
                "unserved": len(unserved_rows),
                "population": total,
                "coverage_pct": (
                    round(100.0 * len(served_rows) / total, 2) if total else None
                ),
                "unserved_handles": [row.get("handle") for row in listed],
                "unserved_listed": len(listed),
                "unserved_truncated": len(unserved_rows) > len(listed),
                "outline": geo_h3.outline_of(sorted(disk)) if disk else None,
            }
        )

    try:
        summary = landuse_store.land_use_summary(drawing_id, layout=scope_layout)
    except Exception:  # noqa: BLE001 - evidence is a bonus here, never a gate
        summary = None

    return {
        "amenity": {
            "asked_for": amenity,
            "chosen_by": chosen_by,
            "count": len(amenities),
            "with_cells": len(amenities) - len(without_cells),
            "without_cells": without_cells,
        },
        "served_use": served_use,
        "population": len(population),
        "bands": bands,
        "method": {
            "how": (
                "each amenity's own cells are grown by whole rings, and the "
                "rings of every amenity are unioned into ONE catchment; a "
                f"{served_use} parcel counts as served when its representative "
                "cell falls inside that union"
            ),
            "why_a_union": (
                "coverage asks whether a parcel is reached by ANY amenity, "
                "not by each one separately. Summing per-amenity counts would "
                "count every parcel that two amenities both reach twice"
            ),
            "resolution": resolution,
            "cells_stored_at": stored_res,
        },
        "caveat": {
            "shape": (
                "a hexagon disk, not a circle. The reach quoted for each band "
                "is centre to centre; the corners reach further, so a parcel "
                "just outside the radius can be counted as served and one "
                "just inside a flat side can be missed"
            ),
            "distance": (
                "straight-line reach across cells, NOT walking distance. "
                "Nothing here knows whether a road, a wall or a wadi lies "
                "between a house and its school. The pedestrian-network "
                "version of this question is not built"
            ),
            "rounding": (
                "a radius is turned into whole rings, so the reach actually "
                "used is the one named in each band, not the radius asked for"
            ),
        },
        "exact_check": (
            "proximity_count at the same radius measures true distance in "
            "drawing coordinates and is the exact cross-check for any single "
            "amenity. It will not agree exactly, and that is the "
            "approximation above rather than a contradiction"
        ),
        "population_note": {
            "parcels_indexed": counted_parcels,
            "note": (
                f"the {len(population)} {served_use} parcels counted here are "
                "those with a cell in this layout. A parcel with no cell is "
                "absent from both served and unserved; the /geo route reports "
                "how many those are"
            ),
        },
        "not_measured": (
            "the true distance from any parcel to any amenity, whether the "
            "amenity is open, what it serves, or whether anything is "
            "reachable on foot. Capacity is not measured either: an amenity "
            "reaching 900 houses is reported the same as one reaching 9"
        ),
        "scope_note": (
            f"layout {scope_layout!r}; {len(amenities)} amenity parcel(s) "
            f"chosen by {chosen_by}; coverage over {len(population)} "
            f"{served_use} parcels"
        ),
        "evidence": summary.get("evidence") if summary else None,
        "land_use_basis": (
            "the labels on these counts are this drawing's own land use "
            "classification, unchanged. This recipe establishes no "
            "classification of its own"
        ),
    }


register(
    Recipe(
        name="catchment_coverage",
        answers=(
            "what share of one land use lies inside the service catchment of "
            "another, at one or more radii, and which parcels lie outside it"
        ),
        when_to_use=(
            "'how much of the housing is within 400 m of a school', 'which "
            "houses are outside the 800 m mosque catchment'. It is radial and "
            "approximate, hexagon rings rather than a circle, and it is not a "
            "walking distance. For one amenity measured exactly, use "
            "proximity_count."
        ),
        params=(
            Param(
                "amenity",
                "text",
                "what the catchments are drawn around: a handle, or a land "
                "use such as `education` to take every parcel of that kind.",
                required=True,
            ),
            Param(
                "radii",
                "text",
                "the catchment radii in metres, comma separated. Default "
                "`400,800`, the two the RCDC poster names.",
                default="400,800",
            ),
            Param(
                "served",
                "text",
                f"the land use whose coverage is measured. Default "
                f"`{DEFAULT_SERVED_USE}`.",
                default=DEFAULT_SERVED_USE,
            ),
            Param(
                "res",
                "number",
                f"the resolution the catchment is built at. Default "
                f"{DEFAULT_RES}; it cannot be finer than the resolution the "
                "cells are stored at.",
                default=DEFAULT_RES,
            ),
        ),
        returns=(
            "amenity",
            "population",
            "bands[].coverage_pct",
            "bands[].served",
            "bands[].unserved",
            "bands[].unserved_handles",
            "bands[].outline",
            "method",
            "caveat",
            "exact_check",
            "not_measured",
        ),
        built_on=(
            "geo_h3.outline_of",
            "store_landuse.crs_for_drawing",
            "store_landuse.land_use_summary",
            "landuse.classify",
            "h3.grid_disk",
            "h3.cell_to_parent",
        ),
        limits={
            "radii": MAX_RADII,
            "radius_m": MAX_RADIUS_M,
            "amenities": MAX_SUBJECTS,
            "unserved_listed_per_band": MAX_ROWS_RETURNED,
        },
        carries_meaning=True,
        run=_run_catchment_coverage,
    )
)
