"""Which H3 cells an entity occupies.

Separate from `store_landuse.py` for the same reason that file is separate
from `store.py`: this is a self-contained piece of arithmetic with one input
and one output, and the argument about whether it is right should be possible
without reading a store.

Three strategies, chosen by what the entity IS rather than by what type the
DXF calls it:

- a closed ring is covered as a polygon;
- an open path is walked as a chain;
- anything else is placed at its one best point.

The order matters and is not arbitrary. A parcel has both a ring and a
centroid; covering it by its ring is the better answer and the centroid is the
fallback its own coverage falls back TO. Reading the type name instead would
mean deciding that an LWPOLYLINE is a parcel, which is exactly the
drawing-specific assumption G6 forbids — the same DXF type is a plot boundary
on one drawing and a road centreline on the next.

Every cell is stored at ONE resolution, the drawing's own. Coarser levels are
derived with `cell_to_parent` and never stored, because the two routes to a
coarse cell do not agree: for a point on a cell boundary, "the res-9 cell this
point is in" and "the res-9 ancestor of its res-13 cell" can be different
cells, and the mosque is one such point at res 7. Mixing them would produce
totals that disagree by a handful of objects with nothing to explain it. See
`tests/test_geo_h3.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final, Iterable, Mapping, Sequence

from .landuse import crs as crs_math

log = logging.getLogger(__name__)

try:  # pragma: no cover - the dependency is pinned; this guards the import
    import h3
except ImportError:  # pragma: no cover
    h3 = None  # type: ignore[assignment]


#: Cells kept for one entity before the covering is truncated.
#:
#: At the default resolution a cell is 43.87 m2, so this is roughly 0.9 km2 of
#: coverage for a single object — an order of magnitude more than the largest
#: parcel on the reference drawing, and still small enough that the array
#: cannot threaten a document. Nothing is silently dropped when it bites:
#: `truncated` says how many were cut and `total` says how many there were,
#: so an aggregate built from these can tell that it is incomplete (G7).
MAX_CELLS_PER_ENTITY = 20_000

#: Segments walked before an open path gives up on being a contiguous chain.
#:
#: `grid_path_cells` is exact and cheap between nearby cells and refuses over
#: long distances, which is the behaviour wanted: a "path" whose consecutive
#: points are kilometres apart is not a path this index should be describing.
MAX_PATH_SEGMENT_CELLS = 4_000


@dataclass(frozen=True)
class CellAssignment:
    """The cells one entity occupies, and how that was decided.

    `note` is populated exactly when `cells` is empty. An entity with no cells
    and no reason is the failure mode this whole dataclass exists to prevent:
    a backfill that quietly indexes 40,000 of 46,754 entities looks identical
    to one that indexed everything.
    """

    cells: tuple[str, ...] = ()
    #: The single cell that best stands for this entity — its centroid's cell
    #: for an area, its first cell for a path. What a map labels, and what a
    #: "which cell is this in" question is answered with.
    primary: str | None = None
    #: "ring", "path", "point", or None when nothing could be assigned.
    strategy: str | None = None
    #: Cells beyond `MAX_CELLS_PER_ENTITY`, counted rather than forgotten.
    truncated: int = 0
    #: True cell count before truncation.
    total: int = 0
    #: Why there are no cells. `None` whenever there are some.
    note: str | None = None
    #: True when this entity is stored under a block layout but its own
    #: coordinates put it inside the drawing, so it was placed in the world
    #: rather than refused. Recorded because a reader must be able to tell a
    #: measured placement from an ordinary modelspace one, and because the
    #: map needs to find these rows: they do not carry the `Model` layout.
    world_placed: bool = False
    #: Set when cells WERE assigned but do not cover the whole object. An
    #: entity that is indexed incompletely is more dangerous than one that is
    #: not indexed at all: it answers vicinity questions, and answers them
    #: short. `None` whenever the coverage is whole.
    coverage_note: str | None = None


def outline_of(cells: Iterable[str]) -> dict[str, Any] | None:
    """One set of cells as a single MultiPolygon boundary, in lat/long.

    What this is for: a catchment drawn as 19 hexagons has 19 outlines, 12 of
    them internal, and every shared edge is drawn twice. `cells_to_h3shape`
    dissolves them into the boundary of the union — the outline somebody
    actually wants to put on a map — and `tight=False` asks for a
    MultiPolygon rather than a Polygon so that a set which is not connected,
    or which has a hole, comes back as the shape it really is instead of
    being flattened into one that is wrong.

    Coordinates come out [lon, lat] already, which is GeoJSON's order and the
    opposite of the (lat, lng) order `LatLngPoly` takes as input. The two live
    in this codebase within a few hundred lines of each other, so it is said
    here rather than assumed.

    `None` for an empty set — not an empty MultiPolygon, which renders as
    nothing while claiming to be a shape.
    """
    cells = list(cells)
    if not cells:
        return None
    try:
        shape = h3.cells_to_h3shape(cells, tight=False)
        geometry = h3.h3shape_to_geo(shape)
    except Exception as exc:  # pragma: no cover - depends on the cell set
        log.debug("outline failed for %d cells: %s", len(cells), exc)
        return None
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "cells": len(cells),
            "note": (
                "the boundary of the union of these cells, dissolved: shared "
                "edges between neighbouring cells are gone and only the "
                "outside remains. It is the hexagons' own edges, so it is as "
                "approximate as they are — not a smooth catchment"
            ),
        },
    }


def available() -> bool:
    """Whether cell assignment can run at all."""
    return h3 is not None


def _cell_of_point(crs: crs_math.Crs, xy: Sequence[float] | None, res: int) -> str | None:
    if not xy or len(xy) < 2:
        return None
    lat, lon = crs.to_lat_lon(float(xy[0]), float(xy[1]))
    return h3.latlng_to_cell(lat, lon, res)


def _ring_cells(
    crs: crs_math.Crs, ring: Sequence[Sequence[float]], res: int, centroid: str | None
) -> tuple[str, ...]:
    """Every cell a closed ring touches.

    Overlap containment, not the default. `polygon_to_cells` decides by cell
    CENTRE, and a 300 m2 parcel — the ordinary residential plot on the
    reference drawing — comes back with zero cells at resolution 9 and one at
    resolution 11 under that rule. A parcel that occupies no cells is a parcel
    that vanishes from every count built on this index, silently, and only for
    the small ones.

    `LatLngPoly` takes (lat, lng) pairs, which is the opposite order from the
    GeoJSON this repo also produces. Written out rather than passed through a
    shared helper precisely because the two orders live in one codebase.
    """
    points = []
    for xy in ring:
        if not xy or len(xy) < 2:
            continue
        points.append(crs.to_lat_lon(float(xy[0]), float(xy[1])))
    if len(points) < 3:
        return ()

    try:
        shape = h3.LatLngPoly(points)
        cells = h3.polygon_to_cells_experimental(shape, res, contain="overlap")
    except Exception as exc:  # pragma: no cover - depends on the geometry
        log.debug("polygon coverage failed: %s", exc)
        cells = []

    cells = sorted(set(cells))
    if cells:
        return tuple(cells)
    # The guarantee: every ring ends with at least one cell. A polygon smaller
    # than a cell, or one degenerate enough that the coverage refuses it, is
    # still somewhere.
    return (centroid,) if centroid else ()


def _path_cells(
    crs: crs_math.Crs, points: Sequence[Sequence[float]], res: int
) -> tuple[str, ...]:
    """An ordered, contiguous chain of cells along an open path.

    Built with `grid_path_cells` between consecutive vertices rather than by
    sampling the path every N metres. Sampling was the obvious approach and it
    does not work: at the default resolution a cell is 4.09 m across, so any
    step chosen in advance either skips cells — leaving a "chain" whose links
    are five cells apart — or is so small it is a guess about the units the
    drawing is in. `grid_path_cells` asks H3 for the line between two cells
    and gets back exactly the cells on it, contiguous by construction, with no
    step and no unit anywhere in the calculation.

    Order is preserved and duplicates are dropped, so a road that doubles back
    on itself is one chain rather than two.
    """
    out: list[str] = []
    seen: set[str] = set()
    previous: str | None = None

    for xy in points:
        cell = _cell_of_point(crs, xy, res)
        if cell is None:
            continue
        if previous is not None and cell != previous:
            try:
                span = h3.grid_path_cells(previous, cell)
            except Exception:
                # Too far apart, or across a place where the grid has no
                # straight line. The vertices themselves are still true; what
                # is lost is the contiguity between these two, and a gap in a
                # chain is better than an invented link.
                span = [cell]
            if len(span) > MAX_PATH_SEGMENT_CELLS:
                span = [previous, cell]
            for step in span:
                if step not in seen:
                    seen.add(step)
                    out.append(step)
        elif cell not in seen:
            seen.add(cell)
            out.append(cell)
        previous = cell

    return tuple(out)


def _space_of(layout: Any) -> str:
    """Which coordinate space a layout's numbers live in.

    `store._space_of`, deferred to call time. Restating it here would be a
    second opinion about which layouts hold real-world coordinates, and the
    two would differ the first time a drawing arrived with a layout named
    something nobody expected.
    """
    from .store import _space_of as space_of

    return space_of(layout if isinstance(layout, str) else None)


#: How far outside a drawing's own extents a point may sit and still be
#: treated as a position in it, as a fraction of the extents' diagonal. Small,
#: and it exists only so that a point exactly on the boundary is not excluded
#: by floating-point noise — not to admit anything that is genuinely outside.
EXTENTS_TOLERANCE = 0.01


def _inside_extents(xy: Sequence[float], extents: Mapping[str, Any] | None) -> bool:
    """Whether a point is a position in this drawing at all.

    The check exists because of five entities on the reference drawing, and
    the shape of what they are is worth stating: block references whose
    insertion point is (0, 0). A block INSERTed at the origin is an everyday
    CAD artefact — the geometry is drawn where it belongs and the reference
    that carries it is parked at zero — but (0, 0) put through a UTM inverse
    is a perfectly well-formed position in the Gulf of Guinea, and two of the
    five had a bounding box sitting squarely in Janadriyah while their
    anchor claimed to be off the coast of Africa.

    A drawing's extents are the bounding box of its own model-space geometry,
    so a model-space point outside them is not a location in the drawing by
    definition. That makes this a general test rather than a guard against one
    file's quirk, and it needs no knowledge of where any particular drawing is.

    No extents means no check. A drawing whose extents could not be computed —
    there is one in the store — is not thereby forbidden from having cells
    (G3): what is unavailable is the check, not the coordinates.
    """
    if not extents:
        return True
    lo = extents.get("min")
    hi = extents.get("max")
    if not lo or not hi or len(lo) < 2 or len(hi) < 2:
        return True
    width = float(hi[0]) - float(lo[0])
    height = float(hi[1]) - float(lo[1])
    margin = EXTENTS_TOLERANCE * ((width**2 + height**2) ** 0.5)
    return (
        float(lo[0]) - margin <= float(xy[0]) <= float(hi[0]) + margin
        and float(lo[1]) - margin <= float(xy[1]) <= float(hi[1]) + margin
    )


def _point_fields(row: Mapping[str, Any]) -> Sequence[tuple[str, Sequence[float] | None]]:
    """The candidate points, in the order `lat_lon_for_entity` uses them.

    Imported rather than restated so the two cannot drift: an entity whose
    cell came from its bounding-box centre while its reported position came
    from its polygon centroid would be an object in two places, and on a
    non-convex shape those two are 72.7 m apart on this drawing.
    """
    from .store_landuse import POINT_FIELDS

    return [(name, row.get(name)) for name, _basis in POINT_FIELDS]



def _block_is_world_placed(
    row: Mapping[str, Any],
    extents: Mapping[str, Any] | None,
    placements: Mapping[str, Any] | None,
) -> bool:
    """Whether a block-resident entity's own coordinates are world coordinates.

    THE MEASUREMENT IS THE INSERT CHAIN, not the extents. `extract.block_placements`
    composes the transform from model space down to each block definition; when
    every link is the identity, the block's stored numbers need no moving and
    already are positions on the earth. That is the causal fact. Extents
    membership is only its CONSEQUENCE, and a consequence can be true by
    coincidence -- a block drawn in local coordinates that happen to fall inside
    a large drawing's extents would pass an extents test while being nowhere.

    Extents keep two jobs, both narrower than deciding:

    - corroboration: an identity chain whose points still land outside the
      drawing is a contradiction, and the safe reading of a contradiction is
      to refuse rather than to place;
    - the origin-anchor guard that already existed, which is what stops a
      block reference parked at (0, 0) being read as a position in the Gulf
      of Guinea.

    Without placement information -- a drawing ingested before this was
    recorded -- there is no measurement to rely on, so the answer is no. A
    re-ingest supplies it; guessing would not.
    """
    if not placements:
        return False
    # Deferred, and read from `extract` rather than restated here: the prefix
    # and its inverse are one fact, and a second copy would drift.
    from .extract import block_name_of, is_block_layout  # noqa: PLC0415

    layout = row.get("layout")
    if not isinstance(layout, str) or not is_block_layout(layout):
        return False
    placement = placements.get(block_name_of(layout))
    if not isinstance(placement, Mapping) or not placement.get("identity"):
        return False
    if placement.get("ambiguous"):
        # The same definition is placed identically in one spot and moved in
        # another. One stored row cannot be both; refusing is the honest half.
        return False
    # Corroboration, not the decision: the chain says these are world
    # coordinates, so they must also land inside the drawing.
    return any(
        xy is not None and len(xy) >= 2 and _inside_extents(xy, extents)
        for _name, xy in _point_fields(row)
    )

def cells_for_entity(
    crs: crs_math.Crs,
    row: Mapping[str, Any],
    resolution: int,
    extents: Mapping[str, Any] | None = None,
    placements: Mapping[str, Any] | None = None,
) -> CellAssignment:
    """The cells one stored entity occupies.

    `row` is an entity document. Only geometry fields are read; nothing here
    looks at the layer name, the type, or anything else a drawing gets to
    name, so a drawing with unfamiliar conventions is indexed exactly as well
    as a familiar one (G10).
    """
    if h3 is None:  # pragma: no cover
        return CellAssignment(note="the h3 package is not installed")

    space = _space_of(row.get("layout"))
    world_placed = False
    if space != "model":
        # The bug this catches was real and silent. Cells were assigned to
        # every entity in the drawing, and 4,058 of them landed in H3 base
        # cells 61 and 75 — the Gulf of Guinea and the South Atlantic — while
        # looking exactly like the 42,691 that landed correctly on Janadriyah.
        #
        # A sheet's coordinates are page geometry: a border on an A0 sheet is
        # 1,189 units long because the sheet is 1,189 mm wide, and putting
        # that through a UTM inverse produces a position off the coast of
        # Africa. A block definition's coordinates are in the block's own
        # frame, and each INSERT places the same numbers somewhere different,
        # so one entity would need many cells or none. Neither is a
        # real-world position, and there is no correction that would make one
        # — the drawing simply does not say where these are.
        #
        # This is the same distinction `_unit_names` draws for lengths
        # (D-076), read from the same function so the two cannot disagree
        # about which layouts are real.
        #
        # BUT the premise above is a statement about coordinates, and it was
        # being decided from a NAME. For a bound xref it is simply false: the
        # xref keeps the coordinates it was drawn in and the INSERT that
        # places it is the identity, so the block's numbers ARE positions on
        # the earth. Sedra is the measured case — 934,013 of its 990,291
        # block-resident entities carry points inside the drawing's own
        # extents — and deciding by layout name left a fully ingested
        # 990,790-entity drawing with an empty map and nothing on screen
        # saying why.
        #
        # So the premise is now tested rather than assumed, with the same
        # `_inside_extents` this module already uses to reject a block
        # reference parked at the origin. A block point inside the drawing's
        # own extents is a position in this drawing BY THE FILE'S OWN
        # DEFINITION of where it is; one outside still is not, and is still
        # refused below. Paper space is unchanged: page geometry is not a
        # position however the numbers happen to fall.
        placed_in_world = space == "block" and _block_is_world_placed(
            row, extents, placements
        )
        if not placed_in_world:
            return CellAssignment(
                note=(
                    f"this entity is in {space} space, whose coordinates are "
                    + (
                        "page geometry rather than a position on the earth"
                        if space == "paper"
                        else "the block's own frame, placed somewhere "
                        "different by every INSERT of it"
                    )
                    + ", so it has no cell"
                )
            )
        world_placed = True

    # The representative point, taken from the first candidate field that has
    # one AND puts it inside the drawing. Skipping an out-of-extents candidate
    # rather than failing on it is what recovers the two INSERTs whose anchor
    # is at the origin but whose bounding box is where they are drawn.
    centroid = None
    rejected_points = 0
    for name, xy in _point_fields(row):
        if not xy or len(xy) < 2:
            continue
        if not _inside_extents(xy, extents):
            rejected_points += 1
            continue
        centroid = _cell_of_point(crs, xy, resolution)
        if centroid is not None:
            break

    strategy: str | None = None
    cells: tuple[str, ...] = ()

    ring = row.get("ring")
    if row.get("ring_status") == "complete" and ring:
        strategy = "ring"
        cells = _ring_cells(crs, ring, resolution, centroid)
    else:
        path = row.get("path_points")
        if not path:
            ends = row.get("ends")
            # `ends` is a pair of endpoints, stored under either shape
            # depending on the entity. Two points are enough for a chain.
            if isinstance(ends, Mapping):
                path = [ends.get("start"), ends.get("end")]
            elif isinstance(ends, Sequence) and not isinstance(ends, (str, bytes)):
                path = list(ends)
        path = [p for p in (path or []) if p]
        if len(path) >= 2:
            strategy = "path"
            cells = _path_cells(crs, path, resolution)

    if not cells and centroid is not None:
        strategy = strategy or "point"
        cells = (centroid,)

    if not cells:
        if rejected_points:
            return CellAssignment(
                note=(
                    f"every point stored for this entity ({rejected_points}) "
                    "lies outside the drawing's own extents, so none of them "
                    "is a position in this drawing — a block reference parked "
                    "at the origin is the usual cause. Placing it anyway would "
                    "put it in the sea"
                )
            )
        return CellAssignment(
            note=(
                "this entity carries no point, no closed ring and no path, so "
                "there is nowhere to put it"
            )
        )

    total = len(cells)
    truncated = 0
    if total > MAX_CELLS_PER_ENTITY:
        truncated = total - MAX_CELLS_PER_ENTITY
        cells = cells[:MAX_CELLS_PER_ENTITY]

    # For an area the representative cell is its centroid's, which is inside
    # the shape; for a chain it is the first, which is one of its ends. Taking
    # `cells[0]` for a polygon would give whichever cell sorted first, a
    # corner, and put every label on the edge of the thing it names.
    primary = centroid if strategy == "ring" and centroid else cells[0]

    return CellAssignment(
        cells=cells,
        primary=primary,
        strategy=strategy,
        truncated=truncated,
        total=total,
        coverage_note=_coverage_gap(strategy, cells, row, resolution),
        world_placed=world_placed,
    )


def _coverage_gap(
    strategy: str | None,
    cells: tuple[str, ...],
    row: Mapping[str, Any],
    resolution: int,
) -> str | None:
    """Whether this entity's cells fall short of the entity, and how.

    One test, and it is deliberately narrow: the object has a MEASURED length,
    that length is further than a cell reaches, and it came out in a single
    cell anyway. Both halves matter. `length` is set by the extractor only for
    things that have a length to measure, so a block reference or a run of
    text — which have a bounding box but no extent worth covering — never
    reaches this at all, and the check does not turn into 29,000 warnings
    about labels.

    Two shapes on the reference drawing land here, and they are different
    faults with the same symptom. An ARC stores only the two points its ends
    are at, so a deep arc's chord is short while the arc sweeps metres past
    it. A CIRCLE stores no ring at all, so it is placed at its centre and its
    circumference is not covered by anything.

    Nothing is invented to close either gap. The bounding box would cover a
    rectangle the object does not occupy, and G6 forbids a bbox standing in
    for a shape. What exists is the length the extractor measured from the
    real geometry, so the gap is STATED although it cannot be filled — and a
    vicinity answer that quietly stops at a chord is exactly the kind of short
    answer nobody thinks to check.

    The comparison puts drawing units against a distance in metres, which is
    sound only because of how the entity got here: cells exist for a drawing
    whose coordinates are a UTM easting and northing, and those are metres by
    definition of the projection.
    """
    if strategy not in ("path", "point"):
        return None
    if len(cells) >= 2:
        return None
    length = row.get("length")
    if not isinstance(length, (int, float)):
        return None
    across = 2 * h3.average_hexagon_edge_length(resolution, unit="m")
    if length <= across:
        return None
    return (
        f"this object measures {length:.2f} in drawing units — further than "
        f"one cell reaches at resolution {resolution} — but the geometry "
        f"stored for it ({row.get('type') or 'unknown type'}) resolves to a "
        "single cell, so its extent is not indexed. A vicinity count over "
        "these cells will be short for this object"
    )



def scope_match(layout: str) -> dict[str, Any]:
    """The Mongo filter for "entities in this scope", as one rule.

    Model scope is not simply `layout == "Model"`. A drawing that keeps its
    content inside a bound xref stores that content under a `[block] ...`
    layout while its coordinates are ordinary positions in the drawing, and
    `cells_for_entity` records that per entity as `h3_world_placed`. A reader
    that filtered on the layout name alone would show an empty map for a fully
    indexed drawing, which is what Sedra did.

    Defined here, once, because two readers need it -- the map feed and the
    cell view -- and a scope rule that differs between them would put a
    parcel on one screen and not the other with nothing to explain it.

    A request for a specific sheet means that sheet, so only model scope is
    widened.
    """
    if _space_of(layout) == "model":
        return {"$or": [{"layout": layout}, {"h3_world_placed": True}]}
    return {"layout": layout}


#: The resolution the VIEWPORT is covered at, and the resolution of the coarse
#: parent cell stored beside each fine one.
#:
#: A viewport cannot be matched against the fine cells directly: at the stored
#: resolution a screenful of a city block is roughly 93,000 cells, far past
#: what an `$in` should carry, and a coarse cover has nothing to match against
#: unless the coarse cell is stored too. So it is stored. `cell_to_parent` is
#: exact and needs no maths of ours, no re-parse and no second index of cells
#: -- one derived field beside the one already written.
#:
#: Resolution 8 covers ~0.74 km2 per cell, so a viewport a few kilometres
#: across is tens of cells rather than tens of thousands, which is what makes
#: the membership query small enough to be worth indexing.
VIEWPORT_RESOLUTION: Final[int] = 8


def coarse_cell(cell: str | None, res: int = VIEWPORT_RESOLUTION) -> str | None:
    """The ancestor of a fine cell at the viewport resolution, or None.

    Derived rather than recomputed from coordinates: the fine cell already
    encodes the position, and `cell_to_parent` is exact, so this cannot drift
    from the cell it is stored beside.
    """
    if not cell or h3 is None:
        return None
    try:
        if h3.get_resolution(cell) <= res:
            return cell
        return h3.cell_to_parent(cell, res)
    except Exception:  # pragma: no cover - a cell we cannot read
        return None


def cells_for_viewport(
    west: float, south: float, east: float, north: float,
    res: int = VIEWPORT_RESOLUTION,
) -> list[str]:
    """The coarse cells a lng/lat viewport covers.

    No projection anywhere: the viewport is already in degrees and H3 takes
    degrees, which is the whole reason this route exists rather than a bbox
    pushed into drawing coordinates. `LatLngPoly` takes (lat, lng) pairs,
    which is the opposite order from the GeoJSON this repo also produces, so
    it is written out here rather than passed through a shared helper.
    """
    if h3 is None:  # pragma: no cover
        return []
    ring = [
        (south, west), (south, east), (north, east), (north, west), (south, west),
    ]
    try:
        shape = h3.LatLngPoly(ring)
        return list(h3.polygon_to_cells_experimental(shape, res, contain="overlap"))
    except Exception as exc:  # pragma: no cover - depends on the box
        log.debug("viewport cover failed: %s", exc)
        return []

def fields_for_storage(
    assignment: CellAssignment, resolution: int
) -> dict[str, Any]:
    """The additive fields written onto an entity document.

    The resolution travels with the cells because a cell index without its
    resolution cannot be read: `8d5...` and `89 5...` are different lengths of
    the same place, and a config change that re-indexes a drawing has to leave
    the old rows identifiable as stale rather than merely different.

    Written for every entity, including the ones that got nothing. A row with
    `h3_cells: []` and a note is a row that was considered and could not be
    placed; a row with no h3 fields at all has not been through the backfill,
    and telling those two apart is the whole of knowing whether an index is
    complete.
    """
    return {
        "h3_res": resolution,
        "h3_cell": assignment.primary,
        # The indexed key a viewport query can actually use. See
        # `VIEWPORT_RESOLUTION`: the fine cell says where this is, the coarse
        # one says which screenful it belongs to.
        "h3_coarse": coarse_cell(assignment.primary),
        "h3_cells": list(assignment.cells),
        "h3_cell_count": assignment.total,
        "h3_strategy": assignment.strategy,
        "h3_cells_truncated": assignment.truncated or None,
        "h3_world_placed": assignment.world_placed or None,
        "h3_note": assignment.note,
        "h3_coverage_note": assignment.coverage_note,
    }


#: The share of an entity population the DEFAULT view must contain.
#:
#: Not a tuning knob so much as a statement of what "where the drawing is"
#: means: the region holding all but a small tail. A drawing's extreme
#: bounding box is not where the drawing is -- Janadriyah's `NBHD *`
#: boundaries are real, correctly placed, and scattered up to 12 km from an
#: estate that then occupies a few pixels. See section 13.6 of
#: docs/INTAKE-TO-AGENT-PLAN.md.
CONTENT_SHARE: Final[float] = 0.98


def cells_holding_most(
    counts: Mapping[str, int], share: float = CONTENT_SHARE
) -> list[str]:
    """The SMALLEST set of cells holding at least `share` of the entities.

    Busiest first, so what is dropped is always the thinnest tail. This is the
    same shape as the origin-anchor guard in `_inside_extents`: a handful of
    objects that are technically in the drawing must not be allowed to speak
    for where the drawing is.

    Returns every cell when the tail cannot be trimmed -- one cell, or a share
    of 1 -- because "most of it" and "all of it" are the same answer then, and
    inventing a difference would be worse than admitting there is none.
    """
    if not counts:
        return []
    total = sum(counts.values())
    if total <= 0:
        return []
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    wanted = total * share
    kept: list[str] = []
    running = 0
    for cell, n in ordered:
        kept.append(cell)
        running += n
        if running >= wanted:
            break
    return kept


def bounds_of_cells(cells: Iterable[str]) -> list[list[float]] | None:
    """The lng/lat box covering these cells, as [[minLng, minLat], [maxLng, maxLat]].

    Cell BOUNDARIES rather than centres: a centre-only box would cut the
    outermost cells in half and clip the very geometry it is meant to frame.
    """
    if h3 is None:
        return None
    lats: list[float] = []
    lngs: list[float] = []
    for cell in cells:
        try:
            for lat, lng in h3.cell_to_boundary(cell):
                lats.append(float(lat))
                lngs.append(float(lng))
        except Exception:  # pragma: no cover - a cell id we cannot read
            continue
    if not lats:
        return None
    return [[min(lngs), min(lats)], [max(lngs), max(lats)]]
