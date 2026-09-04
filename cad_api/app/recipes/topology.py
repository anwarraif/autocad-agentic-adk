"""Two topology recipes: does the ground tile, and can every plot be reached.

Owner: the TOPOLOGY lane — DOSSIER Phases 5a and 5b.

Both recipes answer a question about RELATIONSHIPS between polygons, and
neither of them was askable before Phase 1: until the Dossier said which layers
are **regions** and which are **networks**, there was nothing to relate.

## The two questions, and why they are two

**5a `overlap_scan`.** Two failure modes are reported SEPARATELY, because they
are different accidents with different owners:

* an **overlap** is two parcels claiming the same ground — on a master plan
  that is land sold twice;
* a **gap** is ground inside the site boundary that belongs to no parcel — land
  nobody accounted for.

**5b `frontage_check`.** A plot with no road frontage cannot be built on or
sold. That makes it a review-blocking defect rather than a nicety, and it is
why the parcels are reported **by handle**, named. A count is not actionable; a
handle is clickable.

## The tolerance decides the answer, so it is published, never buried

Two parcels sharing an edge are NOT overlapping, and where that line is drawn
is the whole finding. The same is true of frontage: "touching" is a tolerance,
and a different tolerance produces a different list of landlocked plots.

Every tolerance in this file is therefore

1.  **dimensionless in its constant and dimensioned by the drawing's own
    geometry.** `0.001 × the smaller parcel's area` and
    `0.01 × sqrt(median parcel area)` carry whatever units the drawing carries,
    which is the only thing that works when 11 of 18 drawings are in inches and
    3 declare no units at all (G2). A tolerance written as a number of drawing
    units would be a Janadriyah constant wearing a general name (G1);
2.  **published in the response** with its basis, its rule, and the absolute
    value it took on THIS drawing, in THIS drawing's units (G7);
3.  **never derived from the answer.** `frontage_check` measures the clearance
    from every parcel to the nearest road and reports its distribution — but
    the tolerance is not taken from that distribution. A tolerance read off the
    distances it then judges would report every parcel as having frontage by
    construction.

## What is measured, and what is only approximated — said out loud

Ring-to-ring work here is exact: parcels are stored with their rings, and the
intersection area below is computed from those rings by a decomposition that
assumes nothing about the shape (G6).

Road CENTRELINES are not. `LINE`, `ARC` and open `LWPOLYLINE` rows carry a
bounding box and no vertices — that is the same gap DOSSIER Phase 5c exists to
decide about. So a road entity with no ring is measured against its BOUNDING
BOX, and the response says so per row.

The direction of that error is what makes it usable, and it is why
`frontage_check` gives THREE answers rather than two:

* **frontage established** — the nearest road carries a ring, so the distance
  is exact and the pass is proven;
* **frontage not established** — the parcel came inside the tolerance only
  against a bounding box. A box CONTAINS its entity, so the measured distance
  is a LOWER bound: this is neither a pass nor a failure, and calling it a pass
  would publish something nobody measured;
* **no frontage** — further than the tolerance from every box, and therefore
  further than the tolerance from every road, whatever the boxes hide. This
  list is SOUND: the approximation cannot put a parcel on it wrongly, only keep
  one off it. It is the list the recipe exists to produce, and every row on it
  is named by handle.

The same discipline is why `region.point_in_ring` answers inside, outside, or
boundary rather than picking a side.

## Limits are limits on WORK, not on population size

A pair count does not bound the cost of this scan: seven thousand pairs of
five-vertex plots and seven thousand pairs involving a four-hundred-vertex road
corridor differ by four orders of magnitude, and the second is the shape that
runs for three minutes. Both recipes therefore sweep the candidate pairs and
sum their VERTEX PRODUCTS before a single triangle is clipped, and refuse above
a stated budget with a suggestion — the same discipline as
`store_spatial._join`, which counts its pairs before casting a single ray.

## Layer names never live in this file (G1)

Not one layer name, typology code, or road-layer name is written here. Parcel
layers come from the caller or from this drawing's land use config; road layers
come from the caller, from the config's `network` role, or from the Dossier's
geometric role — never from a name. The `C-ROAD-*` trap is exactly why: those
are Civil3D sheet-layout layers, and a name test would call them roads.

## Riding on, not rewriting

`store_spatial._SpatialIndex` is the bbox prefilter, used unchanged, so that
2,380 parcels do not become 2,380² tests. `store_spatial._ring_distance` is the
edge-to-edge distance, `geometry.signed_area` the shoelace, and
`region.point_in_ring` the point test. A second implementation of any of them
would answer differently on the same polygon, with nothing in either response
to say which one ran.
"""

from __future__ import annotations

import math
from typing import Any, Final, Mapping, Sequence

from .. import evidence as ev
from .. import geometry as geom
from .. import landuse
from .. import store_spatial as spatial
from ..extract import RING_TYPES
from ..mongo import COLL_ENTITIES, coll

# `library` is imported as a MODULE, for one function: the reader that turns
# "role=parcel" into this drawing's layer names. Copying that reader here would
# put a second config interpretation in the repo, and two interpretations of one
# config are how two answers about the same drawing start to differ. The same
# reasoning is why `library` itself calls `store_spatial._join` rather than
# writing a second ray-caster.
from . import library as _library
from .registry import (
    COMPUTED_PROVENANCE,
    MAX_ROWS_RETURNED,
    Param,
    Recipe,
    RecipeRefused,
    refuse_oversize,
    register,
    scope_for,
)

# --- stated limits (G7) ------------------------------------------------------

#: Rings that may enter one pairwise scan. Janadriyah's whole parcel population
#: is 2,380, so this sits an order of magnitude above the largest drawing here
#: and exists so that a filter which filters nothing refuses instead of running
#: for ten minutes.
MAX_SCAN_RINGS: Final[int] = 20_000

#: Pairs that may SURVIVE the bbox prefilter in `overlap_scan`. Counted after
#: the prefilter, for the same reason `store_spatial.PAIRS_BUDGET` is: over the
#: raw product 2,380² is 5.66 million before a single ring has been looked at,
#: and a budget counted there would refuse its own Definition of Done.
MAX_PAIRS_TESTED: Final[int] = 120_000

#: The clipping work ONE scan may do, counted as the sum of (vertices_a x
#: vertices_b) over the pairs that survived the prefilter — and counted BEFORE
#: a single triangle is clipped, so that this is a limit on work rather than a
#: report written after the work has already been done. A pair count alone does
#: not bound it: 7,000 pairs of five-vertex plots and 7,000 pairs involving a
#: 400-vertex road corridor differ by four orders of magnitude in cost, and the
#: second one is the shape that runs for ten minutes.
#:
#: Measured 25 August 2026 on the largest drawing in the store: clipping costs
#: about 11 microseconds per vertex pair, its 2,545 parcel rings come to 224,748
#: vertex pairs, and the same scan with a right-of-way corridor layer added
#: comes to 15,736,813 — which took 173 seconds and is exactly what this refuses.
MAX_VERTEX_PAIRS_TOTAL: Final[int] = 1_500_000

#: Vertex product for ONE pair. Above it that pair's area is WITHHELD together
#: with its handles, never computed with a simplified ring and never reported as
#: zero (G8). A ring may carry up to `geometry.MAX_RING_VERTICES` vertices, and
#: two of those are 16.7 million vertex pairs for a single cell.
MAX_VERTEX_PAIRS_PER_PAIR: Final[int] = 40_000

#: Boundary rings accepted by the gap block. A site boundary is one ring, or a
#: handful; a layer holding hundreds is not a site boundary and answering as if
#: it were would publish a gap that means nothing. It is also a work limit: every
#: parcel is clipped against every boundary ring whose box it meets.
MAX_BOUNDARY_RINGS: Final[int] = 50

#: Road entities that may enter one frontage check.
MAX_ROAD_ENTITIES: Final[int] = 50_000

#: Parcel-to-road pairs that may survive the prefilter in `frontage_check`.
MAX_FRONTAGE_PAIRS: Final[int] = 400_000

#: The frontage check's own vertex-pair budget, and it is deliberately larger
#: than the overlap scan's. The two algorithms differ where it matters: the
#: nearest-road search stops as soon as no remaining candidate can beat the best
#: distance found, so its swept worst case badly over-states the work it really
#: does, while ring clipping has no such exit and pays its whole product.
#:
#: Measured 25 August 2026 on the reference drawing, against a right-of-way
#: layer of 80 corridor rings: the sweep counts 14,732,651 vertex pairs and the
#: run takes 42.8 seconds — which also fixes where this ceiling sits. It is a
#: ceiling, not a latency target.
MAX_FRONTAGE_VERTEX_PAIRS: Final[int] = 20_000_000

# --- the tolerances, as DIMENSIONLESS ratios ---------------------------------
#
# Every number below is a ratio. Not one of them is a length or an area, and
# that is the whole point: a length written here would be right for Janadriyah
# in metres and wrong for the 11 drawings in inches and the 3 that declare
# nothing (G1, G2).

#: An intersection counts as an overlap only above this fraction of the SMALLER
#: parcel's area. Two parcels that share an edge intersect in exactly zero area
#: when their vertices are snapped, and in a sliver when they are not; this
#: fraction is what tells a sliver apart from land sold twice. On a 300-unit²
#: plot it is 0.3 unit² — far below any real double allocation, far above any
#: drafting slop.
DEFAULT_MIN_OVERLAP_FRACTION: Final[float] = 0.001

#: The numerical floor, relative to the pair's own bounding diagonal. Taken
#: from `geometry.BOUNDARY_EPS_REL` rather than copied, so that the floor used
#: here is the floor the ring machinery itself uses.
NUMERICAL_FLOOR_REL: Final[float] = geom.BOUNDARY_EPS_REL

#: The frontage touch tolerance, as a fraction of the typical parcel SIZE
#: (`sqrt(median parcel area)`, a length in the drawing's own units).
DEFAULT_TOUCH_REL: Final[float] = 0.01

#: How far `frontage_check` looks when MEASURING the nearest road. Wider than
#: the touch tolerance on purpose: a clearance distribution measured only inside
#: the tolerance would contain nothing but parcels that already passed, and the
#: one diagnosis that matters — "this road layer is a centreline, so no parcel
#: touches it" — would be invisible.
DEFAULT_SEARCH_REL: Final[float] = 3.0


def _entities():
    """This module's only door to MongoDB, and it goes through `coll()`."""
    return coll(COLL_ENTITIES)


# =============================================================================
# Pure geometry
# =============================================================================
#
# Everything in this section is a function of numbers. No MongoDB, no config,
# no units — which is what lets the fixtures below pin an intersection area to
# an exact value without a database.

Point = tuple[float, float]
Box = tuple[float, float, float, float]


def _box_of(ring: Sequence[Point]) -> Box | None:
    if len(ring) < 3:
        return None
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _box_ring(box: Box) -> list[Point]:
    """A box as a four-vertex ring, so that one distance engine serves both."""
    return [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]


def _box_distance(a: Box, b: Box) -> float:
    """Separation between two axis-aligned boxes; 0 when they overlap or touch."""
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return math.hypot(dx, dy)


def _boxes_overlap(a: Box, b: Box) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _line_cut(prev: Point, cur: Point, a: Point, b: Point) -> Point:
    """Where the segment prev→cur meets the infinite line a→b."""
    d1 = _cross(a, b, prev)
    d2 = _cross(a, b, cur)
    denom = d1 - d2
    if denom == 0.0:
        return cur
    t = d1 / denom
    return (prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1]))


def _clip_by_convex(subject: Sequence[Point], clip: Sequence[Point]) -> list[Point]:
    """Sutherland–Hodgman. Valid because `clip` is always a TRIANGLE here.

    Sutherland–Hodgman is only correct for a convex clip window, which is why
    this file never hands it a parcel: it clips triangle against triangle, and
    the polygons are decomposed into triangles first. A version that clipped
    parcel against parcel directly would be quietly wrong on every concave plot,
    and G6 forbids assuming they are all rectangles.
    """
    out = list(subject)
    n = len(clip)
    for i in range(n):
        if not out:
            return []
        a = clip[i]
        b = clip[(i + 1) % n]
        cut: list[Point] = []
        for j, cur in enumerate(out):
            prev = out[j - 1]
            cur_in = _cross(a, b, cur) >= 0.0
            prev_in = _cross(a, b, prev) >= 0.0
            if cur_in:
                if not prev_in:
                    cut.append(_line_cut(prev, cur, a, b))
                cut.append(cur)
            elif prev_in:
                cut.append(_line_cut(prev, cur, a, b))
        out = cut
    return out


def _triangles(ring: Sequence[Point]) -> list[tuple[list[Point], float]]:
    """A ring as SIGNED triangles fanned from the local origin.

    The decomposition is what makes the intersection below correct for any
    simple polygon, concave ones included, without a polygon-clipping library
    and without assuming a shape (G6): the indicator function of a simple ring
    equals the signed sum of its fan triangles, so the intersection of two rings
    is the double sum of the intersections of their triangles.
    """
    out: list[tuple[list[Point], float]] = []
    n = len(ring)
    for i in range(n):
        b = ring[i]
        c = ring[(i + 1) % n]
        area2 = b[0] * c[1] - c[0] * b[1]
        if area2 == 0.0:
            continue
        tri = [(0.0, 0.0), b, c] if area2 > 0 else [(0.0, 0.0), c, b]
        out.append((tri, 1.0 if area2 > 0 else -1.0))
    return out


def _abs_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(geom.signed_area(points, origin=(0.0, 0.0)))


def ring_intersection_area(
    a: Sequence[Point], b: Sequence[Point]
) -> float | None:
    """The area two rings share, in the rings' own units squared.

    Returns `None` — never `0.0` — when the pair is above
    `MAX_VERTEX_PAIRS_PER_PAIR`. Zero would read as "these two do not overlap",
    which is a claim nobody measured (G8).

    Both rings are translated to a shared bottom-left corner first, for the
    reason that shapes all of `geometry.py`: in projected coordinates every
    cross product is a difference of numbers around 1e6, and translating first
    removes that whole class of error without changing the answer.
    """
    if len(a) < 3 or len(b) < 3:
        return 0.0
    if len(a) * len(b) > MAX_VERTEX_PAIRS_PER_PAIR:
        return None

    ox = min(min(p[0] for p in a), min(p[0] for p in b))
    oy = min(min(p[1] for p in a), min(p[1] for p in b))
    ra = [(p[0] - ox, p[1] - oy) for p in a]
    rb = [(p[0] - ox, p[1] - oy) for p in b]

    # Both rings are normalised counter-clockwise so that the outer signs of the
    # double sum are both +1. Winding direction is a drafting habit, not a fact
    # about the ground, and a scan whose answer flipped sign with it would be
    # reporting the drafter rather than the drawing.
    if geom.signed_area(ra, origin=(0.0, 0.0)) < 0:
        ra = list(reversed(ra))
    if geom.signed_area(rb, origin=(0.0, 0.0)) < 0:
        rb = list(reversed(rb))

    tris_a = _triangles(ra)
    tris_b = _triangles(rb)
    total = 0.0
    for tri_a, sign_a in tris_a:
        box_a = _box_of(tri_a)
        for tri_b, sign_b in tris_b:
            box_b = _box_of(tri_b)
            if box_a is None or box_b is None or not _boxes_overlap(box_a, box_b):
                continue
            piece = _clip_by_convex(tri_a, tri_b)
            if len(piece) < 3:
                continue
            total += sign_a * sign_b * _abs_area(piece)
    # Rounding can leave a hair below zero on two rings that merely touch.
    return max(0.0, total)


def _numerical_floor(diagonal: float) -> float:
    """The area below which a shared sliver is the arithmetic, not the drawing.

    Coordinates are doubles. At the magnitudes a projected drawing uses, two
    rings snapped to the same edge still differ in their last bits, and the area
    that difference produces is not a drafting error — it is the resolution of
    the numbers themselves. Separating that from a real sliver is what stops a
    tidy master plan from returning thousands of findings nobody can act on.
    """
    return NUMERICAL_FLOOR_REL * diagonal * diagonal


def _pair_tolerance(
    area_a: float, area_b: float, diagonal: float, fraction: float
) -> float:
    """The area at or below which an intersection is NOT an overlap.

    Two terms, and the larger one binds:

    * `fraction × min(area_a, area_b)` — the sliver rule. It scales with the
      parcels being compared, so it means the same thing on a 300-unit² plot and
      on a 30,000-unit² school.
    * `NUMERICAL_FLOOR_REL × diagonal²` — the arithmetic floor, for the case
      where a ring's area is itself near zero and the fraction term collapses
      with it.

    Both are dimensionless multiples of quantities measured from THIS pair, so
    the tolerance arrives already in the drawing's units, whatever they are.
    """
    smaller = min(abs(area_a), abs(area_b))
    return max(fraction * smaller, _numerical_floor(diagonal))


# =============================================================================
# Reading the store
# =============================================================================


#: Ring statuses other than `complete`, taken from `store_spatial` rather than
#: re-listed, so that a cause added there is reported here on the same day.
SKIP_CAUSES: Final[tuple[str, ...]] = spatial.SKIP_CAUSES


def _ring_docs(
    drawing_id: str,
    layout: str | None,
    layers: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every ring-bearing row on these layers, plus a census of what was skipped.

    The census is the point. A polygon whose ring is bulged, open, or degenerate
    cannot take part in an area test, and it is counted by CAUSE and named —
    never treated as a polygon of area zero, which would make an unmeasured
    parcel look like a parcel that overlaps nothing (G8).
    """
    query: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "layer": {"$in": list(layers)},
        "type": {"$in": sorted(RING_TYPES)},
    }
    rows = list(
        _entities()
        .find(
            dict(query),
            {
                "handle": 1,
                "layer": 1,
                "type": 1,
                "ring": 1,
                "ring_status": 1,
                "ring_origin": 1,
                "area": 1,
                "polygon_centroid": 1,
                "geometry_note": 1,
                "_id": 0,
            },
        )
        .sort([("layer", 1), ("handle", 1)])
    )

    usable: list[dict[str, Any]] = []
    by_cause: dict[str, int] = {cause: 0 for cause in SKIP_CAUSES}
    examples: dict[str, list[str]] = {}
    zero_area: list[str] = []
    no_ring: list[str] = []

    for doc in rows:
        status = str(doc.get("ring_status") or "unreadable")
        if status != "complete":
            by_cause[status] = by_cause.get(status, 0) + 1
            examples.setdefault(status, [])
            if len(examples[status]) < 5:
                examples[status].append(str(doc.get("handle")))
            continue
        ring = spatial._ring_of(doc)
        if len(ring) < 3:
            no_ring.append(str(doc.get("handle")))
            continue
        area = abs(geom.signed_area(ring))
        if area == 0.0:
            if len(zero_area) < 20:
                zero_area.append(str(doc.get("handle")))
            continue
        doc["_ring"] = ring
        doc["_area"] = area
        doc["_box"] = _box_of(ring)
        usable.append(doc)

    census = {
        "ring_type_entities": len(rows),
        "rings_usable": len(usable),
        "skipped_by_cause": {k: v for k, v in sorted(by_cause.items()) if v},
        "skipped_total": sum(by_cause.values()) + len(no_ring) + len(zero_area),
        "skipped_examples": {k: v for k, v in sorted(examples.items())},
        "rings_complete_but_empty": len(no_ring),
        "rings_of_zero_area": len(zero_area),
        "rings_of_zero_area_examples": zero_area[:5],
        "not_zero_note": (
            "an unmeasurable ring is COUNTED and named, never given an area of "
            "zero. A polygon that could not be tested is not a polygon that "
            "overlaps nothing, and the two must not arrive as the same answer "
            "(G8)."
        ),
    }
    return usable, census


def _index_over(
    boxes: Sequence[Box | None], *, grow: float
) -> "spatial._SpatialIndex":
    """`store_spatial._SpatialIndex` over boxes grown by `grow`.

    The index is used exactly as it was built to be used — as a point-in-box
    lookup — and is not re-implemented here. What makes a point lookup answer a
    BOX question without ever missing a pair is that expansion. Two boxes come
    within `pad` of each other only when
    `|centre_a − centre_b| ≤ half_a + half_b + pad` on both axes; growing every
    indexed box by `pad + the largest half-extent among the QUERY boxes` and
    then querying with the query box's CENTRE therefore always lands inside the
    expanded box. The prefilter over-proposes and never under-proposes, and the
    exact `_box_distance` test removes the surplus.
    """
    padded: list[Box | None] = [
        None if b is None else (b[0] - grow, b[1] - grow, b[2] + grow, b[3] + grow)
        for b in boxes
    ]
    return spatial._SpatialIndex(padded)


def _half_extent(boxes: Sequence[Box | None]) -> float:
    present = [b for b in boxes if b is not None]
    if not present:
        return 0.0
    return max(max((b[2] - b[0]) / 2.0, (b[3] - b[1]) / 2.0) for b in present)


def _self_pairs(
    docs: Sequence[Mapping[str, Any]], *, pad: float = 0.0
) -> tuple[list[tuple[int, int]], int]:
    """Index pairs within one population whose boxes come within `pad`."""
    boxes = [d.get("_box") for d in docs]
    if not any(b is not None for b in boxes):
        return [], 0
    index = _index_over(boxes, grow=pad + _half_extent(boxes))

    proposed = 0
    pairs: list[tuple[int, int]] = []
    for j, box in enumerate(boxes):
        if box is None:
            continue
        for i in index.hits((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0):
            if i >= j:
                continue
            proposed += 1
            other = boxes[i]
            if other is not None and _box_distance(other, box) <= pad:
                pairs.append((i, j))
    pairs.sort()
    return pairs, proposed


def _cross_pairs(
    queries: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    pad: float,
) -> tuple[dict[int, list[int]], int]:
    """For each query, the targets whose box comes within `pad` of its own.

    Two populations rather than one, because a frontage check that indexed
    parcels and roads together would spend most of its budget proposing
    parcel-to-parcel pairs it then throws away.
    """
    target_boxes = [d.get("_box") for d in targets]
    query_boxes = [d.get("_box") for d in queries]
    if not any(b is not None for b in target_boxes):
        return {}, 0
    index = _index_over(target_boxes, grow=pad + _half_extent(query_boxes))

    proposed = 0
    near: dict[int, list[int]] = {}
    for q, box in enumerate(query_boxes):
        if box is None:
            continue
        for t in index.hits((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0):
            proposed += 1
            other = target_boxes[t]
            if other is not None and _box_distance(other, box) <= pad:
                near.setdefault(q, []).append(t)
    return near, proposed


# =============================================================================
# Units and evidence
# =============================================================================


def _unit_reason(units: Mapping[str, Any], key: str) -> str | None:
    if units.get(key):
        return None
    return (
        units.get("why_no_unit")
        or "this drawing declares no units, so none is claimed"
    )


def _area_quantity(
    value: float | None,
    units: Mapping[str, Any],
    *,
    basis: str,
    method: str,
    withheld_reason: str = "this area could not be measured",
) -> dict[str, Any]:
    if value is None:
        return ev.Quantity.withheld(
            basis=basis, method=method, reason=withheld_reason
        ).as_dict()
    return ev.Quantity.measured(
        basis=basis,
        method=method,
        value=float(value),
        unit=units.get("area_unit"),
        unit_reason=_unit_reason(units, "area_unit"),
    ).as_dict()


def _length_quantity(
    value: float | None,
    units: Mapping[str, Any],
    *,
    basis: str,
    method: str,
    withheld_reason: str = "this length could not be measured",
) -> dict[str, Any]:
    if value is None:
        return ev.Quantity.withheld(
            basis=basis, method=method, reason=withheld_reason
        ).as_dict()
    return ev.Quantity.measured(
        basis=basis,
        method=method,
        value=float(value),
        unit=units.get("length_unit"),
        unit_reason=_unit_reason(units, "length_unit"),
    ).as_dict()


def _geometry_observation(
    detail: str, *, locator: str | None, layout: str | None
) -> ev.Observation:
    """A measured fact from coordinates.

    `Origin.GEOMETRY` sits in the `coordinates` corpus, and that corpus NEVER
    states — it infers. That ceiling is the right one and not a limitation to be
    worked around: two rings sharing area is a measurement, and the file itself
    says nothing at all about whether that was intended.
    """
    return ev.Observation(
        origin=ev.Origin.GEOMETRY,
        detail=detail,
        locator=locator,
        layout=layout,
    )


def _absence_observation(
    detail: str, *, locator: str | None, layout: str | None
) -> ev.Observation:
    return ev.Observation(
        origin=ev.Origin.ABSENCE,
        detail=detail,
        locator=locator,
        layout=layout,
    )


def _measured_evidence(
    claim: str,
    *,
    drawing_id: str,
    scope: ev.Scope,
    observations: Sequence[ev.Observation],
    not_established: str,
    how_to_verify: str,
) -> ev.Evidence:
    """Evidence for a claim born from measurement, with the ceiling it earns.

    `tokens` is empty and stays empty. There is no word that, appearing in a
    file string, makes that file STATE that two parcels overlap by 4.7 units².
    The claim can therefore never rise above `inferred`, and that ceiling is
    correct rather than modest.
    """
    return ev.Evidence.of(
        ev.Claim(value=claim, tokens=()),
        drawing_id=drawing_id,
        scope=scope,
        provenance=COMPUTED_PROVENANCE,
        observations=tuple(observations),
        not_established=not_established,
        how_to_verify=how_to_verify,
        ceiling=ev.Grade.INFERRED,
    )


# =============================================================================
# 5a. overlap_scan
# =============================================================================


def _gap_not_computed(reason: str, hint: str) -> dict[str, Any]:
    return {
        "computed": False,
        "gap_area": None,
        "why_not_computed": reason,
        "how_to_compute_it": hint,
        "never_approximated": (
            "a site boundary is NEVER derived from the parcels themselves. "
            "Taking the outline of the parcels as the site would make the gap "
            "zero by construction and would answer a question nobody asked."
        ),
    }


def _boundary_rings(
    drawing_id: str, layout: str | None, boundary_layer: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    docs, census = _ring_docs(drawing_id, layout, [boundary_layer])
    refuse_oversize(
        what=f"boundary rings on layer {boundary_layer!r}",
        size=len(docs),
        limit=MAX_BOUNDARY_RINGS,
        hint=(
            "A site boundary is one ring, or a handful. A layer holding more "
            "than this is not a site boundary, and a gap measured against it "
            "would be a number about something else. Name the layer that really "
            "carries the boundary."
        ),
    )
    return docs, census


def _run_overlap_scan(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    layers: Sequence[str] | None,
    boundary_layer: str | None,
    min_overlap_fraction: float | None,
) -> dict[str, Any]:
    names, layers_source = _library._resolve_layers(drawing_id, layers, role="parcel")

    fraction = (
        DEFAULT_MIN_OVERLAP_FRACTION
        if min_overlap_fraction is None
        else float(min_overlap_fraction)
    )
    if fraction < 0 or fraction >= 1:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"min_overlap_fraction is {min_overlap_fraction!r}; it must be at "
            "least 0 and below 1.",
            "It is a FRACTION of the smaller parcel's area, not an area. A "
            "value of 1 or more would mean no overlap short of total is an "
            "overlap; a negative one would flag every pair that touches.",
        )

    docs, census = _ring_docs(drawing_id, layout, names)
    refuse_oversize(
        what="rings entering the pairwise scan",
        size=len(docs),
        limit=MAX_SCAN_RINGS,
        hint=(
            "Narrow `layers`. A pairwise scan grows as the square of its "
            "population before the bbox prefilter, and a scan that cannot "
            "finish is not a broader answer."
        ),
    )

    pairs, proposed = _self_pairs(docs)
    refuse_oversize(
        what="pairs that survived the bbox prefilter",
        size=len(pairs),
        limit=MAX_PAIRS_TESTED,
        hint=(
            "Narrow `layers` to the population genuinely being checked. This "
            "count is taken AFTER the spatial prefilter, not over the raw "
            f"{len(docs)} x {len(docs)} product, so it is already the work that "
            "would really be done."
        ),
    )

    # The first sweep only COUNTS. Clipping first and reporting the cost
    # afterwards would mean the work this limit exists to refuse has already
    # been done by the time anyone hears about it — the same reason
    # `store_spatial._join` counts its pairs before casting a single ray.
    work = 0
    for i, j in pairs:
        product = len(docs[i]["_ring"]) * len(docs[j]["_ring"])
        if product <= MAX_VERTEX_PAIRS_PER_PAIR:
            work += product
    refuse_oversize(
        what="ring vertex pairs to clip",
        size=work,
        limit=MAX_VERTEX_PAIRS_TOTAL,
        hint=(
            "This is a limit on WORK, not on the population: a scan of "
            f"{len(pairs)} pairs of small plots costs a fraction of a scan of "
            "the same number of pairs involving a road corridor with hundreds "
            "of vertices. Split the scan — run the plot layers together, and "
            "the corridor layers in a second call — or narrow `layers`."
        ),
    )

    overlaps: list[dict[str, Any]] = []
    slivers: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    tolerances_applied: list[float] = []
    overlap_area_sum = 0.0
    touching = 0
    slivers_from_arithmetic = 0
    relationships: dict[str, int] = {}

    for i, j in pairs:
        a, b = docs[i], docs[j]
        shared = ring_intersection_area(a["_ring"], b["_ring"])
        if shared is None:
            withheld.append(
                {
                    "handles": [a.get("handle"), b.get("handle")],
                    "layers": [a.get("layer"), b.get("layer")],
                    "reason": (
                        f"{len(a['_ring'])} x {len(b['_ring'])} vertices is "
                        f"above the per-pair limit of {MAX_VERTEX_PAIRS_PER_PAIR}; "
                        "the shared area is withheld, not simplified and not "
                        "reported as zero"
                    ),
                }
            )
            continue

        box_a, box_b = a["_box"], b["_box"]
        diagonal = math.hypot(
            max(box_a[2], box_b[2]) - min(box_a[0], box_b[0]),
            max(box_a[3], box_b[3]) - min(box_a[1], box_b[1]),
        )
        tolerance = _pair_tolerance(a["_area"], b["_area"], diagonal, fraction)
        floor = _numerical_floor(diagonal)
        tolerances_applied.append(tolerance)

        if shared == 0.0:
            touching += 1
            continue

        smaller_area = min(a["_area"], b["_area"])
        smaller = a if a["_area"] <= b["_area"] else b
        # WHICH overlap this is, because the three cases are three different
        # defects. Two rings covering the same ground is a parcel drawn twice on
        # two layers, and every area total that counts both counts that ground
        # twice. One ring wholly inside another is a parcel nested in a larger
        # allocation. A partial overlap is a boundary in the wrong place.
        covers_a = abs(shared - a["_area"]) <= tolerance
        covers_b = abs(shared - b["_area"]) <= tolerance
        if covers_a and covers_b:
            relationship = (
                "coincident: both rings cover the same ground, so any total "
                "that adds both counts that ground twice"
            )
        elif covers_a or covers_b:
            relationship = "nested: the smaller ring lies wholly inside the larger"
        else:
            relationship = "partial: the two rings cross"
        row = {
            "handles": [a.get("handle"), b.get("handle")],
            "layers": [a.get("layer"), b.get("layer")],
            "shared_area": _area_quantity(
                shared,
                units,
                basis=(
                    f"the ground shared by {a.get('handle')} and {b.get('handle')}"
                ),
                method=(
                    "exact ring intersection: both rings decomposed into signed "
                    "triangles, triangle pairs clipped, areas summed"
                ),
            ),
            "fraction_of_smaller": (
                shared / smaller_area if smaller_area else None
            ),
            "smaller_handle": smaller.get("handle"),
            "smaller_area": _area_quantity(
                smaller_area,
                units,
                basis=f"the area of {smaller.get('handle')}, the smaller of the pair",
                method="shoelace over the stored ring",
            ),
            "tolerance_for_this_pair": _area_quantity(
                tolerance,
                units,
                basis="the area at or below which this pair is not an overlap",
                method=(
                    f"max({fraction:g} x the smaller area, "
                    f"{NUMERICAL_FLOOR_REL:g} x the pair's bounding diagonal squared)"
                ),
            ),
            "relationship": relationship,
        }
        if shared > tolerance:
            overlap_area_sum += shared
            overlaps.append(row)
            key = relationship.split(":")[0]
            relationships[key] = relationships.get(key, 0) + 1
        else:
            row["below_the_arithmetic_floor"] = shared <= floor
            if shared <= floor:
                slivers_from_arithmetic += 1
            slivers.append(row)

    overlaps.sort(
        key=lambda r: (
            -(r["shared_area"].get("value") or 0.0),
            str(r["handles"][0]),
        )
    )
    slivers.sort(
        key=lambda r: (
            -(r["shared_area"].get("value") or 0.0),
            str(r["handles"][0]),
        )
    )

    representative = None
    if tolerances_applied:
        ordered = sorted(tolerances_applied)
        representative = ordered[len(ordered) // 2]

    gap = _gap_block(
        drawing_id,
        layout,
        units,
        boundary_layer=boundary_layer,
        parcels=docs,
        overlap_area_sum=overlap_area_sum,
        overlap_pairs=len(overlaps),
        parcel_layers=names,
        fraction=fraction,
    )

    scope_note = (
        f"layout {layout!r}; {len(docs)} rings with `ring_status == 'complete'` "
        f"and a non-zero area, on {len(names)} layers from {layers_source}; "
        f"{len(pairs)} pairs tested after the bounding-box prefilter; an "
        f"intersection counts as an overlap above {fraction:g} of the smaller "
        "parcel's area"
    )
    scope = scope_for(layout=layout, units=units, note=scope_note)

    observations: list[ev.Observation] = [
        _geometry_observation(
            f"{len(overlaps)} pairs of rings share more ground than the stated "
            f"tolerance; {len(slivers)} share less and are reported as slivers; "
            f"{touching} pairs meet with exactly zero shared area",
            locator=",".join(names[:3]) or None,
            layout=layout,
        )
    ]
    if census["skipped_total"]:
        observations.append(
            _absence_observation(
                f"{census['skipped_total']} polygons on those layers could not "
                "be tested at all — "
                + (
                    ", ".join(
                        f"{v} {k}" for k, v in census["skipped_by_cause"].items()
                    )
                    or "no cause recorded"
                )
                + f", {census['rings_of_zero_area']} of zero area",
                locator=",".join(names[:3]) or None,
                layout=layout,
            )
        )
    if withheld:
        observations.append(
            _absence_observation(
                f"{len(withheld)} pairs were above the per-pair vertex limit "
                "and their shared area is withheld rather than computed",
                locator=str(withheld[0]["handles"][0]),
                layout=layout,
            )
        )

    evidence = _measured_evidence(
        f"{len(overlaps)} pairs of parcels overlap above the stated tolerance "
        f"in layout {layout!r}",
        drawing_id=drawing_id,
        scope=scope,
        observations=observations,
        not_established=(
            "WHY they overlap. An overlap is a measurement, not a verdict: two "
            "rings sharing ground can be land allocated twice, a parcel drawn "
            "over its own hatch, or a revision left in place beside its "
            "replacement, and the file says which of those it is nowhere. Nor "
            "is the absence of an overlap a statement that the ground tiles: "
            "only the pairs whose bounding boxes come together were tested, "
            "and the gap block is the half of the question that covers what "
            "nothing claims"
        ),
        how_to_verify=(
            "open the two handles in the drawing and look at them together; "
            "then re-run with `min_overlap_fraction` an order of magnitude "
            "smaller, and see whether the pair count moves. A finding that "
            "changes with the tolerance is a drafting sliver; one that does not "
            "is a real double allocation"
        ),
    )

    body: dict[str, Any] = {
        "scope_note": scope_note,
        "layers": list(names),
        "layers_source": layers_source,
        "method": (
            "pairwise ring intersection, exact: each ring is decomposed into "
            "signed triangles fanned from a shared local origin, triangle pairs "
            "are clipped against one another, and the signed areas are summed. "
            "No shape is assumed — concave plots and plots with hundreds of "
            "vertices are measured the same way (G6)"
        ),
        "prefilter": (
            "the bounding box of each ring, through `store_spatial._SpatialIndex` "
            "— the same uniform grid `join_labels` uses, unchanged. Without it "
            f"this scan would be {len(docs)} x {len(docs)} tests"
        ),
        "tolerance": {
            "rule": (
                "an intersection is an OVERLAP when its area is strictly "
                "greater than the tolerance for that pair; at or below it the "
                "pair is reported as a sliver, and two parcels sharing an edge "
                "are neither"
            ),
            "min_overlap_fraction": fraction,
            "numerical_floor_rel": NUMERICAL_FLOOR_REL,
            "supplied_by_caller": min_overlap_fraction is not None,
            "per_pair_formula": (
                "max(min_overlap_fraction x the smaller ring's area, "
                "numerical_floor_rel x the pair's bounding diagonal squared)"
            ),
            "representative_value": _area_quantity(
                representative,
                units,
                basis="the median of the tolerances actually applied to the tested pairs",
                method="median over the per-pair tolerances",
                withheld_reason="no pair was tested, so no tolerance was applied",
            ),
            "basis": (
                "both terms are DIMENSIONLESS multiples of quantities measured "
                "from the pair itself, so the tolerance arrives already in this "
                "drawing's units — the same rule in metres, in inches, and in a "
                "drawing that declares no unit at all (G2). A tolerance written "
                "as a number of drawing units would be one drawing's trait "
                "wearing a general name (G1)"
            ),
            "decides": (
                "this number decides the answer. Two parcels sharing an edge are "
                "not overlapping, and where that line is drawn is the finding"
            ),
        },
        "rings_scanned": len(docs),
        "ring_census": census,
        "pairs_proposed_by_prefilter": proposed,
        "pairs_after_prefilter": len(pairs),
        "pairs_tested": len(pairs) - len(withheld),
        "vertex_pairs_clipped": work,
        "vertex_pairs_budget": MAX_VERTEX_PAIRS_TOTAL,
        "pairs_meeting_with_zero_shared_area": touching,
        "pairs_meeting_note": (
            "a shared area of exactly zero is two rings that MEET — a shared "
            "edge or a shared corner. It is the normal way a master plan is "
            "drawn and it is not an overlap"
        ),
        "overlaps": {
            "count": len(overlaps),
            "returned": len(overlaps[:MAX_ROWS_RETURNED]),
            "truncated": len(overlaps) > MAX_ROWS_RETURNED,
            "cap": MAX_ROWS_RETURNED,
            "shared_area_total": _area_quantity(
                overlap_area_sum if overlaps else None,
                units,
                basis=f"the ground claimed twice across {len(overlaps)} pairs",
                method="sum of the pairwise intersection areas above the tolerance",
                withheld_reason="not one pair overlaps above the tolerance",
            ),
            "by_relationship": dict(sorted(relationships.items())),
            "by_relationship_note": (
                "`coincident` is the expensive one: two rings covering the same "
                "ground means the parcel is drawn twice, and every area total "
                "that adds both layers counts that ground twice. `nested` is a "
                "parcel inside a larger allocation. `partial` is a boundary in "
                "the wrong place"
            ),
            "rows": overlaps[:MAX_ROWS_RETURNED],
        },
        "slivers_below_tolerance": {
            "count": len(slivers),
            "below_the_arithmetic_floor": slivers_from_arithmetic,
            "above_the_arithmetic_floor": len(slivers) - slivers_from_arithmetic,
            "returned": len(slivers[:20]),
            "truncated": len(slivers) > 20,
            "cap": 20,
            "note": (
                "these pairs DO share ground, but less than their tolerance. "
                "They are reported rather than dropped: a sliver that keeps "
                "recurring on one layer is a snapping problem, and dropping "
                "them would hide it"
            ),
            "arithmetic_floor_note": (
                "a sliver BELOW the arithmetic floor is the resolution of the "
                "coordinates, not a defect in the drawing: two rings snapped to "
                "the same edge still differ in the last bits of a double. Only "
                "the ones above the floor are worth a drafter's time"
            ),
            "rows": slivers[:20],
        },
        "pairs_withheld": withheld[:20],
        "pairs_withheld_total": len(withheld),
        "pairs_withheld_note": (
            "a pair above the per-pair vertex limit has its shared area WITHHELD "
            "together with both handles. Withheld is not zero (G8)"
        ),
        "gap": gap,
        "two_failure_modes": (
            "an overlap and a gap are different accidents and are reported "
            "separately: an overlap is land sold twice, a gap is land nobody "
            "accounted for. A drawing can carry both, either, or neither"
        ),
        "limits_applied": {
            "rings": MAX_SCAN_RINGS,
            "pairs_after_prefilter": MAX_PAIRS_TESTED,
            "vertex_pairs_total": MAX_VERTEX_PAIRS_TOTAL,
            "vertex_pairs_per_pair": MAX_VERTEX_PAIRS_PER_PAIR,
            "rows_returned": MAX_ROWS_RETURNED,
            "boundary_rings": MAX_BOUNDARY_RINGS,
        },
    }
    return ev.attach(body, evidence)


def _gap_block(
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    *,
    boundary_layer: str | None,
    parcels: Sequence[Mapping[str, Any]],
    overlap_area_sum: float,
    overlap_pairs: int,
    parcel_layers: Sequence[str],
    fraction: float,
) -> dict[str, Any]:
    """Ground inside the boundary that no scanned layer claims — or why not.

    A gap is only computable where a boundary ring EXISTS. When none is named,
    this returns `computed: false` with the reason, and it never derives a site
    boundary from the parcels: an outline taken from the parcels makes the gap
    zero by construction, which is not a measurement of anything.
    """
    if not boundary_layer:
        return _gap_not_computed(
            "no `boundary_layer` was given, and a gap is ground inside a "
            "boundary. Without a boundary there is nothing for the parcels to "
            "fall short of.",
            "name the single ring layer that carries the site boundary in this "
            "drawing. If the drawing does not carry one, then this file cannot "
            "answer the question and that is the honest answer.",
        )

    rings, census = _boundary_rings(drawing_id, layout, boundary_layer)
    if not rings:
        return _gap_not_computed(
            f"layer {boundary_layer!r} carries no polygon with a complete, "
            f"non-zero ring on layout {layout!r}. "
            + (
                "What it does carry: "
                + ", ".join(
                    f"{v} {k}" for k, v in census["skipped_by_cause"].items()
                )
                + "."
                if census["skipped_by_cause"]
                else "It carries no ring-bearing entity at all on this layout."
            ),
            "check the layer name and the layout, or repair the boundary "
            "polygon: a bulged or open boundary cannot bound anything, and its "
            "area is withheld rather than guessed (G8).",
        )

    boundary_area = sum(r["_area"] for r in rings)

    # Boundary rings that overlap EACH OTHER would count their shared ground
    # twice in the total above, which would show up as a gap that is not there.
    self_overlap = 0.0
    if len(rings) > 1:
        pairs, _ = _self_pairs(rings)
        for i, j in pairs:
            shared = ring_intersection_area(rings[i]["_ring"], rings[j]["_ring"])
            if shared:
                self_overlap += shared

    covered = 0.0
    clipped = 0
    partly_outside = 0
    for parcel in parcels:
        inside = 0.0
        for ring in rings:
            if not _boxes_overlap(parcel["_box"], ring["_box"]):
                continue
            shared = ring_intersection_area(parcel["_ring"], ring["_ring"])
            if shared:
                inside += shared
        if inside <= 0.0:
            continue
        clipped += 1
        # A parcel that straddles the boundary contributes only the part inside
        # it. Counting it whole would inflate the covered ground; dropping it
        # would inflate the gap. Both are wrong, and neither would be visible.
        if inside < parcel["_area"] * (1.0 - 1e-9):
            partly_outside += 1
        covered += inside

    covered_net = covered - overlap_area_sum
    gap_area = boundary_area - self_overlap - covered_net

    return {
        "computed": True,
        "boundary_layer": boundary_layer,
        "boundary_rings": len(rings),
        "boundary_ring_census": census,
        "boundary_area": _area_quantity(
            boundary_area - self_overlap,
            units,
            basis=f"the ground inside the {len(rings)} boundary rings",
            method=(
                "shoelace over each boundary ring, less the ground any two of "
                "them share"
            ),
        ),
        "boundary_rings_share_ground": _area_quantity(
            self_overlap if len(rings) > 1 else None,
            units,
            basis="ground claimed by more than one boundary ring",
            method="pairwise ring intersection among the boundary rings",
            withheld_reason="there is one boundary ring, so it cannot overlap another",
        ),
        "covered_area": _area_quantity(
            covered_net,
            units,
            basis="ground inside the boundary claimed by the scanned layers",
            method=(
                "each parcel ring clipped to the boundary rings and summed, "
                "less the overlaps found above"
            ),
        ),
        "gap_area": _area_quantity(
            gap_area,
            units,
            basis="ground inside the boundary that no scanned layer claims",
            method="boundary area less covered area",
        ),
        "gap_fraction": (gap_area / boundary_area) if boundary_area else None,
        "parcels_inside_boundary": clipped,
        "parcels_crossing_the_boundary": partly_outside,
        "parcels_crossing_note": (
            "a parcel that crosses the boundary is CLIPPED to it, not counted "
            "whole and not dropped. Only the part inside the boundary counts as "
            "covered ground"
        ),
        "what_counts_as_covered": (
            "only the layers in `layers` were scanned, so ground covered by any "
            "layer left out of the scan — road corridors and open space among "
            "them — appears here as gap. That is arithmetic, not a defect "
            "finding: name every covering layer in `layers` before reading this "
            "number as unallocated land"
        ),
        "limits": {
            "overlap_correction": (
                f"the {overlap_pairs} overlapping pairs are subtracted at their "
                "FULL area, not clipped to the boundary. Where an overlapping "
                "pair straddles the boundary this over-corrects, and the gap "
                "reads slightly larger than it is"
            ),
            "inclusion_exclusion_order": (
                "the correction is first order. Ground covered by three or more "
                "parcels at once is subtracted once too often; that state also "
                "shows up as several overlapping pairs sharing a handle"
            ),
            "tolerance": (
                f"pairs sharing at or below {fraction:g} of the smaller parcel's "
                "area are NOT subtracted, so the covered area includes those "
                "slivers twice"
            ),
        },
        "scanned_layers": list(parcel_layers),
    }


register(
    Recipe(
        name="overlap_scan",
        answers=(
            "whether the parcels tile the ground — which pairs overlap, and "
            "what is left over inside the site boundary"
        ),
        when_to_use=(
            "'do any plots overlap', 'was any parcel sold twice', 'is any land "
            "inside the site unaccounted for', or before a plot schedule is "
            "signed off. It is the most expensive question in the file and the "
            "one worth asking first: an overlap is land sold twice and a gap is "
            "land nobody accounted for."
        ),
        params=(
            Param(
                "layers",
                "layers",
                "the polygon layers whose rings are compared with one another. "
                "Left empty, they are derived from this drawing's land use "
                "config (role=parcel) and the response names them. Name every "
                "COVERING layer — roads and open space included — when the gap "
                "figure is the one being read.",
            ),
            Param(
                "boundary_layer",
                "layer",
                "the single ring layer carrying the site boundary. Without it "
                "the gap half of the answer is returned as NOT COMPUTED with "
                "its reason: a site boundary is never derived from the parcels, "
                "because an outline taken from the parcels makes the gap zero "
                "by construction.",
            ),
            Param(
                "min_overlap_fraction",
                "number",
                "the fraction of the SMALLER parcel's area above which a shared "
                "area counts as an overlap. A fraction, never an area, so it "
                "means the same thing in metres, in inches, and in a drawing "
                "that declares no units. Left empty it defaults to "
                f"{DEFAULT_MIN_OVERLAP_FRACTION:g}, and the response publishes "
                "the absolute value that fraction took on this drawing.",
                default=None,
            ),
        ),
        returns=(
            "overlaps.count",
            "overlaps.rows[].handles",
            "overlaps.rows[].shared_area",
            "overlaps.rows[].fraction_of_smaller",
            "overlaps.rows[].relationship",
            "overlaps.by_relationship",
            "slivers_below_tolerance",
            "pairs_meeting_with_zero_shared_area",
            "pairs_withheld",
            "gap.computed",
            "gap.gap_area",
            "tolerance",
            "ring_census.skipped_by_cause",
        ),
        built_on=(
            "store_spatial._SpatialIndex",
            "store_spatial._ring_of",
            "geometry.signed_area",
            "recipes.library._resolve_layers",
        ),
        limits={
            "rings": MAX_SCAN_RINGS,
            "pairs_after_prefilter": MAX_PAIRS_TESTED,
            "vertex_pairs_total": MAX_VERTEX_PAIRS_TOTAL,
            "vertex_pairs_per_pair": MAX_VERTEX_PAIRS_PER_PAIR,
            "boundary_rings": MAX_BOUNDARY_RINGS,
            "rows_returned": MAX_ROWS_RETURNED,
        },
        run=_run_overlap_scan,
    )
)


# =============================================================================
# 5b. frontage_check
# =============================================================================


def _road_layers(
    drawing_id: str, layout: str | None, given: Sequence[str] | None
) -> tuple[tuple[str, ...], str, list[str]]:
    """The road layers, from the caller, the config, or the Dossier — never a name.

    Three routes, in this order, and the response says which one answered:

    1.  the caller names them;
    2.  this drawing's land use config declares `role: network` on them;
    3.  the Dossier's GEOMETRIC role for the layer is `network` — open paths
        whose endpoints meet.

    What is refused is the fourth route nobody may take: matching a layer name.
    Civil3D template layers named `C-ROAD-*` are sheet layout, not roads, and a
    name test would call every one of them a road. Meaning never comes from a
    name (G1); the geometric role is what disambiguates.
    """
    if given:
        return tuple(given), "the list of layers handed over by the caller", []

    notes: list[str] = []
    found: set[str] = set()
    sources: list[str] = []

    try:
        config = landuse.for_drawing(drawing_id)
    except Exception as exc:  # noqa: BLE001 -- a broken config must not be an outage
        config = None
        notes.append(
            f"this drawing's land use config could not be read "
            f"({type(exc).__name__}), so no layer arrived through the config route"
        )
    if config is not None:
        named = sorted(
            name
            for name, entry in config.layers.items()
            if entry.role == landuse.NETWORK_ROLE
        )
        if named:
            found.update(named)
            sources.append(
                f"land use config {config.path} (role={landuse.NETWORK_ROLE})"
            )
        else:
            notes.append(
                "this drawing's land use config names no layer with role "
                f"{landuse.NETWORK_ROLE!r}"
            )
    elif not notes:
        notes.append("this drawing has no land use config yet")

    dossier_names, why = _dossier_network_layers(drawing_id, layout)
    if dossier_names:
        found.update(dossier_names)
        sources.append(
            f"the Dossier's geometric role `network` on layout {layout!r}"
        )
    if why:
        notes.append(why)

    if not found:
        raise RecipeRefused(
            "RECIPE_NO_CONFIG",
            f"no road layer could be derived for drawing {drawing_id!r} on "
            f"layout {layout!r}.",
            "Name the road layers in `road_layers`, or give them `role: network` "
            "in this drawing's land use config, or build the Dossier so their "
            "geometric role is known. What will NOT happen is a guess from "
            "layer names: template layers named after roads are sheet layout, "
            "and calling them roads would report frontage that is not there. "
            + (" ".join(notes) if notes else ""),
        )
    return tuple(sorted(found)), " and ".join(sources), notes


def _dossier_network_layers(
    drawing_id: str, layout: str | None
) -> tuple[tuple[str, ...], str | None]:
    """Layers whose GEOMETRIC role is `network`, read through `dossier_read`.

    `dossier_read` is imported behind a guard, as every Dossier caller in this
    repo does: a missing or half-written module must leave this recipe able to
    answer through its other routes, with the reason PUBLISHED rather than
    swallowed. The role itself is read through `dossier_read.profiles_for`, not
    off the raw document, so the storage shape stays that module's business.
    """
    try:
        from .. import dossier_read
    except Exception as exc:  # noqa: BLE001 -- guarded on purpose, see docstring
        return (), (
            f"dossier_read is not importable ({type(exc).__name__}), so no "
            "layer arrived through the geometric-role route"
        )

    try:
        document = dossier_read.dossier_for(drawing_id)
    except Exception as exc:  # noqa: BLE001
        return (), (
            f"the Dossier store could not be read ({type(exc).__name__}); "
            "nothing was established about any layer's geometric role"
        )
    if document is None:
        return (), (
            "no Dossier has been built for this drawing, so no layer has a "
            "geometric role yet. That is 'not computed', not 'there are no "
            "roads'"
        )

    blocks = document.get("layers")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        return (), "this Dossier records no layer buckets"
    names = sorted(
        {
            str(b.get("layer"))
            for b in blocks
            if isinstance(b, Mapping)
            and b.get("layer")
            and (layout is None or str(b.get("layout")) == str(layout))
        }
    )
    if not names:
        return (), f"this Dossier records no layer bucket on layout {layout!r}"

    chunk = max(1, int(getattr(dossier_read, "MAX_PROFILE_LAYERS", 40)))
    network: list[str] = []
    for start in range(0, len(names), chunk):
        window = names[start : start + chunk]
        try:
            profiles = dossier_read.profiles_for(
                drawing_id, window, layout=layout, dossier=document
            )
        except Exception as exc:  # noqa: BLE001
            return tuple(network), (
                f"the Dossier was read but profiling stopped after "
                f"{len(network)} matches ({type(exc).__name__})"
            )
        for name, profile in profiles.items():
            if isinstance(profile, Mapping) and profile.get("role") == "network":
                network.append(str(name))
    if not network:
        return (), (
            f"the Dossier gives no layer on layout {layout!r} the geometric "
            "role `network`"
        )
    return tuple(sorted(set(network))), None


def _road_geometry(
    drawing_id: str, layout: str | None, layers: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every road entity, with the best geometry the store holds for it.

    Two kinds come back, and the difference is published per row rather than
    averaged away:

    * a road drawn as a CLOSED polygon — a corridor, a right of way — carries a
      ring, and its distance to a parcel is exact;
    * a road drawn as a `LINE`, an `ARC`, or an open polyline carries a bounding
      box and nothing else. Its distance is measured to that BOX.

    The direction of the second error is the useful part. A box contains its
    entity, so the measured distance is a LOWER bound on the true one: a parcel
    reported as landlocked really is landlocked, while a parcel reported as
    having frontage may only have a bounding box nearby. The list this recipe
    exists to produce — the parcels with NO frontage — is the sound one.
    """
    rows = list(
        _entities()
        .find(
            {
                "drawing_id": drawing_id,
                "layout": layout,
                "layer": {"$in": list(layers)},
            },
            {
                "handle": 1,
                "layer": 1,
                "type": 1,
                "ring": 1,
                "ring_status": 1,
                "bbox": 1,
                "_id": 0,
            },
        )
        .sort([("layer", 1), ("handle", 1)])
    )

    items: list[dict[str, Any]] = []
    with_ring = 0
    box_only = 0
    no_geometry: list[str] = []
    for doc in rows:
        ring: list[Point] = []
        if doc.get("ring_status") == "complete":
            ring = spatial._ring_of(doc)
        if len(ring) >= 3:
            doc["_ring"] = ring
            doc["_box"] = _box_of(ring)
            doc["_basis"] = "ring"
            with_ring += 1
        else:
            box = doc.get("bbox") or {}
            low, high = box.get("min"), box.get("max")
            if not low or not high or len(low) < 2 or len(high) < 2:
                no_geometry.append(str(doc.get("handle")))
                continue
            flat = (float(low[0]), float(low[1]), float(high[0]), float(high[1]))
            doc["_box"] = flat
            doc["_ring"] = _box_ring(flat)
            doc["_basis"] = "bounding_box"
            box_only += 1
        items.append(doc)

    census = {
        "road_entities": len(rows),
        "measured_from_their_ring": with_ring,
        "measured_from_their_bounding_box": box_only,
        "without_any_geometry": len(no_geometry),
        "without_any_geometry_examples": no_geometry[:5],
        "bounding_box_note": (
            "a road drawn as a LINE, an ARC, or an open polyline carries no "
            "vertices in this store — only a bounding box. Its distance is "
            "therefore measured to that BOX, which CONTAINS the entity, so the "
            "measured distance is a LOWER bound on the true one. The error runs "
            "one way: a parcel reported as landlocked really is landlocked; a "
            "parcel reported as having frontage may only have a box nearby. "
            "Storing path endpoints at ingest is what would make both exact, "
            "and that is DOSSIER Phase 5c's decision"
        ),
        "without_any_geometry_note": (
            "an entity with neither a ring nor a bounding box is COUNTED and "
            "named, never treated as a road at distance zero (G8)"
        ),
    }
    return items, census


def _box_contains(outer: Box, inner: Box) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _nearest(
    parcel: Mapping[str, Any],
    roads: Sequence[Mapping[str, Any]],
    indexes: Sequence[int],
) -> tuple[float | None, Mapping[str, Any] | None]:
    """The closest road among `indexes`, and which one it was.

    Nearest-first with a lower bound, so a parcel with fifty candidates does not
    pay for fifty ring comparisons: the separation of the two BOUNDING BOXES can
    never exceed the true distance, so once the best exact distance is at or
    below the next candidate's box separation, nothing further can win. That is
    an ordering, not an approximation — the answer is identical to comparing
    every candidate.
    """
    best: float | None = None
    best_road: Mapping[str, Any] | None = None
    ordered = sorted(
        indexes, key=lambda k: _box_distance(parcel["_box"], roads[k]["_box"])
    )
    for k in ordered:
        road = roads[k]
        floor = _box_distance(parcel["_box"], road["_box"])
        if best is not None and floor >= best:
            break
        value = spatial._ring_distance(parcel["_ring"], road["_ring"])
        if value is None:
            continue
        if best is None or value < best:
            best = value
            best_road = road
            if best == 0.0:
                break
    return best, best_road


def _closed_ring_candidates(
    drawing_id: str, layout: str | None, tested: Sequence[str]
) -> list[dict[str, Any]]:
    """Layers of closed rings this run did NOT test, ranked by a CORRIDOR SIGNAL.

    Offered when the run established nothing, so that "supply a better road
    layer" is a sentence someone can act on. Read from the Dossier, which
    profiles a layer by its geometry, so a corridor layer whose name says
    nothing still appears.

    **This used to rank by AREA, and that was measured to be wrong.** On the
    reference drawing the largest closed-ring layer is the SITE BOUNDARY.
    Every parcel lies inside the site boundary, so a retry against it comes
    back with universal frontage — a wrong answer wearing a confident label,
    which is worse than the honest refusal it replaced. The layer that really
    carries the right of way is only the second largest and its name contains
    no road word, so neither size nor name finds it.

    What finds it is geometry, and the rule is general: **a road corridor
    contains the road network and not the parcels; a site boundary contains
    both; a parcel layer contains neither.** `recipes.corridor` measures those
    two fractions from points the store already holds and ranks by
    `network_inside x (1 - parcels_inside)`; every row that comes back carries
    both fractions, the sample they were taken from, and one sentence saying
    why it ranked where it did.

    Nothing here claims a layer IS a road. The rows say a layer BEHAVES like a
    corridor on this drawing, which is what geometry can say.
    """
    try:  # pragma: no cover - kept importable-or-silent, as the reader was
        from . import corridor
    except Exception:  # noqa: BLE001
        return []
    try:
        # `rank_region_layers` does not raise: every failure comes back as a
        # block with `ranked: False` and a reason. The guard is for a defect in
        # it, which must degrade this response rather than end it.
        block = corridor.rank_region_layers(drawing_id, layout, exclude=tested)
    except Exception:  # noqa: BLE001
        return []

    rows = list(block.get("candidates") or [])
    # The reason the ordering changed travels ON the rows, because this
    # function can only return a list: the layer a ranking by size would have
    # proposed usually scores zero and therefore does not appear among the
    # rows at all, and a reader who cannot see that it was CONSIDERED and
    # rejected on evidence has only been asked to trust a new order.
    swallowed = block.get("contains_every_sampled_parcel") or {}
    by_size = block.get("ranking_by_size_would_have_proposed") or {}
    context = (
        f"{block.get('candidates_scored')} closed-ring layer(s) on this layout "
        f"were ranked by the corridor signal — {block.get('ranked_by')} — and "
        f"never by size. {swallowed.get('count', 0)} of them contain every "
        "sampled parcel and score zero for that reason"
        + (
            f", including {by_size.get('layer')!r}, the largest, which a "
            "ranking by area would have offered first"
            if by_size.get("layer")
            and (by_size.get("parcels_inside") or 0.0) >= 1.0
            else ""
        )
        + "."
    )
    for row in rows:
        row["ranking_context"] = context
    return rows


def _frontage_safety(
    *,
    established: int,
    unproven: int,
    without: int,
    candidates: Any = None,
) -> dict[str, Any]:
    """Whether a caller may draw a conclusion from this run.

    `usable_as_evidence` is False when nothing was established either way --
    every parcel unproven, none confirmed, none refuted. A zero produced that
    way is not a measurement, and a chain that intersects it inherits the zero
    without inheriting any reason for it.
    """
    total = established + unproven + without
    if total and unproven == total:
        candidates = candidates() if callable(candidates) else (candidates or [])
        return {
            "usable_as_evidence": False,
            "verdict": "NOTHING ESTABLISHED",
            "why": (
                f"all {total} parcels came back unproven: the road geometry "
                "supplied resolves only to bounding boxes, so no parcel could "
                "be confirmed as having frontage or refused it."
            ),
            "do_not": (
                "do NOT read `no_frontage.count == 0` as 'every parcel has "
                "frontage', and do not intersect this empty list with another "
                "answer -- the empty set here means the question was not "
                "settled, not that nothing matched"
            ),
            "how_to_get_an_answer": (
                "pass `road_layers` naming layers whose entities are CLOSED "
                "rings (a right-of-way corridor layer), which can be measured "
                "exactly rather than through a box"
            ),
            # Naming the candidates, because "supply a better layer" is
            # advice nobody can act on without knowing which layers exist and
            # which of them are closed rings. On the reference drawing the
            # corridor layer's name carries no road word at all, so a caller
            # told only to "find a road corridor layer" searches by name and
            # finds nothing -- the same dead end the residual hit. (The name
            # itself is deliberately not written here: a layer name from one
            # drawing does not belong in code, rule G1.)
            #
            # Ranked by the CORRIDOR SIGNAL -- how much of the road network
            # falls inside a layer's rings, against how much of the parcel
            # geometry does -- and offered as candidates, never as a claim
            # that any of them is a road.
            "candidate_road_layers": candidates,
            # The exact call to make, ready to copy. Naming candidates was not
            # enough: told to act on the hint, the agent invented parameters of
            # its own -- it narrowed an unrelated recipe to two layers it chose
            # itself -- and reached the same empty answer by a different wrong
            # road. A model asked to compose a call will compose one; a model
            # handed the call runs it.
            "retry_with": [
                {
                    "recipe": "frontage_check",
                    "params": {"road_layers": [row.get("layer")]},
                    # The row's own sentence, written where the fractions
                    # were measured, so the reason cannot drift from them.
                    "because": (
                        row.get("why")
                        or (
                            f"{row.get('layer')!r} behaves most like a road "
                            "corridor of the untested closed-ring layers here, "
                            "and closed rings can be measured exactly"
                        )
                    ),
                }
                for row in candidates[:3]
                if row.get("layer")
            ],
            "retry_note": (
                "run one of these verbatim -- do not compose parameters of your "
                "own, and change nothing about the other steps. They are "
                "ordered by how much the layer behaves like a road corridor, "
                "so the first is the best candidate and the second is the next "
                "strongest corridor signal"
            ),
            "candidate_basis": (
                "layers in this layout whose entities are closed rings, "
                "excluding the parcel layers this run already tested, ranked "
                "by a CORRIDOR SIGNAL: how much of the road network falls "
                "inside a layer's rings against how much of the parcel "
                "geometry does. A corridor holds the network and not the "
                "parcels; a site boundary holds both and scores zero, which "
                "matters because the site boundary is the largest such layer "
                "and a ranking by size proposes it first. Nothing here claims "
                "any of them IS a road corridor"
            ),
        }
    return {
        "usable_as_evidence": True,
        "verdict": "ESTABLISHED",
        "why": (
            f"{established} parcel(s) confirmed with frontage and {without} "
            f"confirmed without; {unproven} remain unproven and are listed "
            "separately rather than counted either way"
        ),
        "do_not": (
            "the unproven list is not a finding: do not add it to either side"
        ),
    }


def _run_frontage_check(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    parcel_layers: Sequence[str] | None,
    road_layers: Sequence[str] | None,
    touch_tolerance: float | None,
) -> dict[str, Any]:
    parcel_names, parcels_source = _library._resolve_layers(
        drawing_id, parcel_layers, role="parcel"
    )
    road_names, roads_source, road_notes = _road_layers(
        drawing_id, layout, road_layers
    )

    parcels, parcel_census = _ring_docs(drawing_id, layout, parcel_names)
    refuse_oversize(
        what="parcels entering the frontage check",
        size=len(parcels),
        limit=MAX_SCAN_RINGS,
        hint="Narrow `parcel_layers`.",
    )
    roads, road_census = _road_geometry(drawing_id, layout, road_names)
    refuse_oversize(
        what="road entities entering the frontage check",
        size=len(roads),
        limit=MAX_ROAD_ENTITIES,
        hint="Narrow `road_layers` to the layers that really carry the roads.",
    )

    # The scale that dimensions both tolerances, taken from THIS drawing's own
    # parcels. `sqrt(median area)` is a length in the drawing's units, so the
    # ratios below stay pure numbers.
    areas = sorted(p["_area"] for p in parcels)
    typical = math.sqrt(areas[len(areas) // 2]) if areas else None
    if typical is None or typical <= 0:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"not one parcel on {len(parcel_names)} layers has a measurable "
            f"ring in layout {layout!r}, so no touch tolerance can be derived "
            "from this drawing.",
            "Hand over `touch_tolerance` in DRAWING units, or check `layout` "
            "and `parcel_layers`. A tolerance invented here would be a number "
            "from another drawing, and frontage is exactly the answer a wrong "
            "tolerance gets wrong quietly.",
        )

    caller_gave_tolerance = touch_tolerance is not None
    tolerance = (
        float(touch_tolerance)
        if caller_gave_tolerance
        else DEFAULT_TOUCH_REL * typical
    )
    if tolerance < 0:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"touch_tolerance is {touch_tolerance!r}; a distance cannot be "
            "negative.",
            "It is a distance in DRAWING units. Zero means the ring must "
            "genuinely meet the road geometry.",
        )
    window = max(DEFAULT_SEARCH_REL * typical, tolerance)

    near, proposed = _cross_pairs(parcels, roads, pad=window)
    cross_total = sum(len(v) for v in near.values())
    refuse_oversize(
        what="parcel-to-road pairs that survived the bbox prefilter",
        size=cross_total,
        limit=MAX_FRONTAGE_PAIRS,
        hint=(
            "Narrow `parcel_layers` or `road_layers`. This count is taken AFTER "
            "the spatial prefilter, so it is already the work that would really "
            "be done."
        ),
    )

    # Counted before a single distance is taken, for the same reason as the
    # overlap scan: the pair count does not bound the cost, the vertex product
    # does. `_nearest` usually stops long before this, so the budget is
    # deliberately conservative — it bounds the worst case, not the typical one.
    work = sum(
        len(parcels[q]["_ring"]) * len(roads[t]["_ring"])
        for q, targets in near.items()
        for t in targets
    )
    refuse_oversize(
        what="ring-to-road vertex pairs to compare",
        size=work,
        limit=MAX_FRONTAGE_VERTEX_PAIRS,
        hint=(
            "This is a limit on WORK. Narrow `road_layers` to the layers that "
            "really carry the roads, narrow `parcel_layers`, or hand over a "
            "smaller `touch_tolerance`, which narrows the search window with it."
        ),
    )

    ring_roads = [k for k, road in enumerate(roads) if road.get("_basis") == "ring"]
    box_roads = [k for k, road in enumerate(roads) if road.get("_basis") != "ring"]

    established = 0
    unproven: list[dict[str, Any]] = []
    without: list[dict[str, Any]] = []
    unmeasured = 0
    clearances: list[float] = []
    swallowed = 0

    for idx, parcel in enumerate(parcels):
        candidates = set(near.get(idx, []))
        # Two searches, not one, because the two kinds of road answer with
        # different authority. A ring gives an exact distance; a bounding box
        # gives a LOWER bound. Taking the single minimum of the two would let a
        # box that happens to sit closer hide a ring that proves the answer.
        ring_best, ring_road = _nearest(
            parcel, roads, [k for k in ring_roads if k in candidates]
        )
        box_best, box_road = _nearest(
            parcel, roads, [k for k in box_roads if k in candidates]
        )

        measured = [v for v in (ring_best, box_best) if v is not None]
        best = min(measured) if measured else None
        if best is not None:
            clearances.append(best)

        if ring_best is not None and ring_best <= tolerance:
            established += 1
            continue

        if box_best is not None and box_best <= tolerance:
            contains = _box_contains(box_road["_box"], parcel["_box"])
            if contains:
                swallowed += 1
            unproven.append(
                {
                    "handle": parcel.get("handle"),
                    "layer": parcel.get("layer"),
                    "nearest_road": _length_quantity(
                        box_best,
                        units,
                        basis=(
                            f"the distance from {parcel.get('handle')} to the "
                            "bounding box of the nearest road entity"
                        ),
                        method=(
                            "shortest distance between the parcel ring and the "
                            "road entity's BOUNDING BOX — a lower bound on the "
                            "true distance"
                        ),
                    ),
                    "nearest_road_handle": box_road.get("handle"),
                    "nearest_road_layer": box_road.get("layer"),
                    "measured_from": "bounding_box",
                    "bounding_box_contains_the_parcel": contains,
                    "verdict": (
                        "inside the tolerance, but only against a bounding box: "
                        "neither frontage nor its absence is established"
                    ),
                }
            )
            continue

        if best is None:
            unmeasured += 1
            without.append(
                {
                    "handle": parcel.get("handle"),
                    "layer": parcel.get("layer"),
                    "nearest_road": _length_quantity(
                        None,
                        units,
                        basis=f"the distance from {parcel.get('handle')} to the nearest road",
                        method="not measured",
                        withheld_reason=(
                            "no road entity came inside the search window, so "
                            "the distance is greater than the window and was "
                            "not measured. Withheld is not zero, and it is not "
                            "infinity either"
                        ),
                    ),
                    "nearest_road_handle": None,
                    "nearest_road_layer": None,
                    "measured_from": None,
                    "verdict": "no road within the search window",
                }
            )
            continue

        best_road = ring_road if ring_best == best else box_road
        without.append(
            {
                "handle": parcel.get("handle"),
                "layer": parcel.get("layer"),
                "nearest_road": _length_quantity(
                    best,
                    units,
                    basis=f"the distance from {parcel.get('handle')} to the nearest road",
                    method=(
                        "shortest distance between the parcel ring and the "
                        + (
                            "road ring"
                            if best_road is not None
                            and best_road.get("_basis") == "ring"
                            else "road entity's bounding box"
                        )
                    ),
                ),
                "nearest_road_handle": (
                    best_road.get("handle") if best_road is not None else None
                ),
                "nearest_road_layer": (
                    best_road.get("layer") if best_road is not None else None
                ),
                "measured_from": (
                    best_road.get("_basis") if best_road is not None else None
                ),
                "verdict": "beyond the touch tolerance",
            }
        )

    without.sort(key=lambda r: (str(r["layer"]), str(r["handle"])))
    unproven.sort(key=lambda r: (str(r["layer"]), str(r["handle"])))
    clearance_sorted = sorted(clearances)
    median_clearance = (
        clearance_sorted[len(clearance_sorted) // 2] if clearance_sorted else None
    )

    looks_like_centreline = bool(
        median_clearance is not None
        and median_clearance > tolerance
        and len(without) > len(parcels) / 2
    )

    scope_note = (
        f"layout {layout!r}; {len(parcels)} parcels with a complete ring on "
        f"{len(parcel_names)} layers from {parcels_source}, against "
        f"{len(roads)} entities on {len(road_names)} layers from {roads_source}; "
        f"a parcel has frontage when its ring comes within {tolerance:g} "
        f"{units.get('length_unit') or 'drawing units'} of road geometry"
    )
    scope = scope_for(layout=layout, units=units, note=scope_note)

    observations: list[ev.Observation] = [
        _geometry_observation(
            f"of {len(parcels)} parcel rings, {established} come within "
            f"{tolerance:g} of road geometry that carries a ring, "
            f"{len(unproven)} come within it only against a bounding box, and "
            f"{len(without)} are further away than the tolerance",
            locator=",".join(road_names[:3]) or None,
            layout=layout,
        )
    ]
    if road_census["measured_from_their_bounding_box"]:
        observations.append(
            _geometry_observation(
                f"{road_census['measured_from_their_bounding_box']} of "
                f"{road_census['road_entities']} road entities carry no ring in "
                "this store and were measured against their bounding box, which "
                "under-states their distance",
                locator=",".join(road_names[:3]) or None,
                layout=layout,
            )
        )
    if parcel_census["skipped_total"]:
        observations.append(
            _absence_observation(
                f"{parcel_census['skipped_total']} polygons on the parcel layers "
                "have no ring that can be tested and were neither passed nor "
                "failed",
                locator=",".join(parcel_names[:3]) or None,
                layout=layout,
            )
        )

    evidence = _measured_evidence(
        f"{len(without)} of {len(parcels)} parcels do not come within "
        f"{tolerance:g} {units.get('length_unit') or 'drawing units'} of any "
        "road geometry in this scope",
        drawing_id=drawing_id,
        scope=scope,
        observations=observations,
        not_established=(
            "whether a parcel without frontage here is a DEFECT. This measures "
            "geometric proximity to the layers named as roads; access can be "
            "drawn on a layer nobody named, granted by an easement the file does "
            "not carry, or provided by a road on another layout. And the "
            "tolerance decides: a road CENTRELINE never touches a plot "
            "boundary, so a touch tolerance run against a centreline layer "
            "reports the whole population as landlocked and is arithmetically "
            "right about nothing"
        ),
        how_to_verify=(
            "read `clearance` below. If the median clearance sits well above "
            "the tolerance, the road layer is a centreline or a similar offset "
            "geometry: re-run with `touch_tolerance` set to at least half the "
            "carriageway width — a planning number this file does not carry. "
            "Then open the named handles and look"
        ),
    )

    body: dict[str, Any] = {
        "scope_note": scope_note,
        "parcel_layers": list(parcel_names),
        "parcel_layers_source": parcels_source,
        "road_layers": list(road_names),
        "road_layers_source": roads_source,
        "road_layers_notes": road_notes,
        "road_layers_never_by_name": (
            "road layers arrive from the caller, from this drawing's land use "
            "config (role=network), or from the Dossier's GEOMETRIC role — "
            "never from a layer name. Civil3D template layers named after roads "
            "are sheet layout, and a name test would report frontage onto a "
            "title block"
        ),
        "road_layers_caveat": (
            "the Dossier's `network` role is a GEOMETRIC fact — open paths whose "
            "endpoints meet — and not a statement that a layer is a road. A "
            "revision cloud and a kerb line profile as `network` too. The route "
            "that says ROAD is the land use config (role=network); when the "
            "layers below arrived by geometric role alone, read the list before "
            "reading the verdict"
        ),
        "method": (
            "shortest distance between the parcel ring and the road geometry, "
            "through `store_spatial._ring_distance` — the same edge-to-edge "
            "engine `adjacency` uses, which returns 0 for shapes that touch or "
            "overlap"
        ),
        "prefilter": (
            "the bounding box of every ring and every road entity, through "
            "`store_spatial._SpatialIndex`, expanded by the search window so "
            "that no pair inside the window can be missed"
        ),
        "tolerance": {
            "value": _length_quantity(
                tolerance,
                units,
                basis="the distance within which a parcel counts as touching a road",
                method=(
                    "handed over by the caller"
                    if caller_gave_tolerance
                    else f"{DEFAULT_TOUCH_REL:g} x sqrt(median parcel ring area)"
                ),
            ),
            "supplied_by_caller": caller_gave_tolerance,
            "touch_ratio": None if caller_gave_tolerance else DEFAULT_TOUCH_REL,
            "typical_parcel_size": _length_quantity(
                typical,
                units,
                basis="the square root of the median parcel ring area in this scope",
                method="sqrt of the median of the measured ring areas",
            ),
            "basis": (
                "a RATIO of a length measured from this drawing's own parcels, "
                "never a number of drawing units. That is what makes the same "
                "rule behave identically in metres, in inches, and in a drawing "
                "that declares no unit at all (G2), and what keeps one "
                "drawing's trait out of the code (G1)"
            ),
            "not_derived_from_the_answer": (
                "the tolerance is NOT taken from the clearances it then judges. "
                "A tolerance read off the distribution of measured distances "
                "would report every parcel as having frontage by construction, "
                "which is a tautology wearing the clothes of a measurement"
            ),
            "decides": (
                "this number decides the answer. Halve it and parcels move into "
                "the landlocked list; double it and they move out"
            ),
        },
        "search_window": {
            "value": _length_quantity(
                window,
                units,
                basis="how far this check looks when MEASURING the nearest road",
                method=f"max({DEFAULT_SEARCH_REL:g} x the typical parcel size, the tolerance)",
            ),
            "why_wider_than_the_tolerance": (
                "a clearance distribution measured only inside the tolerance "
                "would contain nothing but parcels that already passed, and the "
                "one diagnosis that matters — that the road layer is a "
                "centreline — would be invisible. A parcel with no road inside "
                "this window has its distance WITHHELD, not set to infinity"
            ),
        },
        "parcels_tested": len(parcels),
        "parcel_census": parcel_census,
        "road_census": road_census,
        "pairs_proposed_by_prefilter": proposed,
        "pairs_after_prefilter": cross_total,
        "vertex_pairs_budgeted": work,
        "vertex_pairs_budget": MAX_FRONTAGE_VERTEX_PAIRS,
        "vertex_pairs_note": (
            "the worst case, counted before any distance was taken. The "
            "nearest-first search with a bounding-box lower bound usually stops "
            "far short of it, which is why this budget is larger than the "
            "overlap scan's: ring clipping has no such exit and pays its whole "
            "product"
        ),
        "three_answers_not_two": (
            "frontage established, frontage NOT established, and no frontage. "
            "The middle one exists because a road entity with no ring is "
            "measured against its bounding box, and a box that comes within the "
            "tolerance proves nothing — the box contains the road, so the true "
            "distance can only be larger. Collapsing that into 'has frontage' "
            "would publish a pass nobody measured"
        ),
        "frontage_established": established,
        "frontage_established_note": (
            "measured against road geometry that carries a RING, so the distance "
            "is exact and the pass is proven"
        ),
        "frontage_unproven": {
            "count": len(unproven),
            "returned": len(unproven[:MAX_ROWS_RETURNED]),
            "truncated": len(unproven) > MAX_ROWS_RETURNED,
            "cap": MAX_ROWS_RETURNED,
            "rows": unproven[:MAX_ROWS_RETURNED],
            "bounding_box_swallows_the_parcel": swallowed,
            "why": (
                "these parcels came inside the tolerance only against a road "
                "entity's BOUNDING BOX. That is neither a pass nor a failure: "
                "the box contains the road, so the real distance is at least "
                "this and may be much more. `bounding_box_swallows_the_parcel` "
                "counts the worst case — a box so large it contains the whole "
                "parcel, where the measurement establishes nothing at all"
            ),
            "how_to_settle_it": (
                "store the endpoints of path entities at ingest (DOSSIER Phase "
                "5c's decision) and every one of these becomes exact; or name a "
                "road-corridor layer whose entities are closed rings in "
                "`road_layers`"
            ),
        },
        "no_frontage": {
            "count": len(without),
            "returned": len(without[:MAX_ROWS_RETURNED]),
            "truncated": len(without) > MAX_ROWS_RETURNED,
            "cap": MAX_ROWS_RETURNED,
            "rows": without[:MAX_ROWS_RETURNED],
            "beyond_the_search_window": unmeasured,
            "named_not_counted": (
                "every parcel without frontage is named by handle and layer. A "
                "count is not actionable; a handle is clickable"
            ),
            "why_this_list_is_sound": (
                "a bounding box CONTAINS its entity, so every distance measured "
                "here is a LOWER bound on the true one. A parcel further than "
                "the tolerance from every box is further than the tolerance from "
                "every road, whatever the boxes hide. The approximation cannot "
                "put a parcel on this list that does not belong there — it can "
                "only keep one off it"
            ),
        },
        # Whether this run may be used as EVIDENCE, said in a field a caller
        # can branch on rather than only in prose.
        #
        # The failure this exists to stop, measured today: asked which parcels
        # are both size outliers and without frontage, the agent ran this
        # recipe with its default road layers, got `no_frontage: 0` because
        # every parcel was merely unproven, intersected that empty set with 30
        # outliers, and reported "there are none" -- a clean negative finding
        # inherited from a step that had established nothing. Run against the
        # closed corridors instead, one parcel is in fact both.
        #
        # An empty result and an empty QUESTION look identical downstream
        # unless the response says which it was.
        "conclusion_safety": _frontage_safety(
            established=established,
            unproven=len(unproven),
            without=len(without),
            # Passed as a THUNK, not a value: the corridor scan costs ~2 s and
            # is only ever read when the run established nothing. Evaluated as
            # an argument it ran on every call and its result was discarded on
            # the ones that succeeded -- work paid for inside a request the
            # viewer is waiting on.
            candidates=lambda: _closed_ring_candidates(
                drawing_id, layout, parcel_layers
            ),
        ),
        "clearance": {
            "measured": len(clearances),
            "not_measured": len(parcels) - len(clearances),
            "min": _length_quantity(
                clearance_sorted[0] if clearance_sorted else None,
                units,
                basis="the closest a parcel comes to road geometry",
                method="shortest ring-to-road distance",
                withheld_reason="no parcel had a road inside the search window",
            ),
            "median": _length_quantity(
                median_clearance,
                units,
                basis="the median distance from a parcel to its nearest road",
                method="median over the measured clearances",
                withheld_reason="no parcel had a road inside the search window",
            ),
            "max": _length_quantity(
                clearance_sorted[-1] if clearance_sorted else None,
                units,
                basis="the farthest a parcel sits from road geometry, inside the window",
                method="shortest ring-to-road distance",
                withheld_reason="no parcel had a road inside the search window",
            ),
            "why_it_is_here": (
                "so that the tolerance can be chosen from evidence rather than "
                "from habit. It is reported, and it is deliberately NOT used to "
                "set the tolerance"
            ),
        },
        "road_geometry_warning": (
            (
                "the median clearance sits above the touch tolerance and most "
                "parcels fail. That is the signature of road geometry drawn "
                "AWAY from the plot boundary — a centreline, or a carriageway "
                "edge on another layer — rather than of a scheme without "
                "access. Re-run with `touch_tolerance` at about half the "
                "carriageway width before reading this list as a defect list"
            )
            if looks_like_centreline
            else None
        ),
        "limits_applied": {
            "parcels": MAX_SCAN_RINGS,
            "road_entities": MAX_ROAD_ENTITIES,
            "pairs_after_prefilter": MAX_FRONTAGE_PAIRS,
            "vertex_pairs_total": MAX_FRONTAGE_VERTEX_PAIRS,
            "rows_returned": MAX_ROWS_RETURNED,
        },
    }
    return ev.attach(body, evidence)


register(
    Recipe(
        name="frontage_check",
        answers="which parcels do not touch a road, named by handle",
        when_to_use=(
            "'does every plot have road access', 'which plots cannot be reached "
            "from a road', or before a plot schedule is signed off. A plot with "
            "no road frontage cannot be built on or sold, so this is a "
            "review-blocking defect rather than a nicety."
        ),
        params=(
            Param(
                "parcel_layers",
                "layers",
                "the polygon layers whose rings are checked for frontage. Left "
                "empty, derived from this drawing's land use config "
                "(role=parcel) and the response names them.",
            ),
            Param(
                "road_layers",
                "layers",
                "the layers carrying the road geometry — centrelines, corridors, "
                "rights of way. Left empty they are derived from this drawing's "
                "land use config (role=network) and from the Dossier's GEOMETRIC "
                "role `network`, and the response says which route answered. "
                "They are never derived from a layer NAME: template layers named "
                "after roads are sheet layout.",
            ),
            Param(
                "touch_tolerance",
                "number",
                "the distance within which a parcel counts as touching a road, "
                "in DRAWING units. Left empty it is derived from this drawing's "
                f"own geometry as {DEFAULT_TOUCH_REL:g} x sqrt(median parcel "
                "area), and the response publishes the value it took. Give it "
                "explicitly when the road layer is a CENTRELINE: half a "
                "carriageway width is a planning number, and this file does not "
                "carry it.",
                default=None,
            ),
        ),
        returns=(
            "frontage_established",
            "frontage_unproven.count",
            "no_frontage.count",
            "no_frontage.rows[].handle",
            "no_frontage.rows[].nearest_road",
            "clearance.median",
            "tolerance.value",
            "road_layers_source",
            "road_census.measured_from_their_bounding_box",
            "road_geometry_warning",
        ),
        built_on=(
            "store_spatial._ring_distance",
            "store_spatial._SpatialIndex",
            "dossier_read.profiles_for",
            "recipes.library._resolve_layers",
        ),
        limits={
            "parcels": MAX_SCAN_RINGS,
            "road_entities": MAX_ROAD_ENTITIES,
            "pairs_after_prefilter": MAX_FRONTAGE_PAIRS,
            "vertex_pairs_total": MAX_FRONTAGE_VERTEX_PAIRS,
            "rows_returned": MAX_ROWS_RETURNED,
        },
        run=_run_frontage_check,
    )
)
