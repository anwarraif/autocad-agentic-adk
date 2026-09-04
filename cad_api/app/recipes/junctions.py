"""How the paths on a layer MEET: dead ends, T-junctions, crossroads.

Owner: the TOPOLOGY-ENDPOINTS lane — DOSSIER Phase 5c.

The question this exists for was asked out loud and could not be answered:
*how many T-junctions, crossroads, dead ends and simple connections are there?*
The agent said the computation was not available yet. It was honest and it was
useless, and the reason was one line down in the data layer: the store kept a
BOUNDING BOX for every `LINE` and `ARC` and nothing else. A box has four
corners, a path has two ends, and nothing in the file records which diagonal is
which — so junctions were not merely uncomputed, they were uncomputable from
the store. Phase 1's chain count says `approximate` for exactly that reason.

`extract.py` now stores the two ENDS of every open path, and the VERTICES of
every open polyline. This recipe is what those fields were for.

## Branch degree, not path count

The count that matters is **branches**, and the difference is not a nicety:

    degree(node) = (paths that END at it) + 2 x (paths whose INTERIOR passes
                    through it)

A road that runs THROUGH a junction contributes two branches, not one. Count
"how many paths meet here" instead and a T-junction where the through-road was
never split reads as a dead end, a crossroads reads as a simple connection, and
a drawing with two dozen crossroads reports none at all. The rule is stated
here because it is the whole instrument.

## What is exact, what is bounded, and why no node is guessed

`ends` are exact, so the NODES are exact: two paths meet when their stored ends
fall within the snap tolerance of one another. The second half of the degree
rule needs a path's INTERIOR, and that is where the store still has an edge. So
each entity is put in one of two classes, per row rather than per type:

* **interior known** — a closed polygon with a complete `ring`; an open
  polyline with stored `path_points`, whose arc spans are measured AS ARCS
  rather than as their chords; a `LINE` (the segment between its two ends IS
  the line); and any open path whose stored `length` equals the distance
  between its ends, which is only possible if it is straight;
* **interior unknown** — an `ARC` entity, a `CIRCLE`, a closed polyline whose
  ring could not be completed, an open polyline over the vertex limit, a
  spline. Nothing stored says where the middle of those went.

An entity of the second class is not ignored and is not treated as absent. Its
bounding box is stored, a box CONTAINS its entity, so a node OUTSIDE the box is
proof that this entity's interior does not pass through that node, and a node
inside it is genuinely unknown. Every node therefore carries a degree RANGE:

    degree_min = ends + 2 x (interiors proven to pass)
    degree_max = degree_min + 2 x (interiors that might)

A node is counted as a dead end, a simple connection, a T-junction or a
crossroads only when `degree_min == degree_max`. The rest are published in
`undecided`, each with its range and the handles that made it uncertain. A
number that was merely unmeasurable is never rounded down to a category (G8) —
that is precisely the failure mode that produced "this computation is not
available yet" wearing a different face.

Measured on the reference drawing's live road copy at its own derived
tolerance, 26 August 2026: 410 paths, 561 nodes, **5 undecided**, 263 of 410
interiors known. Crossroads and simple connections come back EXACT; dead ends
and T-junctions carry a range four to five nodes wide, and every one of those
five nodes is undecided because an ARC entity's box reaches it. Storing an
arc's centre, radius and angles would close the last of it; it is left open
deliberately, because the range says so and a reader can see how small it is.

The vertex field is what made that possible, and the measurement that bought
it is worth keeping: with the two endpoints alone the same census left **83 of
561** nodes undecided and put T-junctions at 147-216. Vertices cost 20,518
points across the whole 20-drawing store.

## The tolerance decides the answer, so it is published

Two ends are the same node when they are within `snap_tolerance` of each other.
Raise it and dead ends turn into junctions; lower it and junctions split into
dead ends. It is therefore DERIVED from the drawing — `dossier_network`'s own
rule, `SNAP_FRACTION x the median entity length`, imported rather than copied
so that the chain count and the junction count can never drift apart — and it
is published with its basis, its absolute value in this drawing's units, and a
sensitivity table re-run across a 16x band (G2, G7).

## The triplication trap

The reference layer holds the same estate drawn THREE times: three spatially
detached clusters of 411 entities each. An answer that counts all 1,233 triples
every junction in the estate, and an answer that divides by three without
saying so is worse. So this recipe PARTITIONS the layer into copies — through
`dossier_anomaly`'s own partition, used unchanged — reports every copy's census
separately, and names which copy the headline describes. The live copy is
identified by measurement, not by position: it is the one the REST of the
drawing sits on top of, counted as the number of entities on other layers whose
bounding box starts inside that copy's box. On the reference drawing that count
is 18,986 for the middle copy and 0 for each of the other two, and the three
copies' censuses agree exactly — which is what makes them copies.

## Layer names never live in this file (G1)

Path layers arrive from the caller, from this drawing's land use config
(`role: network`), or from the Dossier's GEOMETRIC role — through
`recipes.topology._road_layers`, the same reader `frontage_check` uses, so that
one config cannot be interpreted two ways. Never from a layer name.
"""

from __future__ import annotations

import math
from typing import Any, Final, Mapping, Sequence

from .. import dossier_anomaly as anomaly
from .. import dossier_network as network
from .. import evidence as ev
from .. import region
from .. import store_spatial as spatial
from ..extract import ENDS_CLOSED, ENDS_NOT_DERIVED, ENDS_OPEN
from ..mongo import COLL_ENTITIES, coll

# `topology` is imported as a MODULE, for its layer reader, its evidence
# helpers and its box arithmetic. Copying any of them here would put a second
# interpretation of the same config — and a second definition of "how far
# apart" — into the repo, and two interpretations of one config are how two
# answers about the same drawing start to differ.
from . import topology as _topology
from .registry import (
    Param,
    Recipe,
    RecipeRefused,
    refuse_oversize,
    register,
    scope_for,
)

Point = tuple[float, float]
Box = tuple[float, float, float, float]


# =============================================================================
# Stated limits (G7)
# =============================================================================

#: Path entities that may enter one census. Janadriyah's whole road layer is
#: 1,233 rows across three copies, so this sits far above the largest drawing
#: in the store and exists so that a filter which filters nothing refuses
#: instead of running for ten minutes.
MAX_PATH_ENTITIES: Final[int] = 50_000

#: Endpoint comparisons the node clustering may make. Taken from
#: `dossier_network` rather than re-chosen, because it is the same union-find
#: over the same kind of points and a second number would make the two lanes
#: refuse at different sizes on the same layer.
MAX_POINT_COMPARISONS: Final[int] = network.MAX_POINT_COMPARISONS

#: The interior-test work ONE census may do, counted as the sum of the vertex
#: counts of the candidate geometries over the (node, entity) pairs that
#: survive the bounding-box prefilter — and counted BEFORE a single distance is
#: taken, for the reason `recipes.topology` counts its vertex pairs first: a
#: limit reported after the work has been done is a report, not a limit.
MAX_INTERIOR_VERTEX_TESTS: Final[int] = 20_000_000

#: Copies listed in the response. A layer holding more detached copies than
#: this is not a drawing with copies, it is a drawing made of tiles, and the
#: census would say nothing about either.
MAX_COPIES_REPORTED: Final[int] = 20

#: Nodes named individually in each bucket. The counts are always complete; it
#: is the row lists that are capped, and every cap is stated in the response.
MAX_NODES_LISTED: Final[int] = 50

#: Above this many paths in the described copy, the sensitivity table is
#: skipped and the omission is stated instead of being timed. Each extra factor
#: is a full re-run of the census.
MAX_PATHS_FOR_SENSITIVITY: Final[int] = 20_000


# =============================================================================
# The tolerance, as a DIMENSIONLESS ratio
# =============================================================================

#: Imported, not re-chosen: `dossier_network` already derives a snap tolerance
#: for exactly this population by exactly this rule, and its chain count and
#: this junction census must never be able to disagree about whether two ends
#: are the same point.
SNAP_FRACTION: Final[float] = network.SNAP_FRACTION
FLOAT_NOISE_FACTOR: Final[float] = network.FLOAT_NOISE_FACTOR

#: How close a path's stored `length` must sit to the distance between its two
#: ends before the path is treated as STRAIGHT and its interior is taken to be
#: the segment between them. Relative, so it carries no unit (G2). A curve is
#: strictly longer than its chord, so equality within this ratio is a proof of
#: straightness and not an assumption about shape (G6).
STRAIGHTNESS_REL: Final[float] = 1e-9

#: Factors the whole census is re-run at, so that a reader can see the
#: tolerance's grip rather than take the headline on trust. The band spans 16x
#: on purpose: a table of 0.5x and 2x can look flat on a number that moves.
SENSITIVITY_FACTORS: Final[tuple[float, ...]] = (0.25, 0.5, 2.0, 4.0)

#: What each branch degree is called. 5 and above has no name here on purpose:
#: five roads meeting at one point is unusual enough that naming it would
#: normalise it, and the response says so and lists the handles.
DEGREE_NAMES: Final[dict[int, str]] = {
    1: "dead_ends",
    2: "simple_connections",
    3: "t_junctions",
    4: "crossroads",
}

#: Types that are not path geometry at all, so they have no ends and are not
#: components either. Taken from `dossier_network` unchanged: a network layer
#: regularly carries a stray label or block, and letting a TEXT's box glue two
#: roads together is a defect that lane already names.
NOT_PATH_TYPES: Final[frozenset[str]] = network._NOT_PATH_TYPES


def _entities():
    """This module's only door to MongoDB, and it goes through `coll()`."""
    return coll(COLL_ENTITIES)


# =============================================================================
# Pure geometry and pure arithmetic
# =============================================================================
#
# Nothing below touches MongoDB, config, or units. That is what lets the tests
# pin a junction census to an exact number without a database.


def _finite(value: Any) -> float | None:
    """A finite float, or None. `True` is not 1.0 and NaN is not a coordinate."""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _point(raw: Any) -> Point | None:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return None
    if len(raw) < 2:
        return None
    x, y = _finite(raw[0]), _finite(raw[1])
    if x is None or y is None:
        return None
    return x, y


def _ends_of(doc: Mapping[str, Any]) -> tuple[Point, Point] | None:
    """The two stored ends of one row, or None when they are not there."""
    raw = doc.get("ends")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    if len(raw) != 2:
        return None
    first, last = _point(raw[0]), _point(raw[1])
    if first is None or last is None:
        return None
    return first, last


def _box_of_points(points: Sequence[Point]) -> Box | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _box_of_doc(doc: Mapping[str, Any]) -> Box | None:
    raw = doc.get("bbox")
    if not isinstance(raw, Mapping):
        return None
    lo, hi = _point(raw.get("min")), _point(raw.get("max"))
    if lo is None or hi is None:
        return None
    return (min(lo[0], hi[0]), min(lo[1], hi[1]), max(lo[0], hi[0]), max(lo[1], hi[1]))


# --- spans: a path as the pieces it is really made of ------------------------
#
# A path is a run of SPANS, each of which is a straight segment or a circular
# arc. Keeping the arc as an arc, rather than chopping it into chords, is what
# makes the interior test exact: the distance from a point to a circular arc
# has a closed form, so there is no flattening resolution to choose, no error
# to bound, and nothing to state as a caveat.
#
#   ("seg", p0, p1)
#   ("arc", centre, radius, start_angle, sweep, p0, p1)

Span = tuple


def _bulge_span(p0: Point, p1: Point, bulge: float) -> Span:
    """One polyline span, as an arc when it bulges and a segment when it does not.

    `bulge` is `tan(quarter of the included angle)`, the DXF way of saying how
    much a span bows away from its chord. Positive is counter-clockwise. The
    construction below is the standard one and is checked in the tests against
    `ezdxf.math.bulge_to_arc`, an independent implementation of the same
    definition -- a number agreeing with itself is not a check.
    """
    chord = math.dist(p0, p1)
    if abs(bulge) <= 1e-9 or chord == 0.0:
        return ("seg", p0, p1)
    theta = 4.0 * math.atan(bulge)
    half = theta / 2.0
    sin_half = math.sin(half)
    if abs(sin_half) < 1e-15:
        return ("seg", p0, p1)
    radius = chord / (2.0 * sin_half)
    mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    # The LEFT normal of the chord. The centre sits along it at
    # `radius * cos(theta/2)`, which is negative for a major arc and puts the
    # centre on the other side -- the same formula covers both.
    normal = (-(p1[1] - p0[1]) / chord, (p1[0] - p0[0]) / chord)
    offset = radius * math.cos(half)
    centre = (mid[0] + normal[0] * offset, mid[1] + normal[1] * offset)
    start = math.atan2(p0[1] - centre[1], p0[0] - centre[0])
    return ("arc", centre, abs(radius), start, theta, p0, p1)


def _spans_of(points: Sequence[Point], bulges: Sequence[float], closed: bool) -> list[Span]:
    n = len(points)
    if n < 2:
        return []
    last = n if closed else n - 1
    out: list[Span] = []
    for i in range(last):
        bulge = bulges[i] if i < len(bulges) else 0.0
        out.append(_bulge_span(points[i], points[(i + 1) % n], bulge))
    return out


def _span_distance(x: float, y: float, span: Span) -> float:
    """Shortest distance from a point to one span. Exact for both kinds.

    Segments go through `region._point_segment_distance` rather than a second
    implementation: `store_spatial._ring_distance` and `recipes.truth` both
    call that same function, and two implementations of one geometric test are
    how two answers about the same shape start to differ. Arcs have no
    equivalent anywhere in this repo, so the closed form is written here --
    `abs(|P - centre| - radius)` when P's bearing falls inside the sweep, and
    the nearer endpoint when it does not.
    """
    if span[0] == "seg":
        return region._point_segment_distance(x, y, span[1], span[2])
    _kind, centre, radius, start, sweep, p0, p1 = span
    bearing = math.atan2(y - centre[1], x - centre[0])
    if sweep >= 0.0:
        along = (bearing - start) % (2.0 * math.pi)
        inside = along <= sweep
    else:
        along = (start - bearing) % (2.0 * math.pi)
        inside = along <= -sweep
    if inside:
        return abs(math.hypot(x - centre[0], y - centre[1]) - radius)
    return min(math.hypot(x - p0[0], y - p0[1]), math.hypot(x - p1[0], y - p1[1]))


def _distance_to_spans(x: float, y: float, spans: Sequence[Span]) -> float:
    if not spans:
        return math.inf
    return min(_span_distance(x, y, span) for span in spans)


class _Path:
    """One entity, reduced to what a junction census needs of it.

    `spans` is the entity's interior where the store DETERMINES it, and `None`
    where it does not. The distinction is carried per row rather than per type,
    because it is not a property of the type: a `LWPOLYLINE` with stored
    vertices is exact whether it curves or not, while its neighbour whose
    vertices were over the storage limit is not.

    Four routes to a known interior, tried in this order because each is
    strictly better evidence than the next:

    1.  a complete stored `ring` -- a closed polygon, in full;
    2.  stored `path_points` (+ `path_bulges`) -- an open polyline, in full,
        with every arc segment kept as an ARC rather than chopped into chords;
    3.  a `LINE` -- the segment between its two stored ends IS the line;
    4.  any open path whose stored `length` equals the distance between its
        ends: only a straight path can be as short as its own chord.
    """

    __slots__ = (
        "index", "handle", "layer", "type", "box", "ends", "status",
        "spans", "shape_basis", "length",
    )

    def __init__(self, index: int, doc: Mapping[str, Any]) -> None:
        self.index = index
        self.handle = str(doc.get("handle") or "")
        self.layer = str(doc.get("layer") or "")
        self.type = str(doc.get("type") or "")
        self.length = _finite(doc.get("length"))
        self.ends = _ends_of(doc)
        self.status = doc.get("ends_status")
        self.spans: list[Span] | None = None
        self.shape_basis = ""

        ring = [p for p in (_point(v) for v in (doc.get("ring") or [])) if p]
        points = [p for p in (_point(v) for v in (doc.get("path_points") or [])) if p]
        bulges = [b for b in (_finite(v) for v in (doc.get("path_bulges") or [])) if b is not None]

        if doc.get("ring_status") == "complete" and len(ring) >= 3:
            self.spans = _spans_of(ring, (), closed=True)
            self.shape_basis = "the stored ring, complete and in full"
        elif len(points) >= 2:
            self.spans = _spans_of(points, bulges, closed=False)
            self.shape_basis = (
                "the open polyline's stored vertices, in full"
                + (
                    "; its arc spans are measured as arcs, not as their chords"
                    if bulges
                    else ""
                )
            )
        elif self.ends is not None and self.type == "LINE":
            self.spans = [("seg", self.ends[0], self.ends[1])]
            self.shape_basis = (
                "the segment between the two stored ends, which for a LINE is "
                "the whole line"
            )
        elif self.ends is not None and self._is_straight():
            self.spans = [("seg", self.ends[0], self.ends[1])]
            self.shape_basis = (
                "the segment between the two stored ends: the stored length "
                "equals the distance between them, and only a straight path can "
                "be as short as its own chord"
            )

        self.box = _box_of_doc(doc)
        if self.box is None and points:
            self.box = _box_of_points(points)
        if self.box is None and self.ends is not None:
            self.box = _box_of_points([self.ends[0], self.ends[1]])

    def _is_straight(self) -> bool:
        if self.length is None or self.ends is None:
            return False
        chord = math.dist(self.ends[0], self.ends[1])
        if self.length <= 0.0:
            return False
        return abs(self.length - chord) <= STRAIGHTNESS_REL * max(1.0, self.length)

    def distance_to(self, x: float, y: float) -> float:
        assert self.spans is not None
        return _distance_to_spans(x, y, self.spans)

    def vertices(self) -> int:
        """The work one interior test costs, in spans. `1` when unknown, which
        is the bounding-box test that is done instead."""
        return len(self.spans) if self.spans is not None else 1


class _Node:
    """One place where path ends meet, and everything known about its degree."""

    __slots__ = ("point", "ends", "ending", "proven", "possible")

    def __init__(self, point: Point, ends: int, ending: set[int]) -> None:
        self.point = point
        self.ends = ends
        self.ending = ending
        self.proven: list[int] = []
        self.possible: list[int] = []

    @property
    def degree_min(self) -> int:
        return self.ends + 2 * len(self.proven)

    @property
    def degree_max(self) -> int:
        return self.degree_min + 2 * len(self.possible)

    @property
    def decided(self) -> bool:
        return not self.possible


def _cluster_endpoints(
    paths: Sequence[_Path], tolerance: float, max_comparisons: int
) -> list[_Node] | None:
    """Union-find over the stored ends; one node per cluster.

    `None` means the comparison cap was hit — the ends are packed more densely
    than the cap allows to scan. That is reported, never truncated.

    The grid is the same construction `dossier_network._chains_at` uses, and for
    the same reason: at a cell size equal to the tolerance, two points within
    the tolerance can only ever be in the same cell or in one of the eight
    around it.
    """
    ends: list[tuple[int, float, float]] = []
    for path in paths:
        if path.ends is None:
            continue
        ends.append((path.index, path.ends[0][0], path.ends[0][1]))
        ends.append((path.index, path.ends[1][0], path.ends[1][1]))
    if not ends:
        return []

    parent = list(range(len(ends)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if tolerance > 0.0:
        grid: dict[tuple[int, int], list[int]] = {}
        for idx, (_pid, x, y) in enumerate(ends):
            grid.setdefault(
                (math.floor(x / tolerance), math.floor(y / tolerance)), []
            ).append(idx)
        tol2 = tolerance * tolerance
        comparisons = 0
        for idx, (_pid, x, y) in enumerate(ends):
            cx = math.floor(x / tolerance)
            cy = math.floor(y / tolerance)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for jdx in grid.get((cx + dx, cy + dy), ()):
                        if jdx <= idx:
                            continue
                        comparisons += 1
                        if comparisons > max_comparisons:
                            return None
                        _pid2, x2, y2 = ends[jdx]
                        if (x - x2) ** 2 + (y - y2) ** 2 <= tol2:
                            union(idx, jdx)
    else:
        # No scale could be derived, so only exact coincidence counts. That
        # SPLITS rather than merges, which makes every degree a floor; the
        # tolerance block says so.
        first_at: dict[Point, int] = {}
        for idx, (_pid, x, y) in enumerate(ends):
            seen = first_at.get((x, y))
            if seen is None:
                first_at[(x, y)] = idx
            else:
                union(seen, idx)

    members: dict[int, list[int]] = {}
    for idx in range(len(ends)):
        members.setdefault(find(idx), []).append(idx)

    nodes: list[_Node] = []
    for root in sorted(members):
        group = members[root]
        xs = [ends[i][1] for i in group]
        ys = [ends[i][2] for i in group]
        nodes.append(
            _Node(
                point=(sum(xs) / len(xs), sum(ys) / len(ys)),
                ends=len(group),
                ending={ends[i][0] for i in group},
            )
        )
    # Sorted by position so that two runs over the same data list the same node
    # first. Determinism that cannot be checked is not determinism.
    nodes.sort(key=lambda n: (n.point[0], n.point[1]))
    return nodes


def _attach_interiors(
    nodes: Sequence[_Node],
    paths: Sequence[_Path],
    tolerance: float,
    budget: int,
) -> int:
    """Fill each node's `proven` and `possible` lists. Returns the work done.

    Two sweeps, and the first one only COUNTS. Taking the distances first and
    reporting the cost afterwards would mean the work the budget exists to
    refuse has already been done by the time anyone hears about it — the same
    discipline as `recipes.topology._run_overlap_scan` and
    `store_spatial._join`.
    """
    grown: list[Box | None] = [
        None
        if path.box is None
        else (
            path.box[0] - tolerance,
            path.box[1] - tolerance,
            path.box[2] + tolerance,
            path.box[3] + tolerance,
        )
        for path in paths
    ]
    index = spatial._SpatialIndex(grown)

    candidates: list[list[int]] = []
    work = 0
    for node in nodes:
        hits = [
            i
            for i in index.hits(node.point[0], node.point[1])
            if paths[i].index not in node.ending
        ]
        hits.sort()
        candidates.append(hits)
        work += sum(paths[i].vertices() for i in hits)

    refuse_oversize(
        what="interior vertex tests to make",
        size=work,
        limit=budget,
        hint=(
            "This is a limit on WORK, not on the number of nodes: a layer of "
            "short lines and a layer whose entities carry a stored ring of "
            "hundreds of vertices differ by orders of magnitude at the same "
            "node count. Narrow `layers`, or name one layout with fewer paths."
        ),
    )

    for node, hits in zip(nodes, candidates):
        for i in hits:
            path = paths[i]
            if path.spans is None:
                node.possible.append(i)
                continue
            if path.distance_to(node.point[0], node.point[1]) <= tolerance:
                node.proven.append(i)
    return work


def _reachable(node: _Node) -> list[int]:
    """Every branch count this node could really have.

    Each untestable interior contributes exactly 0 or 2 branches, never 1, so
    the possible degrees are `degree_min`, `degree_min + 2`, ... up to
    `degree_max` — not the whole interval. That parity is what makes the bound
    useful rather than merely wide: a node reported as 1-to-3 is a dead end or a
    T-junction and can never be a simple connection.
    """
    return list(range(node.degree_min, node.degree_max + 1, 2))


def _histogram(nodes: Sequence[_Node]) -> dict[str, Any]:
    """The census over a list of nodes, decided and undecided kept apart.

    Two numbers per category, never one. `at_least` counts the nodes whose
    branch degree is KNOWN to be this; `at_most` adds the nodes that could be
    this and could also be something else. Publishing only the first would
    report a T-junction whose through-road is a curve as a dead end, which is
    the very mistake this recipe was written to stop.
    """
    at_least = {name: 0 for name in DEGREE_NAMES.values()}
    at_least["five_or_more"] = 0
    at_most = dict(at_least)
    five_plus: list[_Node] = []
    undecided: list[_Node] = []
    zero_degree = 0

    def bucket(degree: int) -> str | None:
        if degree >= 5:
            return "five_or_more"
        return DEGREE_NAMES.get(degree)

    for node in nodes:
        if node.decided:
            name = bucket(node.degree_min)
            if name is None:
                zero_degree += 1
                continue
            at_least[name] += 1
            at_most[name] += 1
            if name == "five_or_more":
                five_plus.append(node)
            continue
        undecided.append(node)
        for degree in _reachable(node):
            name = bucket(degree)
            if name is not None:
                at_most[name] += 1
    return {
        "at_least": at_least,
        "at_most": at_most,
        "five_plus": five_plus,
        "undecided": undecided,
        "zero_degree": zero_degree,
    }


# =============================================================================
# Reading the store
# =============================================================================


#: The fields one census needs, and no more. `ring` is asked for because a
#: closed polygon's interior IS its stored ring, and that is the one curved
#: shape the store describes in full.
_PROJECTION: Final[dict[str, int]] = {
    "handle": 1,
    "layer": 1,
    "type": 1,
    "bbox": 1,
    "length": 1,
    "ends": 1,
    "ends_status": 1,
    "ends_basis": 1,
    "ring": 1,
    "ring_status": 1,
    "path_points": 1,
    "path_bulges": 1,
    "_id": 0,
}


def _read_paths(
    drawing_id: str, layout: str | None, layers: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every row on these layers, plus a census of what will not take part.

    Nothing is dropped in silence. A row that cannot enter the graph is counted
    under the reason it cannot, and named where it can be named, so that the
    layer's totals still reconcile against `counts_by_layer` (G8).
    """
    rows = list(
        _entities()
        .find(
            {
                "drawing_id": drawing_id,
                "layout": layout,
                "layer": {"$in": list(layers)},
            },
            dict(_PROJECTION),
        )
        .sort([("layer", 1), ("handle", 1)])
    )

    by_status: dict[str, int] = {}
    named: dict[str, list[str]] = {}
    not_path = 0
    stale = 0
    for doc in rows:
        dxftype = str(doc.get("type") or "").upper()
        if dxftype in NOT_PATH_TYPES:
            not_path += 1
            continue
        status = doc.get("ends_status")
        if status is None:
            stale += 1
            key = "endpoints_not_stored"
        else:
            key = str(status)
        by_status[key] = by_status.get(key, 0) + 1
        if key not in (ENDS_OPEN,):
            named.setdefault(key, [])
            if len(named[key]) < 5:
                named[key].append(str(doc.get("handle")))

    census = {
        "rows_on_these_layers": len(rows),
        "not_path_geometry": not_path,
        "not_path_geometry_note": (
            "text, blocks, dimensions and hatches carry no ends as a CATEGORY. "
            "They are excluded from the graph and counted here, never handed to "
            "the node clustering: a label's box gluing two roads into one "
            "junction is a wrong answer nobody could see"
        ),
        "by_ends_status": dict(sorted(by_status.items())),
        "examples_by_ends_status": {k: v for k, v in sorted(named.items())},
        "endpoints_not_stored": stale,
        "endpoints_not_stored_note": (
            "a row with no `ends_status` at all was written by an ingest older "
            "than the one that stores path endpoints. It is counted, never "
            "guessed at from its bounding box"
        ),
    }
    return rows, census


def _stale_answer(
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    layers: Sequence[str],
    layers_source: str,
    census: Mapping[str, Any],
) -> dict[str, Any]:
    """The honest answer for a drawing ingested before endpoints were stored.

    "Re-ingest to answer" is the right output here and a number is not. A census
    built from bounding-box corners would look exactly like this one and be
    wrong by an unknown amount, which is worse than saying nothing — it is the
    same defect as reporting a road length that silently counts three copies.
    """
    scope_note = (
        f"layout {layout!r}; {census['rows_on_these_layers']} rows on "
        f"{len(layers)} layers from {layers_source}; NO census was computed, "
        "because not one row on these layers carries stored path endpoints"
    )
    evidence = _topology._measured_evidence(
        "the junction census cannot be computed for this drawing until it is "
        "re-ingested",
        drawing_id=drawing_id,
        scope=scope_for(layout=layout, units=units, note=scope_note),
        observations=[
            _topology._absence_observation(
                f"{census['endpoints_not_stored']} path rows on these layers "
                "carry no `ends` field; the drawing was ingested before path "
                "endpoints were stored",
                locator=",".join(layers[:3]) or None,
                layout=layout,
            )
        ],
        not_established=(
            "everything this recipe measures. Nothing here is a statement that "
            "the layer has no junctions, or that it has any; the geometry that "
            "would answer either way is not in the store for this drawing yet"
        ),
        how_to_verify=(
            "re-ingest the drawing and ask again. `INGEST_VERSION` was raised "
            "when the endpoint field was added, so a plain re-ingest run picks "
            "this drawing up on its own"
        ),
    )
    body = {
        "computed": False,
        "scope_note": scope_note,
        "why_not_computed": (
            "endpoints are not stored for this drawing. Every path row on these "
            "layers predates the ingest that writes them, so the two ends of a "
            "path are not in the store and the nodes cannot be built"
        ),
        "how_to_fix_it": (
            "re-ingest this drawing. `extract.INGEST_VERSION` was raised with "
            "the endpoint field, so `python -m app.ingest` re-reads any drawing "
            "stored below it without `--force`"
        ),
        "never_approximated": (
            "a census built from bounding-box corners is NOT offered as a "
            "fallback. A box has four corners and a path has two ends, the file "
            "does not record which diagonal is which, and a junction count built "
            "on that guess is wrong by an amount nothing in the response could "
            "show. Phase 1's chain count labels itself `approximate` for this "
            "exact reason"
        ),
        "layers": list(layers),
        "layers_source": layers_source,
        "path_census": dict(census),
        "junctions": None,
        "nodes_total": None,
    }
    return ev.attach(body, evidence)


# =============================================================================
# Copies — which estate is this answer about
# =============================================================================


def _copies(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The layer partitioned into spatially detached bodies, with the threshold.

    `dossier_anomaly`'s partition is used unchanged, threshold and all: it cuts
    at voids wider than the widest single feature on the layer, which is a
    measurement of the layer rather than a number chosen by a programmer, and it
    is the instrument that found the reference drawing's triplication in the
    first place. A second partition written here could put a junction in a
    different copy from the one the Dossier reports it in.
    """
    readout = anomaly._read_rows(list(rows))
    if not readout.placed:
        return [], {
            "partitioned": False,
            "why": (
                "not one row on these layers carries a usable bounding box, so "
                "the layer could not be partitioned into copies"
            ),
        }
    threshold, source = anomaly._derive_threshold(readout.placed)
    groups, widest_internal = anomaly._partition(readout.placed, threshold)
    summaries = [anomaly._summarise(group, i) for i, group in enumerate(groups)]
    for summary, group in zip(summaries, groups):
        summary["_handles"] = {p.handle for p in group if p.handle}
    detail = {
        "partitioned": True,
        "clusters_found": len(groups),
        "gap_threshold": threshold,
        "gap_basis": (
            "the layer was cut wherever the void between its entities is wider "
            "than the widest single feature ON THAT LAYER "
            f"({anomaly._fmt(threshold)} drawing units, set by handle "
            f"{getattr(source, 'handle', None)!r}). A void wider than the widest "
            "feature cannot lie inside a feature, so it separates one body of "
            "geometry from another. It fails in the SAFE direction: one "
            "oversized entity raises the threshold and merges bodies a smaller "
            "one would have split, which is a missed finding rather than an "
            "invented one"
        ),
        "largest_void_inside_a_cluster": widest_internal,
        "entities_unplaceable": readout.counts["unplaceable"],
    }
    return summaries, detail


def _anchor_counts(
    drawing_id: str,
    layout: str | None,
    layers: Sequence[str],
    summaries: Sequence[Mapping[str, Any]],
) -> list[int] | None:
    """For each copy, how much of the REST of the drawing sits inside its box.

    This is what identifies the live copy without naming a position, a layer or
    a coordinate (G1). Three copies of an estate are parked side by side; only
    one of them has the plots, the rights of way and the labels drawn on top of
    it, because only one of them is the drawing. The count is taken over the
    entities of OTHER layers in the same layout whose bounding box STARTS inside
    the copy's box — an indexed range query, so it is cheap.

    `None` means the store could not answer, which is reported rather than
    resolved by picking a copy anyway.
    """
    out: list[int] = []
    try:
        for summary in summaries:
            lo = summary["bbox_min"]
            hi = summary["bbox_max"]
            out.append(
                _entities().count_documents(
                    {
                        "drawing_id": drawing_id,
                        "bbox.min.0": {"$gte": float(lo[0]), "$lte": float(hi[0])},
                        "bbox.min.1": {"$gte": float(lo[1]), "$lte": float(hi[1])},
                        "layout": layout,
                        "layer": {"$nin": list(layers)},
                    }
                )
            )
    except Exception:  # noqa: BLE001 - an unavailable count must not be an outage
        return None
    return out


def _choose_copy(
    summaries: Sequence[Mapping[str, Any]],
    anchors: Sequence[int] | None,
    requested: int | None,
) -> tuple[int, str]:
    """Which copy the headline describes, and the rule that chose it."""
    if requested is not None:
        return requested, (
            f"the caller asked for copy {requested} by index; the indices are "
            "the order the copies appear in `copies`, left to right and then "
            "bottom to top"
        )
    if len(summaries) == 1:
        return 0, (
            "there is only one body of geometry on these layers, so the census "
            "describes the whole layer and no copy had to be chosen"
        )
    if anchors and max(anchors) > 0 and anchors.count(max(anchors)) == 1:
        winner = anchors.index(max(anchors))
        return winner, (
            f"copy {winner} is the LIVE one: {max(anchors)} entities on other "
            "layers of this layout have their bounding box inside it, against "
            + ", ".join(
                f"{n} for copy {i}" for i, n in enumerate(anchors) if i != winner
            )
            + ". The rest of the drawing is drawn on top of the live copy and "
            "beside the parked ones, and that is a measurement rather than a "
            "guess from where a copy sits"
        )
    return 0, (
        "no copy could be identified as the live one — nothing on any other "
        "layer of this layout sits inside exactly one of them — so copy 0 is "
        "described, and every copy's own census is listed beside it. Read "
        "`copies` before quoting the headline"
    )


# =============================================================================
# Assembling one census
# =============================================================================


def _tolerance_for(paths: Sequence[_Path], given: float | None) -> tuple[float, dict[str, Any]]:
    """The snap tolerance, derived from this drawing's own geometry (G1, G2).

    Not one number in drawing units is written in this file. The scale comes
    from the median length of the paths being counted, and the fraction applied
    to it is dimensionless, so the same rule behaves identically in metres, in
    inches, and in a drawing that declares no unit at all. A floor relative to
    the largest coordinate magnitude keeps a layer whose paths are all
    zero-length from snapping on exact equality alone.
    """
    lengths = sorted(
        p.length for p in paths if p.length is not None and p.length > 0.0
    )
    magnitude = 0.0
    for path in paths:
        if path.box is None:
            continue
        magnitude = max(magnitude, *(abs(v) for v in path.box))

    scale = network._median(lengths)
    scale_source = "the median non-zero path length on these layers"
    if scale is None:
        diagonals = sorted(
            math.hypot(p.box[2] - p.box[0], p.box[3] - p.box[1])
            for p in paths
            if p.box is not None
        )
        scale = network._median([d for d in diagonals if d > 0.0])
        scale_source = "the median non-zero bounding-box diagonal"
    from_scale = SNAP_FRACTION * scale if scale else 0.0
    from_noise = FLOAT_NOISE_FACTOR * magnitude
    derived = network._sig(max(from_scale, from_noise))

    if given is not None:
        value = float(given)
        basis = (
            "handed over by the caller in DRAWING units. The value this drawing "
            f"would have derived for itself is {network._fmt(derived)}"
        )
    else:
        value = derived
        which = "the scale term" if from_scale >= from_noise else "the float-noise floor"
        basis = (
            f"the larger of {SNAP_FRACTION:g} x {network._fmt(scale or 0.0)} "
            f"({network._fmt(from_scale)}, from {scale_source} over "
            f"{len(lengths)} paths) and {FLOAT_NOISE_FACTOR:g} x "
            f"{network._fmt(magnitude)} ({network._fmt(from_noise)}, a "
            f"float-noise floor from the largest coordinate magnitude); "
            f"{which} decided it. The fraction is dimensionless on purpose, so "
            "the tolerance follows the drawing instead of assuming metres — 11 "
            "of the drawings in this store are in inches and 3 declare no unit "
            "at all (G1, G2)"
        )
    return value, {
        "supplied_by_caller": given is not None,
        "derived_value": derived,
        "scale": scale,
        "scale_source": scale_source,
        "basis": basis,
    }


def _run_census(
    paths: Sequence[_Path], tolerance: float, budget: int
) -> dict[str, Any] | None:
    """Nodes, degrees and the histogram at ONE tolerance. `None` on a cap."""
    nodes = _cluster_endpoints(paths, tolerance, MAX_POINT_COMPARISONS)
    if nodes is None:
        return None
    _attach_interiors(nodes, paths, tolerance, budget)
    result = _histogram(nodes)
    result["nodes"] = nodes
    return result


def _node_row(node: _Node, paths: Sequence[_Path]) -> dict[str, Any]:
    return {
        "at": [round(node.point[0], 6), round(node.point[1], 6)],
        "paths_ending_here": sorted(
            paths[i].handle for i in range(len(paths)) if paths[i].index in node.ending
        )[:8],
        "ends": node.ends,
        "branches_proven": node.degree_min,
        "branches_at_most": node.degree_max,
        "passing_through_proven": sorted(paths[i].handle for i in node.proven),
        "passing_through_unknown": sorted(paths[i].handle for i in node.possible),
    }


def _summary_of(result: Mapping[str, Any]) -> dict[str, Any]:
    """One copy's census, in the shape two copies can be compared in."""
    out: dict[str, Any] = {}
    for name in list(DEGREE_NAMES.values()) + ["five_or_more"]:
        low = result["at_least"][name]
        high = result["at_most"][name]
        out[name] = {"at_least": low, "at_most": high, "exact": low if low == high else None}
    out["undecided"] = len(result["undecided"])
    return out


# =============================================================================
# The recipe
# =============================================================================

# =============================================================================
# Choosing the scope when nobody named one
# =============================================================================
#
# Measured on the reference drawing, 26 August 2026. Asked with no `layers`,
# the census ran over all thirteen layers the Dossier calls `network` -- road
# centrelines, but also parking markings, pedestrian crossings, hammerheads,
# four neighbourhood boundaries and a revision cloud -- and reported 792 nodes
# with dead ends anywhere between 14 and 409. Every number in it was true and
# none of it answered "how many junctions".
#
# Scoped to the layer that actually carries the network, the same recipe
# returns 561 nodes, 171 simple connections and 8 crossroads EXACTLY, with dead
# ends inside a four-node band. The instrument was never the problem.
#
# What separates the two is not the name and not the size: it is whether the
# paths on a layer JOIN EACH OTHER. A road centreline layer is a connected
# network, so most of its ends land where another of its own paths ends. A
# markings layer is a scatter of unconnected strokes, however many of them
# there are, and a boundary is a handful of long rings that touch nothing.
#
# So the layer is ranked by that, from the clustering already computed -- no
# second pass, no name test, no size test (G1).


def _self_connectedness(
    nodes: Sequence[_Node], paths: Sequence[_Path]
) -> list[dict[str, Any]]:
    """Per layer: how much of it joins itself, ranked.

    An end COUNTS as joined when another path of the SAME layer ends at the
    same node. Same-layer on purpose: a marking that happens to stop on a road
    is not thereby part of a network, and letting cross-layer contact count
    would rank the scatter as highly as the thing it is drawn over.
    """
    layer_of = {path.index: path.layer for path in paths}
    ends_on: dict[str, int] = {}
    joined_on: dict[str, int] = {}
    for path in paths:
        if path.ends is not None:
            ends_on[path.layer] = ends_on.get(path.layer, 0) + 2
    for node in nodes:
        per_layer: dict[str, int] = {}
        for index in node.ending:
            layer = layer_of.get(index)
            if layer is None:
                continue
            per_layer[layer] = per_layer.get(layer, 0) + 1
        for layer, count in per_layer.items():
            if count >= 2:
                joined_on[layer] = joined_on.get(layer, 0) + count
    rows = []
    for layer, ends in ends_on.items():
        joined = joined_on.get(layer, 0)
        rows.append(
            {
                "layer": layer,
                "ends": ends,
                "ends_joined_to_own_layer": joined,
                "share_joined": round(joined / ends, 4) if ends else 0.0,
                # Ends alone would crown any large scatter; share alone would
                # crown a three-path layer that happens to close. The product
                # is the one that survived both.
                "network_score": round((joined / ends) * joined, 4) if ends else 0.0,
            }
        )
    rows.sort(key=lambda row: (-row["network_score"], row["layer"]))
    return rows


#: The registered name, in one place, because the retry plan names it too and
#: a plan that names a recipe the registry does not know is dropped silently.
RECIPE_NAME: Final[str] = "junction_census"

#: How far ahead the top layer must be before the scope is chosen for the
#: caller.
#:
#: An absolute threshold was tried first and was the wrong instrument. On the
#: reference drawing the road centreline joins itself at only 18% of its ends
#: -- the clustering runs over every layer in scope, so a tolerance set by
#: thirteen layers splits nodes the road alone would merge -- and any absolute
#: bar high enough to exclude a scatter also excluded the network.
#:
#: What IS unambiguous is the gap. The centreline scores 83.1 and the next
#: layer 3.6, which is not a close call in any units. Three times is the
#: weakest statement that still means "one layer, not two": below it, two
#: layers are comparable, the question of which carries the network is a real
#: one, and this refuses rather than picks (G4).
SCORE_DOMINANCE: Final[float] = 3.0


def _why_this_layer(
    best: Mapping[str, Any], top: float, runner_up: float
) -> str:
    """The argument for the chosen layer, in the numbers that made it."""
    lead = (
        f"{top / runner_up:.1f} times the next layer's {runner_up}"
        if runner_up > 0
        else "and no other layer in scope joins itself at all"
    )
    return (
        f"{str(best['layer'])!r} joins itself more than any other layer in "
        f"scope: {best['ends_joined_to_own_layer']} of its {best['ends']} own "
        f"path ends land where another of its own paths ends "
        f"({float(best['share_joined']) * 100:.1f}%), scoring "
        f"{best['network_score']} — {lead}. Paths that join each other are a "
        "network; paths that do not are strokes drawn near one another. This "
        "says the layer BEHAVES like a network, not that it is a road"
    )


def _scope_safety(
    *,
    ranking: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    scope_was_given: bool,
    summary: Mapping[str, Any],
    layout: str | None,
) -> dict[str, Any]:
    """Whether this census may be quoted, and the call that would settle it.

    A census over a scope NOBODY CHOSE is not wrong; it is a census of a
    different thing. It becomes unusable only when two conditions hold at once:
    the scope was inferred rather than given, and the answer it produced is a
    band rather than a number. Either alone is fine -- a named scope is the
    caller's to defend, and a decided census over an inferred scope has already
    answered the question.

    When both hold, the ready-to-run call is published rather than described.
    `recipes/selfheal` executes it once and demotes the verdict, so the caller
    gets the exact figure AND the fact that the layer was chosen by machine.
    """
    decided = all(
        (summary.get(name) or {}).get("exact") is not None
        for name in ("dead_ends", "simple_connections", "t_junctions", "crossroads")
    )
    if scope_was_given:
        return {
            "usable_as_evidence": True,
            "verdict": "ESTABLISHED",
            "why": (
                f"the census ran over the {len(names)} layer"
                f"{'' if len(names) == 1 else 's'} the caller named"
            ),
            "do_not": (
                "do not read a category's `at_least` alone where `exact` is "
                "absent; quote both bounds"
            ),
        }
    if decided:
        return {
            "usable_as_evidence": True,
            "verdict": "ESTABLISHED",
            "why": (
                "every category came back decided, so the inferred scope did "
                "not cost this answer anything"
            ),
            "do_not": (
                "do not read this as a statement about paths on layers outside "
                "the scope in `layers`"
            ),
        }

    best = ranking[0] if ranking else None
    runner_up = float(ranking[1]["network_score"]) if len(ranking) > 1 else 0.0
    top = float(best["network_score"]) if best else 0.0
    connected = (
        best is not None
        and int(best.get("ends_joined_to_own_layer") or 0) > 0
        and (runner_up <= 0.0 or top >= runner_up * SCORE_DOMINANCE)
    )
    plan = (
        [
            {
                "recipe": RECIPE_NAME,
                "params": {"layers": [str(best["layer"])]},
                "because": _why_this_layer(best, top, runner_up),
            }
        ]
        if connected
        else []
    )
    return {
        "usable_as_evidence": False,
        "verdict": "NOTHING ESTABLISHED",
        "why": (
            f"no layer was named, so the census ran over every layer whose "
            f"geometric role is `network` on layout {layout!r} -- "
            f"{len(names)} of them -- and at least one category came back as a "
            "range rather than a number. Mixing a network with the strokes "
            "drawn over it produces counts that are all true and none of them "
            "an answer"
        ),
        "do_not": (
            "do not quote a figure from this. A band this wide is the scope "
            "talking, not the drawing"
        ),
        "how_to_get_an_answer": (
            "name `layers`. `layer_ranking` in this response ranks the layers "
            "in scope by how much each joins itself, which is what separates a "
            "network from a scatter"
        ),
        "retry_with": plan,
    }




def prepare_network(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    layers: Sequence[str] | None,
    snap_tolerance: float | None,
    copy_index: float | None,
) -> dict[str, Any]:
    """Everything both network recipes need before either can say anything.

    Choosing the road layers, reading their paths, splitting detached copies,
    picking the live one, deriving the snap tolerance and clustering the nodes
    is ONE sequence, and `junction_census` and `road_hierarchy` must not each
    have their own. Two copies would answer at two tolerances on two copies of
    the geometry and disagree about the same drawing, and nothing in either
    response would say which one to believe.

    Returns a dict carrying the whole prepared state. `stale_answer` is not
    None when the layers hold only rows from an ingest too old to store path
    endpoints; the caller returns it as its own answer, because there is
    nothing to compute and saying so is the answer.
    """
    names, layers_source, layer_notes = _topology._road_layers(
        drawing_id, layout, layers
    )

    if snap_tolerance is not None and float(snap_tolerance) < 0:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"snap_tolerance is {snap_tolerance!r}; a distance cannot be negative.",
            "It is a distance in DRAWING units: two path ends are the same "
            "junction when they are this far apart or closer. Zero means they "
            "must coincide exactly, which splits rather than merges and makes "
            "every branch count a floor.",
        )

    rows, path_census = _read_paths(drawing_id, layout, names)
    refuse_oversize(
        what="path entities entering the junction census",
        size=len(rows),
        limit=MAX_PATH_ENTITIES,
        hint=(
            "Narrow `layers` to the layers that really carry the network, or "
            "name one layout. This count is the rows on the layers as they "
            "stand, before anything was excluded."
        ),
    )

    graph_rows = [
        doc
        for doc in rows
        if str(doc.get("type") or "").upper() not in NOT_PATH_TYPES
    ]
    if graph_rows and path_census["endpoints_not_stored"] == len(graph_rows):
        return {
            "stale_answer": _stale_answer(
                drawing_id, layout, units, names, layers_source, path_census
            )
        }

    all_paths = [_Path(i, doc) for i, doc in enumerate(graph_rows)]

    summaries, partition = _copies(graph_rows)
    refuse_oversize(
        what="detached bodies of geometry found on these layers",
        size=len(summaries),
        limit=MAX_COPIES_REPORTED,
        hint=(
            "A layer holding more detached bodies than this is not a drawing "
            "with copies of one estate, it is a drawing made of tiles, and a "
            "per-copy census would describe neither. Narrow `layers`."
        ),
    )

    anchors = (
        _anchor_counts(drawing_id, layout, names, summaries)
        if len(summaries) > 1
        else None
    )

    requested = None if copy_index is None else int(copy_index)
    if requested is not None and not (0 <= requested < max(1, len(summaries))):
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"copy_index is {requested}; this layer holds {len(summaries)} "
            "detached bodies of geometry on this layout.",
            "Leave it empty and the live copy is identified by measurement — "
            "the one the rest of the drawing is drawn on top of — and named in "
            "the response. Indices run from 0.",
        )
    chosen, choice_basis = _choose_copy(summaries, anchors, requested)

    # Per-copy path lists. A copy is a set of handles, so the split is exact and
    # a path can never be counted into two copies.
    if summaries:
        by_copy: list[list[_Path]] = []
        for summary in summaries:
            handles = summary["_handles"]
            by_copy.append([p for p in all_paths if p.handle in handles])
        unplaced = [
            p
            for p in all_paths
            if not any(p.handle in s["_handles"] for s in summaries)
        ]
    else:
        by_copy = [list(all_paths)]
        unplaced = []
        chosen = 0

    described = by_copy[chosen] if by_copy else []
    tolerance, tolerance_detail = _tolerance_for(described or all_paths, snap_tolerance)

    result = _run_census(described, tolerance, MAX_INTERIOR_VERTEX_TESTS)
    if result is None:
        raise RecipeRefused(
            "RECIPE_INPUT_TOO_LARGE",
            f"the path ends on these layers are packed more densely than "
            f"{MAX_POINT_COMPARISONS} endpoint comparisons allow to scan at a "
            f"snap tolerance of {network._fmt(tolerance)}.",
            "Hand over a smaller `snap_tolerance`, or narrow `layers`. A count "
            "produced from a truncated scan would be a number about part of the "
            "layer wearing the name of the whole one.",
        )

    return {
        "stale_answer": None,
        "names": names,
        "layers_source": layers_source,
        "layer_notes": layer_notes,
        "rows": rows,
        "path_census": path_census,
        "graph_rows": graph_rows,
        "all_paths": all_paths,
        "summaries": summaries,
        "partition": partition,
        "anchors": anchors,
        "chosen": chosen,
        "choice_basis": choice_basis,
        "by_copy": by_copy,
        "unplaced": unplaced,
        "described": described,
        "tolerance": tolerance,
        "tolerance_detail": tolerance_detail,
        "result": result,
    }


def _run_junction_census(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    layers: Sequence[str] | None,
    snap_tolerance: float | None,
    copy_index: float | None,
) -> dict[str, Any]:
    prepared = prepare_network(
        drawing_id=drawing_id,
        layout=layout,
        units=units,
        layers=layers,
        snap_tolerance=snap_tolerance,
        copy_index=copy_index,
    )
    if prepared["stale_answer"] is not None:
        return prepared["stale_answer"]

    names = prepared["names"]
    layers_source = prepared["layers_source"]
    layer_notes = prepared["layer_notes"]
    rows = prepared["rows"]
    path_census = prepared["path_census"]
    graph_rows = prepared["graph_rows"]
    all_paths = prepared["all_paths"]
    summaries = prepared["summaries"]
    partition = prepared["partition"]
    anchors = prepared["anchors"]
    chosen = prepared["chosen"]
    choice_basis = prepared["choice_basis"]
    by_copy = prepared["by_copy"]
    unplaced = prepared["unplaced"]
    described = prepared["described"]
    tolerance = prepared["tolerance"]
    tolerance_detail = prepared["tolerance_detail"]
    result = prepared["result"]

    # Every copy gets its own census, at the SAME tolerance, so that "the copies
    # agree" is a checked statement rather than an assumption about copies.
    per_copy: list[dict[str, Any]] = []
    for i, summary in enumerate(summaries):
        if i == chosen:
            copy_result = result
        else:
            copy_result = _run_census(by_copy[i], tolerance, MAX_INTERIOR_VERTEX_TESTS)
        per_copy.append(
            {
                "copy": i,
                "entities": summary["entities"],
                "bbox_min": summary["bbox_min"],
                "bbox_width": summary["bbox_width"],
                "bbox_height": summary["bbox_height"],
                "length_total": summary["length_total"],
                "other_layer_entities_inside_it": (
                    anchors[i] if anchors is not None else None
                ),
                "is_the_copy_described": i == chosen,
                "census": None if copy_result is None else _summary_of(copy_result),
                "nodes": None if copy_result is None else len(copy_result["nodes"]),
            }
        )

    censuses = [c["census"] for c in per_copy if c["census"] is not None]
    copies_agree = len(censuses) > 1 and all(c == censuses[0] for c in censuses[1:])

    # ---- sensitivity ------------------------------------------------------
    sensitivity: list[dict[str, Any]] = []
    sensitivity_note = None
    if len(described) > MAX_PATHS_FOR_SENSITIVITY:
        sensitivity_note = (
            f"{len(described)} paths is above the {MAX_PATHS_FOR_SENSITIVITY} "
            "at which the sensitivity table is re-run, so it was NOT computed. "
            "That is an omission, not a finding of stability"
        )
    else:
        for factor in sorted(SENSITIVITY_FACTORS + (1.0,)):
            alt = (
                result
                if factor == 1.0
                else _run_census(
                    described, tolerance * factor, MAX_INTERIOR_VERTEX_TESTS
                )
            )
            sensitivity.append(
                {
                    "factor": factor,
                    "snap_tolerance": network._sig(tolerance * factor),
                    "nodes": None if alt is None else len(alt["nodes"]),
                    "census": None if alt is None else _summary_of(alt),
                }
            )

    def _t_at(row: Mapping[str, Any]) -> tuple[int, int] | None:
        census = row.get("census")
        if not census:
            return None
        block = census["t_junctions"]
        return block["at_least"], block["at_most"]

    moved = {_t_at(row) for row in sensitivity if _t_at(row) is not None}

    summary = _summary_of(result)
    nodes = result["nodes"]
    undecided = result["undecided"]
    interiors_known = sum(1 for p in described if p.spans is not None)
    with_vertices = sum(1 for p in described if p.shape_basis.startswith("the open"))
    with_ends = sum(1 for p in described if p.ends is not None)
    closed_rings = sum(1 for p in described if p.status == ENDS_CLOSED)
    unknown_ends = sum(1 for p in described if p.status == ENDS_NOT_DERIVED)
    no_geometry = [p.handle for p in described if p.box is None]

    scope_note = (
        f"layout {layout!r}; {len(described)} path entities of the "
        f"{len(all_paths)} on {len(names)} layers from {layers_source}"
        + (
            f", being copy {chosen} of {len(summaries)} detached copies of the "
            "same geometry on those layers"
            if len(summaries) > 1
            else ""
        )
        + f"; two path ends are one junction within {network._fmt(tolerance)} "
        + (units.get("length_unit") or "drawing units")
        + f"; {len(nodes)} junctions found, of which {len(undecided)} carry a "
        "branch-count RANGE rather than a category"
    )
    scope = scope_for(layout=layout, units=units, note=scope_note)

    def _phrase(name: str) -> str:
        row = summary[name]
        if row["exact"] is not None:
            return f"{row['exact']} {name.replace('_', ' ')}"
        return f"{row['at_least']} to {row['at_most']} {name.replace('_', ' ')}"

    observations: list[ev.Observation] = [
        _topology._geometry_observation(
            f"{len(nodes)} places where path ends meet; "
            + ", ".join(
                _phrase(n)
                for n in (
                    "dead_ends",
                    "simple_connections",
                    "t_junctions",
                    "crossroads",
                    "five_or_more",
                )
            ),
            locator=",".join(names[:3]) or None,
            layout=layout,
        )
    ]
    if len(summaries) > 1:
        observations.append(
            _topology._geometry_observation(
                f"these layers hold {len(summaries)} spatially detached copies "
                f"of the same geometry; this census describes copy {chosen} "
                "alone, and every copy's own census is listed beside it",
                locator=",".join(names[:3]) or None,
                layout=layout,
            )
        )
    if undecided:
        observations.append(
            _topology._absence_observation(
                f"{len(undecided)} junctions could not be given a category: a "
                "curve passes close enough that its bounding box reaches them, "
                "and two stored endpoints do not say where the middle of a curve "
                "went. Each is published with its range",
                locator=str(_node_row(undecided[0], described)["at"]),
                layout=layout,
            )
        )
    if unknown_ends or no_geometry:
        observations.append(
            _topology._absence_observation(
                f"{unknown_ends} path entities in this copy carry no derivable "
                f"ends and {len(no_geometry)} carry no bounding box; they are "
                "counted and named, never treated as paths that meet nothing",
                locator=",".join(names[:3]) or None,
                layout=layout,
            )
        )

    evidence = _topology._measured_evidence(
        ", ".join(
            _phrase(n)
            for n in ("t_junctions", "crossroads", "dead_ends", "simple_connections")
        )
        + f" were counted on {len(described)} paths in layout {layout!r}",
        drawing_id=drawing_id,
        scope=scope,
        observations=observations,
        not_established=(
            "that a junction is a JUNCTION on the ground. This counts branches "
            "of drawn geometry: three centrelines meeting at a point is a "
            "T-junction in the drawing, and whether the ground carries a "
            "give-way, a roundabout, or nothing at all is not in this file. Nor "
            "is a dead end a cul-de-sac — a path can end because the estate "
            "ends, because the sheet ends, or because the drafter split it "
            "there. And where an interior could not be tested the degree is a "
            "RANGE: those nodes are in `undecided`, not folded into the nearest "
            "category. Nothing here reads Z either: two paths that cross at "
            "different levels meet in plan and not on the ground, and this "
            "census counts the plan"
        ),
        how_to_verify=(
            "read `tolerance_sensitivity`. A count that holds across the whole "
            "band is a count about the drawing; one that moves is a count about "
            "the tolerance. Then open a handful of the handles in `junctions` "
            "and look at them — every node names the paths that end at it"
        ),
    )

    ranking = _self_connectedness(nodes, all_paths)
    safety = _scope_safety(
        ranking=ranking,
        names=names,
        scope_was_given=layers is not None,
        summary=summary,
        layout=layout,
    )

    body: dict[str, Any] = {
        "computed": True,
        "scope_note": scope_note,
        "layers": list(names),
        "layers_source": layers_source,
        "layers_notes": layer_notes,
        "layers_never_by_name": (
            "path layers arrive from the caller, from this drawing's land use "
            "config (role=network), or from the Dossier's GEOMETRIC role — never "
            "from a layer name. Civil3D template layers named after roads are "
            "sheet layout, and a name test would count the junctions of a title "
            "block"
        ),
        "method": (
            "branch degree at each node. A node is a cluster of stored path ENDS "
            "within the snap tolerance; its degree is (the number of path ends "
            "there) + 2 x (the number of paths whose INTERIOR passes through "
            "it). The doubling is the instrument: a road that runs THROUGH a "
            "junction contributes two branches, and counting paths instead of "
            "branches reports a T-junction as a dead end and a crossroads as a "
            "simple connection"
        ),
        "prefilter": (
            "the bounding box of every path, grown by the snap tolerance, "
            "through `store_spatial._SpatialIndex` — the same uniform grid "
            "`join_labels` uses, unchanged. A node outside a grown box cannot "
            "touch that entity, whatever its shape"
        ),
        "junctions": {
            "dead_ends": summary["dead_ends"],
            "simple_connections": summary["simple_connections"],
            "t_junctions": summary["t_junctions"],
            "crossroads": summary["crossroads"],
            "every_count_is_a_range": (
                "each category carries `at_least`, `at_most` and `exact`. "
                "`at_least` is the nodes whose branch degree is KNOWN to be "
                "this; `at_most` adds the nodes that could be this and could "
                "also be something else, because a shape the store describes "
                "only by its bounding box comes close enough to reach them. "
                "`exact` is filled in only when the two agree — which is the "
                "whole answer wherever the geometry is stored in full. Quote "
                "`exact` where it is there, and BOTH bounds where it is not; "
                "quoting `at_least` alone would report a T-junction whose "
                "through-road is an arc as a dead end"
            ),
            "five_or_more": {
                "count": len(result["five_plus"]),
                "at_most": summary["five_or_more"]["at_most"],
                "note": (
                    "five or more branches at one point is unusual and worth "
                    "opening: it is either a genuine multi-way junction, or two "
                    "junctions that the snap tolerance has merged into one. The "
                    "handles are listed so it can be looked at rather than "
                    "argued about"
                ),
                "rows": [
                    _node_row(n, described)
                    for n in sorted(
                        result["five_plus"], key=lambda n: -n.degree_min
                    )[:MAX_NODES_LISTED]
                ],
                "cap": MAX_NODES_LISTED,
                "truncated": len(result["five_plus"]) > MAX_NODES_LISTED,
            },
            "nodes_with_no_branches": result["zero_degree"],
            "nodes_with_no_branches_note": (
                "a node with zero branches cannot happen -- every node is made "
                "of at least one path end -- so this is a guard rather than a "
                "finding. It is published because a count that is dropped when "
                "it is zero is a count nobody notices when it is not"
            ),
            "what_each_one_means": {
                "dead_ends": "one branch — a cul-de-sac head, or a stub at the boundary",
                "simple_connections": (
                    "two branches — a path continuing through. Often just a "
                    "drafting split rather than anything on the ground"
                ),
                "t_junctions": "three branches",
                "crossroads": "four branches",
            },
        },
        "undecided": {
            "count": len(undecided),
            "why": (
                "a curve comes close enough that its bounding box reaches this "
                "node, and two stored endpoints do not describe a curve. The "
                "node's branch count is therefore a RANGE. It is published as "
                "one rather than resolved: counting the low end would report a "
                "T-junction as a dead end, and counting the high end would "
                "invent junctions"
            ),
            "rows": [
                _node_row(n, described)
                for n in sorted(undecided, key=lambda n: -n.degree_max)[
                    :MAX_NODES_LISTED
                ]
            ],
            "cap": MAX_NODES_LISTED,
            "truncated": len(undecided) > MAX_NODES_LISTED,
            "how_to_close_it": (
                "store an ARC's centre, radius and angles at ingest, the way its "
                "two ends and an open polyline's vertices already are. Endpoints "
                "made the NODES exact and vertices made the polylines exact; the "
                "arc entity is the last shape the store describes only by its "
                "box, and it is what every remaining row here is waiting on"
            ),
        },
        "nodes_total": len(nodes),
        "layer_ranking": ranking,
        "layer_ranking_note": (
            "how much of each layer joins ITSELF: `share_joined` is the "
            "fraction of that layer's own path ends that land where another "
            "path of the same layer ends, and `network_score` multiplies it by "
            "the number of joined ends so that neither a large scatter nor a "
            "tiny closed figure can win on one of the two alone. Nothing here "
            "reads a layer name"
        ),
        "conclusion_safety": safety,
        "paths_in_this_copy": len(described),
        "path_geometry": {
            "with_two_stored_ends": with_ends,
            "with_their_vertices_stored": with_vertices,
            "closed_rings_no_free_ends": closed_rings,
            "ends_not_derivable": unknown_ends,
            "interiors_known": interiors_known,
            "interiors_unknown": len(described) - interiors_known,
            "interiors_note": (
                "an interior is KNOWN for a closed polygon with a complete "
                "stored ring; for an open polyline with its own vertices "
                "stored, whose arc spans are measured AS ARCS and not as their "
                "chords; for a LINE (the segment between its two ends is the "
                "whole line); and for any path whose stored length equals the "
                "distance between its ends, since only a straight path can be "
                "as short as its own chord. It is UNKNOWN for an ARC entity, a "
                "circle, a polyline whose ring could not be completed, a "
                "polyline over the vertex limit, and a spline — and that is "
                "what `undecided` counts"
            ),
            "without_any_geometry": len(no_geometry),
            "without_any_geometry_examples": sorted(no_geometry)[:5],
            "without_any_geometry_note": (
                "a path with neither ends nor a bounding box cannot be placed at "
                "all. It is counted and named here, and NO node's range accounts "
                "for it: where this number is not zero, every branch count above "
                "is a bound over the paths that could be located"
            ),
        },
        "path_census": dict(path_census),
        "tolerance": {
            "value": _topology._length_quantity(
                tolerance,
                units,
                basis="the distance within which two path ends are one junction",
                method=tolerance_detail["basis"],
            ),
            "supplied_by_caller": tolerance_detail["supplied_by_caller"],
            "derived_value_for_this_drawing": tolerance_detail["derived_value"],
            "scale_source": tolerance_detail["scale_source"],
            "snap_fraction": SNAP_FRACTION,
            "shared_with": (
                "`dossier_network`, whose chain count runs on the same rule and "
                "the same constants — imported here rather than copied, so that "
                "the two lanes can never disagree about whether two ends are the "
                "same point"
            ),
            "decides": (
                "this number decides the answer. Raise it and two dead ends "
                "become one junction; lower it and a junction splits into dead "
                "ends. `tolerance_sensitivity` shows how hard it grips on THIS "
                "drawing"
            ),
        },
        "tolerance_sensitivity": {
            "rows": sensitivity,
            "factors": list(sorted(SENSITIVITY_FACTORS + (1.0,))),
            "skipped": sensitivity_note,
            "verdict": (
                None
                if sensitivity_note
                else (
                    "the T-junction count is the same at every tolerance in the "
                    "band, so it is a count about the drawing rather than about "
                    "the tolerance"
                    if len(moved) <= 1
                    else "the T-junction count MOVES across the band ("
                    + ", ".join(
                        f"{_t_at(r)[0]}-{_t_at(r)[1]} at x{r['factor']:g}"
                        for r in sensitivity
                        if _t_at(r) is not None
                    )
                    + "), so the tolerance is doing part of the work and the "
                    "headline must not be quoted without it"
                )
            ),
        },
        "copies": {
            "count": len(summaries),
            "described": chosen,
            "chosen_by": choice_basis,
            "they_agree": copies_agree if len(summaries) > 1 else None,
            "they_agree_note": (
                "every copy was counted separately at the same tolerance. Copies "
                "that agree corroborate each other; copies that do NOT agree are "
                "not copies, and the headline then describes one body of "
                "geometry rather than a repeated estate"
            ),
            "rows": per_copy[:MAX_COPIES_REPORTED],
            "paths_in_no_copy": len(unplaced),
            "partition": partition,
            "why_this_matters": (
                "a layer that holds the same estate drawn three times triples "
                "every junction count taken over the whole layer. An answer that "
                "does that silently is the exact failure this campaign exists to "
                "end, so the census describes ONE copy and says which"
            ),
        },
        "interior_work_note": (
            "the interior tests were counted before any distance was taken, so "
            f"the budget of {MAX_INTERIOR_VERTEX_TESTS} vertex tests is a limit "
            "on work rather than a report written after the work was done"
        ),
        "limits_applied": {
            "path_entities": MAX_PATH_ENTITIES,
            "endpoint_comparisons": MAX_POINT_COMPARISONS,
            "interior_vertex_tests": MAX_INTERIOR_VERTEX_TESTS,
            "copies_reported": MAX_COPIES_REPORTED,
            "nodes_listed": MAX_NODES_LISTED,
            "paths_for_sensitivity": MAX_PATHS_FOR_SENSITIVITY,
        },
    }
    return ev.attach(body, evidence)


register(
    Recipe(
        name=RECIPE_NAME,
        answers=(
            "how the paths on a layer meet — dead ends, simple connections, "
            "T-junctions and crossroads, counted by BRANCH degree at each node"
        ),
        when_to_use=(
            "'how many T-junctions are there', 'how many crossroads', 'how many "
            "cul-de-sacs', 'is this road network broken into pieces', or before "
            "a layout is signed off. It answers from the stored path ENDPOINTS, "
            "so a drawing ingested before those were stored is told to "
            "re-ingest rather than given a number built from bounding boxes."
        ),
        params=(
            Param(
                "layers",
                "layers",
                "the layers whose paths are counted. Left empty they are derived "
                "from this drawing's land use config (role=network) and from the "
                "Dossier's GEOMETRIC role `network`, and the response says which "
                "route answered. They are never derived from a layer NAME: "
                "template layers named after roads are sheet layout.",
            ),
            Param(
                "snap_tolerance",
                "number",
                "the distance within which two path ends count as the same "
                "junction, in DRAWING units. Left empty it is derived from this "
                "drawing's own geometry by the same rule the chain count uses, "
                "and the response publishes the value it took plus a "
                "sensitivity table across a 16x band. Give it explicitly only "
                "when you know the drafting slop of the file.",
                default=None,
            ),
            Param(
                "copy_index",
                "number",
                "which detached copy of the geometry to describe, when the "
                "layers hold the same thing drawn more than once. Left empty, "
                "the LIVE copy is identified by measurement — the one the rest "
                "of the drawing is drawn on top of — and named in the response. "
                "Every copy's own census is listed either way.",
                default=None,
            ),
        ),
        returns=(
            "junctions.dead_ends.at_least",
            "junctions.dead_ends.at_most",
            "junctions.dead_ends.exact",
            "junctions.simple_connections.exact",
            "junctions.t_junctions.at_least",
            "junctions.t_junctions.at_most",
            "junctions.t_junctions.exact",
            "junctions.crossroads.at_least",
            "junctions.crossroads.at_most",
            "junctions.crossroads.exact",
            "junctions.five_or_more.count",
            "undecided.count",
            "undecided.rows[].branches_proven",
            "undecided.rows[].branches_at_most",
            "nodes_total",
            "paths_in_this_copy",
            "tolerance.value",
            "tolerance_sensitivity.rows",
            "copies.count",
            "copies.described",
            "copies.chosen_by",
            "path_geometry.interiors_unknown",
            "path_census.endpoints_not_stored",
        ),
        built_on=(
            "extract._ends_fields",
            "extract._path_point_fields",
            "dossier_network._median",
            "dossier_anomaly._partition",
            "store_spatial._SpatialIndex",
            "region._point_segment_distance",
            "recipes.topology._road_layers",
        ),
        limits={
            "path_entities": MAX_PATH_ENTITIES,
            "endpoint_comparisons": MAX_POINT_COMPARISONS,
            "interior_vertex_tests": MAX_INTERIOR_VERTEX_TESTS,
            "copies_reported": MAX_COPIES_REPORTED,
            "nodes_listed": MAX_NODES_LISTED,
        },
        run=_run_junction_census,
    )
)
