"""Ring geometry: centroid, edge lengths, and shape fingerprint.

One rule determines almost the whole of this file:

    Every second-order quantity is computed after the ring has been
    translated to the lower-left corner of its bounding box, and the
    result is translated back afterwards.

The reason is measured, not theoretical. Shoelace multiplies coordinates
before it subtracts them. In projected coordinates -- this project's
reference drawing sits in UTM around E 691,942 / N 2,750,756 -- each
product lands around 1.9e12, and the float64 ulp there is about 4.7e-4.
What is left after the subtraction is not the polygon's area; it is the
polygon's area plus noise, and that noise is multiplied by the coordinate
magnitude again when the centroid is computed.

Measured on rectangles rotated 49 degrees at those coordinates:

    plot size       absolute frame    after translation
    12 x 25         0.770 m           0.000000 m
    10 x 25         0.462 m           0.000000 m
     5 x  5         9.235 m           0.000000 m
    76 x 76         0.040 m           0.000000 m

The direction of the error runs against expectation: it is INVERSELY
proportional to parcel size, because the area error is set by the coordinate
magnitude while the divisor is that area itself. A small parcel far from the
origin is the worst case, and small parcels are precisely the ones people ask
about most often.

This translation is also the only reason edge detection can work at all: in
absolute coordinates, a polygon does not recognise its own vertices as lying
on its edge. See `region.point_in_ring`.

There is no drawing-specific constant in this file (G1): `RING_ORIGIN` is
derived per ring from its own data, and the only numbers that can be tuned
(`MAX_RING_VERTICES`, `SHAPE_KEY_QUANTUM`) are stated resource limits, not a
property of any one drawing.

The decision is recorded in `docs/DECISIONS-LOG.md` D-083.
"""

from __future__ import annotations

import hashlib
import math
from typing import Final, Iterable, Sequence

Point = tuple[float, float]

#: Vertex limit for a ring stored whole. Raised from 512 because 512 had no
#: measurement behind it and created an exception to the invariant for exactly
#: one polygon. The arithmetic: 4,096 vertices x 30 B BSON = 123 KB per
#: document in the worst case, far below Mongo's 16 MB limit.
MAX_RING_VERTICES = 4096

#: Shape-fingerprint quantum, in DRAWING UNITS -- not metres. A drawing in
#: inches uses the same number and therefore a finer quantum; that makes
#: `find_duplicates` stricter there, not looser, so it fails in the safe
#: direction. The value is echoed in `shape_key_basis`.
SHAPE_KEY_QUANTUM = 1e-3

#: Turn-angle quantum, radians.
TURN_QUANTUM = 1e-3

#: Relative tolerance for declaring that a point lies on a ring's edge.
#: Relative to the ring's diagonal, so it carries no unit at all and behaves
#: identically in metres, in inches, and in drawings that state no unit.
BOUNDARY_EPS_REL = 1e-9


def dedupe(points: Sequence[Point]) -> tuple[list[Point], int]:
    """Drop consecutive identical vertices, including a repeating closer.

    Converted files produce these routinely. A zero-length edge makes the turn
    angle undefined and contaminates the shape fingerprint, so two identical
    polygons can get different keys purely because one of them carries a
    duplicated vertex.
    """
    out: list[Point] = []
    dropped = 0
    for p in points:
        if out and _same(out[-1], p):
            dropped += 1
            continue
        out.append((float(p[0]), float(p[1])))
    while len(out) > 1 and _same(out[0], out[-1]):
        out.pop()
        dropped += 1
    return out, dropped


def _same(a: Point, b: Point) -> bool:
    return a[0] == b[0] and a[1] == b[1]


def ring_origin(points: Sequence[Point]) -> Point:
    """The translation point: the lower-left corner of the ring's bounding box."""
    return (min(p[0] for p in points), min(p[1] for p in points))


def _local(points: Sequence[Point], origin: Point) -> list[Point]:
    ox, oy = origin
    return [(p[0] - ox, p[1] - oy) for p in points]


def signed_area(points: Sequence[Point], origin: Point | None = None) -> float:
    """Signed area, computed in the local frame.

    The sign states the winding direction: positive is counter-clockwise.
    """
    if len(points) < 3:
        return 0.0
    o = origin if origin is not None else ring_origin(points)
    pts = _local(points, o)
    total = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def polygon_centroid(
    points: Sequence[Point], origin: Point | None = None
) -> Point | None:
    """Area centroid of the polygon, or None when the area is zero.

    `None` and not an exception: a polyline that doubles back on itself is a
    normal state in a drawing recovered from hundreds of audit errors, and zero
    area means there is no centroid to report -- not that the centroid is at
    zero.
    """
    if len(points) < 3:
        return None
    o = origin if origin is not None else ring_origin(points)
    pts = _local(points, o)
    a = 0.0
    cx = 0.0
    cy = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if a == 0.0:
        return None
    a *= 0.5
    return (cx / (6.0 * a) + o[0], cy / (6.0 * a) + o[1])


def edge_lengths(points: Sequence[Point], closed: bool) -> list[float]:
    """Length of every edge. Needs no translation: distance already subtracts."""
    n = len(points)
    if n < 2:
        return []
    idx = range(n) if closed else range(n - 1)
    return [math.dist(points[i], points[(i + 1) % n]) for i in idx]


def perimeter(points: Sequence[Point], closed: bool) -> float:
    return sum(edge_lengths(points, closed))


def turn_angles(points: Sequence[Point]) -> list[float]:
    """Turn angle at every vertex, radians, in [-pi, pi].

    Edge lengths alone are not a shape fingerprint: a 12x25 rectangle and a
    12x25 parallelogram have identical edge sequences. It is the turn angles
    that separate the two, and in a rotated grid meeting a curved road the
    trapezium at the grid's edge is an ordinary occurrence, not an edge case.
    """
    n = len(points)
    if n < 3:
        return []
    out: list[float] = []
    for i in range(n):
        prev = points[(i - 1) % n]
        cur = points[i]
        nxt = points[(i + 1) % n]
        a = math.atan2(cur[1] - prev[1], cur[0] - prev[0])
        b = math.atan2(nxt[1] - cur[1], nxt[0] - cur[0])
        out.append((b - a + math.pi) % (2 * math.pi) - math.pi)
    return out


def _least_rotation(seq: Sequence[tuple[int, int]]) -> int:
    """Index of the lexicographically least rotation, Booth's algorithm, O(n).

    Not "start from the longest edge": that is a heuristic which is undefined
    when several edges are the same length -- which in a rectangle means
    always -- and whose way out falls back to a naive O(n^2) comparison.
    """
    if not seq:
        return 0
    s = list(seq) + list(seq)
    n = len(s)
    f = [-1] * n
    k = 0
    for j in range(1, n):
        sj = s[j]
        i = f[j - k - 1]
        while i != -1 and sj != s[k + i + 1]:
            if sj < s[k + i + 1]:
                k = j - i - 1
            i = f[i]
        if sj != s[k + i + 1]:
            if sj < s[k]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return k % len(seq)


def shape_key(
    points: Sequence[Point],
    *,
    quantum: float = SHAPE_KEY_QUANTUM,
    turn_quantum: float = TURN_QUANTUM,
) -> str | None:
    """Shape fingerprint invariant to translation, rotation, and winding.

    Sensitive to scale and to the mirroring of chiral shapes -- both are
    deliberate. Winding direction is normalised because it is an artefact of
    how the file was written: copy and mirror operations in AutoCAD reverse it,
    so two genuinely identical plots would get different keys and duplicate
    detection would report zero when there are dozens.

    MEANINGFUL ONLY WITHIN A SINGLE DRAWING. It is scale-sensitive but
    unit-blind: a 12x25 metre plot and a 12x25 inch component produce exactly
    the same key. Every use of it must filter on `drawing_id` first.
    """
    pts, _ = dedupe(points)
    if len(pts) < 3:
        return None
    if signed_area(pts) < 0:
        pts = list(reversed(pts))
    lens = edge_lengths(pts, closed=True)
    turns = turn_angles(pts)
    if not lens or len(lens) != len(turns):
        return None
    seq = [
        (int(round(l / quantum)), int(round(t / turn_quantum)))
        for l, t in zip(lens, turns)
    ]
    k = _least_rotation(seq)
    canonical = seq[k:] + seq[:k]
    blob = "|".join(f"{a},{b}" for a, b in canonical)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def shape_key_basis(quantum: float = SHAPE_KEY_QUANTUM) -> str:
    """The sentence that accompanies every `shape_key` inside a response."""
    return (
        f"edge lengths quantised to {quantum:g} drawing units and turn angles "
        f"to {TURN_QUANTUM:g} radians, winding direction normalised, canonical "
        "rotation by Booth. Meaningful only within a single drawing: "
        "scale-sensitive, unit-blind."
    )


def diagonal(points: Sequence[Point]) -> float:
    """Diagonal of the ring's bounding box. Used as the scale for tolerances."""
    if not points:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def is_simple(points: Sequence[Point]) -> bool | None:
    """Whether the ring does not intersect itself.

    Returns `None` above 64 vertices: the test is O(n^2), and an answer that
    sometimes cannot be computed is better declared unknown than reported as a
    fact that happens not to have been checked.
    """
    n = len(points)
    if n < 4:
        return True
    if n > 64:
        return None
    for i in range(n):
        a1 = points[i]
        a2 = points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            b1 = points[j]
            b2 = points[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return False
    return True


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    d1 = _orient(b1, b2, a1)
    d2 = _orient(b1, b2, a2)
    d3 = _orient(a1, a2, b1)
    d4 = _orient(a1, a2, b2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def truncate(points: Sequence[Point], limit: int = MAX_RING_VERTICES) -> list[Point]:
    """For diagnostics only. Never used to measure anything."""
    return list(points[:limit])


def summarise(points: Iterable[Point]) -> str:
    pts = list(points)
    return f"{len(pts)} vertex"


#: How far the drawn shape may sit from the true curve, in DRAWING UNITS.
#:
#: Drawing units, not metres, for the same reason `SHAPE_KEY_QUANTUM` is: a
#: drawing in inches gets a finer tolerance and therefore MORE points, which
#: fails in the safe direction. 10 mm on a metric site plan is well under the
#: width of the lines it is drawn with, and far under anything a reviewer
#: measures.
ARC_CHORD_TOLERANCE = 0.01

#: Never subdivide one span into more than this. A tolerance approaching zero
#: -- a huge radius, or a unit this value is wrong for -- would otherwise ask
#: for millions of points for a single edge.
MAX_ARC_SEGMENTS = 256


def arc_segments(radius: float, sweep: float, tolerance: float) -> int:
    """How many straight spans approximate an arc to within `tolerance`.

    The sagitta of a sub-span of angle d on a circle of radius r is
    r*(1 - cos(d/2)). Solving that for d and dividing the sweep by it gives
    the count. Clamped at both ends: at least one span, never more than
    `MAX_ARC_SEGMENTS`.
    """
    sweep = abs(float(sweep))
    if radius <= 0 or sweep <= 0:
        return 1
    if tolerance <= 0 or tolerance >= 2 * radius:
        # A tolerance as large as the circle says "do not subdivide", and
        # `acos` below would be handed a value outside [-1, 1].
        return 1
    step = 2.0 * math.acos(1.0 - tolerance / radius)
    if step <= 0:
        return MAX_ARC_SEGMENTS
    return max(1, min(MAX_ARC_SEGMENTS, int(math.ceil(sweep / step))))


def arc_points(
    centre: Point,
    radius: float,
    start_angle: float,
    end_angle: float,
    tolerance: float = ARC_CHORD_TOLERANCE,
) -> list[Point]:
    """An ARC as real points, endpoints included, angles in RADIANS.

    The endpoints are computed from the angles like every other point rather
    than carried through separately, so the curve cannot disagree with its own
    ends by a rounding step.
    """
    cx, cy = float(centre[0]), float(centre[1])
    sweep = float(end_angle) - float(start_angle)
    n = arc_segments(float(radius), sweep, tolerance)
    return [
        (
            cx + radius * math.cos(start_angle + sweep * i / n),
            cy + radius * math.sin(start_angle + sweep * i / n),
        )
        for i in range(n + 1)
    ]


def flatten_bulges(
    points: Sequence[Point],
    bulges: Sequence[float],
    closed: bool,
    tolerance: float = ARC_CHORD_TOLERANCE,
) -> list[Point]:
    """Replace every bulged span with the arc it actually describes.

    A bulge is `tan(theta/4)` for the included angle `theta` of the span it
    starts. The two vertices either side of it are the CHORD of that arc, and
    a ring made of those chords is not the boundary -- which is exactly why
    such rings were refused rather than stored. Flattening removes the reason
    for the refusal instead of relaxing it.

    `bulges[i]` belongs to the span from `points[i]` to `points[i+1]`; on a
    closed shape the last span wraps to `points[0]`.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        return pts

    spans = len(pts) if closed else len(pts) - 1
    out: list[Point] = []
    for i in range(spans):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        out.append(a)
        bulge = float(bulges[i]) if i < len(bulges) else 0.0
        if abs(bulge) <= 1e-9:
            continue
        theta = 4.0 * math.atan(bulge)
        chord = math.hypot(b[0] - a[0], b[1] - a[1])
        half = math.sin(theta / 2.0)
        if chord <= 0 or abs(half) < 1e-12:
            continue
        radius = chord / (2.0 * half)
        # Centre: the chord's midpoint, pushed along the chord's normal by the
        # apothem. The SIGN of `radius` carries the bulge's direction, so no
        # separate left/right test is needed and none can disagree with it.
        mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
        apothem = radius * math.cos(theta / 2.0)
        ux, uy = (b[0] - a[0]) / chord, (b[1] - a[1]) / chord
        cx, cy = mx - uy * apothem, my + ux * apothem
        start = math.atan2(a[1] - cy, a[0] - cx)
        n = arc_segments(abs(radius), theta, tolerance)
        # Interior points only: `a` is already appended and `b` starts the
        # next span, so neither is emitted twice.
        for k in range(1, n):
            ang = start + theta * k / n
            out.append((cx + abs(radius) * math.cos(ang), cy + abs(radius) * math.sin(ang)))
    if not closed:
        out.append(pts[-1])
    return out


#: Metres per pixel at zoom 0 on the equator, the Web Mercator constant.
_EQUATOR_M_PER_PX: Final[float] = 156543.03392

#: How far a simplified line may sit from the true one, IN SCREEN PIXELS.
#:
#: Under one pixel, so the simplification cannot be seen: the error is smaller
#: than the line drawn over it. Expressed in pixels rather than metres because
#: that is the thing that stays constant as the camera moves -- a curve seen
#: from far away does not need thirty points to look like a curve.
SIMPLIFY_PIXELS: Final[float] = 0.6


def degrees_per_pixel(zoom: float, latitude: float) -> float:
    """Ground resolution at a Web Mercator zoom, in DEGREES of longitude.

    Degrees rather than metres because the coordinates being simplified are
    degrees, and converting them to metres and back would introduce a second
    approximation to remove the first.
    """
    metres = _EQUATOR_M_PER_PX * math.cos(math.radians(latitude)) / (2.0**zoom)
    # 111,320 m per degree of longitude at the equator; the cos above already
    # carries the latitude, so it cancels and this is a plain constant.
    return metres / (111_320.0 * math.cos(math.radians(latitude)) or 1.0)


def simplify(points: Sequence[Point], tolerance: float) -> list[Point]:
    """Ramer-Douglas-Peucker: drop vertices no further than `tolerance` off.

    Endpoints are always kept, so a ring stays closed and a path keeps its
    ends -- which is what makes this safe to apply to geometry that other
    answers are computed from. It is applied to what is SENT, never to what is
    stored: the store keeps the full curve, and a coarser view is a view, not
    a loss.

    Iterative rather than recursive. A flattened arc can carry hundreds of
    points and Python's recursion limit is not a geometry constant.
    """
    n = len(points)
    if n < 3 or tolerance <= 0:
        return [(float(x), float(y)) for x, y in points]

    keep = [False] * n
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        ax, ay = points[first]
        bx, by = points[last]
        dx, dy = bx - ax, by - ay
        span = math.hypot(dx, dy)
        worst, worst_at = -1.0, first
        for i in range(first + 1, last):
            px, py = points[i]
            if span == 0:
                d = math.hypot(px - ax, py - ay)
            else:
                # Perpendicular distance to the segment's LINE, which is what
                # RDP measures; the endpoints are kept regardless.
                d = abs(dy * px - dx * py + bx * ay - by * ax) / span
            if d > worst:
                worst, worst_at = d, i
        if worst > tolerance:
            keep[worst_at] = True
            stack.append((first, worst_at))
            stack.append((worst_at, last))
    return [(float(points[i][0]), float(points[i][1])) for i in range(n) if keep[i]]
