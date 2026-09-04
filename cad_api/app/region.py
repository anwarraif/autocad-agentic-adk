"""Region geometry for area selection — window vs crossing; rect, polygon, circle.

Pure functions over numbers. Nothing here touches MongoDB, ezdxf or FastAPI,
which is deliberate: the selection rules are the part most likely to be subtly
wrong, and they are far easier to pin down in a unit test than through the
database.

**The two AutoCAD modes, and why both exist.** A drafter dragging a selection
box left-to-right means *window*: take only what is wholly inside. Dragging
right-to-left means *crossing*: take anything the box touches. They give very
different answers on the same box, and picking one for the user is not a
simplification — it is being wrong half the time. WPolygon / CPolygon are the
same pair for a hand-drawn polygon.

**What is actually tested, and the honesty about it.** MongoDB stores each
entity's *bounding box*, not its geometry (see PROJECT-KNOWLEDGE section 4).
So "crossing" here means *the entity's bounding box touches the region*, which
is an over-approximation: a long diagonal line has a big empty bounding box,
and a region clipping only that empty corner will select it even though the
line itself is nowhere near. The API says so in its response (`basis`) rather
than implying a precision it does not have. The alternative — storing full
geometry for 92,775 entities to make selection exact — is a much larger change
than the feature justifies, and is recorded as the reviewable trade-off in
DECISIONS-LOG D-027.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

#: (minx, miny, maxx, maxy)
Box = tuple[float, float, float, float]
#: (x, y)
Point = tuple[float, float]


def polygon_bbox(points: Sequence[Point]) -> Box:
    """The axis-aligned bounding box of a polygon.

    Used as the *database* pre-filter: the indexed bbox query narrows tens of
    thousands of entities to a handful before the exact polygon test runs in
    Python. Filtering the polygon in the browser over the whole drawing was
    the alternative and it means shipping every entity to the client.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def boxes_overlap(a: Box, b: Box) -> bool:
    """True if two axis-aligned boxes share any area, edges included."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def box_inside_box(inner: Box, outer: Box) -> bool:
    """True if `inner` is wholly contained in `outer`, touching allowed."""
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Crossing-number point-in-polygon, valid for concave polygons too.

    A ray is cast in +x and the edge crossings are counted; odd means inside.
    The half-open comparison `(yi > y) != (yj > y)` is what makes a vertex
    count exactly once instead of twice, which is the classic way this
    algorithm goes wrong on a horizontal ray through a vertex.

    A point lying exactly on an edge gets a deterministic answer, not an
    undefined one -- measured: for an axis-aligned rectangle the left and
    bottom edges count as inside and the right and top as outside, regardless
    of winding. What is NOT safe is the case that matters for polygons read
    from a file rather than clicked with a mouse: a point sitting exactly on a
    vertex SHARED by two neighbouring polygons is reported inside NEITHER of
    them. Measured on a rotated grid, 684 of 2001 points along a shared edge
    landed in no polygon at all.

    So the mouse-click justification that used to stand here does not travel.
    Callers joining labels to parcels must use `point_in_ring`, which decides
    the boundary case explicitly and counts it, and which translates first --
    without that translation a polygon does not recognise its own vertices as
    lying on its own edge at projected-coordinate magnitudes. This function is
    kept unchanged because `select_region` depends on its exact behaviour.
    """
    inside = False
    count = len(polygon)
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            # x of the edge at height y
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def _segments_cross(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1-p2 crosses segment p3-p4 (touching counts)."""

    def orientation(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Point, b: Point, c: Point) -> bool:
        """c is collinear with a-b; is it between them?"""
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    d1 = orientation(p3, p4, p1)
    d2 = orientation(p3, p4, p2)
    d3 = orientation(p1, p2, p3)
    d4 = orientation(p1, p2, p4)

    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    # Collinear touches.
    if d1 == 0 and on_segment(p3, p4, p1):
        return True
    if d2 == 0 and on_segment(p3, p4, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, p3):
        return True
    if d4 == 0 and on_segment(p1, p2, p4):
        return True
    return False


def _box_edges(box: Box) -> list[tuple[Point, Point]]:
    minx, miny, maxx, maxy = box
    corners: list[Point] = [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
    ]
    return [(corners[i], corners[(i + 1) % 4]) for i in range(4)]


def _polygon_edges(polygon: Sequence[Point]) -> list[tuple[Point, Point]]:
    count = len(polygon)
    return [(polygon[i], polygon[(i + 1) % count]) for i in range(count)]


def box_crosses_polygon(box: Box, polygon: Sequence[Point]) -> bool:
    """True if the box and the polygon share any area — AutoCAD CPolygon.

    Three cases, and all three are needed. Dropping any one of them misses a
    real overlap:

    1. a polygon vertex inside the box — polygon smaller than the box;
    2. a box corner inside the polygon — box smaller than the polygon, or a
       box wholly inside it (which has no crossing edges at all);
    3. an edge of one crossing an edge of the other — partial overlap.
    """
    if not boxes_overlap(polygon_bbox(polygon), box):
        return False

    minx, miny, maxx, maxy = box
    for px, py in polygon:
        if minx <= px <= maxx and miny <= py <= maxy:
            return True

    for corner in ((minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)):
        if point_in_polygon(corner[0], corner[1], polygon):
            return True

    for a, b in _polygon_edges(polygon):
        for c, d in _box_edges(box):
            if _segments_cross(a, b, c, d):
                return True
    return False


def box_inside_polygon(box: Box, polygon: Sequence[Point]) -> bool:
    """True if the box lies wholly within the polygon — AutoCAD WPolygon.

    All four corners inside is *not* sufficient on its own: a C-shaped
    polygon can hold all four corners of a box while its notch bites straight
    through the middle. So no polygon edge may cross the box either.
    """
    minx, miny, maxx, maxy = box
    for corner in ((minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)):
        if not point_in_polygon(corner[0], corner[1], polygon):
            return False

    for a, b in _polygon_edges(polygon):
        for c, d in _box_edges(box):
            if _segments_cross(a, b, c, d):
                return False
    return True


def circle_of(points: Sequence[Point]) -> tuple[Point, float]:
    """Centre and radius from the two points a circle region is drawn with.

    The wire format is the gesture: the first point is where the user pressed
    (the centre) and the second is where they released (a point on the rim).
    Carrying a radius as its own number would have been a second thing to keep
    consistent with the points, and a rim point survives the world<->view
    transforms the rest of this feature already does correctly.
    """
    (cx, cy), (rx, ry) = points[0], points[1]
    return (cx, cy), math.hypot(rx - cx, ry - cy)


def circle_bbox(center: Point, radius: float) -> Box:
    """The circle's bounding box — NOT the bounding box of its two points.

    Worth stating because getting it wrong is silent and costly: the box
    around [centre, rim] is a quarter of the circle, so it would be used as
    the indexed pre-filter and three quarters of the real matches would never
    be considered. The count would come back plausible and wrong.
    """
    cx, cy = center
    return (cx - radius, cy - radius, cx + radius, cy + radius)


def box_inside_circle(box: Box, center: Point, radius: float) -> bool:
    """True if the box lies wholly within the circle — the window rule.

    All four corners inside is sufficient here, and that is a real difference
    from `box_inside_polygon` rather than a shortcut. A disc is convex, so it
    contains the whole convex hull of any points it contains, and a box is
    exactly the convex hull of its corners. A polygon needs the extra
    edge-crossing test because a C-shaped one can hold all four corners while
    its notch bites through the middle; a circle has no notch.
    """
    if radius <= 0:
        return False
    cx, cy = center
    minx, miny, maxx, maxy = box
    r2 = radius * radius
    for px, py in ((minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)):
        if (px - cx) ** 2 + (py - cy) ** 2 > r2:
            return False
    return True


def box_crosses_circle(box: Box, center: Point, radius: float) -> bool:
    """True if the box and the circle share any area — the crossing rule.

    The closest point of the box to the centre is the centre with each
    coordinate clamped into the box's range; the two overlap exactly when that
    point is within the radius. Exact, and it needs no case analysis: it
    already covers the centre being inside the box (clamping changes nothing,
    distance is zero), the circle sitting wholly inside the box, and the box
    touching only at a corner.
    """
    if radius < 0:
        return False
    cx, cy = center
    minx, miny, maxx, maxy = box
    nearest_x = min(max(cx, minx), maxx)
    nearest_y = min(max(cy, miny), maxy)
    return (cx - nearest_x) ** 2 + (cy - nearest_y) ** 2 <= radius * radius


def shape_bbox(kind: str, points: Sequence[Point]) -> Box:
    """The bounding box of whichever shape these points describe.

    One place, because this box is what the indexed pre-filter uses. A shape
    whose bbox is computed as if it were another shape produces a query that
    quietly considers the wrong candidates.
    """
    if kind == "circle":
        center, radius = circle_of(points)
        return circle_bbox(center, radius)
    return polygon_bbox(points)


def box_in_shape(
    kind: str, mode: str, points: Sequence[Point]
) -> Callable[[Box], bool]:
    """The membership test for one shape and one mode, as a function of a box.

    Returned as a closure so the caller's loop over candidate entities stays
    one loop with one call, rather than growing a branch per shape inside the
    hot path.
    """
    if kind == "circle":
        center, radius = circle_of(points)
        if mode == "window":
            return lambda box: box_inside_circle(box, center, radius)
        return lambda box: box_crosses_circle(box, center, radius)
    if mode == "window":
        return lambda box: box_inside_polygon(box, points)
    return lambda box: box_crosses_polygon(box, points)


def entity_box(bbox: dict | None) -> Box | None:
    """The stored `{"min": [...], "max": [...]}` shape as a flat 2D box.

    Returns None for an entity with no bounding box — a few types (empty
    blocks, some proxies) have none, and they simply cannot take part in a
    spatial selection. Reported as a count rather than dropped silently.
    """
    if not bbox:
        return None
    low = bbox.get("min")
    high = bbox.get("max")
    if not low or not high or len(low) < 2 or len(high) < 2:
        return None
    try:
        return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))
    except (TypeError, ValueError):
        return None


def point_in_ring(
    x: float,
    y: float,
    ring: Sequence[Point],
    *,
    origin: Point | None = None,
    eps_rel: float = 1e-9,
) -> str:
    """Where a point sits relative to a closed ring: inside, outside, or boundary.

    Three answers, not two, and the third is the reason this exists. A plot
    number snapped onto the line it labels is not an edge case in a drawing
    produced by a drafter -- it is the normal way the drawing gets made -- and
    a two-valued answer has to silently pick a side or silently lose the label.

    What makes this work at projected-coordinate magnitudes is `eps_rel`, not
    the translation. That distinction was measured, and it was measured only
    because a mutation harness removed the translation and every one of 648
    tests stayed green. On a 25 x 12 m plot at E 691.942 / N 2.750.756 the
    rounding floor of the distance computation is 1,16e-10 in the absolute
    frame and 1,95e-10 after translating -- slightly WORSE, since subtracting
    the origin rounds every coordinate again -- while `eps` is 3,7e-8. Both
    frames therefore find all 2001 points along a shared edge, and a ring
    reports its own vertices at a distance of exactly zero in both. The
    translation here is kept for one reason only: the ring arrives already in
    its local frame from `geometry`, and re-deriving it would be the only
    place in the pipeline that does second-order work on absolute coordinates.
    For the CENTROID that translation is not optional at all -- see D-083.

    `eps_rel` is relative to the ring's diagonal, so it carries no unit and
    behaves identically in metres, in inches, and in a drawing that never says
    which it uses. It also means edge detection has a floor: below roughly
    0,065 drawing units across, `diag * eps_rel` drops under the rounding
    floor and points genuinely on the edge start being missed -- in either
    frame. Every parcel in every drawing here is orders of magnitude above
    that, but a drawing in millimetres-as-units would not be.
    """
    if len(ring) < 3:
        return "outside"
    ox, oy = origin if origin is not None else (
        min(p[0] for p in ring),
        min(p[1] for p in ring),
    )
    local = [(p[0] - ox, p[1] - oy) for p in ring]
    px, py = x - ox, y - oy

    xs = [p[0] for p in local]
    ys = [p[1] for p in local]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    eps = diag * eps_rel

    n = len(local)
    for i in range(n):
        a = local[i]
        b = local[(i + 1) % n]
        if _point_segment_distance(px, py, a, b) <= eps:
            return "boundary"

    return "inside" if point_in_polygon(px, py, local) else "outside"


def _point_segment_distance(px: float, py: float, a: Point, b: Point) -> float:
    """Shortest distance from a point to a segment, in the caller's frame."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))
