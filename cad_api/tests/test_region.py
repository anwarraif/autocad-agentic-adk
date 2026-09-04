"""Region selection geometry — window vs crossing, rectangle vs polygon.

Why this file is worth its length: a selection that is subtly wrong does not
crash and does not look broken. It returns *a* plausible set of objects, and
the only way anyone finds out is by counting by hand in AutoCAD. So the rules
are pinned here, in a form that needs no database and no drawing.

The strongest test in the file is the last one: it feeds the same random boxes
through the rectangle path (range comparisons, the shape MongoDB executes) and
the polygon path (point-in-polygon and segment crossing, executed in Python)
and demands they agree. Those are two independent implementations of the same
rule, and agreement between them is evidence neither has a private mistake.
"""

from __future__ import annotations

import random

import pytest

from app import region
from app.store import _rect_query


# A 10x10 region, used by most of the cases below.
SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


# ---------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------


def test_polygon_bbox():
    assert region.polygon_bbox([(3, 1), (-2, 7), (5, -4)]) == (-2, -4, 5, 7)


@pytest.mark.parametrize(
    "point, expected",
    [
        ((5, 5), True),      # middle
        ((0.1, 0.1), True),  # just inside a corner
        ((-1, 5), False),    # left of it
        ((11, 5), False),    # right of it
        ((5, -1), False),    # below
        ((5, 11), False),    # above
    ],
)
def test_point_in_polygon_square(point, expected):
    assert region.point_in_polygon(point[0], point[1], SQUARE) is expected


def test_point_in_polygon_handles_a_concave_shape():
    """A C: the notch is outside even though it is surrounded on three sides.

    A convex-only implementation (or a bounding-box shortcut) says the notch
    is inside, and every selection over a courtyard or a re-entrant plot
    boundary is then wrong.
    """
    c_shape = [
        (0.0, 0.0), (10.0, 0.0), (10.0, 3.0), (3.0, 3.0),
        (3.0, 7.0), (10.0, 7.0), (10.0, 10.0), (0.0, 10.0),
    ]
    assert region.point_in_polygon(1.0, 5.0, c_shape) is True   # the spine
    assert region.point_in_polygon(7.0, 5.0, c_shape) is False  # the notch
    assert region.point_in_polygon(7.0, 1.0, c_shape) is True   # lower arm


def test_point_in_polygon_ray_through_a_vertex_counts_once():
    """The classic failure: a horizontal ray leaving through a vertex.

    Counted twice, the point flips to "outside" and a selection loses objects
    along one exact latitude — which looks like random dropouts.
    """
    diamond = [(0.0, 5.0), (5.0, 0.0), (10.0, 5.0), (5.0, 10.0)]
    # Every ray cast from y == 5 leaves through the right-hand vertex
    # (10, 5) — exactly the case a naive count gets twice.
    assert region.point_in_polygon(5.0, 5.0, diamond) is True
    assert region.point_in_polygon(2.0, 5.0, diamond) is True
    assert region.point_in_polygon(9.0, 5.0, diamond) is True
    assert region.point_in_polygon(11.0, 5.0, diamond) is False
    assert region.point_in_polygon(-1.0, 5.0, diamond) is False


# ---------------------------------------------------------------------------
# Crossing vs window — the AutoCAD distinction
# ---------------------------------------------------------------------------


def test_crossing_takes_a_box_that_only_touches():
    box = (8.0, 8.0, 20.0, 20.0)  # one corner inside the square
    assert region.box_crosses_polygon(box, SQUARE) is True
    assert region.box_inside_polygon(box, SQUARE) is False


def test_window_takes_only_a_box_wholly_inside():
    box = (2.0, 2.0, 4.0, 4.0)
    assert region.box_inside_polygon(box, SQUARE) is True
    assert region.box_crosses_polygon(box, SQUARE) is True


def test_a_box_swallowing_the_whole_region_still_crosses():
    """No edge crosses and no polygon vertex test alone finds this one.

    The box contains the polygon entirely, so no box corner is inside the
    polygon and no edges intersect. Only the "is a polygon vertex inside the
    box" case catches it — and a drafter dragging a crossing box over a small
    plot absolutely expects that plot selected.
    """
    box = (-5.0, -5.0, 15.0, 15.0)
    assert region.box_crosses_polygon(box, SQUARE) is True
    assert region.box_inside_polygon(box, SQUARE) is False


def test_a_box_far_away_is_in_neither():
    box = (100.0, 100.0, 110.0, 110.0)
    assert region.box_crosses_polygon(box, SQUARE) is False
    assert region.box_inside_polygon(box, SQUARE) is False


def test_window_rejects_a_box_bridging_a_concave_notch():
    """All four corners inside, and still not contained.

    A C-shaped region holds both arms of the C; a box spanning from one arm
    to the other has every corner inside the polygon while its middle sits in
    open air. Testing corners alone would wrongly call this contained.
    """
    c_shape = [
        (0.0, 0.0), (10.0, 0.0), (10.0, 3.0), (3.0, 3.0),
        (3.0, 7.0), (10.0, 7.0), (10.0, 10.0), (0.0, 10.0),
    ]
    bridging = (5.0, 1.0, 9.0, 9.0)
    for corner in ((5.0, 1.0), (9.0, 1.0), (9.0, 9.0), (5.0, 9.0)):
        assert region.point_in_polygon(corner[0], corner[1], c_shape) is True
    assert region.box_inside_polygon(bridging, c_shape) is False
    assert region.box_crosses_polygon(bridging, c_shape) is True


def test_touching_edges_count_as_crossing():
    """Edge-to-edge contact selects, matching AutoCAD's crossing behaviour."""
    box = (10.0, 4.0, 12.0, 6.0)  # left edge lies on the square's right edge
    assert region.box_crosses_polygon(box, SQUARE) is True


# ---------------------------------------------------------------------------
# entity_box: the stored shape
# ---------------------------------------------------------------------------


def test_entity_box_reads_the_stored_shape():
    assert region.entity_box({"min": [1, 2, 0], "max": [3, 4, 5]}) == (1.0, 2.0, 3.0, 4.0)


@pytest.mark.parametrize(
    "bad",
    [None, {}, {"min": [1]}, {"min": [1, 2]}, {"min": [1, 2], "max": None},
     {"min": ["a", "b"], "max": [1, 2]}],
)
def test_entity_box_returns_none_rather_than_guessing(bad):
    """An entity with no usable bounding box cannot be selected spatially.

    Returning a zero box instead would place it at the origin, where it would
    silently join every selection drawn near (0, 0).
    """
    assert region.entity_box(bad) is None


# ---------------------------------------------------------------------------
# The rectangle path, as MongoDB executes it
# ---------------------------------------------------------------------------


def test_rect_query_is_scoped_to_the_drawing_and_layout():
    query = _rect_query("abc", (0, 0, 10, 10), mode="crossing", layout="Model")
    assert query["drawing_id"] == "abc"
    assert query["layout"] == "Model"


def test_rect_query_omits_layout_when_not_given():
    assert "layout" not in _rect_query("abc", (0, 0, 10, 10), mode="crossing", layout=None)


def test_crossing_and_window_compare_different_sides():
    """The asymmetry is the whole point, so it is pinned explicitly.

    Crossing compares each side of the entity against the *opposite* side of
    the region; window compares like against like. Getting these the same way
    round makes the two modes identical, and the feature pointless.
    """
    crossing = _rect_query("d", (0, 0, 10, 10), mode="crossing", layout=None)
    window = _rect_query("d", (0, 0, 10, 10), mode="window", layout=None)

    assert crossing["bbox.min.0"] == {"$lte": 10}   # entity starts before the far edge
    assert crossing["bbox.max.0"] == {"$gte": 0}    # and ends after the near edge
    assert window["bbox.min.0"] == {"$gte": 0}      # entity starts after the near edge
    assert window["bbox.max.0"] == {"$lte": 10}     # and ends before the far edge
    assert crossing != window


def _matches_rect_query(query: dict, box: region.Box) -> bool:
    """Evaluate a `_rect_query` filter against one box, as MongoDB would."""
    values = {
        "bbox.min.0": box[0],
        "bbox.min.1": box[1],
        "bbox.max.0": box[2],
        "bbox.max.1": box[3],
    }
    for field, condition in query.items():
        if field not in values:
            continue
        value = values[field]
        if "$lte" in condition and not value <= condition["$lte"]:
            return False
        if "$gte" in condition and not value >= condition["$gte"]:
            return False
    return True


@pytest.mark.parametrize("mode", ["crossing", "window"])
def test_rectangle_path_and_polygon_path_agree(mode):
    """Two independent implementations of one rule must give one answer.

    A rectangle drawn by the user goes to MongoDB as range comparisons; the
    same rectangle expressed as a four-vertex polygon goes through the Python
    geometry instead. If those ever disagree, one of the two selection shapes
    is lying to the user — and this is the only place that would notice.
    """
    rng = random.Random(20260820)
    region_box = (0.0, 0.0, 10.0, 10.0)
    as_polygon = [
        (region_box[0], region_box[1]),
        (region_box[2], region_box[1]),
        (region_box[2], region_box[3]),
        (region_box[0], region_box[3]),
    ]
    query = _rect_query("d", region_box, mode=mode, layout=None)
    predicate = (
        region.box_inside_polygon if mode == "window" else region.box_crosses_polygon
    )

    for _ in range(2000):
        x0 = rng.uniform(-6, 16)
        y0 = rng.uniform(-6, 16)
        box = (x0, y0, x0 + rng.uniform(0.1, 8), y0 + rng.uniform(0.1, 8))
        assert _matches_rect_query(query, box) is predicate(box, as_polygon), (
            f"{mode}: rectangle path and polygon path disagree on {box}"
        )


# ---------------------------------------------------------------------------
# Input validation
#
# Every case here returned 200 OK with a plausible-looking answer before it
# was fixed, which is the failure mode that matters: a wrong region does not
# crash, it selects the wrong objects and says nothing.
# ---------------------------------------------------------------------------


def test_shoelace_area_is_zero_for_degenerate_polygons():
    from app.main import _shoelace_area

    same = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    collinear = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    assert _shoelace_area(same) == 0.0
    assert _shoelace_area(collinear) == 0.0


def test_shoelace_area_is_positive_whichever_way_round():
    """Winding direction must not decide whether a region is accepted.

    A polygon clicked clockwise and the same polygon clicked anticlockwise are
    the same region to the user.
    """
    from app.main import _shoelace_area

    clockwise = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    anticlockwise = list(reversed(clockwise))
    assert _shoelace_area(clockwise) == pytest.approx(200.0)
    assert _shoelace_area(anticlockwise) == pytest.approx(200.0)


def test_max_coordinate_is_below_where_the_arithmetic_breaks():
    """The bound has to leave the polygon tests in exact arithmetic.

    `box_crosses_polygon` multiplies coordinate differences, so two values near
    the float ceiling overflow to infinity and every comparison silently turns
    False — a polygon enclosing the whole drawing then reports zero matches.
    """
    from app.main import MAX_COORDINATE

    assert MAX_COORDINATE <= 1e15
    limit = MAX_COORDINATE
    huge = [(-limit, -limit), (limit, -limit), (limit, limit), (-limit, limit)]
    # At the limit the geometry must still work, not overflow.
    assert region.box_crosses_polygon((0.0, 0.0, 1.0, 1.0), huge) is True
    assert region.box_inside_polygon((0.0, 0.0, 1.0, 1.0), huge) is True


# --- circle regions -------------------------------------------------------
# A circle is not a many-sided polygon here. Approximating one would have been
# a handful of lines, and would have misclassified every object near the rim
# by up to the sagitta of a segment -- invisibly, and differently at every
# zoom level. These tests pin the exact behaviour instead.


def test_circle_of_reads_centre_and_rim():
    center, radius = region.circle_of([(10.0, 10.0), (13.0, 14.0)])
    assert center == (10.0, 10.0)
    assert radius == pytest.approx(5.0)  # 3-4-5


def test_circle_bbox_is_the_circle_not_the_two_points():
    # The trap this guards: bbox([centre, rim]) is a quarter of the circle,
    # so using it as the pre-filter would silently drop three quarters of the
    # real matches and still return a plausible count.
    center, radius = region.circle_of([(0.0, 0.0), (0.0, 5.0)])
    assert region.circle_bbox(center, radius) == (-5.0, -5.0, 5.0, 5.0)
    assert region.polygon_bbox([(0.0, 0.0), (0.0, 5.0)]) == (0.0, 0.0, 0.0, 5.0)


def test_box_inside_circle_needs_every_corner():
    center, radius = (0.0, 0.0), 10.0
    # Wholly inside.
    assert region.box_inside_circle((-1.0, -1.0, 1.0, 1.0), center, radius)
    # The far corner of a box that reaches past the rim: 3 corners in, 1 out.
    assert not region.box_inside_circle((0.0, 0.0, 9.0, 9.0), center, radius)
    # Exactly on the rim counts as inside -- the boundary belongs to the disc.
    assert region.box_inside_circle((0.0, 0.0, 0.0, 10.0), center, radius)


def test_box_crosses_circle_covers_every_arrangement():
    center, radius = (0.0, 0.0), 10.0
    # Box far away on the diagonal: nearest corner is at distance 14.1.
    assert not region.box_crosses_circle((10.0, 10.0, 20.0, 20.0), center, radius)
    # Same distance along an axis, where the nearest point is an edge not a
    # corner -- the case a corners-only test gets wrong.
    assert region.box_crosses_circle((-50.0, -1.0, -9.0, 1.0), center, radius)
    # Circle wholly inside the box: no corner is in the circle, yet they
    # plainly overlap.
    assert region.box_crosses_circle((-100.0, -100.0, 100.0, 100.0), center, radius)
    # Centre inside the box.
    assert region.box_crosses_circle((-1.0, -1.0, 1.0, 1.0), center, radius)
    # Touching at a single point on an axis.
    assert region.box_crosses_circle((10.0, -1.0, 20.0, 1.0), center, radius)


def test_window_is_stricter_than_crossing_for_circles():
    center, radius = (0.0, 0.0), 10.0
    straddling = (5.0, 5.0, 15.0, 15.0)
    assert region.box_crosses_circle(straddling, center, radius)
    assert not region.box_inside_circle(straddling, center, radius)


def test_zero_radius_circle_selects_nothing_in_window_mode():
    # A click without a drag. Crossing still finds a box containing the point,
    # which is the same thing a zero-area rectangle would do; window finds
    # nothing, because nothing fits inside a point.
    center, radius = (0.0, 0.0), 0.0
    assert not region.box_inside_circle((-1.0, -1.0, 1.0, 1.0), center, radius)
    assert region.box_crosses_circle((-1.0, -1.0, 1.0, 1.0), center, radius)


def test_box_in_shape_dispatches_on_kind_and_mode():
    points = [(0.0, 0.0), (0.0, 10.0)]
    straddling = (5.0, 5.0, 15.0, 15.0)
    assert region.box_in_shape("circle", "crossing", points)(straddling)
    assert not region.box_in_shape("circle", "window", points)(straddling)
    # And the polygon path is untouched by the new branch.
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert region.box_in_shape("polygon", "crossing", square)((5.0, 5.0, 15.0, 15.0))
    assert region.box_in_shape("polygon", "window", square)((1.0, 1.0, 2.0, 2.0))


def test_shape_bbox_matches_the_shape():
    circle_points = [(0.0, 0.0), (3.0, 4.0)]
    assert region.shape_bbox("circle", circle_points) == (-5.0, -5.0, 5.0, 5.0)
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert region.shape_bbox("polygon", square) == (0.0, 0.0, 10.0, 10.0)
