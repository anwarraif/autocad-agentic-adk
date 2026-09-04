"""Ring geometry: invariants that catch the defects Janadriyah's numbers miss.

The tests in this file deliberately use not one number from any single drawing.
Everything checked here is a PROPERTY — shift it and the result shifts by
exactly that much; rotate it and the fingerprint does not change; a polygon
contains its own vertices. Properties like that hold in the 17th drawing whose
layer convention nobody has ever seen, while Janadriyah's numbers will never
catch breakage there.

Two of them were written to fail against code that had previously been taken as
correct: `test_a_translated_ring_moves_by_exactly_that_much` and
`test_a_polygon_contains_its_own_vertices`. Both passed the old implementation
without a single red light.

The last three tests in this file arrived later, from a mutation harness, and
their reason needs stating: the second of the two above did NOT fail because of
the translation. A polygon recognises its own vertices at a distance of exactly
zero even in absolute coordinates; what makes edge detection work is an epsilon
that scales with the ring's diagonal. That is the property pinned below,
together with its floor. D-084 holds the numbers.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import geometry as geo  # noqa: E402
from app import region  # noqa: E402


# The projected coordinates this project really uses. Not decoration: the whole
# class of defects this file tests only shows up at coordinate magnitudes like
# these.
E0, N0 = 691942.0, 2750756.0


def rect(cx: float, cy: float, w: float, h: float, ang: float = 0.0):
    c, s = math.cos(ang), math.sin(ang)
    return [
        (cx + dx * c - dy * s, cy + dx * s + dy * c)
        for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
    ]


def ell(cx: float, cy: float, scale: float = 1.0):
    """An L-polygon: a shape that is NOT symmetric under order reversal.

    A rectangle cannot be used to test winding normalisation, because its edge
    list is palindromic and the test passes even when the normalisation is not
    there at all.
    """
    pts = [(0, 0), (30, 0), (30, 10), (12, 10), (12, 25), (0, 25)]
    return [(cx + x * scale, cy + y * scale) for x, y in pts]


def bbox_centre(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)


# --- precision: the defect that is the reason for D-083 ----------------------


def test_a_rotated_rectangle_has_its_centroid_exactly_at_its_bbox_centre():
    """This is what makes the old spec's justifying number worth suspecting.

    A rectangle has a centre of symmetry, so the difference between those two
    points is ZERO — at whatever rotation angle and at whatever coordinates.
    The difference once reported over 2,380 plots is therefore not a signal;
    it is an error.
    """
    for ang_deg in (0, 17, 30, 49, 63.4, 90):
        pts = rect(E0, N0, 12.0, 25.0, math.radians(ang_deg))
        c = geo.polygon_centroid(pts)
        assert c is not None
        assert math.dist(c, bbox_centre(pts)) == pytest.approx(0.0, abs=1e-9), ang_deg


def test_the_error_grows_as_the_parcel_shrinks():
    """The counter-intuitive error direction, and why translation is required.

    The area error is set by the coordinate magnitude, while its divisor is the
    parcel's area itself — so a small parcel far from the origin is the worst
    case, and small parcels are precisely the ones people ask about most often.
    """

    def absolute_frame_centroid(points):
        a = cx = cy = 0.0
        n = len(points)
        for i in range(n):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n]
            cr = x0 * y1 - x1 * y0
            a += cr
            cx += (x0 + x1) * cr
            cy += (y0 + y1) * cr
        a *= 0.5
        return (cx / (6 * a), cy / (6 * a))

    ang = math.radians(49.0)

    def worst_over_a_population(w, h):
        # The rounding error on ONE polygon is a lottery; what grows as the
        # parcel shrinks is the maximum over a population. Testing a single
        # sample would test the lottery, not the claim.
        worst_abs = worst_local = 0.0
        for i in range(400):
            pts = rect(E0 + (i % 20) * 30.0, N0 + (i // 20) * 30.0, w, h, ang)
            centre = bbox_centre(pts)
            worst_abs = max(worst_abs, math.dist(absolute_frame_centroid(pts), centre))
            worst_local = max(worst_local, math.dist(geo.polygon_centroid(pts), centre))
        return worst_abs, worst_local

    small_abs, small_local = worst_over_a_population(5.0, 5.0)
    large_abs, large_local = worst_over_a_population(76.0, 76.0)

    assert small_abs > large_abs > 0.0
    assert small_abs > 1.0, f"a small parcel should miss by metres, got {small_abs}"

    # And the translation erases both of them, not merely shrinks them.
    assert small_local == pytest.approx(0.0, abs=1e-9)
    assert large_local == pytest.approx(0.0, abs=1e-9)


def test_a_translated_ring_moves_by_exactly_that_much():
    """I-1 — the translation invariant.

    Catches a shoelace computed in the absolute frame. Against the old
    implementation this test fails; against the whole Definition of Done of the
    old spec it was never run at all.
    """
    base = ell(10.0, 20.0)
    shifted = [(x + 1e6, y + 1e6) for x, y in base]

    c0 = geo.polygon_centroid(base)
    c1 = geo.polygon_centroid(shifted)
    assert c1[0] - c0[0] == pytest.approx(1e6, abs=1e-6)
    assert c1[1] - c0[1] == pytest.approx(1e6, abs=1e-6)

    assert geo.shape_key(base) == geo.shape_key(shifted)
    for a, b in zip(geo.edge_lengths(base, True), geo.edge_lengths(shifted, True)):
        assert a == pytest.approx(b, rel=1e-12)


def test_area_survives_being_moved_far_from_the_origin():
    near = ell(0.0, 0.0)
    far = ell(E0, N0)
    assert abs(geo.signed_area(far)) == pytest.approx(abs(geo.signed_area(near)), rel=1e-12)


# --- edges: the second defect the same translation fixes ---------------------


def test_a_polygon_contains_its_own_vertices():
    """I-2 — the cheapest and sharpest invariant in this whole store.

    In projected coordinates, a polygon does not recognise its own vertices as
    lying on its edge unless the computation is translated first. If this
    fails, edge detection is completely blind and every label snapped to its
    plot's line disappears without a trace.
    """
    for pts in (rect(E0, N0, 12.0, 25.0, math.radians(49.0)), ell(E0, N0)):
        for v in pts:
            assert region.point_in_ring(v[0], v[1], pts) == "boundary"


def test_a_point_on_a_shared_edge_is_boundary_not_lost():
    pts = rect(E0, N0, 12.0, 25.0)
    a, b = pts[0], pts[1]
    for i in range(1, 20):
        t = i / 20.0
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        assert region.point_in_ring(x, y, pts) == "boundary"


def test_inside_and_outside_still_answer_plainly():
    pts = rect(E0, N0, 12.0, 25.0)
    assert region.point_in_ring(E0, N0, pts) == "inside"
    assert region.point_in_ring(E0 + 500, N0, pts) == "outside"


def test_point_in_ring_handles_a_concave_shape():
    pts = ell(E0, N0)
    # A point inside the notch of the L: inside the bbox, outside the polygon.
    assert region.point_in_ring(E0 + 25, N0 + 20, pts) == "outside"
    assert region.point_in_ring(E0 + 5, N0 + 5, pts) == "inside"


# --- shape fingerprint -------------------------------------------------------


def test_the_same_shape_drawn_the_other_way_round_keeps_its_key():
    """I-5 — the winding invariant, tested on an L-polygon.

    Tested on a rectangle, this test passes even when winding normalisation is
    not there at all: the edge list [12,25,12,25] is palindromic.
    """
    pts = ell(0.0, 0.0)
    assert geo.shape_key(pts) == geo.shape_key(list(reversed(pts)))


def test_a_rectangle_and_a_parallelogram_do_not_share_a_key():
    """G6 — edge lengths alone are not a shape fingerprint.

    Both have identical edge sequences; only their angles differ, and in a
    rotated grid meeting a curved road shapes like this are ordinary.
    """
    r = rect(0.0, 0.0, 12.0, 25.0)
    skew = [(x + 0.35 * y, y) for x, y in r]
    lens_r = sorted(round(v, 6) for v in geo.edge_lengths(r, True))
    lens_s = sorted(round(v, 6) for v in geo.edge_lengths(skew, True))
    assert lens_r != lens_s or geo.shape_key(r) != geo.shape_key(skew)
    assert geo.shape_key(r) != geo.shape_key(skew)


def test_rotation_does_not_change_the_key_but_scale_does():
    base = ell(0.0, 0.0)
    ang = math.radians(17.0)
    c, s = math.cos(ang), math.sin(ang)
    turned = [(x * c - y * s, x * s + y * c) for x, y in base]
    assert geo.shape_key(base) == geo.shape_key(turned)
    assert geo.shape_key(base) != geo.shape_key(ell(0.0, 0.0, scale=2.0))


def test_the_key_is_blind_to_units_and_says_so():
    """G10 — the same key for 12x25 metres and for 12x25 inches.

    Not a defect; it is scale-sensitive and unit-blind by construction. What is
    mandatory is that every use of it filters on `drawing_id` first, and that
    `shape_key_basis` says so inside the response.
    """
    assert geo.shape_key(rect(0, 0, 12, 25)) == geo.shape_key(rect(999, -999, 12, 25))
    assert "single drawing" in geo.shape_key_basis()


def test_a_duplicated_vertex_does_not_change_the_key():
    """Converted files produce duplicated vertices routinely."""
    pts = ell(0.0, 0.0)
    with_dupe = [pts[0], pts[0], *pts[1:]]
    assert geo.shape_key(pts) == geo.shape_key(with_dupe)


# --- honest absence ----------------------------------------------------------


def test_a_zero_area_ring_has_no_centroid_and_does_not_raise():
    degenerate = [(0.0, 0.0), (10.0, 0.0), (0.0, 0.0), (10.0, 0.0)]
    assert geo.polygon_centroid(degenerate) is None
    assert geo.shape_key([(0.0, 0.0), (1.0, 1.0)]) is None


def test_dedupe_counts_what_it_dropped():
    pts, dropped = geo.dedupe([(0, 0), (0, 0), (1, 0), (1, 1), (0, 0)])
    assert pts == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert dropped == 2


def test_self_intersection_is_unknown_above_the_limit_not_guessed():
    bowtie = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
    assert geo.is_simple(bowtie) is False
    assert geo.is_simple(rect(0, 0, 4, 4)) is True
    big = [(math.cos(i / 100.0) * 50, math.sin(i / 100.0) * 50) for i in range(200)]
    assert geo.is_simple(big) is None


def test_perimeter_from_ring_matches_the_edge_lengths():
    """I-4 — two independent computation paths that are obliged to meet."""
    pts = ell(E0, N0)
    assert geo.perimeter(pts, True) == pytest.approx(
        sum(geo.edge_lengths(pts, True)), rel=1e-12
    )


# --- what actually makes edge detection work --------------------------------
#
# The three tests below were added after a mutation harness removed the
# translation in `point_in_ring` and all 648 tests stayed green. That mutant
# was in fact supposed to pass: measured on a 25 x 12 m parcel at
# E 691,942 / N 2,750,756, the rounding floor of the point-to-segment distance
# is 1.16e-10 in the absolute frame and 1.95e-10 after translation -- slightly
# WORSE -- while `eps` has the value 3.7e-8. What saves edge detection is an
# epsilon that scales with the ring's diagonal, not the translation.
#
# So what is pinned here is that property, not the translation. For the
# centroid the story is an entirely different one, and is already guarded
# above.


def test_edge_tolerance_scales_with_the_ring_not_with_the_coordinates():
    """The same parcel, at the origin and at UTM coordinates: same behaviour.

    If the epsilon were ever made absolute, this is the test that falls --
    because an absolute epsilon that suits one of them does not suit the other.
    """
    for cx, cy in ((0.0, 0.0), (E0, N0), (1e8, -1e8)):
        pts = rect(cx, cy, 12.0, 25.0, math.radians(49.0))
        for v in pts:
            assert region.point_in_ring(v[0], v[1], pts) == "boundary"
        a, b = pts[0], pts[1]
        for i in range(1, 40):
            t = i / 40.0
            assert region.point_in_ring(
                a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, pts
            ) == "boundary"


def test_edge_tolerance_carries_no_unit():
    """The same drawing in metres and in inches answers the same.

    Eleven of the nineteen drawings in this stack are in inches and three state
    no unit at all. An epsilon is a LENGTH; this one is not.
    """
    metre = rect(E0, N0, 12.0, 25.0, math.radians(49.0))
    inch = rect(E0, N0, 12.0 / 0.0254, 25.0 / 0.0254, math.radians(49.0))
    for pts in (metre, inch):
        centre = (
            sum(p[0] for p in pts) / 4.0,
            sum(p[1] for p in pts) / 4.0,
        )
        assert region.point_in_ring(centre[0], centre[1], pts) == "inside"
        for v in pts:
            assert region.point_in_ring(v[0], v[1], pts) == "boundary"


def test_edge_detection_has_a_floor_and_the_floor_is_stated():
    """G7 — a stated limit, not a limit somebody else discovers.

    Below roughly 0.065 drawing units, `diag * eps_rel` falls below the
    rounding floor at projected-coordinate magnitudes, and points that REALLY
    are on the edge start being missed. Every parcel in every drawing here is
    several orders above that; a drawing whose unit is millimetres is not.
    """
    def all_on_edge_detected(side: float) -> bool:
        pts = rect(E0, N0, side, side, math.radians(49.0))
        a, b = pts[0], pts[1]
        return all(
            region.point_in_ring(
                a[0] + (b[0] - a[0]) * (i / 50.0),
                a[1] + (b[1] - a[1]) * (i / 50.0),
                pts,
            ) == "boundary"
            for i in range(51)
        )

    assert all_on_edge_detected(1.0), "one unit must be far above its floor"
    assert all_on_edge_detected(0.5)
    assert not all_on_edge_detected(
        0.001
    ), "if this passes, the floor has moved and the docstring's number is stale"
