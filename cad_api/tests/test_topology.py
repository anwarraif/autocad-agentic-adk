"""DOSSIER Phases 5a and 5b — `overlap_scan` and `frontage_check`.

Owner: the TOPOLOGY lane.

**Known truth first.** A tool whose findings cannot be checked is a tool whose
findings cannot be trusted, so nothing here starts from Janadriyah. It starts
from rings whose intersection area is known by hand — two squares offset by a
half, an L-shape whose notch is empty, two rings that merely share an edge —
and only then goes to the real drawing. The single most important test in this
file is the negative one: **two parcels sharing an edge must NOT be reported as
overlapping.** That is the false-positive guard, and without it a clean master
plan would come back with two thousand findings.

Three kinds of test, in this order:

1.  **Pure geometry**, with exact expected numbers and no database at all.
    Including the case that decides whether the engine is a toy: a CONCAVE ring
    (G6 — nothing here may assume plots are rectangles) and rings at projected
    coordinate magnitudes, where a naive cross product loses its precision.
2.  **Recipe behaviour over a fixture store**, with a fake collection. Two
    overlapping parcels, two edge-sharing parcels, a sliver below the
    tolerance, an unmeasurable bulged ring that must be COUNTED and never
    zeroed, one landlocked parcel and one parcel touching a road.
3.  **Janadriyah**, skipped when there is no database — and recorded as a
    finding either way. A clean drawing is a result, not a failure.

**One note for the integrator.** Importing this file registers two more
recipes, so `EIGHT_RECIPES` in `tests/test_recipes.py` no longer matches. That
pin is the integrator's single-line change; this file deliberately does not
touch it.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import evidence as ev  # noqa: E402
from app import geometry as geom  # noqa: E402
from app import recipes  # noqa: E402
from app.recipes import registry as rr  # noqa: E402
from app.recipes import topology as tp  # noqa: E402


# =============================================================================
# Tooling
# =============================================================================


DRAWING = "596212db022a3397"

#: A drawing id that is deliberately NOT in the store and has no land use
#: config. Used wherever the point of the test is the fixture geometry rather
#: than the real drawing.
FIXTURE = "fixture000000dead"

LAYOUT = "Model"

#: Layer names live in the TEST, never in the module under test (G1).
PLOTS = "fixture-plots"
ROADS = "fixture-roads"
CORRIDOR = "fixture-corridor"
BOUNDARY = "fixture-boundary"

METRE_UNITS = {
    "name": "m",
    "declared_in_file": True,
    "length_unit": "m",
    "area_unit": "m2",
    "space": "model",
}

UNITLESS = {
    "name": "unitless",
    "declared_in_file": False,
    "length_unit": None,
    "area_unit": None,
    "space": "model",
    "why_no_unit": "$INSUNITS is 0",
}

INCH_UNITS = {
    "name": "in",
    "declared_in_file": True,
    "length_unit": "in",
    "area_unit": "in2",
    "space": "model",
}

ALL_UNIT_SHAPES = [
    pytest.param(METRE_UNITS, id="metre"),
    pytest.param(INCH_UNITS, id="inch"),
    pytest.param(UNITLESS, id="undeclared"),
]


class FakeCursor(list):
    """Just enough cursor for `.find(...).sort(...)`."""

    def sort(self, spec):
        for key, direction in reversed(list(spec)):
            list.sort(
                self,
                key=lambda d, k=key: (d.get(k) is None, str(d.get(k))),
                reverse=direction < 0,
            )
        return self


class FakeCollection:
    """Just enough MongoDB for this file, and not one line more.

    It interprets equality and `$in`, which is every operator the two recipes
    use. A fake that interpreted more would be a fake that can lie differently
    from the real cluster.
    """

    def __init__(self, docs=()):
        self.docs = [copy.deepcopy(d) for d in docs]
        self.filters: list[dict] = []

    def _match(self, doc, flt):
        for key, want in flt.items():
            got = doc.get(key)
            if isinstance(want, dict):
                for op, arg in want.items():
                    if op == "$in":
                        if got not in arg:
                            return False
                    else:
                        raise AssertionError(f"operator {op} has not been faked")
            elif got != want:
                return False
        return True

    def find(self, flt, projection=None):
        self.filters.append(copy.deepcopy(flt))
        return FakeCursor(copy.deepcopy(d) for d in self.docs if self._match(d, flt))

    def count_documents(self, flt):
        self.filters.append(copy.deepcopy(flt))
        return sum(1 for d in self.docs if self._match(d, flt))


def square(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def _bbox(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return {"min": [min(xs), min(ys)], "max": [max(xs), max(ys)]}


def ring_doc(handle, layer, ring, *, status="complete", layout=LAYOUT, drawing=FIXTURE):
    return {
        "_id": f"{drawing}:{handle}",
        "drawing_id": drawing,
        "layout": layout,
        "layer": layer,
        "handle": handle,
        "type": "LWPOLYLINE",
        "ring": [[float(p[0]), float(p[1])] for p in ring] if status == "complete" else None,
        "ring_status": status,
        "ring_origin": None,
        "area": abs(geom.signed_area(ring)) if status == "complete" else None,
        "polygon_centroid": None,
        "bbox": _bbox(ring),
        "geometry_note": None if status == "complete" else f"ring_status {status}",
    }


def line_doc(handle, layer, p0, p1, *, layout=LAYOUT, drawing=FIXTURE):
    """A road drawn as a LINE: a bounding box, and no vertices anywhere."""
    return {
        "_id": f"{drawing}:{handle}",
        "drawing_id": drawing,
        "layout": layout,
        "layer": layer,
        "handle": handle,
        "type": "LINE",
        "bbox": {
            "min": [min(p0[0], p1[0]), min(p0[1], p1[1])],
            "max": [max(p0[0], p1[0]), max(p0[1], p1[1])],
        },
    }


@pytest.fixture
def fixture_store(monkeypatch):
    """A small drawing whose every answer is known by hand.

    Its geometry, all on one layout:

    * `P-OVERLAP-A` (0,0)-(10,10) and `P-OVERLAP-B` (5,0)-(15,10) — they share
      50 units², half of either one. A real overlap.
    * `P-EDGE-A` (20,0)-(30,10) and `P-EDGE-B` (30,0)-(40,10) — they share an
      EDGE and nothing else. Shared area exactly 0. **Must not be flagged.**
    * `P-SLIVER-A` (50,0)-(60,10) and `P-SLIVER-B` (59.99,0)-(69.99,10) — they
      share 0.1 units², a hundredth of the smaller parcel's area, which is
      below the tolerance. A sliver, reported as a sliver.
    * `P-BULGE` — a ring the store could not close. Unmeasurable, counted, and
      never treated as a parcel of area zero.
    * `BOUNDARY-1` (0,0)-(100,100) on its own layer, so the gap half of the
      question has something to be measured against.
    * `ROAD-1`, a LINE running just above `P-OVERLAP-A`, 0.05 away.
    """
    docs = [
        ring_doc("P-OVERLAP-A", PLOTS, square(0, 0, 10, 10)),
        ring_doc("P-OVERLAP-B", PLOTS, square(5, 0, 10, 10)),
        ring_doc("P-EDGE-A", PLOTS, square(20, 0, 10, 10)),
        ring_doc("P-EDGE-B", PLOTS, square(30, 0, 10, 10)),
        ring_doc("P-SLIVER-A", PLOTS, square(50, 0, 10, 10)),
        ring_doc("P-SLIVER-B", PLOTS, square(59.99, 0, 10, 10)),
        ring_doc("P-BULGE", PLOTS, square(80, 0, 10, 10), status="bulge"),
        ring_doc("BOUNDARY-1", BOUNDARY, square(0, 0, 100, 100)),
        line_doc("ROAD-1", ROADS, (0.0, 10.05), (100.0, 10.05)),
    ]
    entities = FakeCollection(docs)
    monkeypatch.setattr(tp, "_entities", lambda: entities)
    return entities


def _run(name, params, *, drawing=FIXTURE, units=None, layout=LAYOUT):
    return recipes.run(
        drawing, name, params, layout=layout, units=dict(units or METRE_UNITS)
    )


# =============================================================================
# 1. Known truth: the intersection engine, with no database at all
# =============================================================================


@pytest.mark.parametrize(
    "a,b,expected,why",
    [
        (square(0, 0, 1, 1), square(0.5, 0, 1, 1), 0.5, "half of each square"),
        (square(0, 0, 1, 1), square(1, 0, 1, 1), 0.0, "a shared EDGE is not an overlap"),
        (square(0, 0, 1, 1), square(1, 1, 1, 1), 0.0, "a shared CORNER is not an overlap"),
        (square(0, 0, 1, 1), square(0, 0, 1, 1), 1.0, "identical rings share all of it"),
        (square(0, 0, 1, 1), square(5, 5, 1, 1), 0.0, "disjoint"),
        (square(0, 0, 10, 10), square(2, 2, 3, 4), 12.0, "one wholly inside the other"),
    ],
)
def test_the_intersection_area_is_exact_on_rings_measured_by_hand(a, b, expected, why):
    assert tp.ring_intersection_area(a, b) == pytest.approx(expected, abs=1e-9), why


def test_a_shared_edge_is_not_an_overlap_whichever_way_round_it_is_asked():
    """The false-positive guard, stated twice on purpose.

    Every parcel in a master plan shares an edge with its neighbour. An engine
    that called that an overlap would return a finding for every plot in the
    drawing, and the list would be discarded on the first reading — taking the
    real overlaps with it.
    """
    left = square(0, 0, 25, 12)
    right = square(25, 0, 25, 12)
    assert tp.ring_intersection_area(left, right) == 0.0
    assert tp.ring_intersection_area(right, left) == 0.0


def test_winding_direction_does_not_change_the_answer():
    """Clockwise or counter-clockwise is a drafting habit, not a fact about the
    ground. A scan whose sign flipped with it would be reporting the drafter."""
    a = square(0, 0, 1, 1)
    b = square(0.5, 0, 1, 1)
    assert tp.ring_intersection_area(list(reversed(a)), b) == pytest.approx(0.5)
    assert tp.ring_intersection_area(a, list(reversed(b))) == pytest.approx(0.5)
    assert tp.ring_intersection_area(
        list(reversed(a)), list(reversed(b))
    ) == pytest.approx(0.5)


def test_a_concave_ring_is_measured_and_not_assumed_to_be_a_rectangle():
    """G6: nothing here may assume a plot is convex, is a rectangle, or has
    four to six vertices.

    The L below has area 7. A square dropped into its NOTCH shares nothing with
    it, and a convex-hull engine would report 4 — the classic silent failure.
    """
    shape_l = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]
    assert abs(geom.signed_area(shape_l)) == pytest.approx(7.0)
    assert tp.ring_intersection_area(shape_l, square(1, 1, 2, 2)) == pytest.approx(0.0)
    assert tp.ring_intersection_area(shape_l, square(0, 0, 2, 2)) == pytest.approx(3.0)


def test_the_answer_survives_projected_coordinate_magnitudes():
    """The reference drawing sits at E 691,942 / N 2,750,756.

    Every cross product there is a difference of numbers around 1e6, and an
    engine that did not translate first would lose the sliver it is looking for
    in the rounding. Both rings are moved to a shared local origin before a
    single product is taken.
    """
    ox, oy = 691942.0, 2750756.0
    a = square(ox, oy, 25, 12)
    overlapping = square(ox + 24, oy, 25, 12)
    touching = square(ox + 25, oy, 25, 12)
    assert tp.ring_intersection_area(a, overlapping) == pytest.approx(12.0, abs=1e-6)
    assert tp.ring_intersection_area(a, touching) == 0.0


def test_a_pair_too_large_to_measure_is_withheld_and_never_returned_as_zero():
    """G8. Zero means "checked, they share nothing"; None means "not checked".
    Collapsing the two is how an unmeasured pair becomes a clean bill of
    health."""
    many = [
        (math.cos(i / 400.0) * 10.0, math.sin(i / 400.0) * 10.0)
        for i in range(int(math.sqrt(tp.MAX_VERTEX_PAIRS_PER_PAIR)) + 5)
    ]
    assert len(many) * len(many) > tp.MAX_VERTEX_PAIRS_PER_PAIR
    assert tp.ring_intersection_area(many, many) is None


def test_the_pair_tolerance_scales_with_the_parcels_and_carries_no_unit():
    """The same fraction on a small plot and on a school gives two different
    absolute tolerances — which is the point. A tolerance written as a number of
    drawing units would mean one thing in metres and another in inches."""
    small = tp._pair_tolerance(300.0, 400.0, 30.0, 0.001)
    large = tp._pair_tolerance(30000.0, 40000.0, 300.0, 0.001)
    assert small == pytest.approx(0.3)
    assert large == pytest.approx(30.0)
    # The numerical floor binds only where the fraction term collapses.
    assert tp._pair_tolerance(0.0, 0.0, 1000.0, 0.001) == pytest.approx(
        tp.NUMERICAL_FLOOR_REL * 1e6
    )


def test_box_distance_is_zero_for_boxes_that_touch():
    assert tp._box_distance((0, 0, 1, 1), (1, 0, 2, 1)) == 0.0
    assert tp._box_distance((0, 0, 1, 1), (2, 0, 3, 1)) == pytest.approx(1.0)
    assert tp._box_distance((0, 0, 1, 1), (2, 2, 3, 3)) == pytest.approx(math.sqrt(2))


def test_the_bbox_prefilter_agrees_with_the_full_product():
    """An index that changes the answer is not an optimisation but a bug.

    `store_spatial._SpatialIndex` is reused rather than re-implemented, and what
    is checked here is the thing that could go wrong in the REUSE: the box
    expansion that lets a point-in-box index answer a box-overlap question must
    never drop a pair.
    """
    rings = []
    for i in range(40):
        w = 1.0 + (i % 7)
        h = 1.0 + (i % 3)
        rings.append(
            {
                "_box": (i * 1.7, (i % 5) * 2.3, i * 1.7 + w, (i % 5) * 2.3 + h),
                "handle": str(i),
            }
        )
    pairs, proposed = tp._self_pairs(rings)
    brute = [
        (i, j)
        for i in range(len(rings))
        for j in range(i + 1, len(rings))
        if tp._boxes_overlap(rings[i]["_box"], rings[j]["_box"])
    ]
    assert sorted(pairs) == sorted(brute)
    assert proposed >= len(brute)


def test_the_cross_prefilter_agrees_with_the_full_product_at_a_distance():
    queries = [{"_box": (i * 3.0, 0.0, i * 3.0 + 1.0, 1.0)} for i in range(20)]
    targets = [{"_box": (j * 2.0, 4.0, j * 2.0 + 0.5, 4.5)} for j in range(30)]
    pad = 5.0
    near, _ = tp._cross_pairs(queries, targets, pad=pad)
    brute = {
        q: sorted(
            t
            for t in range(len(targets))
            if tp._box_distance(queries[q]["_box"], targets[t]["_box"]) <= pad
        )
        for q in range(len(queries))
    }
    brute = {q: v for q, v in brute.items() if v}
    assert {q: sorted(v) for q, v in near.items()} == brute


# =============================================================================
# 2. overlap_scan over a fixture store whose answers are known
# =============================================================================


def test_overlap_scan_finds_the_planted_overlap_and_only_that_one(fixture_store):
    r = _run("overlap_scan", {"layers": [PLOTS]})

    assert r["overlaps"]["count"] == 1, r["overlaps"]["rows"]
    row = r["overlaps"]["rows"][0]
    assert sorted(row["handles"]) == ["P-OVERLAP-A", "P-OVERLAP-B"]
    assert row["shared_area"]["value"] == pytest.approx(50.0)
    assert row["shared_area"]["unit"] == "m2"
    assert row["fraction_of_smaller"] == pytest.approx(0.5)
    assert row["layers"] == [PLOTS, PLOTS]


def test_the_edge_sharing_pair_is_not_an_overlap_and_is_counted_as_meeting(
    fixture_store,
):
    """The false-positive guard, at recipe level.

    `P-EDGE-A` and `P-EDGE-B` share a whole 10-unit edge. They are not in the
    overlap list, they are not in the sliver list, and they are not silently
    absent either: they are counted as a pair that MEETS.
    """
    r = _run("overlap_scan", {"layers": [PLOTS]})
    flagged = {h for row in r["overlaps"]["rows"] for h in row["handles"]}
    slivered = {h for row in r["slivers_below_tolerance"]["rows"] for h in row["handles"]}
    assert "P-EDGE-A" not in flagged and "P-EDGE-B" not in flagged
    assert "P-EDGE-A" not in slivered and "P-EDGE-B" not in slivered
    assert r["pairs_meeting_with_zero_shared_area"] >= 1


def test_a_sliver_is_reported_as_a_sliver_rather_than_dropped(fixture_store):
    r = _run("overlap_scan", {"layers": [PLOTS]})
    rows = r["slivers_below_tolerance"]["rows"]
    assert r["slivers_below_tolerance"]["count"] == 1, rows
    assert sorted(rows[0]["handles"]) == ["P-SLIVER-A", "P-SLIVER-B"]
    assert rows[0]["shared_area"]["value"] == pytest.approx(0.1, abs=1e-6)
    # Below the tolerance for its own pair, which is what put it here.
    assert rows[0]["shared_area"]["value"] < rows[0]["tolerance_for_this_pair"]["value"]


def test_an_overlap_says_WHICH_overlap_it_is(fixture_store, monkeypatch):
    """Three different defects wear the same word.

    Two rings covering the SAME ground is a parcel drawn twice, and every area
    total that adds both layers counts that ground twice. One ring inside
    another is a nested allocation. Rings that cross are a boundary in the wrong
    place. A row that only said "overlap" would leave the reader to open the
    drawing to find out which.
    """
    coincident = [
        ring_doc("C-1", PLOTS, square(0, 0, 10, 10)),
        ring_doc("C-2", CORRIDOR, square(0, 0, 10, 10)),
        ring_doc("N-OUTER", PLOTS, square(50, 0, 40, 40)),
        ring_doc("N-INNER", CORRIDOR, square(55, 5, 10, 10)),
    ]
    entities = FakeCollection(coincident)
    monkeypatch.setattr(tp, "_entities", lambda: entities)
    r = _run("overlap_scan", {"layers": [PLOTS, CORRIDOR]})

    rows = {tuple(sorted(row["handles"])): row for row in r["overlaps"]["rows"]}
    assert rows[("C-1", "C-2")]["relationship"].startswith("coincident")
    assert "twice" in rows[("C-1", "C-2")]["relationship"]
    assert rows[("N-INNER", "N-OUTER")]["relationship"].startswith("nested")
    assert r["overlaps"]["by_relationship"] == {"coincident": 1, "nested": 1}


def test_a_sliver_the_size_of_the_arithmetic_is_told_apart_from_a_drafting_one(
    fixture_store,
):
    """Otherwise a tidy master plan returns thousands of findings nobody can act
    on. Coordinates are doubles: two rings snapped to the same edge still differ
    in their last bits, and the area that produces is the resolution of the
    numbers, not a defect in the drawing."""
    r = _run("overlap_scan", {"layers": [PLOTS]})
    block = r["slivers_below_tolerance"]
    assert block["below_the_arithmetic_floor"] + block["above_the_arithmetic_floor"] == (
        block["count"]
    )
    # The planted 0.1 sliver is real drafting, orders of magnitude above the
    # floor, so it must be on the actionable side of that split.
    assert block["above_the_arithmetic_floor"] == 1
    assert block["rows"][0]["below_the_arithmetic_floor"] is False


def test_the_tolerance_decides_and_moving_it_moves_the_answer(fixture_store):
    """If the tolerance did not change the answer it would not be worth
    publishing. Tightened by two orders of magnitude, the sliver becomes an
    overlap."""
    loose = _run("overlap_scan", {"layers": [PLOTS]})
    tight = _run("overlap_scan", {"layers": [PLOTS], "min_overlap_fraction": 0.00001})
    assert loose["overlaps"]["count"] == 1
    assert tight["overlaps"]["count"] == 2
    assert tight["slivers_below_tolerance"]["count"] == 0
    # And the edge-sharing pair stays out of it at ANY tolerance: zero is zero.
    flagged = {h for row in tight["overlaps"]["rows"] for h in row["handles"]}
    assert "P-EDGE-A" not in flagged


def test_the_tolerance_is_published_with_its_basis_and_its_absolute_value(
    fixture_store,
):
    r = _run("overlap_scan", {"layers": [PLOTS]})
    tol = r["tolerance"]
    assert tol["min_overlap_fraction"] == tp.DEFAULT_MIN_OVERLAP_FRACTION
    assert tol["supplied_by_caller"] is False
    assert "sharing an edge" in tol["rule"]
    assert "smaller" in tol["per_pair_formula"]
    assert "G2" in tol["basis"] and "G1" in tol["basis"]
    assert "decides" in tol
    # The fraction is dimensionless; the value it TOOK on this drawing is not,
    # and it comes back with its unit.
    assert tol["representative_value"]["value"] == pytest.approx(0.1)
    assert tol["representative_value"]["unit"] == "m2"


def test_an_unmeasurable_ring_is_counted_and_named_never_treated_as_zero(
    fixture_store,
):
    """G8. `P-BULGE` cannot take part in an area test. It is not a parcel that
    overlaps nothing — it is a parcel nobody could check, and the two must never
    arrive as the same answer."""
    r = _run("overlap_scan", {"layers": [PLOTS]})
    census = r["ring_census"]
    assert census["skipped_by_cause"]["bulge"] == 1
    assert "P-BULGE" in census["skipped_examples"]["bulge"]
    assert census["rings_usable"] == 6
    assert r["rings_scanned"] == 6
    flagged = {h for row in r["overlaps"]["rows"] for h in row["handles"]}
    assert "P-BULGE" not in flagged


def test_without_a_boundary_the_gap_is_not_computed_and_says_why(fixture_store):
    """The instruction that matters most in the gap half: a site boundary is
    NEVER approximated from the parcels. Doing so makes the gap zero by
    construction and answers a question nobody asked."""
    r = _run("overlap_scan", {"layers": [PLOTS]})
    gap = r["gap"]
    assert gap["computed"] is False
    assert gap["gap_area"] is None
    assert "boundary" in gap["why_not_computed"].lower()
    assert "never" in gap["never_approximated"].lower()


def test_with_a_boundary_the_gap_is_the_ground_nothing_claims(fixture_store):
    """Known by hand: a 100x100 boundary is 10,000; six measurable parcels of
    100 each are 600; the one real overlap of 50 is counted once, not twice. So
    the covered ground is 550 and the gap is 9,450."""
    r = _run("overlap_scan", {"layers": [PLOTS], "boundary_layer": BOUNDARY})
    gap = r["gap"]
    assert gap["computed"] is True
    assert gap["boundary_rings"] == 1
    assert gap["boundary_area"]["value"] == pytest.approx(10000.0)
    assert gap["covered_area"]["value"] == pytest.approx(550.0, abs=1e-3)
    assert gap["gap_area"]["value"] == pytest.approx(9450.0, abs=1e-3)
    assert gap["gap_fraction"] == pytest.approx(0.945, abs=1e-5)
    assert gap["parcels_inside_boundary"] == 6
    assert gap["gap_area"]["unit"] == "m2"
    # The limits of the arithmetic travel with the number, not in a doc.
    assert "overlap_correction" in gap["limits"]
    assert "inclusion_exclusion_order" in gap["limits"]
    assert "layer left out of the scan" in gap["what_counts_as_covered"]


def test_a_boundary_layer_with_no_usable_ring_says_so_rather_than_returning_zero(
    monkeypatch,
):
    docs = [
        ring_doc("P-1", PLOTS, square(0, 0, 10, 10)),
        ring_doc("B-BULGE", BOUNDARY, square(0, 0, 100, 100), status="bulge"),
    ]
    entities = FakeCollection(docs)
    monkeypatch.setattr(tp, "_entities", lambda: entities)
    r = _run("overlap_scan", {"layers": [PLOTS], "boundary_layer": BOUNDARY})
    assert r["gap"]["computed"] is False
    assert r["gap"]["gap_area"] is None
    assert "bulge" in r["gap"]["why_not_computed"]


def test_overlap_scan_on_a_drawing_with_nothing_to_scan_answers_rather_than_breaks(
    monkeypatch,
):
    """G8/G3: no data is a normal state. It is an answer with a denominator of
    zero, not an exception and not a fabricated clean bill of health."""
    entities = FakeCollection([])
    monkeypatch.setattr(tp, "_entities", lambda: entities)
    r = _run("overlap_scan", {"layers": [PLOTS]})
    assert r["rings_scanned"] == 0
    assert r["overlaps"]["count"] == 0
    assert r["overlaps"]["shared_area_total"]["value"] is None
    assert r["overlaps"]["shared_area_total"]["withheld_reason"]
    assert r["tolerance"]["representative_value"]["value"] is None


@pytest.mark.parametrize("units", ALL_UNIT_SHAPES)
def test_overlap_scan_never_claims_a_unit_the_drawing_does_not_declare(
    fixture_store, units
):
    """G2. 11 of 18 drawings are in inches and 3 declare nothing at all."""
    r = _run("overlap_scan", {"layers": [PLOTS], "boundary_layer": BOUNDARY}, units=units)
    row = r["overlaps"]["rows"][0]
    assert row["shared_area"]["unit"] == units["area_unit"]
    assert r["gap"]["gap_area"]["unit"] == units["area_unit"]
    if units["declared_in_file"] is False:
        assert row["shared_area"]["unit_reason"]
        assert '"unit": "m2"' not in json.dumps(r, default=str)
    assert ev.audit_response(r) == []


def test_overlap_scan_refuses_a_fraction_that_is_not_a_fraction(fixture_store):
    for bad in (1.0, 2.5, -0.1):
        with pytest.raises(recipes.RecipeRefused) as excinfo:
            _run("overlap_scan", {"layers": [PLOTS], "min_overlap_fraction": bad})
        assert excinfo.value.code == "RECIPE_PARAM_INVALID"
        assert "fraction" in excinfo.value.hint.lower()


# =============================================================================
# 3. frontage_check over a fixture store whose answers are known
# =============================================================================


@pytest.fixture
def frontage_store(monkeypatch):
    """Three parcels of 100 units² each, and one road drawn as a LINE.

    * `F-TOUCH` (0,0)-(10,10) sits 0.05 below the road. It has frontage.
    * `F-NEAR` (0,15)-(10,25) sits 4.95 above the road — inside the search
      window, outside the touch tolerance. Named, with its measured distance.
    * `F-LANDLOCKED` (0,-500)-(10,-490) is nowhere near anything. Named, with
      its distance WITHHELD rather than reported as infinity.

    The typical parcel size is sqrt(100) = 10, so the derived touch tolerance is
    0.1 and the search window is 30.
    """
    docs = [
        ring_doc("F-TOUCH", PLOTS, square(0, 0, 10, 10)),
        ring_doc("F-NEAR", PLOTS, square(0, 15, 10, 10)),
        ring_doc("F-LANDLOCKED", PLOTS, square(0, -500, 10, 10)),
        line_doc("ROAD-1", ROADS, (0.0, 10.05), (100.0, 10.05)),
        ring_doc("CORRIDOR-1", CORRIDOR, square(-5, 10.05, 110, 4)),
    ]
    entities = FakeCollection(docs)
    monkeypatch.setattr(tp, "_entities", lambda: entities)
    return entities


def test_frontage_names_the_landlocked_parcels_rather_than_counting_them(
    frontage_store,
):
    """A count is not actionable; a handle is clickable."""
    r = _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})

    assert r["parcels_tested"] == 3
    named = {row["handle"] for row in r["no_frontage"]["rows"]}
    assert named == {"F-NEAR", "F-LANDLOCKED"}
    assert r["no_frontage"]["count"] == 2
    for row in r["no_frontage"]["rows"]:
        assert row["layer"] == PLOTS
        assert row["handle"]


def test_frontage_gives_three_answers_and_not_two(frontage_store):
    """Established, NOT established, and none — and every parcel lands in
    exactly one of them.

    The middle answer is the reason this recipe can be trusted at all. A road
    with no ring is measured against its bounding box, and a box that comes
    inside the tolerance proves nothing: the box contains the road, so the real
    distance can only be larger. Collapsing that into "has frontage" would
    publish a pass nobody measured.
    """
    r = _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})
    assert r["frontage_established"] == 0
    assert r["frontage_unproven"]["count"] == 1
    assert r["no_frontage"]["count"] == 2
    assert (
        r["frontage_established"]
        + r["frontage_unproven"]["count"]
        + r["no_frontage"]["count"]
        == r["parcels_tested"]
    )
    assert r["frontage_unproven"]["rows"][0]["handle"] == "F-TOUCH"
    assert r["frontage_unproven"]["rows"][0]["measured_from"] == "bounding_box"
    assert "neither" in r["frontage_unproven"]["rows"][0]["verdict"]


def test_the_landlocked_list_is_sound_and_says_why(frontage_store):
    """The direction of the approximation is the whole argument: a bounding box
    contains its entity, so a measured distance is a LOWER bound. That cannot
    put a parcel on this list wrongly — only keep one off it."""
    r = _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})
    why = r["no_frontage"]["why_this_list_is_sound"]
    assert "LOWER bound" in why
    assert "F-TOUCH" not in {row["handle"] for row in r["no_frontage"]["rows"]}


def test_a_parcel_with_no_road_inside_the_window_withholds_its_distance(
    frontage_store,
):
    """G8 again, in the shape it takes here: "no road was found within the
    window" is not "the distance is zero" and is not "the distance is
    infinite"."""
    r = _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})
    rows = {row["handle"]: row for row in r["no_frontage"]["rows"]}
    far = rows["F-LANDLOCKED"]
    assert far["nearest_road"]["value"] is None
    assert far["nearest_road"]["withheld_reason"]
    assert far["nearest_road_handle"] is None
    assert r["no_frontage"]["beyond_the_search_window"] == 1

    near = rows["F-NEAR"]
    assert near["nearest_road"]["value"] == pytest.approx(4.95, abs=1e-6)
    assert near["nearest_road_handle"] == "ROAD-1"
    assert near["measured_from"] == "bounding_box"


def test_the_touch_tolerance_is_derived_from_the_drawings_own_parcels(frontage_store):
    r = _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})
    tol = r["tolerance"]
    assert tol["supplied_by_caller"] is False
    assert tol["touch_ratio"] == tp.DEFAULT_TOUCH_REL
    assert tol["typical_parcel_size"]["value"] == pytest.approx(10.0)
    assert tol["value"]["value"] == pytest.approx(0.1)
    assert tol["value"]["unit"] == "m"
    assert "not_derived_from_the_answer" in tol
    assert "tautology" in tol["not_derived_from_the_answer"]
    assert r["search_window"]["value"]["value"] == pytest.approx(30.0)


def test_the_tolerance_decides_the_frontage_answer_too(frontage_store):
    """Given a tolerance of five units — about half a carriageway — the parcel
    across the road gains frontage. Same drawing, same road, different
    tolerance: which is exactly why the tolerance is published."""
    tight = _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})
    loose = _run(
        "frontage_check",
        {"parcel_layers": [PLOTS], "road_layers": [ROADS], "touch_tolerance": 5.0},
    )
    assert tight["no_frontage"]["count"] == 2
    assert loose["no_frontage"]["count"] == 1
    assert loose["tolerance"]["supplied_by_caller"] is True
    assert loose["tolerance"]["value"]["value"] == 5.0
    assert {row["handle"] for row in loose["no_frontage"]["rows"]} == {"F-LANDLOCKED"}


def test_a_road_drawn_as_a_ring_settles_the_answer_instead_of_leaving_it_open(
    frontage_store,
):
    """A corridor polygon carries a ring, so its distance is exact and the pass
    is PROVEN. Same parcel, same tolerance, better geometry — and the answer
    moves out of `frontage_unproven` and into `frontage_established`."""
    box = _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})
    ring = _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [CORRIDOR]})

    assert box["frontage_established"] == 0
    assert box["frontage_unproven"]["count"] == 1

    assert ring["road_census"]["measured_from_their_ring"] == 1
    assert ring["road_census"]["measured_from_their_bounding_box"] == 0
    assert ring["frontage_established"] == 1
    assert ring["frontage_unproven"]["count"] == 0


def test_the_bounding_box_proxy_states_the_direction_of_its_error(frontage_store):
    """The honest half of 5b. A bounding box CONTAINS its line, so the measured
    distance is a LOWER bound: the landlocked list is sound, the frontage list
    is not. Saying which way an approximation fails is what makes it usable."""
    r = _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})
    note = r["road_census"]["bounding_box_note"]
    assert "LOWER bound" in note
    assert "landlocked" in note
    assert r["road_census"]["measured_from_their_bounding_box"] == 1


def test_a_centreline_shaped_answer_is_flagged_rather_than_reported_as_a_defect(
    frontage_store,
):
    """Most parcels failing while the median clearance sits well above the
    tolerance is the signature of road geometry drawn AWAY from the boundary,
    not of a scheme without access."""
    r = _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})
    assert r["road_geometry_warning"] is not None
    assert "centreline" in r["road_geometry_warning"]
    assert r["clearance"]["median"]["value"] == pytest.approx(4.95, abs=1e-6)
    assert r["clearance"]["measured"] == 2
    assert r["clearance"]["not_measured"] == 1


def test_road_layers_are_never_derived_from_a_layer_name(monkeypatch, frontage_store):
    """G1 and the `C-ROAD-*` trap. With no config, no Dossier and no parameter,
    the recipe REFUSES — it does not fall back to looking for the word "road" in
    a layer name, which would report frontage onto a Civil3D title block."""
    monkeypatch.setattr(
        tp, "_dossier_network_layers", lambda _d, _l: ((), "no Dossier in this test")
    )
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        _run("frontage_check", {"parcel_layers": [PLOTS]})
    assert excinfo.value.code == "RECIPE_NO_CONFIG"
    assert "layer names" in excinfo.value.hint
    assert "role: network" in excinfo.value.hint


def test_road_layers_arrive_through_the_dossier_geometric_role(
    monkeypatch, frontage_store
):
    monkeypatch.setattr(
        tp, "_dossier_network_layers", lambda _d, _l: ((ROADS,), None)
    )
    r = _run("frontage_check", {"parcel_layers": [PLOTS]})
    assert r["road_layers"] == [ROADS]
    assert "geometric role" in r["road_layers_source"]
    assert "never from a layer name" in r["road_layers_never_by_name"]


def test_frontage_refuses_rather_than_inventing_a_tolerance_when_nothing_is_measurable(
    monkeypatch,
):
    docs = [
        ring_doc("P-BULGE", PLOTS, square(0, 0, 10, 10), status="bulge"),
        line_doc("ROAD-1", ROADS, (0.0, 10.05), (100.0, 10.05)),
    ]
    entities = FakeCollection(docs)
    monkeypatch.setattr(tp, "_entities", lambda: entities)
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})
    assert excinfo.value.code == "RECIPE_PARAM_INVALID"
    assert "touch_tolerance" in excinfo.value.hint


@pytest.mark.parametrize("units", ALL_UNIT_SHAPES)
def test_frontage_never_claims_a_unit_the_drawing_does_not_declare(
    frontage_store, units
):
    r = _run(
        "frontage_check",
        {"parcel_layers": [PLOTS], "road_layers": [ROADS]},
        units=units,
    )
    assert r["tolerance"]["value"]["unit"] == units["length_unit"]
    assert r["clearance"]["median"]["unit"] == units["length_unit"]
    if units["declared_in_file"] is False:
        assert r["tolerance"]["value"]["unit_reason"]
        assert '"unit": "m"' not in json.dumps(r, default=str)
    assert ev.audit_response(r) == []


# =============================================================================
# 4. The contract both recipes sign
# =============================================================================


TOPOLOGY_RECIPES = ("frontage_check", "overlap_scan")


def test_both_recipes_are_in_the_catalogue_with_everything_the_model_reads():
    catalogue = {row["recipe"]: row for row in recipes.catalog()["recipes"]}
    for name in TOPOLOGY_RECIPES:
        assert name in catalogue, "the recipe is registered by importing its module"
        row = catalogue[name]
        assert row["answers"].strip()
        assert row["when_to_use"].strip()
        assert row["returns"]
        assert row["built_on"], "where the number comes from, published"
        assert row["limits"], "G7: limits are stated, not assumed"
        for param in row["params"]:
            assert param["about"].strip(), (name, param["name"])


def test_both_recipes_carry_evidence_at_the_grade_the_geometry_earns(fixture_store):
    """A measurement from coordinates INFERS; it never STATES.

    A ring sharing area with another ring is a fact about the drawing. That two
    parcels were sold twice is a conclusion, and the file says nothing about it
    — so `inferred` is the correct ceiling, not a modest one.
    """
    for name, params in (
        ("overlap_scan", {"layers": [PLOTS]}),
        ("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]}),
    ):
        r = _run(name, params)
        block = r["evidence"]
        assert block["grade"] == "inferred"
        assert block["not_established"].strip()
        assert block["how_to_verify"].strip()
        assert "geometry" in {s["origin"] for s in block["sources"]}
        assert block["scope"]["note"] == r["scope_note"]
        assert ev.audit_response(r) == []


def test_both_recipes_state_every_limit_they_apply(fixture_store):
    for name, params in (
        ("overlap_scan", {"layers": [PLOTS]}),
        ("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]}),
    ):
        r = _run(name, params)
        assert r["limits"], "the recipe's declared limits travel in the envelope"
        assert r["limits_applied"], "and the ones this run really applied"
        assert r["pairs_after_prefilter"] is not None
        assert r["pairs_proposed_by_prefilter"] >= r["pairs_after_prefilter"]


def test_both_recipes_are_deterministic(fixture_store):
    """The same input produces the same output, forever. Two sessions can only
    compare their numbers if that holds."""
    for name, params in (
        ("overlap_scan", {"layers": [PLOTS], "boundary_layer": BOUNDARY}),
        ("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]}),
    ):
        first = json.dumps(_run(name, params), sort_keys=True, default=str)
        second = json.dumps(_run(name, params), sort_keys=True, default=str)
        assert first == second, name


def test_the_module_reaches_neither_the_clock_nor_the_network():
    """Tested as a property of the FILE, not relied on as the author's intent."""
    source = pathlib.Path(tp.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import time",
        "import random",
        "import requests",
        "import urllib",
        "import socket",
        "datetime",
        "random.",
    ):
        assert forbidden not in source, forbidden


def test_no_drawing_specific_constant_lives_in_this_module():
    """G1. Both recipes must work on the 19th drawing, whose layer convention
    nobody has seen. A layer name here would mean they do not."""
    source = pathlib.Path(tp.__file__).read_text(encoding="utf-8")
    for token in (
        "VL2",
        "DP4",
        "LP1",
        "TH3",
        "Primary School",
        "BlockBoundary",
        "New Boundry",
        "C-PROP-PlotNumber",
        "00_Prop",
        # A whole word, case-sensitive: `MAX_ROWS_RETURNED` is not the
        # Janadriyah right-of-way layer, and a substring test that could not
        # tell the two apart would be a gate nobody trusts.
        r"\bROW\b",
        "32638",
    ):
        assert re.search(token, source) is None, token


def test_no_tolerance_in_this_module_is_written_as_a_length_or_an_area():
    """The rule that keeps 5a and 5b working in inches.

    Every tolerance constant is a RATIO. A number of drawing units here would be
    right for one drawing in metres and wrong for the eleven in inches and the
    three that declare nothing (G2).
    """
    for ratio in (
        tp.DEFAULT_MIN_OVERLAP_FRACTION,
        tp.NUMERICAL_FLOOR_REL,
        tp.DEFAULT_TOUCH_REL,
        tp.DEFAULT_SEARCH_REL,
    ):
        assert 0 < ratio <= 10, ratio
    assert not any(
        name.endswith(("_M", "_MM", "_METRES", "_INCHES"))
        for name in dir(tp)
    )


def test_an_unknown_parameter_is_refused_rather_than_ignored(fixture_store):
    """A misspelled parameter silently dropped answers a different question
    from the one that was typed, and nothing in the response says so."""
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        _run("overlap_scan", {"layers": [PLOTS], "tolerance_m": 0.5})
    assert excinfo.value.code == "RECIPE_PARAM_UNKNOWN"


def test_both_recipes_refuse_to_run_without_a_layout(fixture_store):
    """Modelspace and paper sheets do not share a coordinate frame, so there is
    no right answer for "all of it"."""
    for name, params in (
        ("overlap_scan", {"layers": [PLOTS]}),
        ("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]}),
    ):
        with pytest.raises(recipes.RecipeRefused) as excinfo:
            recipes.run(FIXTURE, name, params, layout=None, units=METRE_UNITS)
        assert excinfo.value.code == "RECIPE_LAYOUT_REQUIRED"


def test_the_scan_budgets_WORK_and_not_only_the_number_of_pairs(monkeypatch):
    """The limit that a pair count cannot express.

    Measured on the reference drawing: 7,402 pairs of small plots cost 224,748
    vertex pairs and seconds; the same drawing with a road-corridor layer added
    costs 15,736,813 and took 173 seconds. The pair count barely moved. So the
    budget is counted in vertex pairs, and it is counted BEFORE a single
    triangle is clipped — otherwise the work it exists to refuse has already
    been done by the time anyone hears about it.
    """
    docs = [ring_doc(f"P-{i}", PLOTS, square(i * 5, 0, 10, 10)) for i in range(6)]
    entities = FakeCollection(docs)
    monkeypatch.setattr(tp, "_entities", lambda: entities)

    normal = _run("overlap_scan", {"layers": [PLOTS]})
    assert normal["vertex_pairs_clipped"] > 0
    assert normal["vertex_pairs_budget"] == tp.MAX_VERTEX_PAIRS_TOTAL

    monkeypatch.setattr(tp, "MAX_VERTEX_PAIRS_TOTAL", 4)
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        _run("overlap_scan", {"layers": [PLOTS]})
    assert excinfo.value.code == "RECIPE_INPUT_TOO_LARGE"
    assert "limit on WORK" in excinfo.value.hint
    assert "Split the scan" in excinfo.value.hint


def test_the_frontage_check_budgets_work_on_its_own_larger_ceiling(
    monkeypatch, frontage_store
):
    """Two budgets, because the two algorithms differ where it matters.

    The nearest-road search stops as soon as no remaining candidate can beat the
    best distance found, so its swept worst case badly over-states the work it
    really does. Ring clipping has no such exit. Giving both the same ceiling
    would refuse the one query on the reference drawing that produces a PROVEN
    frontage answer.
    """
    assert tp.MAX_FRONTAGE_VERTEX_PAIRS > tp.MAX_VERTEX_PAIRS_TOTAL
    monkeypatch.setattr(tp, "MAX_FRONTAGE_VERTEX_PAIRS", 1)
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        _run("frontage_check", {"parcel_layers": [PLOTS], "road_layers": [ROADS]})
    assert excinfo.value.code == "RECIPE_INPUT_TOO_LARGE"
    assert "limit on WORK" in excinfo.value.hint


def test_the_scan_refuses_a_population_above_its_stated_limit(monkeypatch):
    """G7: the limit is checked BEFORE the work, so it is a limit on work and
    not a report after the work has already been done."""
    monkeypatch.setattr(tp, "MAX_SCAN_RINGS", 3)
    docs = [
        ring_doc(f"P-{i}", PLOTS, square(i * 20, 0, 10, 10)) for i in range(5)
    ]
    entities = FakeCollection(docs)
    monkeypatch.setattr(tp, "_entities", lambda: entities)
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        _run("overlap_scan", {"layers": [PLOTS]})
    assert excinfo.value.code == "RECIPE_INPUT_TOO_LARGE"
    assert "3" in excinfo.value.message


# =============================================================================
# 5. Janadriyah — the real drawing, recorded either way
# =============================================================================


def _live_or_skip():
    """The reference drawing from the store, or skip.

    Rewritten here rather than imported from another lane's test file: a shared
    helper living in a file owned by someone else is a dependency that is
    invisible until that file moves.
    """
    from app import store as live_store

    try:
        drawing = live_store.get_drawing(DRAWING)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip(f"reference drawing {DRAWING} has not been ingested")
    return live_store, drawing


#: A pairwise scan over 2,545 rings costs seconds, and several tests below read
#: the SAME answer from different angles. They are cached rather than re-run, so
#: that adding a question about the reference drawing costs nothing. The
#: determinism test deliberately calls `_live_uncached` — a cache would make it
#: assert that a dict equals itself.
_LIVE_CACHE: dict[tuple, dict] = {}


def _live_uncached(name, params):
    live_store, drawing = _live_or_skip()
    return recipes.run(
        DRAWING,
        name,
        params,
        layout=LAYOUT,
        units=live_store._unit_names(drawing, LAYOUT),
    )


def _live(name, params):
    key = (name, json.dumps(params, sort_keys=True))
    if key not in _LIVE_CACHE:
        _LIVE_CACHE[key] = _live_uncached(name, params)
    return _LIVE_CACHE[key]


def test_janadriyah_overlap_scan_runs_over_the_config_derived_parcels():
    """The acceptance run. Whatever it returns is the finding — a clean drawing
    is a result, not a failure — so what is asserted here is the SHAPE of a
    trustworthy answer, plus the population it really covered."""
    r = _live("overlap_scan", {})

    assert r["rings_scanned"] > 0
    assert "role=parcel" in r["layers_source"]
    assert r["tolerance"]["representative_value"]["unit"] == "m2"
    assert r["pairs_after_prefilter"] < r["rings_scanned"] ** 2, (
        "the bbox prefilter is what makes this tractable at all"
    )
    assert r["gap"]["computed"] is False, (
        "no boundary layer was named, so the gap must come back not-computed "
        "rather than approximated from the parcels"
    )
    assert ev.audit_response(r) == []
    # Every overlap that IS reported is actionable: two handles, an area, and
    # the fraction of the smaller parcel it takes.
    for row in r["overlaps"]["rows"]:
        assert len(row["handles"]) == 2 and all(row["handles"])
        assert row["shared_area"]["value"] > 0
        assert row["shared_area"]["unit"] == "m2"
        assert 0 < row["fraction_of_smaller"] <= 1.0000001


#: Measured 25 August 2026 over the 49 layers this drawing's config gives
#: role=parcel, on layout Model. Every one of the 23 pairs is COINCIDENT — the
#: same ground drawn twice on two layers — which is why the ground-truth number
#: is a total area and not a count of slivers.
GT_OVERLAP_PAIRS = 23
GT_OVERLAP_AREA = 29241.257984789187


def test_janadriyah_has_twenty_three_parcels_drawn_twice():
    """The acceptance number for 5a, and a real defect in the reference drawing.

    Every overlap this drawing carries is a parcel drawn on two layers at once:
    eight school and early-education parcels also exist as `Open Spaces`
    polygons, and fifteen `Linear Park` polygons do too. 29,241.26 m² of ground
    is therefore claimed twice, and every land-use total that adds both layers
    adds that ground twice with it.

    Not one of them is a partial overlap or a sliver — which is the signature of
    a copy, not of a boundary in the wrong place, and is why the relationship
    travels on every row.
    """
    r = _live("overlap_scan", {})
    assert r["overlaps"]["count"] == GT_OVERLAP_PAIRS
    assert r["overlaps"]["shared_area_total"]["value"] == pytest.approx(
        GT_OVERLAP_AREA, abs=1e-6
    )
    assert r["overlaps"]["shared_area_total"]["unit"] == "m2"
    assert r["overlaps"]["by_relationship"] == {"coincident": GT_OVERLAP_PAIRS}
    for row in r["overlaps"]["rows"]:
        assert row["fraction_of_smaller"] == pytest.approx(1.0, abs=1e-6)
        assert row["layers"][0] != row["layers"][1], (
            "the same ground on two different layers is what makes it a copy"
        )


def test_janadriyah_slivers_are_mostly_the_arithmetic_and_the_answer_says_so():
    """2,479 pairs share a trace of ground and 2,477 of them are below the
    resolution of the coordinates themselves.

    Reporting all 2,479 as findings would bury the two that a drafter could act
    on. Reporting none of them would hide those two. The split is what makes the
    number usable.
    """
    r = _live("overlap_scan", {})
    block = r["slivers_below_tolerance"]
    assert block["count"] > 1000
    assert block["above_the_arithmetic_floor"] < 10
    assert (
        block["below_the_arithmetic_floor"] + block["above_the_arithmetic_floor"]
        == block["count"]
    )


def test_janadriyah_parcels_tile_their_blocks_rather_than_overlapping_them():
    """The known ground truth this can be checked against: 2,380 plots fill 197
    block boundaries with a difference of 0.038 m² over the whole drawing. A
    tiling that exact must produce edge contacts, not overlaps — so the number
    of pairs that MEET at exactly zero shared area must dominate."""
    r = _live("overlap_scan", {})
    assert r["pairs_meeting_with_zero_shared_area"] > r["overlaps"]["count"], (
        f"overlaps={r['overlaps']['count']} "
        f"meeting={r['pairs_meeting_with_zero_shared_area']}"
    )


def test_janadriyah_frontage_check_names_what_it_finds():
    """5b on the real drawing. Its road layers arrive from the config and the
    Dossier, never from a name, and every parcel it fails is named."""
    r = _live("frontage_check", {})

    assert r["road_layers"], "some road layer arrived through config or Dossier"
    assert r["parcels_tested"] > 0
    assert r["tolerance"]["value"]["unit"] == "m"
    assert r["tolerance"]["value"]["value"] > 0
    assert (
        r["frontage_established"]
        + r["frontage_unproven"]["count"]
        + r["no_frontage"]["count"]
        == r["parcels_tested"]
    ), "every parcel lands in exactly one of the three answers"
    for row in r["no_frontage"]["rows"]:
        assert row["handle"] and row["layer"]
    for row in r["frontage_unproven"]["rows"]:
        assert row["measured_from"] == "bounding_box"
    assert ev.audit_response(r) == []


def test_janadriyah_road_geometry_is_paths_not_rings_and_the_answer_says_so():
    """The measured state of this drawing, recorded as a finding.

    Every layer the Dossier gives the `network` role carries `LINE`, `ARC` and
    open `LWPOLYLINE` rows — paths, not rings — and this store keeps a bounding
    box for those and no vertices. So frontage here cannot be PROVEN for a
    single parcel, and the recipe says that rather than reporting a pass. That
    is the concrete cost of the gap DOSSIER Phase 5c exists to decide about.
    """
    r = _live("frontage_check", {})
    assert r["road_census"]["measured_from_their_ring"] == 0
    assert r["road_census"]["measured_from_their_bounding_box"] > 0
    assert r["frontage_established"] == 0
    assert r["frontage_unproven"]["count"] > 0
    assert "5c" in r["frontage_unproven"]["how_to_settle_it"]


#: Janadriyah's right-of-way layer. It lives in the TEST, never in production
#: `.py` — G1 — and it is named here because it is the one road geometry in this
#: drawing drawn as CLOSED rings, which is what makes an exact answer possible
#: at all.
CORRIDOR_LAYER = "ROW"

#: Two parcels this drawing really does leave without road frontage, measured
#: 25 August 2026 against the right-of-way corridors. Their distances are the
#: ground truth this recipe is checked against.
GT_LANDLOCKED = {
    "205D12B": 24.999827483342056,
    "205D132": 31.78696286973036,
}


def test_janadriyah_names_the_two_parcels_that_really_have_no_frontage():
    """The acceptance number for 5b, and the answer the recipe exists to give.

    Measured against the right-of-way CORRIDORS rather than the centrelines,
    because those are drawn as closed rings — so every distance here is exact
    and every verdict is proven, not inferred from a bounding box.

    Two parcels come out landlocked, and they are NAMED. `205D12B` on the
    private-school layer sits 25.00 m from the nearest corridor and `205D132`
    on the provisional-utility layer sits 31.79 m. A reviewer can click both.
    """
    r = _live(
        "frontage_check",
        {
            "parcel_layers": ["Private School", "Provisional Utility Area"],
            "road_layers": [CORRIDOR_LAYER],
            # Pinned rather than derived, so that the assertion below tests the
            # measurement and not the median parcel area of a three-parcel
            # sample. The derived tolerance is exercised in its own test.
            "touch_tolerance": 0.5,
        },
    )
    assert r["parcels_tested"] == 3
    assert r["frontage_established"] == 1
    assert r["frontage_unproven"]["count"] == 0
    assert r["no_frontage"]["count"] == 2

    found = {row["handle"]: row for row in r["no_frontage"]["rows"]}
    assert set(found) == set(GT_LANDLOCKED)
    for handle, distance in GT_LANDLOCKED.items():
        row = found[handle]
        assert row["nearest_road"]["value"] == pytest.approx(distance, abs=1e-6)
        assert row["nearest_road"]["unit"] == "m"
        assert row["nearest_road_layer"] == CORRIDOR_LAYER
        # Exact, not a bounding-box lower bound — so this verdict is proven.
        assert row["measured_from"] == "ring"
    assert ev.audit_response(r) == []


def test_janadriyah_layer_names_never_decide_which_layer_is_a_road():
    """The `C-ROAD-*` trap, on the drawing that carries it. Whatever layers come
    back, the response must say through which machinery they arrived."""
    r = _live("frontage_check", {})
    assert (
        "config" in r["road_layers_source"] or "Dossier" in r["road_layers_source"]
    ), r["road_layers_source"]


def test_janadriyah_answers_are_the_same_twice():
    """Two genuinely separate runs over the real store, not one cached dict."""
    a = json.dumps(_live_uncached("overlap_scan", {}), sort_keys=True, default=str)
    b = json.dumps(_live_uncached("overlap_scan", {}), sort_keys=True, default=str)
    assert a == b


def _busiest_ring_layers(drawing_id, limit=5):
    """The layers with the most complete rings, chosen from the DATA.

    Not from a name and not from a config, so that this runs on a drawing whose
    layer convention nobody has ever seen — which is the whole point of an
    invariant test (G5). The Janadriyah numbers will never catch a break on the
    19th drawing.
    """
    from app.mongo import COLL_ENTITIES, coll

    rows = coll(COLL_ENTITIES).aggregate(
        [
            {
                "$match": {
                    "drawing_id": drawing_id,
                    "layout": LAYOUT,
                    "ring_status": "complete",
                }
            },
            {"$group": {"_id": "$layer", "n": {"$sum": 1}}},
            {"$sort": {"n": -1}},
            {"$limit": limit},
        ]
    )
    return [r["_id"] for r in rows if r["_id"]]


def test_invariant_both_recipes_hold_over_every_drawing_in_the_store():
    """G5 — run over EVERY drawing, not only the one the numbers came from.

    The store holds 18 drawings: most of them in inches, some declaring no unit
    at all, one in millimetres, and several with no closed ring on modelspace.
    What is asserted is not a value — values differ per drawing, as they should
    — but that every answer is CONTRACT-CLEAN and never claims a unit the
    drawing does not declare. A graceful refusal counts as passing; a stack
    trace does not.
    """
    from app import store as live_store

    try:
        drawings = live_store.list_drawings()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawings:
        pytest.skip("the store holds no drawing")

    answered = 0
    refused = 0
    for drawing in drawings:
        drawing_id = drawing.get("_id") or drawing.get("drawing_id")
        units = live_store._unit_names(drawing, LAYOUT)
        layers = _busiest_ring_layers(drawing_id)
        if not layers:
            continue
        for name, params in (
            ("overlap_scan", {"layers": layers}),
            ("frontage_check", {"parcel_layers": layers}),
        ):
            try:
                r = recipes.run(
                    drawing_id, name, params, layout=LAYOUT, units=dict(units)
                )
            except recipes.RecipeRefused as exc:
                # G3: no config, no Dossier, or above a stated limit — all
                # refusals, all with a hint that can be acted on.
                assert exc.hint.strip(), (drawing_id, name)
                refused += 1
                continue
            answered += 1
            assert ev.audit_response(r) == [], (drawing_id, name)
            assert r["scope_note"].strip(), (drawing_id, name)
            assert r["evidence"]["grade"] in {"inferred", "unknown"}
            if units.get("declared_in_file") is False:
                text = json.dumps(r, default=str)
                for invented in ('"unit": "m"', '"unit": "m2"', '"unit": "in"'):
                    assert invented not in text, (drawing_id, name, invented)
    assert answered > 0, "no drawing produced an answer at all"
    assert answered + refused > 1, "the invariant covered only one drawing"


def test_the_envelope_holds_for_every_shape_of_units():
    """The envelope contract itself, without a database, over units that
    include a drawing declaring none."""
    for units in (METRE_UNITS, INCH_UNITS, UNITLESS):
        scope = rr.scope_for(layout=LAYOUT, units=units, note="a note")
        assert scope.units == units
        assert scope.layout == LAYOUT
