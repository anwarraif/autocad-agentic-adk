"""The `ring_status` gate itself — not its trace in Mongo.

This file exists because of a mutation harness. The bulge gate was broken
deliberately — `if has_bulge:` replaced with `if False:`, so that every arced
polygon passed as a complete ring and its area was reported as the area of its
chords — and **all 648 tests stayed green**.

The cause was not random carelessness, and that is the part worth remembering:
every test that mentions "bulge" in this repo reads it from the store. They ask
"are there 1,298 polygons with `ring_status: bulge`", and that answer comes from
documents written by an earlier ingest. Breaking the gate changes not one
existing document, so not one test moves. They test that gate's **trace**, not
the gate — and the breakage only shows up on the next ingest, by which time
nobody is watching any more.

So the tests here build their own DXF in memory and call `_ring_fields`
directly. No Mongo, no fixture, no number from any one particular drawing.

One thing deliberately tested and easy to miss: the arced polygons here are
built so that **their chords are perfectly valid** — closed, three distinct
vertices, non-zero area, not self-intersecting. This gate is not a validity
filter. A "fix" that replaces the arc check with a validity check would let
them through, and this is the test that refuses.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest import mock

import ezdxf
import pytest

from app import extract, geometry


def _msp(tmp_path: Path):
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 6  # metres
    return doc, doc.modelspace()


def _first(msp):
    return list(msp)[0]


# --- the arc gate: the mutant that got through ------------------------------


def test_an_arc_polygon_is_stored_as_the_curve_and_not_as_its_chords(tmp_path: Path):
    """The invariant is unchanged; what satisfies it is.

    A box with one curved side. Its vertex sequence -- its chords -- is
    closed, has four distinct vertices and an area of 100: valid by every
    measure of validity. What made it unusable was never its validity but the
    fact that it is **not the real boundary** of the shape that was drawn, and
    the gate refused it for exactly that reason.

    From ingest version 8 the bulged span is expanded into the arc it
    describes, so what is stored IS the boundary and the reason for the
    refusal is gone rather than waived. The old assertion is kept in spirit
    and sharpened: the ring must differ from the chords, and it must differ by
    the arc's own sagitta.
    """
    doc, msp = _msp(tmp_path)
    msp.add_lwpolyline(
        [(0, 0, 0, 0, 0.55), (10, 0, 0, 0, 0), (10, 10, 0, 0, 0), (0, 10, 0, 0, 0)],
        format="xyseb",
        close=True,
    )
    out = extract._ring_fields(_first(msp))

    assert out["ring_status"] == "complete"
    assert out["ring_arc_flattened"] is True
    assert out["ring_arc_tolerance"] == geometry.ARC_CHORD_TOLERANCE
    ring = out["ring"]
    assert ring is not None

    # The curve is not the chord run: more vertices, and they leave the box.
    assert out["ring_vertex_count"] > 4
    # sagitta = (chord / 2) * bulge, so 5 * 0.55. The bulged span runs along
    # y = 0, and a POSITIVE bulge puts the arc outside the box.
    assert min(y for _, y in ring) == pytest.approx(-2.75, abs=0.01)
    assert abs(geometry.signed_area(ring)) == pytest.approx(119.33, abs=0.1)

    # Every point of the flattened span really lies on one circle.
    span = [p for p in ring if p[1] < 0]
    centre_y = -2.75 + (5.0**2 + 2.75**2) / (2 * 2.75)
    radii = {round(math.hypot(x - 5.0, y - centre_y), 6) for x, y in span}
    assert max(radii) - min(radii) < 1e-6, "flattened points must sit on the arc"


def test_a_bulge_on_the_closing_segment_is_flattened_too(tmp_path: Path):
    """The last vertex closes onto the first, and the arc sits right there.

    The closing segment is the one that most easily escapes a loop over
    consecutive pairs that does not wrap around -- which is as true of
    flattening it as it was of detecting it.
    """
    doc, msp = _msp(tmp_path)
    msp.add_lwpolyline(
        [(0, 0, 0, 0, 0), (10, 0, 0, 0, 0), (10, 10, 0, 0, 0), (0, 10, 0, 0, 0.4)],
        format="xyseb",
        close=True,
    )
    out = extract._ring_fields(_first(msp))
    assert out["ring_status"] == "complete"
    assert out["ring_arc_flattened"] is True
    # The closing span runs down x = 0, so the arc leaves the box in -x.
    assert min(x for x, _ in out["ring"]) == pytest.approx(-2.0, abs=0.02)


def test_the_sign_of_a_bulge_decides_which_side_the_arc_falls(tmp_path: Path):
    """An arc the other way round. The sign is direction, not presence.

    Worth asserting as a PAIR: a flattening that ignored the sign would put
    every arc on the same side of its chord, and on a closed shape that is the
    difference between a boundary that contains a point and one that does not.
    """
    doc, msp = _msp(tmp_path)
    msp.add_lwpolyline(
        [(0, 0, 0, 0, -0.55), (10, 0, 0, 0, 0), (10, 10, 0, 0, 0), (0, 10, 0, 0, 0)],
        format="xyseb",
        close=True,
    )
    out = extract._ring_fields(_first(msp))
    assert out["ring_status"] == "complete"
    ring = out["ring"]
    # Negative: the arc bows INTO the box, so nothing goes below y = 0 and the
    # area is smaller than the chords' 100 rather than larger.
    assert min(y for _, y in ring) == pytest.approx(0.0, abs=1e-9)
    assert abs(geometry.signed_area(ring)) == pytest.approx(80.67, abs=0.1)


def test_a_bulged_ring_too_big_to_flatten_is_still_refused(tmp_path: Path):
    """The ceiling still wins, and for the reason it always did.

    Flattening buys vertices, and a ring over `MAX_RING_VERTICES` is not
    stored truncated -- a truncated ring is a different ring. So the old
    refusal survives exactly where it is still true, rather than being
    deleted along with the case it no longer applies to.
    """
    doc, msp = _msp(tmp_path)
    msp.add_lwpolyline(
        [(0, 0, 0, 0, 0.55), (10, 0, 0, 0, 0), (10, 10, 0, 0, 0), (0, 10, 0, 0, 0)],
        format="xyseb",
        close=True,
    )
    entity = _first(msp)
    with mock.patch.object(geometry, "MAX_RING_VERTICES", 5):
        out = extract._ring_fields(entity)
    assert out["ring_status"] == "bulge"
    assert out.get("ring") is None
    assert "over the 5 limit" in out["geometry_note"]


def test_a_bulge_below_the_threshold_is_not_treated_as_an_arc(tmp_path: Path):
    """The other side of the same gate: a bulge value of zero must pass.

    Without this test, "refuse everything that has a bulge field" would look
    correct, and every straight LWPOLYLINE in every drawing would stop being
    measurable — a failure far larger than the one being guarded against.
    """
    doc, msp = _msp(tmp_path)
    msp.add_lwpolyline(
        [(0, 0, 0, 0, 0.0), (10, 0, 0, 0, 0.0), (10, 10, 0, 0, 0.0), (0, 10, 0, 0, 0.0)],
        format="xyseb",
        close=True,
    )
    out = extract._ring_fields(_first(msp))
    assert out["ring_status"] == "complete"
    assert out["ring"] == [[0, 0], [10, 0], [10, 10], [0, 10]]


# --- the other gates, so that fixing one does not knock another down --------


def test_an_open_polygon_is_refused(tmp_path: Path):
    doc, msp = _msp(tmp_path)
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 10), (0, 10)], format="xy", close=False)
    out = extract._ring_fields(_first(msp))
    assert out["ring_status"] == "open"
    assert out.get("ring") is None


def test_fewer_than_three_distinct_vertices_is_degenerate(tmp_path: Path):
    doc, msp = _msp(tmp_path)
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 0), (0, 0)], format="xy", close=True)
    out = extract._ring_fields(_first(msp))
    assert out["ring_status"] == "degenerate"
    assert out["ring_dropped_duplicates"] >= 1


def test_a_ring_that_doubles_back_has_no_centroid_to_report(tmp_path: Path):
    """Signed area of zero. Three distinct vertices, but no inside."""
    doc, msp = _msp(tmp_path)
    msp.add_lwpolyline([(0, 0), (10, 0), (5, 0)], format="xy", close=True)
    assert extract._ring_fields(_first(msp))["ring_status"] == "degenerate"


def test_too_many_vertices_is_refused_rather_than_measured(tmp_path: Path):
    """A resource limit, and it must sound like a limit — not like a polygon
    that passed."""
    doc, msp = _msp(tmp_path)
    n = geometry.MAX_RING_VERTICES + 8
    pts = [(float(i), float(i % 7)) for i in range(n)]
    msp.add_lwpolyline(pts, format="xy", close=True)
    out = extract._ring_fields(_first(msp))
    assert out["ring_status"] == "oversize"
    assert out.get("ring") is None
    assert str(geometry.MAX_RING_VERTICES) in out["geometry_note"]


def test_a_polyline_with_varying_z_is_refused(tmp_path: Path):
    """3D POLYLINE: its projection onto XY can intersect itself."""
    doc, msp = _msp(tmp_path)
    msp.add_polyline3d([(0, 0, 0), (10, 0, 5), (10, 10, 0), (0, 10, 9)], close=True)
    out = extract._ring_fields(_first(msp))
    assert out["ring_status"] == "non_planar"
    assert out.get("ring") is None


# --- invariants that hold for EVERY status ----------------------------------


@pytest.mark.parametrize(
    "points, kwargs, fmt",
    [
        # Two coincident vertices: degenerate, and still refused.
        ([(0, 0), (10, 0)], {"close": True}, "xy"),
        ([(0, 0), (10, 0), (10, 10), (0, 10)], {"close": False}, "xy"),
        ([(0, 0), (10, 0), (10, 0)], {"close": True}, "xy"),
    ],
)
def test_no_refused_ring_ever_carries_a_ring(tmp_path: Path, points, kwargs, fmt):
    """One positive gate, one meaning.

    The old form used two booleans, and its filter was the ABSENCE of both —
    so open polygons and arced polygons passed through as point-in-polygon
    targets as well. This invariant closes that broken form for good: if the
    status is not "complete", there is no ring anybody can use to answer
    anything.
    """
    doc, msp = _msp(tmp_path)
    msp.add_lwpolyline(points, format=fmt, **kwargs)
    out = extract._ring_fields(_first(msp))

    assert out["ring_status"] != "complete"
    for key in ("ring", "ring_origin", "polygon_centroid", "shape_key", "edge_lengths"):
        assert out.get(key) is None, f"{key} leaked at status {out['ring_status']}"
    assert out["geometry_note"], "a refusing status must state its reason"


def test_an_open_polyline_with_a_bulge_is_not_marked_as_a_flattened_ring(tmp_path: Path):
    """The flag must describe what happened, not what was attempted.

    Flattening ran before the closed test, so an OPEN bulged polyline had its
    arc computed, thrown away when it bailed as `open`, and was still stamped
    `ring_arc_flattened`. Measured on Sedra, 23,180 of the 26,069 rows
    carrying that flag had no ring at all. Its curve is not lost -- an open
    polyline keeps `path_points` and `path_bulges` -- but a flag claiming a
    ring was made where none was is the same class of defect as a count that
    disagrees with the store.
    """
    doc, msp = _msp(tmp_path)
    msp.add_lwpolyline(
        [(0, 0, 0, 0, 0.55), (10, 0, 0, 0, 0), (10, 10, 0, 0, 0)],
        format="xyseb",
        close=False,
    )
    out = extract._ring_fields(_first(msp))
    assert out["ring_status"] == "open"
    assert out.get("ring") is None
    assert out.get("ring_arc_flattened") is None, (
        "nothing was flattened into a ring, so nothing may claim it was"
    )

    # And the curve itself is still carried, by the field that owns it.
    path = extract._path_point_fields(_first(msp))
    assert path["path_points"] is not None
    assert path["path_bulges"] and any(path["path_bulges"])
