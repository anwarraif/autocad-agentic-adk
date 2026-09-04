"""DOSSIER Phase 5c — stored path endpoints, and the `junction_census` recipe.

Owner: the TOPOLOGY-ENDPOINTS lane.

**Known truth first, and the known truth here is a rule rather than a number.**
The failure this recipe exists to prevent is not an arithmetic slip, it is a
definition: counting *how many paths meet* instead of *how many branches leave*.
The two agree on a dead end and disagree on everything else — a T-junction whose
through-road was never split reads as a dead end, a crossroads reads as a simple
connection, and a drawing with two dozen crossroads reports none. Two tests
below are that rule, stated on geometry drawn by hand:

* `test_a_stem_landing_on_an_unsplit_road_is_a_t_junction_not_a_dead_end`
* `test_a_road_crossing_an_unsplit_road_is_a_crossroads_not_a_simple_connection`

Everything else supports them. Four kinds of test, in this order:

1.  **`extract.py`'s two geometry fields**, on DXFs built in memory so that each
    test states exactly the condition it checks. Including the three that are
    not obvious: an ARC's ends are on the arc, never on the corners of its box;
    a CLOSED polyline that contains an arc must come back as `closed_ring`,
    because `ring_status` calls it `bulge` — exactly like an open one — and
    would otherwise hand two coincident "ends" to the node clustering; and a
    MIRRORED polyline's bulge must change sign, because the lift into world
    coordinates reverses handedness and an arc left on the wrong side of its
    own chord moves by twice the sagitta with no other number changing.
2.  **Pure functions**, with no database at all: node clustering, branch degree,
    the parity that makes a bounded degree useful rather than merely wide, and
    the bulge-to-arc reconstruction — checked against `ezdxf.math.bulge_to_arc`,
    because a number agreeing with itself is not a check.
3.  **The recipe over a fixture store** whose every junction was counted by
    hand, including the triplication trap and a drawing that predates the field.
4.  **Janadriyah**, skipped when there is no database, and recorded either way.

**Two of this file's ground-truth figures have already been wrong**, once from
the owner's hand count and once from this lane's own independent check, in
different halves of the same computation. What caught both is written out at
`JANADRIYAH_MEASURED`, because the way a measurement failed is worth more than
the number it produced.

**One note for the integrator.** Importing this file registers one more recipe,
so the pinned recipe-name tuple in `tests/test_recipes.py` no longer matches.
That pin and the mount in `app/recipes/__init__.py` are the integrator's
one-line changes; this file deliberately touches neither. The recipe is called
**`junction_census`**.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import sys

import ezdxf
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import extract  # noqa: E402
from app import recipes  # noqa: E402
from app.recipes import junctions as jn  # noqa: E402
from app.recipes import registry as rr  # noqa: E402
from app.recipes import topology as tp  # noqa: E402


DRAWING = "596212db022a3397"
FIXTURE = "fixture0000jncts"
LAYOUT = "Model"

#: Layer names live in the TEST, never in the module under test (G1).
PATHS = "fixture-paths"
OTHER = "fixture-other"

METRE_UNITS = {
    "name": "m",
    "declared_in_file": True,
    "length_unit": "m",
    "area_unit": "m2",
    "space": "model",
}
INCH_UNITS = {
    "name": "in",
    "declared_in_file": True,
    "length_unit": "in",
    "area_unit": "in2",
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
ALL_UNIT_SHAPES = [
    pytest.param(METRE_UNITS, id="metre"),
    pytest.param(INCH_UNITS, id="inch"),
    pytest.param(UNITLESS, id="undeclared"),
]


# =============================================================================
# Tooling
# =============================================================================


class FakeCursor(list):
    def sort(self, spec):
        for key, direction in reversed(list(spec)):
            list.sort(
                self,
                key=lambda d, k=key: (d.get(k) is None, str(d.get(k))),
                reverse=direction < 0,
            )
        return self


def _dotted(doc, key):
    """`bbox.min.0` against a document, the way MongoDB reads it."""
    current = doc
    for part in key.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
    return current


class FakeCollection:
    """Just enough MongoDB for this file, and not one line more.

    It interprets equality, `$in`, `$nin`, `$gte` and `$lte` over dotted keys —
    every operator this recipe uses and no other. A fake that interpreted more
    would be a fake that can lie differently from the real cluster.
    """

    def __init__(self, docs=()):
        self.docs = [copy.deepcopy(d) for d in docs]
        self.filters: list[dict] = []

    def _match(self, doc, flt):
        for key, want in flt.items():
            got = _dotted(doc, key)
            if isinstance(want, dict):
                for op, arg in want.items():
                    if op == "$in" and got not in arg:
                        return False
                    if op == "$nin" and got in arg:
                        return False
                    if op == "$gte" and (got is None or got < arg):
                        return False
                    if op == "$lte" and (got is None or got > arg):
                        return False
                    if op not in ("$in", "$nin", "$gte", "$lte"):
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


def line_doc(handle, p0, p1, *, layer=PATHS, layout=LAYOUT, drawing=FIXTURE):
    """A LINE as the store holds it since Phase 5c: a box AND its two ends."""
    return {
        "_id": f"{drawing}:{handle}",
        "drawing_id": drawing,
        "layout": layout,
        "layer": layer,
        "handle": handle,
        "type": "LINE",
        "length": math.dist(p0, p1),
        "bbox": {
            "min": [min(p0[0], p1[0]), min(p0[1], p1[1])],
            "max": [max(p0[0], p1[0]), max(p0[1], p1[1])],
        },
        "ends": [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]],
        "ends_status": extract.ENDS_OPEN,
        "ends_basis": "LINE.start and LINE.end, read from the file in WCS",
    }


def curve_doc(
    handle, p0, p1, box, length, *, points=None, bulges=None, layer=PATHS,
    drawing=FIXTURE,
):
    """An open path that BENDS: two ends, a length longer than the chord.

    With `points` it is a row as the store holds it since the vertex change,
    and its interior is exact. WITHOUT `points` it is a row whose vertices
    could not be stored -- over the vertex limit, or unreadable -- which is the
    case that keeps the degree-range machinery honest and exercised.
    """
    doc = {
        "_id": f"{drawing}:{handle}",
        "drawing_id": drawing,
        "layout": LAYOUT,
        "layer": layer,
        "handle": handle,
        "type": "LWPOLYLINE",
        "length": length,
        "ring": None,
        "ring_status": "open",
        "bbox": {"min": [box[0], box[1]], "max": [box[2], box[3]]},
        "ends": [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]],
        "ends_status": extract.ENDS_OPEN,
        "ends_basis": "first/last vertex of the open polyline (WCS)",
        "path_points_basis": (
            "the open polyline's own vertices (WCS)"
            if points
            else f"{9999} vertices exceed the 4096 limit"
        ),
    }
    if points:
        doc["path_points"] = [[float(x), float(y)] for x, y in points]
    if bulges:
        doc["path_bulges"] = [float(b) for b in bulges]
    return doc


def text_doc(handle, at, *, layer=PATHS, drawing=FIXTURE):
    """A stray label on a network layer. It has no ends as a CATEGORY."""
    return {
        "_id": f"{drawing}:{handle}",
        "drawing_id": drawing,
        "layout": LAYOUT,
        "layer": layer,
        "handle": handle,
        "type": "TEXT",
        "bbox": {"min": [at[0], at[1]], "max": [at[0] + 2, at[1] + 1]},
    }


def stale_line_doc(handle, p0, p1, *, layer=PATHS, drawing=FIXTURE):
    """A LINE as an ingest OLDER than Phase 5c wrote it: a box, and no ends."""
    doc = line_doc(handle, p0, p1, layer=layer, drawing=drawing)
    doc.pop("ends")
    doc.pop("ends_status")
    doc.pop("ends_basis")
    return doc


#: The hand-counted network. Every junction in it was worked out on paper
#: before a line of the recipe ran, and the arithmetic is in the docstring of
#: `known_store` so that a reader can redo it.
def _known_docs():
    return [
        # --- a T-junction: a stem landing on the middle of an unsplit road ---
        line_doc("L-THROUGH", (0, 10), (40, 10)),
        line_doc("L-STEM", (20, 10), (20, 30)),
        # --- a crossroads: a split road crossing an unsplit one -------------
        line_doc("L-CROSS-A", (60, 0), (60, 40)),
        line_doc("L-CROSS-B1", (50, 20), (60, 20)),
        line_doc("L-CROSS-B2", (60, 20), (70, 20)),
        # --- a simple connection: two paths meeting end to end --------------
        line_doc("L-SIMPLE-1", (100, 0), (100, 10)),
        line_doc("L-SIMPLE-2", (100, 10), (100, 20)),
        # --- a curve whose interior the store cannot describe ---------------
        curve_doc("P-CURVE", (120, 0), (140, 0), (120, -5, 140, 5), 30.0),
        line_doc("L-TOUCH", (130, 0), (130, 20)),
        # --- a stray label, which must not glue anything together -----------
        text_doc("T-LABEL", (5, 5)),
    ]


@pytest.fixture
def known_store(monkeypatch):
    """A network whose census was counted by hand.

    Nodes, worked out on paper:

    * `(0,10)`, `(40,10)`, `(20,30)` — one path end each: **3 dead ends**;
      `(20,10)` — `L-STEM` ends there and `L-THROUGH` passes THROUGH it:
      1 + 2 = **1 T-junction**;
    * `(60,0)`, `(60,40)`, `(50,20)`, `(70,20)` — **4 dead ends**;
      `(60,20)` — `L-CROSS-B1` and `L-CROSS-B2` end there and `L-CROSS-A`
      passes through: 2 + 2 = **1 crossroads**;
    * `(100,0)`, `(100,20)` — **2 dead ends**; `(100,10)` — two ends, nothing
      passing: **1 simple connection**;
    * `(120,0)`, `(140,0)`, `(130,20)` — **3 dead ends**; `(130,0)` —
      `L-TOUCH` ends there and `P-CURVE`'s BOX reaches it while its shape is
      unknown: 1 branch for certain, 3 if the curve really passes — **1
      undecided node, range 1 to 3**.

    Totals: 16 nodes; dead ends 12 (13 at most), simple connections 1,
    T-junctions 1 (2 at most), crossroads 1, five-or-more 0, undecided 1.
    """
    entities = FakeCollection(_known_docs())
    monkeypatch.setattr(jn, "_entities", lambda: entities)
    return entities


def _run(params=None, *, drawing=FIXTURE, units=None, layout=LAYOUT):
    return recipes.run(
        drawing,
        "junction_census",
        {"layers": [PATHS], **(params or {})},
        layout=layout,
        units=dict(units or METRE_UNITS),
    )


# =============================================================================
# 1. extract.py — the endpoint field
# =============================================================================


def _docs_of(tmp_path, build):
    """Extract a DXF built in memory, keeping only what MODELSPACE holds.

    The filter is load-bearing rather than tidy: `ezdxf.new(setup=True)` ships
    standard arrow-head blocks, whose contents are catalogued under `[block] …`
    pseudo-layouts, and several of them are closed LWPOLYLINEs. Without this,
    `by_type["LWPOLYLINE"]` picks an arrow head and the test asserts against a
    shape nobody drew.
    """
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 6
    build(doc.modelspace(), doc)
    path = tmp_path / "fixture.dxf"
    doc.saveas(path)
    _drawing, entities = extract.extract(path)
    drawn = [e for e in entities if e.layout == "Model"]
    return {e.type: e for e in drawn}, [e.to_mongo() for e in drawn]


def test_a_line_stores_its_two_ends_and_not_its_box_corners(tmp_path):
    """The whole point of the field: a box has four corners and a diagonal the
    file does not record, and a path has two ends that it does."""
    by_type, docs = _docs_of(tmp_path, lambda msp, d: msp.add_line((1, 2), (11, 8)))
    line = by_type["LINE"]
    assert line.ends_status == extract.ENDS_OPEN
    assert line.ends == [[1.0, 2.0], [11.0, 8.0]]
    # The other diagonal of the same box -- what a corner-based guess offers.
    assert [1.0, 8.0] not in line.ends and [11.0, 2.0] not in line.ends


def test_an_arcs_ends_are_on_the_arc(tmp_path):
    """Computed from centre, radius and the two angles.

    Cross-checked against ezdxf's own `start_point`/`end_point`, which is an
    independent implementation of the same definition: a number agreeing with
    itself is not a check.
    """
    holder = {}

    def build(msp, _doc):
        holder["arc"] = msp.add_arc((10, 20), 5, 0, 90)

    by_type, _docs = _docs_of(tmp_path, build)
    arc = by_type["ARC"]
    assert arc.ends_status == extract.ENDS_OPEN
    assert arc.ends[0] == pytest.approx([15.0, 20.0], abs=1e-9)
    assert arc.ends[1] == pytest.approx([10.0, 25.0], abs=1e-9)
    reference = holder["arc"]
    assert arc.ends[0] == pytest.approx(
        [reference.start_point.x, reference.start_point.y], abs=1e-9
    )
    assert arc.ends[1] == pytest.approx(
        [reference.end_point.x, reference.end_point.y], abs=1e-9
    )
    # A box corner would be (15, 25) -- on neither end of this quarter arc.
    assert [15.0, 25.0] not in arc.ends


def test_a_full_turn_arc_is_a_closed_ring_and_not_a_zero_length_path(tmp_path):
    """start=0/end=360 reduces to a sweep of zero under the modulo. Reported as
    a closed curve, which is what it is, rather than as two coincident ends."""
    by_type, docs = _docs_of(tmp_path, lambda msp, d: msp.add_arc((0, 0), 3, 0, 360))
    arc = by_type["ARC"]
    assert arc.ends_status == extract.ENDS_CLOSED
    stored = next(d for d in docs if d["type"] == "ARC")
    assert "ends" not in stored, "an absent field, never [0, 0]"
    assert "full turn" in (arc.ends_basis or "")


def test_a_circle_has_no_free_ends_and_says_so(tmp_path):
    by_type, docs = _docs_of(tmp_path, lambda msp, d: msp.add_circle((1, 2), 3))
    assert by_type["CIRCLE"].ends_status == extract.ENDS_CLOSED
    assert "ends" not in next(d for d in docs if d["type"] == "CIRCLE")


def test_an_open_polyline_stores_its_first_and_last_vertex(tmp_path):
    by_type, _docs = _docs_of(
        tmp_path,
        lambda msp, d: msp.add_lwpolyline([(0, 0), (5, 0), (5, 5)], close=False),
    )
    poly = by_type["LWPOLYLINE"]
    assert poly.ends_status == extract.ENDS_OPEN
    assert poly.ends == [[0.0, 0.0], [5.0, 5.0]]


def test_a_closed_polyline_that_contains_an_arc_is_still_reported_as_closed(tmp_path):
    """The regression this field exists to carry, and it is invisible elsewhere.

    `ends_status` must answer CLOSED for a closed polyline that contains an
    arc. A consumer that decided from `ring_status` alone would hand the node
    clustering two "ends" that are the same point, and report a ring as a path
    with a dead end at both tips.

    The PREMISE changed at ingest version 8 and the invariant did not, which
    is why this test is updated rather than deleted. `ring_status` used to
    answer `bulge` here -- the same thing it answers for an OPEN polyline,
    because the bulge gate fired before the closed test -- and that collision
    was the original trap. Now the arc is flattened and the status is
    `complete`, so the two no longer collide; `ends_status` still has to be
    right on its own, and a future change that reintroduced the collision
    would still be caught below.
    """

    def build(msp, _doc):
        msp.add_lwpolyline(
            [(0, 0, 0, 0, 0.5), (5, 0, 0, 0, 0), (5, 5, 0, 0, 0)],
            format="xyseb",
            close=True,
        )

    by_type, docs = _docs_of(tmp_path, build)
    poly = by_type["LWPOLYLINE"]
    assert poly.ring_status == "complete", "the premise of this test"
    assert poly.ring_arc_flattened is True, "the arc became the boundary"
    # The point of the test: decided from CLOSEDNESS, not from ring_status.
    assert poly.ends_status == extract.ENDS_CLOSED
    assert "ends" not in next(d for d in docs if d["type"] == "LWPOLYLINE")


def test_a_spline_reports_that_its_ends_were_not_derived(tmp_path):
    """A spline's control points need not lie on its own curve. Taking the
    first one would be wrong by a plausible amount rather than obviously."""
    by_type, docs = _docs_of(
        tmp_path,
        lambda msp, d: msp.add_spline([(0, 0), (5, 8), (10, 0), (15, 8)]),
    )
    spline = by_type["SPLINE"]
    assert spline.ends_status == extract.ENDS_NOT_DERIVED
    assert "control points" in (spline.ends_basis or "")
    assert "ends" not in next(d for d in docs if d["type"] == "SPLINE")


def test_text_carries_no_endpoint_keys_at_all(tmp_path):
    """A TEXT does not have ends as a CATEGORY, so the keys are absent rather
    than null -- the same rule the ring block and the anchor block follow."""
    by_type, docs = _docs_of(
        tmp_path, lambda msp, d: msp.add_text("ROOM 101").set_placement((1, 1))
    )
    stored = next(d for d in docs if d["type"] == "TEXT")
    for key in ("ends", "ends_status", "ends_basis"):
        assert key not in stored


def test_a_mirrored_polylines_ends_are_in_world_coordinates(tmp_path):
    """Extrusion (0,0,-1) FLIPS the x axis.

    An OCS endpoint published as a WCS one is metres away from where it really
    is, and two paths that genuinely meet would come back as two dead ends --
    with nothing in any number to show it. The ends must agree with the ring
    the same file produces, which is already lifted into WCS.
    """

    def build(msp, _doc):
        msp.add_lwpolyline(
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            close=True,
            dxfattribs={"extrusion": (0, 0, -1)},
        )
        msp.add_lwpolyline(
            [(0, 0), (5, 0), (5, 5)],
            close=False,
            dxfattribs={"extrusion": (0, 0, -1)},
        )

    _by_type, docs = _docs_of(tmp_path, build)
    open_poly = next(
        d for d in docs if d["type"] == "LWPOLYLINE" and d.get("ends_status") == extract.ENDS_OPEN
    )
    assert open_poly["extrusion_non_standard"] is True, "the premise of this test"
    assert open_poly["ends"][0] == [-0.0, 0.0] or open_poly["ends"][0] == [0.0, 0.0]
    assert open_poly["ends"][1] == pytest.approx([-5.0, 5.0])
    closed = next(
        d
        for d in docs
        if d["type"] == "LWPOLYLINE" and d.get("ends_status") == extract.ENDS_CLOSED
    )
    assert "ends" not in closed


def test_an_open_polyline_stores_its_own_vertices(tmp_path):
    """`ring` is the closed counterpart and has existed all along; an OPEN
    polyline had nowhere to put its shape and was stored as a box."""
    by_type, docs = _docs_of(
        tmp_path,
        lambda msp, d: msp.add_lwpolyline([(0, 0), (5, 0), (5, 5)], close=False),
    )
    poly = by_type["LWPOLYLINE"]
    assert poly.path_points == [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]
    stored = next(d for d in docs if d["type"] == "LWPOLYLINE")
    assert "path_bulges" not in stored, "no arc-ness to record, so no key"
    assert "own vertices" in stored["path_points_basis"]


def test_a_bulged_open_polyline_stores_the_bulge_with_its_vertices(tmp_path):
    """Two vertices of a bulged span are the CHORD of an arc. Handing a
    consumer the chord and calling it the path moves the geometry by up to the
    sagitta -- on a road fillet, metres -- with nothing in any number to show
    it."""

    def build(msp, _doc):
        msp.add_lwpolyline(
            [(0, 0, 0, 0, 0.5), (10, 0, 0, 0, 0), (10, 10, 0, 0, 0)],
            format="xyseb",
            close=False,
        )

    by_type, docs = _docs_of(tmp_path, build)
    poly = by_type["LWPOLYLINE"]
    assert poly.path_points == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    # One bulge per SPAN: an open polyline of three vertices has two spans, and
    # the trailing bulge belongs to a span that does not exist.
    assert poly.path_bulges == [0.5, 0.0]
    assert "per-span bulges" in (poly.path_points_basis or "")


def test_an_old_style_polyline_stores_its_vertices_and_bulges_too(tmp_path):
    """`POLYLINE` is not a legacy curiosity here -- it is the normal shape for
    a topographic contour, and it reaches this code by a different branch from
    `LWPOLYLINE`. A branch with no test is a branch that works by luck."""

    def build(msp, _doc):
        poly = msp.add_polyline2d([(0, 0), (10, 0), (10, 10)], close=False)
        poly.vertices[0].dxf.bulge = 0.25

    by_type, _docs = _docs_of(tmp_path, build)
    poly = by_type["POLYLINE"]
    assert poly.path_points == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    assert poly.path_bulges == [0.25, 0.0]
    assert poly.ends == [[0.0, 0.0], [10.0, 10.0]]


def test_a_closed_polyline_keeps_its_shape_in_ring_and_not_twice(tmp_path):
    """Two copies of one shape is two answers to one question."""
    by_type, docs = _docs_of(
        tmp_path,
        lambda msp, d: msp.add_lwpolyline([(0, 0), (5, 0), (5, 5)], close=True),
    )
    stored = next(d for d in docs if d["type"] == "LWPOLYLINE")
    assert stored["ring_status"] == "complete" and stored["ring"]
    assert "path_points" not in stored
    assert "is in `ring`" in stored["path_points_basis"]


def test_a_mirrored_bulged_polyline_bends_the_way_the_drawing_bends(tmp_path):
    """The silent half of the OCS lift.

    Mirroring reverses handedness, so an arc that curves left of its chord in
    OCS curves right of it in WCS. Lifting the points and leaving the bulge
    alone puts the arc on the wrong side of its own chord -- by twice the
    sagitta -- and neither the length nor the box moves to tell you. Checked
    against ezdxf's own OCS-aware flattening.
    """
    from ezdxf import path as ezpath

    holder = {}

    def build(msp, _doc):
        holder["poly"] = msp.add_lwpolyline(
            [(0, 0, 0, 0, 0.5), (10, 0, 0, 0, 0)],
            format="xyseb",
            close=False,
            dxfattribs={"extrusion": (0, 0, -1)},
        )

    by_type, _docs = _docs_of(tmp_path, build)
    poly = by_type["LWPOLYLINE"]
    assert poly.path_bulges == [-0.5], "the sign must flip under the mirror"

    span = jn._bulge_span(
        tuple(poly.path_points[0]), tuple(poly.path_points[1]), poly.path_bulges[0]
    )
    reference = [(v.x, v.y) for v in ezpath.make_path(holder["poly"]).flattening(0.001)]
    for point in reference:
        assert jn._span_distance(point[0], point[1], span) < 0.01, (
            "the reconstructed arc must lie on the curve ezdxf draws in WCS"
        )


def test_an_oversize_polyline_stores_nothing_and_says_why(tmp_path, monkeypatch):
    """The same ceiling `_ring_fields` applies, for the same reason: what is
    over it is not stored at all rather than stored truncated, because a
    truncated path is a different path."""
    monkeypatch.setattr(extract.geometry, "MAX_RING_VERTICES", 4)
    by_type, docs = _docs_of(
        tmp_path,
        lambda msp, d: msp.add_lwpolyline(
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)], close=False
        ),
    )
    stored = next(d for d in docs if d["type"] == "LWPOLYLINE")
    assert "path_points" not in stored
    assert "exceed the 4 limit" in stored["path_points_basis"]
    # Its ENDS are still exact -- the node graph does not lose the entity.
    assert stored["ends"] == [[0.0, 0.0], [5.0, 0.0]]


def test_the_ingest_version_was_raised_so_old_drawings_are_re_read():
    """Without this, every drawing already in the store keeps its old documents
    and every junction question answers "re-ingest" for ever."""
    assert extract.INGEST_VERSION >= 6


def test_every_stored_end_is_a_real_coordinate(tmp_path):
    """Never `[0, 0]`. On a projected drawing that sentinel sits 2.7 million
    units from the geometry, and it would look like a coordinate."""

    def build(msp, _doc):
        msp.add_line((691000, 2750000), (691050, 2750030))
        msp.add_arc((691000, 2750000), 12, 30, 200)
        msp.add_lwpolyline([(691000, 2750000), (691010, 2750010)], close=False)

    _by_type, docs = _docs_of(tmp_path, build)
    stored = [d for d in docs if "ends" in d]
    assert len(stored) == 3
    for doc in stored:
        for point in doc["ends"]:
            assert all(math.isfinite(v) for v in point)
            assert math.hypot(point[0], point[1]) > 1.0


# =============================================================================
# 2. Pure functions -- branch degree, with no database at all
# =============================================================================


def _paths(*docs):
    return [jn._Path(i, doc) for i, doc in enumerate(docs)]


def test_a_lines_interior_is_known_and_a_curves_is_not():
    """The one distinction the whole bound rests on, and it is per ROW rather
    than per type: a polyline as long as its own chord is straight, and a
    polyline whose vertices were never stored is unknown however it is shaped."""
    straight_line, bent, straight_poly = _paths(
        line_doc("A", (0, 0), (10, 0)),
        curve_doc("B", (0, 0), (10, 0), (0, -3, 10, 3), 14.0),
        curve_doc("C", (0, 0), (10, 0), (0, 0, 10, 0), 10.0),
    )
    assert straight_line.spans == [("seg", (0.0, 0.0), (10.0, 0.0))]
    assert bent.spans is None, "a curve with no stored vertices is not two points"
    assert straight_poly.spans == [("seg", (0.0, 0.0), (10.0, 0.0))]
    assert "as short as its own chord" in straight_poly.shape_basis


def test_a_stored_polyline_is_exact_however_it_bends():
    """The second half of the data-layer change: with its own vertices stored,
    a curved open polyline stops being a bounding box and starts being a path."""
    curved = _paths(
        curve_doc(
            "D",
            (0, 0),
            (20, 0),
            (0, -6, 20, 6),
            26.0,
            points=[(0, 0), (10, -5), (20, 0)],
        )
    )[0]
    assert curved.spans == [
        ("seg", (0.0, 0.0), (10.0, -5.0)),
        ("seg", (10.0, -5.0), (20.0, 0.0)),
    ]
    assert "stored vertices" in curved.shape_basis
    # The midpoint of the drawn shape is ON it; the midpoint of the chord is not.
    assert curved.distance_to(10.0, -5.0) == pytest.approx(0.0)
    assert curved.distance_to(10.0, 0.0) > 4.0


def test_a_bulged_span_is_measured_as_an_arc_and_not_as_its_chord():
    """A bulge is an arc, and its chord is up to a sagitta away from it.

    Measuring the chord would put a junction on a road the drawing bends
    around -- the same class of error as measuring a bounding box, only
    smaller and therefore harder to notice.
    """
    arc = _paths(
        curve_doc(
            "E",
            (0, 0),
            (10, 0),
            (0, -5, 10, 0),
            15.7,
            points=[(0, 0), (10, 0)],
            bulges=[1.0],
        )
    )[0]
    assert arc.spans is not None and arc.spans[0][0] == "arc"
    assert "arcs, not as their chords" in arc.shape_basis
    # A semicircle below the chord: its own apex is on it, the chord's midpoint
    # is a full radius away.
    assert arc.distance_to(5.0, -5.0) == pytest.approx(0.0, abs=1e-9)
    assert arc.distance_to(5.0, 0.0) == pytest.approx(5.0, abs=1e-9)


@pytest.mark.parametrize("bulge", [0.2, 0.5, 1.0, 2.0, -0.3, -1.5])
def test_the_bulge_reconstruction_agrees_with_ezdxf(bulge):
    """Checked against `ezdxf.math.bulge_to_arc`, an independent implementation
    of the same DXF definition. A number agreeing with itself is not a check."""
    from ezdxf.math import bulge_to_arc

    p0, p1 = (3.0, 7.0), (13.0, 11.0)
    span = jn._bulge_span(p0, p1, bulge)
    assert span[0] == "arc"
    _kind, centre, radius, start, sweep, _a, _b = span
    ref_centre, ref_start, ref_end, ref_radius = bulge_to_arc(p0, p1, bulge)
    assert centre == pytest.approx((ref_centre.x, ref_centre.y), abs=1e-9)
    assert radius == pytest.approx(ref_radius, abs=1e-9)
    # Both endpoints must sit on the reconstructed circle.
    for point in (p0, p1):
        assert math.dist(point, centre) == pytest.approx(radius, abs=1e-9)
    # And the arc's own midpoint must be at distance zero from it.
    mid_angle = start + sweep / 2.0
    apex = (
        centre[0] + radius * math.cos(mid_angle),
        centre[1] + radius * math.sin(mid_angle),
    )
    assert jn._span_distance(apex[0], apex[1], span) == pytest.approx(0.0, abs=1e-9)
    # A point on the far side of the circle is not on the arc.
    opposite = (
        centre[0] - radius * math.cos(mid_angle),
        centre[1] - radius * math.sin(mid_angle),
    )
    assert jn._span_distance(opposite[0], opposite[1], span) > radius * 0.5


def test_two_paths_meeting_end_to_end_make_one_node_and_two_dead_ends():
    paths = _paths(line_doc("A", (0, 0), (0, 10)), line_doc("B", (0, 10), (0, 20)))
    nodes = jn._cluster_endpoints(paths, 0.01, jn.MAX_POINT_COMPARISONS)
    assert [n.ends for n in nodes] == [1, 2, 1]
    jn._attach_interiors(nodes, paths, 0.01, jn.MAX_INTERIOR_VERTEX_TESTS)
    assert sorted(n.degree_min for n in nodes) == [1, 1, 2]


def test_a_node_beyond_the_tolerance_does_not_merge():
    """The tolerance decides the answer: at 0.01 these are two dead ends, at
    1.0 they are one connection. That is why it is published."""
    paths = _paths(line_doc("A", (0, 0), (0, 10)), line_doc("B", (0, 10.5), (0, 20)))
    assert len(jn._cluster_endpoints(paths, 0.01, jn.MAX_POINT_COMPARISONS)) == 4
    assert len(jn._cluster_endpoints(paths, 1.0, jn.MAX_POINT_COMPARISONS)) == 3


def test_an_unknown_interior_moves_the_degree_by_two_never_by_one():
    """Parity is what makes a bounded degree useful rather than merely wide: a
    node reported as 1-to-3 is a dead end or a T-junction, and can never be a
    simple connection."""
    node = jn._Node((0.0, 0.0), ends=1, ending={0})
    node.possible = [1, 2]
    assert jn._reachable(node) == [1, 3, 5]
    assert node.degree_min == 1 and node.degree_max == 5


def test_the_node_clustering_refuses_rather_than_truncating():
    """A count from a truncated scan is a number about part of a layer wearing
    the name of the whole one (G7)."""
    paths = _paths(*[line_doc(f"L{i}", (0, 0), (1, 1)) for i in range(40)])
    assert jn._cluster_endpoints(paths, 1.0, max_comparisons=5) is None


# =============================================================================
# 3. The rule -- the two tests this recipe exists for
# =============================================================================


def test_a_stem_landing_on_an_unsplit_road_is_a_t_junction_not_a_dead_end(known_store):
    """Count paths and you get 1 here. Count BRANCHES and you get 3."""
    r = _run()
    junctions = r["junctions"]
    assert junctions["t_junctions"]["at_least"] == 1
    assert junctions["simple_connections"]["exact"] == 1
    # And the node itself names what runs through it.
    assert r["undecided"]["count"] == 1, "only the curve is undecided"


def test_a_road_crossing_an_unsplit_road_is_a_crossroads_not_a_simple_connection(
    known_store,
):
    """Two ends plus one road passing through is four branches. An endpoint-to-
    endpoint count sees two paths meeting and calls it a simple connection."""
    r = _run()
    assert r["junctions"]["crossroads"]["exact"] == 1


def test_the_whole_hand_counted_census(known_store):
    """Every number in `known_store`'s docstring, in one place."""
    r = _run()
    junctions = r["junctions"]
    assert r["nodes_total"] == 16
    assert junctions["dead_ends"] == {"at_least": 12, "at_most": 13, "exact": None}
    assert junctions["simple_connections"] == {
        "at_least": 1,
        "at_most": 1,
        "exact": 1,
    }
    assert junctions["t_junctions"] == {"at_least": 1, "at_most": 2, "exact": None}
    assert junctions["crossroads"] == {"at_least": 1, "at_most": 1, "exact": 1}
    assert junctions["five_or_more"]["count"] == 0
    assert r["undecided"]["count"] == 1


def test_the_undecided_node_is_published_with_its_range_and_the_handle(known_store):
    """Not folded into the nearest category, and not dropped (G8)."""
    r = _run()
    row = r["undecided"]["rows"][0]
    assert row["branches_proven"] == 1
    assert row["branches_at_most"] == 3
    assert row["passing_through_unknown"] == ["P-CURVE"]
    assert "L-TOUCH" in row["paths_ending_here"]
    assert "how_to_close_it" in r["undecided"]


def test_a_stored_curve_decides_the_node_its_bounding_box_only_bracketed(
    monkeypatch,
):
    """The whole point of storing the vertices, shown on one node.

    The same geometry twice. Without the polyline's vertices the node at
    (130,0) is 1-to-3 — a dead end or a T-junction, and the store cannot say
    which. With them it is a T-junction, decided, because the polyline really
    does run through that point.
    """
    bracketed = FakeCollection(_known_docs())
    monkeypatch.setattr(jn, "_entities", lambda: bracketed)
    before = _run()
    assert before["undecided"]["count"] == 1
    assert before["junctions"]["t_junctions"] == {
        "at_least": 1,
        "at_most": 2,
        "exact": None,
    }

    docs = [d for d in _known_docs() if d["handle"] != "P-CURVE"]
    docs.append(
        curve_doc(
            "P-CURVE",
            (120, 0),
            (140, 0),
            (120, -5, 140, 5),
            30.0,
            points=[(120, 0), (125, -4), (135, -4), (140, 0)],
        )
    )
    decided = FakeCollection(docs)
    monkeypatch.setattr(jn, "_entities", lambda: decided)
    after = _run()
    assert after["undecided"]["count"] == 0
    # The curve dips BELOW the node, so it does not pass through it: the node
    # is a decided dead end rather than a decided T-junction. Decided is the
    # claim; which way it went is the drawing's business.
    assert after["junctions"]["dead_ends"]["exact"] == 13
    assert after["junctions"]["t_junctions"]["exact"] == 1


def test_a_stored_curve_that_really_passes_through_makes_a_t_junction(monkeypatch):
    """The other half of the same coin: the curve runs through the node."""
    docs = [d for d in _known_docs() if d["handle"] != "P-CURVE"]
    docs.append(
        curve_doc(
            "P-CURVE",
            (120, 0),
            (140, 0),
            (120, -5, 140, 5),
            30.0,
            points=[(120, 0), (125, 3), (130, 0), (135, -3), (140, 0)],
        )
    )
    entities = FakeCollection(docs)
    monkeypatch.setattr(jn, "_entities", lambda: entities)
    r = _run()
    assert r["undecided"]["count"] == 0
    assert r["junctions"]["t_junctions"]["exact"] == 2
    assert r["junctions"]["dead_ends"]["exact"] == 12


def test_a_stray_label_is_excluded_and_counted_not_used_as_geometry(known_store):
    """A TEXT's bounding box gluing two roads into one junction is a wrong
    answer nobody could see in the number."""
    r = _run()
    assert r["path_census"]["not_path_geometry"] == 1
    assert r["paths_in_this_copy"] == 9


# =============================================================================
# 4. The triplication trap
# =============================================================================


def _triplication_docs():
    """The same two-path network drawn three times, ~1 km apart.

    Each copy: two lines meeting end to end -> 1 simple connection, 2 dead
    ends. One entity on ANOTHER layer sits on top of the middle copy, which is
    what makes that copy the live one.
    """
    docs = []
    for index, offset in enumerate((0.0, 1000.0, 2000.0)):
        docs.append(line_doc(f"C{index}-A", (offset, 0), (offset + 10, 0)))
        docs.append(line_doc(f"C{index}-B", (offset + 10, 0), (offset + 10, 10)))
    docs.append(line_doc("OTHER-1", (1002, 2), (1008, 8), layer=OTHER))
    return docs


@pytest.fixture
def triplicated_store(monkeypatch):
    entities = FakeCollection(_triplication_docs())
    monkeypatch.setattr(jn, "_entities", lambda: entities)
    return entities


def test_three_copies_are_counted_once_and_the_live_one_is_named(triplicated_store):
    """The failure this campaign exists to end: an answer that silently counts
    three estates. Six paths across three copies is 3 nodes per copy, not 9."""
    r = _run()
    assert r["copies"]["count"] == 3
    assert r["copies"]["described"] == 1, "the copy the rest of the drawing sits on"
    assert "the LIVE one" in r["copies"]["chosen_by"]
    assert r["copies"]["they_agree"] is True
    assert r["nodes_total"] == 3
    assert r["junctions"]["simple_connections"]["exact"] == 1
    assert r["junctions"]["dead_ends"]["exact"] == 2
    assert r["paths_in_this_copy"] == 2


def test_every_copy_is_counted_separately_so_agreement_is_checked(triplicated_store):
    r = _run()
    rows = r["copies"]["rows"]
    assert [row["copy"] for row in rows] == [0, 1, 2]
    assert [row["other_layer_entities_inside_it"] for row in rows] == [0, 1, 0]
    assert [row["is_the_copy_described"] for row in rows] == [False, True, False]
    assert rows[0]["census"] == rows[1]["census"] == rows[2]["census"]


def test_a_caller_can_name_the_copy_and_the_response_says_which(triplicated_store):
    r = _run({"copy_index": 2})
    assert r["copies"]["described"] == 2
    assert "the caller asked for copy 2" in r["copies"]["chosen_by"]
    assert r["params_used"]["copy_index"] == 2.0


def test_a_copy_index_outside_the_copies_is_refused_with_a_suggestion(
    triplicated_store,
):
    with pytest.raises(rr.RecipeRefused) as excinfo:
        _run({"copy_index": 7})
    assert excinfo.value.code == "RECIPE_PARAM_INVALID"
    assert "3 detached bodies" in excinfo.value.message


# =============================================================================
# 5. A drawing that predates the field -- degrade honestly, never guess
# =============================================================================


@pytest.fixture
def stale_store(monkeypatch):
    docs = [
        stale_line_doc("S-A", (0, 0), (10, 0)),
        stale_line_doc("S-B", (10, 0), (10, 10)),
    ]
    entities = FakeCollection(docs)
    monkeypatch.setattr(jn, "_entities", lambda: entities)
    return entities


def test_a_drawing_without_stored_endpoints_says_re_ingest_and_gives_no_number(
    stale_store,
):
    """"Re-ingest to answer" is the right output. A census built from bounding
    box corners would look exactly like a real one and be wrong by an unknown
    amount -- which is worse than saying nothing."""
    r = _run()
    assert r["computed"] is False
    assert r["junctions"] is None
    assert r["nodes_total"] is None
    assert "re-ingest" in r["how_to_fix_it"].lower()
    assert "endpoints are not stored" in r["why_not_computed"]
    assert "never_approximated" in r
    assert r["path_census"]["endpoints_not_stored"] == 2
    assert r["evidence"]["grade"] in ("inferred", "unsupported", "stated", "corroborated")


def test_the_stale_answer_still_carries_a_scope_and_evidence(stale_store):
    """The envelope contract holds on the degraded path too: a response that
    hides its scope passes the audit in a way nobody can see."""
    r = _run()
    assert r["scope_note"]
    assert "NO census was computed" in r["scope_note"]
    assert r["evidence"]["not_established"]


def test_a_half_migrated_drawing_still_answers_over_the_rows_that_have_ends(
    monkeypatch,
):
    """One stale row does not make the whole layer unanswerable -- it is
    counted, named, and left out of the graph."""
    docs = _known_docs() + [stale_line_doc("S-OLD", (300, 300), (310, 300))]
    entities = FakeCollection(docs)
    monkeypatch.setattr(jn, "_entities", lambda: entities)
    r = _run()
    assert r["computed"] is True
    assert r["path_census"]["endpoints_not_stored"] == 1
    assert r["path_census"]["by_ends_status"]["endpoints_not_stored"] == 1


# =============================================================================
# 6. Tolerance, units, limits, determinism -- the standing rules
# =============================================================================


@pytest.mark.parametrize("units", ALL_UNIT_SHAPES)
def test_no_response_writes_a_unit_the_drawing_does_not_declare(known_store, units):
    """G2. 11 of the drawings in this store are in inches and 3 declare
    nothing; a tolerance labelled "m" because this one happens to be would be
    wrong for most of them."""
    r = _run(units=units)
    quantity = r["tolerance"]["value"]
    assert quantity["unit"] == units["length_unit"]
    if units["length_unit"] is None:
        assert quantity.get("unit_reason")
        assert "m" not in json.dumps(quantity["unit"] or "")


def test_the_tolerance_is_derived_from_this_drawing_and_published(known_store):
    """A number in drawing units written in the module would be one drawing's
    trait wearing a general name (G1)."""
    r = _run()
    block = r["tolerance"]
    assert block["supplied_by_caller"] is False
    assert block["snap_fraction"] == jn.SNAP_FRACTION
    assert block["derived_value_for_this_drawing"] == pytest.approx(
        block["value"]["value"]
    )
    assert "median" in block["scale_source"]
    assert "decides" in block
    assert "dossier_network" in block["shared_with"]


def test_the_tolerance_shares_its_rule_with_the_chain_count():
    """Two implementations of one tolerance is how two answers about the same
    layer start to differ."""
    from app import dossier_network as network

    assert jn.SNAP_FRACTION is network.SNAP_FRACTION
    assert jn.FLOAT_NOISE_FACTOR is network.FLOAT_NOISE_FACTOR


def test_a_caller_supplied_tolerance_is_used_and_labelled(known_store):
    r = _run({"snap_tolerance": 6.0})
    assert r["tolerance"]["supplied_by_caller"] is True
    assert r["tolerance"]["value"]["value"] == 6.0
    # At 6 units the two ends 10.5 apart in the sensitivity test would still be
    # separate; what changes here is that the response SAYS which was used.
    assert "handed over by the caller" in r["tolerance"]["value"]["method"]


def test_a_negative_tolerance_is_refused_with_a_reason(known_store):
    with pytest.raises(rr.RecipeRefused) as excinfo:
        _run({"snap_tolerance": -1})
    assert excinfo.value.code == "RECIPE_PARAM_INVALID"
    assert "cannot be negative" in excinfo.value.message


def test_the_sensitivity_table_is_published_and_spans_a_band(known_store):
    """Published for the same reason the tolerance is: a count that holds
    across the band is a count about the drawing, and one that moves is a count
    about the tolerance."""
    r = _run()
    rows = r["tolerance_sensitivity"]["rows"]
    assert [row["factor"] for row in rows] == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert max(row["snap_tolerance"] for row in rows) / min(
        row["snap_tolerance"] for row in rows
    ) == pytest.approx(16.0, rel=1e-6)
    assert r["tolerance_sensitivity"]["verdict"]
    for row in rows:
        assert row["census"]["t_junctions"]["at_least"] >= 0


def test_every_limit_is_stated_in_the_response(known_store):
    """G7: a limit nobody can read is not a stated limit."""
    r = _run()
    for key in (
        "path_entities",
        "endpoint_comparisons",
        "interior_vertex_tests",
        "copies_reported",
        "nodes_listed",
    ):
        assert key in r["limits_applied"]
        assert key in r["limits"]


def test_an_oversize_layer_is_refused_with_a_suggestion(known_store, monkeypatch):
    monkeypatch.setattr(jn, "MAX_PATH_ENTITIES", 3)
    with pytest.raises(rr.RecipeRefused) as excinfo:
        _run()
    assert excinfo.value.code == "RECIPE_INPUT_TOO_LARGE"
    assert excinfo.value.hint


def test_the_interior_work_is_counted_before_it_is_done(known_store, monkeypatch):
    """A budget checked after the work has run is a report, not a budget."""
    monkeypatch.setattr(jn, "MAX_INTERIOR_VERTEX_TESTS", 1)
    with pytest.raises(rr.RecipeRefused) as excinfo:
        _run()
    assert excinfo.value.code == "RECIPE_INPUT_TOO_LARGE"
    assert "limit on WORK" in excinfo.value.hint


def test_the_same_question_gives_the_same_answer(known_store):
    """Determinism is a property of a recipe, not an intention (registry rule
    1): two sessions can only compare numbers if the numbers do not move."""
    first = _run()
    second = _run()
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(
        second, sort_keys=True, default=str
    )


def test_the_response_names_where_its_layers_came_from(known_store):
    r = _run()
    assert r["layers"] == [PATHS]
    assert "caller" in r["layers_source"]
    assert "never" in r["layers_never_by_name"]


def test_no_layer_name_lives_in_the_module(known_store):
    """G1, checked against the module's own source rather than by discipline."""
    source = pathlib.Path(jn.__file__).read_text(encoding="utf-8")
    for forbidden in ("VL2", "VL4", "DP4", "LP1", "TH3", "00_Prop", "Primary School"):
        assert forbidden not in source
    assert PATHS not in source and OTHER not in source


def test_a_layer_with_no_paths_answers_zero_rather_than_failing(monkeypatch):
    """G8: the data-not-there path is tested, and it returns an answer with a
    reason instead of an exception."""
    entities = FakeCollection([text_doc("T-ONLY", (0, 0))])
    monkeypatch.setattr(jn, "_entities", lambda: entities)
    r = _run()
    assert r["computed"] is True
    assert r["nodes_total"] == 0
    assert r["junctions"]["t_junctions"]["exact"] == 0
    assert r["path_census"]["not_path_geometry"] == 1


def test_the_recipe_is_registered_under_an_honest_name():
    recipe = recipes.get("junction_census")
    assert recipe is not None
    assert recipe.needs_layout is True
    assert recipe.carries_meaning is True
    assert "extract._ends_fields" in recipe.built_on
    assert "recipes.topology._road_layers" in recipe.built_on


def test_layers_are_derived_from_config_and_refused_when_there_is_none(
    known_store, monkeypatch
):
    """G3: a drawing without a config gets a graceful refusal with a
    suggestion, never a guess from layer names."""
    monkeypatch.setattr(
        tp, "_dossier_network_layers", lambda *a, **k: ((), "no Dossier in this test")
    )
    with pytest.raises(rr.RecipeRefused) as excinfo:
        recipes.run(
            FIXTURE, "junction_census", {}, layout=LAYOUT, units=dict(METRE_UNITS)
        )
    assert excinfo.value.code == "RECIPE_NO_CONFIG"
    assert "NOT happen is a guess from" in excinfo.value.hint


# =============================================================================
# 7. Janadriyah -- the real drawing, recorded either way
# =============================================================================

#: Measured on the DXF on 26 August 2026 by `scripts`-free code that shares
#: nothing with this recipe: geometry from ezdxf's own
#: `path.make_path().flattening(0.001)`, open/closed from each entity's own
#: flag, endpoints from the file's first and last vertex. Over the LIVE copy of
#: the reference road layer (408 open paths + 2 closed), at the tolerance this
#: drawing derives for itself (0.0433751 m):
#:
#:      dead ends 188 | simple connections 171 | T-junctions 194 |
#:      crossroads 8 | five-or-more 0           (561 nodes)
#:
#: THE FIGURES THAT WERE HERE BEFORE WERE WRONG, and how they were wrong is
#: worth more than the numbers:
#:
#: 1.  The first hand count (dead 190 / simple 138 / T 189 / crossroads 23 /
#:     five-or-more 19) counted a path's own flattened interior against the
#:     node where that same path ENDS. An arc flattened at 0.2 rad has chords
#:     shorter than the 0.5 m snap on any fillet under ~2.5 m radius, so the
#:     chord next to the shared endpoint registered as a mid-span pass and
#:     added two branches to a node that had not gained any. It moved
#:     degree-2 nodes to 4 and degree-3 nodes to 5, which is exactly the shape
#:     of the 23 crossroads and the 19 five-way junctions: both were artefacts.
#: 2.  The first independent check made here (dead 188 / simple 170 / T 196 /
#:     crossroads 8) had its own bug, in the opposite half of the problem: its
#:     bulge flattening put the FAR end of a bulged polyline in the wrong
#:     place — by up to 26 m on the reference layer — which scattered about six
#:     phantom nodes. Its endpoint set disagreed with the file; the stored
#:     `ends` agree with the file and with ezdxf exactly.
#:
#: Two independent computations, two different bugs, agreeing closely enough
#: that neither looked wrong. Node counts were what exposed both: the clean
#: check above reproduces this recipe's node count at every tolerance
#: (568 / 561 / 559 / 558 across the sensitivity band), and the first one did
#: not.
#:
#: The recipe does not claim these figures exactly — an ARC entity is still
#: stored as a bounding box, so a handful of nodes carry a range. It BRACKETS
#: them, and every figure above must fall inside its own published range.
JANADRIYAH_MEASURED = {
    "dead_ends": 188,
    "simple_connections": 171,
    "t_junctions": 194,
    "crossroads": 8,
}
JANADRIYAH_COPIES = 3

#: The recipe's own figures for the same scope, recorded so that a change in
#: any of them is visible in a diff rather than only in a range that still
#: happens to contain the truth.
JANADRIYAH_EXPECTED = {
    "nodes_total": 561,
    "paths_in_this_copy": 410,
    "undecided": 5,
    "simple_connections_exact": 171,
    "crossroads_exact": 8,
}


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


def _live_road_layers(live_store, drawing):
    """The layers the Dossier or the config calls network, on the real drawing.

    Named through the same reader the recipe uses, so this test cannot pass by
    naming a layer the recipe would never have chosen (G1).
    """
    try:
        return tp._road_layers(DRAWING, LAYOUT, None)[0]
    except rr.RecipeRefused:
        pytest.skip("no network layer is derivable for the reference drawing")


def test_janadriyah_junction_census_is_either_a_bracket_or_an_honest_refusal():
    """The acceptance run, and it is written to be correct BEFORE and AFTER the
    re-ingest that stores endpoints.

    Before: every path row still lacks `ends`, and the only right answer is
    "re-ingest to answer" with no number attached. After: the census runs, and
    every hand-measured figure recorded above must lie inside the published
    range. A test that only worked in one of those two states would go green
    for the wrong reason on the day the store is migrated.
    """
    live_store, drawing = _live_or_skip()
    layers = _live_road_layers(live_store, drawing)
    r = recipes.run(
        DRAWING,
        "junction_census",
        {"layers": list(layers)},
        layout=LAYOUT,
        units=live_store._unit_names(drawing, LAYOUT),
    )

    if not r["computed"]:
        assert r["junctions"] is None
        assert "re-ingest" in r["how_to_fix_it"].lower()
        assert r["path_census"]["endpoints_not_stored"] > 0
        pytest.skip(
            "the reference drawing has not been re-ingested since path "
            "endpoints were added; the recipe refused honestly, which is the "
            "behaviour asserted above"
        )

    assert r["copies"]["count"] == JANADRIYAH_COPIES, (
        "the reference road layer holds the same estate three times; an answer "
        "over all three would triple every junction"
    )
    assert r["copies"]["they_agree"] is True
    assert "the LIVE one" in r["copies"]["chosen_by"]
    for name, measured in JANADRIYAH_MEASURED.items():
        block = r["junctions"][name]
        assert block["at_least"] <= measured <= block["at_most"], (
            f"{name}: the independently measured {measured} is outside the "
            f"published range {block['at_least']}-{block['at_most']}"
        )
    assert r["nodes_total"] == JANADRIYAH_EXPECTED["nodes_total"]
    assert r["paths_in_this_copy"] == JANADRIYAH_EXPECTED["paths_in_this_copy"]
    assert r["undecided"]["count"] == JANADRIYAH_EXPECTED["undecided"]
    assert (
        r["junctions"]["simple_connections"]["exact"]
        == JANADRIYAH_EXPECTED["simple_connections_exact"]
    )
    assert (
        r["junctions"]["crossroads"]["exact"]
        == JANADRIYAH_EXPECTED["crossroads_exact"]
    )
    # Every remaining range is an ARC entity's bounding box, which is the one
    # shape the store still describes only by its box.
    assert r["path_geometry"]["with_their_vertices_stored"] > 0
    assert r["tolerance"]["value"]["unit"] == "m"
    assert r["tolerance_sensitivity"]["rows"]
