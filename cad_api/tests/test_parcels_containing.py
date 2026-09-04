"""From a label to the parcel containing it — the direction that failed in the
meeting.

This file tests `parcels_containing` through its arithmetic, not through
MongoDB, for a reason that has already been expensive in this repo: a mutation
harness showed that every test which proves something through already-ingested
data does not move at all when the code that produced that data is broken. See
D-084.

There is not one Janadriyah number here, and not one layer name. What is
checked is a property: a point inside its plot finds its plot; nested
containers come out smallest first; a label with no insertion point answers
"could not be checked" and not "there is none"; and all of it stays correct at
projected coordinate magnitudes, which is exactly where this class of defect
appears.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from app import store_spatial

#: The real projected coordinates this project uses. A 12 x 25 m plot at
#: E 691,942 / N 2,750,756 is where every precision defect in this repo shows
#: up, and where not one of them shows up near the origin.
E0, N0 = 691942.0, 2750756.0


def rect(cx: float, cy: float, w: float, h: float, ang: float = 0.0):
    c, s = math.cos(ang), math.sin(ang)
    return [
        [cx + dx * c - dy * s, cy + dx * s + dy * c]
        for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
    ]


def ring_doc(handle: str, ring, layer: str, *, layout="Model", area=None) -> dict[str, Any]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return {
        "handle": handle,
        "layer": layer,
        "type": "LWPOLYLINE",
        "layout": layout,
        "ring": ring,
        "ring_status": "complete",
        "ring_origin": [min(xs), min(ys)],
        "bbox": {"min": [min(xs), min(ys)], "max": [max(xs), max(ys)]},
        "area": area,
    }


def label_doc(handle: str, x: float, y: float, *, layout="Model", text="1") -> dict[str, Any]:
    return {
        "handle": handle,
        "layer": "any-label-layer",
        "type": "TEXT",
        "layout": layout,
        "text": text,
        "anchor_point": [x, y],
        "anchor_basis": "TEXT.insert",
    }


class FakeCollection:
    """Just enough Mongo for this function, and no more.

    `find` ignores its filter deliberately: what is tested here is the
    filtering this function does AFTER the index, because the
    `drawing_bbox_min` index can only bind one side and the other side is
    code. A fake that filtered too would hide exactly the part that can be
    wrong.
    """

    def __init__(self, docs):
        self.docs = docs

    def find_one(self, query, projection=None):
        for d in self.docs:
            if d.get("handle") == query.get("handle"):
                return dict(d)
        return None

    def find(self, query, projection=None):
        pt = query.get("bbox.min.0", {}).get("$lte")
        py = query.get("bbox.min.1", {}).get("$lte")
        for d in self.docs:
            box = d.get("bbox")
            if box is None:
                continue
            if box["min"][0] <= pt and box["min"][1] <= py:
                yield dict(d)


@pytest.fixture
def patched(monkeypatch):
    def install(docs):
        monkeypatch.setattr(store_spatial, "coll", lambda name: FakeCollection(docs))
        monkeypatch.setattr(
            store_spatial.landuse_store, "classify_layers", lambda *a, **k: {}
        )
        monkeypatch.setattr(
            store_spatial,
            "_scope_of",
            lambda *a, **k: type("S", (), {"units": {"area_unit": "m2"}})(),
        )
    return install


# --- core properties --------------------------------------------------------


def test_a_label_finds_the_parcel_it_sits_in(patched):
    plot = rect(E0, N0, 12.0, 25.0, math.radians(49.0))
    docs = [ring_doc("P1", plot, "plots", area=300.0), label_doc("L1", E0, N0)]
    patched(docs)

    out = store_spatial.parcels_containing("d", "L1")
    assert [r["handle"] for r in out["contained_by"]] == ["P1"]
    assert out["total"] == 1
    assert out["contained_by"][0]["area"] == 300.0
    assert out["contained_by"][0]["area_unit"] == "m2"
    assert out["selected"]["handle"] == "L1"


def test_a_label_outside_every_parcel_answers_zero_not_null(patched):
    """Zero means checked and there is none. Null means it could not be checked.

    Two different claims, and swapping them is the cheapest way to make "there
    is no school here" sound the same as "I did not look".
    """
    plot = rect(E0, N0, 12.0, 25.0)
    docs = [ring_doc("P1", plot, "plots", area=300.0), label_doc("L1", E0 + 500, N0 + 500)]
    patched(docs)

    out = store_spatial.parcels_containing("d", "L1")
    assert out["contained_by"] == []
    assert out["total"] == 0
    assert out.get("not_measured") is None


def test_nested_containers_all_come_back_smallest_first(patched):
    """A plot number sits in its plot AND in its block AND in its district."""
    docs = [
        ring_doc("BLOCK", rect(E0, N0, 200.0, 200.0), "blocks", area=40000.0),
        ring_doc("PLOT", rect(E0, N0, 12.0, 25.0), "plots", area=300.0),
        ring_doc("DISTRICT", rect(E0, N0, 900.0, 900.0), "districts", area=810000.0),
        label_doc("L1", E0, N0),
    ]
    patched(docs)

    out = store_spatial.parcels_containing("d", "L1")
    assert [r["handle"] for r in out["contained_by"]] == ["PLOT", "BLOCK", "DISTRICT"]
    assert out["total"] == 3
    assert "smallest" in out["nesting_note"].casefold()


def test_a_container_without_a_stored_area_sorts_last_not_first(patched):
    """If a missing area is read as zero, it WINS as the smallest precisely
    because it was not measured — and the first answer, the one most often
    read, becomes the least known one."""
    docs = [
        ring_doc("UNMEASURED", rect(E0, N0, 60.0, 60.0), "x", area=None),
        ring_doc("PLOT", rect(E0, N0, 12.0, 25.0), "plots", area=300.0),
        label_doc("L1", E0, N0),
    ]
    patched(docs)

    out = store_spatial.parcels_containing("d", "L1")
    assert [r["handle"] for r in out["contained_by"]] == ["PLOT", "UNMEASURED"]
    assert out["contained_by"][1]["area"] is None
    assert out["contained_by"][1]["area_quantity"]["withheld_reason"]


def test_a_label_with_no_insertion_point_says_so_instead_of_answering(patched):
    doc = label_doc("L1", 0, 0)
    doc["anchor_point"] = None
    doc["anchor_basis"] = None
    patched([ring_doc("P1", rect(E0, N0, 12.0, 25.0), "plots", area=300.0), doc])

    out = store_spatial.parcels_containing("d", "L1")
    assert out["total"] is None
    assert "insertion point" in out["not_measured"]
    assert "bounding box" in out["not_measured"], "must say why the bbox is not a fallback"


def test_a_point_on_the_plot_line_is_inside_it(patched):
    """A plot number the drafter snapped to its plot line is not an edge case.

    It is how the drawing was made, and a test that drops it loses its label
    without a single sign.
    """
    plot = rect(E0, N0, 12.0, 25.0, math.radians(49.0))
    corner = plot[0]
    docs = [ring_doc("P1", plot, "plots", area=300.0), label_doc("L1", corner[0], corner[1])]
    patched(docs)

    out = store_spatial.parcels_containing("d", "L1")
    assert [r["handle"] for r in out["contained_by"]] == ["P1"]
    assert out["contained_by"][0]["where"] == "boundary"
    assert out["boundary_cases"] == 1


# --- generality: what makes it hold on the 19th drawing ---------------------


def test_layers_are_never_consulted_to_decide_what_is_a_container(patched):
    """The layer name here deliberately means nothing in any language.

    If this function ever learns that "plot" or "parcel" is special, this test
    is the one that falls — and the next drawing, coming from another
    contractor with its own convention, will answer empty for no reason.
    """
    docs = [
        ring_doc("P1", rect(E0, N0, 12.0, 25.0), "حدود", area=300.0),
        label_doc("L1", E0, N0),
    ]
    patched(docs)
    out = store_spatial.parcels_containing("d", "L1")
    assert [r["handle"] for r in out["contained_by"]] == ["P1"]


def test_a_polygon_that_is_not_a_complete_ring_is_never_a_container(patched):
    """An arced polygon is its chord, not the real boundary, and an open
    polygon has no inside. Both answer "inside" confidently for the wrong
    shape."""
    for status in ("bulge", "open", "oversize", "non_planar", "degenerate", "unreadable"):
        ring = ring_doc("P1", rect(E0, N0, 12.0, 25.0), "x", area=300.0)
        ring["ring_status"] = status
        patched([ring, label_doc("L1", E0, N0)])
        out = store_spatial.parcels_containing("d", "L1")
        assert out["contained_by"] == [], f"{status} must never be a container"
        assert out["total"] == 0


def test_containers_in_another_layout_are_not_offered(patched):
    """Modelspace and sheets do not share a coordinate frame, so a point that
    falls inside a polygon of another layout is an arithmetic coincidence."""
    docs = [
        ring_doc("SHEET", rect(E0, N0, 90.0, 90.0), "x", layout="Layout1", area=8100.0),
        ring_doc("MODEL", rect(E0, N0, 12.0, 25.0), "x", layout="Model", area=300.0),
        label_doc("L1", E0, N0, layout="Model"),
    ]
    patched(docs)
    out = store_spatial.parcels_containing("d", "L1")
    assert [r["handle"] for r in out["contained_by"]] == ["MODEL"]


def test_an_entity_is_never_its_own_container(patched):
    """A polygon contains its own insertion point, and answering "you are
    inside yourself" is an answer that is arithmetically correct and of no use
    whatsoever."""
    plot = rect(E0, N0, 12.0, 25.0)
    doc = ring_doc("P1", plot, "plots", area=300.0)
    doc["anchor_point"] = [E0, N0]
    doc["anchor_basis"] = "bbox_centre"
    patched([doc])
    out = store_spatial.parcels_containing("d", "P1")
    assert out["contained_by"] == []


def test_an_unknown_handle_is_refused_rather_than_answered_empty(patched):
    patched([ring_doc("P1", rect(E0, N0, 12.0, 25.0), "x", area=300.0)])
    with pytest.raises(store_spatial.SpatialRefused) as exc:
        store_spatial.parcels_containing("d", "NOPE")
    assert exc.value.as_dict()["code"] == "ENTITY_UNKNOWN"


def test_the_answer_survives_being_far_from_the_origin(patched):
    """The same property tested at the origin and at projected coordinates.

    If the edge test were ever built on an absolute tolerance, the one far from
    the origin is the one that falls first — and that is the only one
    production uses.
    """
    for cx, cy in ((0.0, 0.0), (E0, N0), (1e7, -1e7)):
        plot = rect(cx, cy, 12.0, 25.0, math.radians(31.0))
        patched([ring_doc("P1", plot, "x", area=300.0), label_doc("L1", cx, cy)])
        out = store_spatial.parcels_containing("d", "L1")
        assert [r["handle"] for r in out["contained_by"]] == ["P1"], f"failed at {cx}"


def test_a_polygon_is_tested_at_its_centroid_and_the_answer_says_so(patched):
    """A parcel has no insertion point, and refusing to answer for it means
    "which district is this school in" cannot be asked.

    What is guarded here is not only the answer but its LABEL: a centroid that
    is inside is not the same thing as the whole polygon being inside, and for
    concave shapes the two can differ.
    """
    inner = ring_doc("SCHOOL", rect(E0, N0, 70.0, 70.0), "x", area=4900.0)
    inner["anchor_point"] = None
    inner["polygon_centroid"] = [E0, N0]
    docs = [ring_doc("DISTRICT", rect(E0, N0, 900.0, 900.0), "y", area=810000.0), inner]
    patched(docs)

    out = store_spatial.parcels_containing("d", "SCHOOL")
    assert [r["handle"] for r in out["contained_by"]] == ["DISTRICT"]
    assert out["point_used"] == "the entity's polygon_centroid"
    assert "concave" in out["containment_note"], "the difference between a centroid and the whole polygon must be named"


def test_a_label_still_uses_its_insertion_point_not_a_centroid(patched):
    """The order is not taste: the insertion point is what the file really
    stores."""
    lab = label_doc("L1", E0, N0)
    lab["polygon_centroid"] = [E0 + 10_000, N0 + 10_000]
    patched([ring_doc("P1", rect(E0, N0, 12.0, 25.0), "x", area=300.0), lab])

    out = store_spatial.parcels_containing("d", "L1")
    assert out["point_used"] == "the entity's anchor_point"
    assert out["containment_note"] is None
    assert [r["handle"] for r in out["contained_by"]] == ["P1"]
