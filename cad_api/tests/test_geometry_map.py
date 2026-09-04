"""The map geometry endpoint: what it must never get wrong.

Written the way `test_parcels_containing.py` is written, and for the same
reason (D-084): a test that proves something by reading already-ingested
Janadriyah documents does not move when the code that produced them breaks.
There is not one Janadriyah number below. What is pinned is a set of
properties, each of which would still be the right property for the
seventeenth drawing whose layer conventions nobody has seen.

Four of them exist because they are the mistakes this particular endpoint is
positioned to make:

  * **lon/lat, not lat/lon.** `crs.to_lat_lon()` returns `(lat, lon)`; deck.gl
    and GeoJSON want `[lon, lat]`. There is exactly one swap in the module and
    a test that fails if it disappears or happens twice.
  * **Reprojection is display, measurement is not.** `area` must be byte-for-
    byte the same whether the vertices came back as degrees or as metres. A
    reprojected ring has a different area, and the moment those two travel
    together someone quotes the wrong one.
  * **A refusal, not plausible numbers.** Asked for lng/lat with no CRS, the
    endpoint must fail. Raw eastings formatted as degrees are not obviously
    wrong on screen — they are a valid-looking point in the Gulf of Guinea.
  * **Absence is reported, not implied.** A layout of INSERTs must come back
    saying INSERT has no outline, rather than as an empty list that reads as
    an empty site.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from app import geometry_map
from app.landuse import crs as crs_math


#: The projected magnitudes this project really works at. Every precision
#: defect in this repo appears here and none of them appears near the origin.
E0, N0 = 691942.0, 2750756.0

#: The zone the corpus uses. Named once; nothing below asserts on it.
EPSG = 32638


def rect(cx: float, cy: float, w: float, h: float) -> list[list[float]]:
    return [
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
        [cx - w / 2, cy + h / 2],
    ]


def ring_doc(handle: str, ring, *, layer="PLOTS", layout="Model", area=1.0):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return {
        "handle": handle,
        "type": "LWPOLYLINE",
        "layer": layer,
        "layout": layout,
        "ring": ring,
        "ring_status": "complete",
        "ring_orientation": "ccw",
        "ring_vertex_count": len(ring),
        "polygon_centroid": [sum(xs) / len(xs), sum(ys) / len(ys)],
        "bbox_centre": [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2],
        "area": area,
        "perimeter_from_ring": 2 * (max(xs) - min(xs) + max(ys) - min(ys)),
    }


def path_doc(handle: str, points, *, bulges=None, layer="ROADS", layout="Model"):
    return {
        "handle": handle,
        "type": "LWPOLYLINE",
        "layer": layer,
        "layout": layout,
        "path_points": points,
        "path_bulges": bulges,
        "ring_status": "open",
        "bbox_centre": points[0],
        "length": 10.0,
    }


def line_doc(handle: str, a, b, *, layer="ROADS", layout="Model"):
    """A LINE: two endpoints in WCS, and that IS the whole line."""
    return {
        "handle": handle,
        "type": "LINE",
        "layer": layer,
        "layout": layout,
        "ends": [list(a), list(b)],
        "ends_status": "open_path",
        "ends_basis": "LINE.start/LINE.end (WCS)",
        "bbox_centre": [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2],
        "length": math.dist(a, b),
    }


def arc_doc(handle: str, a, b, *, layer="ROADS", layout="Model"):
    """An ARC: its endpoints were computed from a centre and radius that were
    then NOT stored, so two points is all there is."""
    return {
        "handle": handle,
        "type": "ARC",
        "layer": layer,
        "layout": layout,
        "ends": [list(a), list(b)],
        "ends_status": "open_path",
        "ends_basis": "ARC.center + radius + start_angle/end_angle",
        "bbox_centre": [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2],
        # Deliberately longer than the straight distance: an arc is longer
        # than its chord, and that difference is the error being declared.
        "length": math.dist(a, b) * 1.11,
    }


def refused_ring_doc(handle: str, *, layer="block-circle", layout="Model"):
    """A CLOSED polyline the extractor refused to write a ring for.

    `ring_status: "bulge"` — it contains an arc span. No `ring`, no
    `path_points`, and a two-point `ring_diagnostic` that is not geometry.
    This is the shape that hides from a coverage report.
    """
    return {
        "handle": handle,
        "type": "LWPOLYLINE",
        "layer": layer,
        "layout": layout,
        "ring_status": "bulge",
        "ring": None,
        "path_points": None,
        "bbox_centre": [E0, N0],
    }


def insert_doc(handle: str, x: float, y: float, *, layout="Model"):
    """A block placement: a bounding box and nothing to draw.

    This is the shape of the gap §3.2 of the architecture note calls phase 5,
    and the reason `geometry_coverage` exists at all.
    """
    return {
        "handle": handle,
        "type": "INSERT",
        "layer": "BUILDINGS",
        "layout": layout,
        "block_name": "VILLA-A",
        "bbox_centre": [x, y],
    }


class FakeCursor:
    """The three cursor methods this module uses, and no more.

    Ordering is by `_id` because that is what the route asks for, and on these
    documents `_id` is `drawing_id:handle` with a single drawing, so it is the
    same order the route used to produce by sorting on `handle` here.
    """

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def sort(self, key, direction=1):
        self.docs = sorted(
            self.docs, key=lambda d: str(d.get(key) or ""), reverse=direction < 0
        )
        return self

    def skip(self, n: int):
        self.docs = self.docs[n:]
        return self

    def limit(self, n: int):
        self.docs = self.docs[:n]
        return self

    def __iter__(self):
        return iter(self.docs)


class FakeCollection:
    """Just enough MongoDB for this module, and no more.

    `aggregate` interprets only the four operators `_coverage` actually uses.
    It is a real interpreter rather than a hard-coded answer, so the pipeline
    below is genuinely exercised: change the `$cond` and the coverage test
    goes red. What it does not model is the *index*, and that is deliberate —
    the deployment's `notablescan` behaviour is a property of the server, and
    a fake that granted every query an index would quietly bless the one
    query shape that fails in production.
    """

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.queries: list[dict[str, Any]] = []

    # -- find ---------------------------------------------------------------

    def find(self, query, projection=None):
        self.queries.append(query)
        out = []
        for doc in self.docs:
            if not self._matches(doc, query):
                continue
            out.append(dict(doc) if projection is None else {
                k: v for k, v in doc.items() if k in projection or k == "_id"
            })
        # A cursor, because the module now asks the SERVER to sort and page
        # rather than materialising a layout and slicing it here. It used to
        # fetch every matching row, sort them in the process and take a slice
        # at the very end, so `limit` bought a smaller response and no less
        # work: 25 s for limit=4000 against 32 s unbounded on a 679,878 row
        # drawing.
        return FakeCursor(out)

    def count_documents(self, query, limit=0):
        """`total_matches` is counted, never inferred from a page.

        `limit` is part of the real signature and callers use it to ask "is
        there at least one of these?" without counting the rest. A double that
        omits it fails on the keyword rather than on the behaviour, which
        tells the reader nothing about the code under test.
        """
        self.queries.append(query)
        n = sum(1 for doc in self.docs if self._matches(doc, query))
        return min(n, limit) if limit else n

    def _matches(self, doc, query) -> bool:
        for key, cond in query.items():
            if key == "$or":
                if not any(self._matches(doc, clause) for clause in cond):
                    return False
                continue
            if key == "$and":
                # The scope rule and the kind filter are both `$or` clauses,
                # so the query joins them with `$and` rather than letting the
                # second silently overwrite the first.
                if not all(self._matches(doc, clause) for clause in cond):
                    return False
                continue
            if key in ("drawing_id",):
                continue  # the fake holds one drawing
            value = doc.get(key)
            if isinstance(cond, dict):
                if "$ne" in cond and value == cond["$ne"]:
                    return False
                if "$in" in cond and value not in cond["$in"]:
                    return False
            elif value != cond:
                return False
        return True

    # -- aggregate ----------------------------------------------------------

    def aggregate(self, pipeline, allowDiskUse=False):  # noqa: N803 -- driver's name
        """`allowDiskUse` is part of the real signature.

        A double that omits it fails on the keyword rather than on the
        behaviour, which tells a reader nothing about the code under test.
        """
        self.queries.append(pipeline)
        docs = list(self.docs)
        rows: list[dict[str, Any]] = []
        for stage in pipeline:
            if "$match" in stage:
                docs = [d for d in docs if self._matches(d, stage["$match"])]
            elif "$group" in stage:
                spec = stage["$group"]
                field = spec["_id"].lstrip("$")
                buckets: dict[Any, dict[str, Any]] = {}
                for d in docs:
                    row = buckets.setdefault(d.get(field), {"_id": d.get(field)})
                    for name, acc in spec.items():
                        if name == "_id":
                            continue
                        row[name] = row.get(name, 0) + self._sum_term(acc["$sum"], d)
                rows = list(buckets.values())
            elif "$sort" in stage:
                # Several keys, applied least-significant first. A one-key
                # version silently dropped the tie-break that makes a layer
                # list stably ordered.
                for key, direction in reversed(list(stage["$sort"].items())):
                    rows.sort(key=lambda r, k=key: r[k], reverse=direction < 0)
        return rows

    @staticmethod
    def _sum_term(term, doc) -> int:
        if term == 1:
            return 1
        cond, yes, no = term["$cond"]
        if "$ifNull" in cond:
            # {"$cond": [{"$ifNull": ["$field", False]}, 1, 0]}
            field = cond["$ifNull"][0].lstrip("$")
            return yes if doc.get(field) else no
        # {"$cond": [{"$eq": ["$field", value]}, 1, 0]}
        field, value = cond["$eq"]
        return yes if doc.get(field.lstrip("$")) == value else no


@pytest.fixture
def stub(monkeypatch):
    """Wire the module to a fake store, with a CRS present by default."""

    def install(docs, *, crs=True, layouts=("Model",), units=("m", 6), layers=()):
        collection = FakeCollection(docs)
        drawing = {
            "_id": "d1",
            "layouts": [{"name": n} for n in layouts],
            "units_name": units[0],
            "units_code": units[1],
            "layers": list(layers),
        }

        def coll(name):
            if name == geometry_map.mongo.COLL_DRAWINGS:
                return FakeCollectionOfOne(drawing)
            return collection

        config = (
            type(
                "Cfg",
                (),
                {"crs": crs_math.Crs(epsg=EPSG, name="WGS 84 / UTM zone 38N",
                                     declared_in_file=False)},
            )()
            if crs
            else type("Cfg", (), {"crs": None})()
        )
        monkeypatch.setattr(geometry_map.mongo, "coll", coll)
        # `preconditions` reads the drawing through `store`, which is a real
        # Mongo call and is not what these tests are about. The ETag's own
        # behaviour — the header parsing, the 304 — is exercised against the
        # running route instead.
        monkeypatch.setattr(
            geometry_map.geo_view,
            "preconditions",
            lambda _id: {
                "config_version": 1,
                "stored_resolution": 13,
                "ingest_version": 1,
            },
        )
        monkeypatch.setattr(geometry_map.landuse, "for_drawing", lambda _id: config)
        monkeypatch.setattr(
            geometry_map.store_landuse, "classify_layers", lambda _id, names: {}
        )
        return collection

    return install


class FakeCollectionOfOne:
    def __init__(self, doc):
        self.doc = doc

    def find_one(self, _query, projection=None):
        """`projection` is part of the real signature and the route uses it.

        Honoured rather than ignored: the route projects AWAY the fields it
        never reads -- on Sedra they are 147 KB of a 387 KB document -- and a
        double that returned everything regardless would let a test pass while
        the code read a field it had asked the database not to send.
        """
        if not projection:
            return self.doc
        excluded = {k for k, v in projection.items() if not v and k != "_id"}
        if excluded:
            return {k: v for k, v in self.doc.items() if k not in excluded}
        included = {k for k, v in projection.items() if v}
        return {
            k: v for k, v in self.doc.items() if k in included or k == "_id"
        }


def call(**kwargs):
    params = {
        "drawing_id": "d1",
        "layout": "Model",
        "frame": "auto",
        "kinds": geometry_map.DEFAULT_KINDS,
        "layers": None,
        "limit": geometry_map.DEFAULT_LIMIT,
        "offset": 0,
        # Spelled out because this calls the function rather than the route:
        # FastAPI resolves `Query(None)` to None, and a direct call would
        # otherwise hand the `Query` object itself straight through.
        "bbox": None,
        "request": None,
        "response": None,
    }
    params.update(kwargs)
    return geometry_map.drawing_geometry(**params)


# --- the swap that is always wrong -------------------------------------------


def test_lnglat_is_longitude_first(stub):
    """A metre further EAST must move the first number, not the second.

    The single most repeated bug in any project that handles both orders.
    Asserting on the *direction of change* rather than on a coordinate keeps
    this test correct in any zone and on any drawing.
    """
    stub([ring_doc("A", rect(E0, N0, 2, 2)), ring_doc("B", rect(E0 + 100, N0, 2, 2))])
    out = call(frame="lnglat")
    west, east = (f["centroid"] for f in sorted(out["features"], key=lambda f: f["handle"]))

    assert east[0] > west[0], "100 m east must increase longitude"
    assert math.isclose(east[1], west[1], abs_tol=1e-4), (
        "moving due east must not meaningfully change latitude"
    )
    # And the values must actually be degrees, not eastings wearing the name.
    assert -180 <= west[0] <= 180 and -90 <= west[1] <= 90


def test_world_frame_returns_the_stored_vertices_untouched(stub):
    """No arithmetic at all on the `world` path.

    `frame=world` exists so a drawing with no CRS can still be drawn. If any
    transform leaked into it, that view would be subtly wrong with nothing to
    compare it against.
    """
    ring = rect(E0, N0, 12, 25)
    stub([ring_doc("A", ring)])
    out = call(frame="world")
    assert out["features"][0]["coordinates"] == ring


# --- display is not measurement ----------------------------------------------


def test_area_is_identical_in_both_frames(stub):
    """The whole reason `crs.py` refuses to reproject geometry.

    A reprojected ring has different side lengths and a different area. This
    endpoint reprojects for *drawing*, and the measured figures must come back
    from the store unchanged — otherwise two numbers for one plot exist and
    one of them is wrong.
    """
    stub([ring_doc("A", rect(E0, N0, 12, 25), area=300.0)])
    world = call(frame="world")["features"][0]
    degrees = call(frame="lnglat")["features"][0]

    assert world["area"] == degrees["area"] == 300.0
    assert world["perimeter"] == degrees["perimeter"]
    assert world["coordinates"] != degrees["coordinates"], "the sanity half"


def test_the_response_says_which_frame_and_that_it_is_not_measurable(stub):
    stub([ring_doc("A", rect(E0, N0, 12, 25))])
    out = call(frame="lnglat")
    assert out["frame"] == "lnglat"
    assert "not measurable" in out["frame_basis"].lower()


# --- refusing, rather than producing plausible numbers ------------------------


def test_lnglat_without_a_crs_is_refused(stub):
    """Not a silent fallback to `world`, and certainly not degrees-shaped
    eastings: both would put a drawing somewhere definite and wrong."""
    stub([ring_doc("A", rect(E0, N0, 2, 2))], crs=False)
    with pytest.raises(Exception) as caught:
        call(frame="lnglat")
    assert getattr(caught.value, "code", None) == "CRS_NOT_CONFIGURED"
    assert "crs:" in getattr(caught.value, "hint", "")


def test_auto_falls_back_to_world_and_says_why(stub):
    stub([ring_doc("A", rect(E0, N0, 2, 2))], crs=False)
    out = call(frame="auto")
    assert out["frame"] == "world"
    assert out["placement"]["can_georeference"] is False
    assert out["placement"]["why_not"], "a refusal with no reason is unactionable"


def test_a_drawing_that_is_not_in_metres_cannot_be_georeferenced(stub):
    """`crs.py` reads its input as metres. A drawing in inches with a CRS
    block is a contradiction, and it has to be reported as one — the corpus is
    mostly inches, so this is the common case, not the exotic one."""
    stub([ring_doc("A", rect(E0, N0, 2, 2))], units=("in", 1))
    out = call(frame="auto")
    assert out["placement"]["can_georeference"] is False
    assert any("metres" in r for r in out["placement"]["why_not"])


def test_an_unknown_layout_is_refused_rather_than_answered_empty(stub):
    """An empty feature list for a typo'd layout reads as 'this drawing has no
    geometry', which is the most misleading answer available here."""
    stub([ring_doc("A", rect(E0, N0, 2, 2))])
    with pytest.raises(Exception) as caught:
        call(layout="Modle")
    assert getattr(caught.value, "code", None) == "LAYOUT_NOT_FOUND"
    assert "Model" in getattr(caught.value, "hint", "")


# --- absence is reported, not implied ----------------------------------------


def test_a_layout_of_block_placements_says_insert_has_no_outline(stub):
    """The gap that decides whether a map can show buildings.

    Fifteen villas that are fifteen INSERTs are fifteen documents with a
    bounding box and nothing to draw. An empty `features` list on its own is
    indistinguishable from an empty site.
    """
    stub([insert_doc("A", E0, N0), insert_doc("B", E0 + 30, N0)])
    out = call()
    assert out["features"] == []
    coverage = out["geometry_coverage"]
    assert coverage["entities_in_layout"] == 2
    assert coverage["with_outline"] == 0
    assert coverage["without_outline"] == 2
    assert "INSERT" in coverage["types_with_no_outline"]


def test_coverage_counts_both_halves_of_a_mixed_layout(stub):
    stub([
        ring_doc("A", rect(E0, N0, 12, 25)),
        path_doc("B", [[E0, N0], [E0 + 10, N0]]),
        insert_doc("C", E0, N0),
    ])
    coverage = call()["geometry_coverage"]
    by_type = {r["type"]: r for r in coverage["by_type"]}
    assert by_type["LWPOLYLINE"]["with_outline"] == 2
    assert by_type["INSERT"]["with_outline"] == 0
    assert coverage["with_outline"] == 2 and coverage["without_outline"] == 1


def test_bulges_travel_with_the_path_and_are_declared(stub):
    """A path drawn straight between bulged vertices is a chord, not an arc.

    The endpoint does not flatten them — that is a real decision, and the only
    thing that makes it survivable is that the consumer can tell.
    """
    stub([path_doc("A", [[E0, N0], [E0 + 10, N0]], bulges=[0.41])])
    feature = call()["features"][0]
    assert feature["bulges"] == [0.41]
    assert feature["has_bulge"] is True


# --- the list conventions the rest of cad-api uses ----------------------------


def test_pagination_is_stable_and_disjoint(stub):
    """`offset` has to page a list with a fixed order.

    Without the sort the order is whatever the index returned, and two pages
    of one drawing can then both contain — or both miss — the same plot.
    """
    stub([ring_doc(f"{i:03X}", rect(E0 + i, N0, 2, 2)) for i in range(10)])
    first = call(limit=4, offset=0)
    second = call(limit=4, offset=4)
    rest = call(limit=4, offset=8)

    assert first["total_matches"] == 10
    assert first["truncated"] is True and first["next_offset"] == 4
    assert rest["truncated"] is False and rest["next_offset"] is None

    seen = [f["handle"] for f in first["features"] + second["features"] + rest["features"]]
    assert len(set(seen)) == 10
    assert seen == sorted(seen), "the order is by handle, and it is the same every call"


def test_labels_are_off_unless_asked_for(stub):
    """Janadriyah's Model carries 5,518 TEXT against 3,169 outlines. Sent
    unconditionally they would nearly double the payload to draw annotations
    most callers do not want."""
    docs = [
        ring_doc("A", rect(E0, N0, 12, 25)),
        {
            "handle": "B",
            "type": "TEXT",
            "layer": "LABELS",
            "layout": "Model",
            "text": "17",
            "anchor_point": [E0, N0],
        },
    ]
    stub(docs)
    assert call()["counts"]["label"] == 0
    with_labels = call(kinds="ring,label")
    assert with_labels["counts"]["label"] == 1
    assert with_labels["features"][-1]["text"] == "17"


def test_a_layer_filter_narrows_without_changing_the_coverage_denominator(stub):
    """Coverage answers "what is in this layout", not "what did you just ask
    for". A denominator that moved with the filter would make a layer look
    fully covered whenever it was the only one requested."""
    stub([
        ring_doc("A", rect(E0, N0, 12, 25), layer="PLOTS"),
        ring_doc("B", rect(E0 + 40, N0, 12, 25), layer="OTHER"),
        insert_doc("C", E0, N0),
    ])
    out = call(layers="PLOTS")
    assert [f["handle"] for f in out["features"]] == ["A"]
    assert out["geometry_coverage"]["entities_in_layout"] == 3


def test_unknown_kinds_are_refused_with_the_list_of_known_ones(stub):
    stub([ring_doc("A", rect(E0, N0, 2, 2))])
    with pytest.raises(Exception) as caught:
        call(kinds="polygons")
    assert getattr(caught.value, "code", None) == "BAD_KINDS"
    assert "ring" in getattr(caught.value, "hint", "")


# --- geometry that was stored and unreachable --------------------------------


def test_a_line_is_returned_as_its_two_endpoints(stub):
    """The gap that made a road network draw as a fifth of itself.

    `ends` for a LINE is `LINE.start`/`LINE.end` in WCS. That is not an
    approximation of the line, it IS the line — 16,011 of them across the
    corpus were stored and had no way out over HTTP.
    """
    a, b = [E0, N0], [E0 + 40, N0 + 30]
    stub([line_doc("A", a, b)])
    out = call(frame="world")
    feature = out["features"][0]

    assert feature["kind"] == "segment"
    assert feature["coordinates"] == [a, b]
    assert feature["is_chord"] is False
    assert feature["ends_basis"] == "LINE.start/LINE.end (WCS)"
    assert out["counts"]["segment"] == 1


def test_an_arc_is_returned_but_declares_itself_a_chord(stub):
    """Drawn, and declared. Withholding 450 arcs leaves a road network full of
    gaps with nothing on screen to explain them; drawing them silently asserts
    a straight line through the inside of every bend. Only the flag makes the
    first option honest."""
    stub([arc_doc("A", [E0, N0], [E0 + 14, N0 + 1])])
    feature = call()["features"][0]

    assert feature["kind"] == "segment"
    assert feature["is_chord"] is True
    assert len(feature["coordinates"]) == 2
    # The stored length is the ARC's, not the chord's. A consumer that measured
    # the two returned points would get a different, smaller number — which is
    # exactly why the flag has to travel with them.
    assert feature["length"] > math.dist(*feature["coordinates"])


def test_segments_are_on_by_default(stub):
    """Unlike labels. A line is geometry the drawing asserts; a text anchor is
    an annotation most callers do not want, and the two should not share a
    default."""
    assert "segment" in geometry_map.DEFAULT_KINDS
    assert "label" not in geometry_map.DEFAULT_KINDS


def test_a_refused_ring_is_counted_where_nothing_else_could_see_it(stub):
    """The gap that hides from `types_with_no_outline`.

    A closed polyline containing an arc span gets no `ring` and no
    `path_points`. Its TYPE is LWPOLYLINE, which mostly does have outlines, so
    the by-type report can never name it — 1,291 shapes corpus-wide were absent
    from a coverage report that looked complete.
    """
    stub([
        ring_doc("A", rect(E0, N0, 12, 25)),
        refused_ring_doc("B"),
        refused_ring_doc("C"),
    ])
    out = call()
    coverage = out["geometry_coverage"]

    assert len(out["features"]) == 1, "the refused ones cannot be drawn"
    assert coverage["rings_refused"] == 2
    assert coverage["rings_refused_reason"]
    # The point of the test: the by-type view alone would report nothing wrong.
    assert "LWPOLYLINE" not in coverage["types_with_no_outline"]


def test_no_refusals_reports_none_rather_than_zero_with_a_reason(stub):
    """A reason attached to a count of zero reads as a problem that exists."""
    stub([ring_doc("A", rect(E0, N0, 12, 25))])
    coverage = call()["geometry_coverage"]
    assert coverage["rings_refused"] == 0
    assert coverage["rings_refused_reason"] is None


# --- layers the FILE switches off --------------------------------------------


def test_layers_the_file_switches_off_are_named(stub):
    """The fix for a bug that was invisible from either view alone.

    A caller drawing model space while looking at a paper sheet cannot learn
    this from the SVG: the sheet's `data-off-layers` is empty, because its
    viewport already excludes those layers. So the map drew eleven layers
    AutoCAD keeps switched off — on the real drawing, the road centrelines and
    every neighbourhood boundary, hundreds of objects scattered kilometres from
    the site. It read as a projection fault and was a visibility one.
    """
    stub(
        [ring_doc("A", rect(E0, N0, 12, 25), layer="PLOTS")],
        layers=[
            {"name": "PLOTS", "off": False, "frozen": False},
            {"name": "ROAD-CL", "off": True, "frozen": False},
            {"name": "OLD-REV", "off": False, "frozen": True},
        ],
    )
    out = call()
    names = {row["name"] for row in out["layers_off"]}

    assert names == {"ROAD-CL", "OLD-REV"}, "off AND frozen both mean not shown"
    assert {r["name"]: r["off"] for r in out["layers_off"]}["ROAD-CL"] is True
    assert {r["name"]: r["frozen"] for r in out["layers_off"]}["OLD-REV"] is True
    assert "viewport" in out["layers_off_basis"]


def test_features_on_a_switched_off_layer_are_still_returned(stub):
    """Hidden is a VIEW decision, and the endpoint does not get to make it.

    Dropping them here would make those objects unreachable rather than
    unshown: no way to toggle the layer back on, no way to select one, no way
    for the agent to cite one. The caller is told which layers are off and
    decides what to do about it.
    """
    stub(
        [
            ring_doc("A", rect(E0, N0, 12, 25), layer="PLOTS"),
            ring_doc("B", rect(E0 + 4000, N0, 12, 25), layer="ROAD-CL"),
        ],
        layers=[{"name": "ROAD-CL", "off": True, "frozen": False}],
    )
    out = call()
    assert {f["handle"] for f in out["features"]} == {"A", "B"}
    assert [r["name"] for r in out["layers_off"]] == ["ROAD-CL"]


def test_a_drawing_with_no_off_layers_says_so_with_an_empty_list(stub):
    stub([ring_doc("A", rect(E0, N0, 12, 25))], layers=[
        {"name": "PLOTS", "off": False, "frozen": False},
    ])
    assert call()["layers_off"] == []


# --- the viewport ------------------------------------------------------------
#
#  Aligned with `/geo`, deliberately: the same idea of what a viewport does to
#  a payload and what it must never do to a total. The one difference is which
#  frame the box is in, and it is a difference this endpoint cannot avoid --
#  it answers in drawing coordinates for every drawing without a CRS.


def test_a_viewport_keeps_what_meets_it_and_drops_what_does_not(stub):
    stub(
        [
            ring_doc("NEAR", rect(E0, N0, 10, 10)),
            ring_doc("FAR", rect(E0 + 5000, N0 + 5000, 10, 10)),
        ]
    )
    out = call(frame="world", bbox=f"{E0 - 50},{N0 - 50},{E0 + 50},{N0 + 50}")
    assert [f["handle"] for f in out["features"]] == ["NEAR"]
    assert out["viewport"]["matches_in_view"] == 1
    assert out["viewport"]["matches_outside"] == 1


def test_a_viewport_does_not_restrict_the_total(stub):
    """The rule `/geo` states and this one inherits: a filtered view must
    never be readable as the drawing."""
    stub(
        [
            ring_doc("NEAR", rect(E0, N0, 10, 10)),
            ring_doc("FAR", rect(E0 + 5000, N0 + 5000, 10, 10)),
        ]
    )
    out = call(frame="world", bbox=f"{E0 - 50},{N0 - 50},{E0 + 50},{N0 + 50}")
    assert out["returned"] == 1
    assert out["total_matches"] == 2, "the total is the layout, not the view"
    assert "NOT" in out["viewport"]["note"]


def test_an_object_larger_than_the_viewport_is_still_in_it(stub):
    """Overlap, not containment.

    A parcel that contains the whole box, or a road that crosses it, is what
    somebody at that zoom is looking AT. Requiring containment empties the
    screen exactly when the view gets close.
    """
    stub([ring_doc("HUGE", rect(E0, N0, 4000, 4000))])
    out = call(frame="world", bbox=f"{E0 - 5},{N0 - 5},{E0 + 5},{N0 + 5}")
    assert [f["handle"] for f in out["features"]] == ["HUGE"]


def test_an_empty_viewport_is_an_answer_rather_than_an_error(stub):
    stub([ring_doc("A", rect(E0, N0, 10, 10))])
    out = call(frame="world", bbox=f"{E0 + 9000},{N0 + 9000},{E0 + 9100},{N0 + 9100}")
    assert out["features"] == []
    assert out["total_matches"] == 1
    assert out["viewport"]["empty_is_an_answer"]


def test_the_viewport_is_read_in_the_frame_the_answer_is_in(stub):
    """Degrees when the answer is degrees, drawing units when it is not.

    The box that selects the object in `world` selects nothing in `lnglat`,
    because 691942 is not a longitude -- which is the whole reason the frame
    has to be stated rather than assumed.
    """
    stub([ring_doc("A", rect(E0, N0, 10, 10))])
    world_box = f"{E0 - 50},{N0 - 50},{E0 + 50},{N0 + 50}"

    in_world = call(frame="world", bbox=world_box)
    assert [f["handle"] for f in in_world["features"]] == ["A"]
    assert in_world["viewport"]["frame"] == "world"
    assert "drawing" in in_world["viewport"]["bbox_order"]

    in_degrees = call(frame="lnglat", bbox=world_box)
    assert in_degrees["features"] == []

    # And the same object IS found by a box written in degrees.
    lon, lat = call(frame="lnglat")["features"][0]["centroid"]
    found = call(
        frame="lnglat",
        bbox=f"{lon - 0.001},{lat - 0.001},{lon + 0.001},{lat + 0.001}",
    )
    assert [f["handle"] for f in found["features"]] == ["A"]
    assert found["viewport"]["frame"] == "lnglat"
    assert "degrees" in found["viewport"]["bbox_order"]


@pytest.mark.parametrize(
    "raw", ["1,2,3", "a,b,c,d", "10,10,10,20", "5,5,1,1", ""]
)
def test_a_malformed_viewport_is_refused_not_silently_emptied(stub, raw):
    """A box that parsed to nothing would return an empty payload that looks
    exactly like a viewport over empty ground."""
    stub([ring_doc("A", rect(E0, N0, 10, 10))])
    with pytest.raises(Exception) as caught:
        call(frame="world", bbox=raw)
    # Same idiom as the frame refusal above: the code, and a hint that says
    # what to send instead rather than restating the problem.
    assert getattr(caught.value, "code", None) == "BAD_BBOX"
    assert "minX,minY,maxX,maxY" in getattr(caught.value, "hint", "")


def test_without_a_viewport_nothing_changes(stub):
    stub([ring_doc("A", rect(E0, N0, 10, 10)), ring_doc("B", rect(E0 + 40, N0, 10, 10))])
    out = call()
    assert out["viewport"] is None
    assert out["returned"] == out["total_matches"] == 2
    assert out["truncated"] is False


# --- one boolean rather than three fields to combine -------------------------


def test_the_answer_says_whether_the_drawing_is_georeferenced(stub):
    stub([ring_doc("A", rect(E0, N0, 10, 10))])
    assert call()["georeferenced"] is True
    assert call()["georeferenced_reason"] is None


def test_a_drawing_with_no_crs_says_so_and_still_returns_its_geometry(stub):
    """`/geo` cannot answer without a CRS. This one can, and the difference
    has to be legible: `georeferenced: false` with a reason, and the drawing
    still drawn in its own coordinates."""
    stub([ring_doc("A", rect(E0, N0, 10, 10))], crs=False)
    out = call()
    assert out["georeferenced"] is False
    assert "no coordinate system" in out["georeferenced_reason"]
    assert out["frame"] == "world"
    assert out["returned"] == 1, "a drawing with no CRS is still drawable"


def test_the_layer_list_is_the_scope_not_whatever_was_fetched():
    """Membership must come from the store, not from the pages in hand.

    The panel used to build its list from `geometry.features`, so rows
    appeared as the map paged. Measured on Sedra that was not a small error:
    the first 24,000 features by `_id` all sit on ONE layer while the scope
    holds 215, so most of what the user saw was the SHEET's layer list -- the
    sheet-versus-map conflation surfacing in the UI.

    Guarded as an independence property rather than as a corpus number: the
    answer must not move when the paging does.
    """
    entities = FakeCollection(
        [
            {
                "_id": i,
                "drawing_id": "d",
                "layout": "Model",
                "layer": layer,
                "type": "LWPOLYLINE",
                "ring": [[0, 0], [1, 0], [1, 1]],
            }
            # A layer that appears only LATE in `_id` order is the whole
            # point: a page-derived list has not met it yet.
            for i, layer in enumerate(["EARLY"] * 50 + ["LATE"] * 2)
        ]
    )
    query = {"drawing_id": "d", "layout": "Model"}

    first = geometry_map.layers_in_scope(entities, query, token="v1")
    assert {row["name"]: row["count"] for row in first} == {"EARLY": 50, "LATE": 2}

    # Asked again within one version: the same answer, and no second scan.
    before = len(entities.queries)
    assert geometry_map.layers_in_scope(entities, query, token="v1") == first
    assert len(entities.queries) == before, (
        "a cached layer list must not re-aggregate: it is 2.38 s on the "
        "proving drawing and cannot change within one version"
    )

    # A new version is a new question, and must not be served the old answer.
    entities.docs.append(
        {
            "_id": 99,
            "drawing_id": "d",
            "layout": "Model",
            "layer": "NEW",
            "type": "LWPOLYLINE",
            "ring": [[0, 0], [1, 0], [1, 1]],
        }
    )
    assert {
        row["name"] for row in geometry_map.layers_in_scope(entities, query, token="v2")
    } == {"EARLY", "LATE", "NEW"}


def test_the_layers_parameter_can_exclude_as_well_as_include():
    """`!NAME` hides a layer; a bare name isolates one.

    Both in the one parameter because the two have wildly different sizes on a
    real drawing. Isolating three layers is a short list; hiding ONE of
    Sedra's 215 would need the other 214 spelled out, which url-encodes to
    about 12 KB -- past what a request line carries. `!EXT-BASE` says it in
    one term, and EXT-BASE is the case that matters: 440,340 of the scope's
    610,944 objects, first in `_id` order, so every capped fetch came back
    4,000 of 4,000 EXT-BASE and unchecking it in the panel changed nothing
    that had already been fetched.
    """
    assert geometry_map._layer_filter(None) is None
    assert geometry_map._layer_filter("  ,  ") is None

    only = geometry_map._layer_filter("A,B")
    assert only == ([("A"), ("B")], [])
    assert (only.include, only.exclude) == (["A", "B"], [])

    without = geometry_map._layer_filter("!EXT-BASE")
    assert (without.include, without.exclude) == ([], ["EXT-BASE"])

    both = geometry_map._layer_filter("A,!B")
    assert (both.include, both.exclude) == (["A"], ["B"])

    # A layer whose name really starts with "!" is escaped, so the filter can
    # name it rather than silently hiding the wrong thing. AutoCAD layer names
    # permit almost anything.
    escaped = geometry_map._layer_filter("!!weird")
    assert (escaped.include, escaped.exclude) == (["!weird"], [])

    # And the Mongo clause says the same thing.
    q = geometry_map._mongo_filter("d", "Model", ("ring",), without)
    assert q["layer"] == {"$nin": ["EXT-BASE"]}
    q = geometry_map._mongo_filter("d", "Model", ("ring",), only)
    assert q["layer"] == {"$in": ["A", "B"]}
    q = geometry_map._mongo_filter("d", "Model", ("ring",), both)
    assert q["layer"] == {"$in": ["A"], "$nin": ["B"]}


def test_keeps_layer_agrees_with_the_clause_it_stands_for():
    """The filtered total is SUMMED from the cached per-layer counts.

    `count_documents` over a `$nin` is 22 s on the proving drawing while the
    `find` itself is 67 ms, and every distinct checkbox state is a new cache
    key -- so counting would be paid again on every click. Summing is exact
    only while this predicate matches what Mongo does, which is what is
    asserted here rather than assumed.
    """
    keeps = geometry_map.keeps_layer
    assert keeps("ANY", None) is True

    without = geometry_map._layer_filter("!EXT-BASE")
    assert keeps("EXT-BASE", without) is False
    assert keeps("RD-PRK", without) is True

    only = geometry_map._layer_filter("A,B")
    assert keeps("A", only) is True
    assert keeps("C", only) is False

    both = geometry_map._layer_filter("A,B,!B")
    # Excluded wins, exactly as `$in` and `$nin` together intersect.
    assert keeps("A", both) is True
    assert keeps("B", both) is False
