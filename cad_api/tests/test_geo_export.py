"""Wave 3 W1 — the file somebody drags into a map.

The difference between this and `/geo` is not the shape of the JSON, it is who
reads it. `/geo` is answered to a client that also has the route, the OpenAPI
page and somebody to ask. A FILE gets forwarded: it arrives on its own, and
whatever it needs to be read correctly has to be inside it.

So the tests here are mostly about what travels WITH the geometry — a colour on
every feature so it renders with no style file, and the coordinate-system
caveat on every feature so the one thing that could make it misleading cannot
be separated from it.
"""

from __future__ import annotations

import json

import pytest

h3 = pytest.importorskip("h3")

from app import geo_view, main, store
from conftest import unconfigured_drawing_ids
from app.main import ApiError

JANADRIYAH = "596212db022a3397"
MODEL = "Model"

#: A box holding the mosque and the plots around it. Small enough that the
#: whole file can be compared byte for byte without a slow test.
SMALL_BOX = "46.8930,24.8508,46.8945,24.8522"


def _mongo_or_skip():
    try:
        drawing = store.get_drawing(JANADRIYAH)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip("the reference drawing has not been ingested")
    return drawing


def _export(drawing_id: str = JANADRIYAH, **kwargs):
    args = dict(
        res=None, layout=MODEL, cells=True, parcels=True, bbox=None,
        limit=geo_view.MAX_PARCEL_FEATURES, handles=None, rings=0,
        request=None, response=None,
    )
    args.update(kwargs)
    result = main.drawing_geo_export(drawing_id, **args)
    # The route returns a JSONResponse so it can set headers; the body is what
    # these tests are about.
    return json.loads(result.body)


def _indexed_or_skip():
    _mongo_or_skip()
    body = _export(res=9, parcels=False, bbox=SMALL_BOX)
    if not body.get("features"):
        pytest.skip("no cells yet; run scripts/backfill_h3.py")


@pytest.fixture(scope="module")
def small():
    # `_indexed_or_skip`, not `_mongo_or_skip`. The difference showed itself
    # on a shared cluster: the drawing was re-ingested underneath this suite,
    # `ReplaceOne` took the h3 fields with it, and these tests failed with
    # four different assertions about colours and counts instead of saying the
    # one true thing — that there is no index to export. A missing index must
    # read as a missing index wherever it is met.
    _indexed_or_skip()
    return _export(res=11, bbox=SMALL_BOX)


# ---------------------------------------------------------------------------
# It is a file, and a file has to explain itself
# ---------------------------------------------------------------------------


def test_every_feature_carries_a_colour_so_it_renders_with_no_style_file(small):
    """simplestyle: geojson.io, GitHub and most Leaflet setups read `fill`,
    `stroke` and `fill-opacity` off a feature's properties. A file that needs
    a style sheet shipped beside it is a file that renders grey."""
    assert small["features"]
    for feature in small["features"]:
        properties = feature["properties"]
        assert properties["fill"].startswith("#"), properties
        assert properties["stroke"].startswith("#")
        assert 0.0 <= properties["fill-opacity"] <= 1.0


def test_every_feature_carries_the_coordinate_system_caveat(small):
    """The one thing that could make this file misleading, attached to every
    feature so it cannot be separated from the geometry by any tool that
    filters, splits or re-exports it."""
    for feature in small["features"]:
        assert "INFERRED" in feature["properties"]["caveat"]
        assert feature["properties"]["epsg"] == 32638
        assert feature["properties"]["drawing_id"] == JANADRIYAH
    assert "INFERRED" in small["properties"]["caveat"]
    assert "not_a_survey" in small["properties"]


def test_the_collection_says_what_is_in_it_and_what_is_not(small):
    """`totals_whole_drawing` beside a bbox-filtered feature list is the one
    number a reader could take for the file's contents, so the note says which
    is which."""
    properties = small["properties"]
    assert properties["parcels_in_file"] + properties["cells_in_file"] == len(
        small["features"]
    )
    assert properties["totals_whole_drawing"]["parcels"] > properties[
        "parcels_in_file"
    ]
    assert "WHOLE drawing" in properties["note"]


def test_parcels_and_cells_can_be_told_apart(small):
    """`kind`, and it is not decoration: a parcel names the cell it is counted
    in, so both kinds carry a `cell` property. Counting on that key was this
    function's first bug — 3,190 features reported as 3,190 cells and 2,545
    parcels at the same time."""
    kinds = {f["properties"]["kind"] for f in small["features"]}
    assert kinds == {"parcel", "cell"}
    for feature in small["features"]:
        if feature["properties"]["kind"] == "parcel":
            assert feature["properties"]["handle"]
        else:
            assert feature["properties"]["cell"]
            assert feature["properties"]["parcels"] >= 0


def test_the_cells_are_an_overlay_and_do_not_hide_the_parcels(small):
    """A filled hexagon covers the thing it is counting."""
    for feature in small["features"]:
        if feature["properties"]["kind"] == "cell":
            assert feature["properties"]["fill-opacity"] == 0.0


def test_colour_follows_land_use_not_layer_name(small):
    """G1: a layer name is one drawing's trait; a land use is this repo's
    vocabulary and applies to any drawing. Two parcels of one use are one
    colour however they were drawn."""
    by_use: dict[str, set[str]] = {}
    for feature in small["features"]:
        use = feature["properties"].get("land_use")
        if use:
            by_use.setdefault(use, set()).add(feature["properties"]["fill"])
    assert by_use
    for use, colours in by_use.items():
        assert len(colours) == 1, (use, colours)
        assert colours == {geo_view.LAND_USE_COLOURS[use]}


def test_a_use_nobody_chose_a_colour_for_looks_unchosen():
    """A vocabulary that grows must not silently start colouring two things
    the same."""
    from app import landuse

    uncoloured = sorted(set(landuse.USES) - set(geo_view.LAND_USE_COLOURS))
    # The reserved ones are never parcels, so they never reach the export.
    assert set(uncoloured) <= set(landuse.RESERVED_USES), uncoloured
    assert geo_view.UNKNOWN_COLOUR not in geo_view.LAND_USE_COLOURS.values()


# ---------------------------------------------------------------------------
# It is GeoJSON, the way GeoJSON wants it
# ---------------------------------------------------------------------------


def test_it_is_a_feature_collection_that_serialises(small):
    assert small["type"] == "FeatureCollection"
    assert json.loads(json.dumps(small))["type"] == "FeatureCollection"


def test_the_rings_are_closed_and_lands_on_janadriyah(small):
    for feature in small["features"]:
        for ring in feature["geometry"]["coordinates"]:
            assert ring[0] == ring[-1]
            for lon, lat in ring:
                assert 46.84 < lon < 46.94, (lon, lat)
                assert 24.80 < lat < 24.90, (lon, lat)


# ---------------------------------------------------------------------------
# The same request twice
# ---------------------------------------------------------------------------


def test_the_same_export_is_byte_identical(small):
    """DoD, and what makes the ETag from the previous wave safe on this route:
    nothing here reads a clock or a random number, and the features come out
    of the same ordered aggregation `/geo` uses."""
    again = _export(res=11, bbox=SMALL_BOX)
    assert json.dumps(again, sort_keys=True) == json.dumps(small, sort_keys=True)


def test_the_export_has_its_own_cache_key():
    """`/geo` and `/geo/export.geojson` answer the same question with
    different bytes. One ETag for both would serve one as the other."""
    facts = {
        "config_version": 1,
        "stored_resolution": 13,
        "ingest_version": 7,
    }
    plain = geo_view.etag_for(
        JANADRIYAH, params={"res": 9, "layout": MODEL}, **facts
    )
    exported = geo_view.etag_for(
        JANADRIYAH, params={"export": True, "res": 9, "layout": MODEL}, **facts
    )
    assert plain != exported


# ---------------------------------------------------------------------------
# Refusals keep their shape
# ---------------------------------------------------------------------------


def test_a_drawing_with_no_coordinate_system_returns_an_empty_collection_and_says_why():
    """Still a FeatureCollection, so a map does not crash on it — and
    `georeferenced: false` beside `features: []`, so an empty file is never
    read as an empty site."""
    _mongo_or_skip()
    # Drawings with NO per-drawing config, derived from the config tree: a
    # drawing that ships one is georeferenced on purpose and is not evidence
    # of a leak. See tests/conftest.py.
    others = unconfigured_drawing_ids()
    if not others:
        pytest.skip("every ingested drawing has a config")
    body = _export(others[0])
    assert body["type"] == "FeatureCollection"
    assert body["features"] == []
    assert body["georeferenced"] is False
    assert body["reason"]
    assert body["how_to_fix"]


@pytest.mark.parametrize("res", [-1, 16, 99])
def test_a_resolution_that_is_not_one_is_a_400(res):
    _mongo_or_skip()
    with pytest.raises(ApiError) as caught:
        _export(res=res)
    assert caught.value.status == 400


def test_a_reversed_box_is_refused_here_too():
    """The validation is the route's, not the viewer's: a file built from a
    box that describes no ground would be an empty file."""
    _mongo_or_skip()
    with pytest.raises(ApiError) as caught:
        _export(bbox="46.90,24.85,46.89,24.86")
    assert caught.value.code == "BBOX_REVERSED"


def test_the_parts_can_be_asked_for_separately():
    """A map at neighbourhood zoom wants the cells; one at parcel zoom wants
    the parcels. Fetching both to use one is the megabyte nobody needed."""
    _indexed_or_skip()
    cells_only = _export(res=11, bbox=SMALL_BOX, parcels=False)
    parcels_only = _export(res=11, bbox=SMALL_BOX, cells=False)
    assert {f["properties"]["kind"] for f in cells_only["features"]} == {"cell"}
    assert {f["properties"]["kind"] for f in parcels_only["features"]} == {"parcel"}
    assert cells_only["properties"]["parcels_in_file"] == 0
    assert parcels_only["properties"]["cells_in_file"] == 0


# ---------------------------------------------------------------------------
# Georeferenced, and not indexed — two different absences
# ---------------------------------------------------------------------------


def test_a_layout_that_can_never_have_cells_says_that_rather_than_nothing():
    """Paper space is georeferenced by association and unmappable by nature:
    its coordinates are page geometry. Telling somebody to run the backfill
    would send them to a command that changes nothing, so it does not."""
    _mongo_or_skip()
    body = _export(layout="DMP Layout1")
    assert body["type"] == "FeatureCollection"
    assert body["features"] == []
    assert body["georeferenced"] is True
    assert body["indexed"] is False
    assert "paper space" in body["reason"]
    assert "layout=Model" in body["how_to_fix"]
    assert "backfill" not in body["how_to_fix"]


def test_a_missing_index_is_never_served_as_an_empty_site():
    """The incident this guard exists for: an external re-ingest stripped all
    46,754 entities of their h3 fields, and both endpoints went on answering
    200 with an empty cell list and totals of zero — which reads as a drawing
    with nothing in it."""
    from app import geo_view

    exc = geo_view.GeoNotIndexed("no cells", "run the backfill")
    assert exc.reason and exc.how_to_fix
    # The two absences are separate classes, so a caller can act on them
    # differently rather than reading one sentence and guessing.
    assert not issubclass(geo_view.GeoNotIndexed, geo_view.GeoUnavailable)
    assert not issubclass(geo_view.GeoUnavailable, geo_view.GeoNotIndexed)


def test_the_indexed_flag_is_always_present_to_be_read(small):
    """Present on the good answer too. A key that only appears when something
    is wrong is a key nobody writes a branch for."""
    assert small.get("indexed") is True


# ---------------------------------------------------------------------------
# What an answer asked for, and what came back — three answers, not two
# ---------------------------------------------------------------------------


#: Two education parcels, two TEXT objects, and two tokens that are not
#: objects at all. The last pair are real digits lifted from a vicinity
#: answer's table: an area of 2761 m2 and one of 32912 m2, both hex-shaped by
#: accident.
MIXED = "205D153,205D154,95BD9F,95BDA0,2761,32912"


def test_an_answers_tokens_are_sorted_into_three_kinds_not_two():
    """The bug this closes: the file said "71 asked for, 9 drawn" and called
    the other 62 objects it could not draw. They were table cells. Describing
    a count of 32,912 m2 as a missing parcel invents sixty-two problems, and
    `answerHighlight.notEntities` already names that mistake for the viewer —
    a file that gets forwarded must not make it either."""
    _mongo_or_skip()
    body = _export(res=11, handles=MIXED)

    drawn = {
        f["properties"]["handle"]
        for f in body["features"]
        if f["properties"]["kind"] == "parcel"
    }
    assert drawn == {"205D153", "205D154"}

    # Real objects this file does not draw: named, because somebody could go
    # and look at them.
    assert body["properties"]["answer_handles_undrawable"] == ["95BD9F", "95BDA0"]
    # Tokens that were never objects: counted, and deliberately NOT listed.
    # A list of them reads as a list of things that went missing.
    assert body["properties"]["answer_handles_not_objects"] == 2


def test_the_note_counts_objects_separately_from_tokens():
    _mongo_or_skip()
    note = _export(res=11, handles=MIXED)["properties"]["answer_handles_note"]
    assert "6 token(s) were asked for" in note
    assert "4 are objects in this drawing" in note
    assert "2 of those had a boundary" in note
    assert "not objects in this drawing at all" in note


def test_a_file_that_is_the_whole_drawing_says_nothing_about_answers():
    """These keys describe a filter. Present on an unfiltered export they
    would be three empty fields inviting a branch that never fires."""
    _mongo_or_skip()
    properties = _export(res=9, parcels=False)["properties"]
    assert properties["answer_handles"] is None
    assert properties["answer_handles_note"] is None
    assert properties["answer_handles_undrawable"] is None
    assert properties["answer_handles_not_objects"] is None


def test_the_parcel_count_and_the_note_cannot_disagree():
    """Both read one variable. They were two separate sums, which is two
    chances to answer "how many parcels are in this file" differently."""
    _mongo_or_skip()
    body = _export(res=11, handles=MIXED)
    drawn = sum(1 for f in body["features"] if f["properties"]["kind"] == "parcel")
    assert body["properties"]["parcels_in_file"] == drawn
    assert f"{drawn} of those had a boundary" in body["properties"]["answer_handles_note"]
