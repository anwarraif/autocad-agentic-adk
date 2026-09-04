"""GEO-H3 G1 — lat/long and GeoJSON on the entity response.

Two claims are under test and they are not the same claim.

The first is arithmetic: a ring reprojected vertex by vertex lands where the
cross-referenced points say it lands, in the order GeoJSON specifies, closed
the way RFC 7946 requires. That is checkable here.

The second is that nothing else moved. `get_entity` is the response the agent
reads most often, and this phase adds three keys to it; a test that only
checked the new keys would pass just as happily over a response that had lost
`units` or `labels_inside`. So the shape of the old response is pinned as a
subset, from the same fixture the new keys are read from.

Ground truth comes from `docs/GEO-H3-PLAN.md` and from `test_landuse.py`'s
`GEOREF`, which cross-checked the same two points against pyproj. Nothing here
re-derives a constant.
"""

from __future__ import annotations

import json

import pytest

from app import landuse, main, store, store_landuse
from conftest import unconfigured_drawing_ids
from app.mongo import COLL_ENTITIES, coll

JANADRIYAH = "596212db022a3397"
MODEL = "Model"

#: The two points UPLIFT-14 cross-checked against pyproj, rounded to the five
#: decimals those references carry. Tolerance is 1e-5 degrees, about 1.1 m —
#: tighter than that would be testing their rounding.
MOSQUE = "205D0EB"
SCHOOL = "205D153"
GEOREF = {
    MOSQUE: (24.85142, 46.89358),
    SCHOOL: (24.85975, 46.89974),
}

#: A 4-vertex parcel with a 300 m2 ring, from the plan's verified ground truth.
PLOT_VL4 = "205C7CC"

#: Janadriyah, generously boxed. Anything outside this is not a rounding error
#: in the projection; it is the wrong zone, or swapped coordinates, and both
#: produce a perfectly plausible-looking pair of numbers.
LAT_RANGE = (24.80, 24.90)
LON_RANGE = (46.84, 46.94)


def _mongo_or_skip(drawing_id: str = JANADRIYAH):
    try:
        drawing = store.get_drawing(drawing_id)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip(f"the reference drawing {drawing_id} has not been ingested")
    return drawing


def _all_drawing_ids() -> list[str]:
    try:
        return [d["_id"] for d in store.list_drawings()]
    except Exception:  # pragma: no cover - depends on the environment
        return []


ALL_DRAWINGS = _all_drawing_ids()

#: The set these "must refuse" invariants are really about. A drawing that
#: ships a per-drawing config is georeferenced deliberately and is not
#: evidence that knowledge leaked between drawings. See tests/conftest.py.
UNCONFIGURED = unconfigured_drawing_ids()


def _indexed_or_skip():
    """The drawing is here AND it has cells.

    Two different states, and the tests below need the second. A re-ingest on
    a shared cluster removes the h3 fields — `store.replace_entities` writes
    with `ReplaceOne` — and without this guard the key-set tests fail on the
    seven G2 keys being absent, which says nothing about the keys."""
    _mongo_or_skip()
    from app.mongo import COLL_ENTITIES, coll

    if not coll(COLL_ENTITIES).find_one(
        {"drawing_id": JANADRIYAH, "h3_cell": {"$ne": None}}, {"_id": 1}
    ):
        pytest.skip("no cells yet; run scripts/backfill_h3.py")


@pytest.fixture(scope="module")
def mosque():
    _indexed_or_skip()
    return main.get_entity_detail(JANADRIYAH, MOSQUE)


# ---------------------------------------------------------------------------
# lat/long on the response
# ---------------------------------------------------------------------------


def test_the_entity_response_carries_the_cross_referenced_lat_long(mosque):
    """DoD 1. Through the route, not through the store function — the store
    function was already tested and already worked; what this phase changed is
    that anybody can reach it."""
    lat, lon = GEOREF[MOSQUE]
    assert mosque["lat_lon"]["lat"] == pytest.approx(lat, abs=1e-5)
    assert mosque["lat_lon"]["lon"] == pytest.approx(lon, abs=1e-5)


def test_the_lat_long_on_the_response_carries_the_inferred_caveat(mosque):
    """The rule that outranks the number itself. A lat/long from a zone nobody
    confirmed looks exactly like one from a zone somebody did."""
    block = mosque["lat_lon"]
    assert "INFERRED" in block["caveat"]
    assert block["crs_declared_in_file"] is False
    assert block["crs_confirmed_by"] is None
    assert block["crs_grade"] == "inferred"


def test_a_georeferenced_entity_has_no_note_to_make(mosque):
    """`geo_note` exists to explain an absence. When there is no absence it is
    null rather than a reassuring sentence — prose that appears on the happy
    path is prose nobody reads on the unhappy one."""
    assert mosque["geo_note"] is None


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------


def test_the_feature_is_the_shape_geojson_readers_expect(mosque):
    feature = mosque["geometry_geojson"]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert len(feature["geometry"]["coordinates"]) == 1
    assert feature["properties"]["handle"] == MOSQUE
    # The whole reason this is a Feature and not a bare geometry.
    assert "INFERRED" in feature["properties"]["caveat"]


def test_the_ring_is_closed_and_wound_the_way_the_rfc_says(mosque):
    """RFC 7946: a linear ring repeats its first position last, and an exterior
    ring is counterclockwise. Neither is decoration — a tool that trusts the
    winding reads an unclosed clockwise ring as a hole."""
    ring = mosque["geometry_geojson"]["geometry"]["coordinates"][0]
    assert len(ring) >= 4
    assert ring[0] == ring[-1]
    twice_area = sum(
        ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
        for i in range(len(ring) - 1)
    )
    assert twice_area > 0, "exterior ring must be counterclockwise"


def test_longitude_comes_first_and_the_shape_lands_on_janadriyah(mosque):
    """The single easiest way to get this wrong is to write [lat, lon]. It
    produces a valid-looking feature 22 degrees away, in Somalia, and no
    assertion about ring length or closure would notice."""
    ring = mosque["geometry_geojson"]["geometry"]["coordinates"][0]
    for lon, lat in ring:
        assert LON_RANGE[0] < lon < LON_RANGE[1], (lon, lat)
        assert LAT_RANGE[0] < lat < LAT_RANGE[1], (lon, lat)


def test_the_feature_sits_on_the_point_the_lat_long_reports(mosque):
    """The two new keys are two views of one object, so they have to agree.
    They are computed from different stored fields — the centroid and the ring
    — which is exactly why this can fail."""
    ring = mosque["geometry_geojson"]["geometry"]["coordinates"][0]
    lons = [lon for lon, _ in ring]
    lats = [lat for _, lat in ring]
    assert min(lons) <= mosque["lat_lon"]["lon"] <= max(lons)
    assert min(lats) <= mosque["lat_lon"]["lat"] <= max(lats)


def test_the_feature_states_no_area_or_length(mosque):
    """Display-grade only. An area taken off a projection nobody confirmed is
    an area with no provenance, and a reader who finds one in `properties`
    has no way to know it is not the drawing's own figure."""
    props = mosque["geometry_geojson"]["properties"]
    for key in props:
        assert "area" not in key, key
        assert "length" not in key, key
    assert "not measured off this shape" in props["note"] or (
        "must be read from the" in props["note"]
    )


def test_several_parcels_form_a_feature_collection_that_serialises():
    """DoD 2's machine-checkable half. The half a machine cannot check —
    whether the shapes land on Janadriyah when pasted into geojson.io — is the
    visual proof this phase carries separately."""
    _mongo_or_skip()
    features = []
    for handle in (MOSQUE, SCHOOL, PLOT_VL4):
        feature = store_landuse.geometry_geojson_for_entity(JANADRIYAH, handle)
        if feature is not None:
            features.append(feature)
    assert len(features) >= 2
    collection = {"type": "FeatureCollection", "features": features}
    # Round-trips as JSON with no custom encoder: a BSON value that leaked
    # through would only surface here, at the point where somebody pastes it.
    assert json.loads(json.dumps(collection))["type"] == "FeatureCollection"


def test_a_ring_this_store_will_not_vouch_for_gets_no_geometry():
    """G8. A bulged polygon's stored vertices are the chords of its arcs, not
    its boundary. It gets `null` and the entity's own `geometry_note` says why
    — not a shape that is wrong by an arc's worth and looks like every other
    shape on the map."""
    _mongo_or_skip()
    row = coll(COLL_ENTITIES).find_one(
        # drawing_id first, then the indexed layout; `ring_status` narrows what
        # the index already selected. The cluster runs notablescan.
        {"drawing_id": JANADRIYAH, "layout": MODEL, "ring_status": "bulge"},
        {"handle": 1},
    )
    if not row:
        pytest.skip("no bulged polygon in the reference drawing")
    handle = row["handle"]
    assert store_landuse.geometry_geojson_for_entity(JANADRIYAH, handle) is None
    entity = main.get_entity_detail(JANADRIYAH, handle)
    assert entity["geometry_geojson"] is None
    assert entity["geometry_note"], "the reason must already be in the response"


# ---------------------------------------------------------------------------
# The other drawings (G5, G3, G10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drawing_id", UNCONFIGURED or ["<no database>"])
def test_a_drawing_without_a_crs_config_still_writes_no_lat_long(drawing_id):
    """DoD 3, over every ingested drawing rather than the convenient one. The
    Autodesk samples use local coordinates and eleven of them are in inches;
    a UTM zone applied to those produces nonsense shaped like a position.

    The absence has to be an ANSWER: three keys present, two null, and a
    sentence saying why (G3). An endpoint that simply omitted them would be
    indistinguishable from one that had not been deployed."""
    if not ALL_DRAWINGS:
        pytest.skip("no database: the cross-drawing invariant cannot be run")
    if drawing_id == JANADRIYAH:
        pytest.skip("the one georeferenced drawing is covered by its own tests")

    rows = store.query_entities(drawing_id, limit=1)["entities"]
    if not rows:
        pytest.skip(f"{drawing_id} has no entities")
    entity = main.get_entity_detail(drawing_id, rows[0]["handle"])
    assert "lat_lon" in entity and entity["lat_lon"] is None, drawing_id
    assert entity["geometry_geojson"] is None, drawing_id
    assert entity["geo_note"], drawing_id
    assert "no CRS config" in entity["geo_note"], drawing_id


def test_the_georeferenced_drawing_is_the_only_one_that_answers():
    """G10, stated as a count. If a second drawing starts returning lat/long,
    either somebody added a config — in which case this number is edited
    deliberately — or knowledge has leaked from one drawing to another."""
    if not ALL_DRAWINGS:
        pytest.skip("no database")
    from conftest import configured_drawing_ids

    with_crs = []
    for drawing_id in ALL_DRAWINGS:
        try:
            config = landuse.for_drawing(drawing_id)
        except landuse.ConfigError:
            # A config that cannot be read is not a config that georeferences.
            continue
        if config is not None and config.crs is not None:
            with_crs.append(drawing_id)
    # Configured drawings answer; nothing else does. Read from the config
    # tree so that adding a config is a deliberate act this test follows,
    # rather than a change that breaks it for an unrelated-looking reason.
    assert set(with_crs) <= configured_drawing_ids(), (
        "a drawing answered with a coordinate system it was never given"
    )
    assert JANADRIYAH in with_crs, "the reference drawing must stay georeferenced"


# ---------------------------------------------------------------------------
# Nothing else moved
# ---------------------------------------------------------------------------

#: Every key `get_entity` answered with for this entity before this phase,
#: read off the live response rather than remembered. Thirty-four of them, and
#: the point of listing all thirty-four is that a test which pinned the
#: interesting five would pass over a response that had quietly lost
#: `labels_inside` — which is the field an earlier bug did lose, for a day.
OLD_KEYS = {
    "_id",
    "area",
    "attribs",
    "bbox",
    "bbox_centre",
    "block_name",
    "centroid_inside_ring",
    "comments",
    "contained_by",
    "drawing_id",
    "ends_basis",
    "ends_status",
    "extrusion_non_standard",
    "geometry_note",
    "handle",
    "labels_inside",
    "layer",
    "layout",
    "length",
    "path_points_basis",
    "perimeter_from_ring",
    "polygon_centroid",
    "ring_dropped_duplicates",
    "ring_frame",
    "ring_orientation",
    "ring_origin",
    "ring_simple",
    "ring_status",
    "ring_vertex_count",
    "shape_key",
    "shape_key_basis",
    "text",
    "type",
    "units",
}

#: What G1 adds, and all it adds.
G1_KEYS = {"lat_lon", "geometry_geojson", "geo_note"}

#: What G2 adds. Recorded here deliberately rather than folded in silently:
#: this list is the whole point of the test below, and G2 tripped it on its
#: first full run — which is what it is for.
#:
#: `h3_cells` is NOT among them. The covering of the site-boundary polygon is
#: 20,000 cells and took one entity response from 15 KB to 392 KB, so it is
#: withheld from `get_entity` exactly as `ring` is. `h3_cell` and
#: `h3_cell_count` say which cell stands for the entity and how many there
#: are; the covering belongs to the geo endpoints.
G2_KEYS = {
    "h3_res",
    "h3_cell",
    "h3_cell_count",
    "h3_strategy",
    "h3_cells_truncated",
    "h3_note",
    "h3_coverage_note",
}

#: What G3 adds: one shaped block carrying the covering and the resolution it
#: is at. It exists because the flat `h3_*` fields cannot carry the cells —
#: `h3_cells` is withheld from this response — so a reader of one object had
#: no way to see which cells it occupies. The block caps the list at
#: `MAX_ENTITY_CELLS` and says when it withheld it.
G3_KEYS = {"h3"}

#: What the block-placement work adds, DECLARED rather than folded in.
#:
#: `h3_world_placed` says whether this entity has a position on the earth at
#: all -- decided from the composed INSERT chain, not from the name of its
#: layout. It is the single fact that decided 679,878 of Sedra's entities were
#: on the map and 310,912 were not, so a reader of one object is entitled to
#: it and it is published on purpose.
#:
#: Its sibling `h3_coarse` is NOT here, and that is the same judgement made
#: the other way: the resolution-8 parent exists so a viewport can match an
#: index, it means nothing about the entity, and it is withheld in
#: `store.get_entity` beside `h3_cells` and `ring`.
PLACEMENT_KEYS = {"h3_world_placed"}

NEW_KEYS = G1_KEYS | G2_KEYS | G3_KEYS | PLACEMENT_KEYS


def test_the_response_kept_every_key_it_had(mosque):
    """Additive means additive. The three new keys are worth nothing if the
    agent's other twenty questions stopped being answerable."""
    missing = OLD_KEYS - set(mosque)
    assert not missing, missing


def test_the_new_keys_are_the_only_new_keys(mosque):
    """The other direction, so that a debug field cannot ride into a shared
    response and become somebody's dependency. Written as an exact difference
    rather than a subset: a later phase that adds a key to this endpoint is
    meant to edit this list and say so, not to slip past it.

    It has already earned that twice. G2 added seven keys to every entity
    document and this test failed on the run that introduced them — which is
    how the 392 KB `h3_cells` payload was noticed at all, rather than after
    somebody asked the agent about the site boundary. G3 tripped it again with
    its `h3` block, which is the intended way to find out that a response has
    grown."""
    assert set(mosque) - OLD_KEYS == NEW_KEYS


def test_the_covering_itself_never_rides_on_an_entity_response():
    """G7, measured rather than assumed. `get_entity` goes straight into an
    agent tool call, and one 20,000-cell array is most of a context window."""
    _indexed_or_skip()
    biggest = main.get_entity_detail(JANADRIYAH, "1C0436A")
    assert "h3_cells" not in biggest
    assert biggest["h3_cell_count"] > 20_000
    assert len(json.dumps(biggest, default=str)) < 100_000
