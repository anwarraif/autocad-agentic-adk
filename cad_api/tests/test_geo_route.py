"""GEO-H3 G3 — the map payload.

The endpoint is called through `main.drawing_geo`, with every argument passed
explicitly. Calling it with defaults omitted would hand the function FastAPI's
`Query` objects rather than values, which fails in a way that looks like a bug
in the code under test; the arguments are spelled out so that a failure here
is a failure of the endpoint.

The number this phase turns on is that `?res=9` and `?res=13` agree. They do
because a parcel is counted in ONE cell — its representative one — and coarser
cells are reached with `cell_to_parent`. Both halves are asserted, because
both have an obvious-looking alternative that quietly breaks the other.
"""

from __future__ import annotations

import json

import pytest

h3 = pytest.importorskip("h3")

from app import geo_view, landuse, main, store, store_landuse
from conftest import unconfigured_drawing_ids
from app.main import ApiError

JANADRIYAH = "596212db022a3397"
MODEL = "Model"
MOSQUE = "205D0EB"
#: Pinned in `test_geo_cells.py` as the cell the mosque is stored in.
MOSQUE_CELL_RES_13 = "8d53736546c3a7f"

#: Verified ground truth from `docs/GEO-H3-PLAN.md`, measured live from
#: `/land-use?layout=Model`. The map must report the same drawing the summary
#: reports: two answers to one question is the defect, whichever is right.
REF_COUNTS = {
    "residential": 2380,
    "open_space": 114,
    "utility": 60,
    "education": 9,
    "religious": 6,
    "commercial": 4,
    "community": 3,
}


def _call(drawing_id: str, **kwargs):
    args = dict(
        res=None, layout=MODEL, parcels=True, compact=False,
        limit=geo_view.MAX_PARCEL_FEATURES, bbox=None, outline=False,
        highlight=None,
    )
    args.update(kwargs)
    return main.drawing_geo(drawing_id, **args)


def _mongo_or_skip():
    try:
        drawing = store.get_drawing(JANADRIYAH)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip("the reference drawing has not been ingested")
    return drawing


def _indexed_or_skip():
    _mongo_or_skip()
    body = _call(JANADRIYAH, res=9, parcels=False)
    if not body.get("georeferenced") or not body.get("cells"):
        pytest.skip("no cells yet; run scripts/backfill_h3.py")
    return body


@pytest.fixture(scope="module")
def res9():
    # Indexed, not merely present. A drawing whose cells have been wiped —
    # a re-ingest does that, `ReplaceOne` and all — must skip here rather
    # than fail three tests about counts that were never the subject.
    _indexed_or_skip()
    return _call(JANADRIYAH, res=9, limit=5000)


@pytest.fixture(scope="module")
def res13():
    _indexed_or_skip()
    return _call(JANADRIYAH, res=13, limit=5000)


# ---------------------------------------------------------------------------
# The acceptance number
# ---------------------------------------------------------------------------


def test_two_resolutions_agree_on_every_total(res9, res13):
    """DoD. Exactly, not approximately: a parcel is counted once, in the cell
    that represents it, so changing the resolution regroups the counts without
    changing them."""
    assert res9["totals"]["parcels"] == res13["totals"]["parcels"]
    assert res9["totals"]["counts"] == res13["totals"]["counts"]
    assert res9["totals"]["area_m2"] == res13["totals"]["area_m2"]


def test_the_coarser_view_is_the_finer_one_regrouped(res9, res13):
    """The stronger form of the same claim, and the one that would catch a
    coarse view built by re-indexing the points instead of by
    `cell_to_parent` — that alternative agrees on the grand total while
    putting individual parcels in different cells."""
    from collections import Counter

    rolled: Counter[str] = Counter()
    for cell in res13["cells"]:
        parent = h3.cell_to_parent(cell["cell"], 9)
        for use, count in cell["counts"].items():
            rolled[(parent, use)] += count

    direct: Counter[str] = Counter()
    for cell in res9["cells"]:
        for use, count in cell["counts"].items():
            direct[(cell["cell"], use)] += count

    assert rolled == direct


def test_the_map_counts_the_same_drawing_the_summary_counts(res9):
    """The counts are the plan's verified ground truth. They were 19
    commercial parcels on the first run — 4 polygons and 15 HATCHes drawn over
    them — until a parcel was defined as a polygon, which is the rule
    `land_use_summary` already used."""
    assert res9["totals"]["counts"] == REF_COUNTS


def test_every_cell_is_in_the_region_of_the_golden_set(res9, res13):
    for body in (res9, res13):
        for cell in body["cells"]:
            assert h3.get_base_cell_number(cell["cell"]) == 41, cell["cell"]


def test_the_cells_are_at_the_resolution_that_was_asked_for(res9, res13):
    for body, res in ((res9, 9), (res13, 13)):
        assert body["resolution"] == res
        for cell in body["cells"]:
            assert h3.get_resolution(cell["cell"]) == res


# ---------------------------------------------------------------------------
# The parcels
# ---------------------------------------------------------------------------


def test_every_feature_carries_the_handle_the_brief_guarantees(res9):
    """The one field the brief promises will survive into the final contract:
    it is the same id the agent quotes in chat, so marking an answer on the
    map is answer handles -> their features."""
    features = res9["parcels"]["features"]
    assert features
    for feature in features[:200]:
        assert feature["properties"]["handle"]
        assert feature["properties"]["land_use"] in REF_COUNTS
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Polygon"


def test_the_features_land_on_janadriyah(res9):
    """[lon, lat], and the same box G1's GeoJSON was checked against. A
    swapped pair is still a valid FeatureCollection."""
    for feature in res9["parcels"]["features"][:300]:
        for lon, lat in feature["geometry"]["coordinates"][0]:
            assert 46.84 < lon < 46.94, (lon, lat)
            assert 24.80 < lat < 24.90, (lon, lat)


def test_the_collection_serialises_as_json(res9):
    payload = json.dumps(res9["parcels"])
    assert json.loads(payload)["type"] == "FeatureCollection"


def test_parcels_counted_but_not_drawn_are_counted_out_loud(res9):
    """A parcel with a bulged ring is a real parcel and is counted; it has no
    boundary this store will vouch for, so it is not drawn. The difference
    between the two numbers is stated rather than left to be discovered by
    subtracting them."""
    drawn = len(res9["parcels"]["features"])
    counted = res9["totals"]["parcels"]
    assert counted >= drawn
    assert res9["excluded"]["parcels_without_a_usable_ring"] == counted - drawn
    assert res9["excluded"]["note"]


# ---------------------------------------------------------------------------
# Budgets (G7)
# ---------------------------------------------------------------------------


def test_the_cells_alone_are_small_enough_for_a_map_to_fetch():
    """DoD. The payload a map actually needs at neighbourhood zoom: 23 cells,
    a few kilobytes. The megabyte is the parcels, and they are opt-out."""
    _indexed_or_skip()
    body = _call(JANADRIYAH, res=9, parcels=False)
    assert len(json.dumps(body, default=str)) < 50_000
    assert body["parcels"]["features"] == []


def test_the_response_states_its_own_limits(res9):
    limits = res9["limits"]
    assert limits["max_parcel_features"] == geo_view.MAX_PARCEL_FEATURES
    assert limits["max_cells"] == geo_view.MAX_CELLS
    assert limits["parcels_omitted"] == 0
    assert limits["cells_omitted"] == 0


def test_a_truncated_collection_says_how_many_it_left_out():
    """G7. A collection that silently stops at the limit reads exactly like a
    drawing with that many parcels."""
    _indexed_or_skip()
    body = _call(JANADRIYAH, res=9, limit=10)
    assert len(body["parcels"]["features"]) == 10
    assert body["limits"]["parcels_omitted"] > 0
    assert body["totals"]["parcels"] > 10


def test_compaction_applies_to_the_covering_and_never_to_the_counts():
    """The one place the plan's wording and the arithmetic disagree, resolved
    in favour of the arithmetic: `compact_cells` merges children into a
    parent, and a count cannot survive that merge — seven parcels in seven
    children are not seven parcels in the parent. So the counted cells are
    left alone and the SET the parcels occupy is what gets compacted."""
    _indexed_or_skip()
    body = _call(JANADRIYAH, res=9, parcels=False, compact=True)
    packed = body["coverage_compact"]
    assert packed
    assert len(packed) < 30_000
    assert body["coverage_note"]
    # The counted cells are untouched by the option.
    plain = _call(JANADRIYAH, res=9, parcels=False, compact=False)
    assert body["cells"] == plain["cells"]


# ---------------------------------------------------------------------------
# Refusals (BINDING 3, and clear 400s)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("res", [-1, 16, 99])
def test_a_resolution_that_is_not_one_is_a_400_with_a_way_forward(res):
    _mongo_or_skip()
    with pytest.raises(ApiError) as caught:
        _call(JANADRIYAH, res=res)
    assert caught.value.status == 400
    assert caught.value.code == "H3_RESOLUTION_OUT_OF_RANGE"
    assert caught.value.hint


def test_asking_for_more_detail_than_is_stored_is_refused_not_invented():
    """`cell_to_children` would return seven cells where the drawing supports
    one piece of evidence, and seven cells read as more detail rather than as
    the same fact restated."""
    _indexed_or_skip()
    stored = _call(JANADRIYAH, res=None, parcels=False)["stored_resolution"]
    with pytest.raises(ApiError) as caught:
        _call(JANADRIYAH, res=stored + 1)
    assert caught.value.status == 400
    assert caught.value.code == "H3_RESOLUTION_FINER_THAN_STORED"
    assert str(stored) in caught.value.hint


def test_an_unknown_drawing_is_still_a_404():
    _mongo_or_skip()
    with pytest.raises(ApiError) as caught:
        _call("0000000000000000")
    assert caught.value.status == 404


def _all_drawing_ids() -> list[str]:
    """Collected at import time, so it must not raise when there is no
    database — a parametrize that throws takes the whole module down and the
    other tests here stop running for a reason nobody would guess."""
    try:
        return [d["_id"] for d in store.list_drawings()]
    except Exception:  # pragma: no cover - depends on the environment
        return []


ALL_DRAWINGS = _all_drawing_ids()

#: The set these "must refuse" invariants are really about. A drawing that
#: ships a per-drawing config is georeferenced deliberately and is not
#: evidence that knowledge leaked between drawings. See tests/conftest.py.
UNCONFIGURED = unconfigured_drawing_ids()


@pytest.mark.parametrize("drawing_id", UNCONFIGURED or ["<no database>"])
def test_a_drawing_with_no_crs_says_so_instead_of_answering_empty(drawing_id):
    """BINDING 3, over every ingested drawing.

    The failure this forbids is the quiet one: `cells: []` with a 200 and
    nothing else, which reads as "this drawing has no parcels" rather than
    "this drawing has no coordinate system". The shape is the same either way
    so a client needs no second branch; `georeferenced` is what tells them
    apart, and `reason` says what would have to change."""
    if drawing_id == "<no database>":
        pytest.skip("no database")
    body = _call(drawing_id, parcels=False)
    if drawing_id == JANADRIYAH:
        assert body["georeferenced"] is True
        assert body.get("reason") is None
        return
    assert body["georeferenced"] is False
    assert body["reason"]
    assert body["how_to_fix"]
    assert body["cells"] == []
    assert body["parcels"]["features"] == []


# ---------------------------------------------------------------------------
# Units (G2 of the generality rules)
# ---------------------------------------------------------------------------


def test_areas_are_published_only_where_the_unit_is_the_one_promised(res9):
    """`area_m2` is a promise about the unit, and eleven of the eighteen
    drawings here are in inches. Nothing converts: the drawing says what its
    numbers are, and where that is not m2 the areas are withheld with a
    reason."""
    assert res9["area_unit"] == geo_view.AREA_UNIT
    assert res9["area_note"] is None
    assert res9["totals"]["area_m2"]
    for cell in res9["cells"]:
        for use, area in cell["area_m2"].items():
            assert area > 0, (use, area)


def test_the_area_total_is_the_sum_of_the_cells(res9):
    from collections import defaultdict

    summed = defaultdict(float)
    for cell in res9["cells"]:
        for use, area in cell["area_m2"].items():
            summed[use] += area
    for use, total in res9["totals"]["area_m2"].items():
        assert summed[use] == pytest.approx(total, rel=1e-6)


# ---------------------------------------------------------------------------
# The entity block
# ---------------------------------------------------------------------------


def test_an_entity_carries_its_cells_and_their_resolution():
    """The plan's other half of G3: answer handles -> their cells."""
    _indexed_or_skip()
    entity = main.get_entity_detail(JANADRIYAH, MOSQUE)
    block = entity["h3"]
    assert block["cell"] == "8d53736546c3a7f"
    assert block["resolution"] == 13
    assert block["strategy"] == "ring"
    assert block["cells"] and block["cell"] in block["cells"]
    assert block["cell_count"] == len(block["cells"])


def test_an_entity_with_too_many_cells_withholds_the_list_and_says_so():
    """G7 again, and the same 392 KB lesson: the covering of the site boundary
    is 20,000 cells. Withheld with its count and a pointer, never truncated
    into a shorter list that looks complete."""
    _indexed_or_skip()
    entity = main.get_entity_detail(JANADRIYAH, "1C0436A")
    block = entity["h3"]
    assert block["cells"] is None
    assert block["cell_count"] > store_landuse.MAX_ENTITY_CELLS
    assert block["cells_withheld"]
    assert "/geo" in block["cells_withheld"]


def test_an_entity_with_no_cells_has_no_block_rather_than_an_empty_one():
    """The flat `h3_note` already says why. A block repeating it would be a
    second copy free to disagree."""
    _indexed_or_skip()
    from app.mongo import COLL_ENTITIES, coll

    row = coll(COLL_ENTITIES).find_one(
        {"drawing_id": JANADRIYAH, "layout": "DMP Layout1", "h3_cell": None},
        {"handle": 1},
    )
    if not row:
        pytest.skip("no uncelled entity found")
    entity = main.get_entity_detail(JANADRIYAH, row["handle"])
    assert entity["h3"] is None
    assert entity["h3_note"]


# ---------------------------------------------------------------------------
# P3 — the viewport
# ---------------------------------------------------------------------------

#: A box around the Jumaa Mosque, whose centroid is 24.85142 / 46.89358.
#: Written as the string a client sends, not as a parsed object: the parsing
#: is part of what these tests exercise.
MOSQUE_BOX = "46.891,24.849,46.896,24.854"

#: Desert well south-west of the site. Real coordinates, nothing drawn there.
EMPTY_BOX = "46.700,24.700,46.710,24.710"


def test_a_viewport_keeps_what_is_in_it_and_drops_what_is_not():
    """DoD. The mosque is in the box and the VL4 plot at the far end of the
    site is not, and both facts come out of the same call."""
    _indexed_or_skip()
    body = _call(JANADRIYAH, res=11, bbox=MOSQUE_BOX)
    handles = {f["properties"]["handle"] for f in body["parcels"]["features"]}
    assert MOSQUE in handles
    assert "205C7CC" not in handles
    view = body["viewport"]
    assert view["cells_in_view"] > 0
    assert view["cells_outside"] > view["cells_in_view"]


def test_the_totals_are_never_narrowed_by_a_viewport():
    """The failure this guards against is a picture of four hexagons with
    '2,380 houses' printed beside it, and nothing saying the number came from
    somewhere else on the screen."""
    _indexed_or_skip()
    whole = _call(JANADRIYAH, res=11, parcels=False)
    framed = _call(JANADRIYAH, res=11, parcels=False, bbox=MOSQUE_BOX)
    assert framed["totals"] == whole["totals"]
    assert framed["totals"]["counts"] == REF_COUNTS
    # And what IS narrowed is reported separately, and is smaller.
    in_view = framed["viewport"]["counts_in_view"]
    assert sum(in_view.values()) < sum(framed["totals"]["counts"].values())
    assert "totals" in framed["viewport"]["note"]


def test_an_empty_viewport_is_an_answer_and_not_an_error():
    """DoD. Empty desert 20 km away: 200, no cells, and a sentence saying that
    an empty box means empty ground rather than a failed query."""
    _indexed_or_skip()
    body = _call(JANADRIYAH, res=11, bbox=EMPTY_BOX)
    assert body["cells"] == []
    assert body["parcels"]["features"] == []
    assert body["viewport"]["counts_in_view"] == {}
    assert body["viewport"]["empty_is_an_answer"]
    assert body["totals"]["parcels"] > 0


def test_no_viewport_means_no_viewport_block():
    """An unfiltered answer must not carry an empty frame that a reader could
    take for one."""
    _indexed_or_skip()
    assert _call(JANADRIYAH, res=11, parcels=False)["viewport"] is None


@pytest.mark.parametrize(
    "raw,code",
    [
        ("46.89,24.85,46.90", "BBOX_MALFORMED"),
        ("46.89,24.85,46.90,24.86,1", "BBOX_MALFORMED"),
        ("a,b,c,d", "BBOX_MALFORMED"),
        ("", None),
        ("46.90,24.85,46.89,24.86", "BBOX_REVERSED"),
        ("46.89,24.86,46.90,24.85", "BBOX_REVERSED"),
        ("200,24.85,46.90,24.86", "BBOX_OUT_OF_RANGE"),
        ("46.89,99,46.90,100", "BBOX_OUT_OF_RANGE"),
    ],
)
def test_a_box_that_describes_no_ground_is_refused_rather_than_repaired(raw, code):
    """A reversed box is the one worth refusing loudest. Swapping the corners
    would answer a different question, and its answer — an empty viewport —
    reads exactly like an empty part of the site."""
    if code is None:
        assert geo_view.parse_bbox(raw) is None
        return
    with pytest.raises(geo_view.GeoBadRequest) as caught:
        geo_view.parse_bbox(raw)
    assert caught.value.code == code
    assert caught.value.hint


def test_the_box_is_read_longitude_first():
    """The same order GeoJSON uses, and the same order this API's own
    coordinates use. The pair that gets swapped is the pair whose two
    conventions live in one codebase."""
    box = geo_view.parse_bbox("46.89,24.85,46.90,24.86")
    assert (box.west, box.south, box.east, box.north) == (46.89, 24.85, 46.90, 24.86)
    assert box.contains(46.895, 24.855)
    assert not box.contains(24.855, 46.895)


def test_a_cell_larger_than_the_box_still_counts_as_in_view():
    """The case the obvious two tests miss. A resolution-9 cell is 105,000 m2;
    a viewport inside one has neither the cell's centre nor any of its
    vertices, and the cell would vanish from a view it entirely covers."""
    _indexed_or_skip()
    tiny = "46.8935,24.8513,46.8937,24.8515"
    body = _call(JANADRIYAH, res=9, parcels=False, bbox=tiny)
    assert body["viewport"]["cells_in_view"] >= 1


# ---------------------------------------------------------------------------
# P5 — the site's own outline
# ---------------------------------------------------------------------------


def test_the_coverage_outline_is_one_shape_rather_than_25627_hexagons():
    """A map that wants 'where is this development' draws this instead of the
    whole covering. Measured: 25,627 cells become 99 polygons and 18,731
    vertices — still a lot, and four orders of magnitude fewer edges than the
    hexagons would be."""
    _indexed_or_skip()
    body = _call(JANADRIYAH, res=9, parcels=False, outline=True)
    outline = body["coverage_outline"]
    assert outline["geometry"]["type"] == "MultiPolygon"
    assert outline["properties"]["cells"] > 20_000
    assert body["coverage_outline_note"]
    assert "not the site's surveyed boundary" in body["coverage_outline_note"]


def test_the_coverage_outline_covers_the_site_and_only_the_site():
    _indexed_or_skip()
    body = _call(JANADRIYAH, res=9, parcels=False, outline=True)
    for poly in body["coverage_outline"]["geometry"]["coordinates"]:
        for ring in poly:
            assert ring[0] == ring[-1]
            for lon, lat in ring:
                assert 46.84 < lon < 46.94, (lon, lat)
                assert 24.80 < lat < 24.90, (lon, lat)


def test_no_outline_unless_it_was_asked_for():
    _indexed_or_skip()
    assert "coverage_outline" not in _call(JANADRIYAH, res=9, parcels=False)


# ---------------------------------------------------------------------------
# Wave 4 — the hierarchy the map climbs, and the answer it lights
# ---------------------------------------------------------------------------

#: The ladder, DERIVED from this drawing's stored cells rather than chosen.
#: Read live from the store on 1 September 2026 by rolling the res-13 cells up
#: with `cell_to_parent`, which is exactly what the endpoint now does — so
#: this pins a measurement, not a preference. It is here because the map's
#: zoom bands are tuned against these numbers: if a re-ingest moves them, the
#: bands are showing something other than what they were tuned for, and that
#: is worth failing a test over.
LADDER = {6: 1, 7: 3, 8: 7, 9: 23, 10: 115, 11: 645, 12: 1957, 13: 2546}


def test_the_hierarchy_is_the_measured_ladder(res9):
    """Harsh, 1 Sep: the top is capped to the hexagon covering the project."""
    hierarchy = res9["hierarchy"]
    assert hierarchy["res_stored"] == 13
    assert hierarchy["res_cover"] == 6
    assert {level["res"]: level["cells"] for level in hierarchy["levels"]} == LADDER


def test_the_cap_really_is_one_hexagon_and_the_finest_that_is(res9):
    """`res_cover` is a claim with two halves, and both are checkable: one
    hexagon at that resolution, more than one at the next finer."""
    levels = {level["res"]: level["cells"] for level in res9["hierarchy"]["levels"]}
    cover = res9["hierarchy"]["res_cover"]
    assert levels[cover] == 1
    assert levels[cover + 1] > 1


def test_the_hierarchy_does_not_change_with_the_resolution_asked_for(res9, res13):
    """It describes the DRAWING, not the request. Derived from the stored
    cells rather than from the rolled-up ones, precisely so that asking at
    res 9 cannot make the ladder start at res 9."""
    assert res9["hierarchy"] == res13["hierarchy"]


def test_the_ladder_is_monotonic(res9):
    """A finer resolution can never describe the same site in fewer cells.
    Cheap, and it is the shape of the bug a wrong roll-up direction makes."""
    counts = [level["cells"] for level in res9["hierarchy"]["levels"]]
    assert counts == sorted(counts)
    assert counts[-1] == len(_call(JANADRIYAH, res=13, parcels=False)["cells"])


def test_highlighting_an_answer_lights_its_cells_and_keeps_the_rest(res9):
    """Kurnia's ask, at the API. The grid is NOT filtered: a highlight that
    returned only the lit cells would leave the reader nothing to read them
    against, which is the whole point of lighting them."""
    mosque = _call(JANADRIYAH, res=9, parcels=False, highlight=MOSQUE)
    plain = _call(JANADRIYAH, res=9, parcels=False)

    assert len(mosque["cells"]) == len(plain["cells"])
    lit = [c for c in mosque["cells"] if c["highlighted"]]
    assert len(lit) == 1
    assert mosque["highlight"]["cells"] == [lit[0]["cell"]]
    assert mosque["highlight"]["handles_not_objects"] is None


def test_the_lit_cell_is_the_one_the_golden_cell_rolls_up_into(res9):
    """The flag has to agree with the roll-up the counts use, or a lit hexagon
    is not the hexagon the parcel was counted in."""
    mosque = _call(JANADRIYAH, res=9, parcels=False, highlight=MOSQUE)
    lit = next(c["cell"] for c in mosque["cells"] if c["highlighted"])
    assert lit == h3.cell_to_parent(MOSQUE_CELL_RES_13, 9)


def test_every_cell_answers_the_highlight_question_or_none_of_them_does(res9):
    """Present because the caller asked, absent because they did not — never
    present for some cells and missing for others."""
    lit = _call(JANADRIYAH, res=9, parcels=False, highlight=MOSQUE)
    assert all("highlighted" in cell for cell in lit["cells"])
    assert all("highlighted" not in cell for cell in res9["cells"])
    assert res9["highlight"] is None


def test_a_token_that_is_not_an_object_is_counted_not_reported_missing():
    """An answer's prose carries counts and areas that are hex-shaped by
    accident. Calling those missing parcels invents a problem out of a table
    cell — the same rule the export already follows."""
    body = _call(JANADRIYAH, res=9, parcels=False, highlight=f"2761,{MOSQUE}")
    assert body["highlight"]["handles_not_objects"] == 1
    assert body["highlight"]["cells_lit"] == 1
    assert "not objects in this drawing" in body["highlight"]["note"]


def test_the_hierarchy_is_absent_for_a_drawing_that_has_no_cells():
    """Absent rather than empty. A ladder of zero levels would read as a site
    with no hierarchy, which is a different claim from "not georeferenced"."""
    _mongo_or_skip()
    body = _call("cce5178aeaf3736c", res=9, parcels=False)
    assert body.get("georeferenced") is False
    assert "hierarchy" not in body
