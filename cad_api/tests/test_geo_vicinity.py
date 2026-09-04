"""GEO-H3 G4 — the vicinity recipe, and whether to believe it.

The acceptance question is Dr. Adel's, relayed through Prasang: how many
houses are within 100 metres of the school. This recipe answers it over cells
that already exist, which makes it fast and makes it approximate, and the
tests that matter here are the ones about the approximation rather than the
ones about the count.

Two of them are worth reading before the code.

`test_the_reach_is_measured_not_asserted` pins what k rings actually cover. The
plan says k=2 is "about 100 m at res 11"; measured, it is 99.3 m centre to
centre and 128.0 m into the disk's corners. Both numbers travel in every
response, because quoting only the first is how an approximation becomes a
wrong figure.

`test_it_agrees_with_the_exact_measurement_in_the_right_mode` is the
cross-check the phase is judged by, and it also records a mistake. The first
version of the recipe pointed at `proximity_count` in `centroid` mode, which
answered 45 where this answers 139. Nothing was wrong with either number: the
rings start from every cell the SUBJECT occupies, so they reach outward from
its footprint, while centroid mode measures centre to centre — and the school
is 74 m across. Against `edge` mode, the comparable basis, the same school is
111 against 139.
"""

from __future__ import annotations

import pytest

h3 = pytest.importorskip("h3")

from app import landuse, recipes, store, store_spatial
from conftest import unconfigured_drawing_ids
from app.mongo import COLL_ENTITIES, coll
from app.recipes import vicinity
from app.recipes import registry
from app.recipes.registry import RecipeRefused

JANADRIYAH = "596212db022a3397"
MODEL = "Model"

#: The Primary School the cross-check is run on, and its own size — which is
#: the whole reason centroid mode and edge mode disagree.
SCHOOL = "205D153"
SCHOOL_ACROSS_M = 74.3

#: docs/GEO-H3-PLAN.md: nine education parcels in model space.
SCHOOLS = 9

#: Measured, at resolution 11 with k=2. The plan says "about 100 m"; these are
#: what "about" turns out to be, and both of them are published.
REACH_CENTRE_TO_CENTRE_M = 99.3
REACH_TO_CORNER_M = 128.0


def _mongo_or_skip():
    try:
        drawing = store.get_drawing(JANADRIYAH)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip("the reference drawing has not been ingested")
    return drawing


def _units():
    return store._unit_names(_mongo_or_skip(), MODEL)


def _run(params):
    return recipes.run(
        JANADRIYAH, "h3_vicinity", params, layout=MODEL, units=_units()
    )


def _indexed_or_skip():
    _mongo_or_skip()
    if not coll(COLL_ENTITIES).find_one(
        {"drawing_id": JANADRIYAH, "h3_cell": {"$ne": None}}, {"_id": 1}
    ):
        pytest.skip("no cells yet; run scripts/backfill_h3.py")


def _layers_for(use: str) -> list[str]:
    config = landuse.for_drawing(JANADRIYAH)
    pattern_set = landuse.patterns()
    out = []
    for name in coll(COLL_ENTITIES).distinct("layer", {"drawing_id": JANADRIYAH}):
        hit = landuse.classify(
            JANADRIYAH, str(name or ""), config=config, pattern_set=pattern_set
        )
        if hit is not None and hit.is_parcel and hit.use == use:
            out.append(str(name))
    return sorted(out)


@pytest.fixture(scope="module")
def schools():
    _indexed_or_skip()
    return recipes.run(
        JANADRIYAH,
        "h3_vicinity",
        {"subject": "education"},
        layout=MODEL,
        units=store._unit_names(store.get_drawing(JANADRIYAH), MODEL),
    )


# ---------------------------------------------------------------------------
# The acceptance question
# ---------------------------------------------------------------------------


def test_it_answers_for_every_school(schools):
    assert len(schools["subjects"]) == SCHOOLS
    assert schools["chosen_by"] == "every education parcel"
    assert schools["subjects_without_cells"] == []
    for row in schools["subjects"]:
        assert row["counts"].get("residential", 0) > 0, row["handle"]
        assert row["disk_cells"] > row["own_cells"]


def test_the_reach_is_measured_not_asserted(schools):
    """The plan says k=2 is about 100 m. This is what "about" is."""
    method = schools["method"]
    assert method["resolution"] == 11
    assert method["rings"] == 2
    assert method["reach_centre_to_centre_m"] == pytest.approx(
        REACH_CENTRE_TO_CENTRE_M, abs=0.5
    )
    assert method["reach_to_disk_corner_m"] == pytest.approx(REACH_TO_CORNER_M, abs=0.5)
    # Both numbers, in the caveat, in every response.
    assert str(REACH_CENTRE_TO_CENTRE_M) in schools["caveat"]
    assert str(REACH_TO_CORNER_M) in schools["caveat"]


def test_it_agrees_with_the_exact_measurement_in_the_right_mode(schools):
    """The cross-check this phase is judged by.

    `edge`, because the rings start from every cell the subject occupies and
    therefore reach outward from its footprint. Same ballpark, in a known
    direction: a cell is counted whole, so the hexagonal answer runs at or
    above the measured one, never below by much."""
    exact = store_spatial.proximity_count(
        JANADRIYAH,
        layout=MODEL,
        around_layers=_layers_for("education"),
        count_layers=_layers_for("residential"),
        radius=REACH_CENTRE_TO_CENTRE_M,
        measure_from="edge",
    )
    measured = {row["handle"]: row["count"] for row in exact["rows"]}
    approximate = {
        row["handle"]: row["counts"].get("residential", 0)
        for row in schools["subjects"]
    }
    assert set(measured) == set(approximate)
    for handle, exact_count in measured.items():
        got = approximate[handle]
        assert got >= exact_count * 0.85, (handle, got, exact_count)
        assert got <= exact_count * 1.6, (handle, got, exact_count)


def test_the_response_points_at_the_call_that_settles_it(schools):
    """An approximation that does not name its own check is an approximation
    nobody checks."""
    check = schools["exact_check"]
    assert check["tool"] == "proximity_count"
    assert check["call"]["params"]["measure_from"] == "edge"
    assert check["call"]["params"]["radius"] == pytest.approx(
        REACH_CENTRE_TO_CENTRE_M, abs=0.5
    )
    assert check["bases_differ"]
    # And the mode that would mislead is named as such.
    assert "centroid" in check["do_not_compare_with"]


def test_the_check_it_names_is_a_call_that_can_actually_be_made(schools):
    """The bug this replaces: the block said to run `run_analysis` with recipe
    `proximity_count`, and there is no such recipe -- following the recipe's
    own instructions for settling its number returned RECIPE_UNKNOWN and the
    catalogue. proximity_count is a tool of its own.

    An approximation whose stated check fails is worse than one that states no
    check, because the failure looks like the check.
    """
    check = schools["exact_check"]
    call = check["call"]

    # Whatever it names must be reachable. If it routes through run_analysis
    # then the recipe has to exist; otherwise it must not claim a recipe at
    # all.
    named_recipe = call.get("recipe") or check.get("recipe")
    if call.get("tool") == "run_analysis":
        assert named_recipe in registry.known(), named_recipe
    else:
        assert named_recipe is None, (
            "this names a recipe but does not call run_analysis"
        )
    assert "proximity_count" not in registry.known()

    # And the arguments have to be the ones the tool takes.
    params = call["params"]
    assert params["drawing_id"] == JANADRIYAH
    assert params["layout"] == MODEL
    # Comma-separated, like the tool and the route, not a JSON list.
    assert isinstance(params["around_layers"], str)
    assert isinstance(params["count_layers"], str)


def test_the_direction_of_the_error_is_not_claimed_to_be_one_sided(schools):
    """`bases_differ` used to say the hexagon count comes out "between equal
    and 25 per cent above" the exact one. It does not always: measured on the
    reference drawing one school came out 1.8 per cent BELOW, because a parcel
    inside the radius but beyond a flat side of the disk falls in no counted
    cell.

    The bound this file already tests against allows below (0.85), so the
    prose was contradicting its own test.
    """
    prose = schools["exact_check"]["bases_differ"]
    assert "not one-sided" in prose.lower() or "below" in prose.lower()


def test_centroid_mode_is_a_different_question_and_the_recipe_says_so():
    """The mistake the first version made, kept as a test so it cannot come
    back: at the same radius, centroid mode answers roughly a third, because
    the school is 74 m across and the rings start from its boundary."""
    _indexed_or_skip()
    centroid = store_spatial.proximity_count(
        JANADRIYAH,
        layout=MODEL,
        around_layers=["Primary School"],
        count_layers=_layers_for("residential"),
        radius=REACH_CENTRE_TO_CENTRE_M,
        measure_from="centroid",
    )
    edge = store_spatial.proximity_count(
        JANADRIYAH,
        layout=MODEL,
        around_layers=["Primary School"],
        count_layers=_layers_for("residential"),
        radius=REACH_CENTRE_TO_CENTRE_M,
        measure_from="edge",
    )
    by_centroid = {r["handle"]: r["count"] for r in centroid["rows"]}[SCHOOL]
    by_edge = {r["handle"]: r["count"] for r in edge["rows"]}[SCHOOL]
    assert by_edge > by_centroid * 1.5
    # The school's own width is the difference, not an error in either.
    assert SCHOOL_ACROSS_M > 50


# ---------------------------------------------------------------------------
# What it refuses to pretend
# ---------------------------------------------------------------------------


def test_it_says_what_it_did_not_measure(schools):
    """`carries_meaning` is True, so the envelope demands evidence; the
    approximation still needs `not_measured`, and the recipe carries both."""
    assert schools["evidence"]["grade"]
    assert "true distance" in schools["not_measured"]
    assert "walking" in schools["not_measured"].lower()
    assert schools["scope_note"]


def test_the_labels_are_the_drawings_own_and_say_so(schools):
    assert "land_use_summary" in schools["land_use_basis"]
    assert "establishes no" in schools["land_use_basis"]


def test_a_subject_is_never_counted_as_its_own_neighbour(schools):
    for row in schools["subjects"]:
        assert row["handle"] not in row["neighbours_sample"]


def test_each_neighbour_is_counted_once_however_many_cells_it_touches(schools):
    for row in schools["subjects"]:
        assert row["neighbours_total"] == sum(row["counts"].values())
        assert len(set(row["neighbours_sample"])) == len(row["neighbours_sample"])


# ---------------------------------------------------------------------------
# Parameters (G7, and clear refusals)
# ---------------------------------------------------------------------------


def test_a_bigger_k_reaches_further_and_counts_more():
    _indexed_or_skip()
    near = _run({"subject": SCHOOL, "k": 2})
    far = _run({"subject": SCHOOL, "k": 4})
    assert (
        far["method"]["reach_centre_to_centre_m"]
        > near["method"]["reach_centre_to_centre_m"]
    )
    assert far["subjects"][0]["counts"]["residential"] > near["subjects"][0][
        "counts"
    ]["residential"]


@pytest.mark.parametrize("k", [0, -1, vicinity.MAX_K + 1, 500])
def test_a_ring_count_outside_the_limit_is_refused_with_the_reach(k):
    _indexed_or_skip()
    with pytest.raises(RecipeRefused) as caught:
        _run({"subject": SCHOOL, "k": k})
    assert caught.value.code == "RECIPE_PARAM_INVALID"
    assert caught.value.hint


def test_a_resolution_finer_than_the_stored_one_is_refused():
    """The same rule the /geo route enforces: a finer ring would be drawn
    around cells that were never computed."""
    _indexed_or_skip()
    with pytest.raises(RecipeRefused) as caught:
        _run({"subject": SCHOOL, "res": 15})
    assert caught.value.code == "RECIPE_PARAM_INVALID"
    assert "13" in caught.value.message


def test_a_subject_that_is_neither_a_handle_nor_a_land_use_is_refused():
    _indexed_or_skip()
    with pytest.raises(RecipeRefused) as caught:
        _run({"subject": "somewhere nice"})
    assert caught.value.code == "RECIPE_PARAM_INVALID"
    # The refusal lists what WOULD work.
    assert "education" in caught.value.hint


def test_ringing_every_house_is_refused_rather_than_attempted():
    """G7. 2,380 subjects is a cross-product, not an answer."""
    _indexed_or_skip()
    with pytest.raises(RecipeRefused) as caught:
        _run({"subject": "residential"})
    assert caught.value.code == "RECIPE_INPUT_TOO_LARGE"
    assert str(vicinity.MAX_SUBJECTS) in caught.value.message


# ---------------------------------------------------------------------------
# The rules every recipe is bound by
# ---------------------------------------------------------------------------


def test_it_is_in_the_catalogue_and_adds_no_tool():
    """The plan's constraint, and the reason this is a recipe at all: new
    capability rides `run_analysis`, and the MCP tool count does not move."""
    names = {r["recipe"] for r in recipes.catalog()["recipes"]}
    assert "h3_vicinity" in names
    entry = next(r for r in recipes.catalog()["recipes"] if r["recipe"] == "h3_vicinity")
    assert entry["limits"]["rings"] == vicinity.MAX_K
    assert entry["limits"]["subjects"] == vicinity.MAX_SUBJECTS
    assert "proximity_count" in entry["when_to_use"]


def test_the_same_call_twice_gives_the_same_answer():
    """Determinism is the first rule in `registry.py`, and an answer that
    changes without the drawing changing is not a measurement."""
    _indexed_or_skip()
    first = _run({"subject": SCHOOL})
    second = _run({"subject": SCHOOL})
    assert first["subjects"] == second["subjects"]
    assert first["method"] == second["method"]


def test_a_drawing_with_no_coordinate_system_is_refused_by_name():
    """BINDING 3 again, at the recipe layer. No cells, no rings — and the
    refusal names the recipe that needs no coordinate system at all."""
    _mongo_or_skip()
    # Drawings with NO per-drawing config, derived from the config tree: a
    # drawing that ships one is georeferenced on purpose and is not evidence
    # of a leak. See tests/conftest.py.
    others = unconfigured_drawing_ids()
    if not others:
        pytest.skip("every ingested drawing has a config")
    with pytest.raises(RecipeRefused) as caught:
        recipes.run(
            others[0],
            "h3_vicinity",
            {"subject": "education"},
            layout=MODEL,
            units={},
        )
    assert caught.value.code == "RECIPE_NO_CONFIG"
    assert "proximity_count" in caught.value.hint


def test_an_empty_network_block_is_explained_rather_than_left_to_read_as_none(
    schools,
):
    """The finding this test was written for, the other way round.

    It was written asserting that some road must be near some school, and it
    failed: this drawing classifies NO layer as a network. The road
    centrelines are there — 1,233 of them in model space — on a layer whose
    config has never been given a `role: network` entry. So `network` is
    empty for every subject, and an empty block reads exactly like "there are
    no roads near the school", which is false.

    What the recipe owes is the difference between the two, and that is what
    is asserted here: the absence of a configured role, named as such, with
    the file that would fix it."""
    assert schools["network_layers"] == []
    for row in schools["subjects"]:
        assert row["network"] == {}
    note = schools["network_note"]
    assert "NO layer as a network" in note
    assert "not a statement that there are no roads" in note
    assert "role: network" in note


def test_network_lengths_are_summed_per_layer_with_their_unit(monkeypatch):
    """The capability itself, tested where the config cannot reach it.

    The drawing has no network layer, so the only way to test the code that
    measures one is to classify a layer as network for the length of this
    test. What is faked is the CONFIG — one layer's role — and nothing else:
    the cells, the lengths and the rings are the real ones."""
    _indexed_or_skip()
    road_layer = "00_Prop - Road - CL_"
    real = landuse.classify

    def as_network(drawing_id, layer_name, **kwargs):
        hit = real(drawing_id, layer_name, **kwargs)
        if str(layer_name) == road_layer:
            return landuse.LandUse(
                use="road",
                subtype=None,
                role=landuse.NETWORK_ROLE,
                layer=road_layer,
                sources=(),
                config_layer=(hit.config_layer if hit else None)
                or __import__("app.evidence", fromlist=["x"]).ConfigLayer.NO_CONFIG,
                config_version=None,
            )
        return hit

    monkeypatch.setattr(landuse, "classify", as_network)
    body = _run({"subject": SCHOOL})

    assert road_layer in body["network_layers"]
    assert "Σ over the network objects" in body["network_note"]
    block = body["subjects"][0]["network"].get(road_layer)
    assert block, "the school has no road near it, which cannot be right"
    assert block["objects"] > 0
    assert block["length"] > 0
    assert block["length_unit"] == "m"


# ---------------------------------------------------------------------------
# Wave 2 P5 — ring bands and the dissolved outline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def banded():
    _indexed_or_skip()
    return recipes.run(
        JANADRIYAH,
        "h3_vicinity",
        {"subject": SCHOOL, "bands": True, "outline": True},
        layout=MODEL,
        units=store._unit_names(store.get_drawing(JANADRIYAH), MODEL),
    )


def test_the_bands_sum_to_the_total_they_refine(banded):
    """DoD. A breakdown that does not add up to the number it breaks down is
    two answers to one question, and the reader has no way to tell which is
    the one to quote."""
    row = banded["subjects"][0]
    summed: dict[str, int] = {}
    for band in row["bands"]:
        for use, count in band["counts"].items():
            summed[use] = summed.get(use, 0) + count
    assert summed == row["counts"]
    assert sum(b["total"] for b in row["bands"]) == sum(row["counts"].values())


def test_there_is_one_band_per_ring_including_the_subjects_own(banded):
    """Ring 0 is the subject's own footprint — the parcels sharing its cells.
    Dropping it would lose them from the breakdown while leaving them in the
    total, which is the arithmetic above failing quietly."""
    row = banded["subjects"][0]
    rings = [b["ring"] for b in row["bands"]]
    assert rings == list(range(banded["method"]["rings"] + 1))
    assert row["bands"][0]["from_m"] == 0.0
    assert row["bands"][0]["to_m"] == 0.0


def test_each_band_is_labelled_with_the_step_that_was_measured(banded):
    """The bands carry metres, and the metres are the measured step rather
    than a round number: 49.6 m at resolution 11, not 50."""
    step = banded["method"]["step_between_cells_m"]
    for band in banded["subjects"][0]["bands"][1:]:
        assert band["to_m"] == pytest.approx(band["ring"] * step, abs=0.15)
        assert band["from_m"] == pytest.approx((band["ring"] - 1) * step, abs=0.15)
    assert banded["bands_note"]
    assert "hexagon boundary and not a circle" in banded["bands_note"]


def test_bands_are_further_out_the_further_the_ring(banded):
    row = banded["subjects"][0]
    for near, far in zip(row["bands"], row["bands"][1:]):
        assert far["from_m"] >= near["to_m"]


def test_no_bands_unless_they_were_asked_for():
    """An answer that always carries an optional breakdown is an answer whose
    optional part nobody trusts to be meaningful."""
    _indexed_or_skip()
    plain = _run({"subject": SCHOOL})
    assert plain["subjects"][0]["bands"] is None
    assert plain["bands_note"] is None


def test_the_outline_is_one_dissolved_boundary_not_a_pile_of_hexagons(banded):
    """DoD. A catchment drawn as 37 hexagons has 37 outlines, most of them
    internal, and every shared edge drawn twice."""
    outline = banded["subjects"][0]["outline"]
    assert outline["type"] == "Feature"
    assert outline["geometry"]["type"] == "MultiPolygon"
    assert outline["properties"]["cells"] == banded["subjects"][0]["disk_cells"]
    # One dissolved shape, and far fewer vertices than 37 hexagons would have.
    rings = [r for poly in outline["geometry"]["coordinates"] for r in poly]
    vertices = sum(len(r) for r in rings)
    assert vertices < 37 * 6


def test_the_outline_is_geojson_the_way_geojson_wants_it(banded):
    """[lon, lat], closed rings, MultiPolygon nested one level deeper than a
    Polygon. All three are things a renderer silently gets wrong rather than
    refusing."""
    geometry = banded["subjects"][0]["outline"]["geometry"]
    for poly in geometry["coordinates"]:
        for ring in poly:
            assert ring[0] == ring[-1], "ring is not closed"
            assert len(ring) >= 4
            for lon, lat in ring:
                assert 46.84 < lon < 46.94, (lon, lat)
                assert 24.80 < lat < 24.90, (lon, lat)


def test_the_outline_says_it_is_only_as_good_as_the_hexagons(banded):
    """It looks like a catchment boundary, and somebody will treat it as one.
    The properties say what it actually is."""
    assert "hexagons" in banded["subjects"][0]["outline"]["properties"]["note"]
    assert banded["outline_note"]


def test_no_outline_unless_it_was_asked_for():
    _indexed_or_skip()
    assert _run({"subject": SCHOOL})["subjects"][0]["outline"] is None


def test_an_empty_set_of_cells_has_no_outline_rather_than_an_empty_one():
    """An empty MultiPolygon renders as nothing while claiming to be a shape."""
    from app import geo_h3

    assert geo_h3.outline_of([]) is None


def test_the_two_new_parameters_are_in_the_catalogue_and_add_no_tool():
    entry = next(
        r for r in recipes.catalog()["recipes"] if r["recipe"] == "h3_vicinity"
    )
    names = {p["name"] for p in entry["params"]}
    assert {"bands", "outline"} <= names
    for param in entry["params"]:
        if param["name"] in {"bands", "outline"}:
            assert param["default"] is False, "both must be opt-in"
            assert param["required"] is False
