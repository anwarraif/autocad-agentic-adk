"""`catchment_coverage` — the RCDC poster's 400/800 m service catchment.

The poster names "catchment 400/800 m" as a Spatial Analysis element. This
recipe answers it radially over the cells that ingest already writes, and the
thing that makes it useful rather than decorative is the COMPLEMENT: it names
the parcels that fall outside the catchment. A coverage percentage nobody can
act on is a number, not an answer.

The ground truth pinned here is Janadriyah, measured 3 Sep 2026 through the
recipe itself and cross-checked against `h3_vicinity`, which reads the same
index through the same helper.
"""

from __future__ import annotations

import pytest

from app import recipes, store
from app.recipes.registry import RecipeRefused

JANADRIYAH = "596212db022a3397"
LAYOUT = "Model"

#: Janadriyah holds 2,380 residential parcels. Every coverage figure below is
#: a share of this number, so if it moves, the percentages mean something
#: different and the rest of this file is wrong rather than merely failing.
RESIDENTIAL_PARCELS = 2380

#: Measured: 9 education parcels, 6 religious.
EDUCATION_PARCELS = 9
RELIGIOUS_PARCELS = 6


def _units(drawing_id: str = JANADRIYAH):
    drawing = store.get_drawing(drawing_id)
    return drawing, store._unit_names(drawing, LAYOUT)


def _run(params, drawing_id: str = JANADRIYAH):
    _drawing, units = _units(drawing_id)
    body = recipes.run(
        drawing_id, "catchment_coverage", params, layout=LAYOUT, units=units
    )
    return body.get("result", body)


def test_education_catchment_on_janadriyah_is_the_measured_figure():
    """Ground truth: 400 m reaches 2,284 of 2,380 houses; 800 m reaches all."""
    result = _run({"amenity": "education", "radii": "400,800"})

    assert result["amenity"]["count"] == EDUCATION_PARCELS
    assert result["amenity"]["with_cells"] == EDUCATION_PARCELS
    assert result["population"] == RESIDENTIAL_PARCELS

    near, far = result["bands"]
    assert near["radius_m"] == 400.0
    assert near["served"] == 2284
    assert near["unserved"] == 96
    assert near["coverage_pct"] == 95.97

    assert far["radius_m"] == 800.0
    assert far["served"] == RESIDENTIAL_PARCELS
    assert far["unserved"] == 0
    assert far["coverage_pct"] == 100.0


def test_religious_catchment_on_janadriyah_is_the_measured_figure():
    result = _run({"amenity": "religious", "radii": "400,800"})

    assert result["amenity"]["count"] == RELIGIOUS_PARCELS
    near, far = result["bands"]
    assert near["served"] == 2378
    assert near["unserved"] == 2
    assert near["coverage_pct"] == 99.92
    assert far["coverage_pct"] == 100.0


def test_the_unserved_parcels_are_named_not_merely_counted():
    """The reason this recipe exists rather than a bare percentage."""
    result = _run({"amenity": "education", "radii": "400"})
    band = result["bands"][0]

    assert len(band["unserved_handles"]) == band["unserved"] == 96
    assert band["unserved_truncated"] is False
    # Handles, not indexes: every one has to be openable in the viewer.
    assert all(isinstance(h, str) and h for h in band["unserved_handles"])
    assert "205C78A" in band["unserved_handles"]


def test_served_and_unserved_are_a_partition_of_the_population():
    """They are a set and its complement, so they cannot disagree."""
    result = _run({"amenity": "education", "radii": "400,800"})
    for band in result["bands"]:
        assert band["served"] + band["unserved"] == result["population"]


def test_a_wider_radius_never_serves_fewer_parcels():
    """Monotonic by construction; asserted because an off-by-one in the ring
    arithmetic would not otherwise be visible in a percentage."""
    result = _run({"amenity": "education", "radii": "400,800"})
    served = [band["served"] for band in result["bands"]]
    assert served == sorted(served)


def test_it_agrees_exactly_with_h3_vicinity_on_the_same_subject():
    """The anti-duplication proof, and the reason the index is shared.

    `h3_vicinity` counts what surrounds ONE subject; `catchment_coverage`
    counts the union over a set. Given the same single subject and the same
    ring count they are the same question, so they must return the same
    number. Measured: 843 residential parcels around the Primary School
    (handle 205D153) at k=8, which is what 400 m resolves to at resolution 11.

    If these two ever diverge, the two recipes are reading different
    populations and at least one of them is wrong.
    """
    _drawing, units = _units()
    vicinity = recipes.run(
        JANADRIYAH, "h3_vicinity", {"subject": "205D153", "k": 8},
        layout=LAYOUT, units=units,
    )
    around = vicinity.get("result", vicinity)["subjects"][0]["counts"]["residential"]

    catchment = _run({"amenity": "205D153", "radii": "400"})
    band = catchment["bands"][0]

    assert band["rings"] == 8, "400 m must resolve to k=8 at resolution 11"
    assert band["served"] == around == 843


def test_the_reach_actually_used_is_stated_not_the_radius_asked_for():
    """A radius becomes whole rings, so 400 m is really 397.2 m."""
    band = _run({"amenity": "education", "radii": "400"})["bands"][0]
    assert band["rings"] == 8
    assert band["reach"]["reach_centre_to_centre_m"] == 397.2
    # The corner reaches further than the centre, and both are published.
    assert band["reach"]["reach_to_disk_corner_m"] > 397.2


def test_the_approximation_and_the_exact_check_are_carried():
    """A radial hexagon figure read as a walking distance is the failure mode."""
    result = _run({"amenity": "education", "radii": "400"})
    caveat = result["caveat"]
    assert "hexagon" in caveat["shape"]
    assert "NOT walking distance" in caveat["distance"]
    assert "proximity_count" in result["exact_check"]
    assert "not_measured" in result and result["not_measured"]


def test_it_says_the_counts_are_the_drawings_own_classification():
    result = _run({"amenity": "education", "radii": "400"})
    assert "establishes no" in result["land_use_basis"]


# --- parameters and refusals ------------------------------------------------


def test_radii_defaults_to_the_two_the_poster_names():
    result = _run({"amenity": "education"})
    assert [band["radius_m"] for band in result["bands"]] == [400.0, 800.0]


def test_a_radius_that_is_not_a_number_is_refused_with_an_example():
    with pytest.raises(RecipeRefused) as caught:
        _run({"amenity": "education", "radii": "400,soon"})
    assert caught.value.code == "RECIPE_PARAM_INVALID"
    assert "400,800" in caught.value.hint


def test_a_radius_beyond_the_ceiling_is_refused_and_names_the_alternative():
    with pytest.raises(RecipeRefused) as caught:
        _run({"amenity": "education", "radii": "50000"})
    assert caught.value.code == "RECIPE_PARAM_INVALID"
    assert "parcel_inventory" in caught.value.hint


def test_too_many_radii_are_refused():
    with pytest.raises(RecipeRefused) as caught:
        _run({"amenity": "education", "radii": "100,200,300,400,500,600"})
    assert caught.value.code == "RECIPE_INPUT_TOO_LARGE"


def test_duplicate_radii_are_collapsed_and_sorted():
    result = _run({"amenity": "education", "radii": "800,400,400"})
    assert [band["radius_m"] for band in result["bands"]] == [400.0, 800.0]


def test_an_amenity_that_is_neither_a_handle_nor_a_land_use_is_refused():
    with pytest.raises(RecipeRefused) as caught:
        _run({"amenity": "helipads"})
    assert caught.value.code == "RECIPE_PARAM_INVALID"
    # The refusal has to say what this drawing DOES know.
    assert "education" in caught.value.hint


def test_a_drawing_without_a_coordinate_system_is_refused_not_guessed():
    """No CRS means no cells; inventing a catchment would be inventing a place."""
    ungeoreferenced = "3341288e9fab8bb5"  # lineweights.dxf, no CRS config
    with pytest.raises(RecipeRefused) as caught:
        _run({"amenity": "education"}, drawing_id=ungeoreferenced)
    assert caught.value.code in {"RECIPE_NO_CONFIG", "RECIPE_PARAM_INVALID"}


def test_the_catalogue_entry_states_the_limits_and_what_it_stands_on():
    recipe = recipes.get("catchment_coverage")
    assert recipe is not None
    assert recipe.carries_meaning is True
    assert recipe.limits["radii"] == 5
    assert "h3.grid_disk" in recipe.built_on
    assert "proximity_count" in recipe.when_to_use
