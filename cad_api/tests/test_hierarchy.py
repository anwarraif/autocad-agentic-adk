"""`road_hierarchy` — junction spacing, cul-de-sacs, and corridor width.

The RCDC poster lists "road hierarchy ROW/junctions/cul-de-sac" as a Spatial
Analysis element. Three of those are measurable from what ingest already
stores. The fourth, a road's CLASS, is not in the drawing and is not invented
here, and the test at the bottom holds that line.

Ground truth is Janadriyah, measured 3 Sep 2026. The numbers that matter most
are the ones shared with `junction_census`: both recipes run the same network
preparation, so if they ever disagree about the node count, the path count or
the dead ends, one of them is describing a different copy of the geometry and
the answer is wrong in a way no reader could see.
"""

from __future__ import annotations

import pytest

from app import recipes, store
from app.recipes.registry import RecipeRefused

JANADRIYAH = "596212db022a3397"
LAYOUT = "Model"

#: Measured. The reference road layers hold the same estate four times; copy 1
#: is the live one, identified by measurement rather than by position.
PATHS_IN_LIVE_COPY = 748
NODES_TOTAL = 792

#: Dead ends are a BRACKET, never one number: a node whose through-road is a
#: curve the store could not resolve might be a dead end and might not.
DEAD_ENDS_AT_LEAST = 14
DEAD_ENDS_AT_MOST = 409


def _units(drawing_id: str = JANADRIYAH):
    drawing = store.get_drawing(drawing_id)
    return store._unit_names(drawing, LAYOUT)


def _run(params=None, drawing_id: str = JANADRIYAH):
    body = recipes.run(
        drawing_id,
        "road_hierarchy",
        params or {},
        layout=LAYOUT,
        units=_units(drawing_id),
    )
    return body.get("result", body)


def test_junction_spacing_on_janadriyah_is_the_measured_distribution():
    """25 segments run junction to junction; the median block is 65 m."""
    spacing = _run()["junction_spacing"]

    assert spacing["count"] == 25
    assert spacing["unit"] == "m"
    assert spacing["min"] == 13.818
    assert spacing["p25"] == 55.0
    assert spacing["median"] == 65.0
    assert spacing["p75"] == 153.008
    assert spacing["max"] == 392.212


def test_the_spacing_says_what_it_measured_between_and_on_what_basis():
    """A block length with no stated basis is a number, not a measurement."""
    spacing = _run()["junction_spacing"]

    assert "degree 3 or more" in spacing["measured_between"]
    assert "stored `length`" in spacing["basis"]
    # The snap tolerance decides whether a node exists at all, so it is part
    # of the answer rather than an implementation detail.
    assert spacing["tolerance"]["snap_tolerance"] > 0
    assert spacing["tolerance"]["unit"] == "m"


def test_the_paths_that_are_not_spacings_are_counted_never_dropped():
    """748 paths, 25 spacings: the other 723 have to be accounted for."""
    result = _run()
    spacing = result["junction_spacing"]
    excluded = spacing["excluded"]

    assert set(excluded) == {
        "one_end_only",
        "no_length_stored",
        "not_between_junctions",
    }
    assert spacing["count"] + sum(excluded.values()) == result["network"][
        "paths_in_this_copy"
    ] == PATHS_IN_LIVE_COPY


def test_cul_de_sacs_are_carried_from_the_census_as_a_bracket():
    cul = _run()["cul_de_sacs"]

    assert cul["at_least"] == DEAD_ENDS_AT_LEAST
    assert cul["at_most"] == DEAD_ENDS_AT_MOST
    # The bracket is open, so there is no exact answer and none is invented.
    assert cul["exact"] is None


def test_it_agrees_with_junction_census_about_the_network_it_described():
    """The reason both recipes share one preparation.

    If these diverge, the two recipes clustered different nodes at different
    tolerances on different copies of the same geometry, and every figure in
    both answers is about a different drawing.
    """
    units = _units()
    census_body = recipes.run(
        JANADRIYAH, "junction_census", {}, layout=LAYOUT, units=units
    )
    census = census_body.get("result", census_body)
    hierarchy = _run()

    assert hierarchy["network"]["nodes_total"] == census["nodes_total"] == NODES_TOTAL
    assert (
        hierarchy["network"]["paths_in_this_copy"]
        == census["paths_in_this_copy"]
        == PATHS_IN_LIVE_COPY
    )
    assert hierarchy["cul_de_sacs"]["at_least"] == census["junctions"]["dead_ends"][
        "at_least"
    ]
    assert hierarchy["cul_de_sacs"]["at_most"] == census["junctions"]["dead_ends"][
        "at_most"
    ]


def test_the_corridor_width_is_measured_and_says_how():
    """Measured: the signal ranks `ROW` first and its median width is 32.1 m.

    The layer happens to be NAMED after a right of way, which is a coincidence
    this recipe is not allowed to use: the ranking is by how much of the road
    network falls inside the rings and how few parcels do, and it never reads
    a name. That the name agrees is a corroboration, not the method.
    """
    corridor = _run()["corridor"]

    assert corridor["measured"] is True
    assert corridor["layer"] == "ROW"
    assert corridor["rings_measured"] == 80
    assert corridor["unit"] == "m"
    assert corridor["width"]["median"] == 32.108
    assert corridor["width"]["min"] < corridor["width"]["median"] < corridor["width"]["max"]
    assert "area divided by its perimeter" in corridor["method"]
    assert "long and thin" in corridor["assumption"]
    assert "never reads a layer name" in corridor["chosen_by"]
    # The signal's own score travels with it. Asking for `score` instead of
    # `corridor_score` published a null that read like a failed measurement.
    assert corridor["signal"]["corridor_score"] is not None
    assert corridor["signal"]["corridor_score"] > 0
    assert corridor["signal"]["verdict"]


def test_the_corridor_publishes_the_length_that_tests_its_own_assumption():
    """Width from 2A/P only means anything when the ring is long and thin.

    So the implied length is published beside it: a reader who sees a median
    width of 32 m against a median length of 159 m can see the assumption
    holding, and would see it fail if the two were close.
    """
    corridor = _run()["corridor"]
    assert corridor["implied_length_median"] == 159.0
    assert corridor["implied_length_median"] > corridor["width"]["median"] * 2


def test_it_refuses_to_classify_a_road():
    """The line this recipe does not cross.

    Nothing in a masterplan DXF states that a strip of tarmac is a collector
    rather than a local street. A hierarchy inferred from width alone would be
    a classification wearing a measurement's clothes, and the response says so
    rather than leaving the reader to assume otherwise.
    """
    result = _run()
    assert "CLASS of any road" in result["not_measured"]
    assert "Right of way is not measured" in result["not_measured"]

    # And the registry agrees: this recipe measures, it does not mean.
    recipe = recipes.get("road_hierarchy")
    assert recipe.carries_meaning is False


def test_the_network_block_names_the_copy_and_why_it_was_chosen():
    network = _run()["network"]
    assert network["copies"] == 4
    assert network["copy_described"] == 1
    assert "measurement" in network["copy_chosen_by"]
    assert network["layers_source"]


def test_a_negative_snap_tolerance_is_refused():
    with pytest.raises(RecipeRefused) as caught:
        _run({"snap_tolerance": -1})
    assert caught.value.code == "RECIPE_PARAM_INVALID"


def test_a_copy_index_outside_the_range_is_refused_and_says_the_range():
    with pytest.raises(RecipeRefused) as caught:
        _run({"copy_index": 99})
    assert caught.value.code == "RECIPE_PARAM_INVALID"
    assert "4" in caught.value.message


def test_the_catalogue_entry_stands_on_the_shared_preparation():
    recipe = recipes.get("road_hierarchy")
    assert recipe is not None
    assert "junctions.prepare_network" in recipe.built_on
    assert "corridor.rank_region_layers" in recipe.built_on
    assert recipe.limits["corridor_rings"] > 0
    assert "does NOT classify" in recipe.when_to_use
