"""DOSSIER Phase 8a — the corridor signal.

Owner: the CORRIDOR-SIGNAL lane.

**Known truth first, and the known truth here is a trap.** The fixture drawing
below is built so that the WRONG answer is obvious and available: its largest
closed-ring layer is a site boundary that contains everything, exactly as the
reference drawing's is, and a ranking by area would offer it first. The single
most important test in this file is therefore the negative one: **the layer
that contains every parcel must not be proposed, however large it is.**

Four kinds of test, in this order:

1.  **Pure arithmetic** — the score, the fraction that refuses to divide by
    zero, the point preference, the spread of the sample. No database, no
    Dossier, exact expected numbers.
2.  **A fixture drawing** whose every answer is known by hand: one corridor
    layer, one site boundary, two parcel layers, one network layer, and a
    distractor that is bigger than all of them.
3.  **The degradations** — no Dossier, no network geometry, no parcel
    geometry, a population above the stated cap. Each must state a reason and
    none may fall back to ranking by size.
4.  **Janadriyah**, skipped when there is no database. `ROW` must come first,
    ahead of layer `0` and `New Boundry`, with the two fractions that put it
    there — and layer names live in this file, never in the module.

**One note for the integrator.** This module registers no recipe and is not in
the catalogue, so `test_recipes.py` needs no change from this lane.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import dossier_read  # noqa: E402
from app import geometry as geom  # noqa: E402
from app.recipes import corridor as cr  # noqa: E402


# =============================================================================
# Tooling
# =============================================================================

DRAWING = "596212db022a3397"
FIXTURE = "fixture000000cafe"
LAYOUT = "Model"

#: Layer names live in the TEST, never in the module under test (G1). Every one
#: of these is deliberately meaningless: if the signal worked because a name
#: said "road", these names would break it.
CORRIDOR = "zeta-7"
BOUNDARY = "alpha-1"
PLOTS_A = "kappa-3"
PLOTS_B = "kappa-4"
NETWORK = "mu-9"
EMPTY = "nu-0"


class FakeCursor(list):
    """Just enough cursor for `.find(...).sort(...)`."""

    def sort(self, spec):
        for key, direction in reversed(list(spec)):
            list.sort(
                self,
                key=lambda d, k=key: (d.get(k) is None, str(d.get(k))),
                reverse=direction < 0,
            )
        return self


class FakeCollection:
    """Just enough MongoDB for this file, and not one line more."""

    def __init__(self, docs=()):
        self.docs = [copy.deepcopy(d) for d in docs]
        self.filters: list[dict] = []

    def _match(self, doc, flt):
        for key, want in flt.items():
            got = doc.get(key)
            if isinstance(want, dict):
                for op, arg in want.items():
                    if op == "$in":
                        if got not in arg:
                            return False
                    else:
                        raise AssertionError(f"operator {op} has not been faked")
            elif got != want:
                return False
        return True

    def find(self, flt, projection=None):
        self.filters.append(copy.deepcopy(flt))
        return FakeCursor(copy.deepcopy(d) for d in self.docs if self._match(d, flt))


def square(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def centre(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return [(min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0]


def ring_doc(handle, layer, ring, *, layout=LAYOUT, drawing=FIXTURE):
    return {
        "_id": f"{drawing}:{handle}",
        "drawing_id": drawing,
        "layout": layout,
        "layer": layer,
        "handle": handle,
        "type": "LWPOLYLINE",
        "ring": [[float(p[0]), float(p[1])] for p in ring],
        "ring_status": "complete",
        "ring_origin": None,
        "area": abs(geom.signed_area(ring)),
        "polygon_centroid": centre(ring),
        "bbox_centre": centre(ring),
    }


def path_doc(handle, layer, point, *, layout=LAYOUT, drawing=FIXTURE):
    """A road drawn as a LINE: a bounding-box centre, and no ring anywhere."""
    return {
        "_id": f"{drawing}:{handle}",
        "drawing_id": drawing,
        "layout": layout,
        "layer": layer,
        "handle": handle,
        "type": "LINE",
        "ring_status": None,
        "bbox_centre": [float(point[0]), float(point[1])],
    }


def block(layer, role, *, entities, area=None, layout=LAYOUT):
    return {
        "layer": layer,
        "layout": layout,
        "role": role,
        "entities": entities,
        "measure": {
            "kind": "area" if role == "region" else "length",
            "value": area,
            "unit": "m2" if role == "region" else "m",
            "unit_reason": "the layout declares metres",
        },
    }


#: The fixture drawing, whose every answer is known by hand.
#:
#: A 100 x 100 site. `alpha-1` is a single ring around ALL of it -- the site
#: boundary, and by area five times the biggest thing in the drawing.
#: `zeta-7` is two north-south corridors, 10 wide, at x 20-30 and x 70-80,
#: which the road network runs down and which no plot sits in. `kappa-3` and
#: `kappa-4` are four plots each, in the bands the corridors leave behind.
#: `mu-9` is the road network: ten points, five down each corridor.
#:
#: The corridors are deliberately NOT central. A representative point stands
#: for a whole entity, and the site boundary's own point is its centre: a
#: corridor drawn through the middle of the site would contain that point and
#: the fixture would then be measuring an artefact of the proxy instead of the
#: rule under test. That artefact is real and is stated in the module; it just
#: has no business inside the one test that pins the arithmetic.
#:
#: Measured by hand:
#:   `zeta-7`  holds 10 of 10 network points and 0 of 9 parcel points -> 1.0
#:   `alpha-1` holds 10 of 10 network points and 10 of 10 parcel points -> 0.0
#:   a plot layer holds 0 of 10 network points -> 0.0
def _fixture_docs():
    docs = [ring_doc("SITE", BOUNDARY, square(0, 0, 100, 100))]
    docs.append(ring_doc("COR-W", CORRIDOR, square(20, 0, 10, 100)))
    docs.append(ring_doc("COR-E", CORRIDOR, square(70, 0, 10, 100)))
    # Four plots per layer: `kappa-3` in the western band, `kappa-4` in the
    # middle one. Neither touches a corridor.
    for i in range(4):
        docs.append(ring_doc(f"PA-{i}", PLOTS_A, square(2, 5 + i * 20, 15, 15)))
        docs.append(ring_doc(f"PB-{i}", PLOTS_B, square(35, 5 + i * 20, 15, 15)))
    # Ten road points, five down each corridor.
    for i in range(5):
        docs.append(path_doc(f"R-W{i}", NETWORK, (25.0, 10.0 + i * 20.0)))
        docs.append(path_doc(f"R-E{i}", NETWORK, (75.0, 10.0 + i * 20.0)))
    return docs


FIXTURE_DOSSIER = {
    "drawing_id": FIXTURE,
    "layers": [
        block(BOUNDARY, "region", entities=1, area=10_000.0),
        block(CORRIDOR, "region", entities=2, area=2_000.0),
        block(PLOTS_A, "region", entities=4, area=900.0),
        block(PLOTS_B, "region", entities=4, area=900.0),
        block(NETWORK, "network", entities=10, area=500.0),
    ],
}


@pytest.fixture
def fixture_store(monkeypatch):
    """The fixture drawing, its Dossier, and no land use config."""
    entities = FakeCollection(_fixture_docs())
    monkeypatch.setattr(cr, "_entities", lambda: entities)
    monkeypatch.setattr(
        dossier_read,
        "dossier_for",
        lambda drawing_id: (
            copy.deepcopy(FIXTURE_DOSSIER) if drawing_id == FIXTURE else None
        ),
    )
    # G3: no config. The parcel population must then come from the geometric
    # `region` role, and the answer must still be an answer.
    monkeypatch.setattr(cr.landuse, "for_drawing", lambda drawing_id: None)
    return entities


def _rows(block_out):
    return {str(row["layer"]): row for row in block_out["candidates"]}


# =============================================================================
# 1. Pure arithmetic — no database, no Dossier
# =============================================================================


@pytest.mark.parametrize(
    "network,parcels,expected,why",
    [
        (1.0, 0.0, 1.0, "all of the network, none of the parcels: a corridor"),
        (1.0, 1.0, 0.0, "all of both: a site boundary, and the trap"),
        (0.0, 0.0, 0.0, "none of the network: not a corridor, whatever else"),
        (0.5, 0.5, 0.25, "the product, computed by hand"),
        (0.4, 0.75, 0.1, "the reference drawing's own numbers, rounded"),
    ],
)
def test_the_score_is_the_product_and_its_two_zeros_are_the_two_wrong_answers(
    network, parcels, expected, why
):
    assert cr.corridor_score(network, parcels) == pytest.approx(expected, abs=1e-9), why


def test_a_layer_holding_every_parcel_scores_zero_however_much_network_it_holds():
    """The site-boundary trap, stated on its own because it is the whole point.

    Ranked by area the reference drawing offers the site boundary; every parcel
    is inside it; the retry reports universal frontage. No amount of network
    inside a layer may rescue a layer that contains all the parcels.
    """
    assert cr.corridor_score(1.0, 1.0) == 0.0
    assert cr.corridor_score(0.99, 1.0) == 0.0


def test_a_layer_holding_no_network_scores_zero_however_few_parcels_it_holds():
    assert cr.corridor_score(0.0, 0.0) == 0.0
    assert cr.corridor_score(0.0, 0.01) == 0.0


def test_a_difference_would_rank_a_plot_layer_above_the_real_corridor():
    """Why the score is a PRODUCT, pinned with the measured numbers.

    On the reference drawing a residential plot layer measures (0.025, 0.000)
    and the real corridor measures (0.442, 0.742). A difference prefers the
    plot layer, because a difference rewards a layer for holding nothing.
    """
    plots = (0.025, 0.000)
    corridor_layer = (0.442, 0.742)
    assert plots[0] - plots[1] > corridor_layer[0] - corridor_layer[1]
    assert cr.corridor_score(*corridor_layer) > cr.corridor_score(*plots)


def test_an_unmeasurable_fraction_makes_the_score_absent_and_never_zero():
    """G8. A score of 0 is a finding; a score that could not be taken is a gap,
    and a chain that treats the second as the first inherits a conclusion."""
    assert cr.corridor_score(None, 0.0) is None
    assert cr.corridor_score(0.5, None) is None
    assert cr._fraction(0, 0) is None, "nothing sampled is not a fraction of zero"
    assert cr._fraction(0, 10) == 0.0, "nothing found out of ten IS zero"
    assert cr._fraction(3, 12) == 0.25


@pytest.mark.parametrize(
    "doc,expected_point,expected_field",
    [
        (
            {"polygon_centroid": [1.0, 2.0], "bbox_centre": [9.0, 9.0]},
            (1.0, 2.0),
            "polygon_centroid",
        ),
        ({"bbox_centre": [3.0, 4.0]}, (3.0, 4.0), "bbox_centre"),
        ({"anchor_point": [5.0, 6.0]}, (5.0, 6.0), "anchor_point"),
        ({"polygon_centroid": None, "bbox_centre": [7.0, 8.0]}, (7.0, 8.0), "bbox_centre"),
        ({}, None, None),
        ({"bbox_centre": [None, 1.0]}, None, None),
        ({"bbox_centre": [1.0]}, None, None),
    ],
)
def test_the_representative_point_prefers_the_true_centroid_and_says_which_it_used(
    doc, expected_point, expected_field
):
    point, field = cr._point_of(doc)
    assert point == expected_point
    assert field == expected_field


def test_the_sample_spans_the_population_rather_than_taking_the_first_rows():
    """The first N rows sit wherever the drafter started drawing, and five of
    them describe one corner of a site and call it the layer."""
    rows = list(range(100))
    sample, step = cr._spread(rows, 10)
    assert len(sample) == 10
    assert step == 10
    assert sample[0] == 0 and sample[-1] == 90, sample
    assert max(sample) - min(sample) > 80, "the sample must span, not cluster"


def test_a_population_smaller_than_the_sample_is_taken_whole_and_the_step_is_one():
    sample, step = cr._spread([1, 2, 3], 10)
    assert sample == [1, 2, 3] and step == 1


def test_the_sample_still_spans_when_it_is_most_of_the_population():
    """The defect a slice hides, and it hides it at the CAREFUL sample sizes.

    `rows[::len(rows) // size]` collapses to `rows[::1]` as soon as the sample
    is more than half the population, and then truncates to the first `size`
    rows. Measured on the reference drawing before this was fixed: asking for
    1,000 of 1,574 network points sampled two thirds of a single layer and read
    the winning fraction as 0.177 where an evenly spaced sample of the same
    size reads 0.445 -- a caller who asked for MORE evidence got a worse
    answer, silently.
    """
    rows = list(range(1574))
    sample, step = cr._spread(rows, 1000)
    assert len(sample) == 1000
    assert max(sample) > 1500, "a truncating sample would stop at 999"
    assert 1.5 < step < 1.6
    assert len(set(sample)) == 1000, "no row is sampled twice"


@pytest.mark.parametrize("size", [1, 7, 99, 787, 1000, 1573, 1574, 5000])
def test_the_sample_never_runs_off_the_end_of_the_population(size):
    rows = list(range(1574))
    sample, step = cr._spread(rows, size)
    assert len(sample) == min(size, len(rows))
    assert step >= 1
    assert set(sample) <= set(rows)


def test_the_sample_is_the_same_twice():
    rows = [{"handle": f"H{i}"} for i in range(57)]
    assert cr._spread(rows, 9) == cr._spread(rows, 9)


@pytest.mark.parametrize(
    "rank,expected",
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (21, "21st")],
)
def test_the_ordinal_reads_as_english(rank, expected):
    assert cr._ordinal(rank) == expected


# =============================================================================
# 2. The fixture drawing — every answer known by hand
# =============================================================================


def test_the_corridor_is_ranked_first_and_the_bigger_site_boundary_is_not(
    fixture_store,
):
    """The acceptance shape, on a drawing whose truth is known.

    `alpha-1` is five times the area of `zeta-7` and holds the whole road
    network. It is still not the answer, because it holds every plot too.
    """
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    assert out["ranked"] is True
    assert out["best"]["layer"] == CORRIDOR
    order = [row["layer"] for row in out["candidates"]]
    assert order[0] == CORRIDOR
    assert order.index(CORRIDOR) < order.index(BOUNDARY)


def test_the_corridor_holds_all_the_network_and_none_of_the_plots(fixture_store):
    rows = _rows(cr.rank_region_layers(FIXTURE, LAYOUT))
    corridor_row = rows[CORRIDOR]
    assert corridor_row["network_inside"]["fraction"] == pytest.approx(1.0)
    assert corridor_row["network_inside"]["points_inside"] == 10
    assert corridor_row["parcels_inside"]["fraction"] == pytest.approx(0.0)
    assert corridor_row["parcels_inside"]["points_sampled"] == 9
    assert corridor_row["corridor_score"] == pytest.approx(1.0)


def test_the_site_boundary_holds_both_and_therefore_scores_zero(fixture_store):
    rows = _rows(cr.rank_region_layers(FIXTURE, LAYOUT))
    site = rows[BOUNDARY]
    assert site["network_inside"]["fraction"] == pytest.approx(1.0)
    assert site["parcels_inside"]["fraction"] == pytest.approx(1.0)
    assert site["corridor_score"] == 0.0
    assert "site boundary" in site["why"]


def test_a_plot_layer_holds_neither_and_therefore_scores_zero(fixture_store):
    rows = _rows(cr.rank_region_layers(FIXTURE, LAYOUT))
    plots = rows[PLOTS_A]
    assert plots["network_inside"]["fraction"] == pytest.approx(0.0)
    assert plots["corridor_score"] == 0.0
    assert "holds none of the sampled road network" in plots["why"]


def test_a_layer_does_not_earn_its_score_from_containing_its_own_centroids(
    fixture_store,
):
    """A ring contains its own centroid. That is arithmetic, not evidence, so
    a candidate's own points are excluded from BOTH sides of its parcel
    fraction -- and the denominator it really used is published."""
    rows = _rows(cr.rank_region_layers(FIXTURE, LAYOUT))
    plots = rows[PLOTS_A]
    assert plots["parcels_inside"]["own_points_excluded"] == 4
    assert plots["parcels_inside"]["points_sampled"] == 7, (
        "eleven region points less its own four"
    )
    assert plots["parcels_inside"]["fraction"] == pytest.approx(0.0)


def test_every_candidate_publishes_both_fractions_the_sample_and_a_reason(
    fixture_store,
):
    """Publish the evidence, not just the verdict: a reader must be able to see
    WHY a layer was proposed and disagree with the conclusion while agreeing
    with the numbers."""
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    for row in out["candidates"]:
        for side in ("network_inside", "parcels_inside"):
            fraction = row[side]
            assert set(
                ("fraction", "points_inside", "points_sampled", "population", "basis")
            ) <= set(fraction)
            assert fraction["points_sampled"] >= 0
            assert fraction["basis"].strip()
        assert row["why"].strip()
        assert str(row["layer"]) in row["why"]
        assert row["rank"] >= 1


def test_the_answer_never_claims_the_layer_is_a_road(fixture_store):
    """Claim only what is measured. `behaves like` is the ceiling; `is a road`
    is meaning, and geometry does not carry meaning."""
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    text = json.dumps(out)
    assert "does not say the layer is a road" in out["best"]["why"]
    assert out["what_this_does_not_establish"].strip()
    for forbidden in ("is a road", "is the road", "IS a road corridor,"):
        assert forbidden not in text.replace("does not say the layer is a road", "")


def test_the_response_shows_what_a_ranking_by_size_would_have_done(fixture_store):
    """The change of rule is published so it can be CHECKED, not believed."""
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    old = out["ranking_by_size_would_have_proposed"]
    assert old["layer"] == BOUNDARY, "the largest layer by area"
    assert old["corridor_score"] == 0.0
    assert old["parcels_inside"] == pytest.approx(1.0)
    assert out["contains_every_sampled_parcel"]["count"] == 1
    assert out["contains_every_sampled_parcel"]["rows"][0]["layer"] == BOUNDARY


def test_area_travels_on_every_row_and_decides_nothing(fixture_store):
    """Area is published because a reader wants it. If it decided anything, the
    biggest layer would be first -- and it is not."""
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    areas = [row["area"] for row in out["candidates"]]
    assert all(a is not None for a in areas)
    assert areas != sorted(areas, reverse=True), (
        "the order is not the order area would have produced"
    )


def test_the_sample_size_and_the_step_are_published(fixture_store):
    """G7. A sampled fact that does not admit it was sampled is a confident
    half-truth."""
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    sample = out["sample"]
    assert sample["network_points_sampled"] == 10
    assert sample["network_population"] == 10
    assert sample["parcel_points_sampled"] == 11, "every closed ring in the layout"
    assert sample["network_sample_step"] >= 1
    assert sample["rings_tested"] == 11
    assert sample["point_in_ring_tests"] > 0
    assert out["caps"]["sample_points_per_population"] == cr.DEFAULT_SAMPLE_POINTS


def test_a_caller_asking_for_more_points_than_the_ceiling_gets_the_ceiling(
    fixture_store,
):
    out = cr.rank_region_layers(
        FIXTURE, LAYOUT, sample_size=cr.MAX_SAMPLE_POINTS * 100
    )
    assert out["caps"]["sample_points_per_population"] == cr.MAX_SAMPLE_POINTS


def test_the_layers_already_tested_are_not_offered_again(fixture_store):
    out = cr.rank_region_layers(FIXTURE, LAYOUT, exclude=[CORRIDOR])
    assert CORRIDOR not in [row["layer"] for row in out["candidates"]]


def test_an_excluded_layer_still_counts_as_parcel_evidence(fixture_store):
    """A corridor is recognised by NOT containing the parcels, so removing the
    parcels the caller already tested would remove the evidence."""
    out = cr.rank_region_layers(FIXTURE, LAYOUT, exclude=[PLOTS_A, PLOTS_B])
    rows = _rows(out)
    assert PLOTS_A not in rows and PLOTS_B not in rows
    assert rows[BOUNDARY]["parcels_inside"]["points_sampled"] == 10, (
        "the eight plots the caller already tested are still evidence"
    )
    assert rows[BOUNDARY]["parcels_inside"]["fraction"] == pytest.approx(1.0)
    assert out["best"]["layer"] == CORRIDOR


def test_the_answer_is_the_same_twice(fixture_store):
    first = json.dumps(cr.rank_region_layers(FIXTURE, LAYOUT), sort_keys=True, default=str)
    second = json.dumps(cr.rank_region_layers(FIXTURE, LAYOUT), sort_keys=True, default=str)
    assert first == second


def test_a_point_on_a_corridor_edge_counts_as_inside(fixture_store, monkeypatch):
    """A road centreline snapped to the edge of the corridor that carries it is
    normal draughting. Dropping those points would under-state exactly the
    layer this signal looks for."""
    docs = _fixture_docs()
    # A road point sitting exactly on the western corridor's left edge.
    docs.append(path_doc("R-EDGE", NETWORK, (20.0, 20.0)))
    entities = FakeCollection(docs)
    monkeypatch.setattr(cr, "_entities", lambda: entities)
    dossier = copy.deepcopy(FIXTURE_DOSSIER)
    rows = _rows(cr.rank_region_layers(FIXTURE, LAYOUT))
    assert rows[CORRIDOR]["network_inside"]["points_inside"] == 11
    assert rows[CORRIDOR]["network_inside"]["fraction"] == pytest.approx(1.0)
    assert dossier["layers"], "the fixture dossier is untouched by the call"


def test_the_score_carries_no_unit_and_none_is_invented(fixture_store):
    """G2. Both fractions are dimensionless, which is what makes the same rule
    behave identically in metres, in inches, and in a drawing that declares no
    unit at all."""
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    for row in out["candidates"]:
        assert "unit" not in row["network_inside"]
        assert "unit" not in row["parcels_inside"]
    assert "dimensionless" in out["score_definition"]


# =============================================================================
# 3. Degrading honestly — a stated reason, never a wrong ranking
# =============================================================================


def test_without_a_dossier_nothing_is_ranked_and_the_reason_is_not_computed(
    monkeypatch,
):
    monkeypatch.setattr(dossier_read, "dossier_for", lambda drawing_id: None)
    monkeypatch.setattr(cr.landuse, "for_drawing", lambda drawing_id: None)
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    assert out["ranked"] is False
    assert out["candidates"] == []
    assert "not computed" in out["why_not"]
    assert "there are no" in out["why_not"], "absence is not a finding of none"


def test_a_dossier_that_cannot_be_read_is_a_reason_and_not_an_outage(monkeypatch):
    def explode(drawing_id):
        raise RuntimeError("the cluster said no")

    monkeypatch.setattr(dossier_read, "dossier_for", explode)
    monkeypatch.setattr(cr.landuse, "for_drawing", lambda drawing_id: None)
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    assert out["ranked"] is False
    assert "RuntimeError" in out["why_not"]


def test_without_network_geometry_the_candidates_are_named_but_not_ranked(
    monkeypatch,
):
    """No roads is a correct outcome. What is refused is a confident order over
    a question nobody could ask."""
    entities = FakeCollection([d for d in _fixture_docs() if d["layer"] != NETWORK])
    monkeypatch.setattr(cr, "_entities", lambda: entities)
    dossier = {
        "drawing_id": FIXTURE,
        "layers": [b for b in FIXTURE_DOSSIER["layers"] if b["layer"] != NETWORK],
    }
    monkeypatch.setattr(dossier_read, "dossier_for", lambda drawing_id: copy.deepcopy(dossier))
    monkeypatch.setattr(cr.landuse, "for_drawing", lambda drawing_id: None)

    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    assert out["ranked"] is False
    assert "network" in out["why_not"]
    named = [row["layer"] for row in out["candidates"]]
    assert BOUNDARY in named, "a layer of closed rings is still a fact"
    assert all(row["corridor_score"] is None for row in out["candidates"])
    assert named == sorted(named), "unranked means layer-name order, not size order"


def test_no_degradation_ever_falls_back_to_ranking_by_size(monkeypatch):
    """The defect this module exists to remove, tested as a property of every
    withheld answer rather than of one of them."""
    entities = FakeCollection([d for d in _fixture_docs() if d["layer"] != NETWORK])
    monkeypatch.setattr(cr, "_entities", lambda: entities)
    dossier = {
        "drawing_id": FIXTURE,
        "layers": [b for b in FIXTURE_DOSSIER["layers"] if b["layer"] != NETWORK],
    }
    monkeypatch.setattr(dossier_read, "dossier_for", lambda drawing_id: copy.deepcopy(dossier))
    monkeypatch.setattr(cr.landuse, "for_drawing", lambda drawing_id: None)

    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    assert out["never_ranked_by_size"].strip()
    areas = [row["area"] for row in out["candidates"]]
    assert areas != sorted(areas, reverse=True) or len(areas) < 2


def test_with_only_one_closed_ring_layer_the_parcel_fraction_is_absent_not_zero(
    monkeypatch,
):
    """G8, in the shape it really arrives in: a drawing whose only region layer
    IS the candidate. Excluding its own points leaves nothing to measure
    against, and 0/0 is not 0."""
    docs = [
        ring_doc("COR-V", CORRIDOR, square(45, 0, 10, 100)),
        path_doc("R-1", NETWORK, (50.0, 10.0)),
    ]
    entities = FakeCollection(docs)
    monkeypatch.setattr(cr, "_entities", lambda: entities)
    dossier = {
        "drawing_id": FIXTURE,
        "layers": [
            block(CORRIDOR, "region", entities=1, area=1_000.0),
            block(NETWORK, "network", entities=1, area=100.0),
        ],
    }
    monkeypatch.setattr(dossier_read, "dossier_for", lambda drawing_id: copy.deepcopy(dossier))
    monkeypatch.setattr(cr.landuse, "for_drawing", lambda drawing_id: None)

    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    row = out["candidates"][0]
    assert row["network_inside"]["fraction"] == pytest.approx(1.0)
    assert row["parcels_inside"]["fraction"] is None
    assert row["corridor_score"] is None
    assert out["ranked"] is False
    assert "absence, not a zero" in out["no_corridor_signal"]


def test_a_population_above_the_stated_ceiling_is_withheld_rather_than_sampled_down(
    monkeypatch, fixture_store
):
    monkeypatch.setattr(cr, "MAX_CANDIDATE_RINGS", 3)
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    assert out["ranked"] is False
    assert "3" in out["why_not"] and "ceiling" in out["why_not"]
    assert out["candidates"], "the layers are still named"


def test_a_layer_with_no_closed_ring_is_a_reason_rather_than_an_empty_answer(
    monkeypatch,
):
    entities = FakeCollection([path_doc("R-1", NETWORK, (1.0, 1.0))])
    monkeypatch.setattr(cr, "_entities", lambda: entities)
    dossier = {
        "drawing_id": FIXTURE,
        "layers": [
            block(EMPTY, "region", entities=0, area=None),
            block(NETWORK, "network", entities=1, area=10.0),
        ],
    }
    monkeypatch.setattr(dossier_read, "dossier_for", lambda drawing_id: copy.deepcopy(dossier))
    monkeypatch.setattr(cr.landuse, "for_drawing", lambda drawing_id: None)

    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    assert out["ranked"] is False
    assert "closed cleanly" in out["why_not"]


def test_the_function_never_raises_whatever_the_store_does(monkeypatch, fixture_store):
    class Exploding:
        def find(self, *_a, **_k):
            raise RuntimeError("the cluster said no")

    monkeypatch.setattr(cr, "_entities", lambda: Exploding())
    out = cr.rank_region_layers(FIXTURE, LAYOUT)
    assert out["ranked"] is False
    assert "RuntimeError" in out["why_not"]


# =============================================================================
# 4. The rules, as properties of the file
# =============================================================================


def test_no_drawing_specific_constant_lives_in_this_module():
    """G1, and it is the rule this whole module turns on.

    The corridor layer on the reference drawing carries no road word, so a name
    search cannot find it; template layers named after roads are sheet layout,
    so a name search finds the wrong thing. Neither failure is available to a
    module that never reads a name.
    """
    source = pathlib.Path(cr.__file__).read_text(encoding="utf-8")
    for token in (
        "VL2",
        "VL3",
        "DP4",
        "LP1",
        "TH3",
        "Primary School",
        "BlockBoundary",
        "New Boundry",
        "C-PROP-PlotNumber",
        "00_Prop",
        r"\bROW\b",
        "32638",
    ):
        assert re.search(token, source) is None, token


def test_the_module_reads_no_layer_name_at_all():
    """Stronger than G1's grep: not one comparison in this file may be against
    a name, a token, or a prefix of one."""
    source = pathlib.Path(cr.__file__).read_text(encoding="utf-8")
    for forbidden in (
        ".startswith(",
        ".endswith(",
        "re.search",
        "re.match",
        "casefold() ==",
        ".lower() ==",
    ):
        assert forbidden not in source, forbidden


def test_the_module_reaches_neither_the_clock_nor_the_network():
    source = pathlib.Path(cr.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import time",
        "import random",
        "import requests",
        "import urllib",
        "import socket",
        "datetime",
        "random.",
    ):
        assert forbidden not in source, forbidden


def test_no_constant_in_this_module_is_written_as_a_length_or_an_area():
    """G2. Every number here is a count of points, a count of rings, or a
    fraction. A length would be right in metres and wrong in inches."""
    for count in (
        cr.DEFAULT_SAMPLE_POINTS,
        cr.MAX_SAMPLE_POINTS,
        cr.MAX_CANDIDATE_RINGS,
        cr.MAX_CANDIDATES_RETURNED,
    ):
        assert isinstance(count, int) and count > 0
    assert cr.DEFAULT_SAMPLE_POINTS <= cr.MAX_SAMPLE_POINTS
    assert not any(
        name.endswith(("_M", "_MM", "_METRES", "_INCHES")) for name in dir(cr)
    )


def test_this_module_registers_no_recipe():
    """It is a scoring helper. The catalogue is the integrator's business, and
    a helper that quietly registered itself would change a pinned count."""
    source = pathlib.Path(cr.__file__).read_text(encoding="utf-8")
    assert "register(" not in source
    assert "Recipe(" not in source


# =============================================================================
# 5. Janadriyah — the acceptance run, recorded either way
# =============================================================================


def _live_or_skip():
    from app import store as live_store

    try:
        drawing = live_store.get_drawing(DRAWING)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip(f"reference drawing {DRAWING} has not been ingested")
    return drawing


_LIVE_CACHE: dict[tuple, dict] = {}


def _live(limit=cr.MAX_CANDIDATES_RETURNED):
    _live_or_skip()
    key = (DRAWING, LAYOUT, limit)
    if key not in _LIVE_CACHE:
        _LIVE_CACHE[key] = cr.rank_region_layers(DRAWING, LAYOUT, limit=limit)
    return _LIVE_CACHE[key]


#: The right-of-way layer on the reference drawing, and the two layers a
#: ranking by AREA puts in front of it. All three live in the TEST -- G1 -- and
#: they are named here because the acceptance criterion is an ORDER between
#: them.
GT_CORRIDOR = "ROW"
GT_SITE_BOUNDARY = "0"
GT_OTHER_BOUNDARY = "New Boundry"

#: Measured 26 August 2026 at the default sample size. The corridor holds 106
#: of 240 sampled network points and 178 of 240 sampled parcel points; both
#: boundary layers hold 114 of 240 network points and 240 of 240 parcel points,
#: which is what sends them to zero. The runner-up scores 0.056.
GT_CORRIDOR_SCORE = 0.114097


def test_janadriyah_ranks_the_right_of_way_first():
    """The acceptance criterion for Phase 8a, on the drawing it was written for.

    `ROW` is second by area and its name carries no road word, so neither size
    nor name finds it. The geometry does.
    """
    out = _live()
    assert out["ranked"] is True
    assert out["best"]["layer"] == GT_CORRIDOR
    assert out["candidates"][0]["layer"] == GT_CORRIDOR
    assert out["candidates"][0]["corridor_score"] == pytest.approx(
        GT_CORRIDOR_SCORE, abs=1e-4
    )


def test_janadriyah_puts_the_right_of_way_ahead_of_both_boundary_layers():
    out = _live(limit=60)
    order = [row["layer"] for row in out["candidates"]]
    assert order.index(GT_CORRIDOR) < order.index(GT_SITE_BOUNDARY)
    assert order.index(GT_CORRIDOR) < order.index(GT_OTHER_BOUNDARY)


def test_janadriyah_publishes_the_two_fractions_that_put_it_there():
    out = _live()
    row = out["candidates"][0]
    network = row["network_inside"]
    parcels = row["parcels_inside"]
    assert 0.3 < network["fraction"] < 0.5
    assert 0.6 < parcels["fraction"] < 0.9
    assert network["points_sampled"] == cr.DEFAULT_SAMPLE_POINTS
    assert parcels["points_sampled"] > 0
    assert network["population"] > network["points_sampled"], "it is a SAMPLE"
    assert row["corridor_score"] == pytest.approx(
        network["fraction"] * (1 - parcels["fraction"]), abs=1e-6
    )


def test_janadriyah_shows_that_the_largest_layer_was_considered_and_rejected():
    """The site boundary is not merely absent from the top of the list: it is
    named, with the fraction that disqualified it."""
    out = _live()
    old = out["ranking_by_size_would_have_proposed"]
    assert old["layer"] == GT_SITE_BOUNDARY
    assert old["parcels_inside"] == pytest.approx(1.0)
    assert old["corridor_score"] == 0.0
    swallowed = {row["layer"] for row in out["contains_every_sampled_parcel"]["rows"]}
    assert {GT_SITE_BOUNDARY, GT_OTHER_BOUNDARY} <= swallowed


def test_janadriyah_says_why_a_third_of_the_network_is_inside_no_layer_at_all():
    """The measured reason the winning fraction is 0.44 and not 1.0: that road
    network is drawn three times and two of the copies sit off the site. A
    reader who cannot see this reads a nearly complete signal as a weak one."""
    out = _live()
    sample = out["sample"]
    assert sample["network_points_inside_no_candidate"] > 0
    assert "three times" in sample["outside_everything_note"]


def test_janadriyah_answers_the_same_twice():
    a = json.dumps(cr.rank_region_layers(DRAWING, LAYOUT), sort_keys=True, default=str)
    b = json.dumps(cr.rank_region_layers(DRAWING, LAYOUT), sort_keys=True, default=str)
    assert a == b


def test_invariant_the_signal_holds_over_every_drawing_in_the_store():
    """G5 -- run over EVERY drawing, not only the one the numbers came from.

    What is asserted is not a value: values differ per drawing, as they should.
    It is that every answer is either RANKED with a score every row can show,
    or WITHHELD with a reason a person can read -- and that nothing in between
    ever happens.
    """
    from app import store as live_store

    try:
        drawings = live_store.list_drawings()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawings:
        pytest.skip("the store holds no drawing")

    ranked = 0
    withheld = 0
    for drawing in drawings:
        drawing_id = str(drawing.get("_id") or drawing.get("drawing_id"))
        out = cr.rank_region_layers(drawing_id, LAYOUT)
        assert isinstance(out, dict), drawing_id
        assert out["never_ranked_by_size"].strip(), drawing_id
        if out["ranked"]:
            ranked += 1
            assert out["ranked_by"].startswith("corridor_score"), drawing_id
            for row in out["candidates"]:
                assert row["why"].strip(), (drawing_id, row["layer"])
                assert row["network_inside"]["points_sampled"] >= 0
                score = row["corridor_score"]
                if score is not None:
                    assert 0.0 <= score <= 1.0, (drawing_id, row["layer"], score)
        else:
            withheld += 1
            assert (out["why_not"] or "").strip(), drawing_id
            for row in out["candidates"]:
                assert row["corridor_score"] is None, drawing_id
    assert ranked + withheld == len(drawings)
    assert ranked >= 1, "the reference drawing at least must be rankable"


def test_a_drawing_with_no_road_corridor_produces_no_confident_corridor():
    """A correct outcome, not a failure. Across the store, any drawing whose
    candidates all score zero must SAY there is no corridor signal rather than
    proposing its largest layer."""
    from app import store as live_store

    try:
        drawings = live_store.list_drawings()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")

    seen = 0
    for drawing in drawings:
        drawing_id = str(drawing.get("_id") or drawing.get("drawing_id"))
        out = cr.rank_region_layers(drawing_id, LAYOUT)
        if out.get("best") is None:
            seen += 1
            said = out.get("no_corridor_signal") or out.get("why_not") or ""
            assert said.strip(), drawing_id
    assert seen >= 1, "at least one drawing in this store carries no corridor"


def test_the_reference_drawing_is_scored_inside_a_request_budget():
    """Cost matters: this runs inside a request. Measured 26 August 2026 on the
    largest drawing in the store -- 2,881 rings, 2,760 point-in-ring tests --
    at about 2.5 s warm, of which the Dossier read is roughly half.

    The assertion is deliberately loose: it is a guard against an accidental
    quadratic, not a latency target on someone else's laptop.
    """
    import time

    _live_or_skip()
    started = time.perf_counter()
    out = cr.rank_region_layers(DRAWING, LAYOUT)
    elapsed = time.perf_counter() - started
    assert out["ranked"] is True
    assert elapsed < 30.0, f"{elapsed:.1f}s"
    assert out["sample"]["point_in_ring_tests"] < 100_000, out["sample"]


def test_the_module_is_importable_from_topology_without_a_cycle():
    """`topology` imports `corridor`; `corridor` must never import `topology`
    back, or the recipe registry stops loading and every recipe disappears."""
    source = pathlib.Path(cr.__file__).read_text(encoding="utf-8")
    assert "import topology" not in source
    assert "from .topology" not in source
    assert math.isclose(cr.corridor_score(0.5, 0.5), 0.25)
