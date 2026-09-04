"""DOSSIER Phase 6 — a checker that has never been shown a lie is untested.

Owner: subagent TRUTH. Tests `app/recipes/truth.py` and nothing else.

The order of this file is the order the work was done in, and that order is the
point. A recipe that compares a printed number with a measured one is trivially
easy to write in a way that never disagrees with anything: associate loosely,
compare against whichever measure happens to match, widen the band until
everything passes. Every one of those mistakes produces a green suite and a
useless check. So the first fixture built here is a drawing that **lies** — a
label reading `25.00` sitting over a line that measures `24.00` — and the first
test asserts that it is caught, with both numbers and both handles.

Three fixture labels carry the whole design:

* `T-LIE` — `25.00` over a 24.00 line. Must be reported, with the difference.
* `T-TRUE` — `12.40` over a 12.4 line. Must NOT be reported: a check that
  cries wolf is discarded by its reader within a week.
* `T-AMBIG` — `8.00` sitting between a line of 8.0 and a line of 6.0, both
  inside its search radius. Must be reported as AMBIGUOUS and given no
  verdict. Picking the nearer one would produce a row indistinguishable from a
  checked one, and that is the failure this whole campaign exists to end.

Two kinds of test, both mandatory (G5):

* **Number tests** over the fixture, where the truth is known because it was
  planted, plus the ground-truth figures measured from the store on 25 August
  2026 for `drafting_hygiene`.
* **Invariant tests** run over EVERY drawing in the store, in metre, inch and
  unit-less shapes, because the fixture numbers will never catch a break on the
  nineteenth drawing.

Everything except the clearly marked live section runs **without MongoDB**:
`check_labels`, `hygiene_report`, `hidden_layers` and `text_scale_check` are
pure over Mongo-shaped documents, which is the same split `store_tables.py`
makes and for the same reason — this suite runs in a throwaway container.

NOTE for the integrator: importing this module registers `dimension_truth` and
`drafting_hygiene` into the shared recipe registry. `tests/test_recipes.py`
pins the catalogue to its eight names, so that pin has to grow when the mount
lands in `app/recipes/__init__.py`. This file does not touch either.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import dossier_census as census  # noqa: E402
from app import evidence as ev  # noqa: E402
from app.recipes import registry as rr  # noqa: E402
from app.recipes import truth  # noqa: E402


# --- unit shapes -------------------------------------------------------------
#
# Copied in the TEST, never in the code: 11 of the 18 drawings in this store are
# in inches and 3 declare no unit at all, so every response is run through all
# of these shapes rather than through the one that happens to be convenient.

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
    "why_no_unit": "this drawing declares no units.",
}

SHEET_UNITS = {
    "name": "sheet coordinates",
    "declared_in_file": False,
    "length_unit": None,
    "area_unit": None,
    "space": "paper",
    "why_no_unit": "DXF carries no per-layout units.",
}

ALL_UNIT_SHAPES = [
    pytest.param(METRE_UNITS, id="metre"),
    pytest.param(INCH_UNITS, id="inch"),
    pytest.param(UNITLESS, id="undeclared"),
    pytest.param(SHEET_UNITS, id="sheet"),
]

DRAWING = "fixture-drawing"
LAYOUT = "Model"


# =============================================================================
# The fixture drawing, and the lie planted in it
# =============================================================================


def _line(handle: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    """A LINE in the shape `autocad_entities` really stores.

    Axis-aligned on purpose, and the reason is a real limit of the store rather
    than convenience: endpoint coordinates are NOT stored for path entities, so
    `truth._distance_to` falls back to the bounding box. For an axis-aligned
    line the box IS the line and the distance is exact; for a diagonal one it
    is a lower bound, which the response says on every row. Testing the exact
    case here keeps the planted numbers honest; the lower-bound wording is
    asserted separately.
    """
    return {
        "drawing_id": DRAWING,
        "handle": handle,
        "type": "LINE",
        "layer": "Geometry",
        "layout": LAYOUT,
        "length": math.dist((x0, y0), (x1, y1)),
        "area": None,
        "bbox": {"min": [min(x0, x1), min(y0, y1)], "max": [max(x0, x1), max(y0, y1)]},
        "bbox_centre": [(x0 + x1) / 2, (y0 + y1) / 2],
    }


def _label(handle: str, text: str, x: float | None, y: float | None) -> dict:
    """A TEXT in the shape `autocad_entities` really stores.

    The bounding box is 1.6 wide by 1.2 tall, so the shorter side — the
    character height, which is the search radius when the caller names none —
    is exactly 1.2. Deliberately NOT square, so that a test using the wrong
    side of the box would show up as a different radius rather than as the
    same one by accident.
    """
    row: dict = {
        "drawing_id": DRAWING,
        "handle": handle,
        "type": "TEXT",
        "layer": "Annotation",
        "layout": LAYOUT,
        "text": text,
    }
    if x is None or y is None:
        return row
    row["anchor_point"] = [x, y]
    row["anchor_basis"] = "TEXT.insert"
    row["bbox"] = {"min": [x - 0.8, y - 0.6], "max": [x + 0.8, y + 0.6]}
    row["bbox_centre"] = [x, y]
    return row


#: The planted radius: every label's own character height, 1.2 drawing units.
FIXTURE_RADIUS = 1.2

RING = [[100.0, 0.0], [112.0, 0.0], [112.0, 25.0], [100.0, 25.0]]

PLOT = {
    "drawing_id": DRAWING,
    "handle": "R-PLOT",
    "type": "LWPOLYLINE",
    "layer": "Plots",
    "layout": LAYOUT,
    "length": 74.0,
    "area": 300.0,
    "bbox": {"min": [100.0, 0.0], "max": [112.0, 25.0]},
    "bbox_centre": [106.0, 12.5],
    "ring_status": "complete",
    "ring": RING,
    "ring_origin": [100.0, 0.0],
    "edge_lengths": [12.0, 25.0, 12.0, 25.0],
}

GEOMETRY = [
    _line("L-LIE", 0, 0, 24, 0),        # the line the lying label sits over
    _line("L-TRUE", 0, 10, 12.4, 10),   # the line the truthful label sits over
    _line("L-AMB1", 0, 20, 8, 20),      # two candidates that do NOT agree
    _line("L-AMB2", 0, 22, 6, 22),
    _line("L-TWIN1", 0, 30, 8, 30),     # two candidates that DO agree
    _line("L-TWIN2", 0, 32, 8, 32),
    PLOT,
]

LABELS = [
    _label("T-LIE", "25.00", 12, 0.6),
    _label("T-TRUE", "12.40", 6, 10.6),
    _label("T-AMBIG", "8.00", 3, 21),
    _label("T-TWIN", "8.00", 3, 31),
    _label("T-EDGE", "25.00", 112.6, 12),      # beside the plot's right edge
    _label("T-INSIDE", "25.00", 111, 12.5),    # inside the plot, near that edge
    _label("T-MIDDLE", "25.00", 106, 12.5),    # inside it, in reach of nothing
    _label("T-PAIR", "12.00 x 25.00", 12, 2),  # a size, not one length
    _label("T-NOPOINT", "5.00", None, None),   # nowhere to put it
    _label("T-FAR", "9.99", 500, 500),         # nothing within reach
    _label("T-METRE", "1.5 m", 500, 600),      # states its own unit
    _label("T-PLOTNO", "2092", 12, 4),         # an integer: NOT a dimension
]

#: What was planted, counted by hand. If the code changes these, the code is
#: answering a different question.
GT_LABELS = 11           # `dimension_value` rows; `2092` is a numeric_label
GT_CHECKED = 5           # T-LIE, T-TRUE, T-TWIN, T-EDGE, T-INSIDE
GT_AGREE = 4
GT_DISAGREE = 1          # T-LIE, and only T-LIE
GT_AMBIGUOUS = 1         # T-AMBIG
GT_NOT_ASSOCIATED = 4    # T-NOPOINT, T-FAR, T-METRE (metre drawing), T-MIDDLE
GT_SKIPPED = 1           # T-PAIR
GT_LIE_PRINTED = 25.0
GT_LIE_MEASURED = 24.0


def run(units=METRE_UNITS, **kw):
    return truth.check_labels(
        LABELS, GEOMETRY, drawing_id=DRAWING, layout=LAYOUT, units=units, **kw
    )


def row_for(result, handle):
    return next((r for r in result["rows"] if r["text_handle"] == handle), None)


# =============================================================================
# 1. The lie
# =============================================================================


def test_a_label_that_lies_is_caught_with_both_numbers_and_both_handles():
    """The whole point, and the first thing written.

    `25.00` printed over a line that measures `24.00`. What a reader needs is
    not "1 disagreement": it is both numbers, the difference, and the two
    handles — so the label and the geometry are each one click away.
    """
    out = run(show_all=True)
    lie = row_for(out, "T-LIE")

    assert lie is not None, "the lying label produced no row at all"
    assert lie["agrees"] is False
    assert lie["printed"] == GT_LIE_PRINTED
    assert lie["measured"] == pytest.approx(GT_LIE_MEASURED)
    assert lie["difference"] == pytest.approx(-1.0)
    assert lie["difference_percent"] == pytest.approx(-4.0)
    assert lie["text_handle"] == "T-LIE"
    assert lie["geometry_handle"] == "L-LIE"
    assert lie["geometry_layer"] == "Geometry"
    assert out["labels_disagree"] == GT_DISAGREE


def test_a_row_carries_the_ratio_that_tells_a_defect_from_a_misread_label():
    """Published without a cut-off attached, and that is the point.

    On the reference drawing's sheet this recipe reports `778855.94` against a
    table rule measuring `135.97` — a UTM northing, not a dimension. The ratio
    says so at a glance; naming a threshold would turn that reading into a rule
    and start hiding real defects behind it.
    """
    out = run(show_all=True)
    assert row_for(out, "T-LIE")["printed_over_measured"] == pytest.approx(25 / 24)
    assert "orders of magnitude apart" in out["rows_ordering"]


def test_the_lie_is_the_only_row_listed_by_default():
    """A report that lists everything is a report nobody reads to the end."""
    out = run()
    assert out["rows_showing"] == "disagreements only"
    assert [r["text_handle"] for r in out["rows"]] == ["T-LIE"]


def test_a_truthful_label_is_not_flagged():
    """A check that cries wolf is discarded by its reader within a week."""
    out = run(show_all=True)
    true_row = row_for(out, "T-TRUE")
    assert true_row is not None
    assert true_row["agrees"] is True
    assert true_row["difference"] == pytest.approx(0.0, abs=1e-9)
    assert out["labels_agree"] == GT_AGREE


# =============================================================================
# 2. Ambiguity is reported, never resolved
# =============================================================================


def test_an_ambiguous_label_is_reported_as_ambiguous_and_given_no_verdict():
    """Two candidates that measure different things leave no answer.

    `8.00` sits 1.0 from a line measuring 8.0 and 1.0 from a line measuring
    6.0. Picking the nearer one is not available here — they are equidistant —
    but even if one were nearer, choosing would produce a row that looks
    exactly like a checked one.
    """
    out = run(show_all=True)
    assert row_for(out, "T-AMBIG") is None, "an ambiguous label was given a verdict"

    entry = next(a for a in out["ambiguous"] if a["text_handle"] == "T-AMBIG")
    handles = {c["handle"] for c in entry["candidates"]}
    assert handles == {"L-AMB1", "L-AMB2"}
    assert {c["measured"] for c in entry["candidates"]} == {8.0, 6.0}
    assert entry["candidates_total"] == 2
    assert entry["measured_spread"] == pytest.approx(2.0)
    assert "agrees" not in entry and "difference" not in entry
    assert out["labels_ambiguous"] == GT_AMBIGUOUS


def test_ambiguity_that_could_not_change_the_answer_is_resolved_and_says_why():
    """Two candidates measuring the SAME thing leave exactly one answer.

    Refusing here would be pedantry dressed as rigour: the verdict does not
    depend on the choice, and the row says so in `resolved_because` rather than
    hiding that there were two.
    """
    out = run(show_all=True)
    twin = row_for(out, "T-TWIN")
    assert twin is not None, "a label with two agreeing candidates was refused"
    assert twin["candidates_total"] == 2
    assert twin["agrees"] is True
    assert "the same value within the band" in twin["resolved_because"]


# =============================================================================
# 3. Association: containment and nearest-within-radius, both named
# =============================================================================


def test_a_label_inside_a_polygon_is_associated_by_containment():
    out = run(show_all=True)
    inside = row_for(out, "T-INSIDE")
    assert inside is not None
    assert inside["association_route"] == "containment"
    assert inside["geometry_handle"] == "R-PLOT"


def test_a_label_beside_a_polygon_is_associated_within_its_own_derived_radius():
    out = run(show_all=True)
    edge = row_for(out, "T-EDGE")
    assert edge is not None
    assert edge["association_route"] == "within_radius"
    assert edge["search_radius"] == pytest.approx(FIXTURE_RADIUS)
    assert "SHORTER side" in edge["search_radius_basis"]
    assert edge["distance"]["value"] == pytest.approx(0.6)


def test_the_derived_radius_does_not_grow_with_the_length_of_the_number():
    """The correction the corpus forced, pinned so it cannot come back.

    Two labels of the same character height, one eight characters long and one
    four, must search the same distance. Deriving the radius from the box
    DIAGONAL made the longer one search roughly twice as far, and it was
    reported ambiguous for a reason that had nothing to do with the drawing.
    """
    short = {"bbox": {"min": [0.0, 0.0], "max": [2.0, 1.2]}}
    long = {"bbox": {"min": [0.0, 0.0], "max": [16.0, 1.2]}}
    assert truth._label_radius(short)[0] == pytest.approx(1.2)
    assert truth._label_radius(long)[0] == pytest.approx(1.2)


def test_a_label_whose_box_is_flat_in_one_axis_yields_no_radius_rather_than_zero():
    flat = {"bbox": {"min": [0.0, 5.0], "max": [10.0, 5.0]}}
    radius, why = truth._label_radius(flat)
    assert radius is None
    assert "flat in one axis" in why


def test_a_polygon_that_merely_contains_a_label_is_not_the_thing_it_measures():
    """The correction the corpus forced, and the second one it forced.

    On the reference drawing's sheet the title border contained every label,
    so every one of the 21 labels there came back ambiguous against border
    lines 140 units away. Containment alone is not association: reach decides,
    and a polygon with no edge within a character height of the label is
    reported under `contains_but_out_of_reach` — listed, so a person can still
    open it, and given no measure.
    """
    out = run(show_all=True)
    assert row_for(out, "T-MIDDLE") is None
    entry = next(a for a in out["not_associated"] if a["text_handle"] == "T-MIDDLE")
    assert entry["contains_but_out_of_reach_total"] == 1
    missed = entry["contains_but_out_of_reach"][0]
    assert missed["handle"] == "R-PLOT"
    assert missed["contains_the_label"] is True
    assert missed["nearest_edge_distance"] == pytest.approx(6.0)
    assert "no edge of this polygon is within" in missed["why"]


def test_a_polygon_offers_one_candidate_per_distinct_edge_length_in_reach():
    """A label in the corner of a plot names one of two edges, and which one is
    not knowable — so both are offered and the label comes back ambiguous."""
    corner = _label("T-CORNER", "25.00", 111.5, 0.5)
    out = truth.check_labels(
        [corner], [PLOT], drawing_id=DRAWING, layout=LAYOUT, units=METRE_UNITS
    )
    entry = out["ambiguous"][0]
    assert {c["measured"] for c in entry["candidates"]} == {12.0, 25.0}
    assert all(c["handle"] == "R-PLOT" for c in entry["candidates"])
    assert out["labels_checked"] == 0


def test_a_polygon_offers_its_nearest_edge_and_not_its_perimeter():
    """ONE measure per candidate, chosen by a stated rule.

    This is the decision that keeps the check able to fail. The plot's
    perimeter is 74 and its area 300; a check that offered every measure and
    accepted whichever matched could never find a lie, because something always
    matches.
    """
    out = run(show_all=True)
    for handle in ("T-EDGE", "T-INSIDE"):
        row = row_for(out, handle)
        assert row["measured"] == pytest.approx(25.0), handle
        assert "nearest edge" in row["measure_basis"], handle
        assert row["measured"] != PLOT["length"]


def test_the_distance_to_a_path_entity_admits_it_is_a_lower_bound():
    """Endpoint coordinates are not stored, so a box stands in — and says so."""
    out = run(show_all=True)
    assert "lower bound" in row_for(out, "T-LIE")["distance"]["method"]
    assert "exact" in row_for(out, "T-EDGE")["distance"]["method"]


def test_a_caller_radius_overrides_the_derived_one_and_is_reported():
    out = run(show_all=True, search_radius=0.1)
    # At 0.1 nothing is reachable any more: every label that was checked by
    # proximity falls out, and it falls out as NOT ASSOCIATED, never as agreeing.
    assert row_for(out, "T-LIE") is None
    entry = next(a for a in out["not_associated"] if a["text_handle"] == "T-LIE")
    assert entry["search_radius"] == pytest.approx(0.1)
    assert "caller" in entry["search_radius_basis"]


# =============================================================================
# 4. Coverage: nothing unmeasurable is counted as zero (G8)
# =============================================================================


def test_every_label_lands_in_exactly_one_bucket():
    out = run()
    arithmetic = out["coverage_arithmetic"]
    assert arithmetic["holds"] is True, arithmetic
    assert arithmetic["verdict_holds"] is True
    assert out["labels_total"] == GT_LABELS
    assert out["labels_checked"] == GT_CHECKED
    assert out["labels_ambiguous"] == GT_AMBIGUOUS
    assert out["labels_not_associated"] == GT_NOT_ASSOCIATED
    assert out["labels_skipped"] == GT_SKIPPED


def test_a_label_that_could_not_be_placed_is_counted_and_named():
    out = run()
    assert out["labels_without_point"] == 1
    entry = next(a for a in out["not_associated"] if a["text_handle"] == "T-NOPOINT")
    assert "no insertion point" in entry["reason"]


def test_a_label_with_nothing_in_reach_is_counted_not_treated_as_agreeing():
    out = run()
    entry = next(a for a in out["not_associated"] if a["text_handle"] == "T-FAR")
    assert "no measurable geometry lies within" in entry["reason"]
    assert entry["contains_but_out_of_reach_total"] == 0
    assert out["coverage_fraction"] == pytest.approx(GT_CHECKED / GT_LABELS)
    assert "statement about the CHECK" in out["coverage_note"]


def test_a_pair_label_states_a_size_and_is_not_judged():
    out = run()
    entry = next(s for s in out["skipped"] if s["text_handle"] == "T-PAIR")
    assert "SIZE" in entry["reason"]
    assert entry.get("measured") is None


def test_a_drawing_with_no_dimension_label_says_so_rather_than_passing():
    """Zero findings on zero input is not a clean bill of health."""
    out = truth.check_labels(
        [_label("T-PLOTNO", "2092", 1, 1)],
        GEOMETRY,
        drawing_id=DRAWING,
        layout=LAYOUT,
        units=METRE_UNITS,
    )
    assert out["labels_total"] == 0
    assert out["coverage_fraction"] is None
    assert "measured absence, not a clean bill" in out["coverage_note"]


# =============================================================================
# 5. The tolerance is derived, stated, and in the drawing's own units
# =============================================================================


def test_the_tolerance_comes_from_the_decimals_the_label_prints():
    two_dp = truth.tolerance_for(truth.printed_value("25.00"), 25.0, None)
    one_dp = truth.tolerance_for(truth.printed_value("25.0"), 25.0, None)
    integer = truth.tolerance_for(truth.printed_value("25"), 25.0, None)

    assert two_dp["from_printed_decimals"] == pytest.approx(0.005)
    assert one_dp["from_printed_decimals"] == pytest.approx(0.05)
    assert integer["from_printed_decimals"] == pytest.approx(0.5)
    assert two_dp["value"] < one_dp["value"] < integer["value"]


def test_a_label_that_states_its_own_tolerance_is_read_as_its_author_wrote_it():
    band = truth.tolerance_for(truth.printed_value("25.00 ±0.20"), 25.0, None)
    assert band["from_stated_tolerance"] == pytest.approx(0.20)
    assert band["value"] == pytest.approx(0.20)


def test_the_band_never_closes_below_floating_point_noise():
    band = truth.tolerance_for(truth.printed_value("83962.565"), 83962.565, None)
    assert band["from_arithmetic_noise"] == pytest.approx(
        truth.ARITHMETIC_NOISE_RELATIVE * 83962.565
    )
    assert band["value"] >= band["from_arithmetic_noise"]


def test_the_band_is_published_on_every_row():
    out = run(show_all=True)
    for row in out["rows"]:
        assert row["tolerance"]["value"] is not None
        assert "DRAWING units" in row["tolerance"]["basis"]


def test_a_caller_tolerance_is_used_and_named_as_the_callers():
    out = run(show_all=True, tolerance=2.0)
    lie = row_for(out, "T-LIE")
    assert lie["agrees"] is True, "a 2.0 band should absorb a 1.0 difference"
    assert lie["tolerance"]["basis"].startswith("handed over by the caller")


# =============================================================================
# 6. Units: every number carries one, and it may be absent (G2)
# =============================================================================


@pytest.mark.parametrize("units", ALL_UNIT_SHAPES)
def test_every_measured_length_carries_its_unit_or_says_why_not(units):
    out = run(units=units, show_all=True)
    for row in out["rows"]:
        for key in ("measured_quantity", "distance"):
            quantity = row[key]
            if quantity["value"] is None:
                assert quantity["withheld_reason"]
                continue
            assert quantity["unit"] == units["length_unit"]
            if quantity["unit"] is None:
                assert quantity["unit_reason"], (key, row["text_handle"])


def test_a_label_stating_a_unit_the_drawing_does_not_use_is_never_converted():
    """A wrong conversion produces confident agreement, which is the worst answer."""
    metric = run(units=METRE_UNITS)
    imperial = run(units=INCH_UNITS)

    assert metric["labels_with_unit_mismatch"] == 0
    assert imperial["labels_with_unit_mismatch"] == 1
    entry = next(s for s in imperial["skipped"] if s["text_handle"] == "T-METRE")
    assert entry["printed_unit"] == "m"
    assert entry["drawing_unit"] == "in"
    assert "No conversion is applied" in entry["reason"]


def test_a_unitless_drawing_never_has_a_unit_invented_for_it():
    out = run(units=UNITLESS, show_all=True)
    blob = json.dumps(out)
    assert '"unit": "m"' not in blob
    assert "DRAWING units, none declared" in out["scope_note"]


# =============================================================================
# 7. The census is the authority on the population, and it is asked
# =============================================================================


def test_the_population_is_the_census_class_and_the_two_are_reconciled():
    out = run()
    assert out["census"]["source"] == "dossier_census.annotation_census"
    assert out["census"]["dimension_value_count"] == out["labels_total"]
    assert out["census"]["agrees_with_this_recipe"] is True


def test_an_integer_label_is_not_a_dimension_and_the_census_decides_that():
    """A plot number is not a measurement, and the rule lives in one place."""
    assert census.classify_text("2092")[0] == census.KIND_NUMERIC
    assert census.classify_text("25.00")[0] == census.KIND_DIMENSION
    chosen = {r["handle"] for r in truth.dimension_labels(LABELS)}
    assert "T-PLOTNO" not in chosen
    assert "T-LIE" in chosen


def test_the_value_reader_finds_a_number_exactly_when_the_census_finds_a_dimension():
    """The two must not drift: a classified label whose value cannot be read
    would open a coverage hole with no visible cause."""
    for text in ("25.00", "12.4", "1,234.50", "Ø25.4mm", "25.00 ±0.20", "8"):
        if census.classify_text(text)[0] != census.KIND_DIMENSION:
            continue
        assert truth.printed_value(text)["value"] is not None, text


# =============================================================================
# 8. The response contract
# =============================================================================


def test_the_response_passes_the_evidence_contract_audit():
    assert ev.audit_response(run()) == []


def test_the_verdict_can_never_be_stated_by_the_file():
    """A disagreement is computed, so its ceiling is `inferred`, and that is right."""
    out = run()
    assert out["evidence"]["grade"] == ev.Grade.INFERRED.value
    assert out["evidence"]["not_established"]
    assert out["evidence"]["how_to_verify"]


def test_the_known_hole_is_stated_before_anyone_finds_it():
    """A decimal number in a legend is a dimension label by content alone."""
    out = run()
    assert "legend" in out["content_is_not_intent"]
    assert "WHICH of the two is wrong" in out["which_one_is_wrong"]
    assert "`<>`" in out["evidence"]["not_established"]


def test_it_is_deterministic():
    assert json.dumps(run(show_all=True), sort_keys=True, default=str) == json.dumps(
        run(show_all=True), sort_keys=True, default=str
    )


def test_the_size_limits_are_stated_and_they_bind():
    out = run()
    assert out["limits_applied"]["labels"] == truth.MAX_LABELS
    assert out["limits_applied"]["pairs_after_prefilter"] == truth.PAIRS_BUDGET

    too_many = [_label(f"T{i}", "1.00", i * 1000.0, 0.0) for i in range(truth.MAX_LABELS + 1)]
    with pytest.raises(rr.RecipeRefused) as caught:
        truth.check_labels(
            too_many, [], drawing_id=DRAWING, layout=LAYOUT, units=METRE_UNITS
        )
    assert caught.value.code == "RECIPE_INPUT_TOO_LARGE"
    assert caught.value.hint, "a refusal without a suggestion cannot be acted on"


# =============================================================================
# 9. drafting_hygiene — pure part
# =============================================================================


HYGIENE_DRAWING = {
    "_id": DRAWING,
    "layers": [
        {"name": "0", "frozen": False, "off": False, "entity_count": 9, "plot": True},
        {"name": "Frozen Layer", "frozen": True, "off": False, "entity_count": 4, "plot": True},
        {"name": "Off Layer", "frozen": False, "off": True, "entity_count": 7, "plot": True},
        {"name": "Visible", "frozen": False, "off": False, "entity_count": 20, "plot": True},
    ],
    "layouts": [
        {
            "name": LAYOUT,
            "entity_count": 40,
            "page_setup": {
                "available": True,
                "why": None,
                "paper_size_name": "ISO_A1",
                "paper_units": "mm",
                "plot_scale": {"numerator": 1.0, "denominator": 100.0},
                "plot_scale_text": "1:100",
            },
        }
    ],
    "header": {"$DIMSCALE": 100.0, "$LUPREC": 2},
    "dim_styles": [{"name": "ISO-25", "scale": 100.0}],
}

DEFAULT_LAYER_FACTS = {
    "layer": "0",
    "count": 3,
    "by_type": {"LINE": 2, "TEXT": 1},
    "handles": ["A1", "A2", "A3"],
    "handles_omitted": 0,
}


def hygiene(drawing=HYGIENE_DRAWING, units=METRE_UNITS, **overrides):
    rows, not_known = truth.hidden_layers(drawing)
    counts = overrides.pop("counts", {"Off Layer": 2, "Frozen Layer": 0})
    hidden = [
        {
            **row,
            "count": counts.get(str(row["layer"]), 0),
            "by_type": {"LWPOLYLINE": counts.get(str(row["layer"]), 0)},
            "handles": [f"H{i}" for i in range(counts.get(str(row["layer"]), 0))],
            "handles_omitted": 0,
        }
        for row in rows
    ]
    return truth.hygiene_report(
        drawing_id=DRAWING,
        drawing=drawing,
        layout=LAYOUT,
        units=units,
        default_layer=overrides.pop("default_layer", DEFAULT_LAYER_FACTS),
        hidden=hidden,
        hidden_not_known=not_known,
        **overrides,
    )


def finding(result, name):
    return next(f for f in result["findings"] if f["finding"] == name)


def test_content_on_the_default_layer_is_found_named_and_explained():
    out = hygiene()
    row = finding(out, "entities_on_default_layer")
    assert row["count"] == 3
    assert row["layer"] == "0"
    assert row["handles"] == ["A1", "A2", "A3"]
    assert row["clean"] is False
    assert "cannot be frozen" in row["why_it_matters"]
    assert out["entities_on_default_layer"] == 3


def test_content_hidden_on_a_frozen_or_off_layer_is_found():
    out = hygiene()
    row = finding(out, "content_on_hidden_layers")
    assert row["count"] == 2
    assert [layer["layer"] for layer in row["layers"]] == ["Off Layer"]
    assert row["layers"][0]["states"] == ["off"]
    assert row["layers_hidden_but_empty_here"] == ["Frozen Layer"]
    assert "without being looked at" in row["why_it_matters"]


def test_a_layer_frozen_only_inside_a_viewport_still_counts_as_hidden():
    """Hidden on the sheet that is approved is hidden, whatever the reason.

    The reference drawing carries exactly one of these — a hatch layer frozen
    in two viewports — and it holds twelve entities that would otherwise have
    read as visible.
    """
    drawing = {
        **HYGIENE_DRAWING,
        "layers": [
            {"name": "0", "frozen": False, "off": False, "entity_count": 1},
            {
                "name": "In Viewport",
                "frozen": False,
                "off": False,
                "entity_count": 12,
                "frozen_in_viewports": ["1F2", "1F3"],
            },
        ],
    }
    rows, why = truth.hidden_layers(drawing)
    assert why is None
    assert [r["layer"] for r in rows] == ["In Viewport"]
    assert rows[0]["frozen_in_viewports"] == 2
    assert rows[0]["states"] == ["frozen in 2 viewport(s)"]


def test_a_hidden_layer_reports_its_own_layout_count_beside_the_whole_drawings():
    """Two numbers that mean different things are never mixed into one."""
    out = hygiene()
    layer = finding(out, "content_on_hidden_layers")["layers"][0]
    assert layer["entities_in_this_layout"] == 2
    assert layer["entities_in_drawing"] == 7


def test_a_clean_drawing_reads_as_clean_rather_than_as_unchecked():
    out = hygiene(
        default_layer={"layer": "0", "count": 0, "by_type": {}, "handles": [], "handles_omitted": 0},
        counts={"Off Layer": 0, "Frozen Layer": 0},
    )
    assert finding(out, "entities_on_default_layer")["clean"] is True
    assert finding(out, "content_on_hidden_layers")["clean"] is True
    assert out["findings_raised"] == 0


def test_a_drawing_with_no_layer_table_says_not_known_and_never_zero():
    """"Not extracted yet" and "nothing is hidden" are different answers (G8)."""
    older = {k: v for k, v in HYGIENE_DRAWING.items() if k != "layers"}
    rows, why = truth.hidden_layers(older)
    assert rows == []
    assert "NOT KNOWN" in why

    out = truth.hygiene_report(
        drawing_id=DRAWING,
        drawing=older,
        layout=LAYOUT,
        units=METRE_UNITS,
        default_layer=DEFAULT_LAYER_FACTS,
        hidden=[],
        hidden_not_known=why,
    )
    assert out["layer_table_available"] is False
    assert out["entities_on_hidden_layers"] is None
    assert finding(out, "content_on_hidden_layers")["clean"] is None


def test_text_height_is_reported_as_unmeasured_and_never_as_a_pass():
    """The check that cannot be run says so, and says what would fix it."""
    out = hygiene()
    row = finding(out, "text_height_vs_sheet_scale")
    assert row["clean"] is None
    assert row["count"] is None
    detail = row["detail"]
    assert detail["derivable"] is False
    assert "not stored" in detail["why_not"]
    assert "char_height" in detail["what_would_make_it_derivable"]
    assert "NOT a clean result" in detail["not_a_pass"]
    assert out["checks_run"] == 2 and out["checks_not_run"] == 1


def test_what_is_known_about_the_sheet_is_reported_even_though_the_check_cannot_run():
    known = truth.text_scale_check(HYGIENE_DRAWING, LAYOUT)["what_is_known"]
    assert known["plot_scale_text"] == "1:100"
    assert known["paper_units"] == "mm"
    assert known["header_dimscale"] == 100.0
    assert known["dim_style_scales"] == [{"name": "ISO-25", "scale": 100.0}]


def test_hygiene_passes_the_evidence_contract_audit():
    assert ev.audit_response(hygiene()) == []
    assert hygiene()["evidence"]["grade"] == ev.Grade.INFERRED.value


def test_hygiene_is_deterministic():
    assert json.dumps(hygiene(), sort_keys=True, default=str) == json.dumps(
        hygiene(), sort_keys=True, default=str
    )


@pytest.mark.parametrize("units", ALL_UNIT_SHAPES)
def test_hygiene_never_invents_a_unit(units):
    out = hygiene(units=units)
    if units["length_unit"] != "m":
        assert '"unit": "m"' not in json.dumps(out)


# =============================================================================
# 10. The catalogue entries
# =============================================================================


TWO_RECIPES = ("dimension_truth", "drafting_hygiene")


def test_both_recipes_are_registered_with_everything_the_catalogue_promises():
    for name in TWO_RECIPES:
        recipe = rr.get(name)
        assert recipe is not None, name
        row = recipe.as_dict()
        assert row["answers"].strip(), name
        assert row["when_to_use"].strip(), name
        assert row["returns"], name
        assert row["built_on"], name
        assert row["limits"], name
        for param in row["params"]:
            assert param["about"].strip(), (name, param["name"])


def test_both_recipes_declare_they_carry_meaning():
    """Both say what something IS — a label disagrees, content is hidden — so
    both are bound to publish `evidence` by the registry envelope."""
    for name in TWO_RECIPES:
        assert rr.get(name).carries_meaning is True


def test_the_catalogue_points_at_the_sibling_rather_than_competing_with_it():
    """`schedule_check` compares stated AREAS from a drawn table; this compares
    printed LENGTHS from free annotation. Confusing the two produces a
    confident answer to the wrong question."""
    row = rr.get("dimension_truth").as_dict()
    assert "schedule_check" in row["when_to_use"]


def test_no_drawing_specific_constant_entered_the_code_g1():
    source = (
        pathlib.Path(truth.__file__).read_text(encoding="utf-8")
    )
    forbidden = re.compile(
        r"VL2|DP4|LP1|TH3|Primary School|SchoolHatch|BlockBoundary|32638"
    )
    assert forbidden.search(source) is None


def test_the_only_layer_name_in_the_code_is_the_one_the_format_defines():
    """`0` is the DXF default layer, present in every file ever written. It is
    a fact about the FORMAT, like `RING_TYPES` — not one drawing's habit."""
    assert truth.DEFAULT_LAYER == "0"


# =============================================================================
# 11. Live — every drawing in the store (G5). Skipped without MongoDB.
# =============================================================================


def _live():
    from app import store as live_store

    try:
        drawings = live_store.list_drawings()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawings:
        pytest.skip("the store holds no drawing")
    return live_store, drawings


def _real_layouts(drawing):
    return [
        str(l["name"])
        for l in (drawing.get("layouts") or [])
        if l.get("name")
        and not str(l["name"]).startswith("[block] ")
        and int(l.get("entity_count") or 0) > 0
    ]


def test_live_every_drawing_answers_both_recipes_without_inventing_anything():
    """The invariant sweep. Janadriyah's numbers will never catch the 19th file.

    What is asserted is not a value — values differ per drawing, which is the
    whole point — but the shape of an honest answer: the coverage arithmetic
    holds, the evidence contract holds, and no unit is claimed that the drawing
    does not declare.
    """
    live_store, drawings = _live()
    from app import recipes

    seen = 0
    for summary in drawings:
        drawing = live_store.get_drawing(str(summary["_id"]))
        if not drawing:
            continue
        layouts = _real_layouts(drawing)
        if not layouts:
            continue
        layout = "Model" if "Model" in layouts else layouts[0]
        units = live_store._unit_names(drawing, layout)
        seen += 1

        out = recipes.run(
            str(summary["_id"]), "dimension_truth", {}, layout=layout, units=units
        )
        assert out["coverage_arithmetic"]["holds"] is True, (summary["_id"], layout)
        assert out["coverage_arithmetic"]["verdict_holds"] is True
        assert out["census"]["agrees_with_this_recipe"] is True
        assert ev.audit_response(out) == [], (summary["_id"], layout)
        if units.get("length_unit") is None:
            assert '"unit": "m"' not in json.dumps(out, default=str)

        hyg = recipes.run(
            str(summary["_id"]), "drafting_hygiene", {}, layout=layout, units=units
        )
        assert ev.audit_response(hyg) == [], (summary["_id"], layout)
        assert hyg["checks_not_run"] == 1
        assert finding(hyg, "text_height_vs_sheet_scale")["clean"] is None

    assert seen >= 1, "no drawing in the store had a layout with entities"


def test_live_the_reference_drawing_has_the_hygiene_figures_measured_from_the_store():
    """Measured 25 August 2026 on `JANADRIYAH DMP - 20240506.dxf`, layout Model.

    The id lives in the TEST, never in production `.py` (G1).
    """
    live_store, _ = _live()
    from app import recipes

    reference = "596212db022a3397"
    drawing = live_store.get_drawing(reference)
    if not drawing:
        pytest.skip(f"reference drawing {reference} has not been ingested")

    units = live_store._unit_names(drawing, "Model")
    out = recipes.run(reference, "drafting_hygiene", {}, layout="Model", units=units)

    # 57 entities on layer `0` in modelspace, and 12 layers the layer table
    # marks hidden: 4 off, 7 frozen, and one — `CommercialHatch` — frozen in
    # two viewports, which is hidden on those sheets and nowhere else. Counting
    # it would have been easy to leave out; it holds 12 entities that would
    # then have read as visible.
    assert out["entities_on_default_layer"] == 57
    assert out["hidden_layers_total"] == 12
    assert out["layer_table_available"] is True
    assert out["entities_on_hidden_layers"] > 0


def test_live_a_dimension_whose_text_is_generated_at_draw_time_is_never_counted_as_agreeing():
    """The reference drawing's 9,867 modelspace DIMENSIONs print `<>`.

    `<>` means "show the measured value", which is generated when the drawing
    is drawn and is not stored anywhere in the file. There is therefore nothing
    to compare, and the honest result is that none of them is checked — not
    that all of them agree.
    """
    live_store, _ = _live()
    from app import recipes

    reference = "596212db022a3397"
    drawing = live_store.get_drawing(reference)
    if not drawing:
        pytest.skip(f"reference drawing {reference} has not been ingested")

    assert census.classify_text("<>")[0] == census.KIND_OTHER

    units = live_store._unit_names(drawing, "Model")
    out = recipes.run(reference, "dimension_truth", {}, layout="Model", units=units)
    assert out["labels_agree"] <= out["labels_checked"]
    assert out["labels_checked"] + out["labels_ambiguous"] + out[
        "labels_not_associated"
    ] + out["labels_skipped"] == out["labels_total"]
