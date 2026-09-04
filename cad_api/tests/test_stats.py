"""Acceptance tests for UPLIFT-06 -- distribution statistics in `measure`.

Same two halves as `test_measure.py`, for the same reason. The pure half builds
a population in Python and asserts the shape and the reasoning; the live half
checks the thirteen-row table in the spec, which is a property of ingested data
and cannot be mocked without the test passing while the aggregation is wrong.

Four numbers in `docs/UPLIFT-06-STATS.md` did not survive contact with the
store, and the corrections are asserted here rather than assumed:

* the mode is counted at **2** decimals, not 6. At 6 decimals VL3's mode holds
  265 plots; the 449 the spec's own Definition of Done demands appears only at
  2, and 2 reproduces all thirteen `n modus` values exactly.
* `sd` is the **population** deviation. C10, LP1, VL2 and VL4 each round to a
  different figure under the sample deviation, and to the spec's under this one.
* the tolerance is per column: the table prints `mean` and the sums to 2
  decimals but `sd` and `max` to 1, so 0.01 cannot hold for those two.
* `above_median: 178` and `p75: 305.6` in the spec's illustrative JSON are not
  reachable from the data under any reading. With 449 of 512 values at the
  median, VL3's third quartile IS the median.

Every one of those is recorded in `docs/PROGRESS.md`.
"""

from __future__ import annotations

import math
import statistics as st

import pytest

from app import evidence as ev
from app import store, store_stats


# ---------------------------------------------------------------------------
# Fixtures for the pure half
# ---------------------------------------------------------------------------

METRES = {
    "name": "m",
    "declared_in_file": True,
    "length_unit": "m",
    "area_unit": "m2",
    "space": "model",
}

NO_UNITS = {
    "name": "unitless",
    "declared_in_file": False,
    "length_unit": None,
    "area_unit": None,
    "space": "model",
}

POP = "layer=X AND layout=Model"


def _describe(values, **over):
    kwargs = {"measure": "area", "units": METRES, "population": POP}
    kwargs.update(over)
    return store_stats.describe(values, **kwargs)


#: Every key in a stats block that carries a dimension and must therefore be a
#: Quantity rather than a bare float.
DIMENSIONED = ("mean", "median", "mode", "min", "max", "sd", "p25", "p75", "sum")

#: Every key that is a count of things. A count has no unit and must NOT be
#: dressed up as one -- `Quantity` is for numbers that can be wrong about their
#: unit, and 449 plots cannot.
COUNTS = ("n", "measured_count", "matched_count", "unmeasurable_count",
          "mode_count", "above_median")


# ---------------------------------------------------------------------------
# Pure: the shape of a statistic
# ---------------------------------------------------------------------------


def test_every_dimensioned_statistic_is_a_quantity():
    """A bare 308.76 cannot say it is a mean of areas of 512 of 512 objects on
    one layer. `audit_response` already refuses bare numbers for three of these
    keys; the other six are the same kind of number and get the same treatment."""
    s = _describe([300.0, 300.0, 400.0, 500.0])
    for key in DIMENSIONED:
        q = s[key]
        assert isinstance(q, dict), f"{key} is a bare value, not a Quantity"
        assert q["basis"], f"{key} does not say what it is a number OF"
        assert q["method"], f"{key} does not say how it was measured"
        assert q["unit"] == "m2", f"{key} lost its unit"


def test_counts_are_plain_integers_not_quantities():
    s = _describe([300.0, 300.0, 400.0])
    for key in COUNTS:
        assert isinstance(s[key], int), f"{key} should be a plain count"
        assert not isinstance(s[key], bool)


def test_every_statistic_travels_with_its_denominator():
    """Definition of Done: `measured_count` is reported beside every statistic.
    The mean of 133 of 133 and the mean of 133 of 20,000 are two different
    claims, and the basis of each number has to carry the difference."""
    s = _describe([300.0] * 133, matched_count=20000)
    assert s["measured_count"] == 133
    assert s["matched_count"] == 20000
    assert s["unmeasurable_count"] == 20000 - 133
    for key in DIMENSIONED:
        assert "133" in s[key]["basis"] and "20000" in s[key]["basis"], (
            f"{key} basis hides that it covers 133 of 20000: {s[key]['basis']!r}"
        )


# ---------------------------------------------------------------------------
# Pure: the trap this spec exists for
# ---------------------------------------------------------------------------


def test_unmeasurable_entities_never_enter_the_mean_as_zero():
    """The HATCH trap, stated as a test. Eight objects that carry no area are
    not eight objects of 0 m2; averaging them in drags the mean down with a
    number that is wrong rather than one that is missing."""
    s = _describe([300.0] * 8, matched_count=16)
    # 8 real values of 300 and 8 that carry nothing.
    assert s["mean"]["value"] == pytest.approx(300.0), (
        "the eight unmeasurable objects were averaged in as zero"
    )
    assert s["mean"]["value"] != pytest.approx(150.0)
    assert s["unmeasurable_count"] == 8


def test_nothing_measurable_withholds_every_number_and_says_why():
    """G8: the answer when the data is absent is null plus a reason, never 0."""
    s = _describe([], matched_count=40)
    for key in DIMENSIONED:
        q = s[key]
        assert q["value"] is None, f"{key} produced a number out of nothing"
        assert q["not_zero"] is True
        assert q["withheld_reason"], f"{key} was withheld without a reason"
    assert s["n"] == 0
    assert s["matched_count"] == 40


def test_a_withheld_statistic_cannot_be_summed_into_a_total():
    """The property that makes the withholding stick. A test rather than a
    comment because it is one careless `sum()` away from being untrue."""
    s = _describe([], matched_count=4)
    q = ev.Quantity(**{k: v for k, v in [
        ("basis", s["mean"]["basis"]), ("method", s["mean"]["method"]),
        ("value", s["mean"]["value"]), ("unit", s["mean"]["unit"]),
        ("withheld_reason", s["mean"]["withheld_reason"]),
    ]})
    assert q.is_withheld
    with pytest.raises(ev.EvidenceError):
        sum([q])


def test_a_drawing_that_declares_no_unit_gets_none_and_a_reason():
    """G2. Eleven of eighteen drawings are in inches and three declare nothing;
    writing 'm' because this one happens to be metric is the whole failure.

    The population is chosen so that every one of the nine statistics has a
    value: a withheld quantity carries `withheld_reason` and never claimed a
    unit in the first place, so asking it for a `unit_reason` would test the
    wrong thing."""
    s = _describe([300.0, 300.0, 400.0], units=NO_UNITS)
    for key in DIMENSIONED:
        assert s[key]["value"] is not None, f"{key} should have a value here"
        assert s[key]["unit"] is None, f"{key} invented a unit"
        assert s[key]["unit_reason"], f"{key} dropped the unit without saying so"


def test_a_withheld_number_explains_its_absence_not_its_unit():
    """The other half of G2. A number that does not exist has no unit to be
    wrong about, and `Quantity` enforces exactly that asymmetry."""
    s = _describe([], matched_count=3, units=NO_UNITS)
    for key in DIMENSIONED:
        assert s[key]["unit"] is None
        assert s[key]["unit_reason"] is None
        assert s[key]["withheld_reason"]


# ---------------------------------------------------------------------------
# Pure: the mode, and the rounding that makes it mean anything
# ---------------------------------------------------------------------------


def test_the_mode_is_counted_on_rounded_values():
    """Areas of plots built to one module are stored as 300.0 and 299.99988.
    Counted raw, every value is its own mode and `mode_count` is noise."""
    values = [300.0, 299.9998779296875, 300.0000001, 250.0]
    s = _describe(values)
    assert s["mode"]["value"] == pytest.approx(300.0, abs=0.005)
    assert s["mode_count"] == 3
    assert s["mode_decimals"] == 2


def test_the_mode_rounding_is_reported_because_it_changes_the_answer():
    """The same population has a different mode at a different precision, so a
    mode that does not name its precision is not a checkable number."""
    values = [300.0, 299.9998779296875, 300.0000001, 250.0]
    coarse = _describe(values, mode_decimals=2)
    fine = _describe(values, mode_decimals=6)
    assert coarse["mode_count"] == 3
    assert fine["mode_count"] == 2, "at 6 decimals 299.999878 is its own value"
    assert fine["mode_decimals"] == 6
    assert str(fine["mode_decimals"]) in fine["mode"]["method"]


def test_a_population_where_nothing_repeats_has_no_mode():
    """A 'mode' occurring once in seventeen values is the smallest value with
    the luck of being first, not a typical value. Withheld, with the reason."""
    s = _describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mode"]["value"] is None
    assert s["mode_count"] == 0
    assert "repeats" in s["mode"]["withheld_reason"]


def test_a_tied_mode_says_it_is_tied():
    s = _describe([300.0, 300.0, 400.0, 400.0, 500.0])
    assert s["mode_ties"] == 2
    assert s["mode_note"]


# ---------------------------------------------------------------------------
# Pure: distribution_note is produced by a rule, not by a model
# ---------------------------------------------------------------------------


def test_a_floored_distribution_is_named_as_one():
    """min = median = mode is the signature the spec is built on: a hard floor
    at the design size and spread in one direction only."""
    s = _describe([300.0] * 9 + [310.0, 350.0, 420.0])
    assert s["distribution_note"]
    assert "floored" in s["distribution_note"]


def test_a_floor_is_recognised_through_the_jitter_of_a_computed_area():
    """The floor in the store is 299.9953, not 300.0. Compared at full float
    precision no layer in this drawing is floored and the note never appears;
    compared at the precision the mode already uses, all thirteen are."""
    s = _describe([299.9953, 300.0, 300.0000071, 305.6, 477.8682])
    assert s["distribution_note"] and "floored" in s["distribution_note"]


def test_an_identical_population_is_not_told_it_spreads_upward():
    """Ordering matters, and the spec has it the other way round. `sd == 0`
    satisfies min = median = mode too, so if the floor rule were tried first a
    population of twenty-four identical values would be told its spread runs
    upward -- of a spread that does not exist."""
    s = _describe([250.0] * 24)
    assert "identical" in s["distribution_note"]
    assert "floored" not in s["distribution_note"]


def test_a_right_skewed_population_is_told_the_mean_is_pulled():
    s = _describe([10.0, 100.0, 105.0, 110.0, 115.0, 900.0])
    assert s["distribution_note"]
    assert "median better represents" in s["distribution_note"]


def test_an_ordinary_distribution_gets_no_note_at_all():
    """Definition of Done: the note must NOT appear for data that is not
    floored. A note on every row is a note nobody reads."""
    s = _describe([100.0, 110.0, 120.0, 130.0, 140.0])
    assert s["distribution_note"] is None


def test_a_zero_median_does_not_divide_the_mean_pull_rule_by_luck():
    """median * 1.02 is 0 when the median is 0, and every positive mean then
    'exceeds' it. Guarded, because a degenerate layer is ordinary here (G8)."""
    s = _describe([0.0, 0.0, 0.0, 5.0])
    assert s["distribution_note"] is None or "median better represents" not in (
        s["distribution_note"]
    )


# ---------------------------------------------------------------------------
# Pure: small populations, and the limits that are stated rather than assumed
# ---------------------------------------------------------------------------


def test_a_small_population_says_so_beside_its_median():
    s = _describe([300.0, 300.0, 400.0])
    assert s["small_population_note"]
    assert "3" in s["small_population_note"]


def test_a_large_population_carries_no_small_population_note():
    s = _describe([300.0] * 512)
    assert s["small_population_note"] is None


def test_quartiles_need_two_values_and_say_so_when_they_do_not_have_them():
    s = _describe([300.0])
    for key in ("p25", "p75", "sd"):
        assert s[key]["value"] is None
        assert s[key]["withheld_reason"]
    # ...but the ones that do exist for a single value are still given.
    assert s["mean"]["value"] == pytest.approx(300.0)
    assert s["median"]["value"] == pytest.approx(300.0)


def test_the_statement_carries_the_number_its_unit_and_its_denominator():
    """The sentence an agent quotes. `_measure_statement` exists because both
    wrong numbers this project shipped were composed in prose around a correct
    figure; a stats block without one invites the same."""
    s = _describe([300.0] * 8, matched_count=10)
    assert "300" in s["statement"]
    assert "m2" in s["statement"]
    assert "8" in s["statement"] and "10" in s["statement"]


def test_the_evidence_audit_passes_over_a_flattened_stats_block():
    """`audit_response` refuses a bare number under `mean`, `median` or `sd`.
    Flattening is how a nested block reaches a response body, so it is checked
    in the flattened form rather than in the one that hides the keys."""
    s = _describe([300.0, 400.0])
    payload = {"drawing_id": "d", "scope_note": POP}
    payload.update({k: s[k] for k in ("mean", "median", "sd")})
    assert ev.audit_response(payload) == []


def test_the_audit_would_have_caught_the_shape_the_spec_illustrates():
    """A guard on the guard. The spec's example JSON writes `"mean": 308.76`,
    and this proves that shape is refused rather than merely discouraged."""
    payload = {"drawing_id": "d", "scope_note": POP, "mean": 308.76}
    assert any("V4" in p for p in ev.audit_response(payload))


# ---------------------------------------------------------------------------
# Against the live database -- the thirteen-row table
# ---------------------------------------------------------------------------

JANADRIYAH = "596212db022a3397"

#: The plot typologies. Here and not in `app/` on purpose: these are layer names
#: from one drawing, and G1 keeps drawing-specific constants out of the code.
TYPOLOGY = ["C10", "LP1", "VL2", "LP2", "VL3", "VL4 SPECIAL", "LP4", "VL4",
            "LP3", "DP4 ZEROLOT", "DP4", "LP5", "TH3"]

#: layer -> (n, mean, median, mode, mode_count, min, max, sd, sum)
TABLE = {
    "C10":         (188, 408.13, 400.0, 400, 174, 400.0, 577.9, 30.7,  76729.30),
    "LP1":         (88,  404.59, 400.0, 400,  79, 400.0, 444.9, 13.6,  35603.59),
    "VL2":         (133, 322.07, 300.0, 300,  89, 300.0, 413.1, 33.3,  42835.60),
    "LP2":         (169, 308.92, 300.0, 300, 135, 300.0, 405.9, 20.2,  52207.84),
    "VL3":         (512, 308.76, 300.0, 300, 449, 300.0, 477.9, 25.4, 158087.09),
    "VL4 SPECIAL": (140, 307.15, 300.0, 300, 112, 300.0, 395.0, 18.0,  43001.56),
    "LP4":         (159, 303.69, 300.0, 300, 146, 300.0, 395.2, 14.0,  48287.25),
    "VL4":         (195, 303.33, 300.0, 300, 182, 300.0, 417.1, 14.2,  59148.57),
    "LP3":         (319, 254.90, 250.0, 250, 282, 250.0, 361.6, 17.4,  81313.22),
    "DP4 ZEROLOT": (139, 252.79, 250.0, 250, 131, 250.0, 425.0, 16.2,  35138.38),
    "DP4":         (236, 250.11, 250.0, 250, 235, 250.0, 275.0,  1.6,  59024.99),
    "LP5":         (24,  250.00, 250.0, 250,  24, 250.0, 250.0,  0.0,   6000.00),
    "TH3":         (78,  200.00, 200.0, 200,  78, 200.0, 200.0,  0.0,  15600.00),
}

#: The table prints `mean`, `median`, `min` and the sums to 2 decimals, so the
#: spec's blanket 0.01 holds for those. `sd` and `max` are printed to 1, and
#: 0.01 cannot hold for a figure rounded to 0.1 -- C10's max is 577.9324 against
#: a printed 577.9. Half of the last printed digit is the honest tolerance.
TIGHT = 0.01
LOOSE = 0.05

OVERALL = {"n": 2380, "mean": 299.57, "median": 300.0, "sd": 52.9,
           "sum": 712977.39, "above_median": 485}

#: A real population in the same drawing that is NOT floored: 17 areas from
#: 57.6 to 654, median 270.5, mean below it. The negative half of the
#: Definition of Done needs a real one, not only a synthetic one.
UNFLOORED_LAYER = "Linear Park"


def _mongo_or_skip():
    try:
        drawing = store.get_drawing(JANADRIYAH)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no database reachable: {type(exc).__name__}")
    if not drawing:
        pytest.skip(
            f"reference drawing {JANADRIYAH} is not ingested; "
            "these acceptance figures cannot be checked"
        )
    return drawing


def _typology(drawing, layers=None):
    return store_stats.stats_by_layer(
        JANADRIYAH,
        measure="area",
        layout_name="Model",
        layers=list(TYPOLOGY if layers is None else layers),
        units=store._unit_names(drawing, "Model"),
        dxftype="LWPOLYLINE",
    )


@pytest.fixture(scope="module")
def typology():
    drawing = _mongo_or_skip()
    return _typology(drawing)


def _row(result, name):
    for row in result["by_layer"]:
        if row["name"] == name:
            return row
    raise AssertionError(f"{name} missing from by_layer")


@pytest.mark.parametrize("layer", list(TABLE))
def test_the_thirteen_row_table_is_reproduced(typology, layer):
    """Definition of Done 1. Every column of every row, at the precision the
    spec printed it."""
    n, mean, median, mode, mode_count, lo, hi, sd, total = TABLE[layer]
    s = _row(typology, layer)["stats"]
    assert s["n"] == n
    assert s["mean"]["value"] == pytest.approx(mean, abs=TIGHT)
    assert s["median"]["value"] == pytest.approx(median, abs=TIGHT)
    assert s["mode"]["value"] == pytest.approx(mode, abs=TIGHT)
    assert s["mode_count"] == mode_count
    assert s["min"]["value"] == pytest.approx(lo, abs=TIGHT)
    assert s["max"]["value"] == pytest.approx(hi, abs=LOOSE)
    assert s["sd"]["value"] == pytest.approx(sd, abs=LOOSE)
    assert s["sum"]["value"] == pytest.approx(total, abs=TIGHT)
    assert s["mean"]["unit"] == "m2"


def test_the_overall_row_is_the_union_of_the_thirteen(typology):
    """Definition of Done 2. 2,380 is the sum of the thirteen counts and
    712,977.39 the sum of the thirteen sums, so the overall row is those
    thirteen layers and nothing else -- not 'every polygon in model space'."""
    s = typology["overall"]["stats"]
    assert s["n"] == OVERALL["n"] == sum(r[0] for r in TABLE.values())
    assert s["mean"]["value"] == pytest.approx(OVERALL["mean"], abs=TIGHT)
    assert s["median"]["value"] == pytest.approx(OVERALL["median"], abs=TIGHT)
    assert s["sd"]["value"] == pytest.approx(OVERALL["sd"], abs=LOOSE)
    assert s["sum"]["value"] == pytest.approx(OVERALL["sum"], abs=TIGHT)


def test_the_overall_row_declares_how_many_populations_it_mixed(typology):
    """The portfolio trap. 299.57 m2 describes no plot in the drawing; it lands
    near 300 only because 1,308 of 2,380 plots are of a 300 m2 type. A mean over
    a mixed population has to say how mixed it is."""
    overall = typology["overall"]
    assert overall["populations"] == 13
    assert overall["mixed_population_note"]
    assert "13" in overall["mixed_population_note"]
    assert "13" in overall["stats"]["mean"]["basis"]


def test_a_single_population_carries_no_mixing_note(typology):
    assert _row(typology, "VL3").get("mixed_population_note") is None


def test_the_mode_rounding_is_what_produces_449(typology):
    """Definition of Done 4, and the reason the spec's '6 decimals' is wrong:
    at 6 decimals this is 265."""
    assert _row(typology, "VL3")["stats"]["mode_count"] == 449
    assert _row(typology, "VL3")["stats"]["mode_decimals"] == 2


def test_at_six_decimals_the_documented_mode_count_does_not_appear(typology):
    """The correction, asserted rather than described. Kept because the next
    person to read the spec will try 6 and needs to see what it gives."""
    drawing = _mongo_or_skip()
    fine = store_stats.stats_by_layer(
        JANADRIYAH, measure="area", layout_name="Model", layers=["VL3"],
        units=store._unit_names(drawing, "Model"), dxftype="LWPOLYLINE",
        mode_decimals=6,
    )
    assert _row(fine, "VL3")["stats"]["mode_count"] == 265


def test_the_two_uniform_layers_have_no_spread(typology):
    """Definition of Done 5. Both are 0.0 to the precision the table prints,
    and neither is exactly zero -- LP5's is 0.0007 -- so the note that claims
    every value is identical must NOT fire for them."""
    for layer in ("LP5", "TH3"):
        s = _row(typology, layer)["stats"]
        assert s["sd"]["value"] == pytest.approx(0.0, abs=TIGHT)
        assert s["sd"]["value"] > 0.0
        assert "identical" not in (s["distribution_note"] or "")


def test_every_one_of_the_thirteen_is_reported_as_floored(typology):
    """Definition of Done 3, first half."""
    for layer in TABLE:
        note = _row(typology, layer)["stats"]["distribution_note"]
        assert note and "floored" in note, f"{layer} lost its floor note"


def test_a_real_unfloored_layer_gets_no_floor_note():
    """Definition of Done 3, second half, against real geometry rather than a
    synthetic population."""
    drawing = _mongo_or_skip()
    r = _typology(drawing, layers=[UNFLOORED_LAYER])
    s = _row(r, UNFLOORED_LAYER)["stats"]
    assert s["n"] >= 8
    assert "floored" not in (s["distribution_note"] or "")


def test_every_row_reports_the_denominator_it_was_computed_over(typology):
    """Definition of Done 6."""
    for row in typology["by_layer"] + [typology["overall"]]:
        s = row["stats"]
        assert s["measured_count"] == s["n"]
        assert s["matched_count"] >= s["measured_count"]
        assert str(s["measured_count"]) in s["mean"]["basis"]


def test_the_count_above_the_overall_median_is_the_documented_485(typology):
    """The spec's headline: 485 plots (20.4%) sit above the overall median.
    It is reproduced only when the comparison uses the same rounding as the
    mode -- which is the third piece of evidence that the rounding is 2."""
    assert typology["overall"]["stats"]["above_median"] == OVERALL["above_median"]


def test_a_layer_that_is_not_in_the_drawing_is_reported_absent_not_dropped():
    """G8, and the never-silently-drop rule. A requested population that does
    not exist is a finding; leaving it out of the response makes the answer look
    complete."""
    drawing = _mongo_or_skip()
    r = _typology(drawing, layers=["VL3", "NO SUCH LAYER"])
    assert r["layers_absent"] == ["NO SUCH LAYER"]
    assert [row["name"] for row in r["by_layer"]] == ["VL3"]


def test_the_response_survives_the_evidence_audit(typology):
    assert ev.audit_response(typology) == []
    for row in typology["by_layer"]:
        assert ev.audit_response({**row, "drawing_id": JANADRIYAH,
                                  "scope_note": typology["scope_note"]}) == []


def test_the_sum_matches_what_measure_reports_for_the_same_filter():
    """Cross-check against the tool that already exists. If these two disagree
    the drawing has not changed -- one of them is wrong."""
    drawing = _mongo_or_skip()
    r = _typology(drawing, layers=["VL3"])
    m = store.measure_entities(
        JANADRIYAH, measure="area", layout_name="Model", layer="VL3",
        dxftype="LWPOLYLINE",
    )
    assert _row(r, "VL3")["stats"]["sum"]["value"] == pytest.approx(
        m["total_for_all_matched"], abs=TIGHT
    )
    assert _row(r, "VL3")["stats"]["n"] == m["measured_entities"]


# ---------------------------------------------------------------------------
# `attach`: the one line `measure_entities` is waiting for
# ---------------------------------------------------------------------------


def _measured(**over):
    kwargs = {"measure": "area", "layout_name": "Model", "layer": "VL3",
              "dxftype": "LWPOLYLINE"}
    kwargs.update(over)
    return kwargs, store.measure_entities(JANADRIYAH, **kwargs)


def test_attach_adds_stats_without_measure_having_to_change_shape():
    """The coordinator's patch is `if stats: store_stats.attach(...)`, so this
    is that call and nothing else. Everything `measure` already reported has to
    survive it untouched."""
    _mongo_or_skip()
    kwargs, r = _measured()
    before = {k: v for k, v in r.items() if k != "stats"}
    store_stats.attach(r, JANADRIYAH, layout_name=kwargs["layout_name"],
                       layer=kwargs["layer"], dxftype=kwargs["dxftype"])
    assert r["stats"]["n"] == 512
    for key, value in before.items():
        assert r[key] == value, f"attach changed {key}"


def test_attach_never_disagrees_with_the_counts_measure_already_published():
    """The counts are read from the result rather than recomputed, so `stats`
    and `measured_entities` cannot tell two stories about one population."""
    _mongo_or_skip()
    kwargs, r = _measured(layer="0", dxftype=None)
    store_stats.attach(r, JANADRIYAH, layout_name="Model", layer="0")
    s = r["stats"]
    assert s["matched_count"] == r["total_matches"]
    assert s["measured_count"] == r["measured_entities"]
    assert s["unmeasurable_count"] == r["unmeasurable_entities"]
    assert s["sum"]["value"] == pytest.approx(r["sum_measured_only"], abs=TIGHT)


def test_attach_gives_every_grouped_row_its_own_statistics():
    _mongo_or_skip()
    kwargs, r = _measured(layer=None, group_by="layer", top=5)
    store_stats.attach(r, JANADRIYAH, layout_name="Model", group_by="layer")
    rows = r["by_layer"]
    assert rows
    for row in rows:
        assert "stats" in row, f"{row['name']} got no statistics"
        assert row["stats"]["measured_count"] == row["measured_entities"]


def test_grouping_by_type_gets_statistics_too_rather_than_silently_none():
    """`by_type` rows once came back without a `stats` block and with nothing in
    the response saying why. A row that is quietly missing its statistics reads
    as a row that has none to give."""
    _mongo_or_skip()
    kwargs, r = _measured(layer="0", dxftype=None, group_by="type", top=10)
    store_stats.attach(r, JANADRIYAH, layout_name="Model", layer="0",
                       group_by="type")
    rows = r["by_type"]
    assert len(rows) > 1, "this fixture layer should hold more than one type"
    for row in rows:
        assert "stats" in row, f"type {row['name']} got no statistics"
        assert row["stats"]["measured_count"] == row["measured_entities"]
        assert str(row["name"]) in row["stats"]["mean"]["basis"]


def test_an_unmeasurable_type_is_summarised_as_withheld_not_as_zero():
    """The HATCH trap where it actually lives. A type that carries no area gets
    a row of withheld numbers, and not a row of zeroes."""
    _mongo_or_skip()
    kwargs, r = _measured(layer=None, dxftype="TEXT")
    store_stats.attach(r, JANADRIYAH, layout_name="Model", dxftype="TEXT")
    s = r["stats"]
    assert s["n"] == 0
    assert s["matched_count"] > 0
    for key in DIMENSIONED:
        assert s[key]["value"] is None
        assert s[key]["withheld_reason"]


# ---------------------------------------------------------------------------
# G5: invariants over every ingested drawing, not only the reference one
# ---------------------------------------------------------------------------


def _all_drawings():
    try:
        return store.list_drawings()
    except Exception:  # pragma: no cover - environment dependent
        return []


def _busiest_layer(drawing_id, layout):
    """The layer holding the most measurable areas, or None."""
    rows = list(store_stats.coll(store_stats.COLL_ENTITIES).aggregate([
        {"$match": {"drawing_id": drawing_id, "layout": layout,
                    "area": {"$type": "number"}}},
        {"$group": {"_id": "$layer", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
        {"$limit": 1},
    ]))
    return str(rows[0]["_id"]) if rows else None


@pytest.mark.parametrize("drawing_id", [d["_id"] for d in _all_drawings()])
def test_no_drawing_is_ever_given_a_unit_it_does_not_declare(drawing_id):
    """G2 and G5 together. Eleven of these drawings are in inches and three
    declare nothing; the Janadriyah figures cannot catch a metre stamped on any
    of them, and this is the test that can."""
    _mongo_or_skip()
    drawing = store.get_drawing(drawing_id)
    if not drawing:
        pytest.skip(f"{drawing_id} not readable")
    layout = "Model"
    layer = _busiest_layer(drawing_id, layout)
    if not layer:
        pytest.skip(f"{drawing_id} has no measurable area in {layout}")

    units = store._unit_names(drawing, layout)
    r = store_stats.stats_by_layer(
        drawing_id, measure="area", layout_name=layout, layers=[layer],
        units=units,
    )
    s = _row(r, layer)["stats"]
    for key in DIMENSIONED:
        q = s[key]
        if q["value"] is None:
            assert q["withheld_reason"], f"{key} withheld with no reason"
            assert q["not_zero"] is True
            continue
        assert q["unit"] == units["area_unit"], (
            f"{key} claims {q['unit']!r} while the drawing says "
            f"{units['area_unit']!r}"
        )
        if q["unit"] is None:
            assert q["unit_reason"], f"{key} dropped the unit silently"
    assert s["n"] <= s["matched_count"]
    assert s["mode_decimals"] == store_stats.MODE_DECIMALS
    assert ev.audit_response(r) == []


def test_the_reference_statistics_agree_with_a_plain_recomputation():
    """The aggregation is the part that can be wrong in a way the table cannot
    see, so the values are pulled back and recomputed with the standard library.
    `sd` here is the population deviation, which is the claim being checked."""
    drawing = _mongo_or_skip()
    rows = store_stats.values_for(
        JANADRIYAH, measure="area", layout_name="Model", layers=["VL3"],
        dxftype="LWPOLYLINE",
    )
    vals = rows["values"]["VL3"]
    assert len(vals) == 512
    assert st.fmean(vals) == pytest.approx(308.76, abs=TIGHT)
    assert st.pstdev(vals) == pytest.approx(25.4, abs=LOOSE)
    assert st.stdev(vals) != pytest.approx(25.4, abs=TIGHT), (
        "the sample deviation rounds to 25.5 here; if it ever matches, the "
        "ddof choice has stopped being observable and this test is misleading"
    )
    assert math.fsum(vals) == pytest.approx(158087.09, abs=TIGHT)
