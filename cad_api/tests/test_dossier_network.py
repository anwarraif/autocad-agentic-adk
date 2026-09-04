"""DOSSIER lane 2 — Σ length and connected chains, tested on fixtures alone.

No Mongo, no DXF, no drawing id. `network_measure` is a pure function over rows
that have already been fetched, so everything below is a list of dicts written
by hand and an assertion about the dict that comes back.

Almost nothing here is a number from Janadriyah. What is pinned is PROPERTY
(rule G5): a chain is one chain however far from the origin it is drawn; the
same geometry scaled by a thousand gets a tolerance scaled by a thousand, so
nothing in drawing units is hiding in the code; an entity with no length is
counted, never zeroed; and a drawing that declares no unit gets no unit
invented for it. Those hold for the 19th drawing from a contractor nobody has
met yet. A fixed metre figure never would.

Two tests are here because the bounding box is all this store holds. Rows carry
no endpoint coordinates, so the chain count is an inference — and the tests
pin the module SAYING so (`chains_bound`, `endpoint_quality`, the caveat)
rather than pinning a chain count as if it were measured.
"""

from __future__ import annotations

import copy
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import dossier_network as dn  # noqa: E402


# The projected coordinates this project really works at. Not decoration: a
# tolerance computed as a fraction of a coordinate rather than of a length
# would pass at the origin and fail here by six orders of magnitude.
E0, N0 = 691942.0, 2750756.0


def line(handle: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    """A LINE row shaped exactly as `autocad_entities` stores one."""
    return {
        "handle": handle,
        "type": "LINE",
        "length": math.hypot(x1 - x0, y1 - y0),
        "bbox": {
            "min": [min(x0, x1), min(y0, y1)],
            "max": [max(x0, x1), max(y0, y1)],
        },
        "bbox_centre": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
    }


def arc(handle: str, x0: float, y0: float, x1: float, y1: float, length: float) -> dict:
    """An ARC row. Its bbox bounds the arc, and its ends are NOT identified."""
    return {
        "handle": handle,
        "type": "ARC",
        "length": length,
        "bbox": {
            "min": [min(x0, x1), min(y0, y1)],
            "max": [max(x0, x1), max(y0, y1)],
        },
        "bbox_centre": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
    }


def chain_of(n: int, *, at_x: float, at_y: float, step: float = 30.0) -> list[dict]:
    """`n` LINEs laid end to end, each starting where the last one stopped."""
    return [
        line(f"h{at_x:.0f}-{i}", at_x + i * step, at_y, at_x + (i + 1) * step, at_y)
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# the five cases the lane brief names
# --------------------------------------------------------------------------


def test_a_simple_chain_is_one_chain():
    rows = chain_of(4, at_x=E0, at_y=N0)

    out = dn.network_measure(rows, unit="m")

    assert out["chains"] == 1
    assert out["chain_sizes_top"] == [4]
    assert out["largest_chain"] == 4
    assert out["singleton_chains"] == 0
    assert out["length_total"] == pytest.approx(120.0)
    assert out["measured_entities"] == 4
    assert out["unmeasured_entities"] == 0
    assert out["length_is_floor"] is False


def test_two_disjoint_chains_are_two_chains():
    # Two runs 5 km apart: nothing in either can be within a snap of the other.
    rows = chain_of(3, at_x=E0, at_y=N0) + chain_of(2, at_x=E0 + 5000.0, at_y=N0)

    out = dn.network_measure(rows, unit="m")

    assert out["chains"] == 2
    assert out["chain_sizes_top"] == [3, 2]
    assert out["chained_entities"] == 5


def test_an_entity_with_no_length_is_counted_never_zeroed():
    rows = chain_of(2, at_x=E0, at_y=N0)
    rows.append(
        {
            "handle": "no-length",
            "type": "LWPOLYLINE",
            "length": None,
            "bbox": {"min": [E0 + 60.0, N0], "max": [E0 + 90.0, N0 + 10.0]},
        }
    )

    out = dn.network_measure(rows, unit="m")

    assert out["measured_entities"] == 2
    assert out["unmeasured_entities"] == 1
    # The sum is over the two that were measurable, and it says so.
    assert out["length_total"] == pytest.approx(60.0)
    assert out["length_is_floor"] is True
    assert "floor" in out["length_note"].lower()
    assert "FLOOR" in out["caveat"]
    # And the missing one is still a member of the layer, not a dropped row.
    assert out["entities"] == 3
    assert out["chained_entities"] == 3


def test_empty_input_reports_no_length_and_no_chains_without_inventing_either():
    out = dn.network_measure([], unit="m")

    assert out["entities"] == 0
    # Null, not 0.0: nothing was measured, so nothing may be quoted.
    assert out["length_total"] is None
    assert out["measured_entities"] == 0
    assert out["unmeasured_entities"] == 0
    # Zero chains over an empty set is arithmetic, and the basis says exactly
    # that rather than implying a geometric finding.
    assert out["chains"] == 0
    assert out["snap_tolerance"] is None
    assert out["tolerance_basis"]
    assert out["caveat"]


def test_a_unitless_drawing_is_never_given_a_unit():
    rows = chain_of(3, at_x=E0, at_y=N0)

    out = dn.network_measure(rows, unit=None)

    assert out["unit"] is None
    assert "drawing units" in out["unit_note"]
    assert "must not be labelled" in out["caveat"]
    # The numbers are still there — absence of a unit is not absence of an
    # answer (rule G3) — but nowhere does the response name one.
    assert out["length_total"] == pytest.approx(90.0)
    assert out["chains"] == 1
    for key in ("tolerance_basis", "unit_note", "length_note", "chains_basis"):
        assert " m," not in out[key] and not out[key].endswith(" m")


# --------------------------------------------------------------------------
# the tolerance: derived, published, and honest about its grip
# --------------------------------------------------------------------------


def test_the_snap_tolerance_is_published_with_a_basis_that_names_its_source():
    out = dn.network_measure(chain_of(4, at_x=E0, at_y=N0), unit="m")

    assert out["snap_tolerance"] > 0.0
    basis = out["tolerance_basis"]
    assert "median" in basis
    assert str(dn.SNAP_FRACTION) in basis or f"{dn.SNAP_FRACTION:g}" in basis
    # It has to say which way the knob turns, because the knob is the answer.
    assert "merges" in basis and "splits" in basis


def test_the_tolerance_scales_with_the_drawing_so_nothing_assumes_metres():
    """The G1/G2 property. A constant in drawing units would break here.

    The same geometry, drawn a thousand times larger — a plan in millimetres
    rather than metres, or the inch drawings that are 11 of the 18 in this
    store. A tolerance that suits one must scale to the other, or it silently
    stops snapping anything.
    """
    small = dn.network_measure(chain_of(4, at_x=0.0, at_y=0.0, step=30.0), unit=None)
    large = dn.network_measure(
        chain_of(4, at_x=0.0, at_y=0.0, step=30_000.0), unit=None
    )

    ratio = large["snap_tolerance"] / small["snap_tolerance"]
    assert ratio == pytest.approx(1000.0, rel=1e-6)
    # And the answer itself does not move with the scale.
    assert small["chains"] == large["chains"] == 1


def test_the_response_shows_what_the_tolerance_is_worth():
    rows = chain_of(3, at_x=E0, at_y=N0)
    out = dn.network_measure(rows, unit="m")

    base = out["snap_tolerance"]
    seen = {
        entry["factor"]: (entry["snap_tolerance"], entry["chains"])
        for entry in out["tolerance_sensitivity"]
    }
    assert sorted(seen) == [0.5, 1.0, 2.0]
    assert seen[0.5][0] == pytest.approx(base * 0.5)
    assert seen[2.0][0] == pytest.approx(base * 2.0)
    # Endpoints that coincide exactly stay joined at every tolerance, so this
    # particular answer is stable — and the response is what proves it, rather
    # than a reader having to take the single number on trust.
    assert {value[1] for value in seen.values()} == {1}


def test_a_gap_wider_than_the_tolerance_splits_the_chain():
    """The property behind the whole instrument, stated both ways."""
    rows = [
        line("a", E0, N0, E0 + 30.0, N0),
        # starts 1 unit past where the first one ended: a real gap.
        line("b", E0 + 31.0, N0, E0 + 61.0, N0),
    ]

    out = dn.network_measure(rows, unit="m")

    assert out["snap_tolerance"] < 1.0
    assert out["chains"] == 2
    # Both boxes are degenerate, so both LINEs have exactly located ends and
    # the count is not hedged.
    assert out["chains_bound"] == "exact"

    joined = dn.network_measure(
        [line("a", E0, N0, E0 + 30.0, N0), line("b", E0 + 30.0, N0, E0 + 60.0, N0)],
        unit="m",
    )
    assert joined["chains"] == 1


# --------------------------------------------------------------------------
# what a bounding box can and cannot say about an endpoint
# --------------------------------------------------------------------------


def test_an_all_line_layer_states_its_count_as_a_lower_bound():
    """Four bbox corners are a SUPERSET of a LINE's two endpoints.

    A superset of candidate points can only add edges, and adding edges can
    only merge components. So the count can be too low and never too high, and
    the response has to say which direction it can be wrong in.
    """
    rows = chain_of(3, at_x=E0, at_y=N0)
    rows.append(line("diagonal", E0 + 90.0, N0, E0 + 120.0, N0 + 40.0))

    out = dn.network_measure(rows, unit="m")

    assert out["chains_bound"] == "lower_bound"
    assert out["endpoint_quality"]["endpoints_bounded"] == 1
    assert out["endpoint_quality"]["endpoints_exact"] == 3
    assert "LOWER BOUND" in out["chains_basis"]


def test_an_arc_downgrades_the_answer_to_an_approximation_and_says_why():
    rows = chain_of(2, at_x=E0, at_y=N0)
    rows.append(arc("curve", E0 + 60.0, N0, E0 + 70.0, N0 + 10.0, length=15.707))

    out = dn.network_measure(rows, unit="m")

    assert out["chains_bound"] == "approximate"
    assert out["endpoint_quality"]["endpoints_unknown"] == 1
    assert "APPROXIMATION" in out["chains_basis"]
    assert "no endpoint coordinates" in out["caveat"]


def test_a_closed_curve_has_no_free_ends_and_its_circumference_is_flagged():
    rows = chain_of(2, at_x=E0, at_y=N0)
    rows.append(
        {
            "handle": "roundabout",
            "type": "CIRCLE",
            "length": 62.831853,
            "bbox": {"min": [E0 + 50.0, N0 - 10.0], "max": [E0 + 70.0, N0 + 10.0]},
        }
    )

    out = dn.network_measure(rows, unit="m")

    assert out["endpoint_quality"]["closed_no_ends"] == 1
    # It touches the chain's end, and we cannot know that: it is its own chain.
    assert out["chains"] == 2
    assert out["closed_curve_entities"] == 1
    assert out["closed_curve_length"] == pytest.approx(62.831853)
    assert "CIRCUMFERENCE" in out["caveat"]


def test_text_on_a_network_layer_neither_joins_a_chain_nor_becomes_one():
    """A label sitting between two roads must not weld them together.

    Its bbox corner can fall within a snap of both, and if a TEXT were fed to
    union-find as path geometry the two roads would be reported as one chain —
    and the label itself as a third. Both are wrong, and both were live before
    this test existed.
    """
    rows = [
        line("west", E0, N0, E0 + 30.0, N0),
        line("east", E0 + 31.0, N0, E0 + 61.0, N0),
        {
            "handle": "label",
            "type": "TEXT",
            "text": "2092",
            "length": None,
            "bbox": {"min": [E0 + 30.0, N0], "max": [E0 + 31.0, N0 + 1.0]},
        },
    ]

    out = dn.network_measure(rows, unit="m")

    assert out["chains"] == 2
    assert out["non_path_entities"] == 1
    assert out["chained_entities"] == 2
    assert out["entities"] == 3
    assert "not path geometry" in out["caveat"]


def test_an_entity_with_no_bbox_is_excluded_and_the_exclusion_is_published():
    rows = chain_of(2, at_x=E0, at_y=N0)
    rows.append({"handle": "ghost", "type": "LINE", "length": 12.0, "bbox": None})

    out = dn.network_measure(rows, unit="m")

    assert out["unplaced_entities"] == 1
    assert out["chained_entities"] == 2
    assert out["entities"] == 3
    # It carries a length, so it is still in the sum.
    assert out["measured_entities"] == 3
    assert out["length_total"] == pytest.approx(72.0)
    assert out["chains_bound"] == "approximate"
    assert "take no part" in out["caveat"] or "Excluded:" in out["caveat"]


# --------------------------------------------------------------------------
# absence, in each of the shapes it really arrives in (rule G8)
# --------------------------------------------------------------------------


def test_a_layer_where_nothing_carries_a_length_gets_null_and_not_zero():
    rows = [
        {
            "handle": f"t{i}",
            "type": "LWPOLYLINE",
            "length": None,
            "bbox": {"min": [E0 + i, N0], "max": [E0 + i + 1.0, N0 + 1.0]},
        }
        for i in range(3)
    ]

    out = dn.network_measure(rows, unit="m")

    assert out["length_total"] is None
    assert out["measured_entities"] == 0
    assert out["unmeasured_entities"] == 3
    assert "not a measurement of zero" in out["caveat"]
    # Connectivity is still computable — the two questions are independent.
    assert out["chains"] is not None
    assert out["snap_tolerance"] > 0.0


def test_a_length_of_zero_is_a_measurement_and_a_nan_is_not():
    rows = [
        line("a", E0, N0, E0 + 30.0, N0),
        {
            "handle": "zero",
            "type": "LINE",
            "length": 0.0,
            "bbox": {"min": [E0 + 30.0, N0], "max": [E0 + 30.0, N0]},
        },
        {
            "handle": "nan",
            "type": "LINE",
            "length": float("nan"),
            "bbox": {"min": [E0 + 30.0, N0], "max": [E0 + 60.0, N0]},
        },
    ]

    out = dn.network_measure(rows, unit="m")

    assert out["measured_entities"] == 2
    assert out["zero_length_entities"] == 1
    assert out["unmeasured_entities"] == 1  # the NaN, counted and not summed
    assert out["length_total"] == pytest.approx(30.0)


def test_rows_that_are_not_entity_documents_are_counted_not_dropped():
    rows = chain_of(2, at_x=E0, at_y=N0) + ["not a dict", None]

    out = dn.network_measure(rows, unit="m")

    assert out["entities"] == 4
    assert out["malformed_rows"] == 2
    assert out["measured_entities"] == 2
    assert "not entity documents" in out["caveat"]


def test_a_layer_of_nothing_but_labels_cannot_report_chains():
    rows = [
        {
            "handle": f"t{i}",
            "type": "MTEXT",
            "length": None,
            "bbox": {"min": [E0 + i, N0], "max": [E0 + i + 1.0, N0 + 1.0]},
        }
        for i in range(4)
    ]

    out = dn.network_measure(rows, unit="m")

    # Null, not 0: there is no path geometry to have chains, which is a
    # different statement from "the paths here form no chains".
    assert out["chains"] is None
    assert out["chains_bound"] == "not_computed"
    assert "not zero chains" in out["chains_basis"]
    assert out["length_total"] is None


# --------------------------------------------------------------------------
# properties that must survive the 19th drawing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit", ["m", "in", "ft", "mm", None], ids=["m", "in", "ft", "mm", "none"]
)
def test_the_contract_keys_are_always_present_whatever_the_unit(unit):
    required = (
        "length_total", "measured_entities", "unmeasured_entities", "chains",
        "snap_tolerance", "tolerance_basis", "caveat",
    )
    for rows in ([], chain_of(2, at_x=E0, at_y=N0)):
        out = dn.network_measure(rows, unit=unit)
        for key in required:
            assert key in out, f"{key} missing for unit={unit!r}"
        assert out["unit"] == unit
        assert isinstance(out["caveat"], str)
        assert isinstance(out["tolerance_basis"], str) and out["tolerance_basis"]


@pytest.mark.parametrize("origin", [(0.0, 0.0), (E0, N0), (-E0, -N0)])
def test_a_chain_is_the_same_chain_wherever_it_is_drawn(origin):
    """Translation invariance — the defect that only shows at UTM magnitudes."""
    out = dn.network_measure(chain_of(5, at_x=origin[0], at_y=origin[1]), unit="m")

    assert out["chains"] == 1
    assert out["largest_chain"] == 5
    assert out["length_total"] == pytest.approx(150.0)


def test_three_identical_copies_parked_apart_read_as_three_chains():
    """The shape of Janadriyah's triplication, without any of its numbers.

    Not the same instrument as lane 3's duplicate detection — that partitions
    by spatial gap and compares cluster sizes. This only says the copies do not
    touch, which is why the two counts can disagree and neither is wrong.
    """
    rows: list[dict] = []
    for i, offset in enumerate((0.0, 3400.0, 6500.0)):
        rows += [
            line(f"c{i}-{j}", E0 + offset + j * 30.0, N0, E0 + offset + (j + 1) * 30.0, N0)
            for j in range(4)
        ]

    out = dn.network_measure(rows, unit="m")

    assert out["chains"] == 3
    assert out["chain_sizes_top"] == [4, 4, 4]
    assert out["length_total"] == pytest.approx(360.0)


def test_reported_chain_sizes_state_their_own_truncation():
    """Rule G7: a capped list says how much it left out."""
    rows: list[dict] = []
    for i in range(dn.MAX_CHAIN_SIZES_REPORTED + 3):
        rows += chain_of(2, at_x=E0 + i * 5000.0, at_y=N0)

    out = dn.network_measure(rows, unit="m")

    assert out["chains"] == dn.MAX_CHAIN_SIZES_REPORTED + 3
    assert len(out["chain_sizes_top"]) == dn.MAX_CHAIN_SIZES_REPORTED
    assert out["chain_sizes_truncated"] == 3
    assert out["caps"]["max_chain_sizes_reported"] == dn.MAX_CHAIN_SIZES_REPORTED
    assert out["caps"]["hit"] == []


def test_over_the_entity_cap_the_count_is_refused_and_not_truncated(monkeypatch):
    """Rule G7: above a stated limit, no answer plus a suggestion.

    The cap is lowered rather than fed 200,001 rows, because what is being
    tested is the BEHAVIOUR at the limit, not the limit's value.
    """
    monkeypatch.setattr(dn, "MAX_CHAIN_ENTITIES", 3)
    rows = chain_of(4, at_x=E0, at_y=N0)

    out = dn.network_measure(rows, unit="m")

    assert out["chains"] is None
    assert out["chains_bound"] == "not_computed"
    assert "max_entities_for_chains" in out["caps"]["hit"]
    assert "Narrow the input" in out["chains_basis"]
    # Σ length has no cap and is untouched by the refusal.
    assert out["length_total"] == pytest.approx(120.0)
    assert out["measured_entities"] == 4


def test_over_the_comparison_cap_the_count_is_refused_and_not_partial(monkeypatch):
    monkeypatch.setattr(dn, "MAX_POINT_COMPARISONS", 1)
    rows = chain_of(6, at_x=E0, at_y=N0)

    out = dn.network_measure(rows, unit="m")

    assert out["chains"] is None
    assert out["chains_bound"] == "not_computed"
    assert "max_point_comparisons_per_pass" in out["caps"]["hit"]
    assert out["tolerance_sensitivity"] == []
    assert out["length_total"] == pytest.approx(180.0)


def test_every_stated_cap_is_in_the_response():
    out = dn.network_measure(chain_of(2, at_x=E0, at_y=N0), unit="m")

    caps = out["caps"]
    assert caps["max_entities_for_chains"] == dn.MAX_CHAIN_ENTITIES
    assert caps["max_point_comparisons_per_pass"] == dn.MAX_POINT_COMPARISONS
    # Σ length is a single pass and is exact at any size; the response says so
    # rather than leaving a reader to assume a hidden ceiling.
    assert caps["max_entities_for_length"] is None


def test_the_function_does_not_mutate_the_rows_it_is_given():
    rows = chain_of(3, at_x=E0, at_y=N0)
    before = copy.deepcopy(rows)

    dn.network_measure(rows, unit="m")

    assert rows == before
