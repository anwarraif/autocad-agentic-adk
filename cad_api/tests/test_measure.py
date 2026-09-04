"""Acceptance tests for the measurement endpoint.

Two halves, deliberately separated.

The first half is pure: the sentence a measurement reports itself with, and
the unit naming behind it, are ordinary functions over a dict and are tested
as such. This is the house style everywhere else in `tests/` -- build the
input, assert the output, no server involved.

The second half is not, and that is a departure worth naming. Nothing else in
this suite touches MongoDB: `test_region.py` re-implements the query filter in
Python rather than run one, and `test_extract.py` says outright that in-memory
documents "cannot drift when the sample files change". That convention is a
good one and it cannot cover what these tests are for. The figure this feature
exists to produce -- 133 objects, 10084.024105 metres on one layer -- is a
property of ingested data, and a test that mocks the aggregation away would
pass while the aggregation was wrong. So these run against the live database
and skip, loudly and by name, when it is absent.

The skip is the compromise, and it is a real one: a suite that goes green on a
machine with no database has proved less than it appears to. `pytest -rs` lists
what was skipped; read it before believing a green run.
"""

from __future__ import annotations

import math
from pathlib import Path

import ezdxf
import pytest

from app import extract, store


# ---------------------------------------------------------------------------
# Pure: how a measurement describes itself
# ---------------------------------------------------------------------------


def _result(**over):
    """A measurement result with sensible defaults, for statement tests."""
    base = {
        "measure": "length",
        "units": {
            "name": "m",
            "declared_in_file": True,
            "length_unit": "m",
            "area_unit": "m2",
        },
        "scope_label": "layout=Model AND layer=VL2",
        "total_matches": 133,
        "measured_entities": 133,
        "unmeasurable_entities": 0,
        "sum_measured_only": 10084.024105,
        "total_for_all_matched": 10084.024105,
        "unmeasurable_by_type": [],
        "warnings": [],
    }
    base.update(over)
    return base


def test_unit_names_reads_the_declared_unit():
    # `space` joined this dict when units became layout-aware: a unit is only
    # meaningful for a named space, and "drawing" is what you get for a call
    # that names none.
    assert store._unit_names({"units_name": "m", "units_code": 6}) == {
        "name": "m",
        "declared_in_file": True,
        "length_unit": "m",
        "area_unit": "m2",
        "space": "drawing",
    }


def test_unit_names_refuses_to_invent_a_unit():
    """Three of the ingested drawings declare nothing. Calling those metres
    would be a wrong number wearing a right one's clothes."""
    for drawing in ({"units_name": "unitless"}, {}, {"units_name": None}):
        units = store._unit_names(drawing)
        assert units["name"] == "unitless"
        assert units["declared_in_file"] is False
        assert units["length_unit"] is None
        assert units["area_unit"] is None


def test_statement_carries_number_unit_scope_and_denominator():
    said = store._measure_statement(_result())
    assert "10084.024105" in said
    assert " m," in said
    assert "layout=Model AND layer=VL2" in said
    # The denominator is the point of the sentence.
    assert "133 of 133" in said


def test_statement_marks_a_partial_sum_as_partial():
    """The failure this guards: a partial figure narrated as a total. It must
    be impossible to quote this sentence and sound complete."""
    said = store._measure_statement(
        _result(
            measured_entities=40,
            unmeasurable_entities=93,
            sum_measured_only=1842.5,
            total_for_all_matched=None,
        )
    )
    assert said.startswith("PARTIAL")
    assert "only 40" in said
    assert "not counted as zero" in said


def test_a_partial_sum_says_how_much_is_really_missing():
    """The sentence used to end "the true length is unknown and larger" every
    time. On model space that warns about 16,129 skipped entities when 15,829 of
    them are DIMENSION, TEXT and HATCH - types with no length to have. Only 300
    bulged polylines are length that exists and was not computed.

    Overstating a gap spends the credibility a real warning needs, and a test
    was pinning the overstatement in place.
    """
    said = store._measure_statement(
        _result(
            measured_entities=40,
            unmeasurable_entities=93,
            sum_measured_only=1842.5,
            total_for_all_matched=None,
            unmeasurable_by_type=[
                {"type": "TEXT", "count": 90},
                {"type": "LWPOLYLINE", "count": 3},
            ],
        )
    )
    assert "90 are types that carry no length at all" in said
    assert "3 could have and did not" in said
    assert "larger by that much and no more" in said


def test_a_partial_sum_missing_nothing_says_so_plainly():
    """When every skipped entity is a type with no length, nothing is missing."""
    said = store._measure_statement(
        _result(
            measured_entities=40,
            unmeasurable_entities=93,
            sum_measured_only=1842.5,
            total_for_all_matched=None,
            unmeasurable_by_type=[{"type": "TEXT", "count": 93}],
        )
    )
    assert "not missing anything they could have contributed" in said
    assert "unknown and larger" not in said


def test_a_quotable_total_carries_its_own_high_severity_caveat():
    """A CIRCLE length total is a sum of circumferences. The instruction says to
    quote `statement` and says nothing about carrying `warnings`, so the caveat
    has to be inside the sentence or it is optional."""
    said = store._measure_statement(
        _result(
            warnings=[{
                "code": "CIRCUMFERENCE_IN_LENGTH_TOTAL",
                "severity": "high",
                "message": "CIRCLE entities matched.",
            }],
        )
    )
    assert "CAVEAT:" in said
    assert "CIRCLE entities matched." in said


def test_statement_says_drawing_units_when_none_are_declared():
    said = store._measure_statement(
        _result(
            units={
                "name": "unitless",
                "declared_in_file": False,
                "length_unit": None,
                "area_unit": None,
            }
        )
    )
    assert "drawing units" in said
    assert " m," not in said


def test_statement_distinguishes_nothing_matched_from_nothing_measurable():
    nothing_matched = store._measure_statement(
        _result(total_matches=0, measured_entities=0, sum_measured_only=None,
                total_for_all_matched=None)
    )
    assert "Nothing matched" in nothing_matched

    nothing_measurable = store._measure_statement(
        _result(measured_entities=0, sum_measured_only=None,
                total_for_all_matched=None, unmeasurable_entities=133)
    )
    assert "none of them carry a length" in nothing_measurable
    # and it must not read as a total of zero
    assert "0 m" not in nothing_measurable


# ---------------------------------------------------------------------------
# Against the live database
# ---------------------------------------------------------------------------

#: The reference drawing and filter, pinned. "133 objects, 10084.02" means
#: nothing without the scope that produced it: the same layer measured across
#: all layouts, or without the type filter, is a different number.
JANADRIYAH = "596212db022a3397"
REF_SCOPE = {"layout_name": "Model", "layer": "VL2", "dxftype": "LWPOLYLINE"}
REF_COUNT = 133
REF_LENGTH = 10084.024105


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


def test_reference_total_is_exact():
    """Acceptance 1. The figure the whole feature exists to produce."""
    _mongo_or_skip()
    r = store.measure_entities(JANADRIYAH, measure="length", **REF_SCOPE)
    assert r["total_matches"] == REF_COUNT
    assert r["measured_entities"] == REF_COUNT
    assert r["unmeasurable_entities"] == 0
    assert r["total_for_all_matched"] == pytest.approx(REF_LENGTH, abs=0.01)
    assert r["units"]["length_unit"] == "m"


def test_the_denominator_is_present_and_is_not_optional():
    """Acceptance 1, the half that matters. A total without the count it was
    summed from is the shape of both wrong numbers this project has shipped."""
    _mongo_or_skip()
    r = store.measure_entities(JANADRIYAH, measure="length", **REF_SCOPE)
    for key in ("measured_entities", "unmeasurable_entities",
                "zero_valued_entities", "measured_fraction"):
        assert key in r, f"{key} missing: the total is unreadable without it"
    assert f"{REF_COUNT} of {REF_COUNT}" in r["statement"]


def test_unmeasurable_filter_returns_null_not_zero():
    """Acceptance 2. TEXT carries no length. A bare 0 would read as 'these
    have no length between them', which is a different and false claim."""
    _mongo_or_skip()
    r = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model", dxftype="TEXT"
    )
    assert r["total_matches"] > 0
    assert r["measured_entities"] == 0
    assert r["sum_measured_only"] is None
    assert r["total_for_all_matched"] is None
    assert "NOTHING_MEASURABLE" in [w["code"] for w in r["warnings"]]


def test_partial_measurement_withholds_the_total():
    """Acceptance 3. A mixed layer where some entities cannot be measured must
    not offer a number that can be mistaken for the whole."""
    _mongo_or_skip()
    r = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model", layer="0"
    )
    assert r["unmeasurable_entities"] > 0
    assert r["measured_entities"] < r["total_matches"]
    assert r["total_for_all_matched"] is None
    assert r["total_for_all_matched_withheld"]
    assert r["sum_measured_only"] is not None
    # and it must say WHAT it skipped
    assert r["unmeasurable_by_type"]
    assert r["statement"].startswith("PARTIAL")


@pytest.fixture
def two_measures_doc(tmp_path: Path) -> Path:
    """One entity with a length and no area, one with both, one with neither.

    Built here rather than taken from the sample data because the point is to
    know the answers independently. Every figure below is arithmetic anyone can
    check on paper, which is what makes a wrong one visible.

    The circle is radius 3 deliberately. At radius 2 its area and its
    circumference are both 4*pi -- numerically identical -- so a test using one
    could not tell the two measures apart at all. That is the exact confusion
    this test exists to catch, and it is easy to build a fixture that hides it.
    """
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 6  # metres
    msp = doc.modelspace()

    # length 10, no area: a straight open segment
    msp.add_line((0, 0), (10, 0), dxfattribs={"layer": "L"})
    # closed 5x5 square: perimeter 20, area 25 -- two different right answers
    msp.add_lwpolyline(
        [(0, 0), (5, 0), (5, 5), (0, 5)], close=True, dxfattribs={"layer": "A"}
    )
    # circumference 6*pi, area 9*pi
    msp.add_circle((20, 20), radius=3, dxfattribs={"layer": "C"})
    # neither
    msp.add_text("NOTE", dxfattribs={"layer": "T"}).set_placement((1, 1))

    path = tmp_path / "two_measures.dxf"
    doc.saveas(path)
    return path


def test_area_and_length_are_computed_from_different_geometry(two_measures_doc: Path):
    """Acceptance 4, rewritten.

    The old version of this test asked for HATCH, on the grounds that a hatch
    has an area and no length. That is not true of this data -- all 720 HATCH
    entities carry neither -- so both sides of the comparison were zero and the
    test passed no matter what the code did. It then asserted
    `result["measure"] == "area"`, which only echoes back the argument it just
    passed in. A mutation making `measure="area"` sum the `length` field
    survived it untouched.

    So it is rebuilt on a document whose answers are known by arithmetic.
    """
    doc = ezdxf.readfile(str(two_measures_doc))
    by_type = {e.dxftype(): e for e in doc.modelspace()}

    line_len, line_area = extract._measure(by_type["LINE"])
    assert line_len == pytest.approx(10.0)
    assert line_area is None, "an open segment encloses nothing"

    poly_len, poly_area = extract._measure(by_type["LWPOLYLINE"])
    assert poly_len == pytest.approx(20.0)   # perimeter
    assert poly_area == pytest.approx(25.0)  # enclosed
    assert poly_len != pytest.approx(poly_area), (
        "the whole point: one shape, two different numbers"
    )

    circ_len, circ_area = extract._measure(by_type["CIRCLE"])
    assert circ_len == pytest.approx(2 * math.pi * 3)
    assert circ_area == pytest.approx(math.pi * 9)

    text_len, text_area = extract._measure(by_type["TEXT"])
    assert text_len is None and text_area is None


def test_area_does_not_secretly_report_length():
    """The same confusion, one layer up, where the aggregation happens.

    The fixture above proves the extractor keeps the two apart. This proves the
    aggregation does too, and it needs real stored data to do it: a mutation
    summing the wrong field is invisible unless the two answers are known and
    far apart. On `BlockBoundary` they are -- 55,862 m of boundary enclosing
    712,977 m2 -- so an area that quietly returns the length is a factor of
    thirteen out and cannot hide in a rounding tolerance.
    """
    _mongo_or_skip()
    scope = {"layout_name": "Model", "layer": "BlockBoundary"}
    length = store.measure_entities(JANADRIYAH, measure="length", **scope)
    area = store.measure_entities(JANADRIYAH, measure="area", **scope)

    assert length["total_for_all_matched"] == pytest.approx(55862.414807, abs=0.01)
    assert area["total_for_all_matched"] == pytest.approx(712977.347656, abs=0.01)
    # and the two must not be the same number by any route
    assert area["total_for_all_matched"] != pytest.approx(
        length["total_for_all_matched"], rel=0.01
    )


def test_empty_filter_is_explained_not_left_bare():
    """Acceptance 5. Zero matches has four causes that look identical."""
    _mongo_or_skip()
    r = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model",
        layer="NO-SUCH-LAYER-EXISTS",
    )
    assert r["total_matches"] == 0
    assert r["why_empty"], "an empty measurement must say why"


def test_a_type_passed_as_a_layer_is_refused():
    """Acceptance 6. The 812-versus-818 defect: this drawing has a layer named
    DIM and a type named DIMENSION, and answering the wrong axis fluently is
    how the wrong number got out."""
    _mongo_or_skip()
    with pytest.raises(store.MeasureRefused) as caught:
        store.measure_entities(
            JANADRIYAH, measure="length", layout_name="Model", layer="DIMENSION"
        )
    assert caught.value.code == "AMBIGUOUS_FILTER_VALUE"
    # A refusal without the corrected call sends the caller back to counting
    # entities one at a time, which is the failure being fixed.
    assert "type=" in caught.value.hint


def test_grouped_totals_cover_everything_even_when_rows_are_capped():
    """A truncated group list must never quietly shrink the grand total."""
    _mongo_or_skip()
    whole = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model"
    )
    capped = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model", group_by="layer", top=3
    )
    assert capped["groups_listed"] == 3
    assert capped["grouping_complete"] is False
    assert capped["truncated_note"]
    assert capped["total_matches"] == whole["total_matches"]
    assert capped["measured_entities"] == whole["measured_entities"]
    # The counts above were all this test checked, which left the thing it is
    # named for unguarded: a mutation recomputing the grand total from the
    # truncated group list shrank model space by 57% and the test passed.
    assert capped["sum_measured_only"] == pytest.approx(
        whole["sum_measured_only"], abs=0.01
    ), "a truncated group list must not shrink the sum"
    assert capped["total_for_all_matched"] == whole["total_for_all_matched"]


def test_a_capped_grouping_keeps_a_real_total_intact():
    """The same guard where the total is a number rather than None.

    The test above runs over all of model space, where TEXT and DIMENSION make
    `total_for_all_matched` null on both sides -- so comparing them proves only
    that None equals None. LINE in model space is fully measurable, has seven
    layers, and truncates at top=2, so here the withheld-total branch is not in
    the way and a real figure has to survive the cap.
    """
    _mongo_or_skip()
    scope = {"layout_name": "Model", "dxftype": "LINE"}
    whole = store.measure_entities(JANADRIYAH, measure="length", **scope)
    capped = store.measure_entities(
        JANADRIYAH, measure="length", group_by="layer", top=2, **scope
    )

    assert whole["unmeasurable_entities"] == 0
    assert whole["total_for_all_matched"] is not None
    assert capped["grouping_complete"] is False
    assert capped["groups_listed"] == 2 < capped["total_groups"]
    assert capped["total_for_all_matched"] == pytest.approx(
        whole["total_for_all_matched"], abs=0.01
    )
    assert capped["total_for_all_matched"] == pytest.approx(60410.511116, abs=0.01)


def test_layout_scope_changes_the_answer():
    """Why `layout` is required rather than optional: the same filter measured
    on a sheet and in model space are different questions."""
    _mongo_or_skip()
    model = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model", layer="0"
    )
    sheet = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="DMP Layout1", layer="0"
    )
    assert model["total_matches"] != sheet["total_matches"]


def test_describe_by_handles_still_works_without_a_stored_selection():
    """A named regression, found by a sweep rather than by this suite.

    `describe_selection` accepts either a stored `selection_id` or a bare list
    of handles. Adding the exact-counts lookup put `stored_counts` inside the
    branch that handles the first case, so the second raised UnboundLocalError
    and returned HTTP 500. It was invisible from the browser: the viewer uses
    the handles path to probe whether a sheet layout can answer a region, and
    a failed probe just looks like a layout that cannot.

    The suite missed it because every existing test went through the stored
    path. This one goes through the other.
    """
    _mongo_or_skip()
    region = store.select_region(
        JANADRIYAH,
        points=[(691100.0, 2749850.0), (691600.0, 2750100.0)],
        kind="rect",
        mode="crossing",
        layout="Model",
    )
    handles = region["handles"][:25]
    assert handles, "the fixture region should match something"

    described = store.describe_handles(JANADRIYAH, handles, sample=0)
    assert described["found"] == len(handles)
    assert described["by_layer"]

# ---------------------------------------------------------------------------
# Empty results must explain themselves
#
# Acceptance 5 above covers a bad layer. These cover the two gaps a sweep found
# after it: a bad block name, and a near-miss list that named the wrong kind of
# thing. Both produced an answer that was confidently, quietly wrong -- which is
# the only failure mode this project actually cares about.
# ---------------------------------------------------------------------------


def test_a_block_that_does_not_exist_is_explained():
    """A misspelled block returned zero with nothing said about why.

    "How many trees are there" answered with a bare 0 reads as "there are no
    trees". This drawing has four tree blocks.
    """
    _mongo_or_skip()
    found = store.query_entities(
        JANADRIYAH, layout_name="Model", block_name="NOSUCHBLOCK", limit=1
    )
    assert found["total_matches"] == 0
    why = store.diagnose_empty_query(
        JANADRIYAH, layout_name="Model", block_name="NOSUCHBLOCK"
    )
    assert why.get("why_empty"), "an unmatched block name must say why"
    assert any("NOSUCHBLOCK" in note for note in why["why_empty"])


def test_a_block_named_in_the_wrong_case_says_so():
    """Block matching is exact. 'tree' finds nothing; 'TREE' finds the block."""
    _mongo_or_skip()
    found = store.query_entities(
        JANADRIYAH, layout_name="Model", block_name="tree", limit=1
    )
    assert found["total_matches"] == 0
    why = store.diagnose_empty_query(
        JANADRIYAH, layout_name="Model", block_name="tree"
    )
    joined = " ".join(why.get("why_empty") or [])
    assert "TREE" in joined
    assert "case" in joined.lower()


def test_near_misses_do_not_report_blocks_as_layouts():
    """`drawing["layouts"]` carries block definitions too, tagged "[block] ".

    Janadriyah has 81 entries there and 3 real layouts. Reporting the other 78
    as layouts is a defect this project already fixed once elsewhere; a search
    for TREE was re-introducing it, and telling the reader that TREE was "a
    layout name" sends them looking in the wrong place entirely.
    """
    _mongo_or_skip()
    r = store.search_text(JANADRIYAH, "TREE")
    near = r.get("near_misses") or {}
    for name in near.get("layouts") or []:
        assert not str(name).startswith("[block]"), f"{name} is a block, not a layout"
    # the blocks belong in the blocks list, and should still be there
    assert any("TREE" in str(b).upper() for b in (near.get("blocks") or []))
    assert "block names" in " ".join(r.get("why_empty") or [])


def test_grouped_rows_declare_what_they_are_ordered_by():
    """Ranked by the measure, not by entity count, and it must say so.

    Asked for "top 20 layers by entity count", the agent answered from this
    list: led by a road layer of 1,233 while DIM, holding 9,738 entities and no
    length at all, sat near the bottom. Every figure was real; the ranking
    answered a different question.
    """
    _mongo_or_skip()
    r = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model",
        group_by="layer", top=5,
    )
    assert "groups_ordered_by" in r
    assert "NOT entity count" in r["groups_ordered_by"]
    # and the ordering really is by the measure, not by count
    sums = [row["sum_length_measured_only"] or 0 for row in r["by_layer"]]
    assert sums == sorted(sums, reverse=True)

# ---------------------------------------------------------------------------
# Units belong to a space, not to a drawing
#
# The defect these exist for: 1,867 border lines on a paper layout were
# reported as "2992.231252 m", complete, with warnings: [] and
# declared_in_file: true. $INSUNITS is a drawing-header field describing MODEL
# space; stamping it on a sheet is out by roughly a thousandfold on an A0 page,
# and every signal built to mark a number untrustworthy said it was fine.
# ---------------------------------------------------------------------------

SHEET = "DMP Layout1"
BLOCK_LAYOUT = "[block] A$C1d2a4b45"


def test_unit_names_refuses_a_unit_for_paper_space():
    """Pure. A sheet has no unit to claim, so none is claimed."""
    drawing = {"units_name": "m", "units_code": 6}
    units = store._unit_names(drawing, SHEET)
    assert units["length_unit"] is None
    assert units["area_unit"] is None
    assert units["declared_in_file"] is False
    assert units["space"] == "paper"
    assert "MODEL space" in units["why_no_unit"]
    # and it must not have quietly guessed millimetres from the page size
    assert "mm" not in units["name"]


def test_unit_names_refuses_a_unit_inside_a_block_definition():
    """Block geometry is scaled by each insert, so one number is many lengths."""
    units = store._unit_names({"units_name": "m", "units_code": 6}, BLOCK_LAYOUT)
    assert units["length_unit"] is None
    assert units["space"] == "block"


def test_unit_names_still_reports_model_space_units():
    """The regression that matters: the fix must not cost the working case."""
    units = store._unit_names({"units_name": "m", "units_code": 6}, "Model")
    assert units["length_unit"] == "m"
    assert units["area_unit"] == "m2"
    assert units["declared_in_file"] is True
    assert units["space"] == "model"


def test_statement_never_stamps_a_drawing_unit_on_sheet_geometry():
    """The quotable sentence is where the wrong unit actually reached a reader."""
    said = store._measure_statement(
        _result(
            units={
                "name": "sheet coordinates",
                "declared_in_file": False,
                "length_unit": None,
                "area_unit": None,
                "space": "paper",
            }
        )
    )
    assert " m," not in said
    assert "sheet units" in said
    assert "NOT a real-world length" in said
    # "drawing units" would invite the reader to look the drawing's units up
    # and apply them, which is the mistake being prevented.
    assert "drawing units" not in said


def test_a_sheet_measurement_carries_a_high_severity_warning():
    """Acceptance for the fix, against the live database."""
    _mongo_or_skip()
    r = store.measure_entities(
        JANADRIYAH, measure="length", layout_name=SHEET, dxftype="LINE"
    )
    assert r["total_matches"] > 0, "the fixture sheet should hold lines"
    assert r["units"]["length_unit"] is None
    codes = [w["code"] for w in r["warnings"]]
    assert "NOT_MODEL_SPACE_UNITS" in codes
    assert [w for w in r["warnings"] if w["code"] == "NOT_MODEL_SPACE_UNITS"][0][
        "severity"
    ] == "high"
    assert " m," not in r["statement"]


def test_model_space_measurement_is_untouched_by_the_sheet_fix():
    """The reference figure must survive the change that fixed the sheet."""
    _mongo_or_skip()
    r = store.measure_entities(JANADRIYAH, measure="length", **REF_SCOPE)
    assert r["units"]["length_unit"] == "m"
    assert r["total_for_all_matched"] == pytest.approx(REF_LENGTH, abs=0.01)
    assert "NOT_MODEL_SPACE_UNITS" not in [w["code"] for w in r["warnings"]]


def test_a_selection_takes_its_units_from_the_rows_it_holds():
    """A region is dragged, not asked for by layout, so the space is read back.

    `describe_selection` reported the drawing's header units regardless of where
    the selection actually was - the same defect as `measure`, one call site
    over, and the reason a sweep was needed rather than a patch.
    """
    _mongo_or_skip()
    sheet_rows = store.query_entities(
        JANADRIYAH, layout_name=SHEET, limit=20
    )
    handles = [e["handle"] for e in sheet_rows["entities"]]
    assert handles, "the sheet should hold entities"

    described = store.describe_handles(JANADRIYAH, handles, sample=0)
    assert described["unit_detail"]["space"] == "paper"
    assert described["unit_detail"]["length_unit"] is None
    assert described["units"] != "m"
    if described["total_length"] is not None:
        assert described["totals_warning"], "a sheet total must say what it is not"


def test_a_selection_spanning_two_spaces_claims_no_unit():
    """Adding a sheet coordinate to a model coordinate is not a length."""
    _mongo_or_skip()
    model_rows = store.query_entities(JANADRIYAH, layout_name="Model", limit=5)
    sheet_rows = store.query_entities(JANADRIYAH, layout_name=SHEET, limit=5)
    handles = [e["handle"] for e in model_rows["entities"]] + [
        e["handle"] for e in sheet_rows["entities"]
    ]
    described = store.describe_handles(JANADRIYAH, handles, sample=0)
    assert described["unit_detail"]["space"] == "mixed"
    assert described["unit_detail"]["length_unit"] is None
    assert "does not produce a length" in described["unit_detail"]["why_no_unit"]

# ---------------------------------------------------------------------------
# What a layout-scoped query cannot see
#
# Entities inside a block *definition* are stored under a pseudo-layout
# "[block] <name>", so layout="Model" never matches them and the answer comes
# back looking complete. Measured on this drawing:
#
#   00_Internal Road       0 in model space, 2,557 inside block definitions
#   00_Prop - Road - CL_   1,233 in model space, 411 more inside definitions
#   VL2                    133 in model space, none hidden
#
# The first is the one that matters: a drafter asking for internal road length
# was told nothing matched, while 24 km of it sat one level down.
# ---------------------------------------------------------------------------

HIDDEN_LAYER = "00_Internal Road"
HIDDEN_ENTITIES = 2557
PARTLY_HIDDEN_LAYER = "00_Prop - Road - CL_"
PARTLY_HIDDEN_ENTITIES = 411


def test_an_empty_layout_result_is_not_reported_as_an_absence():
    """The failure this closes: "nothing matched" as an answer about a layer
    whose geometry is entirely inside a block definition."""
    _mongo_or_skip()
    r = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model", layer=HIDDEN_LAYER
    )
    assert r["total_matches"] == 0
    shadow = r.get("block_definition_shadow")
    assert shadow, "the hidden geometry must be reported"
    assert shadow["entities"] == HIDDEN_ENTITIES
    assert shadow["measurable"] > 0
    assert shadow["blocks"], "say which block definitions they are in"

    codes = [w["code"] for w in r["warnings"]]
    assert "GEOMETRY_INSIDE_BLOCK_DEFINITIONS" in codes

    # and the sentence a reader quotes must carry it, not just the warning
    assert "not an absence" in r["statement"]
    assert str(HIDDEN_ENTITIES) in r["statement"]


def test_a_partial_total_says_how_much_it_could_not_see():
    """1,233 matched and 411 more hidden: the reported total is a third short."""
    _mongo_or_skip()
    r = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model",
        layer=PARTLY_HIDDEN_LAYER,
    )
    assert r["total_matches"] > 0
    shadow = r["block_definition_shadow"]
    assert shadow["entities"] == PARTLY_HIDDEN_ENTITIES
    assert "NOTE:" in r["statement"]
    assert str(PARTLY_HIDDEN_ENTITIES) in r["statement"]
    # it must not pretend the hidden length can be added
    assert "cannot simply be added" in r["statement"]


def test_a_layer_with_nothing_hidden_gains_no_note():
    """The reference figure must stay clean, or the warning becomes noise."""
    _mongo_or_skip()
    r = store.measure_entities(JANADRIYAH, measure="length", **REF_SCOPE)
    assert r.get("block_definition_shadow") is None
    assert "GEOMETRY_INSIDE_BLOCK_DEFINITIONS" not in [
        w["code"] for w in r["warnings"]
    ]
    assert "NOTE:" not in r["statement"]


def test_query_entities_reports_the_same_blindness():
    """One helper, every layout-scoped count. Fixing `measure` alone would have
    left `query_entities` answering the same question wrongly."""
    _mongo_or_skip()
    r = store.query_entities(
        JANADRIYAH, layout_name="Model", layer=HIDDEN_LAYER, limit=1
    )
    assert r["total_matches"] == 0
    assert r["total_in_drawing"] == HIDDEN_ENTITIES
    assert r.get("scope_note"), "an empty layout count must say what it skipped"
    assert str(HIDDEN_ENTITIES) in r["scope_note"]


def test_a_measurement_inside_a_block_definition_has_no_shadow_of_its_own():
    """Asked about a block definition directly, there is nothing further down."""
    _mongo_or_skip()
    r = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="[block] A$C1d2a4b45"
    )
    assert r["total_matches"] > 0
    assert r.get("block_definition_shadow") is None

# ---------------------------------------------------------------------------
# An empty filter value is a question, not a filter
#
# `if layer:` reads "" as "no filter given", so a caller who passed an empty
# layer got the whole layout back under a scope label reading `layout=Model`
# -- 20,334 entities presented as one layer's worth. The two readings differ by
# four orders of magnitude and nothing downstream can tell them apart.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"layer": ""},
    {"dxftype": ""},
    {"block_name": ""},
    {"layer": "   "},          # whitespace cannot match a real layer name
    {"group_by": ""},
])
def test_measure_refuses_an_empty_filter_value(kwargs):
    _mongo_or_skip()
    with pytest.raises(store.MeasureRefused) as caught:
        store.measure_entities(
            JANADRIYAH, measure="length", layout_name="Model", **kwargs
        )
    assert caught.value.code == "EMPTY_FILTER_VALUE"
    assert caught.value.hint, "a refusal must say what to do instead"


@pytest.mark.parametrize("kwargs", [
    {"layer": ""},
    {"dxftype": ""},
    {"block_name": ""},
    {"text_contains": ""},
])
def test_query_entities_refuses_an_empty_filter_value(kwargs):
    """The sweep half: fixing `measure` alone would leave the same hole here."""
    _mongo_or_skip()
    with pytest.raises(store.MeasureRefused) as caught:
        store.query_entities(JANADRIYAH, layout_name="Model", limit=1, **kwargs)
    assert caught.value.code == "EMPTY_FILTER_VALUE"


def test_distinct_values_refuses_an_empty_filter_value():
    _mongo_or_skip()
    with pytest.raises(store.MeasureRefused):
        store.distinct_values(JANADRIYAH, "layer", layout_name="Model", dxftype="")


def test_search_text_refuses_an_empty_query():
    _mongo_or_skip()
    with pytest.raises(store.MeasureRefused):
        store.search_text(JANADRIYAH, "")


def test_a_real_filter_is_unaffected_by_the_guard():
    """The guard must cost nothing on the path that was already correct."""
    _mongo_or_skip()
    r = store.measure_entities(JANADRIYAH, measure="length", **REF_SCOPE)
    assert r["total_for_all_matched"] == pytest.approx(REF_LENGTH, abs=0.01)

# ---------------------------------------------------------------------------
# "Not found" has four meanings
#
# D-069 closed the case of a block that does not exist. The case that actually
# fires is a block that DOES exist with no instance in the layout asked about,
# and that returned zero with nothing said. Asked how many trees a drawing held,
# the answer was "there are none" on a drawing defining four tree blocks.
# ---------------------------------------------------------------------------


def test_a_block_placed_elsewhere_says_where():
    _mongo_or_skip()
    why = store.diagnose_empty_query(
        JANADRIYAH, layout_name="Model", block_name="TREE"
    )
    assert why.get("why_empty"), "an existing-but-absent block must be explained"
    said = why["why_empty_statement"]
    assert "exists" in said
    assert "Model" in said
    # it must name where the insertions actually are
    assert why["why_empty_facts"]["block_instances_by_layout"]
    assert "ADA" in said


def test_a_block_defined_but_never_placed_says_exactly_that():
    """Different from "does not exist", and different from "not in this layout"."""
    _mongo_or_skip()
    why = store.diagnose_empty_query(
        JANADRIYAH, layout_name="Model", block_name="Tree 6"
    )
    said = why["why_empty_statement"]
    assert "never placed" in said
    assert "any layout" in said
    assert why["why_empty_facts"]["block_instances_by_layout"] == {}


def test_an_empty_result_offers_one_quotable_sentence():
    """A list of notes gets summarised, and the summary is where meaning is lost.

    The same reason `measure` carries `statement` (D-066, D-068).
    """
    _mongo_or_skip()
    why = store.diagnose_empty_query(
        JANADRIYAH, layout_name="Model", block_name="NOSUCHBLOCK"
    )
    assert isinstance(why["why_empty_statement"], str)
    assert why["why_empty_statement"].strip()
    for note in why["why_empty"]:
        assert note in why["why_empty_statement"]


def test_related_blocks_are_named_so_a_kind_of_object_can_be_counted():
    """"How many trees" spans four blocks; naming one of them answers less."""
    _mongo_or_skip()
    why = store.diagnose_empty_query(
        JANADRIYAH, layout_name="Model", block_name="TREE"
    )
    related = why["why_empty_facts"].get("related_blocks") or []
    assert any("Tree" in str(n) for n in related)

# ---------------------------------------------------------------------------
# Triage fixes: a name held in the facts block is a name the reader never sees
# ---------------------------------------------------------------------------


def test_a_near_miss_layer_is_named_in_the_sentence_not_only_in_the_facts():
    """The drafter wrote "hatch Green"; the agent queried "Green" and was told
    the layer did not exist. The candidate was already in `why_empty_facts`, where
    nothing reads it. 113 entities went unreported."""
    _mongo_or_skip()
    why = store.diagnose_empty_query(JANADRIYAH, layout_name="Model", layer="Green")
    said = why["why_empty_statement"]
    assert "hatch Green" in said
    assert why["why_empty_facts"]["layer_did_you_mean"]


def test_distinct_values_admits_when_its_ranking_stops():
    """`text` still truncates - 2,861 distinct strings is a different kind of
    list - so it must say the rest are not proven absent."""
    _mongo_or_skip()
    r = store.distinct_values(JANADRIYAH, "text", layout_name="Model", top=20)
    assert r["distinct_values"] > len(r["most_common"])
    order = r["most_common_order"]
    assert "descending" in order
    assert "truncated" in order
    assert "must not be described as absent" in order
    assert "proves nothing" in r["most_common_list_is"]


def test_a_structural_vocabulary_is_returned_complete():
    """The label was not enough on its own.

    Asked for school plot areas, a run called distinct_values(field="layer"),
    did not find `Primary School` among the top 25 of 82, and told the drafter
    the layer did not exist. It holds two entities. The response carried a line
    saying the rest "must not be described as absent" and the run said it
    anyway - so the truncation is removed for the drawing's own vocabulary
    rather than explained. 82 layer names cost about 1.6 kB.
    """
    _mongo_or_skip()
    for field in ("layer", "type", "block_name", "layout"):
        r = store.distinct_values(JANADRIYAH, field, top=5)
        assert len(r["most_common"]) == r["distinct_values"], (
            field + " came back truncated; absence from it would be unprovable"
        )
        # The claim must name the scope it is a claim about: a layout-scoped
        # list cannot see values that live only inside block definitions, and
        # "genuinely absent" full stop was read as absent from the drawing.
        said = r["most_common_list_is"]
        assert "absent FROM THIS SCOPE" in said
        assert "block definitions" in said

    names = {
        v["value"] for v in store.distinct_values(
            JANADRIYAH, "layer", layout_name="Model", top=5
        )["most_common"]
    }
    for school in ("Primary School", "Secondary School", "Private School"):
        assert school in names, school + " must be reachable, it holds entities"

# ---------------------------------------------------------------------------
# What the red team found in the fixes themselves
# ---------------------------------------------------------------------------

#: A drawing with no INSERT entities at all: every block definition in it is
#: unreachable, so model space really is empty.
NO_INSERTS = "f5d017fc710820da"


def test_a_never_inserted_block_is_not_used_to_deny_an_absence():
    """The fix for silent absence became a way to deny a real one.

    61 of 78 block definitions on the reference drawing are never inserted, and
    they hold 11,938 of the 23,439 entities a naive shadow picks up. On
    `title_block-iso.dxf` there are zero INSERTs in the whole file, and the
    sentence said "this is not an absence" over geometry drawn nowhere.
    """
    _mongo_or_skip()
    r = store.measure_entities(NO_INSERTS, measure="length", layout_name="Model")
    assert r["total_matches"] == 0
    shadow = r.get("block_definition_shadow")
    assert shadow, "the drawing does hold geometry inside definitions"
    assert shadow["in_placed_blocks"]["entities"] == 0
    assert shadow["in_never_placed_blocks"]["entities"] > 0
    assert "this IS an absence" in r["statement"]
    assert "never inserted anywhere" in r["statement"]


def test_a_placed_block_still_denies_the_absence():
    """The case the shadow exists for must survive the correction."""
    _mongo_or_skip()
    r = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model", layer=HIDDEN_LAYER
    )
    shadow = r["block_definition_shadow"]
    assert shadow["in_placed_blocks"]["entities"] == HIDDEN_ENTITIES
    assert shadow["in_never_placed_blocks"]["entities"] == 0
    assert "not an absence" in r["statement"]
    assert "ARE placed" in r["statement"]


def test_the_shadow_reports_the_measure_it_was_asked_for():
    """A mutant swapping length for area in the shadow survived the suite.

    The tests asserted the shadow's entity COUNT and never its stored figure -
    and the stored figure is what lands inside the quotable sentence. Swapping
    the two put a 55% error into the flagship number of the decision log with
    everything green.
    """
    _mongo_or_skip()
    by_length = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model", layer=HIDDEN_LAYER
    )["block_definition_shadow"]["in_placed_blocks"]
    by_area = store.measure_entities(
        JANADRIYAH, measure="area", layout_name="Model", layer=HIDDEN_LAYER
    )["block_definition_shadow"]["in_placed_blocks"]

    assert by_length["stored_length"] == pytest.approx(24302.319, abs=0.01)
    assert "stored_area" in by_area
    assert by_area["stored_area"] != pytest.approx(by_length["stored_length"], rel=0.01)


def test_an_area_gap_does_not_blame_open_curves():
    """LINE and ARC enclose nothing, so an area they do not have is not a gap.

    One list served both measures and turned a 547-entity area gap into a
    claimed 1,617 - a 66% overstatement, inside the sentence written to stop
    overstatement.
    """
    said = store._measure_statement(
        _result(
            measure="area",
            measured_entities=40,
            unmeasurable_entities=100,
            sum_measured_only=1842.5,
            total_for_all_matched=None,
            unmeasurable_by_type=[
                {"type": "LINE", "count": 60},
                {"type": "ARC", "count": 30},
                {"type": "LWPOLYLINE", "count": 10},
            ],
        )
    )
    assert "90 are types that carry no area at all" in said
    assert "10 could have and did not" in said


def test_the_partition_adds_up_when_the_type_list_truncates():
    """"X of them are this, Y are that" over a capped list lost the remainder."""
    said = store._measure_statement(
        _result(
            measured_entities=10,
            unmeasurable_entities=100,
            sum_measured_only=5.0,
            total_for_all_matched=None,
            unmeasurable_by_type=[
                {"type": "TEXT", "count": 60},
                {"type": "LWPOLYLINE", "count": 20},
            ],
        )
    )
    assert "20 are of types beyond the 2 listed" in said


def test_every_branch_of_the_statement_carries_its_caveats():
    """The caveat reached only the complete-total branch. A partial measurement
    on a layer holding circles is the ordinary case, and it lost it."""
    partial = store._measure_statement(
        _result(
            measured_entities=40,
            unmeasurable_entities=93,
            sum_measured_only=1842.5,
            total_for_all_matched=None,
            warnings=[{"code": "CIRCUMFERENCE_IN_LENGTH_TOTAL",
                       "severity": "high", "message": "CIRCLE entities matched."}],
        )
    )
    assert "CAVEAT:" in partial

    nothing = store._measure_statement(
        _result(
            measured_entities=0,
            unmeasurable_entities=133,
            sum_measured_only=None,
            total_for_all_matched=None,
            warnings=[{"code": "NOT_MODEL_SPACE_UNITS",
                       "severity": "high", "message": "Sheet coordinates."}],
        )
    )
    assert "CAVEAT:" in nothing


def test_spatial_query_refuses_an_empty_filter():
    """The empty-filter sweep missed this door: 20,331 entities came back under
    a scope label reading `layout=Model`."""
    _mongo_or_skip()
    with pytest.raises(store.MeasureRefused) as caught:
        store.spatial_query(
            JANADRIYAH, bbox=(-1e9, -1e9, 1e9, 1e9), layout_name="Model", layer=""
        )
    assert caught.value.code == "EMPTY_FILTER_VALUE"


def test_the_empty_filter_guard_rejects_things_that_are_not_strings():
    """`0`, `False` and `[]` are dropped by a truthiness check exactly as `""`
    is, and the guard waved all three through."""
    for bad in (False, [], {}, ""):
        with pytest.raises(store.MeasureRefused):
            store._reject_empty_filters(layer=bad)
    # a real value, and a legitimate numeric one, must still pass
    store._reject_empty_filters(layer="VL2", top=0)


def test_a_structural_list_says_which_scope_it_is_complete_for():
    """"Genuinely absent" was true of the scope and read as true of the drawing.
    `00_Internal Road` is absent from the Model layer list and holds 2,557
    entities inside a block definition."""
    _mongo_or_skip()
    r = store.distinct_values(JANADRIYAH, "layer", layout_name="Model", top=5)
    names = {v["value"] for v in r["most_common"]}
    assert HIDDEN_LAYER not in names
    said = r["most_common_list_is"]
    assert "absent FROM THIS SCOPE" in said
    assert "layout=Model" in said

# ---------------------------------------------------------------------------
# Both halves of a set
#
# `distinct_values` groups the entities it can see, so it can only return values
# that appear. Everything a drawing declares and never uses is invisible to it,
# and invisible in a way that reads as non-existent because a complete-looking
# list came back. An agent holding one half reports one half.
#
# These are written against whatever the data holds rather than against named
# values, so they keep meaning if the fixture drawing changes.
# ---------------------------------------------------------------------------


def _absent_half(field, **scope):
    r = store.distinct_values(JANADRIYAH, field, **scope)
    present = {v["value"] for v in r["most_common"]}
    return r, present, r.get("defined_but_not_present")


def test_a_declared_set_returns_the_half_that_does_not_appear():
    """The used half alone is what produced two separate wrong answers."""
    _mongo_or_skip()
    r, present, absent = _absent_half("block_name", layout_name="Model")
    assert present, "the scope should hold some of them"
    assert absent, "the other half must travel in the same response"
    assert absent["count"] > 0
    assert absent["count"] == len(
        [b for b in (store.get_drawing(JANADRIYAH).get("blocks") or [])
         if b not in present]
    )
    # every listed value must say whether it is drawn somewhere else, and where
    for row in absent["values"]:
        assert row["value"] not in present
        assert isinstance(row["appears_elsewhere"], bool)
        assert row["where"], "an absent value must say where it does live"


def test_a_value_defined_and_never_used_is_distinguished_from_one_used_elsewhere():
    """Two different facts, and the answer to "how many are there" needs both.

    Derived from the data rather than named, so this keeps its meaning if the
    fixture drawing is replaced.
    """
    _mongo_or_skip()
    _, _, absent = _absent_half("block_name", layout_name="Model")
    drawn = [r for r in absent["values"] if r["appears_elsewhere"]]
    never = [r for r in absent["values"] if not r["appears_elsewhere"]]
    assert drawn, "this drawing has definitions placed outside model space"
    assert never, "and definitions placed nowhere at all"
    assert any("inserted" in r["where"] for r in drawn)
    assert any("never inserted" in r["where"] for r in never)


def test_a_definition_placed_only_inside_an_unplaced_one_is_not_called_drawn():
    """Reachability, not the presence of an INSERT row.

    A definition inserted only into another definition that is itself never
    placed is not on the drawing, and reporting it as used would be false.
    """
    _mongo_or_skip()
    _, _, absent = _absent_half("block_name", layout_name="Model")
    reachable = store._reachable_blocks(JANADRIYAH)
    for row in absent["values"]:
        assert row["appears_elsewhere"] == (row["value"] in reachable), row["value"]


def test_the_second_half_is_not_specific_to_one_field():
    """`layer` has the same shape and has already failed once: an answer used
    the layers present in one layout and called the rest empty drawing-wide."""
    _mongo_or_skip()
    r, present, absent = _absent_half("layer", layout_name="Model")
    assert absent and absent["count"] > 0
    declared = {
        l["name"] for l in (store.get_drawing(JANADRIYAH).get("layers") or [])
    }
    assert absent["count"] == len(declared - present)
    # a declared layer holding nothing anywhere must not read the same as one
    # that simply is not in this layout
    holds_nothing = [r for r in absent["values"] if not r["appears_elsewhere"]]
    assert holds_nothing
    assert any("holds nothing anywhere" in r["where"] for r in holds_nothing)


def test_a_field_with_no_declared_set_has_no_second_half():
    """A DXF type exists because an entity has it. There is nothing to add."""
    _mongo_or_skip()
    r = store.distinct_values(JANADRIYAH, "type", layout_name="Model")
    assert "defined_but_not_present" not in r


def test_the_note_says_the_listed_half_is_only_half():
    """The note is what a reader acts on, so it carries the shape of the answer."""
    _mongo_or_skip()
    _, _, absent = _absent_half("block_name", layout_name="Model")
    note = absent["note"]
    assert "DEFINED by this drawing" in note
    assert "not absent from the drawing" in note
    assert "both halves" in note

# ---------------------------------------------------------------------------
# A total that is three copies of one answer
#
# Measured on the reference drawing and written down in DOSSIER-BASELINE.md:
# `measure(length, layer='00_Prop - Road - CL_', layout=Model)` returns
# 83,962.565009 m. That layer holds the same estate drawn THREE times -- 3
# clusters of 411 entities, 27,987.522 m each, confirmed against the block
# definition's own stored length. The honest road figure is ~28 km; the tool
# reported ~84 km with nothing to say it was answering a different question,
# and a hand analysis published the tripled number.
#
# The arithmetic was never wrong. That is the whole difficulty: every existing
# guard in this response -- the denominator, the withheld total, the shadow --
# is built to catch an INCOMPLETE sum, and this one is complete.
#
# These run without a database on purpose. The Dossier reader is Mongo-backed
# and is written in its own lane; whether `measure` degrades safely when it is
# absent is a property of this code, not of the data, and a test that could
# only run where a Dossier had already been built would prove it exactly where
# it does not matter.
# ---------------------------------------------------------------------------

ROAD_LAYER = "00_Prop - Road - CL_"
ROAD_TOTAL = 83962.565009
ROAD_CLUSTERS = 3
ROAD_PER_CLUSTER_ENTITIES = 411
ROAD_PER_COPY = 27987.52167

METRE_UNITS = {
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


class _FakeReader:
    """Stands in for the Dossier reader module, which lives in another lane.

    Only the two functions `measure` is allowed to call are implemented, so a
    call to anything else fails here rather than passing on a machine where
    the real module happens to have grown one.
    """

    def __init__(self, record=None, dossier=None, raises=None):
        self.record = record
        self.dossier = dossier
        self.raises = raises

    def duplicate_warning(self, drawing_id, *, layer, layout):
        if self.raises:
            raise self.raises
        return self.record

    def dossier_for(self, drawing_id):
        return self.dossier


def _reader(monkeypatch, reader):
    monkeypatch.setattr(store, "_dossier_reader", lambda: reader)


def _warn(monkeypatch, reader, *, measure="length", units=None,
          total=ROAD_TOTAL, layer=ROAD_LAYER, layout="Model"):
    _reader(monkeypatch, reader)
    return store._duplicate_warning(
        JANADRIYAH,
        layer=layer,
        layout=layout,
        measure=measure,
        units=units or METRE_UNITS,
        sum_measured=total,
    )


def _record(**over):
    """A Dossier duplicate record, shaped as Phase 1's anomaly lane records one."""
    base = {
        "layer": ROAD_LAYER,
        "layout": "Model",
        "clusters": ROAD_CLUSTERS,
        "entities_each": ROAD_PER_CLUSTER_ENTITIES,
        "entities_involved": ROAD_CLUSTERS * ROAD_PER_CLUSTER_ENTITIES,
        "length_total_each": ROAD_PER_COPY,
        "instrument": "spatial_cluster",
    }
    base.update(over)
    return base


def test_a_duplicated_layer_names_its_clusters_and_its_per_copy_figure(monkeypatch):
    """The acceptance case, in the block a reader acts on."""
    block = _warn(monkeypatch, _FakeReader(record=_record()))

    assert block["status"] == "duplicate_clusters"
    assert block["checked"] is True
    assert block["clusters"] == ROAD_CLUSTERS
    assert block["entities_per_cluster"] == ROAD_PER_CLUSTER_ENTITIES
    assert block["entities_involved"] == 1233
    assert block["per_copy"] == pytest.approx(ROAD_PER_COPY, abs=0.01)
    # every figure carries its unit (G2), and says what it is a figure OF
    assert block["unit"] == "m"
    assert " m " in block["per_copy_is"]
    assert "ONE copy" in block["per_copy_is"]
    assert "not this layer's total" in block["per_copy_is"]


def test_the_actionable_sentence_carries_the_counts_and_the_correction(monkeypatch):
    """The sentence is the deliverable. A reader told only "3 clusters" checks
    the sum, finds it correct, and quotes it anyway."""
    block = _warn(monkeypatch, _FakeReader(record=_record()))
    said = block["message"]
    assert "3 spatially detached clusters" in said
    assert "411 entities each" in said
    assert "drawn 3 times" in said
    assert str(ROAD_TOTAL) in said
    # the point that makes it act-on-able rather than merely alarming
    assert "arithmetic is not wrong" in said
    assert "how much is drawn on this layer" in said
    assert "One copy measures" in said
    assert block["hint"], "a warning without a next step sends the reader nowhere"


def test_a_per_copy_figure_is_derived_by_division_when_none_is_recorded(monkeypatch):
    """The Dossier's own per-cluster length is the better reading and is not
    always there. Division is the fallback and must admit what it assumes."""
    block = _warn(monkeypatch, _FakeReader(record=_record(length_total_each=None)))
    assert block["per_copy"] == pytest.approx(ROAD_TOTAL / 3, abs=1e-6)
    assert "divided by 3 clusters" in block["per_copy_is"]
    assert "only if the copies really are identical" in block["per_copy_is"]
    assert "does not prove" in block["per_copy_is"]


def test_a_recorded_per_copy_figure_is_preferred_to_the_division(monkeypatch):
    """Two independent readings agreeing to five decimals is the evidence the
    baseline rests on; quietly recomputing one of them throws that away."""
    block = _warn(
        monkeypatch, _FakeReader(record=_record(length_total_each=12345.6789))
    )
    assert block["per_copy"] == pytest.approx(12345.6789)
    assert "independently of the sum above" in block["per_copy_is"]


def test_an_area_measurement_does_not_borrow_the_recorded_length(monkeypatch):
    """`length_total_each` is a length. Reporting it as an area per copy would
    be the measure confusion this suite already tests one layer down."""
    block = _warn(monkeypatch, _FakeReader(record=_record()), measure="area",
                  total=712977.347656)
    assert block["measure"] == "area"
    assert block["unit"] == "m2"
    assert block["per_copy"] != pytest.approx(ROAD_PER_COPY, rel=0.01)
    assert block["per_copy"] == pytest.approx(712977.347656 / 3, abs=1e-6)


def test_a_missing_dossier_is_not_computed_and_never_reads_as_a_clean_bill(monkeypatch):
    """The failure mode this whole campaign exists to end.

    No Dossier means nobody looked. Reported as "no duplicates" it becomes a
    confident all-clear over an unexamined layer, which is worse than the
    silence it replaced.
    """
    block = _warn(monkeypatch, _FakeReader(record=None, dossier=None))
    assert block["status"] == "not_computed"
    assert block["checked"] is False
    assert block["clusters"] is None
    assert block["reason"]
    assert "NOT COMPUTED" in block["message"]
    assert "not the same answer as 'no duplicates'" in block["message"]
    assert "UNKNOWN" in block["message"]
    assert block["hint"], "say how to compute it"


def test_a_dossier_that_records_nothing_is_a_different_answer(monkeypatch):
    """"Checked and clean" and "not checked" must not collapse into one."""
    block = _warn(
        monkeypatch, _FakeReader(record=None, dossier={"drawing_id": JANADRIYAH})
    )
    assert block["status"] == "no_duplicates_recorded"
    assert block["checked"] is True
    assert "Checked:" in block["message"]
    assert "NOT COMPUTED" not in block["message"]
    # and it must not overclaim: this is one layer in one layout
    assert "says nothing about other" in block["message"]


def test_a_record_that_reports_a_single_cluster_is_clean_not_duplicated(monkeypatch):
    """One cluster is a layer drawn once. Two is the smallest duplicate."""
    for record in (_record(clusters=1), _record(duplicate_clusters=False)):
        block = _warn(monkeypatch, _FakeReader(record=record, dossier={"x": 1}))
        assert block["status"] == "no_duplicates_recorded"
        assert block["per_copy"] is None


def test_a_record_whose_numbers_this_tool_cannot_find_still_warns(monkeypatch):
    """The Dossier reader owns the record's shape and it is written in another
    lane. If it renames its fields, the honest degradation is a warning with
    the numbers missing -- not silence, and not an all-clear. The fields it
    actually supplied are named back so the mismatch is visible."""
    block = _warn(monkeypatch, _FakeReader(record={"copies_found": "three"}))
    assert block["status"] == "duplicate_clusters"
    assert block["clusters"] is None
    assert block["per_copy"] is None
    assert "the cluster count is not recorded" in block["per_copy_is"]
    assert block["dossier_fields"] == ["copies_found"]
    assert block["dossier_fields_total"] == 1
    assert block["message"]


def test_an_absent_reader_module_costs_the_measurement_nothing(monkeypatch):
    """`measure` is a core tool and the enrichment lane ships separately. A
    module that is not there yet must not be able to break it."""
    _reader(monkeypatch, None)
    block = store._duplicate_warning(
        JANADRIYAH, layer=ROAD_LAYER, layout="Model", measure="length",
        units=METRE_UNITS, sum_measured=ROAD_TOTAL,
    )
    assert block["status"] == "not_computed"
    assert "not available in this build" in block["reason"]


def test_the_reader_import_is_guarded_rather_than_assumed():
    """The import itself, not a monkeypatched stand-in for it.

    `_dossier_reader()` is called on a machine where the module may simply not
    exist. Whatever it returns, it must return -- not raise.
    """
    assert store._dossier_reader() is None or hasattr(
        store._dossier_reader(), "duplicate_warning"
    )


def test_a_reader_that_raises_is_not_computed_rather_than_an_error(monkeypatch):
    block = _warn(monkeypatch, _FakeReader(raises=RuntimeError("mongo is down")))
    assert block["status"] == "not_computed"
    assert block["checked"] is False
    assert "lookup failed" in block["reason"]


def test_an_unreadable_record_is_not_computed_rather_than_assumed_clean(monkeypatch):
    """A shape this tool does not understand is not evidence of anything, and
    the safe direction to fail in is "unknown", not "fine"."""
    block = _warn(monkeypatch, _FakeReader(record=["not", "a", "mapping"]))
    assert block["status"] == "not_computed"
    assert "cannot read" in block["reason"]


def test_the_warning_never_stamps_metres_on_a_drawing_that_declares_none(monkeypatch):
    """G2. 11 of the drawings in this store are in inches and 3 declare
    nothing; a per-copy figure labelled "m" would be a wrong number wearing a
    right one's clothes."""
    block = _warn(
        monkeypatch, _FakeReader(record=_record(length_total_each=None)),
        units=NO_UNITS,
    )
    assert block["unit"] is None
    assert block["unit_label"] == "drawing units"
    assert block["unit_basis"]
    assert "drawing units" in block["per_copy_is"]
    assert "drawing units" in block["message"]


def test_the_warning_claims_no_unit_for_sheet_geometry(monkeypatch):
    sheet = dict(METRE_UNITS, length_unit=None, area_unit=None, space="paper")
    block = _warn(monkeypatch, _FakeReader(record=_record()), units=sheet)
    assert block["unit"] is None
    assert block["unit_label"] == "sheet units"


# ---------------------------------------------------------------------------
# The same thing through `measure_entities`, without a database
#
# The block above is a function over a dict. What a caller actually receives is
# a `measure` response, and the two things that matter there -- the block being
# attached to the right kind of answer, and every key that was there before
# still being there -- can only be seen from the whole response.
# ---------------------------------------------------------------------------


class _FakeEntities:
    """Just enough aggregation to run `measure_entities` in memory.

    The `$facet` pipeline is the measurement itself; every other pipeline
    reaching this collection is the block-definition shadow, and returning
    nothing for it means "nothing is hidden", which is the ordinary case.
    """

    def __init__(self, facet):
        self._facet = facet

    def aggregate(self, pipeline, *args, **kwargs):
        if any("$facet" in stage for stage in pipeline):
            return iter([self._facet])
        return iter([])


def _in_memory_measure(monkeypatch, reader, **kwargs):
    facet = {
        "totals": [{
            "_id": None,
            "matched": 1233,
            "measured": 1191,
            "zeros": 0,
            "total": ROAD_TOTAL,
        }],
        # 42 of the 1,233 carry no length -- the baseline's own caveat.
        "skipped": [{"_id": "LWPOLYLINE", "count": 42}],
        "types": [{"_id": "LINE", "count": 900}, {"_id": "ARC", "count": 333}],
    }
    monkeypatch.setattr(store, "coll", lambda *_: _FakeEntities(facet))
    monkeypatch.setattr(store, "get_drawing", lambda _id: {
        "drawing_id": JANADRIYAH,
        "units_name": "m",
        "units_code": 6,
        "layers": [{"name": ROAD_LAYER}, {"name": "VL2"}],
        "counts_by_type": {"LINE": 900, "ARC": 333},
    })
    _reader(monkeypatch, reader)
    call = {"measure": "length", "layout_name": "Model", "layer": ROAD_LAYER}
    call.update(kwargs)
    return store.measure_entities(JANADRIYAH, **call)


def test_measure_attaches_the_duplicate_warning_to_a_layer_total(monkeypatch):
    r = _in_memory_measure(monkeypatch, _FakeReader(record=_record()))
    assert r["sum_measured_only"] == pytest.approx(ROAD_TOTAL)
    block = r["duplicate_warning"]
    assert block["status"] == "duplicate_clusters"
    assert block["clusters"] == ROAD_CLUSTERS
    assert block["total_reported"] == pytest.approx(ROAD_TOTAL)
    assert "sum_measured_only" in block["total_reported_is"]


def test_the_triplication_reaches_the_sentence_a_reader_quotes(monkeypatch):
    """A `severity: high` warning sitting outside `statement` is optional, and
    this project has already shipped a wrong number for exactly that reason."""
    r = _in_memory_measure(monkeypatch, _FakeReader(record=_record()))
    codes = [w["code"] for w in r["warnings"]]
    assert "DUPLICATE_GEOMETRY_IN_TOTAL" in codes
    warning = [
        w for w in r["warnings"] if w["code"] == "DUPLICATE_GEOMETRY_IN_TOTAL"
    ][0]
    assert warning["severity"] == "high"
    assert warning["hint"]
    assert "CAVEAT:" in r["statement"]
    assert "drawn 3 times" in r["statement"]


def test_a_measurement_with_no_duplicates_gains_no_caveat(monkeypatch):
    """Or the warning becomes noise, and noise is how a real one gets ignored."""
    r = _in_memory_measure(
        monkeypatch, _FakeReader(record=None, dossier={"drawing_id": JANADRIYAH})
    )
    assert r["duplicate_warning"]["status"] == "no_duplicates_recorded"
    assert "DUPLICATE_GEOMETRY_IN_TOTAL" not in [w["code"] for w in r["warnings"]]


def test_a_measurement_with_no_dossier_still_says_so_in_the_response(monkeypatch):
    """Not loudly enough to drown the statement, and never silently."""
    r = _in_memory_measure(monkeypatch, _FakeReader(record=None, dossier=None))
    assert r["duplicate_warning"]["status"] == "not_computed"
    assert "DUPLICATE_GEOMETRY_IN_TOTAL" not in [w["code"] for w in r["warnings"]]
    assert "CAVEAT:" not in r["statement"]


def test_a_total_over_no_named_layer_carries_no_duplicate_block(monkeypatch):
    """The Dossier records duplicates per layer; a sum across a whole layout
    has no single layer to be three copies of, so the question is not asked."""
    r = _in_memory_measure(monkeypatch, _FakeReader(record=_record()), layer=None)
    assert "duplicate_warning" not in r


#: Every key a `measure` response carried before the duplicate block existed,
#: with the type the viewer and the agent read it as. Written down rather than
#: derived, because a rename is invisible to a test that reads the response it
#: is checking.
MEASURE_KEYS_BEFORE = {
    "drawing_id": str,
    "measure": str,
    "scope": dict,
    "scope_label": str,
    "units": dict,
    "total_matches": int,
    "measured_entities": int,
    "unmeasurable_entities": int,
    "zero_valued_entities": int,
    "measured_fraction": float,
    "sum_measured_only": float,
    "total_for_all_matched": type(None),
    "total_for_all_matched_withheld": str,
    "unmeasurable_by_type": list,
    "unmeasurable_by_type_order": str,
    "warnings": list,
    "statement": str,
}


@pytest.mark.parametrize("reader_kind", [
    "duplicates", "clean", "no_dossier", "raises", "absent",
])
def test_the_measure_response_keeps_every_key_it_had(monkeypatch, reader_kind):
    """ADDITIVE-ONLY, enforced here as well as by the snapshot gate.

    The viewer and `web/app/api/agent/stream/route.ts` read this response BY
    KEY. A rename is invisible to a suite that asserts on values and fatal on
    screen, so the old surface is pinned by name and by type -- in every state
    the enrichment can be in, including the states where it failed.
    """
    readers = {
        "duplicates": _FakeReader(record=_record()),
        "clean": _FakeReader(record=None, dossier={"drawing_id": JANADRIYAH}),
        "no_dossier": _FakeReader(record=None, dossier=None),
        "raises": _FakeReader(raises=RuntimeError("mongo is down")),
        "absent": None,
    }
    r = _in_memory_measure(monkeypatch, readers[reader_kind])
    for key, kind in MEASURE_KEYS_BEFORE.items():
        assert key in r, key + " was lost from the measure response"
        assert isinstance(r[key], kind), (
            key + " changed type: " + type(r[key]).__name__
            + " is not " + kind.__name__
        )
    # and the figures themselves are untouched by the enrichment
    assert r["total_matches"] == 1233
    assert r["measured_entities"] == 1191
    assert r["sum_measured_only"] == pytest.approx(ROAD_TOTAL)
    assert r["statement"].startswith("PARTIAL")


def test_the_duplicate_block_has_one_shape_whatever_it_found(monkeypatch):
    """A block whose keys appear and disappear with the answer forces every
    reader to guess. The status says what happened; the keys are always there."""
    shapes = []
    for reader in (
        _FakeReader(record=_record()),
        _FakeReader(record=None, dossier={"drawing_id": JANADRIYAH}),
        _FakeReader(record=None, dossier=None),
        None,
    ):
        shapes.append(
            set(_in_memory_measure(monkeypatch, reader)["duplicate_warning"])
        )
    assert all(s == shapes[0] for s in shapes), "the block changes shape"
    for required in ("status", "checked", "clusters", "entities_per_cluster",
                     "per_copy", "per_copy_is", "unit", "unit_label",
                     "message", "hint"):
        assert required in shapes[0]


def test_the_reference_road_layer_carries_its_triplication():
    """Acceptance, against the live store and the real Dossier.

    Skips loudly rather than passing quietly where no Dossier has been built:
    a green run on a machine that never looked has proved nothing.
    """
    _mongo_or_skip()
    r = store.measure_entities(
        JANADRIYAH, measure="length", layout_name="Model", layer=ROAD_LAYER
    )
    assert r["sum_measured_only"] == pytest.approx(ROAD_TOTAL, abs=0.01)
    block = r.get("duplicate_warning")
    assert block, "a layer total must always carry this block"
    if block["status"] == "not_computed":
        pytest.skip("no Dossier built for " + JANADRIYAH + ": " + block["reason"])
    assert block["status"] == "duplicate_clusters"
    assert block["clusters"] == ROAD_CLUSTERS
    assert block["entities_per_cluster"] == ROAD_PER_CLUSTER_ENTITIES
    assert block["per_copy"] == pytest.approx(ROAD_PER_COPY, abs=0.1)
    assert "DUPLICATE_GEOMETRY_IN_TOTAL" in [w["code"] for w in r["warnings"]]
