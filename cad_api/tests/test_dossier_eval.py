"""`scripts/dossier_eval.py` -- the eval each drawing generates from itself.

Everything here runs on fixtures. No MongoDB, no cad-api, no ezdxf: the
generator is a pure function of a Dossier document and the comparison is a pure
function of two values, which is what makes it possible to assert the rules
below on a laptop and in CI, on a drawing nobody has ingested yet.

The tests that matter most are the ones written to catch a collapse rather than
to confirm a success:

* `test_an_answer_may_never_be_read_from_the_dossier` and its mirror. This
  eval's entire claim is that the expected value and the observed value come
  from different places. If that ever stops being true it stops measuring
  anything, while continuing to print a green report -- so the rule is enforced
  at construction and asserted here in both directions.
* `test_the_two_sides_of_the_file_do_not_touch_each_other_s_machinery`. The
  banner comment is not the guard; this is. It reads the source and checks that
  the expectation half never reaches for `urllib` and the answer half never
  reaches for `pymongo`.
* `test_coverage_expectation_is_re_derived_and_not_copied`. Quoting
  `coverage.accounted` back at the Dossier would be a tautology that passes
  forever, including on the day a bucket goes missing.
* `test_none_never_compares_equal_to_zero`. "Not measured" and "measured as
  nothing" are different answers; an eval that let them match would pass on
  exactly the bug it exists to find.
* `test_a_matching_number_in_the_wrong_unit_still_fails`. 300 m2 and 300 with
  no unit are not the same fact (rule G2).

Drawing-specific strings appear below only as fixtures and as the pinned golden
figures, which rule G1 exempts (`kecuali di berkas config dan di test`).
"""

from __future__ import annotations

import copy
import importlib.util
import os
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "dossier_eval.py"

# `scripts/` is not a package, so the module is loaded by path rather than
# imported. Doing it here keeps the script standalone -- it is run as
# `python scripts/dossier_eval.py`, and nothing about being testable should
# force it into a package it does not otherwise need.
#
# The skip is not politeness. The suite's own runner mounts ONLY `cad_api/`
# into a one-off container, so `scripts/` is genuinely absent there -- and a
# module-level load that raises takes the WHOLE suite down at collection time,
# turning "one eval could not run" into "1,099 tests did not run". A file this
# module cannot reach is a reason to skip this file, never a reason to stop
# everything else from being checked.
de = None
_why_no_script = ""
if SCRIPT_PATH.is_file():
    _spec = importlib.util.spec_from_file_location("dossier_eval", SCRIPT_PATH)
    if _spec and _spec.loader:
        de = importlib.util.module_from_spec(_spec)
        sys.modules["dossier_eval"] = de
        try:
            _spec.loader.exec_module(de)
        except Exception as exc:  # pragma: no cover - reported, never silent
            de = None
            _why_no_script = f"{SCRIPT_PATH} failed to import: {exc!r}"
    else:
        _why_no_script = f"no import spec for {SCRIPT_PATH}"
else:
    _why_no_script = (
        f"{SCRIPT_PATH} is not on this filesystem. The one-off test container "
        "mounts cad_api/ only, so the eval script is out of reach there; run "
        "it from a checkout, or mount scripts/ as well."
    )

if de is None:
    # `pytestmark` skips the TESTS, but module-level code in this file still
    # reads names off `de` while pytest is collecting -- a parametrize list, a
    # tolerance constant -- and an AttributeError there takes the whole suite
    # down again, which is the failure this guard was added to prevent. So a
    # stand-in answers any attribute with something harmless. Nothing runs
    # against it: every test in the file is skipped.
    class _Missing:
        def __getattr__(self, name: str):
            return 0.0

    de = _Missing()  # type: ignore[assignment]
    _SCRIPT_MISSING = True
else:
    _SCRIPT_MISSING = False

pytestmark = pytest.mark.skipif(_SCRIPT_MISSING, reason=_why_no_script)


# --- fixture builders ---------------------------------------------------------
#
# Two drawings that differ in the ways the corpus actually differs: one declares
# metres, the other declares nothing at all. Rule G2 is not a special case here,
# it is half the fixture.


def measure(kind, value, unit, *, unit_reason=None, measured=0, unmeasured=0, **extra):
    block = {
        "kind": kind,
        "value": value,
        "unit": unit,
        "unit_reason": unit_reason,
        "measured_entities": measured,
        "unmeasured_entities": unmeasured,
    }
    block.update(extra)
    return block


def bucket(layer, layout, entities, role, measure_block, **extra):
    block = {
        "layer": layer,
        "layout": layout,
        "entities": entities,
        "role": role,
        "measure": measure_block,
        "types": {},
        "anomalies": [],
    }
    block.update(extra)
    return block


NO_UNIT_REASON = (
    "The caller stated no unit for this layer x layout, so none is claimed. "
    "$INSUNITS describes model space only."
)


def metric_dossier():
    """A drawing in metres, with one bucket of every role and a duplication."""
    return {
        "_id": "aaaa1111",
        "dossier_version": 1,
        "entity_total": 940,
        "layouts": [
            {"name": "Model", "is_modelspace": True, "declared_entities": 940},
            {"name": "[block] stamp", "is_modelspace": False, "is_block": True},
        ],
        "coverage": {"accounted": 940, "total": 940, "complete": True},
        "layers": [
            bucket(
                "Plots",
                "Model",
                500,
                "region",
                measure("area", 158087.090924, "m2", measured=500),
            ),
            bucket(
                "Centreline",
                "Model",
                300,
                "network",
                measure(
                    "length",
                    83962.565009,
                    "m",
                    measured=280,
                    unmeasured=20,
                    value_is_floor=True,
                ),
            ),
            bucket(
                "Trees",
                "Model",
                90,
                "points",
                measure("count", 90, None, unit_reason="A count carries no unit."),
                census={
                    "applicable": True,
                    "blocks": [
                        {"block_name": "palm", "count": 60},
                        {"block_name": "acacia", "count": 25},
                        {"block_name": "olive", "count": 4},
                        {"block_name": "fig", "count": 1},
                    ],
                    "total": 90,
                },
            ),
            bucket(
                "PlotNumber",
                "Model",
                40,
                "annotation",
                measure("count", 40, None, unit_reason="A count carries no unit."),
                census={
                    "applicable": True,
                    "total": 40,
                    "classified": 38,
                    "classes": [
                        {
                            "kind": "numeric_label",
                            "count": 38,
                            "distinct_texts": 30,
                            "range": {
                                "min": 1990.0,
                                "max": 2110.0,
                                "unit": None,
                                "unit_basis": (
                                    "null, and deliberately: the census reads the "
                                    "text as typed and is never told the drawing's "
                                    "unit."
                                ),
                            },
                        }
                    ],
                    "without_text": {"count": 2},
                },
            ),
            bucket(
                "Scratch",
                "Model",
                10,
                "mixed",
                measure(
                    "count",
                    10,
                    None,
                    unit_reason="A count carries no unit.",
                    why_no_native_measure="No type family reached 60.0%.",
                ),
            ),
        ],
    }


def unitless_dossier():
    """A drawing that declares no unit. Nothing may default it to metres."""
    return {
        "_id": "bbbb2222",
        "dossier_version": 1,
        "entity_total": 130,
        "layouts": [{"name": "Model", "is_modelspace": True}],
        "coverage": {"accounted": 130, "total": 130, "complete": True},
        "layers": [
            bucket(
                "Icons",
                "Model",
                100,
                "network",
                measure(
                    "length",
                    117.398207,
                    None,
                    unit_reason=NO_UNIT_REASON,
                    measured=90,
                    unmeasured=10,
                ),
            ),
            bucket(
                "Vport",
                "Model",
                30,
                "region",
                measure("area", 84.0, None, unit_reason=NO_UNIT_REASON, measured=30),
            ),
        ],
    }


def duplicated_dossier():
    """One layer drawn three times, as the reference drawing's roads are."""
    document = metric_dossier()
    document["_id"] = "cccc3333"
    document["layers"][1]["anomalies"] = [
        {
            "kind": "duplicate_clusters",
            "detail": "3 spatially detached clusters hold 100 entities each.",
            "evidence": {
                "groups": [
                    {
                        "members": 3,
                        "entities_each": 100,
                        "entities_involved": 300,
                    }
                ]
            },
        }
    ]
    return document


class FakeStore:
    """A `DossierStore` that holds documents instead of a database.

    Rule G5 in one object: the generator is handed whatever documents exist and
    must produce cases for all of them, so a test can add a nineteenth drawing
    by appending to a list.
    """

    def __init__(self, documents):
        self._documents = [copy.deepcopy(d) for d in documents]

    def dossiers(self, drawing_id=None):
        if drawing_id:
            return [d for d in self._documents if d["_id"] == drawing_id]
        return list(self._documents)

    def drawing_names(self):
        return {d["_id"]: f"{d['_id']}.dxf" for d in self._documents}


class FakeTools:
    """A `ToolClient` whose answers are written by the test, not by a server."""

    def __init__(self, bodies=None, raises=None):
        self.bodies = bodies or {}
        self.raises = raises
        self.asked = []

    def get(self, endpoint, params=None):
        self.asked.append((endpoint, dict(params or {})))
        if self.raises:
            raise self.raises
        return self.bodies.get(endpoint, {})


def all_generated(documents=None):
    documents = documents or [metric_dossier(), unitless_dossier(), duplicated_dossier()]
    cases = []
    for document in documents:
        cases += de.generate_cases(document, f"{document['_id']}.dxf")
    return cases


# =============================================================================
# The rule that makes this an eval rather than a mirror
# =============================================================================


def test_an_answer_may_never_be_read_from_the_dossier():
    with pytest.raises(de.SourceConfusion):
        de.EvalCase(
            case_id="x",
            origin="generated",
            kind="k",
            drawing_id="d",
            drawing_name="d.dxf",
            layout="Model",
            layer="L",
            question="q",
            expected=1,
            expected_unit=None,
            unit_note="NO UNIT",
            comparison="count",
            expected_from="dossier:autocad_dossiers[d].layers[0].entities",
            answer_from="dossier:autocad_dossiers[d].layers[0].entities",
            answer_call={"endpoint": "/whatever"},
        )


def test_an_expectation_may_never_be_read_from_a_tool():
    with pytest.raises(de.SourceConfusion):
        de.EvalCase(
            case_id="x",
            origin="generated",
            kind="k",
            drawing_id="d",
            drawing_name="d.dxf",
            layout="Model",
            layer="L",
            question="q",
            expected=1,
            expected_unit=None,
            unit_note="NO UNIT",
            comparison="count",
            expected_from="tool:GET /drawings/{id}/measure",
            answer_from="tool:GET /drawings/{id}/measure",
            answer_call={"endpoint": "/drawings/d/measure"},
        )


def test_an_answer_call_may_never_target_the_dossier_collection():
    with pytest.raises(de.SourceConfusion):
        de.EvalCase(
            case_id="x",
            origin="generated",
            kind="k",
            drawing_id="d",
            drawing_name="d.dxf",
            layout=None,
            layer=None,
            question="q",
            expected=1,
            expected_unit=None,
            unit_note="NO UNIT",
            comparison="count",
            expected_from="dossier:autocad_dossiers[d].entity_total",
            answer_from="tool:GET /autocad_dossiers",
            answer_call={"endpoint": f"/{de.DOSSIER_COLLECTION}/d"},
        )


def test_every_generated_case_splits_its_two_sources():
    for case in all_generated():
        assert case.expected_from.startswith(de.EXPECTED_PREFIX), case.case_id
        assert case.answer_from.startswith(de.ANSWER_PREFIX), case.case_id
        assert de.DOSSIER_COLLECTION not in case.answer_call["endpoint"], case.case_id


def test_every_golden_case_is_hand_measured_and_answered_by_a_tool():
    cases = de.golden_cases({de.GOLDEN_DRAWING_ID})
    assert cases, "the golden set must exist when its drawing is in the store"
    for case in cases:
        assert case.origin == "golden"
        assert case.expected_from.startswith(de.HANDWRITTEN_PREFIX), case.case_id
        assert case.answer_from.startswith(de.ANSWER_PREFIX), case.case_id


def test_the_two_sides_of_the_file_do_not_touch_each_other_s_machinery():
    """The banner comments are a promise; this is the guard.

    Read the source, cut it at the two banners, and check that the expectation
    half never reaches for HTTP and the answer half never reaches for the
    database. A generator that peeked at a tool response to decide what to
    expect would pass every other test in this file.
    """
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    start = source.index("# EXPECTATION SIDE")
    middle = source.index("# ANSWER SIDE")
    end = source.index("# COMPARISON")

    def code_only(text: str) -> str:
        """Comments are prose about the rule; this checks the rule itself.

        Both banners describe what their half must not do, which means the
        banned words appear in the comments by design. Full-line comments and
        PEP 8 trailing comments are dropped; a banned word hidden inside a
        string literal would survive, and that is the right way round -- a
        false alarm is cheap here and a miss is not.
        """
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("#"):
                continue
            lines.append(line.split("  # ")[0])
        return "\n".join(lines)

    expectation_half = code_only(source[start:middle])
    answer_half = code_only(source[middle:end])

    for banned in ("urllib.request", "urlopen", "ToolClient(", "answer_for("):
        assert banned not in expectation_half, (
            f"the expectation side reaches for {banned!r}; expectations must come "
            "from the Dossier alone"
        )
    for banned in ("pymongo", "MongoClient", "DossierStore", de.DOSSIER_COLLECTION):
        assert banned not in answer_half, (
            f"the answer side reaches for {banned!r}; answers must come from the "
            "tools alone"
        )


# =============================================================================
# Generation: over every drawing, from the store, never hard-coded (G5)
# =============================================================================


def test_cases_are_generated_for_every_drawing_in_the_store():
    documents = [metric_dossier(), unitless_dossier(), duplicated_dossier()]
    store = FakeStore(documents)
    cases, coverage = de.build_all_cases(store, include_golden=False)
    assert {c.drawing_id for c in cases} == {d["_id"] for d in documents}
    assert set(coverage) == {d["_id"] for d in documents}


def test_a_new_drawing_brings_its_own_cases_with_no_code_change():
    """The nineteenth drawing. Nothing about it is known to this file."""
    newcomer = unitless_dossier()
    newcomer["_id"] = "dddd4444"
    before = FakeStore([metric_dossier()])
    after = FakeStore([metric_dossier(), newcomer])
    grew = len(de.build_all_cases(after, include_golden=False)[0]) - len(
        de.build_all_cases(before, include_golden=False)[0]
    )
    assert grew > 0
    cases, _ = de.build_all_cases(after, include_golden=False)
    assert any(c.drawing_id == "dddd4444" for c in cases)


def test_one_drawing_can_be_selected_without_hiding_that_it_was():
    store = FakeStore([metric_dossier(), unitless_dossier()])
    cases, coverage = de.build_all_cases(
        store, drawing_id="bbbb2222", include_golden=False
    )
    assert {c.drawing_id for c in cases} == {"bbbb2222"}
    assert set(coverage) == {"bbbb2222"}


def test_case_ids_are_unique():
    cases = all_generated() + de.golden_cases({de.GOLDEN_DRAWING_ID})
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))


def test_every_role_present_in_a_drawing_yields_at_least_one_question():
    kinds = {c.kind for c in de.generate_cases(metric_dossier(), "a.dxf")}
    assert {
        "coverage_total",
        "region_count",
        "region_area",
        "network_count",
        "network_length",
        "network_floor",
        "points_count",
        "points_block_count",
        "annotation_count",
        "annotation_values",
        "annotation_range",
        "mixed_count",
        "anomaly_scan",
    } <= kinds


def test_a_bucket_with_no_layer_name_is_not_asked_about():
    """An empty layer name makes an ambiguous filter, not a question."""
    document = metric_dossier()
    document["layers"].append(
        bucket("", "Model", 7, "region", measure("area", 1.0, "m2", measured=7))
    )
    cases = de.generate_cases(document, "a.dxf")
    assert all(c.layer != "" for c in cases)


def test_bucket_coverage_counts_what_was_not_covered():
    """A cap that is not reported is a silent truncation (G7)."""
    document = metric_dossier()
    cases = de.generate_cases(document, "a.dxf", bucket_cap=1)
    cov = de.bucket_coverage(document, cases)
    assert cov["buckets"] == 5
    assert cov["buckets_covered"] == 5 - cov["buckets_not_covered"]
    assert cov["buckets_not_covered"] >= 0


def test_the_bucket_cap_is_honoured_per_role():
    document = metric_dossier()
    document["layers"].append(
        bucket("Plots2", "Model", 400, "region", measure("area", 2.0, "m2", measured=400))
    )
    document["layers"].append(
        bucket("Plots3", "Model", 300, "region", measure("area", 3.0, "m2", measured=300))
    )
    document["layers"].append(
        bucket("Plots4", "Model", 200, "region", measure("area", 4.0, "m2", measured=200))
    )
    cases = de.generate_cases(document, "a.dxf", bucket_cap=2)
    counted = [c for c in cases if c.kind == "region_count"]
    assert len(counted) == 2
    # And the two chosen are the largest, not the first two encountered.
    assert {c.layer for c in counted} == {"Plots", "Plots2"}


def test_block_names_are_capped_and_the_biggest_win():
    cases = de.generate_cases(metric_dossier(), "a.dxf", name_cap=2)
    blocks = [c for c in cases if c.kind == "points_block_count"]
    assert [c.expected for c in blocks] == [60, 25]


# =============================================================================
# Units: every figure carries one, and it may legitimately be absent (G2)
# =============================================================================


def test_every_case_states_its_unit_or_states_that_it_has_none():
    for case in all_generated() + de.golden_cases({de.GOLDEN_DRAWING_ID}):
        assert case.unit_note, case.case_id
        if case.expected_unit is None:
            assert "NO UNIT" in case.unit_note or "null" in case.unit_note, (
                f"{case.case_id} has no unit and does not say so"
            )


def test_a_drawing_that_declares_no_unit_is_never_given_metres():
    cases = de.generate_cases(unitless_dossier(), "b.dxf")
    quantities = [c for c in cases if c.kind in ("network_length", "region_area")]
    assert quantities
    for case in quantities:
        assert case.expected_unit is None
        assert case.expected["unit"] is None
        assert NO_UNIT_REASON in case.unit_note
        assert " m" not in case.unit_note.replace("model space", "")


def test_a_metric_drawing_carries_its_unit_into_the_expectation():
    cases = de.generate_cases(metric_dossier(), "a.dxf")
    area = next(c for c in cases if c.kind == "region_area")
    length = next(c for c in cases if c.kind == "network_length")
    assert area.expected_unit == "m2"
    assert length.expected_unit == "m"


def test_a_count_never_pretends_to_have_a_unit():
    for case in all_generated():
        if case.kind.endswith("_count") or case.kind == "coverage_total":
            assert case.expected_unit is None
            assert "NO UNIT" in case.unit_note


# =============================================================================
# Coverage: re-derived, never quoted back
# =============================================================================


def test_coverage_expectation_is_re_derived_and_not_copied():
    document = metric_dossier()
    document["coverage"]["accounted"] = 999999  # a lie inside the Dossier
    case = de.coverage_case(document, "a.dxf")
    assert case.expected == 940, "the expectation must be the sum of the buckets"
    assert case.note and "INTERNAL INCONSISTENCY" in case.note


def test_a_dropped_bucket_changes_the_coverage_expectation():
    document = metric_dossier()
    full = de.coverage_case(document, "a.dxf").expected
    document["layers"].pop()
    assert de.coverage_case(document, "a.dxf").expected < full


def test_coverage_is_answered_by_a_live_count_not_by_the_drawing_document():
    case = de.coverage_case(metric_dossier(), "a.dxf")
    assert "entities/handles" in case.answer_call["endpoint"]
    assert case.answer_call["params"] == {"limit": 1}


# =============================================================================
# The FLOOR caveat
# =============================================================================


def test_network_length_says_floor_when_something_carries_no_length():
    case = next(
        c for c in de.generate_cases(metric_dossier(), "a.dxf")
        if c.kind == "network_length"
    )
    assert "FLOOR" in (case.caveat or "")
    assert "20 of 300" in case.caveat


def test_network_length_says_total_when_nothing_is_missing():
    document = metric_dossier()
    document["layers"][1]["measure"].update(
        {"measured_entities": 300, "unmeasured_entities": 0, "value_is_floor": False}
    )
    case = next(
        c for c in de.generate_cases(document, "a.dxf") if c.kind == "network_length"
    )
    assert "FLOOR" not in (case.caveat or "")
    assert "total rather than a floor" in case.caveat


def test_the_floor_itself_gets_its_own_case():
    case = next(
        c for c in de.generate_cases(metric_dossier(), "a.dxf")
        if c.kind == "network_floor"
    )
    assert case.expected == {"measured_entities": 280, "unmeasured_entities": 20}


# =============================================================================
# Comparison
# =============================================================================


def count_case(expected=5):
    return de.EvalCase(
        case_id="t.count",
        origin="generated",
        kind="region_count",
        drawing_id="d",
        drawing_name="d.dxf",
        layout="Model",
        layer="L",
        question="how many",
        expected=expected,
        expected_unit=None,
        unit_note="NO UNIT: a count.",
        comparison="count",
        expected_from="dossier:autocad_dossiers[d].layers[0].entities",
        answer_from="tool:GET /drawings/{id}/entities/handles",
        answer_call={
            "endpoint": "/drawings/d/entities/handles",
            "extract": "path:total_matches",
        },
    )


def quantity_case(expected, tolerance=de.REL_TOLERANCE):
    return de.EvalCase(
        case_id="t.quantity",
        origin="generated",
        kind="region_area",
        drawing_id="d",
        drawing_name="d.dxf",
        layout="Model",
        layer="L",
        question="how large",
        expected=expected,
        expected_unit=expected.get("unit"),
        unit_note="NO UNIT: none stated." if not expected.get("unit") else "in unit",
        comparison="quantity",
        tolerance=tolerance,
        expected_from="dossier:autocad_dossiers[d].layers[0].measure",
        answer_from="tool:GET /drawings/{id}/measure",
        answer_call={"endpoint": "/drawings/d/measure"},
    )


def test_a_matching_count_passes_and_a_drifting_one_fails():
    assert de.compare(count_case(5), 5)[0] == de.PASS
    status, detail = de.compare(count_case(5), 6)
    assert status == de.FAIL
    assert "+1" in detail


def test_a_missing_count_is_a_failure_and_not_a_zero():
    status, detail = de.compare(count_case(5), None)
    assert status == de.FAIL
    assert "no count at all" in detail


def test_none_never_compares_equal_to_zero():
    assert de._same(None, 0, None) is False
    assert de._same(0, None, None) is False
    assert de._same(None, "", None) is False
    assert de._same(None, None, None) is True


def test_a_matching_number_in_the_wrong_unit_still_fails():
    """300 m2 and 300 in no unit at all are not the same fact (G2)."""
    case = quantity_case({"value": 300.0, "unit": "m2", "measured_entities": 1})
    status, detail = de.compare(
        case, {"value": 300.0, "unit": None, "measured_entities": 1}
    )
    assert status == de.FAIL
    assert "unit" in detail


def test_rounding_does_not_fail_a_quantity_but_a_real_difference_does():
    case = quantity_case({"value": 83962.565009, "unit": "m", "measured_entities": 1191})
    ok, _ = de.compare(
        case, {"value": 83962.56500901, "unit": "m", "measured_entities": 1191}
    )
    assert ok == de.PASS
    # 0.035 m out of 83 km is 4e-7 relative -- inside a 1e-6 tolerance and
    # outside this one. That difference is one short kerb return, and an eval
    # that shrugged at it would shrug at a missing entity.
    bad, _ = de.compare(
        case, {"value": 83962.6, "unit": "m", "measured_entities": 1191}
    )
    assert bad == de.FAIL


def test_a_failure_quotes_the_evidence_the_extractor_carried():
    case = de.EvalCase(
        case_id="t.values",
        origin="generated",
        kind="annotation_values",
        drawing_id="d",
        drawing_name="d.dxf",
        layout="Model",
        layer="0",
        question="what values",
        expected={"entities_with_a_value": 0, "distinct_values": 0},
        expected_unit=None,
        unit_note="NO UNIT: counts of text values.",
        comparison="mapping",
        expected_from="dossier:autocad_dossiers[d].layers[0].census",
        answer_from="tool:GET /drawings/{id}/distinct",
        answer_call={"endpoint": "/drawings/d/distinct"},
    )
    status, detail = de.compare(
        case,
        {
            "entities_with_a_value": 33,
            "distinct_values": 15,
            "evidence_sample": ["'   '", "'      '"],
        },
    )
    assert status == de.FAIL
    assert "sample=" in detail and "'   '" in detail


def range_case():
    return next(
        c for c in de.generate_cases(metric_dossier(), "a.dxf")
        if c.kind == "annotation_range"
    )


def test_values_inside_the_published_range_pass():
    status, detail = de.compare(
        range_case(), {"values": [1990.0, 2050.0, 2110.0], "seen": 3}
    )
    assert status == de.PASS
    assert "[1990.0, 2110.0]" in detail


def test_a_value_outside_the_published_range_fails():
    status, detail = de.compare(range_case(), {"values": [1990.0, 9999.0], "seen": 2})
    assert status == de.FAIL
    assert "9999" in detail


def test_a_range_with_nothing_numeric_to_check_skips_and_says_so():
    status, detail = de.compare(range_case(), {"values": [], "seen": 12})
    assert status == de.SKIP
    assert "12 distinct values" in detail


def anomaly_case():
    return next(
        c for c in de.generate_cases(duplicated_dossier(), "c.dxf")
        if c.kind == "anomaly_duplicates"
    )


def test_duplicate_arithmetic_closes_against_a_live_count():
    status, detail = de.compare(anomaly_case(), 300)
    assert status == de.PASS
    assert "3 copies x 100" in detail


def test_more_duplicates_than_the_layer_holds_is_a_failure():
    status, detail = de.compare(anomaly_case(), 12)
    assert status == de.FAIL
    assert "counts only 12" in detail


def test_the_dossiers_own_duplicate_arithmetic_is_checked():
    document = duplicated_dossier()
    group = document["layers"][1]["anomalies"][0]["evidence"]["groups"][0]
    group["entities_involved"] = 999
    case = next(
        c for c in de.generate_cases(document, "c.dxf")
        if c.kind == "anomaly_duplicates"
    )
    status, detail = de.compare(case, 100000)
    assert status == de.FAIL
    assert "does not close" in detail


def scan_case(document=None):
    return next(
        c
        for c in de.generate_cases(document or metric_dossier(), "a.dxf")
        if c.kind == "anomaly_scan"
    )


def test_two_quiet_instruments_are_a_clean_bill_of_health():
    status, detail = de.compare(
        scan_case(), {"groups_total": 0, "entities_involved": 0, "targets_considered": 40}
    )
    assert status == de.PASS
    assert "checked and clean" in detail


def test_two_loud_instruments_agree():
    status, _ = de.compare(
        scan_case(duplicated_dossier()),
        {"groups_total": 4, "entities_involved": 9, "targets_considered": 40},
    )
    assert status == de.PASS


def test_instruments_that_disagree_are_a_finding_and_not_a_failure():
    """`find_duplicates` sees closed rings; the Dossier also sees open paths.

    A difference between them says which instrument is blind where. Grading it
    FAIL would make the gate permanently red for a difference nobody intends to
    remove, and a gate like that is one people stop reading.
    """
    status, detail = de.compare(
        scan_case(duplicated_dossier()),
        {"groups_total": 0, "entities_involved": 0, "targets_considered": 40},
    )
    assert status == de.FINDING
    assert "disagree" in detail


def test_the_cross_instrument_case_publishes_the_caveat_that_explains_it():
    assert "TWO DIFFERENT INSTRUMENTS" in scan_case().caveat


def test_the_scan_expectation_is_scoped_to_the_layout_the_tool_is_asked_about():
    """A count of the whole file against a count of one layout is not a check.

    The Dossier profiles every layout including block definitions;
    `find_duplicates` takes one layout and skips block definitions. Comparing
    the two unscoped manufactured disagreements out of nothing but the scope --
    a figure held up against a figure of something else, which is the exact
    defect this campaign keeps finding. The out-of-scope buckets are counted
    and named instead of being dropped.
    """
    document = duplicated_dossier()
    document["layers"].append(
        bucket(
            "Stamp",
            "[block] stamp",
            60,
            "region",
            measure("area", 5.0, None, unit_reason=NO_UNIT_REASON, measured=60),
            anomalies=[
                {
                    "kind": "duplicate_clusters",
                    "detail": "two copies",
                    "evidence": {
                        "groups": [
                            {"members": 2, "entities_each": 30, "entities_involved": 60}
                        ]
                    },
                }
            ],
        )
    )
    case = scan_case(document)
    assert case.layout == "Model"
    assert case.expected == {"layers_flagged": 1}, "the block bucket must not count"
    assert case.note and "[block] stamp" in case.note
    assert "OTHER layouts" in case.note


# =============================================================================
# Extractors
# =============================================================================


def test_the_two_vocabularies_for_an_unmeasured_entity_are_mapped_in_one_place():
    """The tool says `unmeasurable`, the Dossier says `unmeasured`."""
    got = de.ex_length_floor({"measured_entities": 1191, "unmeasurable_entities": 42})
    assert got == {"measured_entities": 1191, "unmeasured_entities": 42}


def test_a_quantity_extractor_reads_the_unit_from_the_tool_not_from_the_case():
    body = {
        "sum_measured_only": 12.5,
        "units": {"length_unit": None, "area_unit": None},
        "measured_entities": 3,
    }
    assert de.ex_length_quantity(body)["unit"] is None
    assert de.ex_area_quantity(body)["unit"] is None


def test_the_residual_extractor_survives_a_use_row_that_is_not_there():
    got = de.ex_road_residual({"uses": []})
    assert got["layer"] is None and got["parcels"] is None


def test_numeric_values_ignores_what_cannot_be_a_number():
    got = de.ex_numeric_values(
        {"most_common": [{"value": "2,110"}, {"value": "n/a"}, {"value": " 12.5 "}],
         "distinct_values": 3}
    )
    assert got["values"] == [2110.0, 12.5]
    assert got["seen"] == 3


def test_dig_returns_none_rather_than_raising_on_a_missing_path():
    assert de._dig({"a": {"b": [1, 2]}}, "a.b[1]") == 2
    assert de._dig({"a": {}}, "a.b.c") is None
    assert de._dig({"a": [1]}, "a[9]") is None


# =============================================================================
# Running and gating
# =============================================================================


def test_a_dead_tool_is_recorded_as_an_error_not_as_a_wrong_answer():
    result = de.run_case(count_case(5), FakeTools(raises=de.ToolError("connection refused")))
    assert result.status == de.ERROR
    assert "connection refused" in result.detail


def test_a_case_runs_the_call_it_declared():
    tools = FakeTools({"/drawings/d/entities/handles": {"total_matches": 5}})
    result = de.run_case(count_case(5), tools)
    assert result.status == de.PASS
    assert tools.asked == [("/drawings/d/entities/handles", {})]


def test_the_exit_code_gates_on_failures_and_errors():
    ok = [de.CaseResult(count_case(), de.PASS)]
    findings = ok + [de.CaseResult(count_case(), de.FINDING)]
    broken = ok + [de.CaseResult(count_case(), de.FAIL)]
    errored = ok + [de.CaseResult(count_case(), de.ERROR)]
    assert de.gate(ok) == 0
    assert de.gate(findings) == 0
    assert de.gate(findings, strict=True) == 1
    assert de.gate(broken) == 1
    assert de.gate(errored) == 1


def test_the_report_names_the_two_sources_and_never_prints_a_secret(capsys):
    tools = FakeTools({"/drawings/d/entities/handles": {"total_matches": 5}})
    results = [de.run_case(count_case(5), tools)]
    de.print_report(results, {}, None, "http://localhost:4311")
    printed = capsys.readouterr().out
    assert de.DOSSIER_COLLECTION in printed
    assert "http://localhost:4311" in printed
    assert "MONGODB_URI" not in printed
    assert "mongodb+srv" not in printed and "mongodb://" not in printed


def test_the_report_totals_the_cases_that_carry_no_unit(capsys):
    tools = FakeTools({"/drawings/d/entities/handles": {"total_matches": 5}})
    de.print_report([de.run_case(count_case(5), tools)], {}, None, "http://x")
    printed = capsys.readouterr().out
    assert "carries NO unit" in printed


# =============================================================================
# The golden set -- hand-written, and the reason anyone believes the rest
# =============================================================================


def test_the_golden_set_skips_loudly_when_its_drawing_is_not_in_the_store():
    assert de.golden_cases(set()) == []
    assert de.golden_cases({"some-other-drawing"}) == []


def test_the_golden_set_carries_the_road_residual_the_campaign_started_for():
    cases = {c.case_id: c for c in de.golden_cases({de.GOLDEN_DRAWING_ID})}
    length = cases["golden.roads.length"]
    assert length.expected["total_matches"] == 1233
    assert length.expected["value"] == 83962.565009
    assert length.expected["unit"] == "m"
    assert "FLOOR" in length.caveat

    duplicates = cases["golden.roads.duplicate_clusters"]
    assert duplicates.expected["clusters"] == 3
    assert duplicates.expected["entities_per_cluster"] == 411

    residual = cases["golden.roads.residual"]
    assert residual.question == "How many roads are there?"
    assert residual.expected["parcels"] == 0
    assert residual.expected["entities"] == 1233
    assert residual.expected["role"] == "network"


def test_the_golden_set_resolves_plot_2092_to_a_parcel_and_not_to_its_text():
    cases = {c.case_id: c for c in de.golden_cases({de.GOLDEN_DRAWING_ID})}
    search = cases["golden.plot2092.search"]
    assert search.expected["total_matches"] == 2
    assert search.expected["handles"] == ["2600C0A", "27321CB"]
    assert "block definition" in search.caveat

    parcel = cases["golden.plot2092.parcel"]
    assert parcel.expected["handle"] == "205C7CC"
    assert parcel.expected["layer"] == "VL4"
    assert parcel.expected["area"] == 300.0
    assert parcel.expected["area_unit"] == "m2"
    assert "SMALLEST container" in parcel.caveat


def test_the_relayed_golden_case_admits_that_it_is_a_relay():
    """`land_use_summary` builds its residual FROM the Dossier.

    Checking it proves the relay works, not that the measurement is right. The
    case has to say so, or it would read as a second independent confirmation
    of a figure that was only ever measured once.
    """
    residual = next(
        c for c in de.golden_cases({de.GOLDEN_DRAWING_ID})
        if c.case_id == "golden.roads.residual"
    )
    assert "RELAY, NOT MEASUREMENT" in (residual.note or "")
    independent = next(
        c for c in de.golden_cases({de.GOLDEN_DRAWING_ID})
        if c.case_id == "golden.roads.length"
    )
    assert "/measure" in independent.answer_call["endpoint"]


def test_generated_and_golden_cases_are_told_apart():
    cases = all_generated() + de.golden_cases({de.GOLDEN_DRAWING_ID})
    assert {c.origin for c in cases} == {"generated", "golden"}
    for case in cases:
        if case.case_id.startswith("golden."):
            assert case.origin == "golden"
        else:
            assert case.origin == "generated"


def test_an_unknown_origin_is_refused():
    with pytest.raises(ValueError):
        de.EvalCase(
            case_id="x",
            origin="vibes",
            kind="k",
            drawing_id="d",
            drawing_name="d.dxf",
            layout=None,
            layer=None,
            question="q",
            expected=1,
            expected_unit=None,
            unit_note="NO UNIT",
            comparison="count",
            expected_from="dossier:autocad_dossiers[d].entity_total",
            answer_from="tool:GET /x",
            answer_call={"endpoint": "/x"},
        )


# =============================================================================
# Against the real store and the real tools -- opt in, and skipped by name
# =============================================================================

LIVE = os.environ.get("DOSSIER_EVAL_LIVE") == "1"
live_only = pytest.mark.skipif(
    not LIVE,
    reason=(
        "needs the real store and a running cad-api; set DOSSIER_EVAL_LIVE=1 to "
        "run it. `pytest -rs` names what was skipped -- read that before "
        "trusting a green run"
    ),
)


@live_only
def test_live_cases_are_generated_for_every_drawing_the_store_holds():
    de.load_env()
    store = de.DossierStore()
    documents = store.dossiers()
    assert documents, "no Dossier documents -- has the backfill run?"
    cases, coverage = de.build_all_cases(store)
    covered = {c.drawing_id for c in cases if c.origin == "generated"}
    assert covered == {str(d["_id"]) for d in documents}
    assert len(coverage) == len(documents)


@live_only
def test_live_the_golden_road_and_plot_answers_still_hold():
    de.load_env()
    store = de.DossierStore()
    ids = {str(d["_id"]) for d in store.dossiers()}
    cases = de.golden_cases(ids)
    if not cases:
        pytest.skip("the golden drawing is not in this store")
    tools = de.ToolClient()
    for case in cases:
        result = de.run_case(case, tools)
        assert result.status == de.PASS, f"{case.case_id}: {result.detail}"
