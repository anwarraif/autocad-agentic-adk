"""`scripts/agent_qa.py` -- unit tests for the GRADING logic only.

Everything here runs on fixtures: hand-written answers graded against
hand-written cases. No network, no SSE, no running stack -- the part of the
harness that talks to `POST /api/agent/stream` is exercised by running the
harness itself, not by this file.

The tests that matter most are the ones written to catch a collapse:

* tolerant number matching must accept every separator and stated rounding of
  a pinned figure (`83,962.565`, `83962.565`, `83,962.57`) and still reject a
  genuinely different number -- a matcher loose enough to pass `84,000` grades
  nothing;
* a NEGATED mention of a forbidden claim must never be a violation ("it is not
  true that there are no roads" contains "no roads"); when in doubt the
  grader prefers MISSING a violation to inventing one;
* the route's zero-tool-calls `ungrounded` label is an automatic FAIL, while a
  refusal (zero tools, no label) is not;
* WEAK is facts-present-caveat-missing and never fatal unless --strict.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "agent_qa.py"

# `scripts/` is not a package, so the module is loaded by path. The skip is not
# politeness: the suite's own runner mounts ONLY `cad_api/` into a one-off
# container, so `scripts/` is genuinely absent there -- and a module-level load
# that raises takes the WHOLE suite down at collection time. A file this module
# cannot reach is a reason to skip this file, never a reason to stop everything
# else from being checked. (Same pattern as test_dossier_eval.py, and for the
# same measured incident.)
qa = None
_why_no_script = ""
if SCRIPT_PATH.is_file():
    _spec = importlib.util.spec_from_file_location("agent_qa", SCRIPT_PATH)
    if _spec and _spec.loader:
        qa = importlib.util.module_from_spec(_spec)
        sys.modules["agent_qa"] = qa
        try:
            _spec.loader.exec_module(qa)
        except Exception as exc:  # pragma: no cover - reported, never silent
            qa = None
            _why_no_script = f"{SCRIPT_PATH} failed to import: {exc!r}"
    else:
        _why_no_script = f"no import spec for {SCRIPT_PATH}"
else:
    _why_no_script = (
        f"{SCRIPT_PATH} is not on this filesystem. The one-off test container "
        "mounts cad_api/ only, so the harness script is out of reach there; "
        "run from a checkout, or mount scripts/ as well."
    )

pytestmark = pytest.mark.skipif(qa is None, reason=_why_no_script)


# --- fixture builders ---------------------------------------------------------


def transcript(**kw):
    defaults = dict(answer="", tools=[], ungrounded=False, errors=[], timed_out=False)
    defaults.update(kw)
    return qa.Transcript(**defaults)


def case(**kw):
    base = {
        "id": "T",
        "question": "q",
        "must_contain_facts": [],
        "must_not_claim": [],
        "required_caveats": [],
        "expected_tools": [],
    }
    base.update(kw)
    return base


ROAD_LENGTH = 83962.565009


# =============================================================================
# Number extraction: separators normalised, neighbours never fused
# =============================================================================


def test_comma_and_space_thousand_separators_are_one_number():
    for text in ["83,962.565 m", "83962.565 m", "83 962.565 m"]:
        tokens = qa.extract_numbers(text)
        assert len(tokens) == 1
        assert tokens[0].value == pytest.approx(83962.565)


def test_adjacent_numbers_are_never_fused_across_punctuation():
    """'2026, 46,754' is two numbers; a greedy separator class fuses them."""
    values = [t.value for t in qa.extract_numbers("in 2026, 46,754 entities")]
    assert values == [2026.0, 46754.0]


def test_decimal_places_are_read_from_the_token():
    assert qa.extract_numbers("27,987.52")[0].decimals == 2
    assert qa.extract_numbers("46,754")[0].decimals == 0


# =============================================================================
# Tolerant matching: every stated rounding counts, nothing else does
# =============================================================================


@pytest.mark.parametrize(
    "text",
    [
        "the total is 83,962.565 m",   # comma separators, dossier precision
        "the total is 83962.565 m",    # bare
        "the total is 83 962.565 m",   # space separators
        "the total is 83,962.57 m",    # rounded to 2 dp
        "the total is 83,963 m",       # rounded to integer
        "the total is 83,962 m",       # truncated to integer (as people write)
        "the total is 83,962.565009 m",  # full precision
    ],
)
def test_stated_roundings_of_the_pinned_length_all_match(text):
    ok, _ = qa.number_in_text(text, ROAD_LENGTH)
    assert ok, text


@pytest.mark.parametrize(
    "text",
    [
        "the total is 84,000 m",  # a different number, not a rounding
        "the total is 8 m",       # magnitude collapse
        "the total is 83,900 m",
        "no numbers here",
    ],
)
def test_numbers_that_are_not_roundings_do_not_match(text):
    ok, _ = qa.number_in_text(text, ROAD_LENGTH)
    assert not ok, text


def test_a_pinned_rounded_figure_accepts_the_tools_fuller_precision():
    """The pin may be the rounded one: 303.33 must accept 303.32602."""
    ok, _ = qa.number_in_text("mean area 303.32602 m2 over 195 plots", 303.33)
    assert ok


def test_integer_facts_demand_equality():
    ok, _ = qa.number_in_text("there are 1,233 entities", 1233, integer=True)
    assert ok
    bad, _ = qa.number_in_text("there are 1,234 entities", 1233, integer=True)
    assert not bad


def test_a_small_integer_needs_its_context_words():
    """A bare '3' is everywhere; '3 identical copies' is the fact."""
    ok, _ = qa.number_in_text(
        "the layer holds 3 identical copies", 3, integer=True,
        context_words=["copies", "cluster"],
    )
    assert ok
    bad, _ = qa.number_in_text(
        "plotted at 1:3 scale on sheet A0", 3, integer=True,
        context_words=["copies", "cluster"],
    )
    assert not bad


def test_an_explicit_tolerance_admits_a_nearby_value():
    ok, _ = qa.number_in_text("about 299.996 m2", 300.0, tolerance=0.5)
    assert ok
    bad, _ = qa.number_in_text("about 310 m2", 300.0, tolerance=0.5)
    assert not bad


# =============================================================================
# Facts: concepts and any-of alternatives
# =============================================================================


def test_a_concept_matches_case_insensitively():
    ok, _ = qa.check_fact(
        {"kind": "any_of", "words": ["parcel"]}, "There are 0 road PARCELS here."
    )
    assert ok


def test_any_of_facts_is_satisfied_by_either_alternative():
    fact = {
        "kind": "any_of_facts",
        "facts": [
            {"kind": "number", "value": 27987.52},
            {"kind": "any_of", "words": ["per copy"]},
        ],
    }
    ok, _ = qa.check_fact(fact, "each copy measures 27,987.52 m")
    assert ok
    ok2, _ = qa.check_fact(fact, "the true length counts one pass per copy")
    assert ok2
    bad, _ = qa.check_fact(fact, "the network is long")
    assert not bad


def test_an_unknown_fact_kind_is_refused_not_ignored():
    with pytest.raises(ValueError):
        qa.check_fact({"kind": "vibes"}, "anything")


# =============================================================================
# Forbidden claims: negation-aware, prefer missing a violation
# =============================================================================

NO_ROADS = {"kind": "phrase_any", "phrases": ["no roads"], "label": "denies roads"}


def test_a_bare_forbidden_phrase_is_a_violation():
    violated, _ = qa.check_forbidden(NO_ROADS, "There are no roads in this drawing.")
    assert violated


def test_a_negated_mention_is_excused_and_says_so():
    violated, detail = qa.check_forbidden(
        NO_ROADS, "It is not true that there are no roads here."
    )
    assert not violated
    assert "excused" in detail


def test_negation_does_not_leak_across_a_sentence_boundary():
    """'not' in the PREVIOUS sentence must not excuse this one."""
    violated, _ = qa.check_forbidden(
        NO_ROADS, "The parcels are not classified. There are no roads."
    )
    assert violated


def test_number_without_flags_the_total_quoted_bare():
    spec = {
        "kind": "number_without",
        "value": ROAD_LENGTH,
        "unless_words": ["copy", "copies", "duplicate"],
    }
    violated, _ = qa.check_forbidden(
        spec, "The road network is 83,962.565 m long."
    )
    assert violated
    excused, _ = qa.check_forbidden(
        spec, "83,962.565 m in total, but that sums all three copies."
    )
    assert not excused
    absent, _ = qa.check_forbidden(spec, "The network is drawn three times.")
    assert not absent


# =============================================================================
# Grading: the three grades and the automatic FAILs
# =============================================================================


def roads_case():
    return case(
        must_contain_facts=[
            {"kind": "any_of", "words": ["parcel"], "label": "frame"},
            {"kind": "number", "value": 1233, "integer": True, "label": "1233"},
        ],
        must_not_claim=[NO_ROADS],
        required_caveats=[
            {"kind": "any_of", "words": ["copies", "duplicate"], "label": "caveat"}
        ],
        expected_tools=["land_use_summary", "measure"],
    )


GOOD_ANSWER = (
    "No road parcels are configured, but the road centreline layer holds "
    "1,233 entities; note the geometry is drawn in three identical copies."
)


def test_a_full_answer_passes():
    g = qa.grade_transcript(
        roads_case(), transcript(answer=GOOD_ANSWER, tools=["land_use_summary"])
    )
    assert g.grade == qa.PASS


def test_facts_present_but_caveat_missing_is_weak():
    answer = "No road parcels are configured; the centreline layer holds 1,233 entities."
    g = qa.grade_transcript(
        roads_case(), transcript(answer=answer, tools=["land_use_summary"])
    )
    assert g.grade == qa.WEAK
    assert g.caveats_missing


def test_a_missing_fact_is_a_fail():
    answer = "No road parcels are configured, drawn in three identical copies."
    g = qa.grade_transcript(
        roads_case(), transcript(answer=answer, tools=["land_use_summary"])
    )
    assert g.grade == qa.FAIL


def test_a_forbidden_claim_is_a_fail_even_with_every_fact_present():
    answer = GOOD_ANSWER + " In short: there are no roads."
    g = qa.grade_transcript(
        roads_case(), transcript(answer=answer, tools=["land_use_summary"])
    )
    assert g.grade == qa.FAIL
    assert g.forbidden_hits


def test_the_ungrounded_label_is_an_automatic_fail():
    """Even a numerically perfect answer: the label means nothing was read."""
    g = qa.grade_transcript(
        roads_case(),
        transcript(answer=GOOD_ANSWER, tools=[], ungrounded=True),
    )
    assert g.grade == qa.FAIL
    assert any("ungrounded" in r for r in g.reasons)


def test_a_timeout_is_a_fail_with_its_reason_named():
    g = qa.grade_transcript(roads_case(), transcript(timed_out=True))
    assert g.grade == qa.FAIL
    assert any("timeout" in r for r in g.reasons)


def test_an_empty_answer_is_a_fail_that_names_the_stream_error():
    g = qa.grade_transcript(
        roads_case(), transcript(answer="", errors=["ANSWER_LOST: no text"])
    )
    assert g.grade == qa.FAIL
    assert any("ANSWER_LOST" in r for r in g.reasons)


def test_none_of_the_expected_tools_cited_is_a_fail():
    g = qa.grade_transcript(
        roads_case(), transcript(answer=GOOD_ANSWER, tools=["search_text"])
    )
    assert g.grade == qa.FAIL
    assert any("expected tools" in r for r in g.reasons)


def test_empty_expected_tools_skips_the_tool_check():
    refusal = case(
        must_contain_facts=[{"kind": "any_of", "words": ["drawing"], "label": "scope"}],
        must_not_claim=[{"kind": "phrase_any", "phrases": ["Paris"], "label": "answers"}],
        expected_tools=[],
    )
    g = qa.grade_transcript(
        refusal,
        transcript(answer="I can only answer questions about this drawing.", tools=[]),
    )
    assert g.grade == qa.PASS


def test_a_refusal_that_answers_anyway_fails():
    refusal = case(
        must_contain_facts=[],
        must_not_claim=[{"kind": "phrase_any", "phrases": ["Paris"], "label": "answers"}],
        expected_tools=[],
    )
    g = qa.grade_transcript(
        refusal, transcript(answer="The capital of France is Paris.", tools=[])
    )
    assert g.grade == qa.FAIL


# =============================================================================
# Retry semantics and the gate
# =============================================================================


def test_the_better_grade_of_two_attempts_stands():
    assert qa.better(qa.FAIL, qa.PASS) == qa.PASS
    assert qa.better(qa.WEAK, qa.FAIL) == qa.WEAK
    assert qa.better(qa.PASS, qa.WEAK) == qa.PASS


def test_the_gate_fails_on_any_fail_and_on_weak_only_under_strict():
    assert qa.gate([qa.PASS, qa.PASS]) == 0
    assert qa.gate([qa.PASS, qa.WEAK]) == 0
    assert qa.gate([qa.PASS, qa.WEAK], strict=True) == 1
    assert qa.gate([qa.PASS, qa.FAIL]) == 1
    assert qa.gate([qa.PASS, qa.FAIL], strict=True) == 1


def test_a_case_result_keeps_both_transcripts_and_grades_by_the_best():
    c = roads_case()
    failed = (transcript(timed_out=True), qa.grade_transcript(c, transcript(timed_out=True)))
    ok_t = transcript(answer=GOOD_ANSWER, tools=["measure"])
    passed = (ok_t, qa.grade_transcript(c, ok_t))
    result = qa.CaseResult(case=c, drawing_name="d.dxf", attempts=[failed, passed])
    assert result.final == qa.PASS
    assert len(result.attempts) == 2
    assert result.best[1].grade == qa.PASS


# =============================================================================
# Drawing selection and runtime derivation (G10): fixtures, no network
# =============================================================================


def listing_fixture():
    def d(_id, name, units, count):
        return {
            "_id": _id,
            "original_filename": name,
            "units_name": units,
            "layouts": [{"name": "Model", "entity_count": count, "is_modelspace": True}],
        }

    return {
        "drawings": [
            d(qa.JANADRIYAH_ID, "JANADRIYAH DMP - 20240506.dxf", "m", 46754),
            d("aaa", "colorwh.dxf", "inch", 36437),
            d("bbb", "arch.dxf", "inch", 1542),
            d("ccc", "tablet.dxf", "unitless", 5412),
            d("ddd", "small.dxf", "inch", 90),
        ]
    }


def test_the_generic_drawing_is_the_largest_inch_one_excluding_colorwh():
    chosen, reason = qa.choose_generic_drawing(listing_fixture())
    assert chosen["_id"] == "bbb"
    assert "arch.dxf" in reason


def test_janadriyah_itself_is_never_the_generic_drawing():
    listing = {"drawings": [listing_fixture()["drawings"][0]]}
    with pytest.raises(RuntimeError):
        qa.choose_generic_drawing(listing)


def test_without_an_inch_drawing_the_largest_remaining_is_taken_and_says_so():
    listing = {
        "drawings": [
            d
            for d in listing_fixture()["drawings"]
            if d["units_name"] != "inch"
        ]
    }
    chosen, reason = qa.choose_generic_drawing(listing)
    assert chosen["_id"] == "ccc"
    assert "no inch drawing" in reason


def detail_fixture():
    return {
        "_id": "bbb",
        "original_filename": "arch.dxf",
        "units_name": "inch",
        "layouts": [{"name": "Model", "is_modelspace": True}],
        "dossier": {
            "available": True,
            "entity_total": 1542,
            "roles": [
                {"role": "network", "entities": 1394},
                {"role": "annotation", "entities": 108},
                {"role": "region", "entities": 9},
            ],
            "layers": [
                {
                    "layer": "Handrail",
                    "layout": "Model",
                    "entities": 269,
                    "role": "network",
                    "measure": {
                        "kind": "length",
                        "value": 5800.004798,
                        "unit": "inch",
                        "measured_entities": 259,
                        "unmeasured_entities": 10,
                        "value_is_floor": True,
                    },
                },
                {
                    "layer": "Small",
                    "layout": "Model",
                    "entities": 3,
                    "role": "network",
                    "measure": {"kind": "length", "value": 1.0, "unit": "inch"},
                },
            ],
        },
    }


def core_fixture():
    return {
        "generic": {
            "cases": [
                {"id": "G01", "derive": "entity_total", "question": "how many?",
                 "expected_tools": ["describe_drawing"]},
                {"id": "G02", "derive": "roles_present", "question": "what is in it?",
                 "expected_tools": ["describe_drawing"]},
                {"id": "G03", "derive": "network_length_unit",
                 "question": "how long is '{layer}' in {layout}?",
                 "expected_tools": ["measure"]},
            ]
        }
    }


def test_generic_facts_come_from_the_dossier_not_from_the_template():
    """Rule G10: no second drawing's numbers live in the question set."""
    cases = {c["id"]: c for c in qa.derive_generic_cases(core_fixture(), detail_fixture())}
    total = cases["G01"]["must_contain_facts"][0]
    assert total["value"] == 1542
    assert "dossier" in cases["G01"]["source"]


def test_the_roles_question_asks_about_the_two_biggest_roles():
    cases = {c["id"]: c for c in qa.derive_generic_cases(core_fixture(), detail_fixture())}
    labels = [f["label"] for f in cases["G02"]["must_contain_facts"]]
    assert len(labels) == 2
    assert any("network" in l for l in labels)
    assert any("annotation" in l for l in labels)


def test_the_measure_question_names_the_biggest_bucket_and_forbids_metres():
    cases = {c["id"]: c for c in qa.derive_generic_cases(core_fixture(), detail_fixture())}
    g03 = cases["G03"]
    assert "'Handrail' in Model" in g03["question"]
    assert g03["must_contain_facts"][0]["value"] == 5800.004798
    assert any(
        "metre" in spec["phrases"] for spec in g03["must_not_claim"]
    ), "an inch drawing answered in metres is the known-wrong reading"
    assert g03["required_caveats"], "10 unmeasured entities demand the FLOOR caveat"


def test_a_drawing_without_a_dossier_refuses_rather_than_inventing():
    detail = detail_fixture()
    detail["dossier"] = {"available": False}
    with pytest.raises(RuntimeError):
        qa.derive_generic_cases(core_fixture(), detail)


# =============================================================================
# The shipped question set itself is well-formed
# =============================================================================


def test_qa_core_json_parses_and_every_case_is_gradable():
    import json

    core_path = REPO_ROOT / "scripts" / "qa_core.json"
    if not core_path.is_file():
        pytest.skip(f"{core_path} not on this filesystem (cad_api/-only mount)")
    core = json.loads(core_path.read_text(encoding="utf-8"))
    ids = []
    for c in core["janadriyah"]["cases"]:
        ids.append(c["id"])
        assert c["question"].strip()
        assert c.get("source"), f"{c['id']} must name where its facts were measured"
        for fact in c.get("must_contain_facts", []) + c.get("required_caveats", []):
            ok, detail = qa.check_fact(fact, "")  # must not raise on any kind
            assert ok is False, detail
        for spec in c.get("must_not_claim", []):
            violated, _ = qa.check_forbidden(spec, "")
            assert violated is False
    for c in core["generic"]["cases"]:
        ids.append(c["id"])
        derive = c.get("derive")
        assert derive in (None, "entity_total", "roles_present", "network_length_unit")
        if derive is None:
            assert "must_contain_facts" in c
            numbers = [
                f
                for f in c["must_contain_facts"]
                if f.get("kind") == "number"
            ]
            assert not numbers, (
                f"{c['id']}: a static generic case may not pin numbers (G10)"
            )
    assert len(ids) == len(set(ids)), "case ids must be unique"


def test_the_janadriyah_golden_numbers_are_the_measured_ones():
    """The pins themselves, cross-checked against the phase-report figures."""
    import json

    core_path = REPO_ROOT / "scripts" / "qa_core.json"
    if not core_path.is_file():
        pytest.skip(f"{core_path} not on this filesystem (cad_api/-only mount)")
    core = json.loads(core_path.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in core["janadriyah"]["cases"]}

    def numbers(case_id):
        out = []
        for f in by_id[case_id]["must_contain_facts"]:
            if f["kind"] == "number":
                out.append(f["value"])
            elif f["kind"] == "any_of_facts":
                out += [s["value"] for s in f["facts"] if s["kind"] == "number"]
        return out

    assert 1233 in numbers("J01.roads_count")
    assert 83962.565009 in numbers("J01.roads_count")
    assert 27987.52 in numbers("J02.road_length_really")
    assert 46754 in numbers("J04.entity_total")
    assert 195 in numbers("J07.vl4_mean_area")
    assert 411 in numbers("J08.drawn_twice")
