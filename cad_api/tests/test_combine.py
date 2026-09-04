"""DOSSIER Phase 8b — `combine_findings`, the recipe that composes two others.

Owner: the COMBINE lane.

**The test that matters most is the negative one.** A composition whose side
established nothing must REFUSE, and must not return an empty list. That single
behaviour is the reason this recipe exists: measured five ways over two days,
the agent asked which parcels are both size outliers and without road frontage
answered "there are none" — an empty intersection inherited from a frontage run
that had settled nothing. The answer is one parcel.

Three kinds of test, in this order:

1.  **The composition engine, over side responses written by hand.** The two
    sides are replaced at `_run_recipe`, so the set arithmetic, the extraction
    rules, the caps and the refusals are tested exactly, with no database and
    no dependence on what the reference drawing happens to contain today. What
    is NOT faked is the registry: the recipe names and their parameter
    declarations are read from the live catalogue, so a side named here that
    stopped existing fails this file.
2.  **Refusals over the real parameter surface** — an unknown recipe, a recipe
    whose answer set is ambiguous, a self-reference, an unknown operation, and
    a parameter aimed at a side that does not accept it.
3.  **Janadriyah**, skipped when there is no database. Three numbers computed
    outside this path on 26 August 2026, which are the acceptance criteria for
    this lane.

**One note for the integrator.** Importing this file registers one more recipe,
so the pinned name tuple in `tests/test_recipes.py` no longer matches. That pin
is the integrator's one-line change; this file deliberately does not touch it,
and neither does `app/recipes/combine.py`.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import recipes  # noqa: E402
from app.recipes import combine as cb  # noqa: E402
from app.recipes import registry as rr  # noqa: E402

NAME = cb.RECIPE_NAME

#: The reference drawing, and the layout every recipe here is run against.
DRAWING = "596212db022a3397"
LAYOUT = "Model"

#: The one layer of CLOSED rings on the reference drawing that a frontage check
#: can be measured against exactly. It lives in the TEST and never in the module
#: under test (G1): it names one drawing's right-of-way corridor and means
#: nothing on the next one — and it carries no road word at all, which is why
#: the residual could never find it by name.
CORRIDOR_LAYER = "ROW"

#: Ground truth, computed on 26 August 2026 outside the path these tests take.
GT_OUTLIERS = 30
GT_BOTH_OUTLIER_AND_NO_FRONTAGE = ("205D12B",)
GT_OUTLIER_AND_OVERLAPPING = 16
GT_OVERLAP_SAMPLE = ("205CD54", "220B253", "205D135", "205D12E", "220B274")

#: A drawing id deliberately not in the store: every faked-side test names it,
#: so that a fake that stopped being called could not accidentally reach the
#: real cluster and pass for the wrong reason.
FIXTURE = "fixture000000dead"

METRE_UNITS = {
    "name": "m",
    "declared_in_file": True,
    "length_unit": "m",
    "area_unit": "m2",
    "space": "model",
}


# =============================================================================
# Side responses written by hand
# =============================================================================


def outliers_body(handles, *, truncated=False, why_empty=None, layer="fixture-plots"):
    """What `outliers` publishes, cut down to the keys the rule reads."""
    body = {
        "recipe": "outliers",
        "drawing_id": FIXTURE,
        "layout": LAYOUT,
        "params_used": {"field": "area", "layers": None, "z": 2.0},
        "scope_note": "a hand-written outliers response",
        "field": "area",
        "outliers": [
            {
                "handle": h,
                "layer": layer,
                "type": "LWPOLYLINE",
                "value": 900.0,
                "unit": "m2",
                "z_score": 3.1,
                "direction": "large",
            }
            for h in handles
        ],
        "outliers_total": None if truncated else len(handles),
        "outliers_truncated": truncated,
        "not_measured": "4 objects match this filter but carry no area",
    }
    if why_empty is not None:
        body["outliers"] = []
        body["outliers_total"] = 0
        body["why_empty"] = why_empty
    return body


def frontage_body(handles, *, usable=True, truncated=False, unproven=0):
    """What `frontage_check` publishes, including its safety block."""
    safety = (
        {
            "usable_as_evidence": True,
            "verdict": "ESTABLISHED",
            "why": "2 confirmed without frontage",
            "do_not": "the unproven list is not a finding",
        }
        if usable
        else {
            "usable_as_evidence": False,
            "verdict": "NOTHING ESTABLISHED",
            "why": (
                "all parcels came back unproven: the road geometry supplied "
                "resolves only to bounding boxes"
            ),
            "do_not": "do NOT read `no_frontage.count == 0` as 'every parcel has frontage'",
            "how_to_get_an_answer": (
                "pass `road_layers` naming layers whose entities are CLOSED rings"
            ),
            "candidate_road_layers": [{"layer": CORRIDOR_LAYER, "area": 1.0}],
            "retry_with": [
                {
                    "recipe": "frontage_check",
                    "params": {"road_layers": [CORRIDOR_LAYER]},
                    "because": "the largest untested layer of closed rings",
                }
            ],
        }
    )
    return {
        "recipe": "frontage_check",
        "drawing_id": FIXTURE,
        "layout": LAYOUT,
        "params_used": {"parcel_layers": None, "road_layers": None, "touch_tolerance": None},
        "scope_note": "a hand-written frontage response",
        "frontage_established": 100,
        "frontage_unproven": {"count": unproven, "rows": []},
        "no_frontage": {
            "count": len(handles) + (5 if truncated else 0),
            "returned": len(handles),
            "truncated": truncated,
            "cap": 200,
            "beyond_the_search_window": 0,
            "rows": [
                {
                    "handle": h,
                    "layer": "fixture-plots",
                    "nearest_road": {"value": 12.5, "unit": "m", "method": "ring to ring"},
                    "verdict": "beyond the touch tolerance",
                }
                for h in handles
            ],
        },
        "road_geometry_warning": None,
        "conclusion_safety": safety,
    }


def overlap_body(pairs, *, truncated=False, withheld=0, slivers=0):
    """What `overlap_scan` publishes. `pairs` is a list of two-handle tuples."""
    return {
        "recipe": "overlap_scan",
        "drawing_id": FIXTURE,
        "layout": LAYOUT,
        "params_used": {"layers": None, "boundary_layer": None},
        "scope_note": "a hand-written overlap response",
        "overlaps": {
            "count": len(pairs),
            "returned": len(pairs),
            "truncated": truncated,
            "cap": 200,
            "rows": [
                {
                    "handles": [a, b],
                    "layers": ["fixture-plots", "fixture-plots"],
                    "shared_area": {"value": 50.0, "unit": "m2", "method": "exact"},
                    "relationship": "partial: the two rings cross",
                }
                for a, b in pairs
            ],
        },
        "slivers_below_tolerance": {"count": slivers},
        "pairs_withheld_total": withheld,
        "ring_census": {"skipped_total": 0},
    }


@pytest.fixture
def sides(monkeypatch):
    """Replace the two side runs, keeping everything else real.

    Only `_run_recipe` moves. The recipe names, their declared parameters and
    the whole envelope still come from the live registry, so a side that
    stopped existing — or a parameter that was renamed — fails here rather than
    being papered over by the fake.
    """
    planned: dict[str, dict] = {}
    calls: list[tuple] = []

    def fake_run(drawing_id, recipe, params=None, *, layout=None, units=None):
        calls.append((drawing_id, recipe, copy.deepcopy(dict(params or {})), layout))
        if recipe not in planned:
            raise AssertionError(f"the test planned no response for {recipe!r}")
        return copy.deepcopy(planned[recipe])

    monkeypatch.setattr(cb, "_run_recipe", fake_run)
    return type("Sides", (), {"planned": planned, "calls": calls})()


def run(params, *, drawing=FIXTURE, layout=LAYOUT):
    return recipes.run(drawing, NAME, params, layout=layout, units=dict(METRE_UNITS))


# =============================================================================
# 1. Registration and the contract with the registry
# =============================================================================


def test_the_recipe_registers_itself_and_is_in_the_catalogue():
    assert NAME in recipes.known()
    entry = next(r for r in recipes.catalog()["recipes"] if r["recipe"] == NAME)
    assert entry["needs_layout"] is True
    assert entry["limits"]["recursion_depth"] == 1
    assert entry["limits"]["runs_per_call"] == 2
    for param in entry["params"]:
        assert param["kind"] in rr.PARAM_KINDS, param
        assert param["about"].strip()


def test_both_sides_go_through_the_registry_entry_point_every_caller_uses():
    """Not a private copy of a recipe and not a second route into one.

    Two routes into the same computation is two answers to the same question,
    with nothing in either response saying which route was taken.
    """
    assert cb._run_recipe is rr.run


def test_every_forwardable_parameter_matches_the_kind_its_target_declares():
    """The flat surface mirrors the real parameters; it does not invent them.

    A parameter declared here as `number` and there as `text` would be coerced
    twice, differently, and the second coercion would refuse a value the first
    one accepted — a failure that reads as the caller's mistake.
    """
    for name, kind, _about, accepted in cb.FORWARDABLE:
        for recipe_name in accepted:
            recipe = rr.get(recipe_name)
            assert recipe is not None, recipe_name
            declared = recipe.param(name)
            assert declared is not None, f"{recipe_name} declares no {name!r}"
            assert declared.kind == kind, (
                f"{recipe_name}.{name} is {declared.kind!r} there and {kind!r} here"
            )


def test_every_parameter_of_a_composable_recipe_can_be_reached():
    """A composable recipe with an unreachable parameter is a recipe that can
    only ever be run one way through here — and nothing in the catalogue would
    say so."""
    reachable = {row[0] for row in cb.FORWARDABLE}
    for rule in cb.ANSWER_SETS:
        recipe = rr.get(rule.recipe)
        assert recipe is not None, rule.recipe
        missing = sorted(p.name for p in recipe.params if p.name not in reachable)
        assert not missing, f"{rule.recipe}: {missing} cannot be set from here"


def test_every_registered_recipe_is_either_composable_or_carries_a_reason():
    """"Unsupported" tells a reader nothing and invites them to try the next
    one. Every refusal names why THAT recipe's answer set is ambiguous."""
    rows = cb._not_composable_rows()
    named = {row["recipe"] for row in rows}
    composable = {rule.recipe for rule in cb.ANSWER_SETS}
    assert named | composable | {NAME} == set(recipes.known())
    for row in rows:
        assert len(row["why"]) > 40, row


def test_the_answer_set_rule_of_every_composable_recipe_is_published():
    for rule in cb.ANSWER_SETS:
        published = rule.as_dict()
        assert published["recipe"] == rule.recipe
        assert "[]" in published["answer_set_from"], (
            "the rule names a key PATH into the side's own response, so a "
            "reader can look it up rather than take it on trust"
        )
        assert published["answer_set_rule"].strip()
        assert published["deliberately_excluded"].strip(), (
            "what a rule leaves OUT is the half a reader cannot infer"
        )


# =============================================================================
# 2. The composition itself
# =============================================================================


def test_intersect_returns_only_the_handles_both_sides_found(sides):
    sides.planned["outliers"] = outliers_body(["H1", "H2", "H3"])
    sides.planned["frontage_check"] = frontage_body(["H2", "H9"])

    out = run({"left": "outliers", "right": "frontage_check", "operation": "intersect"})

    assert out["composed"] is True
    assert out["handles"] == ["H2"]
    assert out["handles_total"] == 1
    assert out["conclusion_safety"]["usable_as_evidence"] is True
    assert len(sides.calls) == 2, "each side runs exactly once"


def test_union_and_difference_answer_different_questions(sides):
    sides.planned["outliers"] = outliers_body(["H1", "H2"])
    sides.planned["frontage_check"] = frontage_body(["H2", "H9"])

    both = run({"left": "outliers", "right": "frontage_check", "operation": "union"})
    assert both["handles"] == ["H1", "H2", "H9"]

    only_left = run(
        {"left": "outliers", "right": "frontage_check", "operation": "difference"}
    )
    assert only_left["handles"] == ["H1"]
    assert "LEFT minus RIGHT" in only_left["operation_rule"] or "LEFT" in only_left[
        "operation_rule"
    ]


def test_every_returned_handle_says_which_side_or_sides_produced_it(sides):
    """A reader must be able to retrace the answer rather than trust it."""
    sides.planned["outliers"] = outliers_body(["H1", "H2"])
    sides.planned["frontage_check"] = frontage_body(["H2", "H9"])

    out = run({"left": "outliers", "right": "frontage_check", "operation": "union"})
    rows = {row["handle"]: row for row in out["rows"]}

    assert rows["H1"]["sides"] == ["left"]
    assert rows["H9"]["sides"] == ["right"]
    assert rows["H2"]["sides"] == ["left", "right"]
    assert rows["H2"]["found_by"] == ["outliers", "frontage_check"]
    assert rows["H9"]["found_by"] == [None, "frontage_check"]

    # And the side's OWN row travels with it, so the number that made it a
    # finding is in front of the reader.
    assert rows["H2"]["left"]["rows"][0]["z_score"] == 3.1
    assert rows["H2"]["right"]["rows"][0]["verdict"] == "beyond the touch tolerance"
    assert rows["H2"]["left"]["answer_set_from"] == "outliers[].handle"
    assert rows["H1"]["layer"] == "fixture-plots"


def test_the_rule_used_to_read_each_side_is_published(sides):
    sides.planned["outliers"] = outliers_body(["H1"])
    sides.planned["frontage_check"] = frontage_body(["H1"])
    out = run({"left": "outliers", "right": "frontage_check"})

    assert out["left"]["answer_set_from"] == "outliers[].handle"
    assert out["right"]["answer_set_from"] == "no_frontage.rows[].handle"
    assert "frontage_unproven" in out["right"]["deliberately_excluded"]
    assert out["left"]["answer_set_size"] == 1
    assert out["operation"] == "intersect", "the default, and it is published"


def test_the_parameters_a_side_really_received_are_published(sides):
    sides.planned["outliers"] = outliers_body(["H1"])
    sides.planned["frontage_check"] = frontage_body(["H1"])

    run(
        {
            "left": "outliers",
            "right": "frontage_check",
            "left_z": 3,
            "right_road_layers": [CORRIDOR_LAYER],
        }
    )
    by_recipe = {call[1]: call[2] for call in sides.calls}
    assert by_recipe["outliers"] == {"z": 3.0}
    assert by_recipe["frontage_check"] == {"road_layers": (CORRIDOR_LAYER,)}


def test_a_parcel_in_several_overlapping_pairs_appears_once_with_all_its_rows(sides):
    """The overlap finding is about a PAIR, so both members are involved — and
    a parcel that overlaps two neighbours is one parcel, not two."""
    sides.planned["outliers"] = outliers_body(["A", "B", "C", "Z"])
    sides.planned["overlap_scan"] = overlap_body([("A", "B"), ("A", "C")])

    out = run({"left": "outliers", "right": "overlap_scan", "operation": "intersect"})
    assert out["handles"] == ["A", "B", "C"]
    rows = {row["handle"]: row for row in out["rows"]}
    assert len(rows["A"]["right"]["rows"]) == 2
    assert len(rows["B"]["right"]["rows"]) == 1


def test_slivers_are_not_overlaps_and_are_carried_as_a_caveat_not_a_result(sides):
    sides.planned["outliers"] = outliers_body(["A"])
    sides.planned["overlap_scan"] = overlap_body([("A", "B")], slivers=2479)

    out = run({"left": "outliers", "right": "overlap_scan"})
    assert out["handles"] == ["A"]
    joined = " ".join(out["caveats_from_the_sides"])
    assert "2479" in joined and "sliver" in joined
    assert out["conclusion_safety"]["usable_as_evidence"] is True


def test_a_side_caveat_is_carried_forward_and_never_summarised_away(sides):
    sides.planned["outliers"] = outliers_body(["H1"])
    sides.planned["frontage_check"] = frontage_body(["H1"], unproven=9)

    out = run({"left": "outliers", "right": "frontage_check"})
    joined = " ".join(out["caveats_from_the_sides"])
    assert "9 parcels came back UNPROVEN" in joined
    assert "carry no area" in joined
    assert out["conclusion_safety"]["usable_as_evidence"] is True, (
        "a caveat bounds how a result is read; it does not empty it"
    )


def test_the_result_is_capped_and_says_how_much_it_left_out(sides):
    """G7: a limit that is stated and that really binds."""
    many = [f"H{i:04d}" for i in range(cb.MAX_HANDLES_RETURNED + 25)]
    sides.planned["outliers"] = outliers_body(many)
    sides.planned["frontage_check"] = frontage_body(many)

    out = run({"left": "outliers", "right": "frontage_check"})
    assert out["handles_total"] == len(many)
    assert out["handles_returned"] == cb.MAX_HANDLES_RETURNED
    assert out["handles_dropped"] == 25
    assert out["handles_truncated"] is True
    assert len(out["rows"]) == cb.MAX_HANDLES_RETURNED


def test_the_same_call_twice_produces_the_same_bytes(sides):
    """Determinism, checked rather than intended: no clock, no ordering by
    dictionary insertion, no set iteration order reaching the response."""
    sides.planned["outliers"] = outliers_body(["H3", "H1", "H2"])
    sides.planned["frontage_check"] = frontage_body(["H2", "H1"])
    params = {"left": "outliers", "right": "frontage_check", "operation": "union"}

    first = json.dumps(run(params), sort_keys=True, default=str)
    second = json.dumps(run(params), sort_keys=True, default=str)
    assert first == second


# =============================================================================
# 3. The refusal — the reason this recipe exists
# =============================================================================


def test_a_side_that_established_nothing_refuses_instead_of_returning_an_empty_set(
    sides,
):
    """The measured failure, in one test.

    A frontage run with default road layers settles nothing, its list of
    parcels without frontage is empty FOR THAT REASON, and intersecting it with
    thirty outliers produced the clean-sounding "there are none". The answer is
    one parcel. So: no empty list, a named side, and a reason.
    """
    sides.planned["outliers"] = outliers_body(["H1", "H2", "H3"])
    sides.planned["frontage_check"] = frontage_body([], usable=False)

    out = run({"left": "outliers", "right": "frontage_check", "operation": "intersect"})

    assert out["composed"] is False
    assert out["handles"] is None, "an empty list here reads as 'there are none'"
    assert out["handles_total"] is None
    assert out["rows"] == []
    assert out["handles_withheld_reason"].strip()

    assert out["refusal"]["side"] == "right"
    assert out["refusal"]["recipe"] == "frontage_check"
    assert out["refusal"]["code"] == "COMBINE_SIDE_ESTABLISHED_NOTHING"
    assert "bounding boxes" in out["refusal"]["why"]
    assert "CLOSED rings" in out["refusal"]["how_to_get_an_answer"]

    assert out["conclusion_safety"]["usable_as_evidence"] is False
    assert out["conclusion_safety"]["verdict"] == "NOTHING TO COMPOSE"
    assert "there are none" in out["conclusion_safety"]["do_not"]

    # SETTLED and COMPLETE are two different failures, and were once read as
    # one: the frontage list here is untruncated — complete — and worth nothing.
    assert out["right"]["answer_set_settled"] is False
    assert out["right"]["answer_set_settled_note"].strip()
    assert out["left"]["answer_set_settled"] is True

    # The other side ran and its answer stands; it is simply not the reason.
    assert out["refusal"]["other_side"]["side"] == "left"
    assert out["left"]["answer_set_size"] == 3


def test_the_refusal_hands_back_this_whole_call_ready_to_run(sides):
    """Naming a candidate was measured to be not enough: told to act on a hint
    without an exact call, the agent invented parameters of its own and was
    wrong by a new route. So the plan is the whole composition."""
    sides.planned["outliers"] = outliers_body(["H1"])
    sides.planned["frontage_check"] = frontage_body([], usable=False)

    out = run(
        {
            "left": "outliers",
            "right": "frontage_check",
            "operation": "intersect",
            "left_z": 2.5,
        }
    )
    plan = out["retry_with"][0]
    assert plan["recipe"] == NAME
    assert plan["params"]["left"] == "outliers"
    assert plan["params"]["right"] == "frontage_check"
    assert plan["params"]["operation"] == "intersect"
    assert plan["params"]["right_road_layers"] == [CORRIDOR_LAYER]
    assert plan["params"]["left_z"] == 2.5, "the untouched side keeps its parameters"
    assert plan["because"]

    # And the side's own plans are repeated unchanged beside it.
    assert out["refusal"]["side_retry_with"][0]["recipe"] == "frontage_check"


def test_the_refusal_never_asks_the_api_layer_to_re_run_one_side(sides):
    """`main._retry_if_nothing_established` re-runs any response whose
    `conclusion_safety.retry_with` is populated — and what it would re-run here
    is a single recipe, replacing a composition with half of itself. So the
    ready-to-run calls live at the top level of the body instead."""
    sides.planned["outliers"] = outliers_body(["H1"])
    sides.planned["frontage_check"] = frontage_body([], usable=False)

    out = run({"left": "outliers", "right": "frontage_check"})
    assert "retry_with" not in out["conclusion_safety"]
    assert out["retry_with"], "the plans are published, just not there"


def test_the_api_layers_auto_retry_leaves_a_refused_composition_alone(sides, monkeypatch):
    """The seam, checked against the real consumer instead of by reading it.

    `main._retry_if_nothing_established` re-runs one recipe whenever a response
    carries `conclusion_safety.usable_as_evidence: false` AND a populated
    `conclusion_safety.retry_with`. Both halves of that condition are this
    file's business: the first is deliberately true here, so the second must be
    deliberately absent, or a refused composition would come back as one side's
    answer with a composition's name on it.
    """
    try:
        from app import main
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"cad-api could not be imported: {type(exc).__name__}: {exc}")

    sides.planned["outliers"] = outliers_body(["H1"])
    sides.planned["frontage_check"] = frontage_body([], usable=False)
    refused = run({"left": "outliers", "right": "frontage_check"})

    def boom(*_args, **_kwargs):  # pragma: no cover - fails the test if reached
        raise AssertionError("the API layer re-ran a side of a refused composition")

    monkeypatch.setattr(recipes, "run", boom)
    passed_through = main._retry_if_nothing_established(
        refused, FIXTURE, LAYOUT, dict(METRE_UNITS)
    )
    assert passed_through is refused
    assert "auto_retry" not in passed_through


def test_an_outliers_distribution_that_could_not_deviate_is_a_refusal_too(sides):
    """`outliers` publishes no `conclusion_safety`; it says the same thing
    through `why_empty`. An empty list from a population of one is an unasked
    question, not a finding."""
    sides.planned["outliers"] = outliers_body(
        [], why_empty="this distribution has no usable standard deviation: the "
        "population holds fewer than two values"
    )
    sides.planned["frontage_check"] = frontage_body(["H1"])

    out = run({"left": "outliers", "right": "frontage_check"})
    assert out["composed"] is False
    assert out["handles"] is None
    assert out["refusal"]["side"] == "left"
    assert "standard deviation" in out["refusal"]["why"]


def test_a_response_whose_answer_set_key_has_moved_refuses_rather_than_empties(sides):
    """A renamed key must not read as 'nothing found'. Withheld is not empty."""
    body = outliers_body(["H1"])
    body.pop("outliers")
    sides.planned["outliers"] = body
    sides.planned["frontage_check"] = frontage_body(["H1"])

    out = run({"left": "outliers", "right": "frontage_check"})
    assert out["composed"] is False
    assert out["handles"] is None
    assert out["refusal"]["code"] == "COMBINE_ANSWER_SET_MISSING"
    assert "outliers[].handle" in out["refusal"]["why"]
    assert out["left"]["answer_set_complete"] is False


def test_a_truncated_side_still_composes_but_cannot_support_an_absence(sides):
    """Positives stay sound — every handle returned really is in both sets —
    while the list may be short, so it cannot carry 'these are the only ones'."""
    sides.planned["outliers"] = outliers_body(["H1", "H2"], truncated=True)
    sides.planned["frontage_check"] = frontage_body(["H2"])

    out = run({"left": "outliers", "right": "frontage_check"})
    assert out["composed"] is True
    assert out["handles"] == ["H2"]
    assert out["left"]["answer_set_complete"] is False
    assert out["conclusion_safety"]["usable_as_evidence"] is False
    assert out["conclusion_safety"]["verdict"] == "INCOMPLETE INPUT SET"
    assert "ABSENCE" in out["conclusion_safety"]["do_not"]


def test_withheld_overlap_pairs_make_the_answer_set_incomplete(sides):
    sides.planned["outliers"] = outliers_body(["A"])
    sides.planned["overlap_scan"] = overlap_body([("A", "B")], withheld=3)

    out = run({"left": "outliers", "right": "overlap_scan"})
    assert out["handles"] == ["A"]
    assert out["right"]["answer_set_complete"] is False
    assert "withheld" in out["right"]["answer_set_incomplete_because"]
    assert out["conclusion_safety"]["usable_as_evidence"] is False


def test_the_left_side_is_reported_first_when_both_settled_nothing(sides):
    """A refusal that moved between sides would look like two problems."""
    sides.planned["outliers"] = outliers_body([], why_empty="nothing could deviate")
    sides.planned["frontage_check"] = frontage_body([], usable=False)

    out = run({"left": "outliers", "right": "frontage_check"})
    assert out["refusal"]["side"] == "left"


# =============================================================================
# 4. Refusals over the parameter surface
# =============================================================================


def test_a_composition_cannot_compose_another_composition():
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        run({"left": NAME, "right": "outliers"})
    assert excinfo.value.code == "RECIPE_PARAM_INVALID"
    assert "compose another composition" in excinfo.value.message
    assert "One level" in excinfo.value.hint


def test_an_unknown_recipe_name_is_refused_with_the_composable_list():
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        run({"left": "no_such_recipe", "right": "outliers"})
    assert excinfo.value.code == "RECIPE_PARAM_INVALID"
    assert "not a registered recipe" in excinfo.value.message
    assert "outliers" in excinfo.value.hint


@pytest.mark.parametrize(
    "recipe,expect",
    [
        ("label_coverage", "capped"),
        ("dimension_truth", "TWO handles"),
        ("drafting_hygiene", "handles_omitted"),
        ("adjacency", "own input"),
    ],
)
def test_a_recipe_whose_answer_set_is_ambiguous_is_refused_with_the_reason(
    recipe, expect
):
    """Refused rather than guessed, on purpose. A recipe reports the handles it
    tested, the ones it skipped and the ones it FOUND; composing the wrong one
    of those returns a wrong answer that looks exactly like a right one."""
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        run({"left": "outliers", "right": recipe})
    assert excinfo.value.code == "RECIPE_PARAM_INVALID"
    assert "cannot be composed" in excinfo.value.message
    assert expect in excinfo.value.message


def test_an_unknown_operation_is_refused_and_names_the_three():
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        run({"left": "outliers", "right": "frontage_check", "operation": "both_ish"})
    assert excinfo.value.code == "RECIPE_PARAM_INVALID"
    for word in ("intersect", "union", "difference"):
        assert word in excinfo.value.message + excinfo.value.hint


def test_a_parameter_aimed_at_a_side_that_cannot_take_it_is_refused_not_dropped():
    """A dropped parameter answers a different question from the one that was
    asked, and nothing in the response would show it."""
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        run({"left": "outliers", "right": "frontage_check", "right_z": 3})
    assert excinfo.value.code == "RECIPE_PARAM_UNKNOWN"
    assert "frontage_check" in excinfo.value.message
    assert "right_road_layers" in excinfo.value.hint


def test_a_parameter_this_recipe_does_not_declare_is_refused_by_the_engine():
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        run({"left": "outliers", "right": "frontage_check", "middle": "x"})
    assert excinfo.value.code == "RECIPE_PARAM_UNKNOWN"


def test_a_side_named_as_an_object_is_refused_by_the_parameter_engine():
    """The shape is flat, and the refusal for the nested one is the registry's
    own — which is the right place for it, because every recipe answers the
    same way to a value of the wrong shape."""
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        run({"left": {"recipe": "outliers"}, "right": "frontage_check"})
    assert excinfo.value.code == "RECIPE_PARAM_INVALID"


def test_a_missing_side_is_refused_because_it_has_no_default():
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        run({"left": "outliers"})
    assert excinfo.value.code == "RECIPE_PARAM_REQUIRED"


def test_the_layout_is_required_and_is_handed_to_both_sides(sides):
    sides.planned["outliers"] = outliers_body(["H1"])
    sides.planned["frontage_check"] = frontage_body(["H1"])
    run({"left": "outliers", "right": "frontage_check"}, layout="Layout1")
    assert [call[3] for call in sides.calls] == ["Layout1", "Layout1"]

    with pytest.raises(recipes.RecipeRefused) as excinfo:
        recipes.run(FIXTURE, NAME, {"left": "outliers", "right": "outliers"})
    assert excinfo.value.code == "RECIPE_LAYOUT_REQUIRED"


# =============================================================================
# 5. Janadriyah — the acceptance numbers, computed outside this path
# =============================================================================


def _live_or_skip():
    """The reference drawing from the store, or skip.

    Written here rather than imported from another lane's test file: a shared
    helper living in a file owned by someone else is a dependency that stays
    invisible until that file moves.
    """
    from app import store as live_store

    try:
        drawing = live_store.get_drawing(DRAWING)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip(f"reference drawing {DRAWING} has not been ingested")
    return live_store, drawing


_LIVE_CACHE: dict[str, dict] = {}


def _live(params):
    """One composition against the real drawing, cached.

    A composition runs both sides in full, and the frontage side alone measures
    2,545 rings against the road geometry. Several tests below read the SAME
    answer from different angles, so it is computed once — except in the
    determinism test, which deliberately goes around the cache.
    """
    key = json.dumps(params, sort_keys=True)
    if key not in _LIVE_CACHE:
        _LIVE_CACHE[key] = _live_uncached(params)
    return _LIVE_CACHE[key]


def _live_uncached(params):
    live_store, drawing = _live_or_skip()
    return recipes.run(
        DRAWING,
        NAME,
        params,
        layout=LAYOUT,
        units=live_store._unit_names(drawing, LAYOUT),
    )


def test_janadriyah_outliers_and_no_frontage_is_one_parcel():
    """The question that was answered wrongly five times.

    Run against the layer of CLOSED rings — the right-of-way corridor, whose
    name carries no road word at all — the frontage check establishes its
    answer, and exactly one parcel is both a size outlier and without frontage.
    """
    out = _live(
        {
            "left": "outliers",
            "right": "frontage_check",
            "operation": "intersect",
            "right_road_layers": [CORRIDOR_LAYER],
        }
    )

    assert out["composed"] is True
    assert tuple(out["handles"]) == GT_BOTH_OUTLIER_AND_NO_FRONTAGE
    assert out["handles_total"] == 1
    assert out["left"]["answer_set_size"] == GT_OUTLIERS
    assert out["right"]["conclusion_safety"]["usable_as_evidence"] is True
    assert out["conclusion_safety"]["usable_as_evidence"] is True

    row = out["rows"][0]
    assert row["sides"] == ["left", "right"]
    assert row["found_by"] == ["outliers", "frontage_check"]
    assert row["layer"], "a handle without its layer is not clickable"
    assert row["left"]["rows"][0]["z_score"]
    assert row["right"]["rows"][0]["nearest_road"]["value"] is not None


def test_a_composition_publishes_a_verdict_only_when_the_side_settled():
    """The single most important number in this file is the one NOT returned.

    A set operation over a side that measured nothing would publish "there are
    none" from an empty list that means "we could not tell". That is the
    failure this guard exists for, and it is an INVARIANT: composed is true
    exactly when the right side settled its answer.

    It used to be written as a corpus fact instead -- "with its default road
    layers the frontage check settles nothing, therefore this composition
    refuses". That stopped being true. `frontage_check` carries its own retry
    plan and the runner executes it (`e2d8a78`, which predates this campaign),
    so on this drawing the default layers now retry onto the corridor layer
    ROW and the check settles with two parcels. The old assertion therefore
    failed or passed according to whether the retry happened to find a
    settling layer -- which also made it sensitive to suite ordering, passing
    in a full run and failing in a subset of the same commit.

    Asserting the linkage instead is both stronger and stable: it holds
    whichever way the retry goes, and it still fails loudly if a composition
    ever publishes handles from a side that did not establish them.
    """
    out = _live({"left": "outliers", "right": "frontage_check", "operation": "intersect"})

    settled = out["right"]["answer_set_settled"]
    assert isinstance(settled, bool), "the side must state whether it settled"
    assert out["composed"] is settled, (
        "a composition may publish a verdict only when the side it composed "
        "actually established one"
    )

    if not settled:
        assert out["handles"] is None
        assert out["handles_total"] is None
        assert out["refusal"]["side"] == "right"
        assert out["refusal"]["recipe"] == "frontage_check"
        assert out["refusal"]["why"], "a refusal without a reason is a shrug"
        assert out["refusal"]["how_to_get_an_answer"]
        assert out["conclusion_safety"]["usable_as_evidence"] is False
        # And the way out is a whole call, not an instruction to assemble one.
        plan = out["retry_with"][0]
        assert plan["recipe"] == NAME
        assert plan["params"]["left"] == "outliers"
        assert plan["params"]["right"] == "frontage_check"
        assert plan["params"]["right_road_layers"]
    else:
        assert out["refusal"] is None
        assert out["handles_total"] == len(out["handles"])
        assert out["conclusion_safety"]["usable_as_evidence"] is True
        if out["right"].get("retried"):
            # A retry that changed the answer must say so, or the reader
            # cannot tell which road layers the verdict rests on.
            assert out["right"]["retried_with"]

    # The left side ran and stands either way; it is not the reason for any
    # refusal, and its size is a golden constant on this drawing.
    assert out["left"]["answer_set_size"] == GT_OUTLIERS


def test_janadriyah_outliers_that_also_overlap_a_neighbour():
    out = _live({"left": "outliers", "right": "overlap_scan", "operation": "intersect"})

    assert out["composed"] is True
    assert out["handles_total"] == GT_OUTLIER_AND_OVERLAPPING
    assert len(out["handles"]) == GT_OUTLIER_AND_OVERLAPPING
    for handle in GT_OVERLAP_SAMPLE:
        assert handle in out["handles"], handle
    assert out["left"]["answer_set_size"] == GT_OUTLIERS
    assert out["right"]["answer_set_from"] == "overlaps.rows[].handles[]"
    assert out["handles"] == sorted(out["handles"]), "a stable order, so two runs compare"


def test_janadriyah_the_result_carries_both_sides_evidence_and_its_own_scope():
    out = _live({"left": "outliers", "right": "overlap_scan", "operation": "intersect"})

    assert out["evidence"]["grade"] == "inferred", (
        "set membership inferred from two measurements can be no stronger"
    )
    assert out["evidence"]["not_established"].strip()
    assert out["evidence"]["how_to_verify"].strip()
    assert "measured nothing" in out["evidence"]["not_established"]
    assert LAYOUT in out["scope_note"]
    assert "overlaps.rows[].handles[]" in out["scope_note"]
    # The side that says what something IS keeps its own evidence block, whole.
    assert out["right"]["evidence"]["grade"] in {
        "unknown", "inferred", "stated", "corroborated"
    }
    assert out["right"]["scope_note"].strip()


def test_janadriyah_the_composition_is_deterministic():
    params = {"left": "outliers", "right": "overlap_scan", "operation": "intersect"}
    first = _live_uncached(params)
    second = _live_uncached(params)
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(
        second, sort_keys=True, default=str
    )
