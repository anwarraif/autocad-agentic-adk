"""DOSSIER Phase 7 — the revision diff and the arrival path.

Two halves, and the second one is the one the owner asked for:

* `app/recipes/revision.py` — two Dossiers compared. Every test of it below
  runs on **fixtures**: no MongoDB, no ezdxf, no cad-api. The engine is a pure
  function of two documents, which is what makes it possible to assert G2 and
  G8 on a laptop, and on a drawing nobody has ingested.
* `app/ingest.py` — the hook that means a drawing which arrives tomorrow is
  understood without anyone remembering to run a script. Its tests are written
  to catch the failure that would actually happen: not "does it work", but
  **"does a profiler that raises take the ingest down with it"**. A drawing
  stored without a profile is recoverable; an ingest that died inside a
  profiler has lost the entity data too.

The tests written to catch a collapse rather than to confirm a success:

* `test_a_metre_is_never_subtracted_from_an_inch` and its per-bucket sibling.
  Eleven of the drawings in this store are in inches and three declare
  nothing. A diff that subtracts across that produces a number, and the number
  means nothing (G2).
* `test_an_absent_measure_is_never_treated_as_zero`. "Nothing here could be
  measured" and "this measures zero" are different facts (G8), and a diff that
  let them meet would publish the whole of one side as if it were the change.
* `test_several_candidates_are_listed_and_none_is_chosen`. The predecessor
  rule's only interesting branch is the ambiguous one, because that is the
  branch where picking a winner would be easy and wrong.
* `test_the_arrival_path_never_raises` and
  `test_a_broken_profiler_does_not_fail_the_ingest`. The hook is only worth
  having if it cannot cost an ingest.
* `test_the_planted_changes_and_nothing_else` — the live acceptance, against
  the synthetic revision pair actually in the store. It skips when the store
  or the fixture is out of reach, and says so.

Drawing-specific strings appear below only as fixtures and as the pinned
figures of the synthetic pair, which rule G1 exempts (*kecuali di berkas config
dan di test*).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
from typing import Any

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import ingest  # noqa: E402
from app.recipes import registry  # noqa: E402
from app.recipes import revision as rev  # noqa: E402


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "dossier_diff.py"
FIXTURE_DIR = REPO_ROOT / "data" / "dxf" / "revision_pair"
MANIFEST_PATH = FIXTURE_DIR / "planted_edits.json"


# --- fixtures -----------------------------------------------------------------


def bucket(
    layer: str,
    layout: str = "Model",
    *,
    entities: int = 3,
    types: dict[str, int] | None = None,
    role: str = "network",
    kind: str | None = "length",
    value: float | None = 100.0,
    unit: str | None = "m",
    anomalies: list[dict[str, Any]] | None = None,
    examined: bool = True,
    chains: int | None = None,
) -> dict[str, Any]:
    """One (layer x layout) block, shaped as `app/dossier.py` stores it."""
    measure: dict[str, Any] = {
        "kind": kind,
        "value": value,
        "unit": unit,
        "unit_reason": None if unit else "this bucket declares no unit",
        "measured_entities": entities,
        "unmeasured_entities": 0,
        "bucket_entities": entities,
        "caveat": None,
    }
    if chains is not None:
        measure["chains"] = chains
    return {
        "layer": layer,
        "layout": layout,
        "entities": entities,
        "types": dict(types or {"LINE": entities}),
        "role": role,
        "role_basis": f"{role} because the geometry says so",
        "measure": measure,
        "anomalies": list(anomalies or []),
        "anomalies_status": {"computed": examined, "why": None},
    }


def dossier(
    drawing_id: str,
    blocks: list[dict[str, Any]],
    *,
    units_name: str | None = "m",
    units_code: int | None = 6,
    entity_total: int | None = None,
    complete: bool = True,
    layers_declared: int | None = None,
    audit_errors: int = 0,
) -> dict[str, Any]:
    """One whole Dossier document, shaped as `app/dossier.build_dossier` does."""
    total = (
        entity_total
        if entity_total is not None
        else sum(int(b["entities"]) for b in blocks)
    )
    layouts: dict[str, int] = {}
    for block in blocks:
        layouts[str(block["layout"])] = layouts.get(str(block["layout"]), 0) + int(
            block["entities"]
        )
    return {
        "_id": drawing_id,
        "drawing_id": drawing_id,
        "computed_at": "2026-08-25T00:00:00+00:00",
        "dossier_version": 1,
        "entity_total": total,
        "layouts": [
            {
                "name": name,
                "declared_entities": count,
                "profiled_entities": count,
                "agrees": True,
                "is_modelspace": name == "Model",
                "is_block": name.startswith("[block]"),
            }
            for name, count in sorted(layouts.items())
        ],
        "layers": list(blocks),
        "file_facts": {
            "xrefs": {"total": 0, "unresolved": 0},
            "text_styles": {"total": 1, "shx_styles": 0},
            "layers": {
                "declared": (
                    layers_declared
                    if layers_declared is not None
                    else len({b["layer"] for b in blocks})
                ),
                "profiled": len({b["layer"] for b in blocks}),
                "declared_but_unprofiled": 0,
            },
            "layouts": {"declared": len(layouts), "disagreeing": 0},
            "embedded_documents": {"available": True, "total": 0},
            "audit": {"errors": audit_errors, "fixes": 0},
            "units": {
                "units_name": units_name,
                "units_code": units_code,
                "declared_in_file": bool(units_code),
            },
        },
        "coverage": {
            "accounted": sum(int(b["entities"]) for b in blocks),
            "total": total,
            "difference": total - sum(int(b["entities"]) for b in blocks),
            "complete": complete,
            "buckets": len(blocks),
            "verdict": "COMPLETE" if complete else "NOT COMPLETE",
        },
    }


def changed_keys(diff: dict[str, Any]) -> set[tuple[str, str]]:
    """Every bucket the diff says moved, added and removed included."""
    out = {(r["layer"], r["layout"]) for r in diff["layers_changed"]}
    out |= {(r["layer"], r["layout"]) for r in diff["layers_added"]}
    out |= {(r["layer"], r["layout"]) for r in diff["layers_removed"]}
    return out


def one(diff: dict[str, Any], layer: str, layout: str = "Model") -> dict[str, Any]:
    for row in diff["layers_changed"]:
        if row["layer"] == layer and row["layout"] == layout:
            return row
    raise AssertionError(
        f"{layer!r} in {layout!r} is not among the changed buckets: "
        f"{sorted(changed_keys(diff))}"
    )


# =============================================================================
# The engine — nothing moved
# =============================================================================


def test_two_identical_dossiers_report_no_bucket_change():
    a = dossier("aaa", [bucket("L1"), bucket("L2", value=50.0)])
    b = copy.deepcopy(a)
    b["_id"] = b["drawing_id"] = "bbb"
    diff = rev.diff_dossiers(a, b)

    assert changed_keys(diff) == set()
    assert diff["totals"]["buckets"]["unchanged"] == 2
    assert diff["totals"]["entity_total"]["delta"] == 0


def test_no_change_is_never_reported_as_two_identical_files():
    """The verdict for "nothing moved" must not read as "same file".

    A Dossier holds aggregates. Saying "no difference" without saying what
    that cannot rule out is the confident half-truth this campaign exists to
    end, and it is precisely the sentence a reader is most likely to quote.
    """
    a = dossier("aaa", [bucket("L1")])
    b = copy.deepcopy(a)
    b["_id"] = b["drawing_id"] = "bbb"
    verdict = rev.diff_dossiers(a, b)["verdict"]
    assert "NOT the same as" in verdict
    assert "cannot_see" in verdict


def test_the_same_drawing_id_is_refused_as_a_comparison():
    a = dossier("aaa", [bucket("L1")])
    diff = rev.diff_dossiers(a, copy.deepcopy(a))
    assert diff["compared"]["same_drawing"] is True
    assert "content hash" in diff["verdict"]


def test_floating_point_noise_is_not_a_change():
    """A delta of 7e-12 is not a revision.

    Every measure the Dossier stores is rounded to six decimals, so the diff
    rounds its subtraction to the same place. Without it, re-exporting a file
    would 'change' half its layers.
    """
    a = dossier("aaa", [bucket("L1", value=83962.565009)])
    b = dossier("bbb", [bucket("L1", value=83962.565009 + 7e-12)])
    assert changed_keys(rev.diff_dossiers(a, b)) == set()


# =============================================================================
# The engine — what it must report
# =============================================================================


def test_a_layer_that_appeared_is_listed_with_what_it_holds():
    a = dossier("aaa", [bucket("L1")])
    b = dossier("bbb", [bucket("L1"), bucket("NEW", entities=1, value=250.0)])
    diff = rev.diff_dossiers(a, b)

    assert [r["layer"] for r in diff["layers_added"]] == ["NEW"]
    row = diff["layers_added"][0]
    assert row["entities"] == 1
    assert row["measure"] == {
        "kind": "length",
        "value": 250.0,
        "unit": "m",
        "unit_reason": None,
    }
    assert diff["totals"]["layers"]["added"] == ["NEW"]


def test_a_layer_that_disappeared_is_listed_with_what_it_held():
    a = dossier("aaa", [bucket("L1"), bucket("GONE", entities=4, value=9.5)])
    b = dossier("bbb", [bucket("L1")])
    diff = rev.diff_dossiers(a, b)

    assert [r["layer"] for r in diff["layers_removed"]] == ["GONE"]
    assert diff["layers_removed"][0]["entities"] == 4
    assert diff["totals"]["layers"]["removed"] == ["GONE"]


def test_entity_and_type_counts_carry_their_deltas():
    a = dossier("aaa", [bucket("L1", entities=7, types={"LINE": 7}, value=880.0)])
    b = dossier("bbb", [bucket("L1", entities=5, types={"LINE": 5}, value=560.0)])
    row = one(rev.diff_dossiers(a, b), "L1")

    assert row["entities"] == {"from": 7, "to": 5, "delta": -2, "changed": True}
    assert row["types"]["changed"] == [
        {
            "type": "LINE",
            "from": 7,
            "to": 5,
            "delta": -2,
            "appeared": False,
            "disappeared": False,
        }
    ]


def test_a_measure_delta_carries_its_unit():
    a = dossier("aaa", [bucket("L1", value=75.819428, unit="inch")], units_name="inch", units_code=1)
    b = dossier("bbb", [bucket("L1", value=94.774285, unit="inch")], units_name="inch", units_code=1)
    row = one(rev.diff_dossiers(a, b), "L1")

    assert row["measure"]["compared"] is True
    assert row["measure"]["delta"] == pytest.approx(18.954857, abs=1e-6)
    assert row["measure"]["delta_unit"] == "inch"


def test_a_role_change_is_reported_with_both_bases():
    a = dossier("aaa", [bucket("L1", role="network", kind="length")])
    b = dossier("bbb", [bucket("L1", role="region", kind="area")])
    row = one(rev.diff_dossiers(a, b), "L1")

    assert row["role"]["from"] == "network"
    assert row["role"]["to"] == "region"
    assert row["role"]["from_basis"] and row["role"]["to_basis"]
    assert "G1" in row["role"]["note"]


def test_an_anomaly_that_appeared_is_reported_and_rolled_up():
    clean = bucket("L1")
    flagged = bucket(
        "L1",
        anomalies=[{"kind": "duplicate_clusters", "detail": "3 copies", "counts": {}}],
    )
    diff = rev.diff_dossiers(dossier("aaa", [clean]), dossier("bbb", [flagged]))
    row = one(diff, "L1")

    assert row["anomalies"]["appeared"] == ["duplicate_clusters"]
    assert diff["anomalies"]["appeared"] == {"duplicate_clusters": 1}


def test_a_layer_that_stopped_being_examined_is_a_change():
    """`examined: false` must never quietly replace a clean bill of health.

    Nothing about the layer moved here — what moved is what is KNOWN about it,
    and a diff that hid that would let "we did not look this time" be read as
    "it is still fine".
    """
    a = dossier("aaa", [bucket("L1", examined=True)])
    b = dossier("bbb", [bucket("L1", examined=False)])
    row = one(rev.diff_dossiers(a, b), "L1")

    assert row["anomalies"]["examined_from"] is True
    assert row["anomalies"]["examined_to"] is False
    assert "not been shown to be gone" in row["anomalies"]["note"]


def test_chain_counts_are_compared_as_counts():
    a = dossier("aaa", [bucket("L1", chains=849)])
    b = dossier("bbb", [bucket("L1", chains=847)])
    row = one(rev.diff_dossiers(a, b), "L1")
    assert row["measure"]["chains"] == {
        "from": 849,
        "to": 847,
        "delta": -2,
        "changed": True,
    }


def test_layouts_and_file_facts_carry_their_own_deltas():
    a = dossier("aaa", [bucket("L1", entities=7)], layers_declared=12)
    b = dossier("bbb", [bucket("L1", entities=5), bucket("NEW", entities=1)], layers_declared=13)
    diff = rev.diff_dossiers(a, b)

    assert diff["layouts_changed"]["changed"][0]["layout"] == "Model"
    assert diff["layouts_changed"]["changed"][0]["profiled_entities"]["delta"] == -1
    facts = {row["fact"]: (row["from"], row["to"]) for row in diff["file_facts_changed"]["changed"]}
    assert facts["layers.declared"] == (12, 13)
    assert facts["layers.profiled"] == (1, 2)


def test_a_unit_change_in_the_file_header_is_itself_a_finding():
    a = dossier("aaa", [bucket("L1", unit="inch")], units_name="inch", units_code=1)
    b = dossier("bbb", [bucket("L1", unit="m")], units_name="m", units_code=6)
    facts = {
        row["fact"]: (row["from"], row["to"])
        for row in rev.diff_dossiers(a, b)["file_facts_changed"]["changed"]
    }
    assert facts["units.units_name"] == ("inch", "m")
    assert facts["units.units_code"] == (1, 6)


# =============================================================================
# G2 — the arithmetic that is refused
# =============================================================================


def test_a_metre_is_never_subtracted_from_an_inch():
    a = dossier("aaa", [bucket("L1", value=100.0, unit="inch")], units_name="inch", units_code=1)
    b = dossier("bbb", [bucket("L1", value=100.0, unit="m")], units_name="m", units_code=6)
    diff = rev.diff_dossiers(a, b)

    assert diff["units"]["agree"] is False
    assert diff["units"]["measure_arithmetic"] == "refused"
    row = one(diff, "L1")
    assert row["measure"]["compared"] is False
    assert row["measure"]["delta"] is None
    assert "G2" in row["measure"]["why_not"]
    # Both figures are still published: a refusal is not a silence.
    assert row["measure"]["value"] == {"from": 100.0, "to": 100.0}
    assert diff["not_compared"], "a refusal that is not listed is a silent one"


def test_counts_still_compare_when_the_units_disagree():
    """A count of objects carries no unit, so it is safe across a unit change.

    Refusing the counts as well would make a diff between an inch drawing and
    its metric re-export report nothing at all, when the thing a reviewer most
    needs to know — whether entities were added or lost in the conversion — is
    a pure count.
    """
    a = dossier("aaa", [bucket("L1", entities=7, types={"LINE": 7}, unit="inch")], units_name="inch", units_code=1)
    b = dossier("bbb", [bucket("L1", entities=5, types={"LINE": 5}, unit="m")], units_name="m", units_code=6)
    row = one(rev.diff_dossiers(a, b), "L1")

    assert row["entities"]["delta"] == -2
    assert row["types"]["changed"][0]["delta"] == -2


def test_one_bucket_whose_own_unit_moved_is_refused_on_its_own_terms():
    """The two files agree; this one bucket does not.

    A layer that moves between model space and a block definition loses its
    unit, because DXF declares none per layout. Subtracting across that move
    compares two different questions.
    """
    a = dossier("aaa", [bucket("L1", value=10.0, unit="m"), bucket("L2", value=1.0, unit="m")])
    b = dossier("bbb", [bucket("L1", value=12.0, unit=None), bucket("L2", value=3.0, unit="m")])
    diff = rev.diff_dossiers(a, b)

    assert diff["units"]["agree"] is True
    assert one(diff, "L1")["measure"]["compared"] is False
    assert one(diff, "L2")["measure"]["compared"] is True
    assert one(diff, "L2")["measure"]["delta"] == 2.0


def test_an_area_is_never_subtracted_from_a_length():
    a = dossier("aaa", [bucket("L1", kind="length", value=10.0, unit="m")])
    b = dossier("bbb", [bucket("L1", kind="area", value=10.0, unit="m")])
    row = one(rev.diff_dossiers(a, b), "L1")

    assert row["measure"]["compared"] is False
    assert "means nothing" in row["measure"]["why_not"]


def test_an_absent_measure_is_never_treated_as_zero():
    """G8. `None` is "nothing could be measured", not "this measures nothing".

    Subtracting a measurement from a blank would publish the whole of one side
    as if it were the change — a 560 m layer that lost its measure would read
    as 560 m of new road.
    """
    a = dossier("aaa", [bucket("L1", value=None)])
    b = dossier("bbb", [bucket("L1", value=560.0)])
    row = one(rev.diff_dossiers(a, b), "L1")

    assert row["measure"]["compared"] is False
    assert row["measure"]["delta"] is None
    assert "G8" in row["measure"]["why_not"]


# =============================================================================
# The engine — scope, caps and the stated limits
# =============================================================================


def test_scoping_to_one_layout_excludes_the_others():
    a = dossier("aaa", [bucket("L1", "Model"), bucket("L1", "[block] B", value=1.0)])
    b = dossier("bbb", [bucket("L1", "Model"), bucket("L1", "[block] B", value=2.0)])

    everything = rev.diff_dossiers(a, b)
    assert ("L1", "[block] B") in changed_keys(everything)

    model_only = rev.diff_dossiers(a, b, layout="Model")
    assert changed_keys(model_only) == set()
    assert "only layout 'Model'" in model_only["scope"]["basis"]


def test_the_default_scope_includes_block_definitions():
    diff = rev.diff_dossiers(dossier("aaa", [bucket("L1")]), dossier("bbb", [bucket("L1")]))
    assert "block definitions among them" in diff["scope"]["basis"]


def test_an_incomplete_dossier_on_either_side_is_said_out_loud():
    a = dossier("aaa", [bucket("L1")], entity_total=99, complete=False)
    b = dossier("bbb", [bucket("L1")])
    coverage = rev.diff_dossiers(a, b)["totals"]["coverage"]

    assert coverage["both_complete"] is False
    assert "part of a file" in coverage["note"]


def test_the_translation_blind_spot_is_published_on_every_response():
    diff = rev.diff_dossiers(dossier("aaa", [bucket("L1")]), dossier("bbb", [bucket("L1")]))
    what = [row["what"] for row in diff["cannot_see"]]
    assert any("MOVED" in item for item in what)
    assert any("WHICH entity" in item for item in what)
    for row in diff["cannot_see"]:
        assert row["why"] and row["remedy"]


def test_every_list_states_its_cap_and_counts_what_it_dropped():
    many_a = [bucket(f"L{i}", value=float(i)) for i in range(rev.MAX_BUCKET_ROWS + 5)]
    many_b = [bucket(f"L{i}", value=float(i) + 1.0) for i in range(rev.MAX_BUCKET_ROWS + 5)]
    diff = rev.diff_dossiers(dossier("aaa", many_a), dossier("bbb", many_b))

    assert len(diff["layers_changed"]) == rev.MAX_BUCKET_ROWS
    assert diff["truncation"]["changed_dropped"] == 5
    # The count is computed over every bucket, not over the listed ones.
    assert diff["totals"]["buckets"]["changed"] == rev.MAX_BUCKET_ROWS + 5


def test_the_engine_refuses_something_that_is_not_a_dossier():
    with pytest.raises(TypeError):
        rev.diff_dossiers("not a document", dossier("bbb", [bucket("L1")]))


# =============================================================================
# The predecessor rule
# =============================================================================


def drawing(drawing_id: str, name: str | None, entities: int = 10) -> dict[str, Any]:
    row: dict[str, Any] = {"_id": drawing_id, "entity_count": entities}
    if name is not None:
        row["original_filename"] = name
    return row


def test_exactly_one_candidate_is_the_predecessor():
    mine = drawing("bbb", "plan.dxf")
    found = rev.find_predecessors(
        mine, [mine, drawing("aaa", "plan.dxf"), drawing("ccc", "other.dxf")]
    )
    assert found["predecessor"] == "aaa"
    assert found["candidate_count"] == 1
    assert found["ambiguous"] is False


def test_no_candidate_says_so_and_names_the_rename_case():
    mine = drawing("bbb", "plan.dxf")
    found = rev.find_predecessors(mine, [mine, drawing("ccc", "other.dxf")])

    assert found["predecessor"] is None
    assert found["candidate_count"] == 0
    assert found["ambiguous"] is False
    assert "renamed" in found["why"]


def test_several_candidates_are_listed_and_none_is_chosen():
    """The one branch where picking a winner would be easy and wrong."""
    mine = drawing("ddd", "plan.dxf")
    found = rev.find_predecessors(
        mine,
        [mine, drawing("aaa", "plan.dxf"), drawing("bbb", "plan.dxf"), drawing("ccc", "x.dxf")],
    )

    assert found["predecessor"] is None
    assert found["ambiguous"] is True
    assert found["candidate_count"] == 2
    assert {row["drawing_id"] for row in found["candidates"]} == {"aaa", "bbb"}
    assert "not resolved" in found["why"]


def test_a_drawing_is_never_its_own_predecessor():
    mine = drawing("aaa", "plan.dxf")
    assert rev.find_predecessors(mine, [mine])["candidate_count"] == 0


def test_a_nameless_drawing_is_not_matched_against_other_nameless_ones():
    mine = drawing("aaa", None)
    found = rev.find_predecessors(mine, [mine, drawing("bbb", None)])

    assert found["predecessor"] is None
    assert found["candidate_count"] == 0
    assert "nothing to match on" in found["why"]
    assert "NOT a statement" in found["why"]


def test_the_name_falls_back_to_name_when_there_is_no_original_filename():
    mine = {"_id": "bbb", "name": "plan.dxf"}
    found = rev.find_predecessors(mine, [mine, {"_id": "aaa", "name": "plan.dxf"}])
    assert found["predecessor"] == "aaa"


def test_the_match_is_exact_and_does_not_fold_case_or_whitespace():
    mine = drawing("bbb", "Plan.dxf")
    found = rev.find_predecessors(mine, [mine, drawing("aaa", "plan.dxf")])
    assert found["candidate_count"] == 0


def test_the_candidate_listing_is_deterministic():
    mine = drawing("zzz", "plan.dxf")
    others = [drawing("ccc", "plan.dxf"), drawing("aaa", "plan.dxf"), drawing("bbb", "plan.dxf")]
    first = rev.find_predecessors(mine, [mine] + others)
    second = rev.find_predecessors(mine, [mine] + list(reversed(others)))
    assert first["candidates"] == second["candidates"]


def test_the_rule_and_its_limits_travel_with_every_answer():
    found = rev.find_predecessors(drawing("aaa", "plan.dxf"), [])
    assert "same drawing name" in found["rule"].lower()
    assert "different content hash" in found["rule"].lower()
    assert any("RENAMED" in limit for limit in found["rule_limits"])
    assert any("not a proven predecessor" in limit for limit in found["rule_limits"])


# =============================================================================
# The recipe
# =============================================================================


def test_the_recipe_is_registered_with_the_shape_the_agent_reads():
    recipe = registry.get("revision_diff")
    assert recipe is not None, "importing app.recipes.revision must register it"
    assert {p.name for p in recipe.params} == {"other_drawing", "scope"}
    assert recipe.param("scope").default == "all"
    # It measures; it says what nothing IS. So it owes `not_measured`, not
    # `evidence` — and a revision question is about a FILE, not about a sheet.
    assert recipe.carries_meaning is False
    assert recipe.needs_layout is False


def _stub_store(monkeypatch, documents: dict[str, dict[str, Any]], found: dict[str, Any]):
    monkeypatch.setattr(rev, "_dossier", lambda drawing_id: documents.get(str(drawing_id)))
    monkeypatch.setattr(rev, "predecessor_for", lambda drawing_id: dict(found))


def test_the_recipe_answers_through_the_envelope_contract(monkeypatch):
    """The body must survive `registry.envelope`, not merely look right.

    `envelope` refuses a recipe that publishes no `scope_note`, and refuses a
    measuring recipe that does not name what it did NOT measure. Asserting the
    body's keys directly would pass while the real call path raised.
    """
    documents = {
        "bbb": dossier("bbb", [bucket("L1", entities=5, value=560.0)]),
        "aaa": dossier("aaa", [bucket("L1", entities=7, value=880.0)]),
    }
    _stub_store(
        monkeypatch,
        documents,
        {"predecessor": "aaa", "candidate_count": 1, "ambiguous": False, "why": "one"},
    )
    out = registry.run("bbb", "revision_diff", {}, layout="Model", units={"length_unit": "m"})

    assert out["recipe"] == "revision_diff"
    assert out["compared"] is True
    assert out["scope_note"]
    assert out["not_measured"]
    assert out["diff"]["totals"]["entity_total"]["delta"] == -2


def test_the_recipe_compares_the_named_drawing_and_says_the_rule_was_not_used(monkeypatch):
    documents = {
        "bbb": dossier("bbb", [bucket("L1")]),
        "zzz": dossier("zzz", [bucket("L1", entities=9)]),
    }
    _stub_store(monkeypatch, documents, {"predecessor": None, "candidate_count": 0})
    out = registry.run(
        "bbb", "revision_diff", {"other_drawing": "zzz"}, layout="Model", units={}
    )

    assert out["compared"] is True
    assert out["predecessor_search"]["source"] == "named by the caller"
    assert out["predecessor_search"]["predecessor"] == "zzz"


def test_an_ambiguous_predecessor_is_an_answer_and_not_an_error(monkeypatch):
    """An agent handed a bare error reaches for another tool and answers wrongly.

    So ambiguity comes back as a well-formed response that says which case it
    was, lists the candidates, and names the parameter that resolves it.
    """
    _stub_store(
        monkeypatch,
        {"bbb": dossier("bbb", [bucket("L1")])},
        {
            "predecessor": None,
            "candidate_count": 2,
            "ambiguous": True,
            "why": "2 other drawings in the store are named 'plan.dxf'",
            "candidates": [{"drawing_id": "aaa"}, {"drawing_id": "ccc"}],
        },
    )
    out = registry.run("bbb", "revision_diff", {}, layout="Model", units={})

    assert out["compared"] is False
    assert out["diff"] is None
    assert out["predecessor_search"]["ambiguous"] is True
    assert "other_drawing" in out["what_to_do"]
    assert out["not_measured"]


def test_no_predecessor_is_also_an_answer(monkeypatch):
    _stub_store(
        monkeypatch,
        {"bbb": dossier("bbb", [bucket("L1")])},
        {"predecessor": None, "candidate_count": 0, "ambiguous": False, "why": "none"},
    )
    out = registry.run("bbb", "revision_diff", {}, layout="Model", units={})

    assert out["compared"] is False
    assert "rename" in out["what_to_do"]


def test_a_missing_dossier_refuses_without_claiming_the_drawing_is_empty(monkeypatch):
    _stub_store(monkeypatch, {}, {"predecessor": None, "candidate_count": 0})
    with pytest.raises(registry.RecipeRefused) as caught:
        registry.run("bbb", "revision_diff", {}, layout="Model", units={})
    assert "NOT a statement that the drawing is empty" in caught.value.hint


def test_scope_layout_restricts_and_scope_all_does_not(monkeypatch):
    documents = {
        "bbb": dossier("bbb", [bucket("L1", "Model"), bucket("L1", "[block] B", value=2.0)]),
        "aaa": dossier("aaa", [bucket("L1", "Model"), bucket("L1", "[block] B", value=1.0)]),
    }
    _stub_store(
        monkeypatch,
        documents,
        {"predecessor": "aaa", "candidate_count": 1, "ambiguous": False, "why": "one"},
    )
    everything = registry.run("bbb", "revision_diff", {}, layout="Model", units={})
    restricted = registry.run(
        "bbb", "revision_diff", {"scope": "layout"}, layout="Model", units={}
    )

    assert everything["diff"]["totals"]["buckets"]["changed"] == 1
    assert restricted["diff"]["totals"]["buckets"]["changed"] == 0
    assert "every layout" in everything["scope_note"]
    assert "layout 'Model' only" in restricted["scope_note"]


def test_an_unknown_scope_is_refused_rather_than_forced(monkeypatch):
    _stub_store(
        monkeypatch,
        {"bbb": dossier("bbb", [bucket("L1")])},
        {"predecessor": None, "candidate_count": 0},
    )
    with pytest.raises(registry.RecipeRefused) as caught:
        registry.run("bbb", "revision_diff", {"scope": "everything"}, layout="Model", units={})
    assert caught.value.code == "RECIPE_PARAM_INVALID"


# =============================================================================
# The arrival path — it may never cost an ingest
# =============================================================================


class _FakeCollection:
    def __init__(self, documents: dict[str, dict[str, Any]] | None = None):
        self.documents = documents or {}
        self.writes: list[str] = []

    def find_one(self, query, projection=None):
        return self.documents.get(str(query.get("_id")))

    def replace_one(self, query, document, upsert=False):
        self.documents[str(query.get("_id"))] = document
        self.writes.append(str(query.get("_id")))

    def update_one(self, query, update):
        target = self.documents.setdefault(str(query.get("_id")), {})
        target.update(update.get("$set") or {})

    def find(self, query=None, projection=None):
        return list(self.documents.values())


def _arrival_env(monkeypatch, *, drawing=None, collection=None):
    """A store and a review directory that exist only inside this test.

    The draft writer is stubbed and that is not tidiness: `profile_on_arrival`
    really does write a YAML file into whatever `CAD_LANDUSE_DRAFT_DIR` names,
    and a test that ran for real would leave a review file for a drawing that
    does not exist sitting in an operator's directory. Caught by noticing an
    `abc.draft.yaml` beside twenty real ones.
    """
    from app import landuse_draft

    collection = collection if collection is not None else _FakeCollection()
    monkeypatch.setattr(
        ingest.store, "get_drawing", lambda drawing_id: drawing, raising=False
    )
    monkeypatch.setattr(ingest, "coll", lambda name: collection)
    monkeypatch.setattr(
        landuse_draft,
        "draft_for_drawing",
        lambda drawing, **kwargs: {"drawing_id": "test", "counts": {}},
    )
    monkeypatch.setattr(
        landuse_draft, "write_draft", lambda draft, **kwargs: pathlib.Path("(not written)")
    )
    return collection


def test_the_arrival_path_never_raises_when_the_store_cannot_be_read(monkeypatch):
    def boom(drawing_id):
        raise RuntimeError("mongo is down")

    _arrival_env(monkeypatch, drawing=None)
    monkeypatch.setattr(ingest.store, "get_drawing", boom, raising=False)
    out = ingest.profile_on_arrival("abc")

    assert out["ok"] is False
    assert out["errors"], "a failure must be reported, not merely survived"
    assert out["dossier"]["ok"] is False


def test_a_broken_profiler_does_not_fail_the_ingest(monkeypatch):
    """The whole point of the hook's error handling, in one test.

    A drawing that is stored but not yet profiled is recoverable — the next
    sweep or the backfill completes it. An ingest that died inside a profiler
    has lost the entity data as well, and re-reading a 136 MB DXF is the
    expensive half.
    """
    collection = _arrival_env(monkeypatch, drawing={"_id": "abc", "layouts": []})
    monkeypatch.setattr(
        ingest,
        "_backfill_module",
        lambda: (_Raising(), "a module whose build_for raises"),
    )
    out = ingest.profile_on_arrival("abc")

    assert out["ok"] is False
    assert out["dossier"]["ok"] is False
    assert "Dossier build failed" in " ".join(out["errors"])
    assert collection.writes == [], "nothing half-built may be stored"


class _Raising:
    def build_for(self, drawing):  # noqa: D102 - a profiler that blows up
        raise ValueError("the profiler exploded")


def test_a_missing_drawing_document_is_reported_not_invented(monkeypatch):
    _arrival_env(monkeypatch, drawing=None)
    out = ingest.profile_on_arrival("abc")

    assert out["ok"] is False
    assert "no drawing document" in out["dossier"]["why"]


def test_the_arrival_failure_reaches_the_ingest_result_and_the_report():
    result = ingest.FileResult(filename="x.dxf", ok=True)
    ingest._attach_arrival(
        result,
        {
            "ok": False,
            "errors": ["Dossier build failed: ValueError: boom"],
            "dossier": {"ok": False, "why": "ValueError: boom"},
            "landuse_draft": {"ok": True, "path": "/tmp/x.draft.yaml"},
            "revision": {"ok": True, "predecessor": None},
        },
    )

    assert result.arrival is not None
    assert any("arrival:" in w for w in result.warnings)
    printed = "\n".join(ingest._arrival_lines(result.arrival))
    assert "NOT BUILT" in printed


def test_the_report_distinguishes_not_attempted_from_failed():
    """`arrival: None` is "nobody asked", not "it went wrong"."""
    not_attempted = "\n".join(ingest._arrival_lines(None))
    assert "not attempted" in not_attempted


def test_the_report_says_when_a_predecessor_was_ambiguous():
    printed = "\n".join(
        ingest._arrival_lines(
            {
                "ok": True,
                "errors": [],
                "dossier": {"ok": True, "built": True, "buckets": 3, "accounted": 9, "total": 9, "coverage_complete": True},
                "landuse_draft": {"ok": True, "path": "p", "ready_to_accept": 1, "needs_a_decision": 2},
                "revision": {
                    "ok": True,
                    "predecessor": None,
                    "ambiguous": True,
                    "candidate_count": 2,
                    "candidates": ["aaa", "bbb"],
                },
            }
        )
    )
    assert "Ambiguous" in printed
    assert "aaa, bbb" in printed


def test_the_hook_runs_on_both_the_fresh_and_the_skipped_ingest_path():
    """A skipped file must still gain a Dossier when it has none.

    Eighteen drawings were ingested before this hook existed. Without the call
    on the skip path a sweep keeps reporting them SKIP/OK while they stay
    unprofiled for ever — which is the same shape as the half-finished-ingest
    bug already recorded in `drawing_is_current`.
    """
    import inspect

    source = inspect.getsource(ingest.ingest_file)
    # Call sites, not mentions: the prose above them names the function too,
    # and a test that counts words breaks the next time somebody explains
    # something.
    assert source.count("profile_on_arrival(") == 2
    assert "rebuild=False" in source, "the skip path must not rebuild for nothing"


def test_a_drawing_that_already_has_a_dossier_is_left_alone_on_the_skip_path(monkeypatch):
    collection = _arrival_env(
        monkeypatch,
        drawing={"_id": "abc", "layouts": []},
        collection=_FakeCollection({"abc": {"_id": "abc", "layers": []}}),
    )
    monkeypatch.setattr(ingest, "_backfill_module", lambda: (None, "not here"))
    out = ingest.profile_on_arrival("abc", rebuild=False)

    assert out["dossier"]["ok"] is True
    assert out["dossier"]["built"] is False
    assert collection.writes == []


def test_a_console_that_cannot_encode_the_report_still_gets_it(capsys):
    """A report lost to `UnicodeEncodeError` is a report nobody read.

    Measured: an em dash inside a diff verdict took down the ingest report on
    a cp1252 terminal after every drawing had been stored and profiled
    correctly — nothing was wrong and everything looked broken.
    """
    ingest._say("1 fewer entities (31 → 30) — Σ length 880.0 … done")
    printed = capsys.readouterr().out
    assert "->" in printed
    assert "→" not in printed
    assert "sum length" in printed


# =============================================================================
# The standalone script
# =============================================================================

#: `scripts/` is genuinely absent from the one-off container the suite's own
#: runner uses, and a module-level load that raises would take the WHOLE suite
#: down at collection time. A file this module cannot reach is a reason to skip
#: these three tests, never a reason to stop everything else from being checked.
_cli = None
_why_no_cli = ""
if SCRIPT_PATH.is_file():
    _spec = importlib.util.spec_from_file_location("dossier_diff", SCRIPT_PATH)
    if _spec is not None and _spec.loader is not None:
        _cli = importlib.util.module_from_spec(_spec)
        try:
            _spec.loader.exec_module(_cli)
        except Exception as exc:  # noqa: BLE001
            _cli, _why_no_cli = None, f"{SCRIPT_PATH} could not be loaded: {exc!r}"
else:
    _why_no_cli = f"{SCRIPT_PATH} is not on this machine"

needs_cli = pytest.mark.skipif(_cli is None, reason=_why_no_cli or "no script")


@needs_cli
def test_the_printed_report_states_every_section_a_reviewer_needs():
    a = dossier("aaa", [bucket("L1", entities=7, value=880.0), bucket("GONE", entities=2)])
    b = dossier("bbb", [bucket("L1", entities=5, value=560.0), bucket("NEW", entities=1)])
    text = "\n".join(_cli.report_lines(rev.diff_dossiers(a, b)))

    for expected in ("VERDICT", "totals", "appeared", "disappeared", "changed", "CANNOT see"):
        assert expected in text, f"the printed report never mentions {expected!r}"


@needs_cli
def test_the_printed_delta_keeps_the_precision_the_dossier_stored():
    """`%g` printed a measured +18.954857 as "+18.9549"; a printed number that
    disagrees with the number it prints is worse than no number."""
    assert _cli._signed(18.954857) == "+18.954857"
    assert _cli._signed(-320.0) == "-320"
    assert _cli._signed(None) == "?"


@needs_cli
def test_the_printed_report_never_writes_a_unit_it_was_not_given():
    assert "drawing units (none declared)" in _cli._fig(5.0, None)
    assert _cli._fig(None, "m") == "absent (not zero)"


# =============================================================================
# The live acceptance — the synthetic revision pair, end to end
# =============================================================================


def _live_pair():
    """`(manifest, before, after)` from the real store, or a reason to skip.

    Everything here is reached exactly as a person would reach it: the two
    drawings are found by the NAME they share, and their Dossiers are read
    from the collection the ingest hook wrote them to. Nothing is built by
    this test, which is the whole claim being checked — that ingesting
    revision B was enough.
    """
    if not MANIFEST_PATH.is_file():
        pytest.skip(
            f"{MANIFEST_PATH} is not on this machine. Build it with: python "
            "scripts/dossier_diff.py --make-revision-pair "
            "data/dxf/<small>.dxf --out data/dxf/revision_pair"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    try:
        from app.mongo import COLL_DOSSIERS, COLL_DRAWINGS, coll

        rows = list(
            coll(COLL_DRAWINGS).find(
                {}, {"_id": 1, "original_filename": 1, "source_path": 1}
            )
        )
    except Exception as exc:  # noqa: BLE001 -- no store here is a skip, not a fail
        pytest.skip(f"the store is not reachable from here: {type(exc).__name__}")

    name = manifest["files"]["name"]
    matching = [r for r in rows if str(r.get("original_filename")) == name]
    ids = {}
    for row in matching:
        path = str(row.get("source_path") or "").replace("\\", "/")
        for side in ("rev_a", "rev_b"):
            if f"/{side}/" in path:
                ids[side] = str(row["_id"])
    if len(ids) != 2:
        pytest.skip(
            f"the synthetic revision pair is not in this store ({len(ids)} of 2 "
            "found). Ingest both files under data/dxf/revision_pair first."
        )

    before = coll(COLL_DOSSIERS).find_one({"_id": ids["rev_a"]})
    after = coll(COLL_DOSSIERS).find_one({"_id": ids["rev_b"]})
    if before is None or after is None:
        pytest.fail(
            "both revisions are in the store but at least one has NO Dossier. "
            "That is exactly the hole DOSSIER Phase 7 closes: ingest is "
            "supposed to build it with no manual step."
        )
    return manifest, before, after, ids


def test_the_planted_changes_and_nothing_else():
    """The acceptance: the diff reports the planted edits, and only those.

    The expectation is not written here. It is read from the manifest the
    fixture builder produced by re-extracting both DXF files through
    `app.extract` — a different path from the Dossier and from the diff, which
    is what makes it an expectation rather than a restatement.
    """
    manifest, before, after, _ids = _live_pair()
    expected = manifest["expected_bucket_changes"]
    diff = rev.diff_dossiers(before, after)

    assert changed_keys(diff) == {
        (row[0], row[1]) for row in expected["bucket_keys"]
    }
    assert diff["totals"]["entity_total"]["from"] == expected["entities_from"]
    assert diff["totals"]["entity_total"]["to"] == expected["entities_to"]

    for row in expected["buckets"]:
        key = (row["layer"], row["layout"])
        if row["entities_from"] is None:
            added = [
                r for r in diff["layers_added"] if (r["layer"], r["layout"]) == key
            ]
            assert len(added) == 1
            assert added[0]["entities"] == row["entities_to"]
            continue
        found = one(diff, row["layer"], row["layout"])
        assert found["entities"]["from"] == row["entities_from"]
        assert found["entities"]["to"] == row["entities_to"]
        if row["length_from"] and found["measure"]["kind"]["to"] == "length":
            assert found["measure"]["compared"] is True
            assert found["measure"]["delta"] == pytest.approx(
                row["length_to"] - row["length_from"], abs=1e-5
            )


def test_the_moved_entity_is_invisible_and_the_report_says_why():
    """The fourth planted edit. It is NOT reported, and that is correct.

    A translation changes no aggregate a Dossier holds. The fixture builder
    verified that by re-extracting both files, so this is a measured blind
    spot rather than a hopeful one — and the diff names it in `cannot_see`
    instead of letting silence stand for "nothing happened".
    """
    manifest, before, after, _ids = _live_pair()
    moved_layer = manifest["moved"]["layer"]
    assert manifest["move_verified"]["measure_unchanged"] is True

    diff = rev.diff_dossiers(before, after)
    assert moved_layer not in {layer for layer, _layout in changed_keys(diff)}
    assert any("MOVED" in row["what"] for row in diff["cannot_see"])


def test_both_revisions_were_profiled_by_ingest_with_no_manual_step():
    """(a) the Dossier exists, and it accounts for the whole file."""
    _manifest, before, after, _ids = _live_pair()
    for side, document in (("rev_a", before), ("rev_b", after)):
        coverage = document.get("coverage") or {}
        assert coverage.get("complete") is True, f"{side}: {coverage.get('verdict')}"


def test_the_later_revision_recorded_its_predecessor_at_ingest():
    """(c) the diff against the predecessor is available without running anything."""
    _manifest, _before, after, ids = _live_pair()
    block = after.get("revision")
    assert block is not None, (
        "the later revision carries no `revision` block, so ingest never asked "
        "the predecessor question"
    )
    assert block["predecessor"] == ids["rev_a"]
    assert block["ambiguous"] is False
    assert block["diff_summary"]["against"] == ids["rev_a"]
    assert block["diff_summary"]["entity_total"]["delta"] == -1


def test_the_earlier_revision_records_that_it_had_no_predecessor():
    """Absence is recorded, not left blank: "never checked" must not look like
    "checked and there is none"."""
    _manifest, before, _after, _ids = _live_pair()
    block = before.get("revision")
    assert block is not None
    assert block["predecessor"] is None
    assert block["candidate_count"] == 0
    assert block["diff_summary"] is None


def test_the_land_use_draft_was_written_for_both_revisions():
    """(b) the draft config exists on disk, keyed by the content hash.

    Written by the ingest hook into whatever review directory the environment
    names. The test finds it the same way anything else would — by asking
    `landuse_draft` where it would be — and skips if this machine is not the
    one that ingested them.
    """
    _manifest, _before, _after, ids = _live_pair()
    from app import landuse_draft

    missing = [
        side
        for side, drawing_id in ids.items()
        if not landuse_draft.draft_path(drawing_id).is_file()
    ]
    if missing:
        pytest.skip(
            "the drafts were written by whichever machine ran the ingest, and "
            f"this one has none for {missing} at "
            f"{landuse_draft.default_draft_dir()}. Set CAD_LANDUSE_DRAFT_DIR "
            "to the directory that ingest wrote to."
        )
    for drawing_id in ids.values():
        text = landuse_draft.draft_path(drawing_id).read_text(encoding="utf-8")
        # It must classify nothing on its own: `layers:` is empty and every
        # proposal beside it is commented out.
        assert "layers:" in text
        assert landuse_draft.ACCEPT_MARKER in text
