"""`app/dossier_read.py` -- the only module that reads `autocad_dossiers`.

Two parts, following the habit of `test_landuse.py` and `test_measure.py`.

The first part is **fixtures only**: `dossier_read` reaches the store through
exactly one private function, `_fetch`, so replacing that one function removes
the database entirely and every rule below gets a guard that is alive on a
laptop, in CI, and on the eighteenth drawing nobody has looked at yet.

The second part reads the **real store** and skips while naming its reason if
it is not there. It exists for the figures that are properties of the stored
data rather than of this code -- 1,233 road entities, 83,962.565 m, three
copies of 27,987.522 -- and for one invariant that must hold over EVERY
drawing in the store, not only the comfortable one (G5). `pytest -rs` names
what was skipped; read that before trusting a green run.

The tests that matter most here are the ones written to catch a collapse
rather than to confirm a success:

* `test_missing_dossier_is_none_not_empty` and its `..._present_but_empty`
  twin. This whole campaign exists because "0" came back for something that
  had never been looked at. Both paths are asserted explicitly, side by side,
  so that a future edit that makes one of them return the other has to delete
  a test on purpose.
* `test_duplicate_warning_never_collapses_not_looked_into_clean`. A `None`
  for "no duplicates" would re-create the same defect inside the one function
  whose signature invites it.
* `test_use_is_a_label_and_never_decides_the_role`. The residual must be
  driven by the caller's TOKENS, never by a table from a land use to a
  geometric role (G1). Swapping the `use` string with the tokens held fixed
  must change nothing at all.
* `test_summary_never_invents_a_unit` / `test_measures_are_never_summed_across_layouts`.
  11 of 18 drawings are in inches and 3 declare nothing (G2).

Drawing-specific strings appear below only as fixtures and pinned reference
figures, which rule G1 exempts (`kecuali di berkas config dan di test`).
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import dossier_read as dr  # noqa: E402


# --- fixture builders ---------------------------------------------------------


def measure(
    kind: str,
    value,
    unit,
    *,
    unit_reason: str | None = None,
    measured: int | None = None,
    unmeasured: int | None = None,
    caveat: str | None = None,
    **extra,
) -> dict:
    """A Dossier measure block. `unit=None` is a first-class state (G2)."""
    out = {
        "kind": kind,
        "value": value,
        "unit": unit,
        "unit_reason": unit_reason
        if unit_reason is not None
        else (None if unit else "the caller established no unit for this bucket"),
        "measured_entities": measured,
        "unmeasured_entities": unmeasured,
        "bucket_entities": (measured or 0) + (unmeasured or 0),
        "basis": f"a {kind} over this bucket",
        "caveat": caveat,
    }
    out.update(extra)
    return out


def bucket(
    layer: str,
    layout: str,
    entities: int,
    role: str | None,
    *,
    measure_block: dict | None = None,
    anomalies=None,
    anomalies_computed: bool | None = True,
    role_shares: dict | None = None,
) -> dict:
    return {
        "layer": layer,
        "layout": layout,
        "entities": entities,
        "types": {},
        "role": role,
        "role_basis": f"{role} decided from geometry alone (rule G1)",
        "role_counts": {},
        "role_shares": role_shares
        or {
            "region": 0.0,
            "network": 0.0,
            "points": 0.0,
            "annotation": 0.0,
            "other": 0.0,
        },
        "role_threshold": 0.6,
        "measure": measure_block or measure("count", entities, None),
        "anomalies": list(anomalies or []),
        "anomalies_status": {"computed": anomalies_computed, "why": None},
    }


def dossier(
    drawing_id: str,
    buckets,
    *,
    entity_total: int | None = None,
    units_name: str | None = None,
    declared_in_file: bool = False,
    declared_but_unprofiled=(),
    unprofiled_dropped: int = 0,
) -> dict:
    buckets = list(buckets)
    accounted = sum(b["entities"] for b in buckets)
    total = entity_total if entity_total is not None else accounted
    return {
        "_id": drawing_id,
        "drawing_id": drawing_id,
        "computed_at": "2026-08-25T00:00:00+00:00",
        "dossier_version": 1,
        "entity_total": total,
        "layouts": [
            {
                "name": name,
                "declared_entities": None,
                "profiled_entities": sum(
                    b["entities"] for b in buckets if b["layout"] == name
                ),
                "agrees": None,
                "is_modelspace": name == "Model",
                "is_block": name.startswith("[block]"),
            }
            for name in sorted({b["layout"] for b in buckets})
        ],
        "layers": buckets,
        "file_facts": {
            "units": {
                "units_name": units_name,
                "units_code": 6 if declared_in_file else 0,
                "declared_in_file": declared_in_file,
            },
            "layers": {
                "declared": len(declared_but_unprofiled) + unprofiled_dropped,
                "declared_but_unprofiled": len(declared_but_unprofiled)
                + unprofiled_dropped,
                "declared_but_unprofiled_items": list(declared_but_unprofiled),
                "declared_but_unprofiled_truncation": {
                    "cap": 25,
                    "present": len(declared_but_unprofiled) + unprofiled_dropped,
                    "reported": len(declared_but_unprofiled),
                    "dropped": unprofiled_dropped,
                },
            },
        },
        "coverage": {
            "accounted": accounted,
            "total": total,
            "difference": total - accounted,
            "complete": total == accounted,
            "buckets": len(buckets),
            "verdict": "COMPLETE" if total == accounted else "NOT COMPLETE",
        },
    }


DUPLICATE_ANOMALY = {
    "kind": "duplicate_clusters",
    "detail": "3 spatially detached clusters hold 411 entities each",
    "counts": {"duplicate_groups": 1, "clusters": 3, "entities_involved": 1233},
    "evidence": {
        "instrument": "spatial_cluster",
        "unit": "m",
        "groups": [
            {
                "members": 3,
                "entities_each": 411,
                "entities_involved": 1233,
                "bbox_width": 1778.3872624223586,
                "bbox_height": 1585.990376851987,
                "length_total_each": 27987.521669703445,
                "length_agrees": True,
                "clusters": [],
            }
        ],
    },
}


@pytest.fixture
def store(monkeypatch):
    """Replace the module's single database seam with a dict of documents.

    Everything in `dossier_read` reads through `_fetch`, so this is the whole
    database as far as the first half of this file is concerned.
    """

    documents: dict[str, dict] = {}
    failure: dict[str, str] = {}

    def fake_fetch(drawing_id, *, projection=None):
        if failure:
            return None, failure["why"]
        if not isinstance(drawing_id, str) or not drawing_id.strip():
            return None, "no drawing id was given"
        document = documents.get(drawing_id)
        if document is None:
            return None, "no Dossier has been built for this drawing yet"
        return json.loads(json.dumps(document)), None

    monkeypatch.setattr(dr, "_fetch", fake_fetch)

    class Store:
        def add(self, document):
            documents[document["_id"]] = document
            return document

        def break_connection(self, why="the Dossier store could not be read"):
            failure["why"] = why

    return Store()


# --- reference fixtures -------------------------------------------------------

ROAD = "00_Prop - Road - CL_"
SHEET_ROAD = "C-ROAD-PROF-GRID-MINR"
ROAD_TOKENS = ("road", "roads", "street", "carriageway", "highway", "junction")


def janadriyah_like() -> dict:
    """The shape the residual rule was written against, in miniature.

    A real road network and a Civil3D sheet-layout layer that merely has the
    word in its name, plus one classified parcel layer and one layer that
    matches nothing.
    """
    return dossier(
        "fixture-road",
        [
            bucket(
                ROAD,
                "Model",
                1233,
                "network",
                measure_block=measure(
                    "length",
                    83962.565009,
                    "m",
                    measured=1191,
                    unmeasured=42,
                    caveat="length_total is a FLOOR: 42 entities carry no length",
                    chains=849,
                    snap_tolerance=0.0433751428,
                    value_is_floor=True,
                ),
                anomalies=[DUPLICATE_ANOMALY],
                role_shares={
                    "region": 0.0,
                    "network": 0.997567,
                    "points": 0.002433,
                    "annotation": 0.0,
                    "other": 0.0,
                },
            ),
            bucket(
                ROAD,
                "[block] A$C1d2a4b45",
                411,
                "network",
                measure_block=measure(
                    "length",
                    27987.52167,
                    None,
                    unit_reason=(
                        "these numbers are inside a block definition and each "
                        "insertion scales them"
                    ),
                    measured=397,
                    unmeasured=14,
                ),
            ),
            bucket(
                SHEET_ROAD,
                "Model",
                6,
                "annotation",
                measure_block=measure("count", 6, None),
            ),
            bucket("VL2", "Model", 133, "region", measure_block=measure("area", 10084.02, "m2")),
            bucket("DIM", "Model", 9738, "annotation", measure_block=measure("count", 9738, None)),
        ],
        units_name="m",
        declared_in_file=True,
        declared_but_unprofiled=["C-ESMT-ROAD", "A-BLDG-FPRT"],
        unprofiled_dropped=185,
    )


# ==============================================================================
# Part 1 -- fixtures only. No database.
# ==============================================================================


# --- the distinction the whole campaign is about ------------------------------


def test_missing_dossier_is_none_not_empty(store):
    """`None` means NOT COMPUTED, at every entry point.

    This is the half of the pair that must never quietly become a well-formed
    empty answer: a caller that receives `None` has to say "this has not been
    computed", and it cannot do that if the module hands it a tidy zero.
    """
    assert dr.dossier_for("never-built") is None
    assert dr.summary("never-built") is None
    assert dr.residual_for("never-built", "road", ROAD_TOKENS, []) is None
    assert dr.duplicate_warning("never-built", layer=ROAD, layout="Model") is None
    # `profiles_for` returns a mapping by contract, so its "nothing" is `{}` --
    # which is why it is the caller's job to have asked `summary`/`dossier_for`
    # first. The docstring says so out loud.
    assert dr.profiles_for("never-built", [ROAD]) == {}


def test_present_but_nothing_matched_is_a_checked_empty(store):
    """The other half. A Dossier that exists and genuinely has no match is a
    well-formed block with an EMPTY LIST -- and it says what it checked."""
    store.add(janadriyah_like())

    block = dr.residual_for(
        "fixture-road", "cemetery", ("cemetery", "graveyard"), [], layout="Model"
    )

    assert block is not None
    assert block["matches"] == []
    assert block["matches_total"] == 0
    assert block["checked"] is True
    assert block["searchable"] is True
    assert block["unclassified_layers_examined"] > 0
    assert "CHECKED and none found" in block["verdict"]
    # and it must not overclaim: names were checked, not meanings.
    assert "NAMES" in block["verdict"] or "names" in block["caveat"]


def test_no_tokens_is_a_third_state_and_not_a_checked_zero(store):
    """A use config lists no vocabulary for is an UNASKED QUESTION.

    `evidence.load_vocabulary` allows an empty token list on purpose -- such a
    claim can never be `stated`. An empty result there must not read like the
    empty result above, or the difference between "we looked and found none"
    and "there was nothing to look for" is lost.
    """
    store.add(janadriyah_like())

    block = dr.residual_for("fixture-road", "unknown", (), [], layout="Model")

    assert block is not None
    assert block["matches"] == []
    assert block["searchable"] is False
    assert block["checked"] is False
    assert "NOT CHECKED" in block["verdict"]


def test_store_failure_is_reported_as_a_different_reason(store):
    """A database that is down is not a Dossier that was never built.

    Both come back as `None` -- the contract says so -- but the reason survives
    for `not_computed_block`, so an agent is never told "not computed" when the
    truth is "nothing was established either way".
    """
    store.break_connection("the Dossier store could not be read (ServerSelectionTimeoutError)")

    assert dr.summary("fixture-road") is None
    block = dr.not_computed_block("fixture-road")
    assert block["available"] is False
    assert "could not be read" in block["why"]
    assert "NOT COMPUTED" in block["meaning"]
    assert "dossier_backfill" in block["how_to_build"]


# --- the residual rule --------------------------------------------------------


def test_the_road_residual(store):
    """The campaign's origin case, and its Definition of Done.

    A use reporting zero, unclassified layers holding entities, and a block
    that publishes what those layers ARE -- ranked so the layer that holds the
    geometry is first, and profiled so the sheet-layout layer beside it is
    visibly a different thing.
    """
    store.add(janadriyah_like())

    block = dr.residual_for(
        "fixture-road", "road", ROAD_TOKENS, ["VL2"], layout="Model"
    )

    assert block is not None
    assert [m["layer"] for m in block["matches"]] == [ROAD, SHEET_ROAD]

    network, sheet = block["matches"]
    assert network["role"] == "network"
    assert network["entities"] == 1233
    assert network["measure"]["kind"] == "length"
    assert network["measure"]["value"] == 83962.565009
    assert network["measure"]["unit"] == "m"
    assert network["matched_token"] == "road"

    # The name proposes; the geometry disposes. Same word, different thing.
    assert sheet["role"] == "annotation"
    assert sheet["entities"] == 6
    assert sheet["measure"]["kind"] == "count"

    assert "not an absence" in block["verdict"].lower()
    assert "83962.565009 m" in block["verdict"]


def test_the_residual_never_names_an_already_classified_layer(store):
    """A layer a config already maps is not a residual.

    Matching is case-insensitive on purpose: over-including a layer as
    "classified" only ever costs a duplicate row elsewhere, while
    under-including it puts an answered layer in the block that exists for
    unanswered ones.
    """
    store.add(janadriyah_like())

    block = dr.residual_for(
        "fixture-road", "road", ROAD_TOKENS, ["00_prop - road - cl_"], layout="Model"
    )

    assert [m["layer"] for m in block["matches"]] == [SHEET_ROAD]
    assert block["classified_layers_excluded"] == 1


def test_use_is_a_label_and_never_decides_the_role(store):
    """G1: there is no map from a land use to an expected geometric role.

    Two calls with the SAME tokens and a different `use` must return the same
    matches with the same roles; a call with the same `use` and different
    tokens must return something else. That is the whole difference between a
    vocabulary in config and an ontology in code.
    """
    store.add(janadriyah_like())

    as_road = dr.residual_for("fixture-road", "road", ROAD_TOKENS, [], layout="Model")
    as_nonsense = dr.residual_for(
        "fixture-road", "not_a_real_land_use", ROAD_TOKENS, [], layout="Model"
    )

    assert [m["layer"] for m in as_road["matches"]] == [
        m["layer"] for m in as_nonsense["matches"]
    ]
    assert [m["role"] for m in as_road["matches"]] == [
        m["role"] for m in as_nonsense["matches"]
    ]

    other_tokens = dr.residual_for(
        "fixture-road", "road", ("dimension",), [], layout="Model"
    )
    assert other_tokens["matches"] == []


def test_the_residual_uses_the_repos_own_literal_test(store):
    """Matching is `evidence.speaks_for`, not `in`.

    `CAR PARKING AREA` does not state `park`, and a token has to appear as a
    WORD. Re-implementing that here with a substring test would quietly undo
    the fence that stops a three-letter token matching half the layer table.
    """
    store.add(
        dossier(
            "fixture-words",
            [
                bucket("CAR PARKING AREA", "Model", 40, "region"),
                bucket("Central Park", "Model", 12, "region"),
            ],
        )
    )

    block = dr.residual_for("fixture-words", "open_space", ("park",), [], layout="Model")

    assert [m["layer"] for m in block["matches"]] == ["Central Park"]


def test_declared_but_empty_layers_say_when_their_search_was_partial(store):
    """A truncated haystack cannot produce a finding of "none".

    The Dossier stores the declared-but-empty layer list already capped (25 of
    210 on the reference drawing), so a search over it is incomplete and must
    say so. "Found none in a list I could only half see" is exactly the shape
    of claim this campaign exists to end.
    """
    store.add(janadriyah_like())

    block = dr.residual_for("fixture-road", "road", ROAD_TOKENS, [], layout="Model")

    assert block["declared_but_empty_matches"] == ["C-ESMT-ROAD"]
    assert block["declared_but_empty_search_complete"] is False
    assert "does NOT mean there are none" in block["declared_but_empty_basis"]


# --- size budgets (G7) --------------------------------------------------------


def test_a_truncated_summary_counts_what_it_dropped(store):
    """176 buckets do not go into every `describe_drawing` call.

    What is capped is the NAMING. The counts and role totals beside it are
    computed over every bucket in scope, so a truncated list never moves a
    published number.
    """
    buckets = [
        bucket(f"L{i:03d}", "Model", 1000 - i, "region",
               measure_block=measure("area", float(i), "m2"))
        for i in range(40)
    ]
    store.add(dossier("fixture-big", buckets))

    block = dr.summary("fixture-big", top=5)

    assert len(block["layers"]) == 5
    truncation = block["truncation"]
    assert truncation["cap_applied"] == 5
    assert truncation["layers_present"] == 40
    assert truncation["layers_dropped"] == 35
    assert truncation["entities_dropped"] == sum(1000 - i for i in range(5, 40))
    # ranked by entity count, biggest first
    assert [row["layer"] for row in block["layers"]] == [f"L{i:03d}" for i in range(5)]
    # the totals were NOT truncated with the list
    assert block["entities_in_scope"] == sum(b["entities"] for b in buckets)
    assert block["buckets_in_scope"] == 40
    region = next(r for r in block["roles"] if r["role"] == "region")
    assert region["buckets"] == 40


def test_summary_clamps_a_caller_that_asks_for_too_much_and_says_so(store):
    store.add(janadriyah_like())

    block = dr.summary("fixture-road", top=10_000)

    assert block["truncation"]["cap_requested"] == 10_000
    assert block["truncation"]["cap_applied"] == dr.MAX_SUMMARY_TOP
    assert block["truncation"]["cap_ceiling"] == dr.MAX_SUMMARY_TOP


def test_profiles_for_omits_a_layer_with_no_entry_and_says_when_it_stopped_looking(
    store,
):
    """Two different absences, kept apart.

    A layer with no Dossier entry is ABSENT from the result -- a row of nulls
    would read as "we looked and there is nothing there". A name beyond the
    cap is PRESENT with `profiled: false`, because "we stopped looking" is not
    the same sentence and must not be delivered by omission.
    """
    store.add(janadriyah_like())

    result = dr.profiles_for("fixture-road", [ROAD, "no such layer"], layout="Model")
    assert set(result) == {ROAD}
    assert result[ROAD]["profiled"] is True

    many = [f"ghost-{i}" for i in range(dr.MAX_PROFILE_LAYERS + 3)]
    overflow = dr.profiles_for("fixture-road", [ROAD] + many, layout="Model")
    beyond = [name for name, row in overflow.items() if row.get("profiled") is False]
    assert beyond, "names past the cap must be reported, not dropped"
    assert "NOT examined" in overflow[beyond[0]]["why"]


def test_profiles_for_deduplicates_and_accepts_one_name(store):
    store.add(janadriyah_like())
    assert set(dr.profiles_for("fixture-road", ROAD, layout="Model")) == {ROAD}
    assert set(dr.profiles_for("fixture-road", [ROAD, ROAD], layout="Model")) == {ROAD}


def test_anomaly_flags_are_capped_and_their_evidence_is_not_republished(store):
    flags = [
        {"kind": f"kind_{i}", "detail": "x" * 900, "counts": {"n": i}, "evidence": {"big": "y" * 5000}}
        for i in range(9)
    ]
    store.add(
        dossier("fixture-anom", [bucket("L", "Model", 5, "region", anomalies=flags)])
    )

    profile = dr.profiles_for("fixture-anom", ["L"])["L"]
    anomalies = profile["anomalies"]

    assert anomalies["total"] == 9
    assert anomalies["reported"] == dr.MAX_ANOMALIES_PER_LAYER
    assert anomalies["dropped"] == 9 - dr.MAX_ANOMALIES_PER_LAYER
    assert all(len(f["detail"]) <= dr.MAX_TEXT for f in anomalies["flags"])
    assert all(f["detail_truncated"] for f in anomalies["flags"])
    assert "evidence" not in json.dumps(anomalies["flags"])


# --- units (G2) ---------------------------------------------------------------


def test_summary_never_invents_a_unit(store):
    """A drawing that declares nothing gets nothing. Never a defaulted metre.

    3 of the 18 drawings in the store declare no unit at all and 11 are in
    inches, so a module that wrote "m" because the drawing it was written
    against used metres would be wrong on most of the corpus.
    """
    store.add(
        dossier(
            "fixture-unitless",
            [
                bucket(
                    "PATHS",
                    "Model",
                    50,
                    "network",
                    measure_block=measure("length", 1234.5, None),
                ),
                bucket(
                    "RINGS",
                    "Model",
                    20,
                    "region",
                    measure_block=measure("area", 99.0, None),
                ),
            ],
            units_name=None,
            declared_in_file=False,
        )
    )

    block = dr.summary("fixture-unitless")

    assert block["units"]["declared_in_file"] is False
    for row in block["layers"]:
        assert row["measure"]["unit"] is None
        assert row["measure"]["unit_reason"]
    for role in block["roles"]:
        for total in role["totals"]:
            assert total["unit"] is None
    assert '"unit": "m"' not in json.dumps(block)


def test_role_totals_are_grouped_by_unit_and_never_added_across_them(store):
    """Metres and a unit-less block definition are two totals, not one.

    A single summed number here would be in no unit whatsoever, and would look
    exactly like a number that had one.
    """
    store.add(janadriyah_like())

    block = dr.summary("fixture-road")
    network = next(r for r in block["roles"] if r["role"] == "network")
    units = {t["unit"] for t in network["totals"]}

    assert units == {"m", None}
    metres = next(t for t in network["totals"] if t["unit"] == "m")
    assert metres["value"] == pytest.approx(83962.565009)
    unitless = next(t for t in network["totals"] if t["unit"] is None)
    assert unitless["value"] == pytest.approx(27987.52167)
    assert unitless["unit_reason"]


def test_a_bucket_with_no_value_is_counted_never_summed_as_zero(store):
    """G8. An absent measurement is an absence, and it says so beside the
    total rather than being folded into it."""
    store.add(
        dossier(
            "fixture-absent",
            [
                bucket(
                    "A", "Model", 10, "network",
                    measure_block=measure("length", 100.0, "m"),
                ),
                bucket(
                    "B", "Model", 10, "network",
                    measure_block=measure("length", None, "m"),
                ),
            ],
        )
    )

    block = dr.summary("fixture-absent")
    network = next(r for r in block["roles"] if r["role"] == "network")

    assert network["buckets_without_value"] == 1
    assert [t["value"] for t in network["totals"]] == [100.0]
    absent = next(r for r in block["layers"] if r["layer"] == "B")
    assert absent["measure"]["value"] is None
    assert "not zero" in absent["measure"]["value_is_absent_not_zero"]


def test_measures_are_never_summed_across_layouts(store):
    """One layer, two layouts, two different units -- and one profile.

    The profile describes the largest bucket and NAMES the other; it does not
    add them. The reference road layer is the case that proves it: 83,962.565
    m in model space, 27,987.522 with no unit at all inside a block.
    """
    store.add(janadriyah_like())

    profile = dr.profiles_for("fixture-road", [ROAD])[ROAD]

    assert profile["layout"] == "Model"
    assert profile["measure"]["value"] == 83962.565009
    assert profile["measure"]["unit"] == "m"
    assert profile["scope"]["buckets_for_this_layer"] == 2
    assert profile["scope"]["entities_all_buckets"] == 1233 + 411
    other = profile["scope"]["other_layouts"][0]
    assert other["layout"] == "[block] A$C1d2a4b45"
    assert other["unit"] is None
    assert "never summed across layouts" in profile["scope"]["chosen_basis"]


def test_layout_scope_is_honoured(store):
    store.add(janadriyah_like())

    model_only = dr.profiles_for("fixture-road", [ROAD], layout="Model")[ROAD]
    assert model_only["scope"]["buckets_for_this_layer"] == 1
    assert model_only["scope"]["other_layouts"] == []

    block_only = dr.profiles_for("fixture-road", [ROAD], layout="[block] A$C1d2a4b45")
    assert block_only[ROAD]["measure"]["unit"] is None

    assert dr.profiles_for("fixture-road", [ROAD], layout="No Such Layout") == {}


# --- the duplicate warning ----------------------------------------------------


def test_duplicate_warning_names_the_cluster_count_and_the_per_copy_figure(store):
    """83,962.565 m is three copies of 27,987.522 m.

    A total quoted without that is three times the truth, so the warning
    carries both numbers, both units, and the ratio between them -- computed
    from the two published figures rather than asserted.
    """
    store.add(janadriyah_like())

    warning = dr.duplicate_warning("fixture-road", layer=ROAD, layout="Model")

    assert warning["warn"] is True
    assert warning["checked"] is True
    group = warning["groups"][0]
    assert group["copies"] == 3
    assert group["entities_each"] == 411
    assert group["per_copy"]["value"] == pytest.approx(27987.5216697)
    assert group["per_copy"]["unit"] == "m"
    assert group["layer_total"]["value"] == 83962.565009
    assert group["total_over_per_copy"] == pytest.approx(3.0)
    assert "3 identical copies" in warning["statement"]


def test_duplicate_warning_never_collapses_not_looked_into_clean(store):
    """The one signature that invites the campaign's original defect.

    `None` for "no duplicates" would make "not computed" and "nothing found"
    the same return value. So `None` means "no Dossier" and nothing else, and
    the three other outcomes are told apart by `warn` and `checked`.
    """
    store.add(janadriyah_like())

    clean = dr.duplicate_warning("fixture-road", layer="DIM", layout="Model")
    assert clean["warn"] is False and clean["checked"] is True

    absent_bucket = dr.duplicate_warning("fixture-road", layer="ghost", layout="Model")
    assert absent_bucket["warn"] is False and absent_bucket["checked"] is False
    assert "not a clean bill of health" in absent_bucket["why"]

    store.add(
        dossier(
            "fixture-unchecked",
            [bucket("L", "Model", 10, "network", anomalies_computed=False)],
        )
    )
    unchecked = dr.duplicate_warning("fixture-unchecked", layer="L", layout="Model")
    assert unchecked["warn"] is False and unchecked["checked"] is False

    assert dr.duplicate_warning("never-built", layer="L", layout="Model") is None

    # All three "no warning" answers are distinguishable from each other.
    assert (
        len({(clean["checked"], "bucket"), (absent_bucket["checked"], "no-bucket"),
             (unchecked["checked"], "not-run")})
        == 3
    )


def test_duplicate_warning_states_its_one_sided_failure(store):
    """The detector misses rather than invents, and the response says so, so a
    clean answer is never read as a proof."""
    store.add(janadriyah_like())

    clean = dr.duplicate_warning("fixture-road", layer="DIM", layout="Model")

    assert "one-sided" in clean["meaning"]
    assert "missed finding, never an invented one" in clean["meaning"]


def test_duplicate_warning_across_every_layout(store):
    """`layout=None` means a `measure` call that was not scoped to one."""
    store.add(janadriyah_like())

    warning = dr.duplicate_warning("fixture-road", layer=ROAD, layout=None)

    assert warning["warn"] is True
    assert warning["groups"][0]["layout"] == "Model"


# --- degrading gracefully (G8) ------------------------------------------------


def test_a_dossier_with_no_layers_is_a_well_formed_empty_answer(store):
    """A drawing that really holds nothing is a DIFFERENT answer from a
    drawing whose Dossier was never built, and both must be sayable."""
    store.add(dossier("fixture-empty", [], entity_total=0))

    block = dr.summary("fixture-empty")

    assert block is not None
    assert block["available"] is True
    assert block["layers"] == []
    assert block["roles"] == []
    assert block["buckets_in_scope"] == 0
    assert block["truncation"]["layers_dropped"] == 0
    assert dr.residual_for("fixture-empty", "road", ROAD_TOKENS, [])["matches"] == []


def test_a_malformed_dossier_does_not_raise(store):
    """A damaged document is normal (G8): report what is missing, do not
    crash the tool that was only enriching its answer."""
    store.add({"_id": "fixture-broken", "layers": "not a list"})

    block = dr.summary("fixture-broken")

    assert block["layers"] == []
    assert block["coverage"]["complete"] is None
    assert block["units"]["declared_in_file"] is None
    assert dr.profiles_for("fixture-broken", ["L"]) == {}
    assert dr.residual_for("fixture-broken", "road", ROAD_TOKENS, [])["matches"] == []


def test_a_caller_handing_in_the_wrong_type_is_told(store):
    store.add(janadriyah_like())
    with pytest.raises(dr.DossierReadError):
        dr.summary("fixture-road", dossier=["not", "a", "mapping"])
    with pytest.raises(dr.DossierReadError):
        dr.profiles_for("fixture-road", 17)


def test_a_prefetched_dossier_is_used_instead_of_a_second_read(store):
    """`land_use_summary` asks for a residual per use that reports zero.

    Six blocks must not be six reads of a 1.4 MB document, so a caller may
    hand the document in. The test proves it is genuinely used: the store is
    broken first, and the answer still arrives.
    """
    document = janadriyah_like()
    store.break_connection()

    block = dr.residual_for(
        "fixture-road", "road", ROAD_TOKENS, [], layout="Model", dossier=document
    )

    assert [m["layer"] for m in block["matches"]] == [ROAD, SHEET_ROAD]


# ==============================================================================
# Part 2 -- on top of the real store. Skips, loudly, when it is not there.
# ==============================================================================

JANADRIYAH = "596212db022a3397"

#: Measured from the live store on 25 August 2026 and pinned in
#: `DOSSIER-01-DESIGN.md` and `DOSSIER-PHASE-1-REPORT.md`.
REF_ROAD_ENTITIES = 1233
REF_ROAD_LENGTH = 83962.565009
REF_ROAD_UNIT = "m"
REF_COPIES = 3
REF_PER_COPY = 27987.521669703445
REF_ENTITY_TOTAL = 46754
REF_BUCKETS = 176


def _mongo_or_skip(drawing_id: str = JANADRIYAH):
    """Skip while naming the reason -- and name the RIGHT one.

    `dossier_read` degrades gracefully by design, so an unreachable database
    and a drawing that was never profiled both arrive here as `None`. A skip
    line that said "has no stored Dossier" when the truth was "no
    MONGODB_URI" would be the same conflation this module exists to prevent,
    in the test report. `_fetch` keeps the two reasons apart; this uses them.
    """
    try:
        document = dr.dossier_for(drawing_id)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no database reachable: {type(exc).__name__}")
    if not document:
        why = dr.not_computed_block(drawing_id)["why"]
        pytest.skip(
            f"no Dossier for the reference drawing {drawing_id}: {why} These "
            "acceptance figures cannot be checked."
        )
    return document


def _all_dossier_ids():
    from app.mongo import COLL_DOSSIERS, coll

    return [d["_id"] for d in coll(COLL_DOSSIERS).find({}, {"_id": 1})]


def test_live_reference_layer_reads_exactly():
    """The profile the whole enrichment exists to deliver, from the real store."""
    _mongo_or_skip()

    profile = dr.profiles_for(JANADRIYAH, [ROAD], layout="Model")[ROAD]

    assert profile["role"] == "network"
    assert profile["entities"] == REF_ROAD_ENTITIES
    assert profile["measure"]["kind"] == "length"
    assert profile["measure"]["value"] == pytest.approx(REF_ROAD_LENGTH)
    assert profile["measure"]["unit"] == REF_ROAD_UNIT
    assert profile["measure"]["unmeasured_entities"] == 42


def test_live_road_residual_is_the_definition_of_done():
    """On the real drawing, with the road layer left unclassified: the
    residual names it, as a network, in metres -- and the layers that merely
    have the word in their name are visibly a different thing."""
    _mongo_or_skip()

    block = dr.residual_for(
        JANADRIYAH, "road", ROAD_TOKENS, [], layout="Model"
    )

    first = block["matches"][0]
    assert first["layer"] == ROAD
    assert first["role"] == "network"
    assert first["entities"] == REF_ROAD_ENTITIES
    assert first["measure"]["value"] == pytest.approx(REF_ROAD_LENGTH)
    assert first["measure"]["unit"] == REF_ROAD_UNIT

    roles = {m["role"] for m in block["matches"]}
    assert len(roles) > 1, (
        "every match profiling as the same role would mean the geometry is "
        "not being read -- the point of the residual is that it separates a "
        "road network from a layer that only sounds like one"
    )


def test_live_duplicate_warning_is_the_triplication():
    _mongo_or_skip()

    warning = dr.duplicate_warning(JANADRIYAH, layer=ROAD, layout="Model")

    assert warning["warn"] is True
    group = warning["groups"][0]
    assert group["copies"] == REF_COPIES
    assert group["entities_each"] == 411
    assert group["per_copy"]["value"] == pytest.approx(REF_PER_COPY)
    assert group["total_over_per_copy"] == pytest.approx(REF_COPIES, abs=0.01)


def test_live_summary_is_size_budgeted_and_states_its_truncation():
    _mongo_or_skip()

    block = dr.summary(JANADRIYAH)

    assert block["entity_total"] == REF_ENTITY_TOTAL
    assert block["coverage"]["complete"] is True
    assert block["coverage"]["buckets"] == REF_BUCKETS
    assert len(block["layers"]) == dr.SUMMARY_TOP_LAYERS
    assert block["truncation"]["layers_dropped"] > 0
    assert block["truncation"]["entities_dropped"] > 0
    # It rides on every describe_drawing call, so it has to stay small enough
    # to be worth carrying. `cad_mcp` budgets the whole response at 90,000
    # characters; this block may not be most of it.
    assert len(json.dumps(block, default=str)) < 45_000


def test_live_every_drawing_republishes_units_verbatim_or_not_at_all():
    """G5 + G2, over EVERY drawing in the store rather than the comfortable one.

    A unit in a summary must be the unit the Dossier bucket carries -- 11 of
    18 drawings are in inches and 3 declare nothing, so a defaulted metre
    would be wrong on most of the corpus and right on the one this code was
    written against.
    """
    _mongo_or_skip()
    ids = _all_dossier_ids()
    assert ids, "the store holds no Dossiers to check"

    checked = 0
    for drawing_id in ids:
        document = dr.dossier_for(drawing_id)
        stored = {
            (str(b.get("layer")), str(b.get("layout"))): (b.get("measure") or {})
            for b in document.get("layers", [])
        }
        block = dr.summary(drawing_id, top=dr.MAX_SUMMARY_TOP)
        assert block is not None
        for row in block["layers"]:
            key = (str(row["layer"]), str(row["layout"]))
            assert row["measure"]["unit"] == stored[key].get("unit")
            assert row["measure"]["value"] == stored[key].get("value")
            if row["measure"]["unit"] is None:
                assert row["measure"]["unit_reason"] is not None
            checked += 1
    assert checked > 0


def test_live_a_drawing_that_declares_no_unit_gets_none():
    """At least one drawing in the store declares no unit at all, and its
    summary must not carry one anywhere."""
    _mongo_or_skip()

    unitless = [
        drawing_id
        for drawing_id in _all_dossier_ids()
        if not (
            (dr.dossier_for(drawing_id).get("file_facts") or {}).get("units") or {}
        ).get("declared_in_file")
    ]
    if not unitless:
        pytest.skip("no unit-less drawing is currently in the store")

    for drawing_id in unitless:
        block = dr.summary(drawing_id, top=dr.MAX_SUMMARY_TOP)
        assert block["units"]["declared_in_file"] in (False, None)
        for row in block["layers"]:
            assert row["measure"]["unit"] is None
        for role in block["roles"]:
            for total in role["totals"]:
                assert total["unit"] is None


def test_live_no_dossier_for_an_unknown_drawing_is_none():
    """The `None` path, on the real store rather than on a fake one."""
    _mongo_or_skip()

    unknown = "0" * 16
    assert dr.dossier_for(unknown) is None
    assert dr.summary(unknown) is None
    assert dr.residual_for(unknown, "road", ROAD_TOKENS, []) is None
    assert dr.profiles_for(unknown, [ROAD]) == {}
    assert dr.duplicate_warning(unknown, layer=ROAD, layout="Model") is None
    assert "no Dossier has been built" in dr.not_computed_block(unknown)["why"]
