"""The Drawing Dossier profiler: fixtures only, no Mongo, no network.

Every fixture in this file is a hand-written list of dicts, which is the whole
point of `app/dossier.py` being pure: the profiler can be interrogated without
an ingested drawing, and the properties below hold on the eighteenth drawing
nobody has looked at yet.

Two tests here were written to fail rather than to pass, and they are the two
that matter:

* `test_coverage_fails_loudly_when_a_bucket_is_dropped` -- an invariant that is
  only ever exercised on correct input is not an invariant, it is a decoration.
  This one removes a bucket and demands that the document say so, in the
  arithmetic AND in words, and that `assert_coverage` stop.
* `test_role_ignores_the_layer_name` -- the C-ROAD trap. A layer named after a
  road but holding closed rings must profile as `region`. Code that reads names
  passes every test written from a drawing whose names happen to be honest.

Drawing-specific strings appear below only as fixtures, which rule G1 exempts
(`kecuali di berkas config dan di test`) -- and the names used are deliberately
ones that LIE about their contents.
"""

from __future__ import annotations

import copy
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import dossier  # noqa: E402


# --- fixture builders ---------------------------------------------------------


def row(handle: str, dxftype: str, **kw) -> dict:
    out = {"handle": handle, "type": dxftype}
    out.update(kw)
    return out


def ring(handle: str, area=None, status: str = "complete") -> dict:
    return row(handle, "LWPOLYLINE", ring_status=status, area=area)


def line(handle: str, length=None) -> dict:
    return row(handle, "LINE", length=length)


def rows_of(dxftype: str, count: int, prefix: str = "H", **kw) -> list[dict]:
    return [row(f"{prefix}{i}", dxftype, **kw) for i in range(count)]


class StubNetwork:
    """Stand-in for lane 2, returning the measured reference figures."""

    calls: list = []

    @staticmethod
    def network_measure(rows, *, unit):
        StubNetwork.calls.append((len(rows), unit))
        return {
            "length_total": 83962.565009,
            "measured_entities": 1191,
            "unmeasured_entities": 42,
            "chains": 3,
            "snap_tolerance": 0.001,
            "tolerance_basis": "stated snap tolerance, published not assumed",
            "caveat": "CIRCLE circumference is inside this sum.",
        }


class ExplodingNetwork:
    @staticmethod
    def network_measure(rows, *, unit):
        raise ZeroDivisionError("a chain of length zero")


class StubAnomaly:
    @staticmethod
    def anomalies(rows, *, layer):
        return [{"kind": "duplicate_clusters", "detail": layer, "evidence": {}, "counts": {}}]


class StubCensus:
    @staticmethod
    def block_census(rows):
        return {"blocks": {"tree": len(rows)}, "truncated": 0}

    @staticmethod
    def annotation_census(rows):
        return {"classes": [], "total": len(rows), "truncated": 0}


@pytest.fixture
def no_lanes(monkeypatch):
    """Lanes 2, 3 and 4 absent -- the state every lane starts in."""
    monkeypatch.setattr(dossier, "_NETWORK", None)
    monkeypatch.setattr(dossier, "_NETWORK_WHY", "no module named dossier_network")
    monkeypatch.setattr(dossier, "_ANOMALY", None)
    monkeypatch.setattr(dossier, "_ANOMALY_WHY", None)
    monkeypatch.setattr(dossier, "_CENSUS", None)
    monkeypatch.setattr(dossier, "_CENSUS_WHY", None)


DRAWING = {
    "_id": "abc123def456",
    "units_name": "m",
    "units_code": 6,
    "entity_count": 10,
    "layouts": [
        {"name": "Model", "entity_count": 10, "is_modelspace": True, "is_block": False}
    ],
    "layers": [{"name": "A"}, {"name": "B"}],
    "xrefs": [
        {"block_name": "x1", "path": "p1", "resolved": False, "kind": "image"},
        {"block_name": "x2", "path": "p2", "resolved": True, "kind": "dwg-xref"},
    ],
    "text_styles": [
        {"name": "s1", "font": "SIMPLEX.SHX"},
        {"name": "s2", "font": "arial.ttf"},
    ],
    "audit_errors": 261,
    "audit_fixes": 3,
}


def two_buckets():
    return {
        ("A", "Model"): [ring(f"a{i}", area=100.0) for i in range(6)],
        ("B", "Model"): [line(f"b{i}", length=5.0) for i in range(4)],
    }


# --- role: geometry, never a name ---------------------------------------------


def test_role_ignores_the_layer_name(no_lanes):
    """The C-ROAD trap: a road-shaped name holding rings is a region."""
    lying_road = dossier.profile_rows(
        [ring(f"r{i}", area=300.0) for i in range(20)],
        layer="C-ROAD-CENTERLINE",
        layout="Model",
        unit="m",
    )
    lying_parcel = dossier.profile_rows(
        [line(f"l{i}", length=45.0) for i in range(20)],
        layer="C-PARCEL-BOUNDARY",
        layout="Model",
        unit="m",
    )
    assert lying_road["role"] == "region"
    assert lying_parcel["role"] == "network"
    # And the same names, swapped, do not move the verdict.
    assert (
        dossier.profile_rows(
            [ring(f"r{i}", area=300.0) for i in range(20)],
            layer="C-PARCEL-BOUNDARY",
            layout="Model",
            unit="m",
        )["role"]
        == "region"
    )


def test_a_closed_ring_and_an_open_one_are_different_families():
    """Type alone decides nothing: `ring_status` is the gate."""
    assert dossier.classify_entity(ring("a", area=1.0)) == "region"
    assert dossier.classify_entity(ring("b", status="open")) == "network"
    assert dossier.classify_entity(ring("c", status=None)) == "network"


def test_every_entity_lands_in_exactly_one_family():
    rows = (
        [ring("r", area=1.0)]
        + [line("l", length=1.0)]
        + [row("i", "INSERT", block_name="tree")]
        + [row("t", "MTEXT", text="2092")]
        + [row("d", "DIMENSION_LINEAR", text="25.00")]
        + [row("h", "HATCH")]
    )
    counts = dossier.family_counts(rows)
    assert sum(counts.values()) == len(rows)
    assert counts == {
        "region": 1,
        "network": 1,
        "points": 1,
        "annotation": 2,
        "other": 1,
    }


def test_the_dominance_threshold_is_named_and_published(no_lanes):
    rows = [ring(f"r{i}", area=1.0) for i in range(7)] + [
        line(f"l{i}", length=1.0) for i in range(3)
    ]
    block = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")
    assert block["role"] == "region"
    assert block["role_threshold"] == dossier.ROLE_DOMINANCE
    # The threshold's VALUE is inside the sentence, not only in the constant.
    assert str(dossier.ROLE_DOMINANCE) in block["role_basis"]
    assert "ROLE_DOMINANCE" in block["role_basis"]
    # The runner-up travels too, so a 70% call is visibly not a 99% call.
    assert block["role_shares"]["region"] == 0.7
    assert block["role_shares"]["network"] == 0.3
    assert "30.0%" in block["role_basis"]


def test_the_threshold_is_above_a_simple_majority_so_ties_cannot_happen():
    """Two families at 0.5 would both qualify under <= 0.5 and the winner would
    be arbitrary. This is the arithmetic reason for the constant's value."""
    assert dossier.ROLE_DOMINANCE > 0.5


def test_a_bucket_below_the_threshold_is_mixed_and_says_why(no_lanes):
    rows = [ring(f"r{i}", area=1.0) for i in range(5)] + [
        line(f"l{i}", length=1.0) for i in range(5)
    ]
    block = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")
    assert block["role"] == "mixed"
    assert "No type family reaches" in block["role_basis"]
    assert block["measure"]["kind"] == "count"
    assert block["measure"]["value"] == 10
    assert "why_no_native_measure" in block["measure"]


def test_a_bucket_that_is_mostly_unrecognised_is_mixed_not_region(no_lanes):
    """`other` drags the winning share down; it never names a role itself."""
    rows = rows_of("HATCH", 7) + [ring(f"r{i}", area=1.0) for i in range(3)]
    block = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")
    assert block["role"] == "mixed"
    assert block["role_counts"]["other"] == 7
    assert "other" in block["role_basis"]


def test_an_empty_bucket_has_no_role_rather_than_a_guessed_one(no_lanes):
    block = dossier.profile_rows([], layer="X", layout="Model", unit="m")
    assert block["role"] is None
    assert block["entities"] == 0
    assert "no geometric role can be decided" in block["role_basis"]
    assert block["measure"]["value"] is None


# --- measures: absence is None and a reason, never zero -----------------------


def test_an_unmeasured_entity_is_counted_not_zeroed(no_lanes):
    rows = [ring("a", area=300.0), ring("b", area=200.0), ring("c", area=None)]
    block = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")
    m = block["measure"]
    assert m["kind"] == "area"
    assert m["value"] == 500.0  # not 500/3, not 166.7, and c is not a zero
    assert m["measured_entities"] == 2
    assert m["unmeasured_entities"] == 1
    assert m["bucket_entities"] == 3
    assert "carry no stored area" in (m["caveat"] or "")


def test_nothing_measurable_gives_none_and_never_zero(no_lanes):
    rows = [ring(f"a{i}", area=None) for i in range(4)]
    m = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")["measure"]
    assert m["value"] is None
    assert m["value"] != 0
    assert m["measured_entities"] == 0
    assert m["unmeasured_entities"] == 4


def test_a_unit_is_never_defaulted_to_metres(no_lanes):
    rows = [ring(f"a{i}", area=10.0) for i in range(4)]
    without = dossier.profile_rows(rows, layer="X", layout="Sheet 1")["measure"]
    assert without["unit"] is None
    assert without["unit_reason"]
    assert "metre" not in str(without["unit"] or "")

    inches = dossier.profile_rows(rows, layer="X", layout="Model", unit="in")["measure"]
    assert inches["unit"] == "in2"
    assert inches["unit_reason"] is None

    metres = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")["measure"]
    assert metres["unit"] == "m2"


def test_a_whole_unit_dict_carries_its_own_reason_through(no_lanes):
    rows = [ring(f"a{i}", area=10.0) for i in range(4)]
    paper = {
        "length_unit": None,
        "area_unit": None,
        "why_no_unit": "these numbers are page geometry on a sheet",
    }
    m = dossier.profile_rows(rows, layer="X", layout="Sheet 1", unit=paper)["measure"]
    assert m["unit"] is None
    assert m["unit_reason"] == "these numbers are page geometry on a sheet"


def test_a_count_carries_no_unit_and_says_so(no_lanes):
    rows = rows_of("INSERT", 12, block_name="tree")
    m = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")["measure"]
    assert m["kind"] == "count"
    assert m["value"] == 12
    assert m["unit"] is None
    assert "not a length or an area" in m["unit_reason"]
    # Counting cannot fail, so this zero is established rather than assumed.
    assert m["unmeasured_entities"] == 0
    assert "counting cannot" in m["basis"]


def test_negative_areas_are_summed_as_stored_and_flagged(no_lanes):
    rows = [ring("a", area=100.0), ring("b", area=-40.0)] + [
        ring(f"c{i}", area=10.0) for i in range(3)
    ]
    m = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")["measure"]
    assert m["value"] == 90.0
    assert "negative" in (m["caveat"] or "")


# --- G7: nothing is silently truncated ----------------------------------------


def test_the_type_map_states_its_cap_and_its_truncation(no_lanes):
    rows = []
    for i in range(dossier.MAX_TYPES_REPORTED + 10):
        rows.extend(rows_of(f"TYPE_{i:03d}", i + 1, prefix=f"t{i}_"))
    block = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")
    trunc = block["types_truncation"]
    assert trunc["cap"] == dossier.MAX_TYPES_REPORTED
    assert trunc["types_present"] == dossier.MAX_TYPES_REPORTED + 10
    assert trunc["types_reported"] == dossier.MAX_TYPES_REPORTED
    assert trunc["types_dropped"] == 10
    assert trunc["entities_dropped"] > 0
    # The count that coverage depends on is untouched by truncation.
    assert block["entities"] == len(rows)
    assert sum(block["types"].values()) == len(rows) - trunc["entities_dropped"]


def test_truncation_is_reported_even_when_nothing_was_dropped(no_lanes):
    block = dossier.profile_rows(
        [line("a", length=1.0)], layer="X", layout="Model", unit="m"
    )
    assert block["types_truncation"]["types_dropped"] == 0
    assert block["types_truncation"]["entities_dropped"] == 0


# --- the guarded seams to lanes 2, 3, 4 ---------------------------------------


def test_a_dossier_still_builds_when_lane_2_is_missing(no_lanes):
    rows = [line(f"l{i}", length=5.0) for i in range(10)]
    m = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")["measure"]
    assert m["kind"] == "length"
    assert m["value"] is None
    assert m["not_computed"] == (
        "dossier_network not available (no module named dossier_network)"
    )
    # Absent, not zero -- and the entity count is still exact.
    assert m["measured_entities"] is None
    assert m["bucket_entities"] == 10


def test_a_lane_that_raises_is_reported_not_propagated(monkeypatch, no_lanes):
    monkeypatch.setattr(dossier, "_NETWORK", ExplodingNetwork)
    rows = [line(f"l{i}", length=5.0) for i in range(10)]
    m = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")["measure"]
    assert m["value"] is None
    assert "raised ZeroDivisionError" in m["not_computed"]
    assert "a chain of length zero" in m["not_computed"]


def test_a_lane_module_without_its_function_is_reported(monkeypatch, no_lanes):
    monkeypatch.setattr(dossier, "_NETWORK", object())
    m = dossier.profile_rows(
        [line("a", length=1.0)], layer="X", layout="Model", unit="m"
    )["measure"]
    assert "does not define it" in m["not_computed"]


def test_missing_anomalies_are_not_reported_as_an_empty_finding(no_lanes):
    """`[]` reads as 'this layer is unremarkable'. Nobody made that claim."""
    block = dossier.profile_rows(
        [line("a", length=1.0)], layer="X", layout="Model", unit="m"
    )
    assert block["anomalies"] == []
    assert block["anomalies_status"]["computed"] is False
    assert block["anomalies_status"]["why"]


def test_lane_3_findings_are_carried_through(monkeypatch, no_lanes):
    monkeypatch.setattr(dossier, "_ANOMALY", StubAnomaly)
    block = dossier.profile_rows(
        [line("a", length=1.0)], layer="ROW", layout="Model", unit="m"
    )
    assert block["anomalies_status"]["computed"] is True
    assert block["anomalies"][0]["kind"] == "duplicate_clusters"
    assert block["anomalies"][0]["detail"] == "ROW"


def test_the_census_is_attached_only_where_it_applies(monkeypatch, no_lanes):
    monkeypatch.setattr(dossier, "_CENSUS", StubCensus)
    points = dossier.profile_rows(
        rows_of("INSERT", 10, block_name="tree"), layer="X", layout="Model"
    )
    texts = dossier.profile_rows(rows_of("MTEXT", 10, text="2092"), layer="X", layout="Model")
    rings = dossier.profile_rows(
        [ring(f"r{i}", area=1.0) for i in range(10)], layer="X", layout="Model"
    )
    assert points["census"]["source"] == "dossier_census.block_census"
    assert points["census"]["blocks"] == {"tree": 10}
    assert texts["census"]["source"] == "dossier_census.annotation_census"
    assert rings["census"]["applicable"] is False
    assert "points" in rings["census"]["why"]


def test_a_missing_census_lane_says_not_computed(no_lanes):
    points = dossier.profile_rows(
        rows_of("INSERT", 5, block_name="tree"), layer="X", layout="Model"
    )
    assert points["census"]["applicable"] is True
    assert "not available" in points["census"]["not_computed"]


# --- the coverage invariant ---------------------------------------------------


def test_the_coverage_invariant_holds_when_every_bucket_is_present(no_lanes):
    doc = dossier.build_dossier(DRAWING, two_buckets(), entity_total=10)
    cov = doc["coverage"]
    assert cov["accounted"] == 10
    assert cov["total"] == 10
    assert cov["difference"] == 0
    assert cov["complete"] is True
    assert cov["buckets"] == 2
    assert cov["verdict"].startswith("COMPLETE")
    dossier.assert_coverage(doc)  # does not raise


def test_coverage_fails_loudly_when_a_bucket_is_dropped(no_lanes):
    """The test this file exists for: the invariant must break, not bend."""
    buckets = two_buckets()
    del buckets[("B", "Model")]  # 4 entities vanish from the account

    doc = dossier.build_dossier(DRAWING, buckets, entity_total=10)
    cov = doc["coverage"]

    assert cov["complete"] is False
    assert cov["accounted"] == 6
    assert cov["total"] == 10
    assert cov["difference"] == 4
    assert cov["verdict"].startswith("NOT COMPLETE")
    # It says WHICH WAY it failed, because under- and over-reporting are
    # different bugs with different fixes.
    assert "in no bucket" in cov["verdict"]
    assert "under-reports" in cov["verdict"]

    with pytest.raises(dossier.CoverageError) as caught:
        dossier.assert_coverage(doc)
    assert "10" in str(caught.value) and "6" in str(caught.value)
    assert caught.value.accounted == 6
    assert caught.value.total == 10
    assert caught.value.difference == 4

    # And the same failure can be made to stop the build itself.
    with pytest.raises(dossier.CoverageError):
        dossier.build_dossier(DRAWING, buckets, entity_total=10, strict=True)


def test_coverage_fails_loudly_when_a_bucket_is_counted_twice(no_lanes):
    buckets = two_buckets()
    buckets[("B", "Model copy")] = list(buckets[("B", "Model")])
    doc = dossier.build_dossier(DRAWING, buckets, entity_total=10)
    cov = doc["coverage"]
    assert cov["complete"] is False
    assert cov["accounted"] == 14
    assert cov["difference"] == -4
    assert "more than once" in cov["verdict"]
    assert "over-reports" in cov["verdict"]
    with pytest.raises(dossier.CoverageError):
        dossier.assert_coverage(doc)


def test_coverage_cannot_pass_when_there_is_no_total_to_check_against(no_lanes):
    """An unknown total filled in from `accounted` would make the invariant a
    tautology that passes on every drawing forever."""
    drawing = {k: v for k, v in DRAWING.items() if k != "entity_count"}
    doc = dossier.build_dossier(drawing, two_buckets(), entity_total=None)
    cov = doc["coverage"]
    assert cov["total"] is None
    assert cov["complete"] is False
    assert cov["difference"] is None
    assert "unknown is not" in cov["verdict"]
    with pytest.raises(dossier.CoverageError):
        dossier.assert_coverage(doc)


def test_coverage_fails_when_the_two_sides_of_the_comparison_disagree(no_lanes):
    """The caller says 10, the drawing document says 11. Both cannot be right,
    and choosing one is a guess."""
    drawing = dict(DRAWING, entity_count=11)
    doc = dossier.build_dossier(drawing, two_buckets(), entity_total=10)
    cov = doc["coverage"]
    assert cov["accounted"] == 10
    assert cov["complete"] is False
    assert cov["disagreement"]
    assert "11" in cov["disagreement"] and "10" in cov["disagreement"]
    with pytest.raises(dossier.CoverageError):
        dossier.assert_coverage(doc)


def test_a_document_with_no_coverage_block_is_refused(no_lanes):
    with pytest.raises(dossier.CoverageError):
        dossier.assert_coverage({"_id": "x"})


def test_file_level_facts_are_never_added_to_the_coverage_sum(no_lanes):
    """2 xrefs + 2 styles must not sneak into `accounted`."""
    doc = dossier.build_dossier(DRAWING, two_buckets(), entity_total=10)
    assert doc["file_facts"]["xrefs"]["total"] == 2
    assert doc["file_facts"]["text_styles"]["total"] == 2
    assert doc["coverage"]["accounted"] == 10
    assert "NOT added" in doc["coverage"]["basis"]


# --- the assembled document ---------------------------------------------------


def test_the_document_has_the_shape_the_design_doc_specifies(no_lanes):
    doc = dossier.build_dossier(DRAWING, two_buckets(), entity_total=10)
    for key in (
        "_id",
        "computed_at",
        "dossier_version",
        "entity_total",
        "layouts",
        "layers",
        "file_facts",
        "coverage",
    ):
        assert key in doc, key
    assert doc["_id"] == "abc123def456"
    assert doc["dossier_version"] == dossier.DOSSIER_VERSION
    assert doc["entity_total"] == 10
    block = doc["layers"][0]
    for key in ("layer", "layout", "entities", "types", "role", "role_basis", "measure"):
        assert key in block, key


def test_a_dossier_without_a_drawing_id_is_refused(no_lanes):
    with pytest.raises(ValueError):
        dossier.build_dossier({"entity_count": 0}, {}, entity_total=0)


def test_layers_are_ordered_deterministically(no_lanes):
    doc_a = dossier.build_dossier(
        DRAWING, two_buckets(), entity_total=10, computed_at="fixed"
    )
    reversed_input = dict(reversed(list(two_buckets().items())))
    doc_b = dossier.build_dossier(
        DRAWING, reversed_input, entity_total=10, computed_at="fixed"
    )
    assert doc_a == doc_b
    assert [b["entities"] for b in doc_a["layers"]] == [6, 4]


def test_buckets_may_arrive_as_pairs_as_well_as_a_mapping(no_lanes):
    pairs = list(two_buckets().items())
    doc = dossier.build_dossier(DRAWING, pairs, entity_total=10, computed_at="f")
    assert doc == dossier.build_dossier(
        DRAWING, two_buckets(), entity_total=10, computed_at="f"
    )


def test_units_are_applied_per_layout_and_never_borrowed_from_the_header(no_lanes):
    """A model-space unit stamped on a sheet is how this project published a
    2992 m A0 border. The sheet bucket gets nothing, and says why."""
    buckets = {
        ("A", "Model"): [ring(f"a{i}", area=100.0) for i in range(6)],
        ("A", "Sheet 1"): [ring(f"s{i}", area=1.0) for i in range(4)],
    }
    doc = dossier.build_dossier(
        DRAWING, buckets, entity_total=10, units_by_layout={"Model": "m"}
    )
    by_layout = {b["layout"]: b for b in doc["layers"]}
    assert by_layout["Model"]["measure"]["unit"] == "m2"
    assert by_layout["Sheet 1"]["measure"]["unit"] is None
    assert by_layout["Sheet 1"]["measure"]["unit_reason"]
    assert doc["file_facts"]["units"]["units_name"] == "m"


def test_the_layouts_block_cross_checks_declared_against_profiled(no_lanes):
    buckets = {
        ("A", "Model"): [ring(f"a{i}", area=1.0) for i in range(6)],
        ("A", "Sheet 1"): [ring(f"s{i}", area=1.0) for i in range(4)],
    }
    doc = dossier.build_dossier(DRAWING, buckets, entity_total=10)
    by_name = {entry["name"]: entry for entry in doc["layouts"]}
    assert by_name["Model"]["declared_entities"] == 10
    assert by_name["Model"]["profiled_entities"] == 6
    assert by_name["Model"]["agrees"] is False
    # A layout the file never declared still appears, with the hole named.
    assert by_name["Sheet 1"]["declared_entities"] is None
    assert by_name["Sheet 1"]["declared_entities_reason"]
    assert by_name["Sheet 1"]["profiled_entities"] == 4
    assert doc["file_facts"]["layouts"]["disagreeing"] == 1


def test_declared_but_unprofiled_layers_are_counted(no_lanes):
    drawing = dict(DRAWING, layers=[{"name": "A"}, {"name": "B"}, {"name": "EMPTY"}])
    doc = dossier.build_dossier(drawing, two_buckets(), entity_total=10)
    layers = doc["file_facts"]["layers"]
    assert layers["declared"] == 3
    assert layers["profiled"] == 2
    assert layers["declared_but_unprofiled"] == 1
    assert layers["declared_but_unprofiled_items"] == ["EMPTY"]


def test_embedded_documents_are_absent_not_zero(no_lanes):
    doc = dossier.build_dossier(DRAWING, two_buckets(), entity_total=10)
    embedded = doc["file_facts"]["embedded_documents"]
    assert embedded["available"] is False
    assert embedded["total"] is None
    assert "not zero" in embedded["why"]

    with_docs = dict(DRAWING, embedded_documents=[{"kind": "xlsx"}])
    doc2 = dossier.build_dossier(with_docs, two_buckets(), entity_total=10)
    assert doc2["file_facts"]["embedded_documents"]["total"] == 1


def test_shx_styles_are_detected_by_font_file_not_by_name(no_lanes):
    doc = dossier.build_dossier(DRAWING, two_buckets(), entity_total=10)
    styles = doc["file_facts"]["text_styles"]
    assert styles["total"] == 2
    assert styles["shx_styles"] == 1
    assert styles["shx_items"][0]["font"] == "SIMPLEX.SHX"


# --- the measured anchors from Phase 0 ----------------------------------------


def test_the_reference_road_layer_profiles_as_the_design_doc_says(monkeypatch, no_lanes):
    """The anchor row of the design doc's document shape, reproduced.

    1,233 entities (LINE 513, ARC 450, LWPOLYLINE 264 and 6 others), role
    network, 83,962.565009 summed from 1,191 with 42 unmeasured. Lane 2 is
    stubbed with its measured answer; what is under test here is that lane 1
    hands it the WHOLE bucket and publishes the 42 rather than the 1,191/1,191
    that would make the hole disappear.
    """
    monkeypatch.setattr(dossier, "_NETWORK", StubNetwork)
    StubNetwork.calls.clear()
    rows = (
        rows_of("LINE", 513, prefix="L", length=45.004)
        + rows_of("ARC", 450, prefix="A", length=15.707)
        + rows_of("LWPOLYLINE", 264, prefix="P", ring_status="open", length=30.0)
        + rows_of("HATCH", 6, prefix="H")
    )
    assert len(rows) == 1233

    block = dossier.profile_rows(rows, layer="road-centrelines", layout="Model", unit="m")

    assert block["entities"] == 1233
    assert block["types"] == {"LINE": 513, "ARC": 450, "LWPOLYLINE": 264, "HATCH": 6}
    assert block["role"] == "network"
    assert block["role_counts"]["network"] == 1227
    m = block["measure"]
    assert m["kind"] == "length"
    assert m["value"] == 83962.565009
    assert m["unit"] == "m"
    assert m["measured_entities"] == 1191
    assert m["unmeasured_entities"] == 42
    assert m["measured_entities"] + m["unmeasured_entities"] == 1233
    assert m["chains"] == 3
    assert m["snap_tolerance"] == 0.001
    assert m["tolerance_basis"]
    assert "CIRCLE" in m["caveat"]
    # Lane 2 saw all 1,233 rows, not the 1,227 that classified as network.
    assert StubNetwork.calls == [(1233, "m")]


# --- purity and hygiene -------------------------------------------------------


def test_the_profiler_imports_no_database_and_no_app(no_lanes):
    """The property that makes every test above possible, pinned in source."""
    source = pathlib.Path(dossier.__file__).read_text(encoding="utf-8")
    forbidden = {"pymongo", "motor", "bson", "ezdxf", "fastapi", "pydantic", "requests"}
    for raw in source.splitlines():
        match = re.match(r"\s*(?:import|from)\s+([\w\.]+)", raw)
        if not match:
            continue
        head = match.group(1)
        assert head.split(".")[0] not in forbidden, raw
        assert head not in {".mongo", ".main", ".store"}, raw
    assert "import pymongo" not in source
    assert ".main" not in source.replace("app.main", "")


def test_profiling_does_not_mutate_the_rows_it_is_given(no_lanes):
    rows = [ring("a", area=1.0), line("b", length=2.0), row("c", "INSERT", block_name="t")]
    before = copy.deepcopy(rows)
    dossier.profile_rows(rows, layer="X", layout="Model", unit="m")
    assert rows == before


def test_a_malformed_row_is_counted_not_dropped(no_lanes):
    rows = [ring("a", area=1.0), None, "not a dict", 7]
    block = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")
    assert block["entities"] == 4  # coverage must never lose a row
    assert block["malformed_rows"] == 3
    assert block["role_counts"]["other"] == 3
    assert block["types"][dossier.UNKNOWN_TYPE] == 3


def test_an_infinite_or_nan_area_is_treated_as_unmeasured(no_lanes):
    rows = [
        ring("a", area=10.0),
        ring("b", area=float("inf")),
        ring("c", area=float("nan")),
        ring("d", area=True),  # a bool is not a measurement
    ]
    m = dossier.profile_rows(rows, layer="X", layout="Model", unit="m")["measure"]
    assert m["value"] == 10.0
    assert m["measured_entities"] == 1
    assert m["unmeasured_entities"] == 3


def test_no_drawing_specific_constant_lives_in_the_module(no_lanes):
    """Rule G1's own grep, aimed at this file."""
    source = pathlib.Path(dossier.__file__).read_text(encoding="utf-8")
    for needle in (
        "VL2",
        "VL4",
        "DP4",
        "LP1",
        "TH3",
        "Primary School",
        "SchoolHatch",
        "BlockBoundary",
        "32638",
        "00_Prop",
    ):
        assert needle not in source, needle
