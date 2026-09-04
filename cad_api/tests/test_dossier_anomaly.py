"""Dossier lane 3 — anomalies. Fixtures only; this file never opens Mongo.

The fixtures reproduce the three layers Phase 0 measured against the live
store (`docs/DOSSIER-BASELINE.md`), because those three are the whole case for
the instrument:

    00_Prop - Road - CL_   1,233 entities -> 3 clusters of 411, each
                           1778.387 x 1585.99 -> copies
    ROW                       91 entities -> 1 cluster -> not a duplicate
    VL4                      195 entities -> 1 cluster -> not a duplicate

The test that matters most is none of those three. It is
`test_two_clusters_that_differ_are_not_copies` and its siblings: an instrument
that answers "copies" for the road layer is worth nothing if it also answers
"copies" for two estates that merely sit side by side. Detached is not the
finding — detached AND identical is.

The second group of tests exists because of rule G2. Half the drawings in this
store are in inches and three declare no unit at all, so every one of these
cases is also run scaled by 39.3701, and the verdict is required not to move.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import dossier_anomaly as da  # noqa: E402


# --------------------------------------------------------------------------
# fixture builders — plain dicts in the shape the store actually returns
# --------------------------------------------------------------------------

# The measured per-copy extents of `00_Prop - Road - CL_`, to 3 dp, and the
# measured per-copy length. Used to BUILD the fixture, so a reader can see the
# anchor and the input are the same numbers.
ROAD_W = 1778.387
ROAD_H = 1585.99
ROAD_N = 411
ROAD_LENGTH_PER_COPY = 27987.52167

# Projected coordinates of the real drawing's neighbourhood. Not decoration:
# float cancellation at magnitudes like these is exactly why the size match is
# a relative tolerance and not an equality.
E0, N0 = 690892.9, 2750756.0

# Metres -> inches. Any factor would do; this one is the conversion that turns
# the reference drawing into one of the eleven that are in inches.
INCH = 39.3701


def entity(handle, x0, y0, x1, y1, *, etype="LINE", length=None, shape_key=None):
    """One row as `autocad_entities` hands it over."""
    row = {
        "handle": handle,
        "type": etype,
        "bbox": {"min": [min(x0, x1), min(y0, y1)], "max": [max(x0, x1), max(y0, y1)]},
        "bbox_centre": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
        "length": length,
        "area": None,
        "shape_key": shape_key,
    }
    return row


#: How much bigger each fixture cell is than the lattice step. Anything above
#: 1.0 makes neighbouring cells overlap, which is what makes the lattice ONE
#: connected body under a threshold derived from its own cell size. A sparser
#: lattice would legitimately be cut into its own rows and columns — correct
#: behaviour on a fixture that had stopped describing a connected estate.
OVERLAP = 1.2


def body(n, w, h, *, origin=(0.0, 0.0), prefix="A", cell=None, length=None):
    """`n` overlapping segments filling a `w` x `h` box exactly.

    The lattice is roughly square in cell count, and the cell size is derived
    from the lattice rather than fixed, so the same builder produces a
    connected body for 4 entities and for 411. `cell` overrides that when a
    test wants to control the widest feature directly.
    """
    cols = max(1, min(n, math.ceil(math.sqrt(n * (w / h if h else 1.0)))))
    rows = max(1, math.ceil(n / cols))
    if cell is not None:
        cw, ch = cell
    else:
        cw = w if cols == 1 else OVERLAP * w / cols
        ch = h if rows == 1 else OVERLAP * h / rows
    step_x = 0.0 if cols == 1 else (w - cw) / (cols - 1)
    step_y = 0.0 if rows == 1 else (h - ch) / (rows - 1)
    ox, oy = origin
    out = []
    for i in range(n):
        c, r = i % cols, i // cols
        x = ox + c * step_x
        y = oy + r * step_y
        out.append(entity(f"{prefix}{i:04X}", x, y, x + cw, y + ch, length=length))
    return out


def road_layer(scale=1.0):
    """The measured road case: three copies, parked ~3.4k and ~6.9k to the side.

    `scale` converts the whole fixture into another unit system. The verdict
    must not move when it does.
    """
    rows = []
    per_entity = ROAD_LENGTH_PER_COPY / ROAD_N
    for tag, dx in (("A", 0.0), ("B", 3400.0), ("C", 6900.0)):
        rows.extend(
            body(
                ROAD_N,
                ROAD_W * scale,
                ROAD_H * scale,
                origin=((E0 + dx) * scale, N0 * scale),
                prefix=tag,
                cell=(120.0 * scale, 30.0 * scale),
                length=per_entity * scale,
            )
        )
    return rows


def row_layer():
    """The measured ROW case: 91 entities, one body, no copy anywhere."""
    return body(91, 1781.366, 1590.138, origin=(E0, N0), prefix="R")


def vl4_layer():
    """The measured VL4 case: 195 entities, one body."""
    return body(195, 1767.003, 1501.938, origin=(E0, N0), prefix="V")


def only(findings, kind, instrument=None):
    out = [f for f in findings if f["kind"] == kind]
    if instrument is not None:
        out = [f for f in out if f["evidence"]["instrument"] == instrument]
    assert len(out) == 1, f"expected exactly one {kind}/{instrument}, got {len(out)}"
    return out[0]


# --------------------------------------------------------------------------
# the contract itself
# --------------------------------------------------------------------------


def test_signature_is_the_contract():
    """`anomalies(rows, *, layer)` — extra parameters must all be optional."""
    sig = inspect.signature(da.anomalies)
    params = list(sig.parameters.values())
    assert params[0].name == "rows"
    assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    layer = sig.parameters["layer"]
    assert layer.kind is inspect.Parameter.KEYWORD_ONLY
    assert layer.default is inspect.Parameter.empty
    for name, p in sig.parameters.items():
        if name not in ("rows", "layer"):
            assert p.default is not inspect.Parameter.empty, (
                f"{name} must be optional or lane 1's call breaks"
            )


def test_every_finding_matches_the_shape_and_the_kind_list():
    for rows, layer in ((road_layer(), "road"), ([], "empty"), (broken_rows(), "b")):
        for finding in da.anomalies(rows, layer=layer):
            assert set(finding) == {"kind", "detail", "evidence", "counts"}
            assert finding["kind"] in da.KINDS
            assert isinstance(finding["detail"], str) and finding["detail"]
            assert isinstance(finding["evidence"], dict)
            assert isinstance(finding["counts"], dict)
            assert all(isinstance(v, int) for v in finding["counts"].values())
            assert finding["evidence"]["instrument"]
            json.dumps(finding)  # must survive the trip into Mongo


def test_an_unremarkable_layer_returns_an_empty_list_never_none():
    assert da.anomalies(row_layer(), layer="ROW") == []
    assert da.anomalies(vl4_layer(), layer="VL4") == []


def test_the_function_does_not_touch_its_input():
    rows = road_layer()
    before = copy.deepcopy(rows)
    da.anomalies(rows, layer="road")
    assert rows == before


def test_the_same_rows_answer_the_same_way_twice():
    rows = road_layer()
    assert json.dumps(da.anomalies(rows, layer="road")) == json.dumps(
        da.anomalies(rows, layer="road")
    )


# --------------------------------------------------------------------------
# the road case — three identical clusters
# --------------------------------------------------------------------------


def test_the_road_case_reports_three_clusters_of_411():
    rows = road_layer()
    assert len(rows) == 1233

    found = only(
        da.anomalies(rows, layer="00_Prop - Road - CL_"),
        "duplicate_clusters", "spatial_cluster",
    )
    group = found["evidence"]["groups"][0]

    assert found["counts"]["clusters"] == 3
    assert group["members"] == 3
    assert group["entities_each"] == 411
    assert found["counts"]["entities_involved"] == 1233
    assert round(group["bbox_width"], 3) == ROAD_W
    assert round(group["bbox_height"], 3) == ROAD_H


def test_the_three_clusters_agree_to_three_decimal_places():
    """The Phase 0 wording, checked as arithmetic rather than as prose."""
    scan = da.cluster_scan(road_layer(), layer="road")
    widths = {round(c["bbox_width"], 3) for c in scan["clusters"]}
    heights = {round(c["bbox_height"], 3) for c in scan["clusters"]}
    counts = {c["entities"] for c in scan["clusters"]}
    assert widths == {ROAD_W}
    assert heights == {ROAD_H}
    assert counts == {411}


def test_the_length_total_corroborates_and_says_so_separately():
    """Σ length is a second reading, reported as its own verdict."""
    group = only(
        da.anomalies(road_layer(), layer="road"), "duplicate_clusters",
        "spatial_cluster",
    )["evidence"]["groups"][0]
    assert group["length_agrees"] is True
    assert group["length_total_each"] == pytest.approx(ROAD_LENGTH_PER_COPY, rel=1e-9)


def test_the_road_finding_names_clustering_and_shape_keys_blindness():
    found = only(da.anomalies(road_layer(), layer="road"), "duplicate_clusters")
    assert found["evidence"]["instrument"] == "spatial_cluster"
    assert "spatial clustering" in found["detail"]
    # the whole reason this instrument exists: the exact one saw nothing
    assert found["evidence"]["keyed_rows"] == 0
    assert found["evidence"]["unkeyed_rows"] == 1233
    assert "shape_key is null" in found["detail"]


def test_the_void_between_clusters_is_wider_than_the_void_inside_one():
    """The margin that justifies the cut, published so a person can check it."""
    scan = da.cluster_scan(road_layer(), layer="road")
    assert scan["largest_void_inside_a_cluster"] < scan["gap_threshold"]
    assert scan["smallest_void_between_clusters"] > scan["gap_threshold"]
    assert (scan["smallest_void_between_clusters"]
            > scan["largest_void_inside_a_cluster"])


# --------------------------------------------------------------------------
# one cluster — not a duplicate
# --------------------------------------------------------------------------


def test_one_cluster_is_never_a_duplicate():
    for rows, layer in ((row_layer(), "ROW"), (vl4_layer(), "VL4")):
        scan = da.cluster_scan(rows, layer=layer)
        assert scan["clusters_total"] == 1
        assert scan["duplicate_groups"] == []
        assert scan["smallest_void_between_clusters"] is None
        assert da.anomalies(rows, layer=layer) == []


def test_row_and_vl4_keep_their_measured_entity_counts():
    assert da.cluster_scan(row_layer(), layer="ROW")["clusters"][0]["entities"] == 91
    assert da.cluster_scan(vl4_layer(), layer="VL4")["clusters"][0]["entities"] == 195


# --------------------------------------------------------------------------
# THE FALSE-POSITIVE GUARD — two clusters that differ
# --------------------------------------------------------------------------


def two_bodies(n_a, w_a, h_a, n_b, w_b, h_b, gap=4000.0):
    return body(n_a, w_a, h_a, origin=(E0, N0), prefix="A") + body(
        n_b, w_b, h_b, origin=(E0 + gap, N0), prefix="B"
    )


def test_two_clusters_that_differ_are_not_copies():
    """Detached is not the finding. Detached AND identical is."""
    rows = two_bodies(411, ROAD_W, ROAD_H, 410, ROAD_W, ROAD_H)
    scan = da.cluster_scan(rows, layer="two")
    assert scan["clusters_total"] == 2, "the two bodies must still be SEEN as two"
    assert scan["duplicate_groups"] == []
    assert da.anomalies(rows, layer="two") == []


def test_same_count_but_a_different_width_is_not_a_copy():
    rows = two_bodies(411, ROAD_W, ROAD_H, 411, ROAD_W + 5.0, ROAD_H)
    assert da.cluster_scan(rows, layer="two")["clusters_total"] == 2
    assert da.anomalies(rows, layer="two") == []


def test_same_count_and_width_but_a_different_height_is_not_a_copy():
    rows = two_bodies(411, ROAD_W, ROAD_H, 411, ROAD_W, ROAD_H + 5.0)
    assert da.cluster_scan(rows, layer="two")["clusters_total"] == 2
    assert da.anomalies(rows, layer="two") == []


def test_a_difference_far_below_the_tolerance_is_still_a_copy():
    """The tolerance is a stated number, so pin both sides of it."""
    nudge = ROAD_W * da.DUPLICATE_MATCH_REL_TOL / 100.0
    rows = two_bodies(411, ROAD_W, ROAD_H, 411, ROAD_W + nudge, ROAD_H)
    assert da.cluster_scan(rows, layer="two")["duplicate_groups"][0]["members"] == 2


def test_a_difference_far_above_the_tolerance_is_not():
    nudge = ROAD_W * da.DUPLICATE_MATCH_REL_TOL * 1000.0
    rows = two_bodies(411, ROAD_W, ROAD_H, 411, ROAD_W + nudge, ROAD_H)
    assert da.cluster_scan(rows, layer="two")["duplicate_groups"] == []


def test_three_bodies_of_which_only_two_match_reports_two_not_three():
    rows = (
        body(411, ROAD_W, ROAD_H, origin=(E0, N0), prefix="A")
        + body(411, ROAD_W, ROAD_H, origin=(E0 + 4000, N0), prefix="B")
        + body(300, ROAD_W, ROAD_H, origin=(E0 + 8000, N0), prefix="C")
    )
    groups = da.cluster_scan(rows, layer="three")["duplicate_groups"]
    assert len(groups) == 1
    assert groups[0]["members"] == 2
    assert groups[0]["entities_each"] == 411


def test_a_field_of_identical_points_is_not_a_field_of_copies():
    """The loudest false positive this instrument can produce, and its guard.

    Zero-extent entities make every separation wider than the widest feature.
    Without the guard, 200 survey points report themselves as 200 mutual
    copies. The instrument must decline, in writing.
    """
    rows = [
        entity(f"P{i:04X}", E0 + i * 50.0, N0, E0 + i * 50.0, N0) for i in range(200)
    ]
    scan = da.cluster_scan(rows, layer="points")
    assert scan["duplicate_groups"] == []
    assert scan["skipped"] and "zero-extent" in scan["skipped"]
    assert da.anomalies(rows, layer="points") == []


def test_two_lone_entities_far_apart_are_not_copies_of_each_other():
    rows = [
        entity("AAA1", E0, N0, E0 + 10, N0 + 10),
        entity("BBB1", E0 + 9000, N0, E0 + 9010, N0 + 10),
    ]
    scan = da.cluster_scan(rows, layer="two-lines")
    assert scan["clusters_total"] == 2
    assert scan["duplicate_groups"] == []
    assert da.cluster_scan(rows, layer="x")["clusters"][0]["entities"] == 1


# --------------------------------------------------------------------------
# the gap threshold: derived from the data, stated, and unit-free
# --------------------------------------------------------------------------


def test_the_threshold_is_derived_and_stated_with_the_entity_that_set_it():
    scan = da.cluster_scan(road_layer(), layer="road")
    assert scan["gap_threshold"] == pytest.approx(120.0)  # the widest fixture feature
    assert scan["threshold_source"]["handle"]
    assert scan["threshold_source"]["extent"] == pytest.approx(120.0)
    assert "widest single feature" in scan["gap_basis"]
    assert "drawing units" in scan["gap_basis"]


def test_the_threshold_moves_with_the_data_not_with_the_code():
    """Double every feature and the derived threshold doubles. No constant."""
    small = da.cluster_scan(body(50, 800.0, 600.0, cell=(40.0, 20.0)), layer="s")
    large = da.cluster_scan(body(50, 800.0, 600.0, cell=(80.0, 20.0)), layer="l")
    assert large["gap_threshold"] == pytest.approx(2 * small["gap_threshold"])


def test_the_verdict_survives_a_change_of_unit():
    """G2. The same estate in inches is the same estate."""
    metres = da.cluster_scan(road_layer(), layer="road")
    inches = da.cluster_scan(road_layer(scale=INCH), layer="road", unit="in")

    assert metres["clusters_total"] == inches["clusters_total"] == 3
    assert metres["duplicate_groups"][0]["members"] == 3
    assert inches["duplicate_groups"][0]["members"] == 3
    assert inches["duplicate_groups"][0]["entities_each"] == 411
    # and the threshold is expressed in the drawing's own units, not a constant
    assert inches["gap_threshold"] == pytest.approx(metres["gap_threshold"] * INCH)


def test_a_drawing_that_declares_no_unit_is_never_given_metres():
    """G2, the version that has already bitten this project."""
    found = only(da.anomalies(road_layer(), layer="road"), "duplicate_clusters",
                 "spatial_cluster")
    ev = found["evidence"]
    assert ev["unit"] is None
    assert ev["unit_label"] == "drawing units"
    assert "must not be read as metres" in ev["unit_basis"]
    assert "drawing units" in found["detail"]
    blob = json.dumps(found)
    for forbidden in ('"m"', '"metre"', '"metres"', '"meter"'):
        assert forbidden not in blob


def test_a_declared_unit_is_echoed_and_nothing_else_is():
    found = only(da.anomalies(road_layer(), layer="road", unit="in"),
                 "duplicate_clusters", "spatial_cluster")
    assert found["evidence"]["unit"] == "in"
    assert found["evidence"]["unit_label"] == "in"
    assert " in." in found["detail"] or " in " in found["detail"]


# --------------------------------------------------------------------------
# declared but empty
# --------------------------------------------------------------------------


def test_a_declared_layer_with_no_entities_says_so():
    found = only(da.anomalies([], layer="C-ANNO-EMPTY"), "declared_but_empty")
    assert found["counts"]["entities"] == 0
    assert found["evidence"]["layer"] == "C-ANNO-EMPTY"
    assert found["evidence"]["rows_received"] == 0
    assert "not the same answer as 'there are none'" in found["detail"]


def test_an_empty_layer_produces_that_finding_and_no_other():
    assert [f["kind"] for f in da.anomalies([], layer="X")] == ["declared_but_empty"]


def test_none_is_treated_as_empty_not_as_a_crash():
    assert da.anomalies(None, layer="X")[0]["kind"] == "declared_but_empty"


# --------------------------------------------------------------------------
# unreadable
# --------------------------------------------------------------------------


def broken_rows():
    good = body(20, 500.0, 400.0, origin=(E0, N0), prefix="G")
    return good + [
        {"handle": "BAD1", "type": "3DSOLID", "length": None},               # no bbox
        {"handle": "BAD2", "type": "SPLINE", "bbox": {"min": [1.0], "max": [2.0, 3.0]}},
        {"handle": "BAD3", "type": "LINE", "bbox": {"min": [float("nan"), 0.0],
                                                    "max": [1.0, 1.0]}},
        {"handle": "BAD4", "type": "LINE",                                # min > max
         "bbox": {"min": [9.0, 9.0], "max": [1.0, 1.0]}},
        {"type": "LINE", "bbox": {"min": [0.0, 0.0], "max": [1.0, 1.0]}},    # no handle
    ]


def test_entities_that_cannot_be_placed_are_counted_and_named():
    found = only(da.anomalies(broken_rows(), layer="mixed"), "unreadable")
    b = found["evidence"]["breakdown"]
    assert b["missing_bbox"] == 1
    assert b["malformed_bbox"] == 2       # bad arity, and a NaN corner
    assert b["reversed_bbox"] == 1
    assert b["unnameable"] == 1
    assert found["counts"]["unplaceable"] == 3
    for handle in ("BAD1", "BAD2", "BAD3", "BAD4"):
        assert handle in found["evidence"]["named"]


def test_the_unreadable_totals_reconcile_with_the_row_count():
    """Nothing dropped: placed + unplaceable == rows."""
    rows = broken_rows()
    found = only(da.anomalies(rows, layer="mixed"), "unreadable")
    c = found["counts"]
    assert c["rows"] == len(rows)
    assert c["placed"] + c["unplaceable"] == c["rows"]


def test_an_unplaceable_entity_is_excluded_and_the_exclusion_is_stated():
    rows = road_layer() + [{"handle": "GHOST", "type": "3DSOLID"}]
    found = only(da.anomalies(rows, layer="road"), "duplicate_clusters",
                 "spatial_cluster")
    assert found["counts"]["entities_clustered"] == 1233
    assert found["counts"]["entities_unplaceable"] == 1
    assert only(da.anomalies(rows, layer="road"), "unreadable")


def test_text_without_a_length_is_counted_but_is_not_itself_a_defect():
    rows = [
        entity(f"T{i:03X}", E0 + i, N0, E0 + i + 2, N0 + 2, etype="TEXT")
        for i in range(10)
    ]
    findings = da.anomalies(rows, layer="anno")
    assert findings == [], "a layer of well-formed TEXT is unremarkable"
    # and yet the count is available to whoever wants to reconcile
    assert da.cluster_scan(rows, layer="anno")["placed"] == 10


def test_a_row_that_is_not_a_record_at_all_is_counted_not_swallowed():
    rows = body(10, 300.0, 200.0, origin=(E0, N0)) + ["not a dict", 42]
    found = only(da.anomalies(rows, layer="junk"), "unreadable")
    assert found["evidence"]["breakdown"]["not_a_record"] == 2


# --------------------------------------------------------------------------
# shape_key — the exact instrument, where it exists
# --------------------------------------------------------------------------


def keyed_rows():
    """Closed rings: 4 sharing one key, 2 sharing another, 3 unique."""
    plan = ["k-aaa"] * 4 + ["k-bbb"] * 2 + ["k-1", "k-2", "k-3"]
    return [
        entity(f"K{i:03X}", E0 + i * 5.0, N0, E0 + i * 5.0 + 30.0, N0 + 25.0,
               etype="LWPOLYLINE", shape_key=key)
        for i, key in enumerate(plan)
    ]


def test_shape_key_duplicates_are_found_and_the_instrument_is_named():
    found = only(da.anomalies(keyed_rows(), layer="VL4"), "duplicate_clusters",
                 "shape_key")
    assert found["counts"]["repeated_keys"] == 2
    assert found["counts"]["entities_involved"] == 6
    assert "shape_key" in found["detail"]
    assert found["evidence"]["repeated_keys"][0]["entities"] == 4
    assert found["evidence"]["repeated_keys"][0]["shape_key"] == "k-aaa"


def test_the_keyed_finding_publishes_how_much_it_could_see():
    """The blindness is part of the reading, not a footnote."""
    rows = keyed_rows() + body(5, 200.0, 100.0, origin=(E0 + 9000, N0), prefix="U")
    found = only(da.anomalies(rows, layer="mix"), "duplicate_clusters", "shape_key")
    assert found["evidence"]["keyed_rows"] == 9
    assert found["evidence"]["unkeyed_rows"] == 5


def test_both_instruments_can_speak_about_one_layer_and_stay_distinguishable():
    rows = (
        keyed_rows()
        + body(30, 400.0, 300.0, origin=(E0 + 9000, N0), prefix="P")
        + body(30, 400.0, 300.0, origin=(E0 + 20000, N0), prefix="Q")
    )
    found = da.anomalies(rows, layer="both")
    instruments = {f["evidence"]["instrument"] for f in found
                   if f["kind"] == "duplicate_clusters"}
    assert instruments == {"shape_key", "spatial_cluster"}


def test_a_null_shape_key_is_never_treated_as_a_shared_value():
    """Every open entity has `shape_key: None`. That is not a group of copies."""
    rows = road_layer()
    assert all(r["shape_key"] is None for r in rows)
    keyed = [f for f in da.anomalies(rows, layer="road")
             if f["evidence"]["instrument"] == "shape_key"]
    assert keyed == []


# --------------------------------------------------------------------------
# G1 — nothing is decided from a layer name
# --------------------------------------------------------------------------


def test_the_layer_name_changes_the_prose_and_nothing_else():
    rows = road_layer()
    a = da.anomalies(rows, layer="00_Prop - Road - CL_")
    b = da.anomalies(rows, layer="طريق")  # an Arabic layer name
    strip = lambda fs: [  # noqa: E731
        {k: v for k, v in f["evidence"].items() if k not in ("layer",)}
        for f in fs
    ]
    assert [f["counts"] for f in a] == [f["counts"] for f in b]
    assert json.dumps(strip(a)) == json.dumps(strip(b))


def test_an_empty_layer_name_is_not_an_error():
    assert da.anomalies(road_layer(), layer="")[0]["kind"] == "duplicate_clusters"


# --------------------------------------------------------------------------
# G7 — caps are stated, and what is omitted is counted
# --------------------------------------------------------------------------


def test_a_capped_cluster_list_says_what_it_left_out():
    step = 5000.0
    rows = []
    for i in range(da.MAX_CLUSTERS_REPORTED + 7):
        rows.extend(body(4, 300.0, 200.0, origin=(E0 + i * step, N0), prefix=f"C{i}_"))
    scan = da.cluster_scan(rows, layer="many")
    assert scan["clusters_total"] == da.MAX_CLUSTERS_REPORTED + 7
    assert scan["clusters_listed"] == da.MAX_CLUSTERS_REPORTED
    assert scan["clusters_omitted"] == 7
    found = only(da.anomalies(rows, layer="many"), "duplicate_clusters",
                 "spatial_cluster")
    assert found["evidence"]["clusters_omitted"] == 7
    assert found["evidence"]["clusters_cap"] == da.MAX_CLUSTERS_REPORTED
    # the COUNT is never truncated even when the LIST is
    assert found["counts"]["clusters"] == da.MAX_CLUSTERS_REPORTED + 7


def test_a_capped_handle_list_says_what_it_left_out():
    found = only(da.anomalies(road_layer(), layer="road"), "duplicate_clusters",
                 "spatial_cluster")
    cluster = found["evidence"]["groups"][0]["clusters"][0]
    assert len(cluster["example_handles"]) == da.MAX_HANDLES_LISTED
    assert cluster["handles_omitted"] == 411 - da.MAX_HANDLES_LISTED


def test_a_row_count_above_the_stated_ceiling_declines_in_writing():
    scan = da.cluster_scan(
        [{"handle": "x"}] * (da.MAX_ROWS_CLUSTERED + 1), layer="huge"
    )
    assert scan["duplicate_groups"] == []
    assert str(da.MAX_ROWS_CLUSTERED) in scan["skipped"]


# --------------------------------------------------------------------------
# a body that a long feature holds together
# --------------------------------------------------------------------------


def test_one_long_feature_closes_the_space_it_crosses():
    """Intervals, not points.

    Two dense patches 900 apart would be cut in two — unless one polyline runs
    from one to the other, in which case they are one body and the instrument
    must say so. This is the case that a point-based clustering gets wrong.
    """
    left = body(20, 300.0, 200.0, origin=(E0, N0), prefix="L", cell=(60.0, 20.0))
    right = body(20, 300.0, 200.0, origin=(E0 + 1200.0, N0), prefix="R",
                 cell=(60.0, 20.0))
    assert da.cluster_scan(left + right, layer="split")["clusters_total"] == 2

    bridge = [entity("BR01", E0 + 100.0, N0 + 50.0, E0 + 1250.0, N0 + 50.0,
                     etype="LWPOLYLINE")]
    joined = da.cluster_scan(left + right + bridge, layer="joined")
    assert joined["clusters_total"] == 1
    assert joined["threshold_source"]["handle"] == "BR01"
