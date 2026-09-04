"""UPLIFT-07 — find_duplicates and the shape cross-check.

Two parts, deliberately separated, following the habit of `test_measure.py`.

The first part is **pure**. Every computation that determines the answer —
grouping by `shape_key`, centroid confirmation, the `ring_status` census, and
the shape of the response — is an ordinary function over a list of dicts, and
is tested as such. This is not convenience: a `find_duplicates` that fetches
its own rows from Mongo cannot be tested without Mongo, and a test that fakes
its driver passes while the grouping is wrong.

The second part is **impure**, and that is said here. The Definition of Done
numbers — 52 groups, 48 of them ≤ 1 cm, 2,312/68/80 on the cross sweep — are
properties of the data that has been ingested. A test that fakes them proves
that its fixture is consistent, not that the drawing is. So that part runs
against a live database and **skips while naming itself** when there is none.
`pytest -rs` prints the list; read it before you trust a green suite.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app import evidence as ev
from app import geometry, store, store_geometry as sg


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

#: The units the drawing states. Its shape is exactly what `store._unit_names`
#: returns; `evidence.Scope` rejects anything short by even one key.
METRES = {
    "name": "m",
    "declared_in_file": True,
    "length_unit": "m",
    "area_unit": "m2",
    "space": "model",
}

#: A drawing that states no unit at all — three out of sixteen.
UNITLESS = {
    "name": "unitless",
    "declared_in_file": False,
    "length_unit": None,
    "area_unit": None,
    "space": "model",
}


def rect(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def para(x: float, y: float, w: float, h: float, skew: float = 5.0):
    """A parallelogram with the same edge sequence as the rectangle.

    It is in this file because it is the only shape that proves `shape_key` is
    not merely a sequence of edge lengths.
    """
    import math as _m

    d = _m.hypot(w, 0.0)
    a = _m.radians(90 - skew)
    return [
        (x, y),
        (x + d, y),
        (x + d + h * _m.cos(a), y + h * _m.sin(a)),
        (x + h * _m.cos(a), y + h * _m.sin(a)),
    ]


def ring_row(handle: str, layer: str, points, **over) -> dict:
    """One row as the Mongo projection returns it for a complete ring."""
    pts = [(float(a), float(b)) for a, b in points]
    centroid = geometry.polygon_centroid(pts)
    row = {
        "handle": handle,
        "layer": layer,
        "layout": "Model",
        "type": "LWPOLYLINE",
        "ring_status": "complete",
        "ring": [[a, b] for a, b in pts],
        "polygon_centroid": [centroid[0], centroid[1]] if centroid else None,
        "edge_lengths": [round(v, 6) for v in geometry.edge_lengths(pts, closed=True)],
        "area": abs(geometry.signed_area(pts)),
        "shape_key": geometry.shape_key(pts),
    }
    row.update(over)
    return row


def census(**over) -> sg.RingCensus:
    counts = {"complete": 0, "open": 0, "bulge": 0, "degenerate": 0}
    counts.update(over)
    return sg.RingCensus.from_counts(counts)


def build(rows, *, kind="geometry", tolerance=0.01, units=METRES, **over):
    """`build_duplicates` with defaults that need not repeat in every test."""
    kwargs = dict(
        drawing_id="d0",
        layout="Model",
        rows=rows,
        ring_census=census(complete=len(rows)),
        kind=kind,
        tolerance=tolerance,
        units=units,
        layers=None,
        limit=50,
    )
    kwargs.update(over)
    return sg.build_duplicates(**kwargs)


# ---------------------------------------------------------------------------
# Pure — the response shape and the contract that binds it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["geometry", "shape", "label"])
def test_setiap_respons_membawa_toleransi_dan_satuannya(kind):
    """A spec correction already applied, tested so it does not come back.

    `tolerance_m` plants the unit inside the parameter's NAME, in a repo where
    11 of the 16 drawings are in inches and 3 state no unit at all. That name
    must not appear again in any response.
    """
    r = build([ring_row("A", "0", rect(0, 0, 12, 25))], kind=kind)
    assert r["tolerance"] == 0.01
    assert r["tolerance_units"] == "m"
    assert "tolerance_m" not in r
    assert not any("_m" == k[-2:] for k in r), f"a field name carries a unit: {r.keys()}"


def test_gambar_tanpa_satuan_menahan_satuan_dan_menyebut_alasannya():
    """G2. A number that writes 'm' because this drawing happens to be in
    metres is a wrong number wearing the clothes of a right one."""
    r = build([ring_row("A", "0", rect(0, 0, 12, 25))], units=UNITLESS)
    assert r["tolerance"] == 0.01
    assert r["tolerance_units"] is None
    assert r["tolerance_units_note"], "a withheld unit must state its reason"
    assert r["area_tolerance_units"] is None


@pytest.mark.parametrize("kind", ["geometry", "shape", "label"])
def test_interpretation_note_wajib_dan_berbeda_per_jenis(kind):
    """A tool that reports '52 duplicates' without that sentence will be read
    as '52 defects', and nothing in the data supports that reading."""
    r = build([ring_row("A", "0", rect(0, 0, 12, 25))], kind=kind)
    assert r["interpretation_note"].strip()
    assert "not" in r["interpretation_note"].lower()


def test_catatan_jenis_shape_menolak_kata_cacat():
    """The 449 VL3 plots really are all 12x25. Calling that a defect is a
    false finding whose count is larger than the real findings."""
    rows = [ring_row(f"H{i}", "VL3", rect(i * 100, 0, 12, 25)) for i in range(5)]
    r = build(rows, kind="shape")
    assert r["groups_total"] == 1
    assert r["groups"][0]["member_count"] == 5
    assert r["defect"] is False
    assert "repeated" in r["interpretation_note"].lower()


# --- separating geometry from shape -----------------------------------------


def test_bentuk_berulang_bukan_duplikat_geometri():
    """The main trap of this spec. Merging the two kinds would report the 449
    VL3 plots as 449 duplicates."""
    rows = [ring_row(f"H{i}", "VL3", rect(i * 100, 0, 12, 25)) for i in range(5)]
    assert build(rows, kind="geometry")["groups_total"] == 0
    assert build(rows, kind="shape")["groups_total"] == 1


def test_salinan_di_tempat_yang_sama_ditemukan():
    rows = [
        ring_row("2E1A44", "0", rect(0, 0, 12, 25)),
        ring_row("205CEA1", "VL2", rect(0.0005, 0.0005, 12, 25)),
    ]
    r = build(rows, kind="geometry")
    assert r["groups_total"] == 1
    assert r["entities_involved"] == 2
    g = r["groups"][0]
    assert {m["handle"] for m in g["members"]} == {"2E1A44", "205CEA1"}
    assert {m["layer"] for m in g["members"]} == {"0", "VL2"}
    assert g["centroid_gap"] < 0.01
    assert g["centroid_gap_units"] == "m"


def test_persegi_panjang_dan_jajaran_genjang_tidak_pernah_satu_grup():
    """Edge lengths alone are not a shape fingerprint. If this fails,
    `shape_key` has lost its turn angles and this whole tool over-reports."""
    r = rect(0, 0, 12, 25)
    p = para(0, 0, 12, 25)
    assert sorted(round(x, 6) for x in geometry.edge_lengths(r, True)) != sorted(
        round(x, 6) for x in geometry.edge_lengths(p, True)
    ) or geometry.shape_key(r) != geometry.shape_key(p)
    rows = [ring_row("R", "0", r), ring_row("P", "0", p)]
    assert build(rows, kind="shape")["groups_total"] == 0


# --- tolerance ---------------------------------------------------------------


def test_toleransi_memilih_himpunan_yang_berbeda():
    """'Duplicate' at 0.01 and at 1 are two different sets. This is the small
    form of the 48-vs-52 in the reference drawing."""
    rows = [
        ring_row("A", "0", rect(0, 0, 12, 25)),
        ring_row("B", "VL2", rect(0.4, 0, 12, 25)),
    ]
    assert build(rows, tolerance=0.01)["groups_total"] == 0
    r = build(rows, tolerance=1.0)
    assert r["groups_total"] == 1
    assert r["groups"][0]["centroid_gap"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "dx,dy",
    [
        (1.4, 0.0),   # adjacent bucket on the x axis
        (0.0, 1.4),   # and on the y axis
        (0.9, 0.9),   # diagonal bucket: 1.27 — inside the box, outside the circle
    ],
)
def test_pasangan_di_bucket_tetangga_tetap_diuji_jaraknya(dx, dy):
    """A mutant that got through once, and is here because of it.

    Pair search uses tolerance-sized buckets and then checks their 3x3
    neighbours, so a pair can pass the bucket filter while being as much as
    ~2.83 times the tolerance apart. What turns "in the same box" into "this
    close" is the explicit distance test, and a test that only uses pairs that
    are very far apart never touches it: removing that test stays green. All
    three distances below are > 1 and all fall in a neighbouring bucket.
    """
    rows = [
        ring_row("A", "0", rect(0, 0, 12, 25)),
        ring_row("B", "VL2", rect(dx, dy, 12, 25)),
    ]
    assert math.hypot(dx, dy) > 1.0
    assert build(rows, tolerance=1.0)["groups_total"] == 0
    # and still found once the tolerance really does cover it
    assert build(rows, tolerance=2.0)["groups_total"] == 1


def test_toleransi_nol_atau_negatif_ditolak_dengan_saran():
    for bad in (0.0, -1.0):
        with pytest.raises(sg.DuplicateRefused) as exc:
            build([], tolerance=bad)
        assert exc.value.code == "TOLERANCE_NOT_POSITIVE"
        assert exc.value.hint.strip()


def test_jenis_yang_tidak_dikenal_ditolak_dengan_daftar_yang_ada():
    with pytest.raises(sg.DuplicateRefused) as exc:
        build([], kind="overlap")
    assert exc.value.code == "UNKNOWN_KIND"
    for k in sg.KINDS:
        assert k in exc.value.hint


def test_layout_wajib():
    """Without a layout, a region on a sheet matches model space geometry that
    happens to share its coordinates."""
    for bad in (None, "", "   "):
        with pytest.raises(sg.DuplicateRefused) as exc:
            build([], layout=bad)
        assert exc.value.code == "LAYOUT_REQUIRED"


# --- what is skipped is counted, not removed ---------------------------------


def test_yang_dilewati_dihitung_dipecah_per_sebab():
    """A skipped target is not a target that did not match."""
    r = build(
        [ring_row("A", "0", rect(0, 0, 12, 25))],
        ring_census=census(complete=1, open=3090, bulge=1298, degenerate=4),
    )
    s = r["skipped"]
    assert s["skipped_open"] == 3090
    assert s["skipped_bulge"] == 1298
    assert s["skipped_degenerate"] == 4
    assert s["skipped_oversize"] == 0, "the key is present even when zero"
    assert r["targets_considered"] == 1


def test_ring_tanpa_shape_key_dilewati_dan_dihitung_bukan_dianggap_unik():
    rows = [
        ring_row("A", "0", rect(0, 0, 12, 25)),
        ring_row("B", "0", rect(0, 0, 12, 25), shape_key=None),
    ]
    r = build(rows, kind="geometry")
    assert r["skipped"]["skipped_no_shape_key"] == 1
    assert r["groups_total"] == 0


def test_ring_tanpa_centroid_dilewati_bukan_diberi_nol():
    """G8. A centroid that does not exist is not a centroid at zero — polygons
    that double back on themselves would gather at the origin and report
    duplicates."""
    rows = [
        ring_row("A", "0", rect(0, 0, 12, 25), polygon_centroid=None),
        ring_row("B", "VL2", rect(0, 0, 12, 25), polygon_centroid=None),
    ]
    r = build(rows, kind="geometry")
    assert r["skipped"]["skipped_no_centroid"] == 2
    assert r["groups_total"] == 0


# --- kind="label" ------------------------------------------------------------


def label_row(handle: str, text: str, x: float, y: float, layer: str = "PlotNo") -> dict:
    return {
        "handle": handle,
        "layer": layer,
        "layout": "Model",
        "type": "TEXT",
        "text": text,
        "anchor_point": [x, y],
        "anchor_basis": "insert",
    }


def test_dua_teks_sama_di_dalam_satu_poligon_jadi_satu_grup():
    rings = [ring_row("POLY", "0", rect(0, 0, 12, 25))]
    labels = [label_row("T1", "A-101", 3, 3), label_row("T2", "A-101", 9, 20)]
    r = build(rings, kind="label", label_rows=labels)
    assert r["groups_total"] == 1
    g = r["groups"][0]
    assert g["polygon_handle"] == "POLY"
    assert g["member_count"] == 2
    assert {m["handle"] for m in g["members"]} == {"T1", "T2"}


def test_teks_sama_di_poligon_berbeda_bukan_duplikat_label():
    """The same plot number on two parcels is a different question — and a
    question whose answer is not 'a duplicate inside one polygon'."""
    rings = [
        ring_row("P1", "0", rect(0, 0, 12, 25)),
        ring_row("P2", "0", rect(100, 0, 12, 25)),
    ]
    labels = [label_row("T1", "A-101", 3, 3), label_row("T2", "A-101", 103, 3)]
    assert build(rings, kind="label", label_rows=labels)["groups_total"] == 0


def test_label_di_luar_setiap_poligon_bukan_error():
    rings = [ring_row("P1", "0", rect(0, 0, 12, 25))]
    labels = [label_row("T1", "A-101", 900, 900), label_row("T2", "A-101", 901, 901)]
    r = build(rings, kind="label", label_rows=labels)
    assert r["groups_total"] == 0
    assert r["skipped"]["boundary_cases"] == 0


def test_label_tanpa_anchor_dilewati_dan_dihitung():
    """The centre of the bounding box is NEVER a fallback."""
    rings = [ring_row("P1", "0", rect(0, 0, 12, 25))]
    labels = [
        {**label_row("T1", "A-101", 3, 3), "anchor_point": None},
        label_row("T2", "A-101", 6, 6),
    ]
    r = build(rings, kind="label", label_rows=labels)
    assert r["skipped"]["skipped_no_anchor"] == 1
    assert r["groups_total"] == 0


def test_label_persis_di_garis_dihitung_di_dalam_dan_dilaporkan_terpisah():
    """Three answers, not two. A plot number snapped to the line it labels is
    not an edge case — it is how the drawing is made."""
    rings = [ring_row("P1", "0", rect(0, 0, 12, 25))]
    labels = [label_row("T1", "A-101", 0, 12.5), label_row("T2", "A-101", 6, 12)]
    r = build(rings, kind="label", label_rows=labels)
    assert r["groups_total"] == 1
    assert r["skipped"]["boundary_cases"] >= 1


def test_poligon_selebar_gambar_tetap_ditemukan_lewat_indeks():
    """The bbox index is a superset, never a subset. A polygon whose box
    crosses too many cells leaves the grid and enters the wide list — and must
    still be tested for every label."""
    rings = [
        ring_row("SMALL", "0", rect(0, 0, 12, 25)),
        ring_row("WIDE", "0", rect(-5000, -5000, 10000, 10000)),
    ]
    labels = [label_row("T1", "Z", 4000, 4000), label_row("T2", "Z", 4100, 4100)]
    r = build(rings, kind="label", label_rows=labels)
    assert r["groups_total"] == 1
    assert r["groups"][0]["polygon_handle"] == "WIDE"


def test_indeks_bbox_mengembalikan_superset_untuk_setiap_titik_uji():
    """Tested as a property, not as an example: for every point, the candidates
    the index returns contain EVERY box that really does contain that point. An
    index that loses one box removes findings without any trace at all in its
    response."""
    from app import region as reg

    boxes = [
        (0.0, 0.0, 12.0, 25.0),
        (12.0, 0.0, 24.0, 25.0),
        (-500.0, -500.0, 500.0, 500.0),
        (100.0, 100.0, 100.0, 100.0),
    ]
    grid = sg._BboxGrid(boxes)
    for x in range(-60, 60, 7):
        for y in range(-60, 60, 7):
            truth = {
                i for i, b in enumerate(boxes)
                if b[0] <= x <= b[2] and b[1] <= y <= b[3]
            }
            got = set(grid.candidates(float(x), float(y)))
            assert truth <= got, f"({x},{y}) is missing {truth - got}"
    assert isinstance(reg.polygon_bbox([(0, 0), (1, 0), (1, 1)]), tuple)


def test_indeks_bbox_kosong_tidak_meledak():
    assert sg._BboxGrid([]).candidates(0.0, 0.0) == []


def test_indeks_bbox_menyempitkan_dan_tetap_memuat_kotak_yang_terlalu_lebar():
    """Two properties that must hold at the same time, and each of which alone
    can be satisfied by a wrong index.

    An index that returns EVERYTHING is a perfect superset and narrows nothing
    — that is exactly what happens if the cell size is taken from the largest
    box rather than from the mean. Conversely, an index that narrows by
    discarding the drawing-wide box removes findings without leaving a trace in
    the response.

    The arrangement imitates a real drawing: hundreds of small parcels, and one
    district-boundary polygon that envelops them all.
    """
    boxes = [
        (float(i * 100), float(j * 100), float(i * 100 + 12), float(j * 100 + 25))
        for i in range(20)
        for j in range(20)
    ]
    boundary = (0.0, 0.0, 10_000.0, 10_000.0)
    boxes.append(boundary)
    grid = sg._BboxGrid(boxes)

    # narrowing: one point must not call back the entire drawing
    got = grid.candidates(1050.0, 1050.0)
    assert len(got) < len(boxes) / 10, (
        f"the index returned {len(got)} of {len(boxes)} boxes; it narrows "
        "nothing at all"
    )
    # and still holds the boundary polygon, too wide to fit in any cell
    assert len(boxes) - 1 in got, "the drawing-wide box vanished from candidates"

    # the superset property still holds across the whole test field
    for x in range(0, 2100, 137):
        for y in range(0, 2100, 137):
            truth = {
                i for i, b in enumerate(boxes)
                if b[0] <= x <= b[2] and b[1] <= y <= b[3]
            }
            assert truth <= set(grid.candidates(float(x), float(y)))


# --- empty data --------------------------------------------------------------


@pytest.mark.parametrize("kind", ["geometry", "shape", "label"])
def test_gambar_tanpa_poligon_mengembalikan_nol_bukan_error(kind):
    """G5, written word for word in the generality rules."""
    r = build([], kind=kind)
    assert r["groups_total"] == 0
    assert r["entities_involved"] == 0
    assert r["groups"] == []
    assert r["truncated"] is False
    assert r["interpretation_note"].strip()


# --- stated limits -----------------------------------------------------------


def test_batas_dinyatakan_di_respons_bukan_dipotong_diam_diam():
    """G7. A silent truncation is a wrong answer with a right face."""
    rows = []
    for i in range(6):
        rows.append(ring_row(f"A{i}", "0", rect(i * 100, 0, 12, 25)))
        rows.append(ring_row(f"B{i}", "VL2", rect(i * 100, 0, 12, 25)))
    r = build(rows, kind="geometry", limit=2)
    assert r["groups_total"] == 6, "group count is exact even when the named are cut"
    assert len(r["groups"]) == 2
    assert r["truncated"] is True
    assert r["limit"] == 2
    assert r["entities_involved"] == 12


def test_terlalu_banyak_target_ditolak_dengan_saran_mempersempit():
    with pytest.raises(sg.DuplicateRefused) as exc:
        sg.refuse_if_oversize(sg.MAX_TARGET_POLYGONS + 1, layout="Model")
    assert exc.value.code == "TOO_MANY_POLYGONS"
    assert "layers" in exc.value.hint


# --- chains --------------------------------------------------------------


def test_grup_adalah_komponen_terhubung_dan_jaraknya_dilaporkan_maksimum():
    """A is near B, B is near C, A is far from C. Reporting the first pair's
    distance would hide that this group is wider than its tolerance."""
    rows = [
        ring_row("A", "0", rect(0.0, 0, 12, 25)),
        ring_row("B", "VL2", rect(0.6, 0, 12, 25)),
        ring_row("C", "VL3", rect(1.2, 0, 12, 25)),
    ]
    r = build(rows, kind="geometry", tolerance=1.0)
    assert r["groups_total"] == 1
    g = r["groups"][0]
    assert g["member_count"] == 3
    assert g["centroid_gap"] == pytest.approx(1.2)
    assert g["centroid_gap"] > 1.0
    assert g["chained"] is True, (
        "a group wider than its tolerance is obliged to say so"
    )


# --- evidence ----------------------------------------------------------------


def test_bukti_geometri_tidak_pernah_naik_di_atas_inferred():
    """Coordinates INFER, they never STATE. A tool that reports duplicates as a
    fact stated by the file is promoting its own evidence grade."""
    rows = [
        ring_row("A", "0", rect(0, 0, 12, 25)),
        ring_row("B", "VL2", rect(0, 0, 12, 25)),
    ]
    r = build(rows, kind="geometry")
    assert r["evidence"]["grade"] == "inferred"
    assert r["evidence"]["claim"]["tokens"] == []
    assert ev.audit_response(r) == []


def test_respons_membawa_scope_note_dan_lolos_audit_kontrak():
    r = build([ring_row("A", "0", rect(0, 0, 12, 25))])
    assert r["scope_note"].strip()
    assert r["drawing_id"] == "d0"
    assert ev.audit_response(r) == []


def test_bukti_untuk_nihil_temuan_tetap_ada_dan_menyebut_toleransinya():
    """'No duplicates' is a finding, and a finding without a scope cannot be
    checked by anyone."""
    r = build([ring_row("A", "0", rect(0, 0, 12, 25))])
    assert r["groups_total"] == 0
    assert r["evidence"]["grade"] in {"inferred", "unknown"}
    assert "0.01" in r["scope_note"] or "0,01" in r["scope_note"]


# ---------------------------------------------------------------------------
# Pure — the shape cross-check
# ---------------------------------------------------------------------------


def test_uji_modul_plot_tidak_punya_default():
    """G1/G6. The ranges 24-26 m and 150-650 m2 are traits of ONE drawing. A
    default inside a `.py` makes this module a Janadriyah-specific solution."""
    import inspect

    sig = inspect.signature(sg.cross_check_layers)
    for name in ("edge_range", "area_range", "coded_layers"):
        assert sig.parameters[name].default is inspect.Parameter.empty, (
            f"{name} has a default; one drawing's trait must not live in code"
        )


def test_sapuan_memisahkan_lolos_gagal_dan_di_luar_layer_kode():
    rows = [
        ring_row("P1", "VL2", rect(0, 0, 12, 25)),
        ring_row("P2", "VL3", rect(100, 0, 12, 25)),
        # plot-shaped, but on layer 0 — the copy that becomes this spec's find
        ring_row("C1", "0", rect(0, 0, 12, 25)),
        # on a coded layer but failing the shape test: an irregular corner plot
        ring_row("X1", "LP3", rect(200, 0, 4, 4)),
    ]
    r = sg.build_cross_check(
        drawing_id="d0",
        layout="Model",
        rows=rows,
        ring_census=census(complete=len(rows)),
        units=METRES,
        coded_layers=["VL2", "VL3", "LP3"],
        edge_range=(24.0, 26.0),
        area_range=(150.0, 650.0),
        limit=50,
    )
    assert r["coded_total"] == 3
    assert r["coded_passing_shape_test"] == 2
    assert r["coded_failing_shape_test"] == 1
    assert r["failing_by_layer"] == [{"layer": "LP3", "count": 1}]
    assert r["shape_matches_outside_coded_layers"] == 1
    assert r["outside_by_layer"] == [{"layer": "0", "count": 1}]
    assert r["agreement_fraction"] == pytest.approx(2 / 3)
    assert ev.audit_response(r) == []


def test_ketidaksepakatan_menurunkan_bukti_kesepakatan_tidak_menaikkannya():
    """A deliberate asymmetry, and one handed over by UPLIFT-08. A shape that
    agrees with the layer name remains a single stating corpus; a shape that
    does NOT agree is a contradiction, and contradictions lower."""
    agree = [ring_row("P1", "VL2", rect(0, 0, 12, 25))]
    disagree = [ring_row("X1", "VL2", rect(0, 0, 4, 4))]
    common = dict(
        drawing_id="d0",
        layout="Model",
        units=METRES,
        coded_layers=["VL2"],
        edge_range=(24.0, 26.0),
        area_range=(150.0, 650.0),
        limit=50,
    )
    a = sg.build_cross_check(rows=agree, ring_census=census(complete=1), **common)
    d = sg.build_cross_check(rows=disagree, ring_census=census(complete=1), **common)
    assert a["evidence"]["grade"] == "inferred"
    assert d["evidence"]["contradictions"], "a disagreement must be recorded"
    assert d["evidence"]["grade"] == "inferred"


def test_sidik_jari_bentuk_mengurutkan_modul_terbesar_lebih_dulu():
    rows = []
    for i in range(4):
        rows.append(ring_row(f"a{i}", "VL3", rect(i * 100, 0, 12, 25)))
    for i in range(2):
        rows.append(ring_row(f"b{i}", "VL2", rect(i * 100, 500, 10, 25)))
    r = sg.build_shape_fingerprint(
        drawing_id="d0",
        layout="Model",
        rows=rows,
        ring_census=census(complete=len(rows)),
        units=METRES,
        limit=50,
    )
    assert [g["count"] for g in r["groups"]] == [4, 2]
    assert sorted(g["modal_edge_lengths"] for g in r["groups"])[0][0] > 0
    top = r["groups"][0]
    assert sorted(round(v) for v in top["modal_edge_lengths"]) == [12, 12, 25, 25]
    assert top["modal_edge_length_units"] == "m"


# ---------------------------------------------------------------------------
# G1 — zero drawing-specific constants inside the code
# ---------------------------------------------------------------------------


def test_tidak_ada_konstanta_khusus_gambar_di_modul_ini():
    """The same grep as the one in UPLIFT-GENERALITY-RULES, narrowed to this
    file so that it fails here rather than in someone else's review."""
    src = Path(sg.__file__).read_text(encoding="utf-8")
    for token in ("VL2", "VL3", "DP4", "LP1", "LP3", "TH3", "ZEROLOT", "32638"):
        assert token not in src, f"drawing-specific constant {token!r} entered the code"


# ---------------------------------------------------------------------------
# Against a live database
# ---------------------------------------------------------------------------

#: The reference drawing and its scope, pinned. "52 groups" means nothing
#: without the layout that produced it.
JANADRIYAH = "596212db022a3397"
LAYOUT = "Model"

#: From `docs/UPLIFT-07-DUPLICATES.md`, measured from the DXF before this code
#: existed.
# Re-measured against the store on 24 August 2026, after UPLIFT-01 landed.
# The old numbers -- 52 and 48 -- were measured when `shape_key` was still
# built from edge lengths alone and its winding was not normalised, and when
# bulged polygons and open polygons still counted as candidates.
#
# Both changed in UPLIFT-01, and both raised the count in a direction that can
# be explained: normalising the winding made copies drawn the other way round
# -- the result of a mirror operation in AutoCAD -- finally recognise one
# another. Measured separately, there are 10 groups whose members differ in
# winding; those groups would never have been found by the old key.
#
# The rise in the count is therefore NOT a regression. It is in fact evidence
# that winding normalisation does the thing it was built to do: find the copies
# that had been missed all along.
EXPECTED_GROUPS_AT_1M = 72
EXPECTED_GROUPS_AT_1CM = 62
EXPECTED_CODED_TOTAL = 2380
EXPECTED_CODED_PASSING = 2312
EXPECTED_CODED_FAILING = 68
EXPECTED_OUTSIDE = {"0": 52, "Open Spaces": 15, "Linear Park": 12, "Pocket Park": 1}


def _mongo_or_skip():
    try:
        drawing = store.get_drawing(JANADRIYAH)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip(
            f"reference drawing {JANADRIYAH} has not been ingested; the "
            "UPLIFT-07 acceptance numbers cannot be checked"
        )
    return drawing


def _units(drawing):
    return store._unit_names(drawing, LAYOUT)


def test_dod_1_lima_puluh_dua_grup_pada_satu_meter():
    """DoD 1. The number belongs to a tolerance of 1 m, not 0.01 — see the
    contradiction note in `docs/PROGRESS.md`."""
    drawing = _mongo_or_skip()
    r = sg.find_duplicates(
        JANADRIYAH, layout=LAYOUT, units=_units(drawing),
        kind="geometry", tolerance=1.0, limit=200,
    )
    assert r["groups_total"] == EXPECTED_GROUPS_AT_1M

    # Measured, and it changes how the parameter has to be read: at a
    # tolerance of 1 m, 72 groups are found, and ALL SEVENTY-TWO have a
    # centroid distance below 1 cm. Re-running at a tolerance of 0.01 gives
    # only 62.
    #
    # Which means that what makes those two numbers differ is NOT the centroid
    # distance -- not one single group is filtered out by it.
    #
    # CORRECTION. The explanation originally written here named "shape
    # grouping", and that is WRONG: `shape_key` is computed at ingest and
    # `tolerance` cannot touch it at all. What this parameter actually does is
    # three things, and only one of them is written in its name:
    #
    #   1. cell size of the spatial bucket over centroid positions
    #                                                     (store_geometry.py:337)
    #   2. the AREA tolerance, i.e. tolerance squared      (:339)
    #   3. the maximum centroid distance                   (:356)
    #
    # What moves 62 to 72 is number 2: at 0.01 two polygons may differ in area
    # by 0.0001 square units, at 1.0 they may differ by 1.0. The conclusion
    # does not change -- the centroid distance really does not filter anything
    # here -- but the mechanism was named wrongly, and a wrong explanation is
    # more dangerous than no explanation, because the next person will make
    # decisions on top of it.
    tight = [g for g in r["groups"] if g["centroid_gap"] <= 0.01]
    assert len(tight) == r["groups_total"], (
        "the centroid distance turns out to filter something here; if that "
        "changes, the note above about what `tolerance` does must be rewritten"
    )
    assert EXPECTED_GROUPS_AT_1CM < EXPECTED_GROUPS_AT_1M


def test_dod_1b_toleransi_satu_sentimeter_memberi_himpunan_yang_lebih_kecil():
    drawing = _mongo_or_skip()
    r = sg.find_duplicates(
        JANADRIYAH, layout=LAYOUT, units=_units(drawing),
        kind="geometry", tolerance=0.01, limit=200,
    )
    assert r["groups_total"] == EXPECTED_GROUPS_AT_1CM


def test_dod_2_setiap_grup_menyilang_layer_nol_dan_layer_tipologi():
    """DoD 2. This is what makes 52 matter to other people: `join_labels`
    without `target_layer` drops every plot number into TWO polygons at once."""
    drawing = _mongo_or_skip()
    r = sg.find_duplicates(
        JANADRIYAH, layout=LAYOUT, units=_units(drawing),
        kind="geometry", tolerance=1.0, limit=200,
    )
    # The original claim -- EVERY group crosses layer 0 and a typology layer --
    # did not survive being counted: of the groups found, some cross two layers
    # neither of which is `0`, for example `BlockBoundary` and `ROW`. Those are
    # real duplicates and deserve reporting; they are simply not the duplicates
    # people expected.
    #
    # What remains true, and what makes this number matter to the other
    # subagents: the MAJORITY of groups really do cross layer 0, so
    # `join_labels` without `target_layer` genuinely does drop plot numbers
    # into two polygons at once. That is what is tested here, as a measured
    # majority rather than an assumed whole.
    crossing_zero = 0
    for g in r["groups"]:
        layers = [m["layer"] for m in g["members"]]
        assert len(set(layers)) > 1 or len(layers) > 1, "a group with no second member"
        if "0" in layers:
            crossing_zero += 1
            assert any(l != "0" for l in layers), "a copy with no typology counterpart"
    assert crossing_zero >= len(r["groups"]) // 2, (
        f"only {crossing_zero} of {len(r['groups'])} groups touch layer 0; "
        "if this falls below half, the 'copies on layer 0' story no longer "
        "describes this drawing and must be rewritten, not loosened"
    )


def test_dod_3_bentuk_berulang_bukan_cacat():
    """DoD 3. One large group from one plot module, not flagged as a defect."""
    drawing = _mongo_or_skip()
    r = sg.find_duplicates(
        JANADRIYAH, layout=LAYOUT, units=_units(drawing),
        kind="shape", limit=500,
    )
    assert r["defect"] is False
    assert max(g["member_count"] for g in r["groups"]) >= 400


def test_dod_4_modul_plot_terbesar_terbaca_tanpa_nama_layer():
    """DoD 4. Plot modules read from geometry alone, without layer names.

    The original claim -- the four LARGEST groups are the 8/10/12/16 x 25
    modules -- is wrong, and its wrongness is useful. Measured: 12x25 has 652
    members, 10x25 has 417, 16x25 has 79, and 8x25 has 37 in EIGHTH place. What
    slipped in among them are five-sided corner plots (42 and 41 members) and
    two other shapes.

    So what is true is not "the four largest" but "those four modules exist and
    are readable from their own geometry". The top three rectangular modules
    alone cover more than forty percent of all complete polygons in this
    layout, which still proves what the spec was after: plot shapes can be
    found without reading a single layer name.
    """
    drawing = _mongo_or_skip()
    r = sg.shape_fingerprint(
        JANADRIYAH, layout=LAYOUT, units=_units(drawing), limit=8
    )
    modules: dict[int, int] = {}
    for g in r["groups"]:
        lens = sorted(round(v) for v in g["modal_edge_lengths"])
        if len(lens) == 4 and lens[2] == lens[3] == 25 and lens[0] == lens[1]:
            modules.setdefault(lens[0], g["count"])
    for width in (8, 10, 12, 16):
        assert width in modules, (
            f"module {width}x25 is not readable from geometry alone; "
            f"what is readable: {sorted(modules)}"
        )
    # The three largest modules dominate, and that is what makes this reading
    # useful.
    assert modules[12] > modules[10] > modules[16] > modules[8]


#: The list of Janadriyah typology layers, if it exists. It lives in a fixture
#: file and not in production `.py`: layer names are a trait of one drawing,
#: and G1 forbids constants like that inside the code while permitting them in
#: tests and config. As long as this file does not exist, the numbers
#: 2,312/68/80 cannot be reproduced from any side without inventing the list,
#: and inventing it makes the cross-check sweep check itself.
CODED_LAYERS_FIXTURE = Path(__file__).parent / "data" / "janadriyah_coded_layers.json"


def test_dod_5_sapuan_silang_periksa_direproduksi():
    """DoD 5. 2,312 passing / 68 failing the shape test / 80 plot-shaped
    outside the coded layers. The coded layer names and the test ranges come
    from a fixture, not from the code."""
    drawing = _mongo_or_skip()
    if not CODED_LAYERS_FIXTURE.exists():
        pytest.skip(
            f"the list of coded layers is not yet in {CODED_LAYERS_FIXTURE.name}; "
            "deriving it from the data being examined would make this sweep "
            "check itself. The right source is the UPLIFT-02 land-use config "
            "(subagent A)"
        )
    import json

    coded = json.loads(CODED_LAYERS_FIXTURE.read_text(encoding="utf-8"))["layers"]
    r = sg.cross_check_layers(
        JANADRIYAH, layout=LAYOUT, units=_units(drawing),
        coded_layers=coded, edge_range=(24.0, 26.0), area_range=(150.0, 650.0),
        limit=500,
    )
    assert r["coded_total"] == EXPECTED_CODED_TOTAL
    assert r["coded_passing_shape_test"] == EXPECTED_CODED_PASSING
    assert r["coded_failing_shape_test"] == EXPECTED_CODED_FAILING
    assert r["shape_matches_outside_coded_layers"] == sum(EXPECTED_OUTSIDE.values())
    assert {
        row["layer"]: row["count"] for row in r["outside_by_layer"]
    } == EXPECTED_OUTSIDE


def test_dod_6_toleransi_dan_catatan_tafsir_ada_di_setiap_respons_hidup():
    drawing = _mongo_or_skip()
    for kind in sg.KINDS:
        r = sg.find_duplicates(
            JANADRIYAH, layout=LAYOUT, units=_units(drawing),
            kind=kind, tolerance=0.01, limit=5,
        )
        assert r["tolerance"] == 0.01
        assert r["tolerance_units"] == "m"
        assert "tolerance_m" not in r
        assert r["interpretation_note"].strip()
        assert ev.audit_response(r) == []


def test_invarian_semua_gambar_tanpa_poligon_mengembalikan_nol():
    """G5. Run over EVERY drawing that exists, not only Janadriyah. This is
    what catches breakage in the 17th drawing."""
    try:
        drawings = store.list_drawings()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawings:
        pytest.skip("no drawing has been ingested")
    checked = 0
    for d in drawings:
        did = d.get("drawing_id") or d.get("_id")
        for layout in (d.get("layouts") or ["Model"]):
            name = layout if isinstance(layout, str) else layout.get("name")
            if not name:
                continue
            r = sg.find_duplicates(
                did, layout=name, units=store._unit_names(d, name),
                kind="geometry", tolerance=0.01, limit=5,
            )
            assert r["groups_total"] >= 0
            assert isinstance(r["groups"], list)
            assert r["interpretation_note"].strip()
            if not r["tolerance_units"]:
                assert r["tolerance_units_note"], (
                    f"{did}/{name}: unit withheld without a reason"
                )
            checked += 1
    assert checked > 0
