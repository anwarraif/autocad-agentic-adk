"""`join_labels`: the text INSIDE a parcel, not the text near it.

Two parts, deliberately kept apart, following the habit of `test_measure.py`.

The first part is **pure**: `_join` takes Mongo-shaped documents and returns
rows. No database, no FastAPI. What is tested there is a PROPERTY — a bounding
box answers a different question from its polygon; a point on the line is
reported inside AND counted; an exceeded limit refuses instead of truncating
in silence. A property like that holds on the 17th drawing nobody has ever
seen (G5), while the Janadriyah numbers will never catch a breakage there.

The second part uses a **live database** and skips, naming what it skipped, if
there is none. The UPLIFT-03 Definition of Done numbers — 8 of 9 education
parcels, 2,380 of 2,380 plot numbers — are properties of data that has been
ingested, and a test that mocks the query would be green while the query is
wrong. `pytest -rs` names what was skipped; read that before believing a green
suite.

One Definition of Done number is NOT reproduced, and that is a finding not a
failure: "six mosque parcels" was counted before the `ring_status` gate was
installed. Two of the five `Local Mosque` parcels are bulged, so their vertex
sequence is the chord and not the real boundary. Under the correct filter the
targets are **four**, and the other two are skipped and counted. See
`test_the_mosques_are_four_targets_not_six_and_the_spec_says_six`.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import store_spatial as sp  # noqa: E402


# The real projected coordinates this project uses, and the Janadriyah grid
# angle. Not decoration: the whole class of defect this pure part tests only
# appears at coordinate magnitudes like these, and the bbox-vs-polygon gap only
# appears if the shape is rotated with respect to the axes.
E0, N0 = 691942.0, 2750756.0
GRID_ANGLE = math.radians(49.0)


def rect(cx: float, cy: float, w: float, h: float, ang: float = 0.0):
    c, s = math.cos(ang), math.sin(ang)
    return [
        [cx + dx * c - dy * s, cy + dx * s + dy * c]
        for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
    ]


def target(handle: str, ring, layer: str = "L", area: float | None = None) -> dict:
    """A Mongo-shaped target document — exactly the projection the code reads."""
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return {
        "handle": handle,
        "layer": layer,
        "ring": [list(p) for p in ring],
        "ring_origin": [min(xs), min(ys)],
        "area": area,
    }


def label(handle: str, x: float | None, y: float | None, **over) -> dict:
    doc = {
        "handle": handle,
        "layer": "T",
        "type": "TEXT",
        "text": "t",
        "anchor_point": None if x is None else [x, y],
        "anchor_basis": None if x is None else "TEXT.insert",
    }
    doc.update(over)
    return doc


# ---------------------------------------------------------------------------
# Pure: why this tool exists
# ---------------------------------------------------------------------------


def test_a_bounding_box_answers_a_different_question_from_the_polygon():
    """Why `join_labels` exists, in a form that needs not one drawing.

    Measured in Janadriyah on the `Private School 205D12B` parcel: the bbox
    test pulls in 42 texts, the correct point-in-polygon pulls in 1. The cause
    is not tolerance — the plot sits on a rotated grid, so the axis-aligned box
    is far larger than its polygon. Here the same property is generated: a
    rectangle rotated by 49 degrees has a bounding box far larger in area than
    itself, and the difference is not a small error.
    """
    ring = rect(E0, N0, 12.0, 25.0, GRID_ANGLE)
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    box = (min(xs), min(ys), max(xs), max(ys))

    # Points spread evenly inside its BOX.
    pts = []
    n = 40
    for i in range(n):
        for j in range(n):
            pts.append(
                (
                    box[0] + (box[2] - box[0]) * (i + 0.5) / n,
                    box[1] + (box[3] - box[1]) * (j + 0.5) / n,
                )
            )

    labels = [label(f"L{k}", x, y) for k, (x, y) in enumerate(pts)]
    result = sp._join([target("T1", ring)], labels, budget=sp.PAIRS_BUDGET)

    in_box = len(pts)  # all of them, by definition
    in_polygon = result["rows"][0]["label_count"]

    assert in_polygon < in_box
    # The area ratio is about 0.56 for 12x25 at 49 degrees; what is tested is
    # not the exact number but that the bbox answer overstates materially, not
    # marginally.
    assert in_polygon < in_box * 0.75, (
        "if the bbox test and the polygon test are nearly the same, what is "
        "being used is not the polygon"
    )


def test_the_prefilter_leaves_pairs_on_the_order_of_the_labels_not_their_product():
    """The 2,000,000 limit is counted AFTER the prefilter, and here is why.

    Counted over the raw product, the UPLIFT-03 plot number test itself
    (2,380 x 2,380 = 5,664,400) hits its own limit, and its advice
    — "narrow label_layer" — is already narrowed as far as it goes.
    After the spatial prefilter, the pairs that survive are on the order of the
    label count.
    """
    n = 60
    targets = []
    labels = []
    for i in range(n):
        cx, cy = E0 + i * 40.0, N0
        targets.append(target(f"T{i}", rect(cx, cy, 12.0, 25.0, GRID_ANGLE)))
        labels.append(label(f"L{i}", cx, cy))

    result = sp._join(targets, labels, budget=sp.PAIRS_BUDGET)

    assert result["pairs_after_prefilter"] <= 4 * len(labels), (
        "the prefilter must bring the pairs down to the order of the label "
        f"count, not {n * n} raw pairs"
    )
    assert result["pairs_tested"] == result["pairs_after_prefilter"]
    assert sum(r["label_count"] for r in result["rows"]) == n


def test_a_target_far_larger_than_the_grid_cell_is_still_found():
    """G6: do not assume a shape, and do not assume a size.

    One parcel covering the whole extent alongside hundreds of small parcels is
    a normal state (`BlockBoundary`, `NBHD A`). A grid index that only stores a
    target in the cells it crosses will MISS that wide target if it is not
    handled; if it is stored in every cell, the insertion cost explodes. What
    is tested: it is still found.
    """
    big = target("BIG", rect(E0, N0, 8000.0, 8000.0), layer="BlockBoundary")
    small = [
        target(f"S{i}", rect(E0 + i * 30.0, N0, 12.0, 25.0, GRID_ANGLE))
        for i in range(100)
    ]
    labels = [label(f"L{i}", E0 + i * 30.0, N0) for i in range(100)]

    result = sp._join([big] + small, labels, budget=sp.PAIRS_BUDGET)
    rows = {r["target_handle"]: r for r in result["rows"]}

    assert rows["BIG"]["label_count"] == 100, "the wide target lost its labels"
    assert result["labels_matched_more_than_one_target"] == 100, (
        "each label falls inside its small parcel AND inside the big block; "
        "that is a topological fact to be reported, not hidden"
    )


def test_the_grid_agrees_with_brute_force_on_awkward_shapes():
    """A spatial index is an optimisation. One that changes the answer is a bug."""
    import random

    rng = random.Random(20260824)
    targets = []
    for i in range(120):
        cx = E0 + rng.uniform(-500, 500)
        cy = N0 + rng.uniform(-500, 500)
        targets.append(
            target(
                f"T{i}",
                rect(cx, cy, rng.uniform(2, 300), rng.uniform(2, 300), rng.uniform(0, 3)),
            )
        )
    pts = [
        (E0 + rng.uniform(-600, 600), N0 + rng.uniform(-600, 600)) for _ in range(400)
    ]
    labels = [label(f"L{k}", x, y) for k, (x, y) in enumerate(pts)]

    fast = sp._join(targets, labels, budget=sp.PAIRS_BUDGET)
    slow = sp._join(targets, labels, budget=sp.PAIRS_BUDGET, _index=False)

    def shape(res):
        return {
            r["target_handle"]: sorted(l["handle"] for l in r["labels"])
            for r in res["rows"]
        }

    assert shape(fast) == shape(slow)
    assert fast["boundary_cases"] == slow["boundary_cases"]


def test_a_point_on_the_line_is_reported_inside_and_counted_as_a_boundary():
    """A plot number the drafter snapped to its plot line is not an edge case.

    It is how the drawing was made. A two-valued answer must either silently
    choose a side or silently lose the label; the right thing is to report
    both.
    """
    ring = rect(E0, N0, 12.0, 25.0, GRID_ANGLE)
    on_edge = (
        (ring[0][0] + ring[1][0]) / 2.0,
        (ring[0][1] + ring[1][1]) / 2.0,
    )
    result = sp._join(
        [target("T1", ring)],
        [label("ON", *on_edge), label("IN", E0, N0)],
        budget=sp.PAIRS_BUDGET,
    )
    row = result["rows"][0]

    assert sorted(l["handle"] for l in row["labels"]) == ["IN", "ON"]
    assert {l["handle"]: l["where"] for l in row["labels"]} == {
        "ON": "boundary",
        "IN": "inside",
    }
    assert result["boundary_cases"] == 1


def test_boundary_cases_is_present_even_when_it_is_zero():
    """"No boundary cases in this data" must be a measurement, not an assumption."""
    result = sp._join(
        [target("T1", rect(E0, N0, 12.0, 25.0))],
        [label("IN", E0, N0)],
        budget=sp.PAIRS_BUDGET,
    )
    assert result["boundary_cases"] == 0
    assert "boundary_cases" in result


def test_a_label_with_no_anchor_point_is_skipped_and_counted():
    """The box centre is NEVER a silent fallback.

    A text box depends on the font, and the reference drawing's SHX font is
    missing, so the box itself is already a guess. Testing a guess while
    reporting an insertion point is a wrong number wearing the clothes of a
    right one.
    """
    result = sp._join(
        [target("T1", rect(E0, N0, 100.0, 100.0))],
        [label("HAS", E0, N0), label("NONE", None, None)],
        budget=sp.PAIRS_BUDGET,
    )
    assert result["skipped_no_anchor"] == 1
    assert result["rows"][0]["label_count"] == 1
    assert result["candidates_examined"] == 2


def test_the_pair_budget_refuses_instead_of_truncating_in_silence():
    """G7: the limit is stated, and above it it refuses with a suggestion."""
    n = 40
    ring = rect(E0, N0, 500.0, 500.0)
    targets = [target(f"T{i}", ring) for i in range(n)]
    labels = [label(f"L{i}", E0, N0) for i in range(n)]

    with pytest.raises(sp.SpatialRefused) as caught:
        sp._join(targets, labels, budget=100)

    assert caught.value.code == "JOIN_TOO_LARGE"
    assert "100" in caught.value.message
    assert caught.value.hint, "a refusal with no suggestion sends the agent back to guessing"


def test_the_budget_is_measured_after_the_prefilter_not_before():
    """The number that cancels this spec's Definition of Done, as a test.

    2,380 targets x 2,380 labels = 5,664,400 raw pairs. Here the same shape is
    scaled down: the raw product is far above the budget, the pairs after the
    prefilter are far below it, and the call must SUCCEED.
    """
    n = 200
    targets = [
        target(f"T{i}", rect(E0 + i * 40.0, N0, 12.0, 25.0, GRID_ANGLE))
        for i in range(n)
    ]
    labels = [label(f"L{i}", E0 + i * 40.0, N0) for i in range(n)]

    raw = n * n
    budget = raw // 4
    result = sp._join(targets, labels, budget=budget)

    assert raw > budget, "this test means nothing if the product is below the budget"
    assert result["pairs_after_prefilter"] < budget
    assert sum(r["label_count"] for r in result["rows"]) == n


def test_a_target_without_a_label_is_a_row_not_an_omission():
    """`targets_without_label` prevents the classic mistake: counting labels
    as a count of parcels."""
    result = sp._join(
        [
            target("WITH", rect(E0, N0, 50.0, 50.0)),
            target("WITHOUT", rect(E0 + 1000.0, N0, 50.0, 50.0)),
        ],
        [label("L1", E0, N0)],
        budget=sp.PAIRS_BUDGET,
    )
    rows = {r["target_handle"]: r for r in result["rows"]}
    assert rows["WITHOUT"]["label_count"] == 0
    assert rows["WITHOUT"]["labels"] == []
    assert len(result["rows"]) == 2


def test_a_degenerate_ring_never_swallows_a_point():
    """G8: a broken drawing is a normal state.

    A ring with fewer than three vertices has no inside. It must not throw, and
    it must not answer "inside".
    """
    result = sp._join(
        [target("T1", [[E0, N0], [E0 + 1, N0]])],
        [label("L1", E0, N0)],
        budget=sp.PAIRS_BUDGET,
    )
    assert result["rows"][0]["label_count"] == 0


# ---------------------------------------------------------------------------
# Against a live database
# ---------------------------------------------------------------------------

#: The reference drawing, and the numbers measured from it on 23 August 2026.
JANADRIYAH = "596212db022a3397"

#: The nine education parcels and the handles of the T4 texts inside them. The
#: first eight carry the text `'lvtR jugdlD'`; `205D12B` carries a DIFFERENT T4
#: text — see `test_every_education_parcel_holds_one_t4_label_and_one_of_them_differs`.
EDUCATION = {
    "205D12D": "70C91A7",
    "205D12F": "70C91A8",
    "205D12E": "70C91AF",
    "205D130": "70C91B0",
    "205CD54": "70C91B7",
    "205D153": "70C91C9",
    "205D135": "70C91CD",
    "205D154": "70C91DF",
    "205D12B": "70C91E2",
}

#: The education facility text, verbatim from the file. Its reading (مرفق
#: تعليمي) is NOT asserted here: the SHX glyph map belongs to UPLIFT-04, and
#: writing the pair `'lvtR jugdlD' -> 'مرفق تعليمي'` into the code would be a
#: drawing-specific constant (G1).
EDUCATION_TEXT = "lvtR jugdlD"

EDUCATION_LAYERS = [
    "Intermediate School",
    "Secondary School",
    "Early Education ELC 600",
    "Primary School",
    "Early Education ELC 300",
    "Private School",
]

#: Thirteen plot typology layers. Their `complete` polygon count is exactly
#: 2,380. Drawing-specific layer names may ONLY live in config and in tests
#: (G1).
PLOT_TYPOLOGY_LAYERS = [
    "VL2", "VL3", "VL4", "VL4 SPECIAL",
    "LP1", "LP2", "LP3", "LP4", "LP5",
    "DP4", "DP4 ZEROLOT",
    "C10", "TH3",
]
PLOT_COUNT = 2380


def _mongo_or_skip():
    try:
        from app.mongo import COLL_DRAWINGS, coll

        drawing = coll(COLL_DRAWINGS).find_one({"_id": JANADRIYAH})
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database could be reached: {type(exc).__name__}")
    if not drawing:
        pytest.skip(
            f"the reference drawing {JANADRIYAH} has not been ingested; the "
            "UPLIFT-03 acceptance numbers cannot be checked"
        )
    return drawing


def test_primary_school_returns_two_rows_with_one_label_each():
    """Definition of Done 1."""
    _mongo_or_skip()
    r = sp.join_labels(
        JANADRIYAH, layout="Model", target_layer="Primary School", label_layer="T4"
    )
    assert r["total_targets"] == 2
    assert r["targets_with_label"] == 2
    assert r["label_count"] == 2
    got = {row["target_handle"]: [l["handle"] for l in row["labels"]] for row in r["rows"]}
    assert got == {"205D153": ["70C91C9"], "205D154": ["70C91DF"]}


def test_every_education_parcel_holds_one_t4_label_and_one_of_them_differs():
    """A SPEC CONTRADICTION, recorded not patched. Definition of Done 2.

    The spec demands "8 labelled, 1 (`205D12B`) empty", and its example
    response writes `"labels": []` for that parcel. Measured: **all nine**
    parcels contain exactly one `T4` layer text. What is right about the
    original claim is the narrower part — eight texts reading `'lvtR jugdlD'`
    fall exactly inside the eight school/ELC parcels — and `205D12B` contains a
    DIFFERENT T4 text (`70C91E2`, `'j{hvD'`).

    So what is wrong is not the measurement but the conclusion drawn from it:
    "contains no education text" was read as "contains no text". The irony is
    on target — `targets_without_label` is named by the spec as the field that
    prevents the classic mistake of counting labels as a count of parcels, and
    the ground truth number behind it was born of the same kind of mistake.

    What is NOT done here: adding a text-content filter so that the number goes
    back to 8. A tool that filters on text whose answer is already known stops
    being able to find what is not yet known. Recorded in docs/PROGRESS.md.
    """
    _mongo_or_skip()
    r = sp.join_labels(
        JANADRIYAH,
        layout="Model",
        target_layer=EDUCATION_LAYERS,
        label_layer="T4",
        limit=20,
    )
    assert r["total_targets"] == 9
    assert r["targets_with_label"] == 9
    assert r["targets_without_label"] == 0
    assert "empty_note" not in r

    got = {
        row["target_handle"]: [l["handle"] for l in row["labels"]] for row in r["rows"]
    }
    assert got == {k: [v] for k, v in EDUCATION.items()}

    by_text = {
        row["target_handle"]: row["labels"][0]["text"] for row in r["rows"]
    }
    education = [h for h, t in by_text.items() if t == EDUCATION_TEXT]
    assert len(education) == 8
    assert "205D12B" not in education
    assert by_text["205D12B"] != EDUCATION_TEXT


def test_the_mosques_are_four_targets_not_six_and_the_spec_says_six():
    """A SPEC CONTRADICTION, recorded not patched.

    The UPLIFT-03 Definition of Done demands "6 of 6" mosque parcels labelled:
    five `Local Mosque` and one `Jumaa Mosque`. Measured against the store
    after the UPLIFT-01 migration, two of the five `Local Mosque` — `205CD4A`
    and `205CD4B` — are `ring_status: bulge`. Their vertex sequence is the
    CHORD, not the real boundary, so it must not be used to test whether a
    point is inside it.

    The number 6 was measured on 23 August, before the `ring_status` gate
    existed; it was born of a filter that let bulged polygons through. What is
    right under the correct filter is **4 targets, 4 labelled, 2 skipped and
    counted as `skipped_bulge`** — and that is a SMALLER answer and a more
    correct one. Recorded in docs/PROGRESS.md.
    """
    _mongo_or_skip()
    r = sp.join_labels(
        JANADRIYAH,
        layout="Model",
        target_layer=["Local Mosque", "Jumaa Mosque"],
        label_layer="T4",
        limit=20,
    )
    assert r["total_targets"] == 4
    assert r["targets_with_label"] == 4
    assert r["skipped_bulge"] == 2
    assert r["skipped_targets_total"] == 2
    texts = sorted(l["text"] for row in r["rows"] for l in row["labels"])
    assert texts.count("ls{] lpgD") == 3
    assert texts.count("ls{] {hlU") == 1


def test_every_plot_holds_exactly_one_plot_number_in_one_call():
    """Definition of Done 4 and 10 — the sharpest regression test for
    point-in-polygon.

    2,380 typology plots, 2,449 plot number texts in the same layout, and
    exactly one text inside every plot. Any result other than 2,380/2,380 means
    the algorithm is broken. Run in ONE call, and the pairs after the prefilter
    must be far below 2,000,000 — the raw product is 5,828,620.
    """
    _mongo_or_skip()
    r = sp.join_labels(
        JANADRIYAH,
        layout="Model",
        target_layer=PLOT_TYPOLOGY_LAYERS,
        label_layer="C-PROP-PlotNumber",
        limit=5,
    )
    assert r["total_targets"] == PLOT_COUNT
    assert r["targets_with_label"] == PLOT_COUNT
    assert r["targets_without_label"] == 0
    assert all(row["label_count"] == 1 for row in r["rows"])
    assert r["label_count"] == PLOT_COUNT

    assert r["pairs_after_prefilter"] < sp.PAIRS_BUDGET
    assert r["pairs_budget"] == sp.PAIRS_BUDGET
    assert r["candidates_examined"] > PLOT_COUNT
    assert r["truncated"] is True and r["returned"] == 5


def test_no_education_label_falls_inside_a_house_plot():
    """Definition of Done 5, the negative test.

    The eight `'lvtR jugdlD'` texts each fall exactly inside one school parcel,
    and ZERO in a house plot. That is what makes the label a second source of
    evidence independent of the layer name; if it fell in houses too, it would
    state nothing.
    """
    _mongo_or_skip()
    r = sp.join_labels(
        JANADRIYAH,
        layout="Model",
        target_layer=PLOT_TYPOLOGY_LAYERS,
        label_layer="T4",
        limit=200,
    )
    hits = [
        l for row in r["rows"] for l in row["labels"] if l["text"] == "lvtR jugdlD"
    ]
    assert hits == []


def test_the_bbox_answer_and_the_polygon_answer_on_the_same_parcel():
    """Definition of Done 6 — why this tool exists, measured on the reference
    drawing.

    `Private School 205D12B` sits on a grid rotated by about 49 degrees, so its
    axis-aligned bounding box is far larger than its polygon. The bbox test
    pulls in dozens of texts; point-in-polygon pulls in one.

    What is tested is not the number 42 but the DISTANCE between them: the bbox
    answer here is not a rough answer, it is the wrong answer.
    """
    _mongo_or_skip()
    from app.mongo import COLL_ENTITIES, coll

    doc = coll(COLL_ENTITIES).find_one(
        {"drawing_id": JANADRIYAH, "handle": "205D12B"}, {"bbox": 1}
    )
    box = doc["bbox"]
    by_bbox = coll(COLL_ENTITIES).count_documents(
        {
            "drawing_id": JANADRIYAH,
            "layout": "Model",
            "type": "TEXT",
            "anchor_point.0": {"$gte": box["min"][0], "$lte": box["max"][0]},
            "anchor_point.1": {"$gte": box["min"][1], "$lte": box["max"][1]},
        }
    )
    r = sp.join_labels(
        JANADRIYAH,
        layout="Model",
        target_handles=["205D12B"],
        label_type="TEXT",
    )
    by_polygon = r["label_count"]

    assert by_bbox >= 10, "the reference parcel should have many texts in its box"
    assert by_polygon < by_bbox / 5, (
        f"bbox {by_bbox} vs polygon {by_polygon}: if the two are close, what is "
        "being tested is not the polygon"
    )


def test_polygons_that_are_not_complete_are_skipped_and_counted_by_cause():
    """Definition of Done 8.

    The old filter — `ring` is not null and `ring_truncated` is false —
    accepted 4,388 bulged and open polygons across the whole store, MORE than
    the 3,485 that are genuinely valid. A target that was skipped is not a
    target that did not match, so it is counted, broken down by cause.
    """
    _mongo_or_skip()
    r = sp.join_labels(JANADRIYAH, layout="Model", label_layer="T4", limit=1)
    for key in (
        "skipped_bulge",
        "skipped_open",
        "skipped_oversize",
        "skipped_degenerate",
        "skipped_non_planar",
        "skipped_unreadable",
    ):
        assert key in r, f"{key} must be present even when zero"
        assert isinstance(r[key], int)
    assert r["skipped_bulge"] > 0, "the reference drawing has 1,298 bulged polygons"
    assert r["skipped_targets_total"] == sum(
        r[f"skipped_{cause}"] for cause in sp.SKIP_CAUSES
    )


def test_a_join_never_pairs_across_layouts():
    """A G5 invariant.

    Polygons in Model and text on a sheet do not share a coordinate frame in
    any meaningful way. A cross-layout join produces pairs that look convincing
    and mean nothing at all.
    """
    _mongo_or_skip()
    from app.mongo import COLL_ENTITIES, coll

    r = sp.join_labels(
        JANADRIYAH, layout="Model", target_layer="Primary School", label_layer="T4"
    )
    handles = [l["handle"] for row in r["rows"] for l in row["labels"]]
    layouts = coll(COLL_ENTITIES).distinct(
        "layout", {"drawing_id": JANADRIYAH, "handle": {"$in": handles}}
    )
    assert layouts == ["Model"]


def test_the_layout_is_required_and_saying_nothing_is_refused():
    """`layout` is mandatory and has no default."""
    _mongo_or_skip()
    for bad in ("", "   ", None):
        with pytest.raises(sp.SpatialRefused) as caught:
            sp.join_labels(JANADRIYAH, layout=bad, target_layer="Primary School")
        assert caught.value.code == "LAYOUT_REQUIRED"


def test_the_response_carries_its_evidence_and_its_scope():
    """The UPLIFT-08 contract: every response that carries MEANING carries
    evidence."""
    _mongo_or_skip()
    from app import evidence as ev

    r = sp.join_labels(
        JANADRIYAH, layout="Model", target_layer="Primary School", label_layer="T4"
    )
    assert r["scope_note"].strip()
    assert ev.audit_response(r) == []
    block = r["evidence"]
    # Point-in-polygon is a COORDINATE corpus, and coordinates never STATE.
    # They corroborate; they do not speak.
    assert block["grade"] in {"inferred", "unknown"}
    assert block["not_established"] is not None or block["grade"] == "inferred"


def test_an_area_never_arrives_as_a_bare_number():
    """G2: every number carries its unit, and a unit is allowed to be absent."""
    _mongo_or_skip()
    r = sp.join_labels(
        JANADRIYAH, layout="Model", target_layer="Primary School", label_layer="T4"
    )
    area = r["rows"][0]["target_area"]
    assert isinstance(area, dict)
    assert area["unit"] == "m2"
    assert area["method"]
    assert area["value"] == pytest.approx(2748.875, abs=0.01)


def test_labels_inside_answers_what_this_plot_is_in_one_call():
    """Definition of Done 7 — the `get_entity` enrichment.

    This is the shape of the question that failed in the 23 August meeting:
    someone clicked a plot number and then asked for its parcel area, and was
    answered about the TEXT entity they clicked. The parcel `Primary School
    205D153` contains three texts — its block number, its plot number, and its
    facility label — and all three are answers to "what is this plot". Not
    narrowed to one layer: narrowing it to the layer whose answer is already
    known makes this tool able only to confirm, not to find.
    """
    _mongo_or_skip()
    out = sp.labels_inside(JANADRIYAH, "205D153")
    assert out["total"] == 3
    assert "70C91C9" in [l["handle"] for l in out["labels"]]
    assert {l["layer"] for l in out["labels"]} == {
        "C-PROP-BlockNumber",
        "C-PROP-PlotNumber",
        "T4",
    }
    assert out["method"]


def test_join_labels_holds_its_invariants_on_every_ingested_drawing():
    """G5: invariant tests are run over ALL the drawings, not one.

    The Janadriyah numbers will never catch a breakage in the 17th drawing. The
    properties tested here hold anywhere: one response belongs to one drawing
    and one layout; its evidence contract is intact; a drawing with no
    `complete` polygon answers ZERO together with its reason, rather than
    throwing.

    Eleven of the 18 drawings are in inches, three declare no units at all, and
    some contain only block definitions. A clean `audit_response` across all
    three kinds is what proves no unit was written because the drawing that
    happened to be open uses metres (G2).
    """
    _mongo_or_skip()
    from app import evidence as ev
    from app.mongo import COLL_DRAWINGS, COLL_ENTITIES, coll

    entities = coll(COLL_ENTITIES)
    ids = [d["_id"] for d in coll(COLL_DRAWINGS).find({}, {"_id": 1})]
    assert len(ids) >= 16, "this invariant means nothing over a handful of drawings"

    checked = 0
    empty = 0
    for did in ids:
        busiest = list(
            entities.aggregate(
                [
                    {"$match": {"drawing_id": did}},
                    {"$group": {"_id": "$layout", "n": {"$sum": 1}}},
                    {"$sort": {"n": -1}},
                    {"$limit": 1},
                ]
            )
        )
        if not busiest:
            continue
        layout = busiest[0]["_id"]
        r = sp.join_labels(did, layout=layout, label_type="ANY", limit=3)
        checked += 1

        assert r["drawing_id"] == did
        assert r["layout"] == layout
        assert ev.audit_response(r) == [], f"{did}: the evidence contract is not intact"
        assert isinstance(r["boundary_cases"], int)
        assert r["pairs_after_prefilter"] <= r["pairs_budget"]

        if r["total_targets"] == 0:
            empty += 1
            assert r["label_count"] == 0
            assert r["evidence"]["grade"] == "unknown"
            assert r["evidence"]["not_established"]
            continue

        # Not one pair crosses a layout.
        handles = [row["target_handle"] for row in r["rows"]]
        handles += [l["handle"] for row in r["rows"] for l in row["labels"]]
        if handles:
            layouts = entities.distinct(
                "layout", {"drawing_id": did, "handle": {"$in": handles}}
            )
            assert layouts == [layout], f"{did}: a cross-layout pair"

    assert checked >= 16
    assert empty >= 1, (
        "at least one drawing should have no closed polygon at all; if there "
        "is none, the 'no data' path is never tested"
    )


def test_labels_inside_on_a_bulge_polygon_says_why_rather_than_guessing():
    """G8: the correct answer is nearly always null + a reason, not 0."""
    _mongo_or_skip()
    out = sp.labels_inside(JANADRIYAH, "205CD4A")
    assert out["labels"] == []
    assert out["total"] is None
    assert "bulge" in out["not_measured"]


# ===========================================================================
# UPLIFT-05 — distance
# ===========================================================================

#: The nine education parcels, and the row keys the spec uses for its matrix.
#: The handle is what binds; a key is only a row name so that the matrix can be
#: read by a human.
SCHOOLS = {
    "205D135": "ELC300",
    "205CD54": "ELC600",
    "205D12D": "INT-1",
    "205D12E": "INT-2",
    "205D153": "PRI-1",
    "205D154": "PRI-2",
    "205D12B": "PRV",
    "205D12F": "SEC-1",
    "205D130": "SEC-2",
}

#: The reference matrix, metres, `centroid` mode, measured 23 August 2026 and
#: reproduced on 24 August from the corrected `polygon_centroid`. Written per
#: handle pair, not per key, because the handle is identity and the key is
#: presentation.
REFERENCE_PAIRS = {
    ("205D12D", "205D12F"): 89,
    ("205D12E", "205D130"): 100,
    ("205D12E", "205D154"): 150,
    ("205D154", "205D130"): 188,
    ("205D135", "205D12D"): 346,
    ("205D135", "205D12B"): 365,
    ("205D135", "205D12F"): 404,
    ("205D135", "205D153"): 441,
    ("205CD54", "205D154"): 449,
    ("205CD54", "205D130"): 450,
    ("205D153", "205D12F"): 470,
    ("205D153", "205D12D"): 499,
    ("205CD54", "205D12E"): 521,
    ("205D153", "205D12B"): 576,
    ("205D153", "205D154"): 733,
    ("205D12B", "205D12F"): 755,
    ("205D12E", "205D12F"): 788,
    ("205D154", "205D12F"): 789,
    ("205D153", "205D12E"): 819,
    ("205D12D", "205D12E"): 877,
    ("205D154", "205D12D"): 877,
    ("205D12F", "205D130"): 889,
    ("205D153", "205D130"): 903,
    ("205D12D", "205D130"): 978,
    ("205D135", "205D154"): 1076,
    ("205CD54", "205D153"): 1078,
    ("205D135", "205D12E"): 1117,
    ("205D135", "205D130"): 1214,
    ("205CD54", "205D12F"): 1229,
    ("205D154", "205D12B"): 1305,
    ("205CD54", "205D12D"): 1315,
    ("205D12E", "205D12B"): 1377,
    ("205D130", "205D12B"): 1468,
    ("205D135", "205CD54"): 1476,
    ("205CD54", "205D12B"): 1649,
    ("205D12D", "205D12B"): 707,
}

EAST_CLUSTER = {"205D12D", "205D12F", "205D135", "205D12B", "205D153"}
WEST_CLUSTER = {"205D12E", "205D130", "205D154", "205CD54"}


def _matrix_lookup(r: dict, a: str, b: str) -> float:
    idx = {item["handle"]: i for i, item in enumerate(r["items"])}
    return r["matrix"][idx[a]][idx[b]]


# --- pure -------------------------------------------------------------------


def test_the_row_keys_are_unique_and_run_east_to_west():
    """A matrix with no row names cannot be read by anyone.

    The rule has to be general: layer names are shortened to the shortest
    unique prefix, and a layer that has more than one parcel gets the suffix
    `-1`, `-2` **running from east to west**. A hand-picked abbreviation table
    would be a drawing-specific constant (G1).
    """
    items = [
        {"handle": "A", "layer": "Intermediate School", "point": (100.0, 0.0)},
        {"handle": "B", "layer": "Intermediate School", "point": (10.0, 0.0)},
        {"handle": "C", "layer": "Secondary School", "point": (50.0, 0.0)},
        {"handle": "D", "layer": "Primary School", "point": (50.0, 0.0)},
        {"handle": "E", "layer": "Private School", "point": (50.0, 0.0)},
        {"handle": "F", "layer": "Early Education ELC 300", "point": (50.0, 0.0)},
        {"handle": "G", "layer": "Early Education ELC 600", "point": (50.0, 0.0)},
    ]
    keys = sp._row_keys(items)

    assert len(set(keys.values())) == len(items), "the keys must be unique"
    assert keys["A"] == "INT-1" and keys["B"] == "INT-2", "east comes first"
    assert keys["C"] == "SEC"
    assert keys["F"] == "ELC300" and keys["G"] == "ELC600"
    # `Primary` and `Private` collide at three letters, so both are extended
    # until unique. The spec writes them PRI and PRV, which were hand-picked;
    # see the note in docs/UPLIFT-05-DISTANCE.md.
    assert keys["D"] != keys["E"]
    assert keys["D"].startswith("PRI") and keys["E"].startswith("PRI")


def test_edge_distance_is_zero_when_two_parcels_touch():
    """Mode `edge` answers "are the two parcels adjacent", and two parcels that
    share a boundary line are adjacent — their distance is zero, not nearly
    zero."""
    a = rect(E0, N0, 20.0, 20.0)
    b = rect(E0 + 20.0, N0, 20.0, 20.0)
    assert sp._ring_distance(
        [(p[0], p[1]) for p in a], [(p[0], p[1]) for p in b]
    ) == pytest.approx(0.0, abs=1e-6)


def test_edge_distance_is_shorter_than_centre_distance_for_separated_parcels():
    """Two modes answer two different questions, and the difference must be
    real."""
    a = [(p[0], p[1]) for p in rect(E0, N0, 20.0, 20.0)]
    b = [(p[0], p[1]) for p in rect(E0 + 100.0, N0, 20.0, 20.0)]
    edge = sp._ring_distance(a, b)
    assert edge == pytest.approx(80.0, abs=1e-6)


def test_edge_distance_is_zero_for_a_parcel_inside_another():
    """One polygon entirely inside another has no crossing edge at all. A test
    that only looks at edges would report a positive distance for two shapes
    that overlap."""
    outer = [(p[0], p[1]) for p in rect(E0, N0, 200.0, 200.0)]
    inner = [(p[0], p[1]) for p in rect(E0, N0, 20.0, 20.0)]
    assert sp._ring_distance(outer, inner) == 0.0
    assert sp._ring_distance(inner, outer) == 0.0


def test_single_linkage_joins_through_a_chain_not_only_direct_neighbours():
    """Single-linkage is a stated choice, not an accident: three parcels 500 m
    apart in a row are ONE cluster at a 600 m threshold even though the
    outermost are 1,000 m apart."""
    pts = {"a": (0.0, 0.0), "b": (500.0, 0.0), "c": (1000.0, 0.0)}
    groups = sp._single_linkage(pts, 600.0)
    assert len(groups) == 1 and set(groups[0]) == {"a", "b", "c"}
    assert len(sp._single_linkage(pts, 400.0)) == 3


# --- against a live database ------------------------------------------------


def test_the_nine_school_matrix_reproduces_the_reference():
    """Definition of Done 1 — a 9x9 matrix, tolerance 1 m."""
    _mongo_or_skip()
    r = sp.distance_matrix(
        JANADRIYAH, layout="Model", handles=sorted(SCHOOLS), mode="centroid"
    )
    assert len(r["items"]) == 9
    assert r["pairs_total"] == 36
    assert r["unit"] == "m"

    for (a, b), expected in REFERENCE_PAIRS.items():
        got = _matrix_lookup(r, a, b)
        assert got == pytest.approx(expected, abs=1.0), f"{a}-{b}: {got} != {expected}"
        assert _matrix_lookup(r, b, a) == pytest.approx(got, abs=1e-9)
    for item in r["items"]:
        assert _matrix_lookup(r, item["handle"], item["handle"]) == 0.0


def test_the_distance_comes_from_the_polygon_centroid_not_the_bounding_box():
    """Why UPLIFT-01 renamed `centroid` to `bbox_centre`.

    Computing a distance from the bounding box centre on geometry sitting on a
    rotated grid introduces errors of up to 2.135 m without a single sign in
    the response. What is tested here is not the size of that error but that
    the number returned comes from the polygon centroid — two different
    sources, and only one of them right.
    """
    _mongo_or_skip()
    import math

    from app.mongo import COLL_ENTITIES, coll

    entities = coll(COLL_ENTITIES)
    docs = {
        d["handle"]: d
        for d in entities.find(
            {"drawing_id": JANADRIYAH, "handle": {"$in": ["205D12D", "205D12F"]}},
            {"handle": 1, "polygon_centroid": 1, "bbox_centre": 1},
        )
    }
    from_polygon = math.dist(
        docs["205D12D"]["polygon_centroid"], docs["205D12F"]["polygon_centroid"]
    )
    from_bbox = math.dist(
        docs["205D12D"]["bbox_centre"], docs["205D12F"]["bbox_centre"]
    )
    r = sp.distance_matrix(
        JANADRIYAH, layout="Model", handles=["205D12D", "205D12F"], mode="centroid"
    )
    got = _matrix_lookup(r, "205D12D", "205D12F")

    assert got == pytest.approx(from_polygon, abs=1e-6)
    assert abs(got - from_bbox) > 1e-6, (
        "the two points happen to coincide on this parcel, so this test "
        "distinguishes nothing — pick another parcel"
    )


def test_nearest_finds_the_two_touching_school_pairs():
    """Definition of Done 2."""
    _mongo_or_skip()
    r = sp.distance_matrix(
        JANADRIYAH, layout="Model", handles=sorted(SCHOOLS), mode="centroid"
    )
    nearest = {row["from_handle"]: row for row in r["nearest"]}
    assert nearest["205D12D"]["to_handle"] == "205D12F"
    assert nearest["205D12D"]["distance"]["value"] == pytest.approx(89, abs=1.0)
    assert nearest["205D12E"]["to_handle"] == "205D130"
    assert nearest["205D12E"]["distance"]["value"] == pytest.approx(100, abs=1.0)
    assert nearest["205D12D"]["distance"]["unit"] == "m"


def test_stats_are_min_89_max_1649_mean_791():
    """Definition of Done 3."""
    _mongo_or_skip()
    r = sp.distance_matrix(
        JANADRIYAH, layout="Model", handles=sorted(SCHOOLS), mode="centroid"
    )
    stats = r["stats"]
    assert stats["min"]["value"] == pytest.approx(89, abs=1.0)
    assert stats["max"]["value"] == pytest.approx(1649, abs=1.0)
    assert stats["mean"]["value"] == pytest.approx(791, abs=1.0)
    assert stats["median"]["value"] == pytest.approx(789, abs=1.5)
    for key in ("min", "max", "mean", "median"):
        assert stats[key]["unit"] == "m"
        assert stats[key]["method"]


def test_clustering_at_600_metres_splits_east_from_west():
    """Definition of Done 4 — two groups, with exact members."""
    _mongo_or_skip()
    r = sp.distance_matrix(
        JANADRIYAH,
        layout="Model",
        handles=sorted(SCHOOLS),
        mode="centroid",
        cluster_threshold=600.0,
    )
    clusters = r["clusters"]
    assert clusters["count"] == 2
    members = [set(group) for group in clusters["members"]]
    assert EAST_CLUSTER in members
    assert WEST_CLUSTER in members
    assert "single-linkage" in clusters["method"]
    assert clusters["threshold"]["value"] == 600.0
    assert clusters["threshold"]["unit"] == "m"


def test_the_cluster_threshold_is_a_parameter_not_a_number_in_the_code():
    """G1: 600 m is a planning number of this drawing, not a property of a CAD
    drawing.

    Without a threshold there are no clusters — and that is said, not silently
    dropped.
    """
    _mongo_or_skip()
    r = sp.distance_matrix(
        JANADRIYAH, layout="Model", handles=sorted(SCHOOLS), mode="centroid"
    )
    assert r["clusters"]["count"] is None
    assert r["clusters"]["not_computed"]


def test_edge_mode_measures_shorter_than_centroid_mode_on_the_reference_pair():
    """Definition of Done 5."""
    _mongo_or_skip()
    pair = ["205D12D", "205D12F"]
    centroid = sp.distance_matrix(
        JANADRIYAH, layout="Model", handles=pair, mode="centroid"
    )
    edge = sp.distance_matrix(JANADRIYAH, layout="Model", handles=pair, mode="edge")
    assert _matrix_lookup(edge, *pair) < _matrix_lookup(centroid, *pair)
    assert edge["mode"] == "edge"
    assert "edges" in edge["measure"]


def test_every_distance_response_says_it_is_not_a_walking_distance():
    """Definition of Done 6, without exception.

    Planning standards are written in walking distance; handing over a
    straight-line number without saying it is a straight line is the easiest
    way to make a correct answer be used for the wrong thing.
    """
    _mongo_or_skip()
    calls = [
        sp.distance_matrix(
            JANADRIYAH, layout="Model", handles=sorted(SCHOOLS), mode="centroid"
        ),
        sp.distance_matrix(
            JANADRIYAH, layout="Model", handles=["205D12D", "205D12F"], mode="edge"
        ),
        sp.proximity_count(
            JANADRIYAH,
            layout="Model",
            around_layers=["Primary School"],
            count_layers=["VL2"],
            radius=400.0,
        ),
    ]
    for r in calls:
        assert "walking distance" in r["not_measured"]
        assert "road network" in r["not_measured"]


def test_forty_one_items_is_a_refusal_with_a_suggestion():
    """Definition of Done 7 — not a result truncated in silence."""
    _mongo_or_skip()
    with pytest.raises(sp.SpatialRefused) as caught:
        sp.distance_matrix(
            JANADRIYAH, layout="Model", layers=["VL2"], mode="centroid"
        )
    assert caught.value.code == "TOO_MANY_ITEMS"
    assert "40" in caught.value.message
    assert caught.value.hint


def test_edge_mode_is_capped_harder_than_centroid_mode():
    """The cost of mode `edge` is O(n·m) per edge pair, so its limit is
    stricter, and that stricter limit is stated (G7)."""
    _mongo_or_skip()
    from app.mongo import COLL_ENTITIES, coll

    handles = [
        d["handle"]
        for d in coll(COLL_ENTITIES)
        .find(
            {"drawing_id": JANADRIYAH, "layout": "Model", "layer": "VL2",
             "ring_status": "complete"},
            {"handle": 1},
        )
        .limit(20)
    ]
    assert len(handles) == 20
    ok = sp.distance_matrix(
        JANADRIYAH, layout="Model", handles=handles[:16], mode="edge"
    )
    assert len(ok["items"]) == 16
    assert ok["max_items_effective"] == 16
    with pytest.raises(sp.SpatialRefused) as caught:
        sp.distance_matrix(JANADRIYAH, layout="Model", handles=handles[:17], mode="edge")
    assert caught.value.code == "TOO_MANY_ITEMS"


def test_proximity_count_returns_one_row_per_school_and_agrees_with_a_hand_count():
    """Definition of Done 8 — Harsh's question at minute 16:22.

    The numbers are re-checked by counting independently from
    `polygon_centroid` in Mongo, not by calling `distance_matrix` written by
    the same author: a cross-check against one's own code only proves the code
    is consistent. 2,380 plots around 9 schools is also far above the 40-item
    matrix limit, which is precisely why `proximity_count` was split out — its
    output is O(n), not O(n²).
    """
    _mongo_or_skip()
    import math

    from app.mongo import COLL_ENTITIES, coll

    r = sp.proximity_count(
        JANADRIYAH,
        layout="Model",
        around_layers=EDUCATION_LAYERS,
        count_layers=PLOT_TYPOLOGY_LAYERS,
        radius=400.0,
    )
    assert len(r["rows"]) == 9
    assert r["measure_from"] == "centroid"

    entities = coll(COLL_ENTITIES)
    centres = {
        d["handle"]: d["polygon_centroid"]
        for d in entities.find(
            {
                "drawing_id": JANADRIYAH,
                "layout": "Model",
                "layer": {"$in": EDUCATION_LAYERS},
                "ring_status": "complete",
            },
            {"handle": 1, "polygon_centroid": 1},
        )
    }
    plots = [
        d["polygon_centroid"]
        for d in entities.find(
            {
                "drawing_id": JANADRIYAH,
                "layout": "Model",
                "layer": {"$in": PLOT_TYPOLOGY_LAYERS},
                "ring_status": "complete",
            },
            {"polygon_centroid": 1},
        )
    ]
    assert len(plots) == PLOT_COUNT

    by_hand = {
        h: sum(1 for p in plots if math.dist(c, p) <= 400.0)
        for h, c in centres.items()
    }
    got = {row["handle"]: row["count"] for row in r["rows"]}
    assert got == by_hand
    assert sum(by_hand.values()) > 0, "a 400 m radius should contain something"


def test_proximity_count_says_straight_line_and_which_point_it_measured_from():
    """Definition of Done 9."""
    _mongo_or_skip()
    r = sp.proximity_count(
        JANADRIYAH,
        layout="Model",
        around_layers=["Primary School"],
        count_layers=["VL2"],
        radius=400.0,
    )
    assert "straight-line" in r["measure"]
    assert r["measure_from"] == "centroid"
    assert r["radius"]["unit"] == "m"
    row = r["rows"][0]
    assert row["total_area"]["unit"] == "m2"
    assert row["nearest"]["method"] and row["farthest"]["method"]


def test_a_radius_larger_than_the_drawing_says_so_instead_of_returning_everything():
    """G7: "everything", served as though it meant something, is a silent
    truncation running the other way."""
    _mongo_or_skip()
    r = sp.proximity_count(
        JANADRIYAH,
        layout="Model",
        around_layers=["Primary School"],
        count_layers=["VL2"],
        radius=1_000_000.0,
    )
    assert r["radius_exceeds_extents"] is True
    assert r["extents_note"]
    small = sp.proximity_count(
        JANADRIYAH,
        layout="Model",
        around_layers=["Primary School"],
        count_layers=["VL2"],
        radius=100.0,
    )
    assert small["radius_exceeds_extents"] is False


def test_the_radius_parameter_does_not_carry_its_unit_in_its_name():
    """A SPEC CONTRADICTION, fixed the same way as in UPLIFT-07.

    The spec writes `radius_m: float`. That plants a unit inside a parameter
    name, in a repo where 11 of 18 drawings are in inches and 3 declare no
    units at all — exactly the same violation as `tolerance_m` in UPLIFT-07,
    and fixed the same way: `radius` + the unit echoed from the drawing.

    `radius_m` is not accepted as an alias. An alias that re-plants the unit
    just pulled out only moves the defect somewhere harder to see.
    """
    _mongo_or_skip()
    import inspect

    params = inspect.signature(sp.proximity_count).parameters
    assert "radius" in params
    assert "radius_m" not in params
    r = sp.proximity_count(
        JANADRIYAH,
        layout="Model",
        around_layers=["Primary School"],
        count_layers=["VL2"],
        radius=400.0,
    )
    assert r["radius"]["value"] == 400.0
    assert r["radius"]["unit"] == "m"


def test_distance_needs_a_layout_and_something_to_measure():
    _mongo_or_skip()
    with pytest.raises(sp.SpatialRefused) as caught:
        sp.distance_matrix(JANADRIYAH, layout="", handles=sorted(SCHOOLS))
    assert caught.value.code == "LAYOUT_REQUIRED"

    with pytest.raises(sp.SpatialRefused) as caught:
        sp.distance_matrix(JANADRIYAH, layout="Model")
    assert caught.value.code == "NOTHING_TO_MEASURE"


def test_distance_never_claims_a_unit_a_drawing_does_not_declare():
    """G2 + G5, over all the drawings.

    11 of the 18 drawings are in inches and 3 declare no units at all. A
    distance that comes out with the unit "m" because the reference drawing
    happens to use metres is a wrong number wearing the clothes of a right one.
    """
    _mongo_or_skip()
    from app.mongo import COLL_DRAWINGS, COLL_ENTITIES, coll

    entities = coll(COLL_ENTITIES)
    checked = 0
    for drawing in coll(COLL_DRAWINGS).find({}, {"_id": 1, "units_name": 1,
                                                 "units_code": 1}):
        did = drawing["_id"]
        groups = list(
            entities.aggregate(
                [
                    {"$match": {"drawing_id": did, "ring_status": "complete"}},
                    {
                        "$group": {
                            "_id": {"layout": "$layout", "layer": "$layer"},
                            "n": {"$sum": 1},
                        }
                    },
                    {"$match": {"n": {"$gte": 2, "$lte": 30}}},
                    {"$limit": 1},
                ]
            )
        )
        if not groups:
            continue
        layout = groups[0]["_id"]["layout"]
        layer = groups[0]["_id"]["layer"]
        r = sp.distance_matrix(did, layout=layout, layers=[layer], mode="centroid")
        checked += 1

        declared = bool(drawing.get("units_code")) and drawing.get(
            "units_name"
        ) not in (None, "unitless")
        is_model = layout == "Model"
        if declared and is_model:
            assert r["unit"] == drawing["units_name"]
        else:
            assert r["unit"] is None, f"{did}/{layout}: a unit claimed with no basis"
            assert r["unit_reason"], "a unit that is absent must come with a reason"
        assert "walking distance" in r["not_measured"]

    assert checked >= 3, "this invariant needs more than one drawing to mean anything"
