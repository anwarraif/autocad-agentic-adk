"""GEO-H3 G2 — the cells themselves.

The binding numbers are in `docs/GEO-H3-PLAN.md` and are asserted here as
written. Two of them changed what the code does rather than what the test
expects, and both are worth knowing before reading the assertions:

**Only model space gets cells.** The first run indexed everything and put
4,058 entities in H3 base cells 61 and 75 — the Gulf of Guinea and the South
Atlantic — because a sheet's coordinates are page geometry and a block
definition's are the block's own frame. Neither is a position on the earth.
The DoD "no cell outside the base-cell region" is what caught it.

**A point outside the drawing is not in the drawing.** Five model-space block
references are anchored at (0, 0); two of them had a bounding box squarely in
Janadriyah while their anchor claimed the Atlantic. Candidate points are now
checked against the drawing's own extents, which recovered those two and
refused the other three by name.

One DoD item is asserted differently from the way it is written, and it is
called out rather than quietly relaxed — see
`test_every_road_is_a_chain_except_the_ones_shorter_than_a_cell`.
"""

from __future__ import annotations

import pytest

h3 = pytest.importorskip("h3")

from app import geo_backfill, geo_h3, landuse, store
from conftest import unconfigured_drawing_ids
from app.mongo import COLL_DRAWINGS, COLL_ENTITIES, coll

JANADRIYAH = "596212db022a3397"
MODEL = "Model"

#: docs/GEO-H3-PLAN.md, and already pinned in `test_geo_h3.py`. Repeated here
#: as the value this phase must PRODUCE rather than compute.
MOSQUE = "205D0EB"
MOSQUE_CELL_RES_13 = "8d53736546c3a7f"

#: Every golden cell is in this base cell. A cell outside it is not a rounding
#: error; it is another continent.
BASE_CELL = 41

#: The layer the plan's ground truth calls the road centrelines. A drawing
#: constant, and it lives here in a test rather than in app code (G1) — the
#: assignment code never reads a layer name.
ROAD_LAYER = "00_Prop - Road - CL_"

#: Verified ground truth from the plan: 3,221 entities carry a full ring.
#: 2,885 of them are in model space; the rest are inside block definitions,
#: which get no cells at all.
MODEL_RINGS = 2885


def _mongo_or_skip():
    try:
        drawing = store.get_drawing(JANADRIYAH)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip("the reference drawing has not been ingested")
    return drawing


def _indexed_or_skip():
    _mongo_or_skip()
    if not coll(COLL_ENTITIES).find_one(
        {"drawing_id": JANADRIYAH, "h3_cell": {"$ne": None}}, {"_id": 1}
    ):
        pytest.skip(
            "this drawing has no cells yet; run scripts/backfill_h3.py --drawing "
            + JANADRIYAH
        )


# ---------------------------------------------------------------------------
# The binding numbers
# ---------------------------------------------------------------------------


def test_the_mosque_lands_in_its_golden_cell():
    """DoD, and the one number the whole phase is measured by. It is produced
    here by the store — the point comes out of Mongo, through the configured
    CRS, into a cell — rather than from a lat/long typed into a test."""
    _indexed_or_skip()
    row = coll(COLL_ENTITIES).find_one({"_id": f"{JANADRIYAH}:{MOSQUE}"}, {"h3_cell": 1})
    assert row["h3_cell"] == MOSQUE_CELL_RES_13


def test_every_parcel_has_at_least_one_cell():
    """DoD. The guarantee that makes the index usable: `polygon_to_cells`
    decides containment by cell centre and returns NOTHING for a 300 m2 parcel
    at coarse resolutions, so a parcel that occupies no cells would vanish
    from every count built on this — silently, and only the small ones."""
    _indexed_or_skip()
    total = coll(COLL_ENTITIES).count_documents(
        {"drawing_id": JANADRIYAH, "layout": MODEL, "ring_status": "complete"}
    )
    assert total == MODEL_RINGS
    empty = coll(COLL_ENTITIES).count_documents(
        {
            "drawing_id": JANADRIYAH,
            "layout": MODEL,
            "ring_status": "complete",
            "h3_cells": {"$size": 0},
        }
    )
    assert empty == 0


def test_no_cell_lands_outside_the_region_of_the_golden_set():
    """DoD, and the test that found the bug this phase's design turns on.

    Every cell, not every primary cell: the failure it catches was in the
    coverings, and a check over one cell per entity would have missed most of
    it."""
    _indexed_or_skip()
    outside: dict[int, int] = {}
    total = 0
    for row in coll(COLL_ENTITIES).find({"drawing_id": JANADRIYAH}, {"h3_cells": 1}):
        for cell in row.get("h3_cells") or []:
            total += 1
            base = h3.get_base_cell_number(cell)
            if base != BASE_CELL:
                outside[base] = outside.get(base, 0) + 1
    assert total > 0
    assert outside == {}, outside


def test_counts_at_a_coarser_resolution_are_the_sum_of_their_children():
    """DoD. True by construction — coarse cells are derived with
    `cell_to_parent` and never stored — and asserted because the alternative
    construction is the tempting one: re-indexing each point at resolution 9
    would disagree for every object near a boundary, by a handful of objects,
    with nothing in the output to explain the difference."""
    _indexed_or_skip()
    from collections import Counter

    fine: Counter[str] = Counter()
    for row in coll(COLL_ENTITIES).find(
        {"drawing_id": JANADRIYAH, "layout": MODEL, "ring_status": "complete"},
        {"h3_cell": 1},
    ):
        if row.get("h3_cell"):
            fine[row["h3_cell"]] += 1

    coarse: Counter[str] = Counter()
    for cell, count in fine.items():
        coarse[h3.cell_to_parent(cell, 9)] += count

    assert sum(coarse.values()) == sum(fine.values()) == MODEL_RINGS
    assert len(coarse) < len(fine)


def test_every_road_is_a_chain_except_the_ones_shorter_than_a_cell():
    """DoD, asserted as the data is rather than as the plan guessed it.

    The plan says every road has at least two cells forming a neighbour chain.
    Measured: 1,194 of 1,233 model-space centrelines do. The other 39 come out
    in one cell, and for 36 of them that is the correct answer — they are arc
    fillets at junctions with a mean length of 2.4 m, some of them under a
    millimetre, and a cell is 8.18 m across. An object smaller than a cell is
    in one cell.

    The remaining three are CIRCLEs 22 m around whose stored geometry is a
    centre point, and they are not quietly accepted: each carries an
    `h3_coverage_note` saying its extent is not indexed. Asserting "at least
    two cells" for all 1,233 would have meant either inflating the tiny ones
    or lowering the resolution until the drawing stopped being resolvable."""
    _indexed_or_skip()
    across = 2 * h3.average_hexagon_edge_length(13, unit="m")
    chains = 0
    single_small = 0
    single_large_flagged = 0
    single_large_unflagged = []

    for row in coll(COLL_ENTITIES).find(
        {"drawing_id": JANADRIYAH, "layout": MODEL, "layer": ROAD_LAYER},
        {"h3_cells": 1, "length": 1, "handle": 1, "h3_coverage_note": 1},
    ):
        cells = row.get("h3_cells") or []
        if len(cells) >= 2:
            chains += 1
            continue
        length = row.get("length") or 0
        if length <= across:
            single_small += 1
        elif row.get("h3_coverage_note"):
            single_large_flagged += 1
        else:
            single_large_unflagged.append(row.get("handle"))

    assert chains >= 1000
    assert single_small > 0
    # The one thing that must never happen: an object bigger than a cell,
    # indexed as a single cell, with nothing saying so.
    assert single_large_unflagged == [], single_large_unflagged


def test_a_road_chain_is_contiguous():
    """What "chain" has to mean. Built with `grid_path_cells` rather than by
    sampling every N metres: at resolution 13 a cell is 4.09 m across, so any
    step fixed in advance either skips cells or is a guess about the drawing's
    units."""
    _indexed_or_skip()
    checked = 0
    for row in coll(COLL_ENTITIES).find(
        {"drawing_id": JANADRIYAH, "layout": MODEL, "layer": ROAD_LAYER},
        {"h3_cells": 1, "handle": 1},
    ).limit(200):
        cells = row.get("h3_cells") or []
        if len(cells) < 2:
            continue
        for i in range(min(len(cells) - 1, 300)):
            assert h3.grid_distance(cells[i], cells[i + 1]) == 1, row.get("handle")
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# What must NOT get a cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("space,layout", [("paper", "DMP Layout1"), ("block", None)])
def test_coordinates_that_are_not_positions_get_no_cell(space, layout):
    """The bug the base-cell check caught, kept caught.

    A sheet border is 1,189 units long because an A0 page is 1,189 mm wide,
    not because anything in the world is; a block definition's geometry is
    placed somewhere different by every INSERT of it. Both produce perfectly
    well-formed positions in the sea."""
    _indexed_or_skip()
    query = {"drawing_id": JANADRIYAH}
    if layout:
        query["layout"] = layout
    else:
        query["layout"] = {"$regex": r"^\[block\] "}

    rows = list(
        coll(COLL_ENTITIES).find(query, {"h3_cells": 1, "h3_cell": 1, "h3_note": 1}).limit(50)
    )
    assert rows, space
    for row in rows:
        assert row.get("h3_cells") == [], row
        assert row.get("h3_cell") is None, row
        assert space in (row.get("h3_note") or ""), row


def test_a_point_outside_the_drawing_is_refused_by_name():
    """The other half of the same lesson. A block reference parked at (0, 0)
    is an everyday CAD artefact and (0, 0) through a UTM inverse is the Gulf
    of Guinea. Excluded with a reason, never placed."""
    _indexed_or_skip()
    rows = list(
        coll(COLL_ENTITIES).find(
            {
                "drawing_id": JANADRIYAH,
                "layout": MODEL,
                "h3_note": {"$regex": "outside the drawing's own extents"},
            },
            {"handle": 1, "h3_cells": 1},
        )
    )
    assert rows
    for row in rows:
        assert row.get("h3_cells") == []


def test_the_extents_check_is_a_general_rule_and_degrades_to_nothing():
    """Pure. A drawing whose extents could not be computed — there is one in
    the store — is not forbidden from having cells (G3): what is missing is
    the check, not the coordinates."""
    extents = {"min": [0.0, 0.0], "max": [100.0, 100.0]}
    assert geo_h3._inside_extents([50.0, 50.0], extents)
    assert geo_h3._inside_extents([0.0, 0.0], extents)
    assert not geo_h3._inside_extents([-5000.0, 0.0], extents)
    assert geo_h3._inside_extents([-5000.0, 0.0], None)
    assert geo_h3._inside_extents([-5000.0, 0.0], {})


# ---------------------------------------------------------------------------
# Resolution is configuration (BINDING 2)
# ---------------------------------------------------------------------------


def test_a_drawing_that_says_nothing_gets_the_default_resolution():
    assert landuse._build_h3_resolution(None, where="test") == 13
    assert landuse._build_h3_resolution({}, where="test") == 13
    assert landuse.DEFAULT_H3_RESOLUTION == 13


def test_a_drawing_may_choose_its_own_resolution():
    assert landuse._build_h3_resolution({"resolution": 9}, where="test") == 9


@pytest.mark.parametrize("bad", [16, -1, 99, "thirteen"])
def test_a_resolution_that_is_not_one_is_refused_rather_than_defaulted(bad):
    """Quietly substituting the default for a resolution somebody typed on
    purpose would index at a level nobody asked for, and look like working."""
    with pytest.raises(landuse.ConfigError):
        landuse._build_h3_resolution({"resolution": bad}, where="test")


def test_the_stored_cells_say_which_resolution_they_are():
    """A cell index without its resolution cannot be read, and a config change
    has to leave the old rows identifiable as stale rather than merely
    different."""
    _indexed_or_skip()
    row = coll(COLL_ENTITIES).find_one({"_id": f"{JANADRIYAH}:{MOSQUE}"}, {"h3_res": 1})
    config = landuse.for_drawing(JANADRIYAH)
    assert row["h3_res"] == config.h3_resolution


# ---------------------------------------------------------------------------
# The backfill itself (BINDING 3, 5, 6)
# ---------------------------------------------------------------------------


def test_a_drawing_with_no_coordinate_system_refuses_loudly():
    """BINDING 3. Not an exception, and not a run that wrote nothing and
    returned success — `ok: false` with a sentence."""
    _mongo_or_skip()
    # Drawings with NO per-drawing config, derived from the config tree: a
    # drawing that ships one is georeferenced on purpose and is not evidence
    # of a leak. See tests/conftest.py.
    others = unconfigured_drawing_ids()
    if not others:
        pytest.skip("every ingested drawing has a config")
    for drawing_id in others:
        report = geo_backfill.assign_cells(drawing_id, dry_run=True)
        assert report["ok"] is False, drawing_id
        assert report["reason"], drawing_id
        assert report["assigned"] == 0, drawing_id


def test_no_other_drawing_has_cells():
    """G10, as a count. Cells on a drawing nobody georeferenced would mean a
    coordinate system had leaked from one drawing to another."""
    _indexed_or_skip()
    for drawing_id in unconfigured_drawing_ids():
        assert (
            coll(COLL_ENTITIES).count_documents(
                {"drawing_id": drawing_id, "h3_cell": {"$ne": None}}
            )
            == 0
        ), drawing_id


def test_every_entity_without_a_cell_is_named_and_counted():
    """BINDING 5. A backfill that indexes 40,000 of 46,754 looks exactly like
    one that indexed everything, unless it says what it left out."""
    _mongo_or_skip()
    report = geo_backfill.assign_cells(JANADRIYAH, dry_run=True)
    assert report["ok"]
    named = sum(
        len(block["handles"]) for block in report["excluded_by_reason"].values()
    )
    assert named == report["excluded"]
    assert report["assigned"] + report["excluded"] == report["entities"]
    for reason, block in report["excluded_by_reason"].items():
        assert reason.strip()
        assert block["count"] == len(block["handles"])
    # And the iterator hands over every one of them, not a sample.
    assert len(list(geo_backfill.excluded_handles(report))) == report["excluded"]


def test_the_report_names_what_it_could_not_cover_whole():
    """Not an exclusion — these rows DO answer vicinity questions, and answer
    them short, which is the harder failure to notice of the two."""
    _mongo_or_skip()
    report = geo_backfill.assign_cells(JANADRIYAH, dry_run=True)
    for row in report["partial_coverage"]:
        assert row["handle"]
        assert row["note"]
    for row in report["truncated"]:
        assert row["total"] > row["kept"]


def test_the_printed_report_says_how_much_it_is_not_printing():
    """G7. A list that silently stops at twenty reads exactly like a list of
    twenty."""
    _mongo_or_skip()
    report = geo_backfill.assign_cells(JANADRIYAH, dry_run=True)
    text = geo_backfill.format_report(report, sample=3)
    assert "more" in text
    assert f"{report['excluded']:,}" in text


def test_a_dry_run_writes_nothing():
    """The safety the CLI's `--dry-run` promises, asserted rather than
    assumed."""
    _indexed_or_skip()
    before = coll(COLL_ENTITIES).count_documents(
        {"drawing_id": JANADRIYAH, "h3_cell": {"$ne": None}}
    )
    geo_backfill.assign_cells(JANADRIYAH, dry_run=True)
    after = coll(COLL_ENTITIES).count_documents(
        {"drawing_id": JANADRIYAH, "h3_cell": {"$ne": None}}
    )
    assert before == after


def test_the_backfill_is_batched_rather_than_held_in_memory():
    """BINDING 6. A file ten times the size has to be a longer run, not a
    redesign, so the batch is a knob and the default is stated."""
    assert geo_backfill.BATCH == 1_000
    report = geo_backfill.assign_cells(JANADRIYAH, dry_run=True, batch=50)
    assert report["ok"] or report["reason"]


# ---------------------------------------------------------------------------
# The assignment, without a database
# ---------------------------------------------------------------------------


@pytest.fixture
def utm38n():
    return landuse._build_crs(
        {
            "epsg": 32638,
            "name": "WGS 84 / UTM zone 38N",
            "declared_in_file": False,
            "sources": [{"origin": "geometry", "detail": "where the extents sit"}],
        },
        where="test",
    )


def test_a_point_entity_gets_exactly_one_cell(utm38n):
    row = {
        "layout": MODEL,
        "anchor_point": [691332.06, 2749824.77],
    }
    got = geo_h3.cells_for_entity(utm38n, row, 13)
    assert got.strategy == "point"
    assert len(got.cells) == 1
    assert got.primary == got.cells[0]


def test_an_entity_with_no_geometry_says_so_rather_than_returning_empty(utm38n):
    """G8. `None` plus a reason, never a silent nothing."""
    got = geo_h3.cells_for_entity(utm38n, {"layout": MODEL}, 13)
    assert got.cells == ()
    assert got.note


def test_a_ring_is_covered_by_overlap_not_by_centre(utm38n):
    """The 300 m2 parcel that comes back with zero cells under the default
    containment rule. Asserted at resolution 11, where the two rules give
    visibly different answers."""
    ring = [
        [691332.0, 2749824.0],
        [691342.0, 2749824.0],
        [691342.0, 2749834.0],
        [691332.0, 2749834.0],
    ]
    row = {"layout": MODEL, "ring_status": "complete", "ring": ring,
           "polygon_centroid": [691337.0, 2749829.0]}
    got = geo_h3.cells_for_entity(utm38n, row, 11)
    assert got.strategy == "ring"
    assert len(got.cells) >= 1
    # The representative cell of an area is inside it, not a corner of it.
    assert got.primary == h3.latlng_to_cell(
        *utm38n.to_lat_lon(691337.0, 2749829.0), 11
    )


def test_the_storage_fields_carry_the_reason_when_there_are_no_cells(utm38n):
    """A row with `h3_cells: []` and a note was considered and could not be
    placed. A row with no h3 fields at all has not been through the backfill.
    Telling those apart is the whole of knowing whether an index is
    complete."""
    got = geo_h3.cells_for_entity(utm38n, {"layout": MODEL}, 13)
    fields = geo_h3.fields_for_storage(got, 13)
    assert fields["h3_cells"] == []
    assert fields["h3_cell"] is None
    assert fields["h3_res"] == 13
    assert fields["h3_note"]
