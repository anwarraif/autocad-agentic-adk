"""Placing an entity by where its coordinates ARE, not by its layout's name.

Sedra forced this. It ingested perfectly -- 990,790 entities, 705 layers, zero
audit errors -- and every map view was empty, because whether an entity had a
place on the earth was decided from a string:

    space = _space_of(row["layout"])      # "[block] ..." -> "block"
    if space != "model":  refuse

That premise is a claim about COORDINATES and it was being read off a NAME.
For a title block it is true: its entities really do sit at [1009.85, 41.96],
metres from their own origin. For a BOUND XREF it is false: the xref keeps the
coordinates it was drawn in and the INSERT that places it is the identity, so
its entities sit at [677347.398, 2749578.042] -- inside the drawing's own
declared extents, which is the definition of being somewhere in this drawing.
934,013 of Sedra's 990,291 block-resident entities are of the second kind.

The rule is now measured with the same `_inside_extents` this module already
used to reject a block reference parked at the origin, so one test decides
both directions. It needs no per-drawing configuration and no knowledge of how
any particular file was built, which is what makes it general: Janadriyah's
door-symbol blocks fail it and stay refused, Sedra's xref passes it.
"""

from __future__ import annotations

import pytest

from app import geo_h3, landuse


MODEL = "Model"
BLOCK = "[block] XR_BUILDING FOOTPRINT-SED-2A"
SHEET = "RRE-RY-RY1-A2A-MP014-SAP-CH-LSM-A0-DWG-00101"

#: Sedra's declared extents, from the file's own header.
EXTENTS = {"min": [666597.6, 2743400.2], "max": [684006.3, 2755046.9]}

#: A real stored point from `[block] XR_BUILDING FOOTPRINT-SED-2A`.
INSIDE = [677347.398, 2749578.042]

#: A real stored point from `[block] ROSHN TITLE BLOCK`, which is page-local.
OUTSIDE = [1009.85, 41.96]

#: What `extract.block_placements` records for Sedra's bound xrefs: model
#: space places them through a chain of identity INSERTs, measured
#: `insert=(0,0,0)` with no scale and no rotation on all 19 of them.
IDENTITY_CHAIN = {"XR_BUILDING FOOTPRINT-SED-2A": {
    "identity": True, "ambiguous": False, "placements": 1, "depth": 1}}

#: A block that model space moves. Its stored numbers are NOT world numbers.
MOVED_CHAIN = {"XR_BUILDING FOOTPRINT-SED-2A": {
    "identity": False, "ambiguous": False, "placements": 1, "depth": 1}}

#: The same definition placed identically once and moved elsewhere. One
#: stored row cannot be in both places.
AMBIGUOUS_CHAIN = {"XR_BUILDING FOOTPRINT-SED-2A": {
    "identity": True, "ambiguous": True, "placements": 2, "depth": 1}}


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


# --- the scope rule, which two readers share -------------------------------


def test_model_scope_also_admits_entities_measured_into_the_world():
    """Otherwise a fully indexed drawing answers with an empty map."""
    match = geo_h3.scope_match(MODEL)
    assert match == {"$or": [{"layout": MODEL}, {"h3_world_placed": True}]}


def test_asking_for_one_sheet_means_that_sheet():
    """Only model scope is widened; a sheet request is not a whole-drawing one."""
    assert geo_h3.scope_match(SHEET) == {"layout": SHEET}
    assert geo_h3.scope_match(BLOCK) == {"layout": BLOCK}


# --- the placement decision ------------------------------------------------


def test_a_block_placed_by_an_identity_chain_is_in_the_world(utm38n):
    """The Sedra case, and the whole point of this change.

    The decision is the INSERT chain: model space places this definition
    through identity transforms, so its stored coordinates need no moving.
    """
    row = {"layout": BLOCK, "anchor_point": INSIDE}

    got = geo_h3.cells_for_entity(utm38n, row, 13, EXTENTS, IDENTITY_CHAIN)

    assert got.cells, "an identity chain means these coordinates are world ones"
    assert got.world_placed is True
    assert got.note is None


def test_a_block_the_drawing_MOVES_is_refused_however_its_numbers_look(utm38n):
    """The causal test, and the one an extents rule could never make.

    These coordinates sit inside the drawing's extents, so an extents-only
    rule would place them. The chain says the block is moved when it is
    inserted, so the stored numbers are NOT where the geometry ends up.
    """
    row = {"layout": BLOCK, "anchor_point": INSIDE}

    got = geo_h3.cells_for_entity(utm38n, row, 13, EXTENTS, MOVED_CHAIN)

    assert got.cells == ()
    assert got.world_placed is False


def test_a_block_placed_two_different_ways_is_refused(utm38n):
    """One stored row cannot be in two places; picking one would be a guess
    wearing the clothes of a measurement."""
    row = {"layout": BLOCK, "anchor_point": INSIDE}

    got = geo_h3.cells_for_entity(utm38n, row, 13, EXTENTS, AMBIGUOUS_CHAIN)

    assert got.cells == ()
    assert got.world_placed is False


def test_extents_still_corroborate_and_can_veto(utm38n):
    """An identity chain whose points land outside the drawing is a
    contradiction, and the safe reading of a contradiction is to refuse."""
    row = {"layout": BLOCK, "anchor_point": OUTSIDE}

    got = geo_h3.cells_for_entity(utm38n, row, 13, EXTENTS, IDENTITY_CHAIN)

    assert got.cells == ()
    assert got.world_placed is False


def test_a_block_with_no_recorded_placement_is_refused(utm38n):
    """A drawing ingested before INGEST_VERSION 7 carries no chain.

    There is then no measurement to rely on, and the answer is no. A
    re-ingest supplies it; guessing would not.
    """
    row = {"layout": BLOCK, "anchor_point": OUTSIDE}

    got = geo_h3.cells_for_entity(utm38n, row, 13, EXTENTS)

    assert got.cells == ()
    assert "block" in got.note
    assert got.world_placed is False


def test_paper_space_is_not_relaxed_even_when_the_numbers_would_fit(utm38n):
    """A border is 1,189 units long because the sheet is 1,189 mm wide.

    That is page geometry whatever the drawing's extents happen to be, so
    sheets keep the old refusal and only block space is measured.
    """
    row = {"layout": SHEET, "anchor_point": INSIDE}

    got = geo_h3.cells_for_entity(utm38n, row, 13, EXTENTS, IDENTITY_CHAIN)

    assert got.cells == ()
    assert "paper" in got.note


def test_an_ordinary_modelspace_entity_is_not_marked_as_world_placed(utm38n):
    """The flag distinguishes a measured placement from an ordinary one."""
    row = {"layout": MODEL, "anchor_point": INSIDE}

    got = geo_h3.cells_for_entity(utm38n, row, 13, EXTENTS, IDENTITY_CHAIN)

    assert got.cells
    assert got.world_placed is False


def test_without_extents_the_chain_still_decides(utm38n):
    """Extents corroborate; they are not the measurement.

    A drawing whose extents could not be computed still has an INSERT chain,
    and `_inside_extents` answers True when it has nothing to compare
    against, so the placement stands on the chain alone. That is deliberate
    and recorded here so a future change to it is a visible decision.
    """
    row = {"layout": BLOCK, "anchor_point": INSIDE}

    got = geo_h3.cells_for_entity(utm38n, row, 13, None, IDENTITY_CHAIN)

    assert got.world_placed is True


def test_the_flag_is_persisted_so_the_map_can_find_the_row():
    """These rows do not carry the `Model` layout, so without a stored flag
    the map query could never select them."""
    placed = geo_h3.CellAssignment(cells=("8d5",), primary="8d5", world_placed=True)
    ordinary = geo_h3.CellAssignment(cells=("8d5",), primary="8d5")

    assert geo_h3.fields_for_storage(placed, 13)["h3_world_placed"] is True
    # None rather than False: an absent key costs nothing on 46,754 rows that
    # will never be world-placed, and matches how the other optional h3
    # fields are written.
    assert geo_h3.fields_for_storage(ordinary, 13)["h3_world_placed"] is None


def test_the_backfill_reads_the_placements_it_then_uses(monkeypatch):
    """A projection that omitted the field the next line read.

    This shipped: the drawing was fetched with `{"extents": 1}` and
    `block_placement` was read off the result, so the placements were always
    empty and every block-resident entity was refused. Nothing raised and
    nothing was logged; a 990,790 entity drawing came out with 19 cells,
    which looks exactly like a drawing that has little in it.
    """
    from app import geo_backfill

    seen: dict = {}

    class FakeDrawings:
        def find_one(self, _query, projection=None):
            seen["projection"] = projection or {}
            return {"extents": {"min": [0, 0], "max": [1, 1]}, "block_placement": {"B": {"identity": True}}}

    monkeypatch.setattr(geo_backfill, "coll", lambda _name: FakeDrawings())

    extents, placements = geo_backfill.drawing_facts("d")

    assert "block_placement" in seen["projection"], (
        "fetched without the field the placement rule needs"
    )
    assert placements == {"B": {"identity": True}}
    assert extents == {"min": [0, 0], "max": [1, 1]}


def test_a_drawing_ingested_before_placements_were_recorded_gets_an_empty_map(
    monkeypatch,
):
    """No measurement is not the same as a refusal; the rule reads it as
    "nothing recorded" and a re-ingest supplies it."""
    from app import geo_backfill

    class FakeDrawings:
        def find_one(self, _query, projection=None):
            return {"extents": None}

    monkeypatch.setattr(geo_backfill, "coll", lambda _name: FakeDrawings())
    extents, placements = geo_backfill.drawing_facts("old")
    assert extents is None
    assert placements == {}
