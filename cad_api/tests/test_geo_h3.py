"""The H3 constants this campaign is measured against.

Everything here is a fact about the library and about six points on the
ground — no database, no drawing, no route. That is deliberate: these are the
numbers every later phase asserts against, so they must be checkable before
any of the code that produces them exists, and on a machine with no Mongo.

The constants are copied from `docs/GEO-H3-PLAN.md`, which computed them with
h3 4.5.0 and confirmed on h3geo.org that they land on Janadriyah. They are
binding. If one of these tests fails the answer is never to recompute the
constant — a cell number is a claim about where something is, and a test that
edits the claim to match the code has stopped testing anything.
"""

import pytest

from app import geo_h3

h3 = pytest.importorskip("h3")


#: The version the golden cells were computed with. A cell index is a function
#: of the implementation that produced it, so a different version is a
#: different answer and the string comparisons below would be measuring the
#: upgrade rather than our code.
H3_VERSION = "4.5.0"

#: Proven live in the container, and the same two points the UTM inverse is
#: cross-checked against in `test_landuse.py` (`GEOREF`).
MOSQUE = (24.851418547, 46.893576218)   # handle 205D0EB, Jumaa Mosque
SCHOOL = (24.85975, 46.89974)           # handle 205D153, Primary School

#: docs/GEO-H3-PLAN.md, "GOLDEN H3 CELLS". Hex strings, never integers.
MOSQUE_CELLS = {
    5: "85537367fffffff",
    7: "875373650ffffff",
    9: "8953736546fffff",
    11: "8b53736546c3fff",
    13: "8d53736546c3a7f",
}
SCHOOL_CELL_RES_11 = "8b5373654222fff"

#: Every golden cell shares this base cell, because they are all the same
#: patch of Riyadh. Later phases use it as the cheap "is this cell even in the
#: right part of the world" check on cells nobody has pasted into h3geo.org.
#: Read off the library rather than assumed, and it is not a constant from the
#: plan: the plan pins cells, and this is a property of those cells.
BASE_CELL = 41


def test_the_pinned_version_is_the_one_the_constants_were_computed_with():
    assert h3.__version__ == H3_VERSION


@pytest.mark.parametrize("res,cell", sorted(MOSQUE_CELLS.items()))
def test_the_mosque_lands_in_its_golden_cell_at_every_resolution(res, cell):
    assert h3.latlng_to_cell(MOSQUE[0], MOSQUE[1], res) == cell


def test_the_school_lands_in_its_golden_cell():
    assert h3.latlng_to_cell(SCHOOL[0], SCHOOL[1], 11) == SCHOOL_CELL_RES_11


def test_every_golden_cell_sits_in_the_base_cell_the_others_do():
    """The cheap sanity check the later phases reuse. It does not prove a cell
    is in Janadriyah, but it does catch a cell that is on another continent,
    which is what a swapped lat/lon or a wrong resolution produces."""
    for cell in [*MOSQUE_CELLS.values(), SCHOOL_CELL_RES_11]:
        assert h3.get_base_cell_number(cell) == BASE_CELL, cell


@pytest.mark.parametrize("res,cell", sorted(MOSQUE_CELLS.items()))
def test_each_golden_cell_reports_the_resolution_it_is_filed_under(res, cell):
    assert h3.get_resolution(cell) == res


#: The resolutions at which the mosque's res-13 cell has the golden cell as its
#: index ancestor. Res 7 is missing from this list ON PURPOSE — see below.
NESTED_RESOLUTIONS = (5, 9, 11)


@pytest.mark.parametrize("res", NESTED_RESOLUTIONS)
def test_the_index_ancestor_agrees_with_the_point_at_most_resolutions(res):
    assert h3.cell_to_parent(MOSQUE_CELLS[13], res) == MOSQUE_CELLS[res]


def test_h3_is_not_perfectly_nested_and_res_7_is_where_it_shows():
    """The finding that decides how every later phase aggregates.

    Hexagons cannot tile a hierarchy exactly, so "the cell this point is in at
    res 7" and "the res-7 ancestor of the cell this point is in at res 13" are
    two different questions with two different answers. For the mosque they
    differ: it sits on a res-7 boundary, and the two candidates are immediate
    neighbours — grid distance 1, not an error and not a bug.

    Both answers below are correct. The rule this pins is that a resolution
    must be reached by ONE of the two routes and never by both: counts derived
    with `cell_to_parent` and counts derived by re-indexing the point would
    disagree for every object near a boundary, and disagree silently. The plan
    chose `cell_to_parent`, and G2 stores res 13 only.

    If this test ever starts failing, H3 has changed something fundamental —
    that is a reason to read the changelog, not to edit this file."""
    from_point = h3.latlng_to_cell(MOSQUE[0], MOSQUE[1], 7)
    from_ancestry = h3.cell_to_parent(MOSQUE_CELLS[13], 7)
    assert from_point == MOSQUE_CELLS[7]
    assert from_ancestry != from_point
    assert h3.are_neighbor_cells(from_point, from_ancestry)
    assert h3.grid_distance(from_point, from_ancestry) == 1


def test_every_golden_cell_is_a_valid_index_written_as_a_string():
    for cell in [*MOSQUE_CELLS.values(), SCHOOL_CELL_RES_11]:
        assert isinstance(cell, str)
        assert h3.is_valid_cell(cell)


def test_the_snake_case_names_the_plan_relies_on_all_exist():
    """The Python binding renamed everything in v4. Every name below is used
    by a later phase, and finding out at runtime that one of them is spelled
    differently costs a debugging session for a typo."""
    for name in (
        "latlng_to_cell",
        "cell_to_latlng",
        "polygon_to_cells",
        "h3shape_to_cells",
        "grid_disk",
        "grid_ring",
        "cell_to_parent",
        "cell_to_children",
        "compact_cells",
        "cell_area",
        "is_valid_cell",
        "are_neighbor_cells",
    ):
        assert hasattr(h3, name), name


#: How G2 asks for overlap rather than centre containment. A plain string
#: keyword, not the `ContainmentMode` enum the C library documents — the
#: Python binding at 4.5.0 does not export one, and code written against that
#: enum fails at import with no hint about which name to use instead.
OVERLAP_MODE = "overlap"


def test_the_overlap_mode_g2_depends_on_is_reachable_and_spelled_this_way():
    """`polygon_to_cells` decides containment by cell centre, so a 300 m2
    parcel can come back with no cells at all. G2 needs the overlapping mode;
    this pins both that it exists and what it is called."""
    import inspect

    assert hasattr(h3, "polygon_to_cells_experimental")
    params = inspect.signature(h3.polygon_to_cells_experimental).parameters
    assert "contain" in params
    allowed = getattr(params["contain"].annotation, "__args__", ())
    assert OVERLAP_MODE in allowed, allowed


def test_a_single_distant_object_cannot_decide_the_camera():
    """The guard section 13.6 asks for, as a property rather than a corpus number.

    Janadriyah's `NBHD *` boundaries are real, correctly placed and up to 12 km
    from the estate, and letting them set the default camera shrank the thing
    a person came to look at into a few pixels. Every drawing with a site
    boundary, a key plan or a north arrow parked far from the work does the
    same, Sedra included.

    So: a cell holding one object must never survive into the content box when
    another holds the bulk.
    """
    counts = {"THE_ESTATE": 999, "A_LONE_NORTH_ARROW": 1}
    kept = geo_h3.cells_holding_most(counts)
    assert kept == ["THE_ESTATE"]
    assert "A_LONE_NORTH_ARROW" not in kept

    # And the tail is trimmed from the THINNEST end, however many cells it has.
    counts = {"BULK_A": 500, "BULK_B": 480, **{f"OUTLIER_{i}": 1 for i in range(20)}}
    kept = geo_h3.cells_holding_most(counts)
    assert set(kept) == {"BULK_A", "BULK_B"}, kept


def test_the_content_box_is_all_of_it_when_there_is_no_tail_to_trim():
    """"Most of it" and "all of it" are the same answer for one cell.

    Inventing a difference there would be worse than admitting there is none:
    a drawing whose objects sit in a single cell has no outlier to exclude.
    """
    assert geo_h3.cells_holding_most({"ONLY": 42}) == ["ONLY"]
    assert geo_h3.cells_holding_most({}) == []
    assert geo_h3.cells_holding_most({"A": 0, "B": 0}) == []
    # A share of 1 keeps everything, by the same rule.
    counts = {"A": 10, "B": 1}
    assert set(geo_h3.cells_holding_most(counts, share=1.0)) == {"A", "B"}


def test_the_content_box_covers_whole_cells_not_their_centres():
    """A centre-only box clips the very geometry it exists to frame."""
    cell = h3.latlng_to_cell(24.85, 46.75, geo_h3.VIEWPORT_RESOLUTION)
    box = geo_h3.bounds_of_cells([cell])
    assert box is not None
    (min_lng, min_lat), (max_lng, max_lat) = box
    centre_lat, centre_lng = h3.cell_to_latlng(cell)
    assert min_lng < centre_lng < max_lng
    assert min_lat < centre_lat < max_lat
    # every vertex of the cell is inside the box it produced
    for lat, lng in h3.cell_to_boundary(cell):
        assert min_lat <= lat <= max_lat and min_lng <= lng <= max_lng
