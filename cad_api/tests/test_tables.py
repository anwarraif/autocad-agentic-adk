"""The drawing's tables (UPLIFT-11): layers, text styles, linetypes, dim styles,
header, page setup, and `find_by_name`.

Two kinds of test, both of them asked for by G5.

**The number tests** use Janadriyah values measured on 24 August 2026 from
`JANADRIYAH DMP - 20240506.dwg` RE-converted through `app.convert` — the same
path as ingest — rather than copied from the spec. The file lives in `data/`,
which is gitignored, so it cannot be a fixture: what comes in here is the
RESULT of the measurement, and how it was measured is recorded in
`docs/PROGRESS.md`. Janadriyah's layer names and style names appear verbatim in
this file, and that is legitimate — G1 allows drawing-specific constants in
config files and in tests, precisely so that they need not live in the code.

That conversion source matters and is not a detail: the old DXF copy in
`data/_to_delete/` LOST the whole layer state — zero switched-off layers, zero
per-viewport freezes, one non-plotting layer — while a fresh conversion from
the DWG gives four, two, and six. Any figure about layer state taken from that
copy is wrong. See PROGRESS.md.

**The invariant tests** build their own DXF in memory, following the habit of
`test_extract.py`: a test that states its own conditions cannot drift when the
sample file changes. One further test sweeps every DXF in `data/dxf` when that
directory is mounted, and SKIPS with a reason when it is not — an invariant
over 16 drawings that goes silent when the drawings are absent is an invariant
people stop reading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ezdxf
import pytest

from app import evidence as ev
from app import extract, store_tables

# --- the DXF fixtures -------------------------------------------------------


def _paper_doc(
    tmp_path: Path,
    *,
    insunits: int,
    paper: tuple[float, float] = (841, 1189),
    paper_units: str = "mm",
) -> Path:
    """One paper layout with a page setup, on a drawing whose unit is `insunits`.

    Two unrelated units live in the same file on purpose: `$INSUNITS` describes
    the model, `plot_paper_units` describes the paper. Treating them as the same
    thing is D-076.
    """
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = insunits
    layout = doc.layouts.get("Layout1")
    layout.page_setup(
        size=paper, units=paper_units, scale=(1, 1), name="A0", device="DWG To PDF.pc3"
    )
    layout.add_line((0, 0), (10, 0))
    path = tmp_path / f"paper_{insunits}_{paper_units}.dxf"
    doc.saveas(path)
    return path


@pytest.fixture
def tables_doc(tmp_path: Path) -> Path:
    """A drawing that switches on every column of the layer table at once.

    One layer for each state that has to be tellable apart — locked, frozen,
    off, non-plotting, with a linetype, with a lineweight, with a description,
    frozen in one viewport — plus one plain layer as the control. Without a
    control, a test that passes because ALL layers carry that property cannot
    be told apart from a test that passes because the code is right.
    """
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 6

    plain = doc.layers.add("PLAIN")
    assert plain is not None

    styled = doc.layers.add("STYLED", color=3, linetype="DASHED", lineweight=35)
    styled.lock()
    styled.description = "a layer that carries a description"

    doc.layers.add("NOPLOT", plot=False)
    doc.layers.add("FROZEN").freeze()
    doc.layers.add("OFF").off()
    doc.layers.add("VPFROZEN")

    style = doc.styles.add("ARABIC-SHX", font="xarb.shx")
    style.dxf.bigfont = "Y-ARAB1b.SHX"
    doc.styles.add("ABSOLUTE-PATH", font="C:/ACAD/ESA/X-ARAB1b.SHX")
    doc.styles.add("TRUETYPE", font="arial.ttf")

    doc.dimstyles.add("SCALED").dxf.dimscale = 2.0
    # `dimstyles.add` writes dimscale=1.0 for us, and that is exactly the case
    # this fixture must NOT have: four of Janadriyah's seven dimension styles
    # carry no dimscale at all, and telling them apart from the ones that set
    # it is the whole point of the column.
    doc.dimstyles.add("UNSCALED").dxf.discard("dimscale")

    layout = doc.layouts.get("Layout1")
    layout.page_setup(
        size=(841, 1189), units="mm", scale=(1, 1), name="A0", device="DWG To PDF.pc3"
    )
    viewport = layout.add_viewport(
        center=(400, 600), size=(400, 400), view_center_point=(0, 0), view_height=100
    )
    viewport.frozen_layers = ["VPFROZEN"]

    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0), dxfattribs={"layer": "PLAIN"})
    msp.add_line((0, 0), (0, 10), dxfattribs={"layer": "PLAIN"})

    path = tmp_path / "tables.dxf"
    doc.saveas(path)
    return path


def _layer(drawing: extract.DrawingDoc, name: str) -> dict[str, Any]:
    for row in drawing.layers:
        if row["name"] == name:
            return row
    raise AssertionError(f"layer {name!r} not extracted")


def _layout(drawing: extract.DrawingDoc, name: str) -> dict[str, Any]:
    for row in drawing.layouts:
        if row["name"] == name:
            return row
    raise AssertionError(f"layout {name!r} not extracted")


# --- the layer table: eleven fields ------------------------------------------


LAYER_FIELDS = {
    "name",
    "color",
    "frozen",
    "off",
    "entity_count",
    "linetype",
    "lineweight",
    "plot",
    "locked",
    "description",
    "frozen_in_viewports",
}


def test_layer_table_has_eleven_fields(tables_doc: Path):
    """Eleven, not five. Those six new ones are the whole of UPLIFT-11."""
    drawing, _ = extract.extract(tables_doc)
    assert drawing.layers
    assert len(LAYER_FIELDS) == 11
    for row in drawing.layers:
        assert set(row) == LAYER_FIELDS, row["name"]


def test_layer_display_properties_are_read_not_defaulted(tables_doc: Path):
    """Every new column is read from the file, and the control stays plain."""
    drawing, _ = extract.extract(tables_doc)

    styled = _layer(drawing, "STYLED")
    assert styled["linetype"] == "DASHED"
    assert styled["lineweight"] == 35
    assert styled["locked"] is True
    assert styled["description"] == "a layer that carries a description"

    plain = _layer(drawing, "PLAIN")
    assert plain["linetype"] == "Continuous"
    assert plain["lineweight"] == -3  # ByLayer default, as declared by the file
    assert plain["locked"] is False
    assert plain["description"] is None

    assert _layer(drawing, "NOPLOT")["plot"] is False
    assert plain["plot"] is True
    assert _layer(drawing, "FROZEN")["frozen"] is True
    assert _layer(drawing, "OFF")["off"] is True


def test_a_switched_off_layer_keeps_its_colour():
    """DXF encodes "off" as a NEGATIVE colour. ACI has no such thing.

    Four Janadriyah layers are off, and a raw read reports their colours as
    −20, −8, −1, −40. Being off is already said by `off`; a colour number that
    lies on four layers is a number nobody can use.
    """
    doc = ezdxf.new("R2010")
    doc.layers.add("LIT", color=20)
    doc.layers.add("DARK", color=20).off()
    assert doc.layers.get("DARK").dxf.color == -20  # the raw DXF encoding

    by_name = {l["name"]: l for l in extract._layer_table(doc, {})}
    assert by_name["DARK"]["color"] == 20
    assert by_name["DARK"]["off"] is True
    assert by_name["LIT"]["color"] == 20
    assert by_name["LIT"]["off"] is False


def test_frozen_in_viewports_names_the_viewport_that_froze_it(tables_doc: Path):
    """A per-viewport freeze is a VIEWPORT property, inverted into the layer table.

    The empty list is present on every other layer on purpose: it means
    "checked, there are none", which is not the same thing as not checked.
    """
    drawing, _ = extract.extract(tables_doc)
    frozen_here = _layer(drawing, "VPFROZEN")["frozen_in_viewports"]
    assert len(frozen_here) == 1
    assert isinstance(frozen_here[0], str) and frozen_here[0]

    assert _layer(drawing, "PLAIN")["frozen_in_viewports"] == []


def test_a_broken_layer_description_does_not_sink_the_ingest(tables_doc: Path):
    """G8: 261 audit errors in the largest file make this no edge case."""

    class Exploding:
        dxf = type("D", (), {"name": "X", "get": staticmethod(lambda *a: None)})()

        @property
        def description(self):  # noqa: D401
            raise RuntimeError("extension dictionary is corrupt")

    assert extract._layer_description(Exploding()) is None


# --- text style, linetype, dim style, header --------------------------------


def test_text_styles_carry_font_and_bigfont_verbatim(tables_doc: Path):
    """A font file name is stored verbatim, absolute path included.

    `C:/ACAD/ESA/X-ARAB1b.SHX` is evidence that the font once existed on
    somebody's disk. Normalising it to a basename erases that evidence, and it
    is exactly that evidence which is taken to the drawing's author when asking
    for it.
    """
    drawing, _ = extract.extract(tables_doc)
    by_name = {s["name"]: s for s in drawing.text_styles}

    assert by_name["ARABIC-SHX"]["font"] == "xarb.shx"
    assert by_name["ARABIC-SHX"]["bigfont"] == "Y-ARAB1b.SHX"
    assert by_name["ABSOLUTE-PATH"]["font"] == "C:/ACAD/ESA/X-ARAB1b.SHX"
    assert by_name["TRUETYPE"]["font"] == "arial.ttf"
    assert by_name["TRUETYPE"]["bigfont"] is None


def test_shx_styles_raise_a_warning_rather_than_a_guess(tables_doc: Path):
    """An SHX font not shipped with the drawing is a request to a person, not a bug.

    The right thing to do is to name it and move on. Guessing the glyphs would
    put a supposition where everybody reads it as fact (T-11).
    """
    drawing, _ = extract.extract(tables_doc)
    shx_warnings = [w for w in drawing.warnings if "SHX" in w]
    assert len(shx_warnings) == 1
    assert "not shipped" in shx_warnings[0]


def test_linetype_pattern_length_distinguishes_zero_from_unknown(tables_doc: Path):
    """`Continuous` has a 0.0 that MEANS something; an unreadable one is None."""
    drawing, _ = extract.extract(tables_doc)
    by_name = {t["name"]: t for t in drawing.linetypes}

    assert by_name["Continuous"]["pattern_length"] == 0.0
    assert by_name["DASHED"]["pattern_length"] > 0
    assert "Dashed" in (by_name["DASHED"]["description"] or "")

    assert extract._pattern_length(object()) is None


def test_dim_style_scale_is_read_from_the_style(tables_doc: Path):
    drawing, _ = extract.extract(tables_doc)
    by_name = {d["name"]: d for d in drawing.dim_styles}
    assert by_name["SCALED"]["scale"] == 2.0


def test_dim_style_scale_is_null_when_the_style_does_not_declare_it():
    """What is compared is a style that sets its scale against one that does not.

    Writing the DXF default of 1.0 onto the one that does not set it makes both
    look the same, and erases the difference being asked about: four of
    Janadriyah's seven dimension styles carry no `dimscale` at all, while `moh`
    sets it to 2.0.

    Tested without saving the file first, and that is not laziness: ezdxf
    WRITES `dimscale` 1.0 on export, so a DXF created and read back by ezdxf
    would never hold a style without a dimscale. AutoCAD leaves it out, and it
    is that real file which has to be read correctly.
    """
    doc = ezdxf.new("R2010")
    doc.dimstyles.add("UNSCALED").dxf.discard("dimscale")
    doc.dimstyles.add("SCALED").dxf.dimscale = 2.0

    by_name = {d["name"]: d for d in extract._dim_styles(doc)}
    assert by_name["UNSCALED"]["scale"] is None
    assert by_name["SCALED"]["scale"] == 2.0


def test_header_keys_are_always_present_and_absence_is_null(tmp_path: Path):
    """Three of the 16 drawings declare no unit. That is information.

    Leaving the key out makes "not declared" impossible to tell apart from
    "never read".
    """
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 0
    path = tmp_path / "bare.dxf"
    doc.saveas(path)

    drawing, _ = extract.extract(path)
    assert set(drawing.header) == set(extract.HEADER_VARS)
    assert drawing.header["$INSUNITS"] == 0
    assert drawing.header["$ACADVER"] == "AC1024"


# --- page setup -------------------------------------------------------------


def test_page_setup_is_read_for_a_paper_layout(tables_doc: Path):
    """DoD: it is read, and does not fail silently the way it does today."""
    drawing, _ = extract.extract(tables_doc)
    setup = _layout(drawing, "Layout1")["page_setup"]

    assert setup["available"] is True
    assert setup["why"] is None
    assert setup["paper_width"] == 841.0
    assert setup["paper_height"] == 1189.0
    assert setup["device"] == "DWG To PDF.pc3"
    assert setup["plot_scale_text"] == "1:1"
    assert setup["source"]


@pytest.mark.parametrize(
    "insunits,units_name", [(1, "inch"), (6, "m"), (0, "unitless")]
)
def test_paper_units_never_borrow_the_model_units(
    tmp_path: Path, insunits: int, units_name: str
):
    """G2, and D-076 in miniature.

    This A0 sheet is 841 x 1189 mm in all three drawings, including the one
    whose model is in inches and the one that declares no unit at all. A
    `paper_units` that follows `$INSUNITS` would write "inch" onto figures that
    are plainly millimetres, and that is the shape of the defect that once
    reported 1,867 paper lines as "2992.231252 m".
    """
    path = _paper_doc(tmp_path, insunits=insunits)
    drawing, _ = extract.extract(path)

    assert drawing.units_name == units_name
    setup = _layout(drawing, "Layout1")["page_setup"]
    assert setup["paper_units"] == "mm"
    assert setup["paper_units_code"] == 1


def test_block_pseudo_layouts_carry_no_page_setup(tables_doc: Path):
    """A block definition is not plotted onto paper.

    Giving it that key would invent a question that does not apply to it.
    """
    drawing, _ = extract.extract(tables_doc)
    blocks = [l for l in drawing.layouts if l["is_block"]]
    assert blocks
    for row in blocks:
        assert "page_setup" not in row
    for row in (l for l in drawing.layouts if not l["is_block"]):
        assert "page_setup" in row


def test_ingest_version_moved_so_stale_drawings_are_re_extracted(tables_doc: Path):
    """A document without style tables must read as stale, not as empty."""
    assert extract.INGEST_VERSION >= store_tables.TABLES_INGEST_VERSION
    drawing, _ = extract.extract(tables_doc)
    assert drawing.ingest_version == extract.INGEST_VERSION


# --- the invariant over every mounted drawing --------------------------------


def _sample_drawings() -> list[Path]:
    directory = Path(os.environ.get("CAD_DXF_DIR", "/data/dxf"))
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.dxf"))


def _sample_params() -> list[Any]:
    """One param per drawing, or one param that SKIPS with a reason.

    A `parametrize` over an empty list disappears from the report without a
    trace, and an invariant that disappears without a trace is one people stop
    reading.
    """
    paths = _sample_drawings()
    if not paths:
        return [
            pytest.param(
                None,
                id="no-drawings-mounted",
                marks=pytest.mark.skip(
                    reason=(
                        "no drawings under CAD_DXF_DIR (default /data/dxf): the "
                        "every-drawing invariant did not run. Mount data/dxf "
                        "read-only to run it."
                    )
                ),
            )
        ]
    return [pytest.param(p, id=p.stem) for p in paths]


@pytest.mark.parametrize("path", _sample_params())
def test_tables_hold_for_every_mounted_drawing(path: Path):
    """G5: run over every drawing there is, not over one.

    Not figures but shape — the eleven fields are always there, the paper unit
    never follows the model unit, and a page setup that could not be read says
    why. This is what will catch the breakage on the 17th drawing.

    A file that cannot be opened AT ALL is skipped with its reason named rather
    than counted as a failure: two sample files have been broken since the
    LibreDWG conversion (T-4), and leaving them red here would turn this gate
    into a light that is always on and stops being read. What is tested is the
    shape of the tables, not whether the file is intact.
    """
    try:
        drawing, _ = extract.extract(path)
    except extract.ExtractionError as exc:
        pytest.skip(f"{path.name} cannot be opened at all: {exc}")

    for row in drawing.layers:
        assert set(row) == LAYER_FIELDS, (path.name, row.get("name"))
        assert isinstance(row["frozen_in_viewports"], list)

    for row in drawing.layouts:
        if row["is_block"]:
            assert "page_setup" not in row
            continue
        setup = row["page_setup"]
        if setup["available"]:
            assert setup["paper_units"] in (None, "inch", "mm", "pixel")
        else:
            assert setup["why"], (path.name, row["name"])
            assert setup["paper_units"] is None

    assert set(drawing.header) == set(extract.HEADER_VARS)
    for style in drawing.text_styles:
        assert "name" in style and "font" in style and "bigfont" in style


# --- find_by_name: the Janadriyah figures ------------------------------------


#: A slice of Janadriyah's `autocad_drawings` document, MEASURED from its DXF on
#: 24 August 2026, not copied from the spec.
#:
#: The layer table here is a slice, not all 292 of its rows. What makes this
#: slice enough: the same measurement swept ALL 292 layers and returned exactly
#: five names containing "school", exactly three containing "mosque", and
#: exactly one containing "vl2". The decoy layers below are there so that a
#: search which returns everything still fails.
#: The text style table is ALL 23 rows verbatim.
JANADRIYAH: dict[str, Any] = {
    "_id": "janadriyah-slice",
    "ingest_version": 4,
    "units_code": 6,
    "units_name": "m",
    "layers": [
        {"name": "Private School", "entity_count": 1, "linetype": "Continuous"},
        {"name": "Intermediate School", "entity_count": 2, "linetype": "Continuous"},
        {"name": "Secondary School", "entity_count": 2, "linetype": "Continuous"},
        {"name": "Primary School", "entity_count": 2, "linetype": "Continuous"},
        {"name": "SchoolHatch", "entity_count": 8, "linetype": "Continuous"},
        {"name": "Local Mosque", "entity_count": 5, "linetype": "Continuous"},
        {"name": "Jumaa Mosque", "entity_count": 1, "linetype": "Continuous"},
        {
            "name": "CS-Land use-00_Religious_Mosque",
            "entity_count": 0,
            "linetype": "Continuous",
        },
        {"name": "VL2", "entity_count": 133, "linetype": "Continuous"},
        {
            "name": "C-PROP-BlockNumber",
            "entity_count": 2446,
            "linetype": "Continuous",
            "color": 20,
            "frozen": False,
            "off": True,
            "locked": False,
            "plot": True,
            "frozen_in_viewports": [],
        },
        {
            "name": "CommercialHatch",
            "entity_count": 12,
            "linetype": "Continuous",
            "frozen_in_viewports": ["F9FD70", "B0BCD1"],
        },
        {"name": "Defpoints", "entity_count": 0, "linetype": "Continuous", "plot": False},
        {"name": "V-CTRL-HCPT", "entity_count": 3, "linetype": "CENTER2", "locked": True},
        {"name": "NBHD A", "entity_count": 1, "linetype": "Continuous", "frozen": True},
    ],
    "blocks": ["Title Block", "ADA"],
    "layouts": [
        {"name": "Model", "entity_count": 23169},
        {"name": "DMP Layout1", "entity_count": 42},
        {"name": "Layout1", "entity_count": 36},
    ],
    "text_styles": [
        {"name": "Standard", "font": "simplex.shx", "bigfont": None},
        {"name": "S4", "font": "simplex", "bigfont": None},
        {"name": "SHR", "font": "SIMPLEX", "bigfont": None},
        {"name": "SOSA", "font": "sosa.shx", "bigfont": None},
        {"name": "xarab", "font": "xarb.shx", "bigfont": None},
        {"name": "DTE", "font": "xarab.shx", "bigfont": "Y-ARAB1b.SHX"},
        {"name": "STYLE1", "font": "xarb.shx", "bigfont": None},
        {"name": "ana--n", "font": "xarb.shx", "bigfont": None},
        {"name": "ROMANC", "font": "ROMANC", "bigfont": None},
        {"name": "ROMAN_TRIPLEX", "font": "ROMANT", "bigfont": None},
        {"name": "PLOTTER_1", "font": "MONOTXT", "bigfont": None},
        {"name": "XARB", "font": "xarb.shx", "bigfont": None},
        {"name": "BOLD-SOFT", "font": "swisscb.ttf", "bigfont": None},
        {"name": "A2", "font": "C:/ACAD/ESA/X-ARAB1b.SHX", "bigfont": "Y-ARAB1b.SHX"},
        {"name": "CC-A-TITLE", "font": "xarb.shx", "bigfont": None},
        {"name": "road name$0$XARB", "font": "xarb.shx", "bigfont": None},
        {"name": "Warefa Context$0$_STYLE", "font": "XARB.SHX", "bigfont": None},
        {"name": "Warefa Context$0$STYLE1", "font": "xarb.shx", "bigfont": None},
        {"name": "_STYLE", "font": "XARB.SHX", "bigfont": None},
        {"name": "RS-DIM", "font": "xarb.shx", "bigfont": None},
        {"name": "DynamicDimensions", "font": "arial.ttf", "bigfont": None},
        {"name": "moh -2", "font": "xarab.shx", "bigfont": None},
        {"name": "Arial W0.8", "font": "arial.ttf", "bigfont": None},
    ],
    "linetypes": [
        {"name": "Continuous", "description": "Solid line", "pattern_length": 0.0},
        {"name": "CENTER2", "description": "Center (.5x)", "pattern_length": 1.125},
    ],
    "dim_styles": [
        {"name": "Standard", "scale": None},
        {"name": "moh", "scale": 2.0},
    ],
    "header": {"$INSUNITS": 6, "$LTSCALE": 100.0, "$DIMSCALE": 2.0},
}


def test_find_by_name_school_returns_the_five_layers():
    """DoD: `school` -> 5 layers. `search_text("SCHOOL")` used to give 0 results.

    Both are correct, and that is exactly the point: there is not one TEXT
    reading "school" in this drawing, while five LAYERS say it in their names.
    """
    out = store_tables.search_names(JANADRIYAH, "school", ["layer"])
    assert out["ok"] is True
    assert {h["name"] for h in out["hits"]} == {
        "Private School",
        "Intermediate School",
        "Secondary School",
        "Primary School",
        "SchoolHatch",
    }
    assert out["hits_total"] == 5
    assert all(h["kind"] == "layer" for h in out["hits"])


def test_find_by_name_is_case_insensitive_and_substring():
    out = store_tables.search_names(JANADRIYAH, "SCHOOL", ["layer"])
    assert out["hits_total"] == 5
    assert store_tables.search_names(JANADRIYAH, "chool", ["layer"])["hits_total"] == 5


def test_find_by_name_mosque_flags_the_empty_layer():
    """DoD: 3 layers, one of them empty and it MUST be flagged.

    A layer named after a mosque that holds nothing is a different answer from
    a layer named after a mosque that holds its parcel, and whoever asked needs
    to be able to tell them apart without making another call.
    """
    out = store_tables.search_names(JANADRIYAH, "mosque", ["layer"])
    assert out["hits_total"] == 3
    by_name = {h["name"]: h for h in out["hits"]}
    assert by_name["CS-Land use-00_Religious_Mosque"]["is_empty"] is True
    assert by_name["Local Mosque"]["is_empty"] is False
    assert out["empty_hits"] == ["CS-Land use-00_Religious_Mosque"]


def test_find_by_name_arab_matches_styles_through_their_font_files():
    """DoD: `arab` -> text styles TOGETHER WITH their font files.

    Measured: four styles match, through six fields. `xarab` matches through
    its NAME; `DTE`, `A2` and `moh -2` say "arab" nowhere except in their font
    file names. The spec writes "6 text styles"; the six are the fields that
    matched, not the styles — this discrepancy is recorded in PROGRESS.md.
    Both are readable from this response, and that is what `matched_on` is for.
    """
    out = store_tables.search_names(JANADRIYAH, "arab", ["text_style"])
    by_name = {h["name"]: h for h in out["hits"]}

    assert set(by_name) == {"xarab", "DTE", "A2", "moh -2"}
    assert by_name["xarab"]["matched_on"] == ["name"]
    assert by_name["DTE"]["matched_on"] == ["font", "bigfont"]
    assert by_name["moh -2"]["matched_on"] == ["font"]
    assert sum(len(h["matched_on"]) for h in out["hits"]) == 6

    assert by_name["DTE"]["font"] == "xarab.shx"
    assert by_name["DTE"]["bigfont"] == "Y-ARAB1b.SHX"
    assert by_name["A2"]["font"] == "C:/ACAD/ESA/X-ARAB1b.SHX"
    assert all(h["font_is_shx"] for h in out["hits"])


def test_the_switched_off_label_layer_is_now_answerable_from_data():
    """The `C-PROP-BlockNumber` claim — CONFIRMED, and that is the opposite of
    what was supposed.

    `GROUND-TRUTH-2.md` Q2.6 writes that this layer is switched off: 2,446
    labels that are not displayed. UPLIFT-11 recorded it as "an old claim that
    is not proven", because the 23 August read got `is_on() == True` and zero
    switched-off layers out of 292.

    Re-measured on 24 August from the DWG through ingest's own conversion path:
    this layer IS off, its colour 20 is encoded as −20, and it is one of four
    switched-off layers whose names are exactly the ones Q2.6 listed. What was
    wrong is the 23 August read — it took the old DXF copy that had already
    lost the layer state.

    The spec writes its own resolution: "store the layer table as data that can
    be tested, not as a sentence in a document". This is that data.
    """
    out = store_tables.search_names(JANADRIYAH, "C-PROP-BlockNumber", ["layer"])
    assert out["hits_total"] == 1
    assert out["hits"][0]["entity_count"] == 2446

    layer = next(
        l for l in JANADRIYAH["layers"] if l["name"] == "C-PROP-BlockNumber"
    )
    assert layer["off"] is True
    assert layer["frozen"] is False
    assert layer["frozen_in_viewports"] == []
    assert layer["color"] == 20

    summary = store_tables.tables_summary(JANADRIYAH)
    assert summary["layers_off"] == 1


def test_find_by_name_vl2_returns_one_layer_with_its_count():
    """DoD: `VL2` -> 1 layer, 133 entities."""
    out = store_tables.search_names(JANADRIYAH, "VL2", ["layer"])
    assert out["hits_total"] == 1
    assert out["hits"][0]["entity_count"] == 133
    assert out["hits"][0]["is_empty"] is False
    assert out["hits"][0]["count_basis"]


def test_a_style_usage_count_is_null_and_never_zero():
    """G8: "not counted" must not disguise itself as "empty".

    Entities do not carry their text style yet (that is UPLIFT-04's request),
    so a style's usage genuinely cannot be counted from this store. What is
    wrong is answering 0 — somebody would throw away a style used 153 times
    because of that number.
    """
    out = store_tables.search_names(JANADRIYAH, "DTE", ["text_style"])
    hit = out["hits"][0]
    assert hit["entity_count"] is None
    assert hit["is_empty"] is None
    assert "not stored" in hit["count_basis"]


def test_a_linetype_reports_the_layers_that_declare_it():
    """What is not known is said; what is known is still answered."""
    out = store_tables.search_names(JANADRIYAH, "CENTER2", ["linetype"])
    hit = out["hits"][0]
    assert hit["entity_count"] is None
    assert hit["layers_using"] == 1
    assert hit["pattern_length"] == 1.125


def test_names_do_not_claim_meaning():
    """G10 as a sentence: a name is not a classification.

    A response that returns `Primary School` without saying this will be read
    as "there is a school here", and that is a claim nobody has made.
    """
    out = store_tables.search_names(JANADRIYAH, "school", ["layer"])
    assert "not a claim about what the object IS" in out["name_is_not_meaning"]
    assert "search_text" in out["what_this_searches"]
    assert "find_by_name" in out["what_this_searches"]


# --- find_by_name: behaviour at the edges ------------------------------------


def test_empty_query_is_refused_with_a_suggestion():
    out = store_tables.search_names(JANADRIYAH, "   ")
    assert out["ok"] is False
    assert out["code"] == "EMPTY_QUERY"
    assert out["hint"]


def test_an_unknown_kind_is_named_rather_than_ignored():
    out = store_tables.search_names(JANADRIYAH, "school", ["layer", "hatch"])
    assert out["ok"] is False
    assert out["code"] == "UNKNOWN_KIND"
    assert "hatch" in out["message"]


def test_a_drawing_ingested_before_the_style_tables_says_so():
    """G3/G8: not 0 results and not an error — a reason.

    Zero results on an old drawing cannot be told apart from zero results on a
    drawing that genuinely has no such style, and the first one can be fixed by
    re-ingesting.
    """
    stale = {k: v for k, v in JANADRIYAH.items() if k != "text_styles"}
    stale["ingest_version"] = 3
    out = store_tables.search_names(stale, "arab")

    assert out["ok"] is True
    assert "text_style" not in out["kinds_searched"]
    missing = [n for n in out["kinds_not_searched"] if n["kind"] == "text_style"]
    assert len(missing) == 1
    assert "re-ingest" in missing[0]["why"]


def test_a_block_kind_without_counts_reports_null_not_zero():
    out = store_tables.search_names(JANADRIYAH, "title", ["block"], block_counts=None)
    assert out["hits_total"] == 1
    assert out["hits"][0]["entity_count"] is None
    assert out["hits"][0]["is_empty"] is None

    counted = store_tables.search_names(
        JANADRIYAH, "title", ["block"], block_counts={"Title Block": 3}
    )
    assert counted["hits"][0]["entity_count"] == 3


def test_the_result_limit_is_stated_and_the_overflow_is_counted():
    """G7: the cap is stated in the response, not cut silently."""
    many = dict(JANADRIYAH)
    many["layers"] = [
        {"name": f"L{i}", "entity_count": i, "linetype": "Continuous"}
        for i in range(store_tables.MAX_HITS + 25)
    ]
    out = store_tables.search_names(many, "L", ["layer"])

    assert out["truncated"] is True
    assert out["hits_total"] == store_tables.MAX_HITS + 25
    assert out["hits_returned"] == store_tables.MAX_HITS
    assert len(out["hits"]) == store_tables.MAX_HITS
    assert "Narrow the query" in out["suggestion"]


def test_no_hits_is_a_clean_answer_not_an_error():
    out = store_tables.search_names(JANADRIYAH, "zzzz-nothing-named-this")
    assert out["ok"] is True
    assert out["hits"] == []
    assert out["hits_total"] == 0
    assert out["empty_hits"] == []


# --- the `tables` block and the SHX font hole --------------------------------


def test_tables_summary_counts_what_is_worth_saying_unasked():
    out = store_tables.tables_summary(JANADRIYAH)
    assert out["layer_count"] == len(JANADRIYAH["layers"])
    assert out["layers_locked"] == 1
    assert out["layers_not_plotted"] == 1
    assert out["layers_frozen"] == 1
    assert out["layers_off"] == 1
    assert out["layers_frozen_in_some_viewport"] == 1
    assert out["layers_non_continuous"] == 1
    assert out["text_style_count"] == 23
    # 15 of the 23 draw through an SHX font. Thirteen of those name one of the
    # three Arabic files; the other two are `simplex.shx` and `sosa.shx`, which
    # are Latin. The spec says eleven point at the Arabic files -- measured, it
    # is thirteen. Recorded in PROGRESS.md rather than quietly adjusted.
    assert out["text_styles_using_shx"] == 15
    arabic = [
        s
        for s in JANADRIYAH["text_styles"]
        if any(
            n in str(s.get(f) or "").lower()
            for f in ("font", "bigfont")
            for n in ("xarb.shx", "xarab.shx", "x-arab1b.shx")
        )
    ]
    assert len(arabic) == 13
    assert set(out["shx_font_files"]) == {
        "simplex.shx",
        "sosa.shx",
        "xarb.shx",
        "xarab.shx",
        "XARB.SHX",
        "C:/ACAD/ESA/X-ARAB1b.SHX",
        "Y-ARAB1b.SHX",
    }
    assert out["tables_stale"] is False


def test_tables_summary_marks_a_pre_uplift_drawing_stale():
    stale = {k: v for k, v in JANADRIYAH.items() if k != "linetypes"}
    out = store_tables.tables_summary(stale)
    assert out["linetype_count"] is None
    assert out["tables_stale"] is True
    assert "re-ingest" in out["tables_stale_why"]


def test_the_shx_gap_is_unknown_and_says_what_is_missing_and_whom_to_ask():
    """One of the three things that cannot be solved by coding.

    The right thing to do: `verified: false`, name which files are missing and
    whom to ask, then move on. An empty `unknown` is an answer that is easy to
    skip past, and the contract refuses to publish one.
    """
    out = store_tables.shx_font_gap(JANADRIYAH)

    assert out["verified"] is False
    assert out["font_files_present"] is False
    assert out["evidence"]["grade"] == ev.Grade.UNKNOWN.value
    assert "xarb.shx" in out["evidence"]["not_established"]
    assert "ask the drawing's author" in out["evidence"]["how_to_verify"]
    assert out["scope_note"]
    assert ev.audit_response(out) == []


def test_the_shx_gap_on_a_drawing_without_shx_fonts_still_answers():
    """G3: absence is answered gracefully, not with an error and not in silence."""
    truetype_only = dict(JANADRIYAH)
    truetype_only["text_styles"] = [
        {"name": "Arial W0.8", "font": "arial.ttf", "bigfont": None}
    ]
    out = store_tables.shx_font_gap(truetype_only)

    assert out["shx_font_files"] == []
    assert out["evidence"]["grade"] == ev.Grade.UNKNOWN.value
    assert "no text style" in out["evidence"]["not_established"]
    assert ev.audit_response(out) == []
