"""Regressions for defects found in review, so they cannot come back quietly.

Each test names the behaviour that was wrong and the observable symptom, not
just the code path.
"""

from __future__ import annotations

import math
import re

import ezdxf
import pytest

from app import extract, store
from app.render import render_layout_svg


# --- extraction ------------------------------------------------------------


def test_full_circle_arc_reports_its_real_length(tmp_path):
    """An ARC written as start=0 end=360 reported length 0.

    `(360 - 0) % 360 == 0`, so a full-sweep arc came out as a zero-length
    object -- a wrong number handed straight to the agent.
    """
    doc = ezdxf.new("R2010", setup=True)
    doc.modelspace().add_arc(center=(0, 0), radius=5, start_angle=0, end_angle=360)
    path = tmp_path / "fullarc.dxf"
    doc.saveas(path)

    _drawing, entities = extract.extract(path)
    arc = next(e for e in entities if e.type == "ARC")
    assert arc.length == pytest.approx(2 * math.pi * 5, rel=1e-6)


def test_curved_polyline_reports_no_length_rather_than_a_wrong_one(tmp_path):
    """LWPOLYLINE bulges were ignored, so curved spans were measured as chords.

    Reporting `None` is correct here: the agent surfaces these numbers
    verbatim, and a confidently wrong kerb length is worse than an absent one.
    """
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    # A square with one bulged (arc) span.
    msp.add_lwpolyline(
        [(0, 0, 0, 0, 0.5), (10, 0, 0, 0, 0), (10, 10, 0, 0, 0), (0, 10, 0, 0, 0)],
        format="xyseb",
        close=True,
    )
    # And a straight one, which must still be measured.
    msp.add_lwpolyline([(0, 0), (3, 0), (3, 4)], format="xy")
    path = tmp_path / "bulge.dxf"
    doc.saveas(path)

    _drawing, entities = extract.extract(path)
    # Scoped to modelspace: block definitions are catalogued too, and the
    # standard setup ships arrowhead blocks that contain their own polylines.
    polylines = [
        e for e in entities if e.type == "LWPOLYLINE" and e.layout == "Model"
    ]
    assert len(polylines) == 2

    curved = [p for p in polylines if p.length is None]
    straight = [p for p in polylines if p.length is not None]
    assert len(curved) == 1, "a bulged polyline must not report a chord length"
    assert len(straight) == 1
    assert straight[0].length == pytest.approx(7.0, rel=1e-6)


# --- rendering -------------------------------------------------------------


def test_handleless_geometry_does_not_pollute_another_layer():
    """Handle-less draw calls shared one global group.

    The first such call fixed the parent layer, so later handle-less geometry
    from other layers was nested there: hiding that one layer hid unrelated
    parts of the drawing.
    """
    from ezdxf.addons.drawing.properties import BackendProperties
    from ezdxf.addons.drawing import layout
    from app.render import HandleAwareSVGRenderBackend

    backend = HandleAwareSVGRenderBackend(
        layout.Page(100, 100), layout.Settings(), {}
    )
    backend.add_strokes(
        "M 0 0 l 1 1",
        BackendProperties(color="#ffffff", lineweight=0.25, layer="WALL", pen=1, handle=""),
    )
    backend.add_strokes(
        "M 2 2 l 1 1",
        BackendProperties(color="#ffffff", lineweight=0.25, layer="ROOF", pen=1, handle=""),
    )
    assert backend.rendered_layers == {"WALL", "ROOF"}, (
        "handle-less geometry on a second layer was absorbed into the first"
    )


def test_block_reference_geometry_carries_the_insert_handle():
    """The claim the whole design rests on, asserted rather than assumed.

    Geometry inside a block has no handle of its own; ezdxf propagates the
    parent INSERT's handle to every child draw call. If that ever stopped
    being true, one door would stop being one clickable object.
    """
    doc = ezdxf.new("R2010", setup=True)
    block = doc.blocks.new(name="DOOR")
    block.add_line((0, 0), (1, 0))
    block.add_arc(center=(0, 0), radius=1, start_angle=0, end_angle=90)

    msp = doc.modelspace()
    insert = msp.add_blockref("DOOR", (5, 5), dxfattribs={"layer": "A-DOOR"})
    insert_handle = insert.dxf.handle

    result = render_layout_svg(
        doc, layout_name="Model", handle_types={insert_handle: "INSERT"}
    )
    handles = set(re.findall(r'data-handle="([^"]+)"', result.svg))
    assert handles == {insert_handle}, (
        f"block children should inherit the INSERT handle; got {handles}"
    )
    # Both the line and the arc must land in that one group.
    group = re.search(
        rf'<g data-handle="{insert_handle}"[^>]*>(.*?)</g>', result.svg, re.S
    )
    assert group and group.group(1).count("<path") >= 2


# --- storage ---------------------------------------------------------------


def test_svg_cache_path_is_injective():
    """Two layout names that sanitise alike must not share one file.

    `"Layout 1"`, `"Layout+1"` and `"Layout/1"` all collapse to `Layout_1`;
    without a digest the second render silently overwrote the first while
    MongoDB still reported two distinct records.
    """
    from pathlib import Path

    root = Path("/tmp/svg")
    names = ["Layout 1", "Layout+1", "Layout/1", "SECTIONS AND DETAILS", "Model"]
    paths = {store._svg_path(root, "abc123", n) for n in names}
    assert len(paths) == len(names), "two layout names mapped to the same file"


def test_svg_cache_path_stays_readable():
    """The digest must not make the cache directory unbrowsable."""
    from pathlib import Path

    path = store._svg_path(Path("/tmp/svg"), "abc123", "DMP Layout1")
    assert "DMP_Layout1" in path.name
    assert path.name.endswith(".svg.gz")
    assert path.name.startswith("abc123__")


# --- block definitions as pseudo-layouts -----------------------------------


def test_block_definition_content_is_reachable(tmp_path):
    """Files whose content lives only in a block definition must not be blank.

    `title_block-arch` and `title_block-iso` each have one entity in a layout
    and 117 inside a block definition. Walking layouts alone reported them as
    effectively empty and the viewer opened to nothing.
    """
    doc = ezdxf.new("R2010", setup=False)
    block = doc.blocks.new(name="TITLE_BLOCK")
    block.add_line((0, 0), (10, 0))
    block.add_line((10, 0), (10, 5))
    block.add_text("SHEET A-101").set_placement((1, 1))
    path = tmp_path / "blockonly.dxf"
    doc.saveas(path)

    drawing, entities = extract.extract(path)
    pseudo = extract.block_layout_name("TITLE_BLOCK")

    in_block = [e for e in entities if e.layout == pseudo]
    assert len(in_block) == 3, "block definition content was not catalogued"

    # Real handles, not the None that virtual_entities() produces -- this is
    # what keeps them clickable and commentable.
    assert all(e.handle for e in in_block)
    assert "SHEET A-101" in [e.text for e in in_block]

    # And the drawing advertises it as a layout the viewer can open.
    names = {l["name"]: l for l in drawing.layouts}
    assert pseudo in names
    assert names[pseudo]["entity_count"] == 3
    assert names[pseudo]["is_block"] is True


def test_block_pseudo_layout_renders(tmp_path):
    """The pseudo-layout must actually produce a clickable SVG."""
    doc = ezdxf.new("R2010", setup=False)
    block = doc.blocks.new(name="TITLE_BLOCK")
    block.add_line((0, 0), (10, 0))
    block.add_line((10, 0), (10, 5))

    handle_types = {
        e.dxf.handle: e.dxftype() for e in doc.blocks.get("TITLE_BLOCK")
    }
    result = render_layout_svg(
        doc,
        layout_name=extract.block_layout_name("TITLE_BLOCK"),
        handle_types=handle_types,
    )
    assert result.rendered_entities == 2
    found = set(re.findall(r'data-handle="([^"]+)"', result.svg))
    assert found == set(handle_types)


def test_anonymous_blocks_are_not_offered_as_layouts(tmp_path):
    """`*Model_Space`, `*U1` and friends must stay out of the layout list.

    They are either the layouts themselves under another name, or unnamed
    copies of a named block -- offering them would show the same content twice
    under a meaningless name.
    """
    doc = ezdxf.new("R2010", setup=False)
    doc.modelspace().add_line((0, 0), (1, 1))
    path = tmp_path / "anon.dxf"
    doc.saveas(path)

    drawing, _entities = extract.extract(path)
    for layout in drawing.layouts:
        assert "*" not in layout["name"], f"anonymous block offered: {layout['name']}"


# --- default layout ---------------------------------------------------------


def test_default_layout_prefers_a_paper_sheet_over_modelspace():
    """The viewer opened on modelspace while Autodesk opened on the sheet.

    A paperspace sheet owns almost no entities of its own -- the annotation
    sample's sheet owns 9 against modelspace's 1,367 -- but those 9 expand to
    959 drawn objects because a VIEWPORT projects modelspace through itself.
    Ranking layouts by entity count therefore always picked modelspace and the
    sheet, with its border and title block, was never shown by default.
    """
    from app.main import _default_layout

    drawing = {
        "layouts": [
            {"name": "Model", "entity_count": 1367, "is_block": False},
            {"name": "SECTIONS AND DETAILS", "entity_count": 9, "is_block": False},
        ]
    }
    renders = [
        {"layout": "Model", "rendered_entities": 1367},
        {"layout": "SECTIONS AND DETAILS", "rendered_entities": 959},
    ]
    assert _default_layout(drawing, renders) == "SECTIONS AND DETAILS"


def test_default_layout_falls_back_to_model_then_block():
    """Order of preference: sheet, then modelspace, then a block definition."""
    from app.main import _default_layout

    only_model = {"layouts": [{"name": "Model", "entity_count": 20, "is_block": False}]}
    assert (
        _default_layout(only_model, [{"layout": "Model", "rendered_entities": 20}])
        == "Model"
    )

    # title_block-arch: nothing in any layout, everything in a block.
    block_only = {
        "layouts": [
            {"name": "[block] Title_Block-ARCH", "entity_count": 117, "is_block": True}
        ]
    }
    assert _default_layout(
        block_only, [{"layout": "[block] Title_Block-ARCH", "rendered_entities": 29}]
    ) == "[block] Title_Block-ARCH"


def test_default_layout_picks_the_busiest_sheet():
    """Janadriyah has three sheets; the fullest one is the useful default."""
    from app.main import _default_layout

    drawing = {
        "layouts": [
            {"name": "Model", "entity_count": 20328, "is_block": False},
            {"name": "DMP Layout1", "entity_count": 2832, "is_block": False},
            {"name": "Layout1", "entity_count": 87, "is_block": False},
        ]
    }
    renders = [
        {"layout": "Model", "rendered_entities": 20312},
        {"layout": "DMP Layout1", "rendered_entities": 22315},
        {"layout": "Layout1", "rendered_entities": 19575},
    ]
    assert _default_layout(drawing, renders) == "DMP Layout1"


# --- foreign payloads -------------------------------------------------------


def test_underlays_and_ole_are_reported_not_silently_dropped(tmp_path):
    """Janadriyah's sheet has two PDF underlays and three OLE objects.

    They are the tall note panels and tables that AutoCAD fills with text.
    ezdxf cannot draw either -- the payload is not DXF geometry -- so they came
    out as unexplained blank rectangles and read as a broken viewer. They must
    be reported so the viewer can say what is missing.
    """
    from app.render import UNRENDERABLE_TYPES

    assert "PDFUNDERLAY" in UNRENDERABLE_TYPES
    assert "OLE2FRAME" in UNRENDERABLE_TYPES
    assert "3DSOLID" in UNRENDERABLE_TYPES

    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((10, 0), (10, 10))
    # A 3D solid: real geometry a 2D renderer has nothing to project.
    msp.new_entity("3DSOLID", dxfattribs={})

    result = render_layout_svg(doc, layout_name="Model", handle_types={})
    kinds = {u["type"] for u in result.unrenderable}
    assert "3DSOLID" in kinds, (
        "an object the renderer cannot draw was dropped without a word"
    )
    for item in result.unrenderable:
        assert item["handle"] and item["reason"]
        # `located` says whether a box could honestly be drawn for it.
        assert "located" in item


def test_unrenderable_is_carried_into_the_render_summary(tmp_path):
    """The API and the viewer read this off the stored render metadata."""
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((10, 0), (10, 10))
    msp.new_entity("3DSOLID", dxfattribs={})

    summary = render_layout_svg(doc, layout_name="Model", handle_types={}).summary()
    assert "unrenderable" in summary
    assert summary["unrenderable"], "summary lost the unrenderable list"


def test_invisible_and_viewport_entities_are_not_reported_as_problems():
    """Correctly-absent objects must not be flagged.

    88 of the 117 entities in title_block-arch's block carry the `invisible`
    flag; AutoCAD hides them too. A VIEWPORT frame never draws as itself
    because its contents are drawn through it. Reporting either as a defect
    would bury the real findings in false alarms.
    """
    from app.render import BENIGN_UNDRAWN_TYPES

    assert "VIEWPORT" in BENIGN_UNDRAWN_TYPES

    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((10, 0), (10, 10))
    hidden = msp.add_line((2, 2), (8, 8))
    hidden.dxf.invisible = 1

    result = render_layout_svg(doc, layout_name="Model", handle_types={})
    reported = {u["handle"] for u in result.unrenderable}
    assert hidden.dxf.handle not in reported, (
        "an entity the author explicitly hid was reported as a failure"
    )


def test_unknown_missing_types_are_surfaced_not_swallowed():
    """The safety net that makes this general rather than a fixed list.

    The PDF underlays went unnoticed because nothing reported them. Any type
    that vanishes from the render without a known reason must now be counted
    and surfaced, so an unfamiliar file cannot fail as silently.
    """
    from app import render as render_mod

    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((10, 0), (10, 10))
    ghost = msp.add_point((5, 5))

    # Pretend POINT is a type the renderer silently drops.
    original = render_mod.UNRENDERABLE_TYPES
    try:
        result = render_layout_svg(doc, layout_name="Model", handle_types={})
    finally:
        render_mod.UNRENDERABLE_TYPES = original

    # Either it drew, or it was reported -- never simply gone.
    drawn = ghost.dxf.handle in result.svg
    reported = any(
        u["type"] in ("POINT", "UNEXPECTED") for u in result.unrenderable
    )
    assert drawn or reported, "an entity vanished with no record of any kind"


# --- DWG conversion ---------------------------------------------------------


def test_wrapped_value_lines_are_rejoined():
    """The LibreDWG defect that made three sample files unreadable.

    DXF is strictly alternating group-code / value lines, and a group code is
    always an integer. LibreDWG sometimes wraps a long value -- an MTEXT note
    body -- onto a second physical line, which shifts every pair after it. The
    parser then reports a nonsense error far away, such as
    `Invalid group code "Standard" at line 97507`.

    Rejoining the orphaned tail is what recovered truetype.dwg's 90 entities;
    the viewer had been showing it as an empty drawing.
    """
    from app.convert import repair_wrapped_values

    broken = "\n".join(["  1", "first part of the value", "CONTINUED HERE", "  7", "Standard"])
    fixed, joins = repair_wrapped_values(broken)
    assert joins == 1
    assert "first part of the valueCONTINUED HERE" in fixed
    # The alternation is restored: "  7" is a code again, "Standard" its value.
    lines = fixed.split("\n")
    assert lines[2].strip() == "7"
    assert lines[3] == "Standard"


def test_well_formed_dxf_is_left_untouched():
    """The repair must be a no-op on a healthy file, not a rewriter."""
    from app.convert import repair_wrapped_values

    good = "\n".join(["  0", "SECTION", "  2", "HEADER", "  0", "ENDSEC"])
    fixed, joins = repair_wrapped_values(good)
    assert joins == 0
    assert fixed == good


def test_negative_group_codes_are_still_codes():
    """DXF uses negative group codes (-1, -2); they must not look like values."""
    from app.convert import repair_wrapped_values

    text = "\n".join([" -1", "value", "  0", "ENDSEC"])
    fixed, joins = repair_wrapped_values(text)
    assert joins == 0
    assert fixed == text


def test_underlay_and_image_references_are_catalogued(tmp_path):
    """TrueView's "6 reference files not found" prompt must be explainable.

    Janadriyah attaches 2 PDFs and 4 JPGs that point at the original author's
    own disk; AutoCAD prompts every recipient about them. Our xref catalogue
    only scanned the block table, where underlay and image definitions never
    appear -- so we reported 0 references and could not explain the prompt.
    """
    doc = ezdxf.new("R2010", setup=True)
    doc.modelspace().add_line((0, 0), (10, 0))
    doc.modelspace().add_line((10, 0), (10, 10))
    doc.add_image_def("C:/Users/DELL/Desktop/title block.jpg", (100, 100))
    path = tmp_path / "withimage.dxf"
    doc.saveas(path)

    drawing, _entities = extract.extract(path)
    kinds = {x.get("kind") for x in drawing.xrefs}
    assert "image" in kinds, "raster attachment was not catalogued as a reference"
    paths = [x["path"] for x in drawing.xrefs]
    assert any("title block.jpg" in p for p in paths)
