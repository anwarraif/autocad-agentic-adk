"""Rendering: the handle -> SVG element mapping.

This is the link the whole viewer rests on. It is also the most fragile part
of the system, because it depends on ezdxf internals that are not part of a
documented API. These tests are the tripwire: if a future ezdxf reorganises
`SVGRenderBackend`, they fail here rather than the viewer silently losing its
ability to map a click back to an entity.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import ezdxf
import pytest
from ezdxf.addons.drawing.properties import BackendProperties

from app.render import (
    MAX_STROKE_PX,
    DEFAULT_COORDINATE_SPACE,
    EmptyRenderError,
    render_layout_svg,
)


@pytest.fixture
def doc():
    document = ezdxf.new("R2010", setup=True)
    document.layers.add("WALL")
    document.layers.add("ANNO")
    msp = document.modelspace()
    msp.add_line((0, 0), (10, 0), dxfattribs={"layer": "WALL"})
    msp.add_line((10, 0), (10, 10), dxfattribs={"layer": "WALL"})
    msp.add_circle((5, 5), radius=3, dxfattribs={"layer": "ANNO"})
    return document


@pytest.fixture
def handle_types(doc):
    return {e.dxf.handle: e.dxftype() for e in doc.modelspace() if e.dxf.handle}


def test_backend_properties_still_carries_handle():
    """The assumption the entire design depends on.

    If ezdxf ever drops `handle` from BackendProperties, no amount of
    subclassing recovers the mapping, and this must be the first thing to
    fail.
    """
    assert "handle" in BackendProperties._fields
    assert "layer" in BackendProperties._fields
    props = BackendProperties(color="#ffffff", lineweight=0.25, layer="L", pen=1, handle="2F1A")
    assert props.handle == "2F1A"


def test_every_entity_gets_a_data_handle_group(doc, handle_types):
    result = render_layout_svg(doc, layout_name="Model", handle_types=handle_types)
    found = set(re.findall(r'data-handle="([^"]+)"', result.svg))
    assert found == set(handle_types), (
        "the set of handles in the SVG does not match the entities rendered"
    )


def test_handles_are_not_duplicated(doc, handle_types):
    """One handle must map to exactly one group.

    An entity split across two groups means a click highlights half of it.
    """
    handles = re.findall(r'data-handle="([^"]+)"', render_layout_svg(
        doc, layout_name="Model", handle_types=handle_types
    ).svg)
    assert len(handles) == len(set(handles))


def test_data_type_matches_the_dxf_type(doc, handle_types):
    svg = render_layout_svg(doc, layout_name="Model", handle_types=handle_types).svg
    for handle, dxftype in handle_types.items():
        match = re.search(
            rf'data-handle="{handle}" data-type="([^"]+)"', svg
        )
        assert match, f"handle {handle} has no data-type"
        assert match.group(1) == dxftype


def test_entities_are_nested_under_their_layer(doc, handle_types):
    """Layer groups exist and wrap the entity groups, so hiding a layer is one
    attribute write rather than a walk over every entity."""
    root = ET.fromstring(render_layout_svg(
        doc, layout_name="Model", handle_types=handle_types
    ).svg)
    layer_groups = root.findall(".//{*}g[@data-layer]")
    names = {g.get("data-layer") for g in layer_groups}
    assert {"WALL", "ANNO"} <= names

    for group in layer_groups:
        for child in group.findall("{*}g"):
            assert child.get("data-handle"), (
                "a layer group has a child that carries no handle"
            )


def test_paths_live_inside_handle_groups(doc, handle_types):
    """No orphan geometry: every drawn path must be attributable."""
    root = ET.fromstring(render_layout_svg(
        doc, layout_name="Model", handle_types=handle_types
    ).svg)
    total_paths = len(root.findall(".//{*}path"))
    attributed = sum(
        len(g.findall("{*}path"))
        for g in root.findall(".//{*}g[@data-handle]")
    )
    assert total_paths > 0
    assert attributed == total_paths


def test_viewbox_is_finite_and_non_degenerate(doc, handle_types):
    """A 1e+20 or zero-size viewBox puts the drawing off-screen."""
    svg = render_layout_svg(doc, layout_name="Model", handle_types=handle_types).svg
    match = re.search(r'viewBox="([^"]+)"', svg)
    assert match
    x, y, w, h = (float(v) for v in match.group(1).split())
    assert w > 0 and h > 0
    assert all(abs(v) < 1e19 for v in (x, y, w, h))


def test_coordinate_space_is_high_enough_to_preserve_geometry(doc, handle_types):
    """Regression: rendering at a low coordinate space destroys the drawing.

    At 1_600 on an 11 km site plan every glyph collapsed to "l 0 -0 l -0 -0".
    The failure is silent -- the SVG is well-formed and simply contains
    nothing -- so it is pinned here.
    """
    assert DEFAULT_COORDINATE_SPACE >= 100_000

    coarse = render_layout_svg(
        doc, layout_name="Model", handle_types=handle_types, coordinate_space=100
    ).svg
    fine = render_layout_svg(
        doc, layout_name="Model", handle_types=handle_types
    ).svg
    # Degenerate output is dramatically smaller; a real render is not.
    assert len(fine) > len(coarse)


def test_stylesheet_ships_with_the_svg(doc, handle_types):
    """The viewer relies on CSS that travels with the document."""
    svg = render_layout_svg(doc, layout_name="Model", handle_types=handle_types).svg
    assert "vector-effect:non-scaling-stroke" in svg
    assert "data-layer-hidden" in svg
    assert "data-selected" in svg


def test_layer_filter_reduces_the_render(doc, handle_types):
    only_wall = render_layout_svg(
        doc, layout_name="Model", handle_types=handle_types, layers=["WALL"]
    )
    assert "ANNO" not in set(only_wall.rendered_layers)
    assert "WALL" in set(only_wall.rendered_layers)


def test_max_entities_truncates_and_says_so(doc, handle_types):
    """Truncation must be reported, never silent."""
    # Two, not one: the fixture's first entity is a horizontal line, and one
    # entity alone gives the page zero height (see the zero-extent test).
    result = render_layout_svg(
        doc, layout_name="Model", handle_types=handle_types, max_entities=2
    )
    assert result.truncated is True
    assert result.truncated_reason
    assert result.rendered_entities <= 2


def test_empty_layout_raises_a_typed_error():
    """A layout with nothing drawable is an expected case, not a crash."""
    document = ezdxf.new("R2010", setup=True)
    document.layouts.get("Layout1").add_viewport(
        center=(0, 0), size=(1, 1), view_center_point=(0, 0), view_height=1
    )
    with pytest.raises(EmptyRenderError):
        render_layout_svg(document, layout_name="Layout1", handle_types={})


def test_missing_handle_type_still_renders(doc):
    """A handle absent from the type map must not drop the geometry."""
    result = render_layout_svg(doc, layout_name="Model", handle_types={})
    assert result.rendered_entities > 0
    assert "data-handle=" in result.svg


def test_stroke_widths_are_pixel_sized_because_of_non_scaling_stroke(doc, handle_types):
    """The bug that made the whole drawing render as five coloured blobs.

    The stylesheet sets `vector-effect: non-scaling-stroke`, which tells the
    browser to ignore the viewBox transform and read `stroke-width` as device
    pixels. ezdxf's defaults emit 150 -- correct as viewBox units (0.2 px on
    screen), catastrophic as pixels: every line drew ~700x too thick and, with
    round linecaps, swallowed the drawing.

    Attribute counts and file sizes all looked perfectly healthy while this was
    happening, which is exactly why this assertion is on the *number*.
    """
    svg = render_layout_svg(doc, layout_name="Model", handle_types=handle_types).svg

    assert "vector-effect:non-scaling-stroke" in svg, (
        "if this is ever removed, the pixel-sized widths below become "
        "hairlines instead - the two settings must change together"
    )

    widths = [
        float(w)
        for w in re.findall(r"stroke-width:\s*([0-9.]+)\s*;", svg)
    ]
    assert widths, "no stroke widths emitted at all"
    assert max(widths) <= MAX_STROKE_PX, (
        f"stroke-width {max(widths)} is being read as {max(widths)} CSS pixels; "
        f"anything above {MAX_STROKE_PX} px turns line work into blobs"
    )
    assert min(widths) >= 1, "sub-pixel strokes disappear on screen"


def test_lineweight_differences_survive(doc):
    """Thick lines must still render thicker than thin ones.

    Guards the fix from being 'simplified' into a single fixed width.
    """
    doc.layers.add("HEAVY")
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0), dxfattribs={"layer": "HEAVY", "lineweight": 200})
    msp.add_line((0, 1), (10, 1), dxfattribs={"layer": "WALL", "lineweight": 9})

    svg = render_layout_svg(doc, layout_name="Model", handle_types={}).svg
    widths = {float(w) for w in re.findall(r"stroke-width:\s*([0-9.]+)\s*;", svg)}
    assert len(widths) > 1, (
        "every line got the same width; lineweight information was thrown away"
    )


def test_sheet_is_white_and_aci7_resolves_to_black(handle_types):
    """Five of sixteen drawings once rendered invisibly on a dark background.

    CAD colours are chosen against the background the drawing is meant to be
    seen on, and these files disagree: some draw dark-for-paper, others use
    ACI 7 which ezdxf turns white when it thinks the background is dark. Only
    setting *both* the sheet colour and the colour resolution satisfies both.
    """
    document = ezdxf.new("R2010", setup=True)
    # ACI 7 = "white or black, depending on the background".
    # Two lines, not one: a single horizontal line has zero height, and ezdxf
    # fits the page to the content bounding box, so the page collapses.
    document.modelspace().add_line((0, 0), (10, 0), dxfattribs={"color": 7})
    document.modelspace().add_line((10, 0), (10, 10), dxfattribs={"color": 7})
    svg = render_layout_svg(document, layout_name="Model", handle_types={}).svg

    background = re.search(r'<rect fill="([^"]+)"', svg)
    assert background and background.group(1) == "#ffffff", (
        "the sheet must be white; a dark sheet hides every dark-for-paper drawing"
    )
    assert "#ffffff" not in re.findall(r"stroke:\s*(#[0-9a-fA-F]{6})", svg), (
        "ACI 7 resolved to white on a white sheet - set_colors() was not applied"
    )


def test_zero_extent_content_is_reported_not_cached_blank():
    """A flat drawing must raise, not silently produce a blank sheet.

    ezdxf fits the page to the content bbox and returns an empty <svg> with no
    error when that bbox has zero width or height. Cached, that is
    indistinguishable from a broken render.
    """
    document = ezdxf.new("R2010", setup=True)
    document.modelspace().add_line((0, 0), (10, 0))  # zero height
    with pytest.raises(EmptyRenderError):
        render_layout_svg(document, layout_name="Model", handle_types={})


# ---------------------------------------------------------------------------
# The drawing -> viewBox affine
#
# Region selection lives or dies on this attribute: it is the only bridge
# between what the user drew on screen and the coordinates MongoDB stores. A
# transposed matrix or a dropped y-flip would not crash anything -- it would
# silently select the wrong objects, which is worse.
# ---------------------------------------------------------------------------


def _world_to_svg(svg: str):
    """The affine from the rendered SVG, as a callable on (x, y)."""
    root = ET.fromstring(svg)
    raw = root.get("data-world-to-svg")
    assert raw, "the SVG root carries no data-world-to-svg"
    a, b, c, d, e, f = (float(v) for v in raw.split())
    return lambda x, y: (a * x + c * y + e, b * x + d * y + f)


def test_world_transform_is_emitted(doc, handle_types):
    svg = render_layout_svg(doc, layout_name="Model", handle_types=handle_types).svg
    root = ET.fromstring(svg)
    values = root.get("data-world-to-svg", "").split()
    assert len(values) == 6
    assert all(_is_finite_number(v) for v in values)


def test_world_transform_maps_content_into_the_viewbox(doc, handle_types):
    """The drawing's own corners must land inside the page it was fitted to.

    The fixture spans (0,0)-(10,10) in drawing units. Whatever the fit does,
    both corners have to fall within the viewBox -- and if the y-flip were
    lost, one of them would land outside it.
    """
    result = render_layout_svg(doc, layout_name="Model", handle_types=handle_types)
    root = ET.fromstring(result.svg)
    _, _, width, height = (float(v) for v in root.get("viewBox").split())
    to_svg = _world_to_svg(result.svg)

    for point in ((0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)):
        x, y = to_svg(*point)
        assert -1 <= x <= width + 1, f"{point} -> x={x} outside 0..{width}"
        assert -1 <= y <= height + 1, f"{point} -> y={y} outside 0..{height}"


def test_world_transform_flips_y_like_the_render_does(doc, handle_types):
    """CAD is y-up, SVG is y-down.

    Without this, a region drawn over the top half of the screen would select
    the bottom half of the drawing -- a failure that looks like "selection is
    just wrong" and is very hard to read back to its cause.
    """
    svg = render_layout_svg(doc, layout_name="Model", handle_types=handle_types).svg
    to_svg = _world_to_svg(svg)
    _, low = to_svg(0.0, 0.0)
    _, high = to_svg(0.0, 10.0)
    assert high < low, "a larger drawing y must map to a smaller SVG y"


def test_world_transform_round_trips_through_its_inverse(doc, handle_types):
    """What the viewer actually does: invert it and go back.

    The browser inverts these six numbers to turn a screen rectangle into
    drawing coordinates. If the inverse does not return the original point to
    within a hair, every selection is off by that error.
    """
    svg = render_layout_svg(doc, layout_name="Model", handle_types=handle_types).svg
    root = ET.fromstring(svg)
    a, b, c, d, e, f = (float(v) for v in root.get("data-world-to-svg").split())
    det = a * d - b * c
    assert det != 0

    def inverse(sx, sy):
        sx, sy = sx - e, sy - f
        return ((d * sx - c * sy) / det, (a * sy - b * sx) / det)

    to_svg = _world_to_svg(svg)
    for point in ((0.0, 0.0), (3.5, 7.25), (10.0, 10.0)):
        back = inverse(*to_svg(*point))
        assert back[0] == pytest.approx(point[0], abs=1e-6)
        assert back[1] == pytest.approx(point[1], abs=1e-6)


def test_world_transform_agrees_with_the_rendered_geometry(doc, handle_types):
    """The strongest check: compare against the paths ezdxf actually emitted.

    The transform is derived from the fitted matrix; the paths are written by
    a separate code path in ezdxf. If the two ever disagree the attribute is
    describing a render that does not exist, and this catches it without any
    knowledge of ezdxf internals -- it only reads the numbers in the file.
    """
    line_handle = next(
        h for h, t in handle_types.items() if t == "LINE"
    )
    result = render_layout_svg(doc, layout_name="Model", handle_types=handle_types)
    root = ET.fromstring(result.svg)
    group = root.find(f".//*[@data-handle='{line_handle}']")
    assert group is not None

    numbers = []
    for path in group.iter():
        d = path.get("d")
        if d:
            numbers += [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d)]
    assert numbers, "the LINE produced no path data"

    # The fixture's two lines both run between points of the 0..10 square, so
    # every emitted coordinate must be inside the transformed square.
    to_svg = _world_to_svg(result.svg)
    corners = [to_svg(x, y) for x in (0.0, 10.0) for y in (0.0, 10.0)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    tolerance = span * 0.02  # SVGRenderBackend rounds coordinates to integers

    # Path data is emitted as absolute moves/lines in pairs.
    for x, y in zip(numbers[0::2], numbers[1::2]):
        assert min(xs) - tolerance <= x <= max(xs) + tolerance
        assert min(ys) - tolerance <= y <= max(ys) + tolerance


def _is_finite_number(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return number == number and abs(number) != float("inf")
