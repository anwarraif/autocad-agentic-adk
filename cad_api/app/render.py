"""DXF -> SVG rendering with a stable `handle` on every entity group.

Why this module exists
----------------------
The viewer must be able to map a click on a `<path>` back to the DWG entity it
came from. ezdxf's stock `SVGBackend` groups output by *colour class*
(``.C1``, ``.C2``), so the handle is discarded on the way out.

How the injection works (verified against ezdxf 1.4.4, see tests/test_render.py)
--------------------------------------------------------------------------------
`SVGBackend` is a `Recorder`: it records every draw call, then replays it onto
an `SVGRenderBackend` that builds the XML tree. Two facts make a clean hook
possible:

1. `SVGBackend.make_backend()` carries the docstring "Override this method to
   use a customized render backend." — a sanctioned extension point.
2. Every `<path>` element in the output is created by exactly two methods,
   `SVGRenderBackend.add_strokes()` and `.add_filling()`, and both are handed
   the `BackendProperties` namedtuple whose fields are
   ``(color, lineweight, layer, pen, handle)``.

So we subclass `SVGRenderBackend`, override those two methods to re-parent the
element into a `<g data-handle=...>`, and delegate the actual colour /
stroke-width resolution back to the library. No geometry maths is
re-implemented here; if ezdxf changes its internals the tests fail loudly
rather than the viewer silently losing its handles.

Output shape
------------
    <svg viewBox="0 0 W H">
      <g>                                  <- SVGRenderBackend.entities
        <g data-layer="A-DOOR">            <- one per layer, cheap layer toggle
          <g data-handle="2F1A" data-type="INSERT">
            <path class="C1" d="..."/>
          </g>
        </g>
      </g>
    </svg>

Two levels of grouping, not one, because they answer different questions:
hiding a layer must be a single attribute write (there can be hundreds of
layers but tens of thousands of entities), while a click must resolve to one
entity.
"""

from __future__ import annotations

import collections
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing.config import Configuration, LineweightPolicy
from ezdxf.addons.drawing.properties import BackendProperties
from ezdxf.addons.drawing.svg import SVGBackend, SVGRenderBackend
from ezdxf.document import Drawing
from ezdxf.layouts import Layout
from ezdxf.math import Vec3

from .extract import block_name_of, is_block_layout

log = logging.getLogger(__name__)

# A layer name is not a legal XML id, and handles are only unique per drawing,
# so both are carried as data-* attributes rather than as `id`.
ATTR_HANDLE = "data-handle"
ATTR_LAYER = "data-layer"
ATTR_TYPE = "data-type"

#: The drawing-coordinates -> viewBox-coordinates affine, written on the <svg>
#: root as six numbers "a b c d e f" in SVG's own `matrix(...)` order:
#:
#:     X = a*x + c*y + e        Y = b*x + d*y + f
#:
#: The viewer needs the INVERSE of this to turn a region the user drew on
#: screen into drawing coordinates, which is the only form in which a region
#: means anything: viewBox units are an artefact of how this particular layout
#: was fitted to a page, while drawing coordinates are what MongoDB stores and
#: what /spatial queries. Deriving it in the browser was the alternative and it
#: is guesswork -- the fit depends on ezdxf's page logic, the y-flip, and the
#: content bounding box, none of which the browser can see.
ATTR_WORLD_TO_SVG = "data-world-to-svg"

# Entities that the frontend draws but that carry no usable handle (anonymous
# construction geometry). They are still rendered, just not clickable.
_UNKNOWN_HANDLE = "_"

#: `SVGRenderBackend` rounds every output coordinate to an integer, so this
#: number is the precision of the whole drawing, not a pixel size. Measured on
#: Janadriyah (11.7 km wide): at 1_600 every text glyph collapsed to
#: "l 0 -0 l -0 -0" — 27k paths of pure noise, because one unit was 7 metres.
#: 1_000_000 (ezdxf's own default) puts one unit at ~1 cm. The viewBox is
#: unitless and the browser scales it, so a large value costs file size, not
#: correctness; a small one destroys geometry irreversibly.
DEFAULT_COORDINATE_SPACE = 1_000_000

#: Widest stroke, **in CSS pixels**, because the stylesheet sets
#: `vector-effect: non-scaling-stroke`.
#:
#: This pairing is the whole point and was got wrong once, expensively.
#: ezdxf sizes strokes in *viewBox units*: with the default settings it emits
#: `stroke-width: 150`, which against a 1,000,000-unit viewBox drawn into a
#: ~1,400 px pane is 0.2 px — a hairline, exactly right. But
#: `non-scaling-stroke` tells the browser to ignore the viewBox transform and
#: read that number as **150 device pixels**. Every line then rendered ~700x
#: too thick, and with ezdxf's round linecaps the whole drawing collapsed into
#: a handful of coloured blobs.
#:
#: So the emitted numbers must themselves be pixel-sized. Settings are derived
#: from this constant rather than hard-coded, so the two can never drift apart
#: again; `tests/test_render.py` pins the invariant.
MAX_STROKE_PX = 4

#: Sheet colour. See the note in `render_layout_svg` -- this decides whether
#: dark-on-paper drawings are visible at all, not merely how they look.
WHITE_SHEET = "#ffffff"

#: Entity types that carry a *foreign payload* rather than DXF geometry, and
#: that therefore no DXF renderer can draw.
#:
#: Janadriyah's DMP sheet has two `PDFUNDERLAY` and three `OLE2FRAME` objects.
#: They are the two tall note panels and the tables on the right-hand side --
#: in AutoCAD they are full of text, and in our render they were simply blank
#: boxes with no explanation, which reads as a broken viewer rather than as a
#: known limitation. (The underlay *definitions* did not even survive the
#: LibreDWG conversion: `get_underlay_def()` raises.)
#:
#: `3DSOLID` and friends are included for the same reason: they are real
#: geometry, but a 2D renderer has nothing to project. `visualization_-_aerial`
#: is five of them and nothing else.
UNRENDERABLE_TYPES: frozenset[str] = frozenset(
    {
        "PDFUNDERLAY",
        "DWFUNDERLAY",
        "DGNUNDERLAY",
        "OLE2FRAME",
        "OLEFRAME",
        "ACAD_PROXY_ENTITY",
        "3DSOLID",
        "BODY",
        "REGION",
        "MESH",
        "SURFACE",
        "PLANESURFACE",
        "EXTRUDEDSURFACE",
        "REVOLVEDSURFACE",
        "SWEPTSURFACE",
        "LOFTEDSURFACE",
        "LIGHT",
        "SUN",
        "POINTCLOUD",
        "POINTCLOUDEX",
    }
)

#: Types whose absence from the output is *correct* and needs no report.
#:
#: `VIEWPORT` is the frame of a paperspace window: its contents are drawn
#: through it, so the frame itself never appears as its own object. Anything
#: carrying the `invisible` flag was hidden by the author -- 88 of the 117
#: entities in `title_block-arch`'s block are flagged that way, and AutoCAD
#: hides them too.
BENIGN_UNDRAWN_TYPES: frozenset[str] = frozenset({"VIEWPORT"})

#: Human wording for the placeholder, per type.
_UNRENDERABLE_LABEL: dict[str, str] = {
    "PDFUNDERLAY": "PDF underlay - not rendered",
    "DWFUNDERLAY": "DWF underlay - not rendered",
    "DGNUNDERLAY": "DGN underlay - not rendered",
    "OLE2FRAME": "embedded OLE object - not rendered",
    "OLEFRAME": "embedded OLE object - not rendered",
    "ACAD_PROXY_ENTITY": "proxy entity - needs the originating application",
    "3DSOLID": "3D solid - not drawn by the 2D renderer",
    "BODY": "3D body - not drawn by the 2D renderer",
    "REGION": "region - not drawn by the 2D renderer",
}

#: Never draw more than this many placeholders; past a handful they stop
#: informing and start obscuring.
MAX_PLACEHOLDERS = 40


def _stroke_settings(coordinate_space: int) -> dict[str, float]:
    """Layout settings that make ezdxf emit pixel-sized stroke widths.

    ezdxf computes:
        max    = int(output_coordinate_space * max_stroke_width)
        min    = int(max * min_stroke_width)
        fixed  = int(max * fixed_stroke_width)

    so asking for `MAX_STROKE_PX` at the top means dividing by the coordinate
    space here. The result is a 1..4 px range, which keeps thin and thick
    lineweights visibly different without any line becoming a blob.
    """
    return {
        "max_stroke_width": MAX_STROKE_PX / coordinate_space,
        "min_stroke_width": 0.25,  # -> 1 px
        "fixed_stroke_width": 0.25,  # -> 1 px
    }


def _tag_world_transform(root: ET.Element, matrix) -> None:
    """Write the drawing -> viewBox affine onto the <svg> root.

    Read back by the viewer, inverted, to convert a region drawn on screen
    into drawing coordinates. See `ATTR_WORLD_TO_SVG`.

    The six numbers are obtained by pushing three points through the very
    matrix the renderer used -- the same `matrix.transform()` call that places
    the unrenderable-object outlines -- rather than by reading the matrix's
    cells. ezdxf composes with row vectors, so cell indices would have to be
    transposed relative to SVG's column-vector `matrix(a,b,c,d,e,f)`; that is
    a convention detail nobody should have to remember twice, and getting it
    silently backwards would put every selection in the wrong place.

    Args:
        root: the `<svg>` element to annotate.
        matrix: ezdxf's fitted transformation, or None if the page never got
            one. Nothing is written in that case, and the viewer disables
            region selection rather than guessing.
    """
    if matrix is None:
        return
    try:
        origin = matrix.transform(Vec3(0, 0, 0))
        unit_x = matrix.transform(Vec3(1, 0, 0))
        unit_y = matrix.transform(Vec3(0, 1, 0))
    except Exception as exc:  # noqa: BLE001 - annotation is best-effort
        log.warning("could not derive world transform: %s", exc)
        return

    a, b = unit_x.x - origin.x, unit_x.y - origin.y
    c, d = unit_y.x - origin.x, unit_y.y - origin.y
    e, f = origin.x, origin.y

    # A singular matrix cannot be inverted, so the attribute would be a
    # promise the viewer could not keep. Better absent than wrong.
    if abs(a * d - b * c) < 1e-12:
        log.warning("world transform is singular; not tagging the SVG")
        return

    # 15 significant digits: a double carries ~15.95, and these coordinates
    # run to 7 figures before the point (UTM eastings), so trimming to the
    # usual 6 would move a selection edge by centimetres.
    root.set(ATTR_WORLD_TO_SVG, " ".join(f"{v:.15g}" for v in (a, b, c, d, e, f)))


def _mark_unrenderable(
    parent: ET.Element,
    entities: list,
    matrix,
    rendered: set[str],
) -> list[dict[str, object]]:
    """Record -- and where possible outline -- objects no DXF renderer can draw.

    Janadriyah's DMP sheet carries two `PDFUNDERLAY` and three `OLE2FRAME`
    objects. Those are the two tall note panels and the tables that AutoCAD
    fills with text; here they were blank rectangles with no explanation,
    which reads as a broken viewer rather than a known limitation.

    A dashed outline is drawn only when the object's bounding box is actually
    known. On this file it is not: LibreDWG dropped the placement attributes
    along with the underlay definitions, so `bbox.extents()` reports no data
    and `insert`/`u_size` are absent. Drawing a box at a guessed position
    would be worse than drawing nothing -- it would assert a location the file
    no longer contains. So those objects are reported instead, and the viewer
    says what is missing without inventing where.
    """
    from ezdxf import bbox as _bbox

    found: list[dict[str, object]] = []
    unexpected: collections.Counter[str] = collections.Counter()

    for entity in entities:
        dxftype = entity.dxftype()
        handle = entity.dxf.get("handle")
        if not handle or handle in rendered:
            continue

        # Correctly absent: a viewport frame, or something the author hid.
        if dxftype in BENIGN_UNDRAWN_TYPES or entity.dxf.get("invisible", 0):
            continue

        if dxftype not in UNRENDERABLE_TYPES:
            # The safety net. A type nobody anticipated went missing without a
            # word; count it so a new file cannot fail silently the way the
            # PDF underlays did. Reported as a warning, not as fact about the
            # object, because we do not know why it did not draw.
            unexpected[dxftype] += 1
            continue

        if len(found) >= MAX_PLACEHOLDERS:
            continue

        reason = _UNRENDERABLE_LABEL.get(dxftype, f"{dxftype} - not rendered")
        record: dict[str, object] = {
            "handle": handle,
            "type": dxftype,
            "layer": entity.dxf.get("layer", "0"),
            "reason": reason,
            "located": False,
        }

        box = None
        try:
            box = _bbox.extents([entity], fast=True)
        except Exception as exc:  # noqa: BLE001 - outlining is best-effort
            log.debug("bbox failed for %s %s: %s", dxftype, handle, exc)

        if box is not None and box.has_data and matrix is not None:
            lo, hi = matrix.transform(box.extmin), matrix.transform(box.extmax)
            x0, x1 = sorted((lo.x, hi.x))
            y0, y1 = sorted((lo.y, hi.y))
            if x1 > x0 and y1 > y0:
                group = ET.SubElement(parent, "g")
                group.set(ATTR_HANDLE, handle)
                group.set(ATTR_TYPE, dxftype)
                group.set("data-unrenderable", "true")
                ET.SubElement(
                    group, "rect",
                    x=f"{x0:.0f}", y=f"{y0:.0f}",
                    width=f"{x1 - x0:.0f}", height=f"{y1 - y0:.0f}",
                )
                label = ET.SubElement(group, "text")
                label.set("x", f"{(x0 + x1) / 2:.0f}")
                label.set("y", f"{(y0 + y1) / 2:.0f}")
                label.set(
                    "font-size",
                    f"{max(min((x1 - x0) / 26, (y1 - y0) / 3), 1):.0f}",
                )
                label.text = reason
                record["located"] = True

        found.append(record)

    if unexpected:
        log.warning(
            "entities disappeared from the render without a known reason",
            extra={"types": dict(unexpected.most_common(10))},
        )
        found.append(
            {
                "handle": "",
                "type": "UNEXPECTED",
                "layer": "",
                "reason": "not drawn, reason unknown: "
                + ", ".join(f"{n} x {t}" for t, n in unexpected.most_common(6)),
                "located": False,
            }
        )
    return found


class HandleAwareSVGRenderBackend(SVGRenderBackend):
    """`SVGRenderBackend` that nests every path under `<g data-handle=...>`.

    Args:
        page: page definition, forwarded to the base class.
        settings: layout settings, forwarded to the base class.
        handle_types: maps entity handle -> DXF type (``"INSERT"``, ``"LINE"``
            ...). Supplied by the extraction pass, because
            `BackendProperties` carries the handle and the layer but not the
            type. A handle missing from this map still renders; it simply gets
            no ``data-type``.
    """

    def __init__(
        self,
        page: layout.Page,
        settings: layout.Settings,
        handle_types: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(page, settings)
        self._handle_types: Mapping[str, str] = handle_types or {}
        # layer name -> <g data-layer="...">
        self._layer_groups: dict[str, ET.Element] = {}
        # handle -> <g data-handle="...">. Keyed by handle alone, not by
        # (layer, handle): an entity with BYBLOCK colour can reach the backend
        # under more than one resolved layer name, and two groups carrying the
        # same handle would mean a click highlights only half the object.
        self._entity_groups: dict[str, ET.Element] = {}
        self._inject_stylesheet()

    # -- public ------------------------------------------------------------

    @property
    def rendered_handles(self) -> set[str]:
        """Handles that actually produced at least one SVG element.

        Not the same as "handles in the drawing": entities on a frozen layer,
        or with zero-extent geometry, never reach the backend. The API reports
        both numbers so a mismatch is visible instead of silent.
        """
        return {h for h in self._entity_groups if h != _UNKNOWN_HANDLE}

    @property
    def rendered_layers(self) -> set[str]:
        """Layer names that produced at least one SVG element."""
        return set(self._layer_groups)

    # -- overrides ---------------------------------------------------------

    def add_strokes(self, d: str, properties: BackendProperties) -> None:
        with self._reparent(properties):
            super().add_strokes(d, properties)

    def add_filling(self, d: str, properties: BackendProperties) -> None:
        with self._reparent(properties):
            super().add_filling(d, properties)

    # -- internals ---------------------------------------------------------

    def _inject_stylesheet(self) -> None:
        """Keep stroke width constant under zoom, for every path at once.

        `vector-effect` is not an inherited SVG property, so setting it on the
        parent `<g>` would not work; setting it on every `<path>` would bloat
        a 20k-entity document. One CSS rule costs ~60 bytes total.
        """
        style = ET.SubElement(self.root, "style")
        style.text = (
            # Constant stroke width under zoom. Without this, lines go blocky
            # as soon as the viewer zooms in.
            "path{vector-effect:non-scaling-stroke}"
            # Layer visibility: the viewer sets this attribute on one <g> per
            # layer, so hiding a layer is a single attribute write rather than
            # a walk over tens of thousands of nodes. `visibility:hidden`, not
            # `display:none`: hidden entities keep their screen geometry, so a
            # comment on a switched-off layer still has a hover zone (clicks
            # pass through either way — hit-testing skips invisible elements).
            "[data-layer-hidden='true']{visibility:hidden}"
            # Selection. A drop-shadow rather than a fill override because CAD
            # entities are a mix of stroked outlines and filled glyphs;
            # forcing `fill` would flood every open path with colour.
            "[data-selected='true']{filter:drop-shadow(0 0 3px #0284c7)"
            " drop-shadow(0 0 9px #0284c7)}"
            "[data-selected='true'] path{stroke:#0284c7!important;"
            "stroke-width:3!important}"
            # Entities carrying a comment, marked by the viewer.
            "[data-commented='true'] path{stroke:#c2410c!important;"
            "stroke-width:3!important}"
            # Foreign payloads AutoCAD can draw and a DXF renderer cannot
            # (PDF underlays, embedded OLE, 3D solids). Marked rather than
            # left as an unexplained blank rectangle.
            "[data-unrenderable] rect{fill:#f1f5f9;stroke:#94a3b8;"
            "stroke-width:2;stroke-dasharray:12 8;vector-effect:non-scaling-stroke}"
            "[data-unrenderable] text{fill:#64748b;text-anchor:middle;"
            "font-family:sans-serif}"
        )
        # <style> is appended after <defs>/<rect>; move it to the front so it
        # applies before the content is parsed.
        self.root.remove(style)
        self.root.insert(0, style)

    def _reparent(self, properties: BackendProperties) -> "_ParentSwap":
        """Point `self.entities` at this entity's group for one draw call.

        The base-class methods create their element with
        ``ET.SubElement(self.entities, "path", ...)``. Swapping `self.entities`
        for the duration of the call is what lets us reuse all of the library's
        colour and stroke-width resolution instead of copying it.
        """
        return _ParentSwap(self, self._group_for(properties))

    def _group_for(self, properties: BackendProperties) -> ET.Element:
        layer = properties.layer or "0"
        handle = properties.handle or _UNKNOWN_HANDLE

        # Handle-less geometry is bucketed per layer, not globally. With a
        # single shared bucket the first such draw call fixed the parent
        # layer, and every later handle-less path -- whatever layer it
        # belonged to -- was appended there, so hiding that one layer hid
        # unrelated geometry and a layer whose only output was handle-less
        # never appeared in the layer list at all.
        key = f"{layer}\x00{handle}" if handle == _UNKNOWN_HANDLE else handle

        group = self._entity_groups.get(key)
        if group is not None:
            return group

        layer_group = self._layer_groups.get(layer)
        if layer_group is None:
            layer_group = ET.SubElement(self.entities, "g")
            layer_group.set(ATTR_LAYER, layer)
            self._layer_groups[layer] = layer_group

        group = ET.SubElement(layer_group, "g")
        group.set(ATTR_HANDLE, handle)
        dxftype = self._handle_types.get(handle)
        if dxftype:
            group.set(ATTR_TYPE, dxftype)
        self._entity_groups[key] = group
        return group


class _ParentSwap:
    """Context manager that temporarily retargets `backend.entities`."""

    __slots__ = ("_backend", "_group", "_saved")

    def __init__(self, backend: SVGRenderBackend, group: ET.Element) -> None:
        self._backend = backend
        self._group = group
        self._saved: ET.Element | None = None

    def __enter__(self) -> ET.Element:
        self._saved = self._backend.entities
        self._backend.entities = self._group
        return self._group

    def __exit__(self, *exc_info: object) -> None:
        assert self._saved is not None
        self._backend.entities = self._saved


class HandleAwareSVGBackend(SVGBackend):
    """`SVGBackend` wired to emit `data-handle` / `data-layer` / `data-type`."""

    def __init__(self, handle_types: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self._handle_types = handle_types or {}
        self._render_backend: HandleAwareSVGRenderBackend | None = None

    def make_backend(
        self, page: layout.Page, settings: layout.Settings
    ) -> SVGRenderBackend:
        # Note: called once per get_string()/get_xml_root_element() call.
        backend = HandleAwareSVGRenderBackend(page, settings, self._handle_types)
        self._render_backend = backend
        return backend

    @property
    def render_backend(self) -> HandleAwareSVGRenderBackend | None:
        """The render backend from the most recent `get_string()` call."""
        return self._render_backend


class EmptyRenderError(RuntimeError):
    """The layout produced no drawable geometry.

    Not a failure of the renderer. A paperspace layout often holds nothing but
    a VIEWPORT frame, which the frontend does not draw, leaving the recording
    empty and its bounding box undefined. The caller should skip the layout and
    say so, rather than log a stack trace that implies something broke.
    """


@dataclass
class RenderResult:
    """Outcome of one render, with the numbers needed to trust it."""

    svg: str
    layout_name: str
    rendered_entities: int
    rendered_layers: list[str]
    elapsed_ms: int
    truncated: bool
    truncated_reason: str | None = None
    #: Foreign-payload objects present on the layout that no DXF renderer can
    #: draw (PDF underlays, embedded OLE, 3D solids). Surfaced so a blank area
    #: of the sheet is explained rather than mysterious.
    unrenderable: list[dict[str, object]] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        """Render metadata without the (potentially huge) SVG payload."""
        return {
            "layout": self.layout_name,
            "rendered_entities": self.rendered_entities,
            "rendered_layers": len(self.rendered_layers),
            "elapsed_ms": self.elapsed_ms,
            "truncated": self.truncated,
            "truncated_reason": self.truncated_reason,
            "svg_bytes": len(self.svg.encode("utf-8")),
            "unrenderable": self.unrenderable,
        }


def render_layout_svg(
    doc: Drawing,
    *,
    layout_name: str = "Model",
    handle_types: Mapping[str, str] | None = None,
    layers: Iterable[str] | None = None,
    max_entities: int | None = None,
    coordinate_space: int = DEFAULT_COORDINATE_SPACE,
) -> RenderResult:
    """Render one layout to SVG with a `data-handle` on every entity group.

    Args:
        doc: an open ezdxf document.
        layout_name: layout to render, e.g. ``"Model"`` or ``"Sheet-1"``.
        handle_types: handle -> DXF type map, used to fill ``data-type``.
        layers: if given, only entities on these layers are drawn. This is the
            cheapest of the size-reduction levers, applied before rendering.
        max_entities: refuse to render more than this many entities and say so
            in the result, instead of hanging the request. ``None`` = no cap.
        coordinate_space: resolution of the SVG coordinate system. See
            `DEFAULT_COORDINATE_SPACE` — do not lower this casually.

    Raises:
        KeyError: if `layout_name` does not exist in the document.
    """
    started = time.perf_counter()

    # A "layout" here may be a real layout or a block definition surfaced as a
    # pseudo-layout (see extract.BLOCK_LAYOUT_PREFIX). Several files keep all
    # of their content inside a block definition and would otherwise be
    # undrawable; the entities in a block *definition* carry real handles, so
    # they stay clickable.
    if is_block_layout(layout_name):
        target = doc.blocks.get(block_name_of(layout_name))
        if target is None:
            raise KeyError(f"no block definition named {block_name_of(layout_name)!r}")
    else:
        target = doc.layouts.get(layout_name)

    entities = list(target)
    if layers is not None:
        wanted = set(layers)
        entities = [e for e in entities if e.dxf.get("layer", "0") in wanted]

    truncated = False
    truncated_reason: str | None = None
    if max_entities is not None and len(entities) > max_entities:
        truncated = True
        truncated_reason = (
            f"layout has {len(entities)} entities, rendering the first "
            f"{max_entities}; narrow by layer or raise CAD_RENDER_MAX_ENTITIES"
        )
        entities = entities[:max_entities]

    backend = HandleAwareSVGBackend(handle_types=handle_types)
    context = RenderContext(doc)

    # Layers switched OFF or frozen in the file are rendered anyway on
    # model-like layouts, then hidden by default in the viewer (the SVG root
    # carries their names in `data-off-layers`). Two reasons, both measured
    # on Janadriyah: the reviewer's comment sits on C-PROP-BlockNumber, which
    # the file keeps OFF — skipping the layer made the comment unhoverable —
    # and AutoCAD's own layer panel lets those layers be switched on, so the
    # viewer's should too. Paper-space sheets are left exactly as plotted.
    off_layers: list[str] = []
    if layout_name == "Model" or is_block_layout(layout_name):
        for table_layer in doc.layers:
            if table_layer.is_frozen() or not table_layer.is_on():
                name = table_layer.dxf.name
                off_layers.append(name)
                props = context.layers.get(name.lower())
                if props is not None:
                    props.is_visible = True

    # Render onto a white sheet, and tell the context so before drawing.
    #
    # This is not a theme choice, it decides whether the drawing is visible at
    # all. CAD colours are picked against the background the drawing is meant
    # to be seen on, and these files disagree with each other:
    #   - architectural_example-imperial draws in #4c0000 (dark maroon), which
    #     on ezdxf's default black background is invisible. Five of the sixteen
    #     drawings were black-on-black.
    #   - title_block-arch draws in ACI 7, which ezdxf resolves to *white* when
    #     it believes the background is dark - invisible the other way round.
    #
    # `set_colors("#ffffff")` fixes both at once: it paints the sheet white and
    # re-resolves ACI 7 to black, matching a plotted sheet and Autodesk's own
    # viewer. Setting only the backend's background rect would have fixed the
    # first case and broken the second.
    # Two separate things, and both are needed:
    #   set_colors()     -> how ACI 7 ("BYLAYER white/black") resolves
    #   set_background() -> the colour of the sheet rectangle itself
    # `Frontend.draw_entities()` sets neither, so the Recorder keeps its
    # default black background; setting only one of the two leaves either
    # dark-on-black or white-on-white.
    context.current_layout_properties.set_colors(WHITE_SHEET)
    backend.set_background(WHITE_SHEET)
    # RELATIVE, not RELATIVE_FIXED: it maps each entity's real lineweight into
    # the 1..MAX_STROKE_PX band, so a heavy boundary still reads as heavier
    # than a hairline. RELATIVE_FIXED collapses every line to one width.
    config = Configuration(
        lineweight_policy=LineweightPolicy.RELATIVE,
        min_lineweight=0.05,
    )
    frontend = Frontend(context, backend, config=config)
    frontend.draw_entities(entities)

    # Page(0, 0) means "fit the page to the content bounding box". The bbox is
    # computed from the recorded geometry, so the $EXTMIN/$EXTMAX 1e+20
    # sentinel problem cannot reach the viewBox. The y-flip (CAD is y-up, SVG
    # is y-down) is applied by SVGBackend itself.
    page = layout.Page(0, 0)
    settings = layout.Settings(
        output_coordinate_space=coordinate_space,
        **_stroke_settings(coordinate_space),
    )
    try:
        # get_xml_root_element(), not get_string(): the placeholders below need
        # `backend.transformation_matrix`, which is only set once the content
        # has been fitted to the page.
        root = backend.get_xml_root_element(page, settings=settings)
    except ValueError as exc:
        # ezdxf raises ValueError("empty bounding box") when nothing drawable
        # was recorded. Distinguish that expected case from a real fault.
        if "empty bounding box" in str(exc):
            raise EmptyRenderError(
                f"layout {layout_name!r} contains {len(entities)} entities but "
                "none of them produce drawable geometry"
            ) from exc
        raise

    # ezdxf returns a bare, empty <svg> element -- no error -- when the fitted
    # page has zero width or height. That happens whenever the content has no
    # extent in one axis (a single horizontal line, a row of collinear
    # points). Left alone it would be cached as a blank sheet that looks like
    # a broken render, so it is reported as the same "nothing to draw" case as
    # an empty layout.
    if root.find("{*}rect") is None and root.find("rect") is None:
        raise EmptyRenderError(
            f"layout {layout_name!r} produced an empty page: its {len(entities)} "
            "entities have no extent in at least one axis"
        )

    if off_layers:
        # The viewer reads this to start those layers hidden (matching how
        # the file looks in AutoCAD) while keeping them toggleable/hoverable.
        root.set("data-off-layers", ",".join(sorted(off_layers)))

    _tag_world_transform(root, backend.transformation_matrix)

    render_backend = backend.render_backend
    rendered_now = render_backend.rendered_handles if render_backend else set()

    unrenderable: list[dict[str, object]] = []
    if render_backend is not None and backend.transformation_matrix is not None:
        unrenderable = _mark_unrenderable(
            render_backend.entities,
            entities,
            backend.transformation_matrix,
            rendered_now,
        )
        if unrenderable:
            log.info(
                "marked unrenderable objects",
                extra={"layout": layout_name, "count": len(unrenderable)},
            )

    svg = ET.tostring(root, encoding="unicode")
    rendered_handles = render_backend.rendered_handles if render_backend else set()
    rendered_layers = sorted(render_backend.rendered_layers) if render_backend else []

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "rendered layout",
        extra={
            "layout": layout_name,
            "entities_in": len(entities),
            "handles_out": len(rendered_handles),
            "svg_bytes": len(svg),
            "elapsed_ms": elapsed_ms,
        },
    )
    return RenderResult(
        svg=svg,
        layout_name=layout_name,
        rendered_entities=len(rendered_handles),
        rendered_layers=rendered_layers,
        elapsed_ms=elapsed_ms,
        truncated=truncated,
        truncated_reason=truncated_reason,
        unrenderable=unrenderable,
    )
