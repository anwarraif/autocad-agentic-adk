"""DXF -> canonical JSON.

Produces the two document shapes that everything downstream depends on:
a `DrawingDoc` (one per file) and a stream of `EntityDoc` (one per top-level
entity), matching the schema in the `cad-graph-model` skill.

Three decisions are baked in here, all of them load-bearing:

1. **Every layout is walked, not just modelspace.** Four of the sample files
   have zero entities in modelspace and their whole content in a paperspace
   layout or in a block definition. Iterating `doc.modelspace()` alone would
   report them as empty files.

2. **Extents are computed, never read from the header.** `$EXTMIN`/`$EXTMAX`
   are only refreshed by AutoCAD on regen and routinely hold the sentinel
   `1e+20`. Feeding that to an SVG viewBox makes the drawing vanish.

3. **"Structure" means top-level entity.** `insert.virtual_entities()` returns
   copies with `handle = None`, so geometry inside a block has no stable
   identity. One INSERT = one handle = one commentable object. Block *contents*
   are still catalogued, under `blocks`, so "what door types exist" is
   answerable.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator, Mapping

import ezdxf
from ezdxf import bbox, recover
from ezdxf.document import Drawing
from ezdxf.entities import DXFGraphic

from . import geometry, region

log = logging.getLogger(__name__)

# Bump when the extraction logic changes in a way that makes stored documents
# stale. Ingest compares this against the stored value and re-ingests.
#: v8: an ARC keeps its centre, radius and angles and is flattened to points,
#: so curves stop drawing as straight chords -- measured, a 15.708 m arc drawn
#: as a 14.142 m line, on a drawing holding 49,167 arcs. A closed polyline
#: containing a bulge is flattened the same way instead of being refused, so
#: 1,291 shapes that were unreachable become real boundaries. Both ride ONE
#: bump, as docs/INTAKE-TO-AGENT-PLAN.md section 9 requires.
#:
#: v7: block placement transforms, so block-resident geometry can be placed by
#: its INSERT chain rather than guessed at from a layout name.
INGEST_VERSION = 8

# $INSUNITS codes. 0 means the file does not declare its units -- that is
# information, not a default, so it is reported as such rather than assumed.
UNIT_NAMES: dict[int, str] = {
    0: "unitless",
    1: "inch",
    2: "feet",
    3: "mile",
    4: "mm",
    5: "cm",
    6: "m",
    7: "km",
    8: "microinch",
    9: "mil",
    10: "yard",
    11: "angstrom",
    12: "nanometer",
    13: "micron",
    14: "dm",
    15: "dam",
    16: "hm",
    17: "gm",
    18: "au",
    19: "lightyear",
    20: "parsec",
}

# Text longer than this is stored in full but truncated in list responses.
TEXT_PREVIEW_CHARS = 80

# ezdxf's bbox for a single entity can be surprisingly costly on hatches and
# splines; a shared cache makes the whole-file pass roughly linear.
_BBOX_FAST = True


class ExtractionError(RuntimeError):
    """Raised when a DXF cannot be opened at all."""


@dataclass
class DrawingDoc:
    """One document per CAD file -- the `autocad_drawings` shape."""

    _id: str
    source_path: str
    original_filename: str
    dxf_version: str
    acad_release: str
    units_code: int
    units_name: str
    extents: dict[str, list[float]] | None
    layers: list[dict[str, Any]]
    blocks: list[str]
    layouts: list[dict[str, Any]]
    #: The drawing's dictionary tables (UPLIFT-11). They live on DrawingDoc,
    #: not on EntityDoc, because these are ONE-file settings: 23 text styles
    #: are paid for once, not 46,754 times.
    text_styles: list[dict[str, Any]]
    linetypes: list[dict[str, Any]]
    dim_styles: list[dict[str, Any]]
    header: dict[str, Any]
    entity_count: int
    counts_by_type: dict[str, int]
    counts_by_layer: dict[str, int]
    xrefs: list[dict[str, Any]]
    file_bytes: int
    audit_errors: int
    audit_fixes: int
    extract_ms: int
    ingest_version: int = INGEST_VERSION
    warnings: list[str] = field(default_factory=list)
    #: What the FILE states about its coordinate system, or None if it states
    #: nothing. See `_declared_crs`. Defaulted so that a drawing document
    #: written before this field existed reads back as "stated nothing"
    #: rather than as a missing key nobody handles.
    declared_crs: dict[str, Any] | None = None
    #: How model space places each block definition, keyed by block name.
    #: `{"identity": bool, "placements": int, "ambiguous": bool, ...}`. This is
    #: what decides whether a block's stored coordinates are also world
    #: coordinates; see `block_placements`.
    block_placement: dict[str, Any] = field(default_factory=dict)

    def to_mongo(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


#: Types that can have a ring. Every ring field is written only for these.
RING_TYPES = frozenset({"LWPOLYLINE", "POLYLINE"})

#: Types that have an insertion point -- the point tested when matching a label
#: to a parcel. The centre of the bounding box may NEVER stand in for it.
ANCHOR_TYPES = frozenset(
    {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "MULTILEADER", "MLEADER", "INSERT"}
)

#: Types that can be a PATH -- something with a run, and therefore possibly two
#: free ends. Every one of them carries the `ends` block below; a TEXT does not,
#: for the same reason it carries no `ring_status` (see `EntityDoc`).
#:
#: `CIRCLE` is in the list although a circle never has free ends: the whole
#: point of the block is that "this is a closed curve, it HAS no ends" is an
#: answer, and it must be distinguishable from "nobody looked". `ELLIPSE` and
#: `SPLINE` are in it for the same reason -- they arrive as `not_derived` with
#: the reason attached rather than being silently missing.
PATH_TYPES = frozenset(
    {"LINE", "ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE", "ELLIPSE", "SPLINE"}
)

#: `ends_status` vocabulary. A closed list, so that a consumer can branch on it
#: and a new state cannot appear unannounced.
ENDS_OPEN: str = "open_path"
ENDS_CLOSED: str = "closed_ring"
ENDS_NOT_DERIVED: str = "not_derived"
ENDS_STATUSES: tuple[str, ...] = (ENDS_OPEN, ENDS_CLOSED, ENDS_NOT_DERIVED)

_RING_KEYS = (
    "ring_status",
    "ring",
    "ring_diagnostic",
    "ring_vertex_count",
    "ring_dropped_duplicates",
    "ring_frame",
    "ring_origin",
    "ring_orientation",
    "extrusion_non_standard",
    "polygon_centroid",
    "centroid_inside_ring",
    "edge_lengths",
    "perimeter_from_ring",
    "shape_key",
    "shape_key_basis",
    "ring_simple",
    "geometry_note",
    "ring_arc_flattened",
    "ring_arc_tolerance",
)

#: Types that carry a run of points WITHOUT being a polyline. An ARC is the
#: whole reason this exists: it is a curve, it is not a RING_TYPE, and while
#: the path block was written for RING_TYPES only it was stored as its two
#: endpoints and drawn as a straight chord.
CURVE_TYPES = frozenset({"ARC"})

#: The open-path shape block, written for RING_TYPES and CURVE_TYPES alike.
_PATH_BLOCK_KEYS = (
    "path_points",
    "path_bulges",
    "path_points_basis",
    "arc",
)

#: Popped INDIVIDUALLY when they have nothing to say, while
#: `path_points_basis` stays behind to say why they are absent.
_PATH_POINT_KEYS = ("path_points", "path_bulges", "arc")

_ANCHOR_KEYS = ("anchor_point", "anchor_basis", "text_style")

_ENDS_KEYS = ("ends", "ends_status", "ends_basis")


@dataclass
class EntityDoc:
    """One document per top-level entity -- the `autocad_entities` shape.

    The geometry block is written ONLY for `RING_TYPES`, the anchor block ONLY
    for `ANCHOR_TYPES`, and the endpoint block ONLY for `PATH_TYPES`. On other
    types those keys are absent altogether -- not
    null-valued. Two reasons, both real: writing empty field names into every
    document costs ~81 B x 46,754 documents = 3.8 MB, nearly five times the
    ring data itself; and in meaning, a TEXT does not have `ring_status` as a
    category. "The key is not there" says that, while `null` claims "it was
    checked and the result was empty".
    """

    _id: str
    drawing_id: str
    handle: str
    type: str
    layer: str
    layout: str
    block_name: str | None
    text: str | None
    attribs: dict[str, str] | None
    bbox: dict[str, list[float]] | None
    bbox_centre: list[float] | None
    length: float | None
    area: float | None

    # --- geometry block, RING_TYPES only ---------------------------------
    ring_status: str | None = None
    ring: list[list[float]] | None = None
    ring_diagnostic: list[list[float]] | None = None
    ring_vertex_count: int | None = None
    ring_dropped_duplicates: int | None = None
    ring_frame: str | None = None
    ring_origin: list[float] | None = None
    ring_orientation: str | None = None
    extrusion_non_standard: bool | None = None
    polygon_centroid: list[float] | None = None
    centroid_inside_ring: bool | None = None
    edge_lengths: list[float] | None = None
    perimeter_from_ring: float | None = None
    shape_key: str | None = None
    shape_key_basis: str | None = None
    ring_simple: bool | None = None
    geometry_note: str | None = None

    # --- open-path shape, OPEN polylines only -----------------------------
    #: The vertices of an OPEN polyline, `[[x, y], ...]`, in WCS.
    #:
    #: `ring` is the closed counterpart of this field and has existed all
    #: along; an OPEN polyline had nowhere to put its vertices, so its shape
    #: was stored as a bounding box and nothing else. The consequence was
    #: measured before this field was added: on the reference drawing's road
    #: layer, 83 of 561 junctions could not be given a category, because the
    #: only thing known about a curved open polyline was that its BOX reached
    #: the junction. With this field the same census decides every node.
    #:
    #: Written for open polylines ONLY. A closed one already carries `ring`,
    #: and duplicating it here would give two answers to one question.
    path_points: list[list[float]] | None = None
    #: `bulges[i]` is the arc-ness of the span from `path_points[i]` to
    #: `path_points[i+1]`: zero is straight, otherwise `tan(quarter of the
    #: included angle)` of a circular arc. ABSENT when every span is straight,
    #: which is the common case and keeps the field off most documents.
    #:
    #: Without it a consumer measuring against `path_points` would be measuring
    #: against the CHORDS of the arcs, silently, and reporting a distance to a
    #: line the drawing does not contain.
    path_bulges: list[float] | None = None
    #: How the vertices were derived, or why they are absent. A phrase, for the
    #: same reason `ends_basis` is one.
    path_points_basis: str | None = None
    #: An ARC's own parameters -- centre, radius, both angles in radians.
    #: `path_points` carries the same curve flattened to within
    #: `geometry.ARC_CHORD_TOLERANCE`; this is kept so a consumer wanting the
    #: exact curve is not left re-deriving it from the flattening.
    arc: dict[str, Any] | None = None
    #: Set on a closed polyline whose bulged spans were expanded into the arcs
    #: they describe. Its `ring` IS the boundary, not the chords of one.
    ring_arc_flattened: bool | None = None
    ring_arc_tolerance: float | None = None

    # --- anchor block, ANCHOR_TYPES only ---------------------------------
    anchor_point: list[float] | None = None
    anchor_basis: str | None = None
    #: This entity's text style name. Without this field `UPLIFT-04` cannot be
    #: done at all: an SHX reading is only valid for text that was DRAWN in
    #: that font, and the only way to apply it without knowing the style is to
    #: apply it to all text -- which is guessing. The font file itself is
    #: missing and cannot be guessed; what this opens up is the ability to name
    #: which text is affected, so the hole is counted instead of estimated.
    text_style: str | None = None

    # --- open-path endpoints block, PATH_TYPES only ----------------------
    #: The two free ends of an open path, `[[x, y], [x, y]]`, in WCS.
    #:
    #: Until this field existed the store held a bounding box for every LINE and
    #: ARC and nothing else, so no question about how paths MEET could be
    #: answered from the store at all: the chain count in `dossier_network` had
    #: to guess endpoints from box corners and label itself `approximate`, and
    #: `recipes.topology.frontage_check` had to measure a plot's distance to a
    #: road's BOX. Two points per path close both holes.
    #:
    #: The field is ABSENT -- popped, not null -- whenever the two ends could
    #: not be derived. `[0, 0]` would be a coordinate nobody measured, and on a
    #: projected drawing it is 2.7 million units from the geometry.
    ends: list[list[float]] | None = None
    #: `open_path`, `closed_ring`, or `not_derived`. This is the field a
    #: consumer branches on, and it carries one fact nothing else in the
    #: document does: whether a polyline is CLOSED. `ring_status` cannot say so
    #: -- a closed polyline with an arc segment is reported as `bulge`, exactly
    #: like an open one, because the bulge gate fires first.
    ends_status: str | None = None
    #: How the two points were derived, or why there are none.
    #:
    #: Deliberately TERSE -- a phrase, not a sentence. Janadriyah alone holds
    #: 18,074 entities of a path type, so every extra character here is paid
    #: 18,074 times: the paragraph this phrase abbreviates would add ~2.9 MB of
    #: prose to one drawing's documents, which is more than the whole ring
    #: store. The reasoning lives once, in `_ends_fields` and in
    #: `recipes.junctions`; what travels with each row is the label.
    ends_basis: str | None = None

    def to_mongo(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items()}
        if self.type not in RING_TYPES:
            for k in _RING_KEYS:
                out.pop(k, None)
        if self.type not in RING_TYPES and self.type not in CURVE_TYPES:
            for k in _PATH_BLOCK_KEYS:
                out.pop(k, None)
        if self.type not in ANCHOR_TYPES:
            for k in _ANCHOR_KEYS:
                out.pop(k, None)
        for k in _PATH_POINT_KEYS:
            if out.get(k) is None:
                out.pop(k, None)
        if self.type not in PATH_TYPES:
            for k in _ENDS_KEYS:
                out.pop(k, None)
        elif out.get("ends") is None:
            # Absent, not null: a closed ring HAS no ends and a spline's ends
            # were not derived, and neither of those is "two points that came
            # out empty". `ends_status` and `ends_basis` stay, so the reason
            # travels with the absence.
            out.pop("ends", None)
        return out


def compute_drawing_id(path: Path) -> str:
    """`sha256(file content)[:16]`.

    Content-addressed, not name-addressed: filenames change and content does
    not, re-ingesting the same file is a no-op, and two revisions of the same
    drawing get distinct ids automatically.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def load_document(path: Path) -> tuple[Drawing, int, int]:
    """Open a DXF, recovering from the damage typical of converted files.

    Returns the document plus the audit error and fix counts, so a caller can
    report "opened, with 261 recovered errors" instead of pretending the file
    was clean.
    """
    try:
        doc, auditor = recover.readfile(str(path))
    except IOError as exc:
        raise ExtractionError(f"cannot read {path.name}: {exc}") from exc
    except ezdxf.DXFStructureError as exc:
        raise ExtractionError(f"{path.name} is not a valid DXF: {exc}") from exc
    return doc, len(auditor.errors), len(auditor.fixes)


def _text_of(entity: DXFGraphic) -> str | None:
    """Pull the human-readable text out of an entity, or None.

    MTEXT needs `plain_text()`: its raw `.text` is interleaved with formatting
    codes (`\\pxq`, `\\f`) that would poison the text index.
    """
    dxftype = entity.dxftype()
    try:
        if dxftype == "MTEXT":
            return entity.plain_text() or None
        if dxftype in ("TEXT", "ATTRIB", "ATTDEF"):
            return entity.dxf.get("text") or None
        if dxftype.startswith("DIMENSION") or dxftype == "DIMENSION":
            # An empty DIMENSION text means "show the measured value", which
            # is not a string we have; report it as absent rather than "".
            return entity.dxf.get("text") or None
        if dxftype in ("MULTILEADER", "MLEADER"):
            ctx = getattr(entity, "context", None)
            mtext = getattr(ctx, "mtext", None) if ctx else None
            return getattr(mtext, "default_content", None) or None
    except (AttributeError, ValueError) as exc:
        log.debug("text extraction failed for %s: %s", dxftype, exc)
    return None


def _attribs_of(entity: DXFGraphic) -> dict[str, str] | None:
    """ATTRIB tag -> value for a block reference.

    This is where title-block data, room numbers and equipment tags live, so
    it is worth more than the geometry for answering real questions.
    """
    if entity.dxftype() != "INSERT":
        return None
    try:
        attribs = {
            att.dxf.tag: att.dxf.text
            for att in entity.attribs
            if att.dxf.get("tag")
        }
    except (AttributeError, TypeError) as exc:
        log.debug("attrib extraction failed: %s", exc)
        return None
    return attribs or None


def _finite(value: float) -> bool:
    """Reject the 1e+20 sentinel and NaN before they reach the database."""
    return math.isfinite(value) and abs(value) < 1e19


def _entity_bbox(
    entity: DXFGraphic, cache: bbox.Cache
) -> tuple[dict[str, list[float]] | None, list[float] | None]:
    """Compute one entity's bounding box and its centre, or (None, None)."""
    try:
        box = bbox.extents([entity], fast=_BBOX_FAST, cache=cache)
    except Exception as exc:  # ezdxf raises a wide range here on odd geometry
        log.debug("bbox failed for handle %s: %s", entity.dxf.get("handle"), exc)
        return None, None
    if not box.has_data:
        return None, None
    lo, hi = box.extmin, box.extmax
    if not all(_finite(v) for v in (lo.x, lo.y, hi.x, hi.y)):
        return None, None
    return (
        {"min": [lo.x, lo.y], "max": [hi.x, hi.y]},
        [(lo.x + hi.x) / 2.0, (lo.y + hi.y) / 2.0],
    )


def _extrusion_non_standard(entity: DXFGraphic) -> bool:
    """True when the extrusion vector is not (0,0,1).

    This is the gate for the most perfectly silent defect in this whole spec.
    LWPOLYLINE vertices are stored in the Object Coordinate System;
    `bbox.extents` and text insertion points are in World. For a polygon that
    has been mirrored the extrusion is often (0,0,-1), which FLIPS the x axis.
    An OCS ring pitted against a WCS test point answers "zero labels inside"
    with confidence -- indistinguishable from "this parcel genuinely has no
    label". And length and area do not shift in the slightest under extrusion,
    so not one old number moves to tell you.
    """
    try:
        e = entity.dxf.extrusion
        return not (
            abs(float(e[0])) < 1e-9
            and abs(float(e[1])) < 1e-9
            and abs(float(e[2]) - 1.0) < 1e-9
        )
    except Exception:
        return False


def _raw_ring(
    entity: DXFGraphic,
) -> tuple[list[tuple[float, float]], bool, bool, bool, list[float]] | None:
    """(WCS points, closed, has bulge, non-planar, per-span bulges), or None.

    `bulges[i]` belongs to the span from `points[i]` to `points[i+1]`, and it is
    `tan(quarter of the included angle)` of a circular arc: zero is a straight
    span. It is returned alongside the points rather than reduced to the
    `has_bulge` flag because a bulged span's two vertices are the CHORD of an
    arc, not the arc, and a consumer that is handed the vertices without the
    bulges cannot tell those two apart -- it would measure a distance to a
    straight line that the drawing does not contain.

    The bulge sign is FLIPPED under a non-standard extrusion, and that is not
    cosmetic. Mirroring reverses handedness, so an arc that curves left of its
    chord in OCS curves right of it in WCS. Lifting the points and leaving the
    bulges alone would put every arc on the wrong side of its own chord -- by
    twice the sagitta, silently, on exactly the entities the OCS lift exists to
    get right.
    """
    dxftype = entity.dxftype()
    try:
        if dxftype == "LWPOLYLINE":
            raw = list(entity.get_points("xyseb"))
            bulges = [float(p[4]) for p in raw]
            pts2 = [(float(p[0]), float(p[1])) for p in raw]
            closed = bool(entity.closed)
            non_planar = False
        elif dxftype == "POLYLINE":
            verts = list(entity.vertices)
            bulges = [float(v.dxf.get("bulge", 0.0) or 0.0) for v in verts]
            locs = [v.dxf.location for v in verts]
            pts2 = [(float(v.x), float(v.y)) for v in locs]
            zs = [float(v.z) for v in locs]
            non_planar = bool(zs) and (max(zs) - min(zs)) > 1e-9
            closed = bool(entity.is_closed)
        else:
            return None
        has_bulge = any(abs(b) > 1e-9 for b in bulges)
    except Exception as exc:
        log.debug("ring read failed for %s: %s", dxftype, exc)
        return None

    if _extrusion_non_standard(entity):
        try:
            ocs = entity.ocs()
            elevation = float(entity.dxf.get("elevation", 0.0) or 0.0)
            pts2 = [
                (float(w.x), float(w.y))
                for w in (ocs.to_wcs((x, y, elevation)) for x, y in pts2)
            ]
            bulges = [-b for b in bulges]
        except Exception as exc:
            log.debug("OCS to WCS failed for %s: %s", dxftype, exc)
            return None
    return pts2, closed, has_bulge, non_planar, bulges


def _text_style_of(entity: DXFGraphic) -> str | None:
    """This entity's text style name, or None when it has none.

    MTEXT uses `style`, TEXT and ATTRIB use `style` as well, but some
    converted files write it empty instead of dropping the attribute. Empty
    and absent are different things and both are returned as None here,
    because both mean the same thing to the consumer: this entity's style is
    unknown, so no reading may be applied to it.
    """
    try:
        value = entity.dxf.get("style", None)
    except Exception:
        return None
    if value is None:
        return None
    name = str(value).strip()
    return name or None


def _ring_fields(entity: DXFGraphic) -> dict[str, Any]:
    """The whole geometry block for one ring-bearing entity.

    One gate, `ring_status`, and `ring` is None for every value other than
    "complete". This shape replaces the two booleans previously used to filter
    point-in-polygon targets: a filter shaped as the ABSENCE of two flags let
    bulged polygons and open polygons through, and both answer "inside" with
    confidence for a shape that is not the real boundary. In the reference
    drawing there are 320 bulged polygons on one layer alone, so this is not
    an edge case.
    """
    out: dict[str, Any] = {
        "ring_frame": "wcs",
        "extrusion_non_standard": _extrusion_non_standard(entity),
        "ring_dropped_duplicates": 0,
    }
    read = _raw_ring(entity)
    if read is None:
        out.update(
            ring_status="unreadable",
            ring_vertex_count=0,
            geometry_note="the vertices could not be read from the file",
        )
        return out

    pts, closed, has_bulge, non_planar, _bulges = read
    clean, dropped = geometry.dedupe(pts)
    out["ring_dropped_duplicates"] = dropped
    out["ring_vertex_count"] = len(clean)

    def bail(status: str, note: str, keep_diagnostic: bool = False) -> dict[str, Any]:
        out["ring_status"] = status
        out["geometry_note"] = note
        if keep_diagnostic and clean:
            out["ring_diagnostic"] = [[x, y] for x, y in geometry.truncate(clean)]
        return out

    if has_bulge and closed:
        # Only a CLOSED shape is flattened here. An open polyline carrying a
        # bulge is not a ring and never becomes one -- it falls through to the
        # `open` bail below, and its curve is kept properly by
        # `_path_point_fields` as `path_points` plus `path_bulges`. Flattening
        # it here computed an arc and threw it away, and worse, stamped
        # `ring_arc_flattened` on a row that got no ring: measured on Sedra,
        # 23,180 of the 26,069 rows carrying that flag were `open`. A flag
        # that claims a ring was made where none was is the same class of
        # defect as a count that disagrees with the store.
        #
        # FLATTENED, not refused. The refusal was right about the reason --
        # the chords are not the boundary and answering "inside" from them is
        # wrong -- so the fix is to remove the reason rather than relax the
        # gate: the bulged spans become the arcs they describe, and what is
        # stored IS the boundary, to within `ARC_CHORD_TOLERANCE`.
        curved = geometry.flatten_bulges(pts, _bulges, closed)
        curved, dropped = geometry.dedupe(curved)
        if len(curved) > geometry.MAX_RING_VERTICES:
            # Over the ceiling the old refusal still stands, and for the old
            # reason. A truncated ring is a different ring.
            return bail(
                "bulge",
                f"the polygon contains an arc; flattening it needs "
                f"{len(curved)} vertices, over the "
                f"{geometry.MAX_RING_VERTICES} limit, so the chords are all "
                "that is left and they are not the true boundary",
                keep_diagnostic=True,
            )
        clean = curved
        out["ring_dropped_duplicates"] = dropped
        out["ring_vertex_count"] = len(clean)
        out["ring_arc_flattened"] = True
        out["ring_arc_tolerance"] = geometry.ARC_CHORD_TOLERANCE
    if non_planar:
        return bail(
            "non_planar",
            "a POLYLINE with a varying Z; its projection onto the XY plane can "
            "intersect itself",
            keep_diagnostic=True,
        )
    if not closed:
        return bail(
            "open",
            "the polygon is open; ray casting would close it implicitly and "
            "answer inside for a shape that has no inside",
        )
    if len(clean) < 3:
        return bail("degenerate", "fewer than three distinct vertices")
    if len(clean) > geometry.MAX_RING_VERTICES:
        return bail(
            "oversize",
            f"{len(clean)} vertices exceed the {geometry.MAX_RING_VERTICES} "
            "limit; what is stored is for diagnosis only and must not be "
            "measured",
            keep_diagnostic=True,
        )

    origin = geometry.ring_origin(clean)
    signed = geometry.signed_area(clean, origin)
    if signed == 0.0:
        return bail(
            "degenerate",
            "the signed area is zero; the polygon doubles back on itself, so "
            "there is no centroid to report",
        )

    centroid = geometry.polygon_centroid(clean, origin)
    lens = geometry.edge_lengths(clean, closed=True)

    out.update(
        ring_status="complete",
        ring=[[x, y] for x, y in clean],
        ring_origin=[origin[0], origin[1]],
        ring_orientation="ccw" if signed > 0 else "cw",
        polygon_centroid=[centroid[0], centroid[1]] if centroid else None,
        centroid_inside_ring=(
            region.point_in_ring(centroid[0], centroid[1], clean, origin=origin)
            != "outside"
            if centroid
            else None
        ),
        edge_lengths=[round(v, 6) for v in lens],
        perimeter_from_ring=sum(lens),
        shape_key=geometry.shape_key(clean),
        shape_key_basis=geometry.shape_key_basis(),
        ring_simple=geometry.is_simple(clean),
        geometry_note=None,
    )
    return out


def _anchor_fields(entity: DXFGraphic) -> dict[str, Any]:
    """The insertion point used to match a label to a parcel, and its origin.

    For TEXT with an alignment other than bottom-left, AutoCAD IGNORES
    `dxf.insert` and uses `dxf.align_point`. A plot number sitting neatly in
    the middle of its plot is almost certainly centre-aligned, so using
    `insert` shifts the test point by half the text width -- the result is not
    zero out of 2,380 but about 2,340 out of 2,380, and someone will tune the
    algorithm instead of fixing the point.

    The centre of the bounding box is NEVER used as a fallback. If no
    insertion point can be derived, the result is None and that entity is
    reported as skipped -- not tested at the wrong point while claiming the
    right one.
    """
    dxftype = entity.dxftype()
    try:
        if dxftype == "TEXT":
            halign = int(entity.dxf.get("halign", 0) or 0)
            valign = int(entity.dxf.get("valign", 0) or 0)
            if halign or valign:
                point = entity.dxf.get("align_point", None)
                basis = "TEXT.align_point"
                if point is None:
                    point = entity.dxf.get("insert", None)
                    basis = "TEXT.insert"
            else:
                point = entity.dxf.get("insert", None)
                basis = "TEXT.insert"
            if point is None:
                return {"anchor_point": None, "anchor_basis": None}
            return {
                "anchor_point": [float(point[0]), float(point[1])],
                "anchor_basis": basis,
            }
        for attr in ("insert", "location"):
            value = entity.dxf.get(attr, None)
            if value is not None:
                return {
                    "anchor_point": [float(value[0]), float(value[1])],
                    "anchor_basis": f"{dxftype}.{attr}",
                }
    except Exception as exc:
        log.debug("anchor extraction failed for %s: %s", dxftype, exc)
    return {"anchor_point": None, "anchor_basis": None}


def _path_point_fields(entity: DXFGraphic) -> dict[str, Any]:
    """The vertices of an OPEN polyline, with the arc-ness of each span.

    The second half of DOSSIER Phase 5c's data-layer change, and the reason it
    is a second half is that it was DECIDED with a number in front of it rather
    than assumed. Storing the two ends made the junction NODES exact and left
    the interiors of curves unknown: on the reference drawing's road layer that
    was 83 of 561 junctions that could only be given a range, every one of them
    because a curved open polyline's bounding box reached the junction and
    nothing stored said where the polyline itself went. The whole store holds
    20,518 open-polyline vertices, so closing that hole costs well under a
    megabyte -- and the same re-ingest carries it.

    Bulges travel WITH the points, never separately and never dropped. Two
    vertices of a bulged span are the CHORD of an arc; handing a consumer the
    chord and calling it the path would move the geometry by up to the sagitta
    -- on a road fillet, metres -- with nothing in any number to show it. Where
    every span is straight, `path_bulges` is absent, which is both smaller and
    truthful: there is no arc-ness to record.

    Closed polylines are NOT written here. A closed one with a clean ring
    already carries `ring`, and a second copy of it would be a second answer to
    one question; a closed one with an arc segment carries neither, which is a
    named hole rather than a silent one -- `recipes.junctions` counts it and
    bounds it by the bounding box.
    """
    out: dict[str, Any] = {
        "path_points": None,
        "path_bulges": None,
        "path_points_basis": None,
        #: The ARC's own parameters, kept so a consumer that wants the exact
        #: curve is not left re-deriving it from the flattened points.
        "arc": None,
    }

    def basis(reason: str) -> dict[str, Any]:
        """Record how the points were derived, or why there are none."""
        out["path_points_basis"] = reason
        return out

    # An ARC is a curve, and it was stored as its two endpoints -- so every
    # curve in the store DREW as a straight chord. Measured on the reference
    # case, a 15.708 m arc drawn as a 14.142 m line; Sedra holds 49,167 arcs,
    # so on that drawing it is what its curved roads and boundaries look like.
    # The extractor already has the centre, radius and both angles in hand
    # here, so this keeps them rather than throwing them away.
    if entity.dxftype() == "ARC":
        angles = _arc_end_angles(entity)
        if angles is None:
            return basis("ARC sweeps a full turn: a circle, drawn from `ends`")
        centre = entity.dxf.center
        radius = float(entity.dxf.radius)
        cz = float(centre.z)
        start, end = angles
        pts = geometry.arc_points((float(centre.x), float(centre.y)), radius, start, end)
        if _extrusion_non_standard(entity):
            lifted = _ocs_to_wcs(entity, pts, cz)
            if lifted is None:
                # Same rule as `ends`: an OCS point published as a WCS one is
                # wrong by the width of the drawing, so none is published.
                return basis("ARC extrusion: OCS not liftable to WCS")
            pts = lifted
        out["path_points"] = [[x, y] for x, y in pts]
        out["arc"] = {
            "center": [float(centre.x), float(centre.y)],
            "radius": radius,
            "start_angle": start,
            "end_angle": end,
            "angles_in": "radians",
        }
        return basis(
            f"ARC.center + radius + start/end angle, flattened to {len(pts)} "
            f"points within {geometry.ARC_CHORD_TOLERANCE} drawing units of "
            "the true curve"
        )

    read = _raw_ring(entity)
    if read is None:
        return basis("polyline vertices could not be read")
    points, closed, _has_bulge, non_planar, bulges = read
    if closed:
        return basis("the polyline is closed: its shape is in `ring`")
    if len(points) < 2:
        return basis(f"{len(points)} vertices: no span to describe")
    if len(points) > geometry.MAX_RING_VERTICES:
        # The same ceiling `_ring_fields` applies, and applied for the same
        # reason: what is over it is not stored at all rather than stored
        # truncated, because a truncated path is a different path.
        return basis(
            f"{len(points)} vertices exceed the {geometry.MAX_RING_VERTICES} limit"
        )

    # One bulge per SPAN. A closed polyline has as many spans as vertices; an
    # open one has one fewer, and the trailing bulge belongs to a span that
    # does not exist.
    spans = bulges[: len(points) - 1]
    out["path_points"] = [[x, y] for x, y in points]
    phrase = "the open polyline's own vertices (WCS)"
    if any(abs(b) > 1e-9 for b in spans):
        out["path_bulges"] = [round(b, 12) for b in spans]
        phrase += " + per-span bulges"
    if non_planar:
        phrase += "; Z varies, so these are its plan projection"
    return basis(phrase)


def _ocs_to_wcs(
    entity: DXFGraphic, points: list[tuple[float, float]], elevation: float
) -> list[tuple[float, float]] | None:
    """Lift OCS points into WCS, or None if the transform cannot be built.

    The same gate `_raw_ring` uses, for the same reason: for a mirrored entity
    the extrusion is (0,0,-1), which FLIPS the x axis. An OCS endpoint pitted
    against a WCS endpoint on the next entity is metres away from where it
    really is, and the two paths that genuinely meet would be reported as two
    separate dead ends -- the exact answer this field exists to get right.
    """
    try:
        ocs = entity.ocs()
        return [
            (float(w.x), float(w.y))
            for w in (ocs.to_wcs((x, y, elevation)) for x, y in points)
        ]
    except Exception as exc:  # noqa: BLE001 - one bad extrusion must not sink ingest
        log.debug("OCS to WCS failed for ends of %s: %s", entity.dxftype(), exc)
        return None


def _arc_end_angles(entity: DXFGraphic) -> tuple[float, float] | None:
    """(start, end) sweep angles in radians, or None when the arc is a circle.

    A full circle is written as start=0, end=360. That reduces to a sweep of
    zero under the modulo -- the same trap `_measure` guards against when it
    would otherwise report an arc length of zero -- and here it means something
    stronger: a closed curve has no free ends at all, so the answer is not two
    points, it is `closed_ring`.
    """
    start = float(entity.dxf.start_angle)
    end = float(entity.dxf.end_angle)
    if (end - start) % 360.0 == 0.0:
        return None
    return math.radians(start), math.radians(end)


def _ends_fields(entity: DXFGraphic) -> dict[str, Any]:
    """The two free ends of one path entity, in WCS -- or why there are none.

    This is DOSSIER Phase 5c's data-layer change, and the reason it exists is
    measurable rather than tidy: before it, an open path was stored as a
    BOUNDING BOX. A box has four corners and a path has two ends, the file does
    not record which diagonal is which, and so nothing about how paths MEET was
    answerable from the store. `dossier_network` had to offer union-find all
    four corners and label its chain count `approximate`; `frontage_check` had
    to measure a plot's clearance to a road's box and publish a lower bound
    instead of a distance. Counting T-junctions was not possible at all.

    Three outcomes, and the difference between the last two is the whole point:

    * `open_path` -- two points, exact, in `ends`;
    * `closed_ring` -- this curve HAS no free ends. `ends` is absent, and that
      absence is a measured fact rather than a gap. It also carries something
      no other field in the document does: `ring_status` cannot tell a closed
      polyline from an open one once the polyline contains an arc, because the
      bulge gate fires before the closed test and both come back as `bulge`;
    * `not_derived` -- the ends could not be established from what the file
      records. `ends` is absent and `ends_basis` says why. A spline's control
      points need not lie on its own curve, so taking the first one would be a
      coordinate nobody measured, and it would be wrong by a plausible-looking
      amount rather than obviously (G8).

    Two points, deliberately, and not a flattening. What that costs is real and
    is stated where it is paid: a consumer can test whether a node lies ON a
    LINE (the segment between the two ends IS the line), and cannot test the
    INTERIOR of an arc or of a curved polyline, because two points do not
    describe a curve. `recipes.junctions` publishes that hole as a per-node
    degree RANGE rather than resolving it in silence.

    `ends_basis` is a PHRASE and not a sentence, for a measured reason: 18,074
    of the reference drawing's 46,754 entities are of a path type, so a
    paragraph here is a paragraph stored 18,074 times. The prose belongs in
    this docstring, which is stored once.
    """
    dxftype = entity.dxftype()
    out: dict[str, Any] = {"ends": None, "ends_status": None, "ends_basis": None}

    def closed(reason: str) -> dict[str, Any]:
        out["ends_status"] = ENDS_CLOSED
        out["ends_basis"] = reason
        return out

    def undecided(reason: str) -> dict[str, Any]:
        out["ends_status"] = ENDS_NOT_DERIVED
        out["ends_basis"] = reason
        return out

    def opened(points: list[tuple[float, float]], basis: str) -> dict[str, Any]:
        flat = [v for p in points for v in p]
        if len(points) != 2 or not all(_finite(v) for v in flat):
            # No point at all rather than a sentinel one.
            return undecided(f"{basis}: coordinates not finite")
        out["ends"] = [[points[0][0], points[0][1]], [points[1][0], points[1][1]]]
        out["ends_status"] = ENDS_OPEN
        out["ends_basis"] = basis
        return out

    try:
        if dxftype == "LINE":
            # LINE stores group 10/11 in WCS, not OCS -- it is one of the few
            # entities the DXF reference defines that way -- so there is no
            # extrusion correction to make here and making one would move it.
            start, end = entity.dxf.start, entity.dxf.end
            return opened(
                [(float(start.x), float(start.y)), (float(end.x), float(end.y))],
                "LINE.start/LINE.end (WCS)",
            )

        if dxftype == "ARC":
            angles = _arc_end_angles(entity)
            if angles is None:
                return closed("ARC sweeps a full turn: a circle, no free ends")
            centre = entity.dxf.center
            radius = float(entity.dxf.radius)
            cx, cy, cz = float(centre.x), float(centre.y), float(centre.z)
            points = [
                (cx + radius * math.cos(a), cy + radius * math.sin(a))
                for a in angles
            ]
            # The points the arc really starts and ends at, never the corners
            # of its bounding box.
            basis = "ARC.center + radius + start_angle/end_angle"
            if _extrusion_non_standard(entity):
                lifted = _ocs_to_wcs(entity, points, cz)
                if lifted is None:
                    # An OCS point published as a WCS one is wrong by the
                    # width of the drawing, so none is published at all.
                    return undecided("ARC extrusion: OCS not liftable to WCS")
                return opened(lifted, basis + ", OCS lifted to WCS")
            return opened(points, basis)

        if dxftype == "CIRCLE":
            return closed("CIRCLE is a closed curve: no free ends")

        if dxftype in RING_TYPES:
            read = _raw_ring(entity)
            if read is None:
                return undecided("polyline vertices could not be read")
            points, is_closed, _has_bulge, _non_planar, _bulges = read
            if is_closed:
                # The only field that says so: a closed polyline containing an
                # arc is reported by `ring_status` as `bulge`, exactly like an
                # open one, because the bulge gate fires before the closed test.
                return closed("polyline is closed: no free ends")
            if len(points) < 2:
                return undecided(
                    f"polyline carries {len(points)} vertices: no two ends"
                )
            return opened(
                # `_raw_ring` has already lifted these out of OCS where the
                # extrusion required it.
                [points[0], points[-1]],
                "first/last vertex of the open polyline (WCS)",
            )

        if dxftype == "ELLIPSE":
            start = float(entity.dxf.get("start_param", 0.0) or 0.0)
            end = float(entity.dxf.get("end_param", 0.0) or 0.0)
            if (end - start) % (2 * math.pi) == 0.0:
                return closed("ELLIPSE sweeps a full turn: no free ends")
            first, last = entity.start_point, entity.end_point
            return opened(
                [(float(first.x), float(first.y)), (float(last.x), float(last.y))],
                "ELLIPSE.start_point/end_point (centre + axis + params)",
            )

        if dxftype == "SPLINE":
            # A SPLINE's control points need NOT lie on its own curve -- only a
            # clamped spline's first and last do, and nothing stored says
            # whether this one is clamped. Taking them would publish a
            # coordinate wrong by a plausible amount rather than obviously.
            return undecided("SPLINE control points need not lie on the curve")
    except Exception as exc:  # noqa: BLE001 - one odd entity must not sink ingest
        log.debug("endpoint extraction failed for %s: %s", dxftype, exc)
        return undecided(f"reading this {dxftype} raised {type(exc).__name__}")

    return undecided(f"{dxftype}: no rule for its ends is written yet")


def _length_and_area(
    pts: list[tuple[float, float]], closed: bool
) -> tuple[float | None, float | None]:
    """Length, and area when closed -- the area computed in a local frame.

    Shoelace multiplies coordinates before subtracting them, so in projected
    coordinates it sums numbers around 1e12 to produce a number around 1e2.
    What is left is the area plus noise. For area the error is small (around
    4e-7 relative) and has been invisible all along; for the centroid it is
    multiplied back up by the magnitude of the coordinates and becomes metres.
    One subtraction per vertex removes both.
    """
    length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    if not closed:
        return length, None
    length += math.dist(pts[-1], pts[0])
    return length, abs(geometry.signed_area(pts))


def _measure(entity: DXFGraphic) -> tuple[float | None, float | None]:
    """(length, area) where the entity type makes them meaningful."""
    dxftype = entity.dxftype()
    try:
        if dxftype == "LINE":
            return (entity.dxf.start - entity.dxf.end).magnitude, None
        if dxftype == "CIRCLE":
            r = float(entity.dxf.radius)
            return 2 * math.pi * r, math.pi * r * r
        if dxftype == "ARC":
            r = float(entity.dxf.radius)
            start = float(entity.dxf.start_angle)
            end = float(entity.dxf.end_angle)
            sweep = (end - start) % 360.0
            # A full circle is written as start=0, end=360, which reduces to
            # 0 under the modulo and would report an arc length of zero.
            if sweep == 0.0 and end != start:
                sweep = 360.0
            return 2 * math.pi * r * (sweep / 360.0), None
        if dxftype == "LWPOLYLINE":
            # Bulge is the arc-ness of each span. get_points("xy") discards it,
            # so treating every span as a straight chord under-reports length
            # and mis-reports area on any polyline with a rounded corner or an
            # arc segment. A wrong number presented confidently is worse than
            # no number, and these values are surfaced verbatim by the agent's
            # get_entity tool -- so a curved polyline reports None.
            if any(abs(float(p[4])) > 1e-9 for p in entity.get_points("xyseb")):
                return None, None
            pts = [(p[0], p[1]) for p in entity.get_points("xy")]
            if len(pts) < 2:
                return None, None
            closed = bool(entity.closed)
            return _length_and_area(pts, closed)
        if dxftype == "POLYLINE":
            # A closed POLYLINE is the normal shape for topographic contours,
            # and previously it had no branch here at all: it got a ring and a
            # centroid but its length and area were null, which made the
            # perimeter-against-length cross-check hollow precisely on the
            # drawings that needed it most.
            verts = list(entity.vertices)
            if any(
                abs(float(v.dxf.get("bulge", 0.0) or 0.0)) > 1e-9 for v in verts
            ):
                return None, None
            pts = [
                (float(v.dxf.location.x), float(v.dxf.location.y)) for v in verts
            ]
            if len(pts) < 2:
                return None, None
            return _length_and_area(pts, bool(entity.is_closed))
    except (AttributeError, ValueError, TypeError) as exc:
        log.debug("measurement failed for %s: %s", dxftype, exc)
    return None, None


#: Prefix marking a pseudo-layout that is really a block definition.
BLOCK_LAYOUT_PREFIX = "[block] "


def block_layout_name(block_name: str) -> str:
    """Pseudo-layout name for a block definition."""
    return f"{BLOCK_LAYOUT_PREFIX}{block_name}"


def is_block_layout(layout_name: str) -> bool:
    return layout_name.startswith(BLOCK_LAYOUT_PREFIX)


def block_name_of(layout_name: str) -> str:
    return layout_name[len(BLOCK_LAYOUT_PREFIX) :]


def _renderable_blocks(doc: Drawing) -> list[str]:
    """Named block definitions that actually contain geometry.

    Anonymous blocks (`*Model_Space`, `*U1`, `*Paper_Space`) are excluded:
    the first two are the layouts themselves under another name, and the `*U`
    ones are unnamed copies whose content is already reachable through the
    named block they duplicate.
    """
    out: list[str] = []
    for block in doc.blocks:
        if block.name.startswith("*"):
            continue
        if any(True for _ in block):
            out.append(block.name)
    return out



#: How deep a chain of nested INSERTs is followed when working out where a
#: block definition is placed. Sedra's bound xrefs nest four levels deep in
#: their layer names; this is a guard against a cyclic or pathological file,
#: not a judgement about real drawings, and a chain that hits it is recorded
#: as unresolved rather than silently truncated.
MAX_PLACEMENT_DEPTH: Final[int] = 8

#: How far a composed transform may sit from the identity and still be called
#: one. An identity INSERT in these files is exactly identity, so this only
#: absorbs floating-point noise from composing matrices.
IDENTITY_EPSILON: Final[float] = 1e-9

_IDENTITY_ROWS: Final[tuple[tuple[float, ...], ...]] = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _is_identity(matrix: Any) -> bool:
    try:
        rows = tuple(tuple(float(v) for v in row) for row in matrix.rows())
    except Exception:  # noqa: BLE001 - a matrix we cannot read is not identity
        return False
    return all(
        abs(a - b) <= IDENTITY_EPSILON
        for row, ident in zip(rows, _IDENTITY_ROWS)
        for a, b in zip(row, ident)
    )


def block_placements(doc: Drawing) -> dict[str, dict[str, Any]]:
    """How model space places each block definition, by composing transforms.

    THE QUESTION THIS ANSWERS. Entities inside a block definition are stored
    with the block's own coordinates. Whether those coordinates are also
    positions on the earth depends entirely on the transform chain that places
    the block: if every INSERT from model space down to it is the identity,
    the block's numbers ARE world numbers and nothing needs moving. If any
    link scales, rotates or offsets, they are not, and they would have to be
    transformed to become so.

    That is a question about MATRICES, and it was previously being answered
    from a layout NAME (`"[block] ..."` -> "not a place"), which is why a
    990,790-entity drawing whose content sits in bound xrefs came out with an
    empty map. A bound xref keeps the coordinates it was drawn in and is
    placed by an identity INSERT, so the name said one thing and the geometry
    said the opposite.

    Composition order is `child @ parent`: ezdxf matrices are row-vector, so a
    point in the child's frame is carried out through the child's own
    placement first and then through its parent's.

    A block reached by several different chains is reported as such rather
    than resolved. One stored row cannot be in two places, and picking one
    would be a guess that looks like a measurement.
    """
    from ezdxf.math import Matrix44  # noqa: PLC0415 - only needed here

    found: dict[str, list[Any]] = {}
    depth_of: dict[str, int] = {}
    truncated: set[str] = set()
    seen: set[tuple[str, tuple[float, ...]]] = set()

    stack: list[tuple[str, Any, int]] = []
    try:
        msp = doc.modelspace()
    except Exception:  # noqa: BLE001 - a document with no model space
        return {}
    for entity in msp:
        if entity.dxftype() != "INSERT":
            continue
        name = entity.dxf.get("name")
        if not name:
            continue
        try:
            stack.append((str(name), entity.matrix44(), 1))
        except Exception:  # noqa: BLE001 - an INSERT we cannot read
            continue

    while stack:
        name, matrix, depth = stack.pop()
        if depth > MAX_PLACEMENT_DEPTH:
            truncated.add(name)
            continue
        try:
            key = (name, tuple(round(v, 9) for row in matrix.rows() for v in row))
        except Exception:  # noqa: BLE001
            continue
        if key in seen:
            continue
        seen.add(key)
        found.setdefault(name, []).append(matrix)
        depth_of[name] = min(depth_of.get(name, depth), depth)

        block = doc.blocks.get(name) if name in doc.blocks else None
        if block is None:
            continue
        for child in block:
            if child.dxftype() != "INSERT":
                continue
            child_name = child.dxf.get("name")
            if not child_name:
                continue
            try:
                stack.append((str(child_name), child.matrix44() @ matrix, depth + 1))
            except Exception:  # noqa: BLE001
                continue

    placements: dict[str, dict[str, Any]] = {}
    for name, matrices in found.items():
        identities = [m for m in matrices if _is_identity(m)]
        placements[name] = {
            "placements": len(matrices),
            "identity": bool(identities),
            "ambiguous": bool(identities) and len(identities) != len(matrices),
            "depth": depth_of.get(name),
            "chain_truncated": name in truncated,
        }
    return placements

def _containers(doc: Drawing) -> Iterator[tuple[str, Any]]:
    """Every place a top-level entity can live, with the name to record.

    Layouts first, then named block definitions as pseudo-layouts. Blocks are
    included because for several of these files they are the *only* place the
    content exists: `title_block-arch` and `title_block-iso` each have one
    entity in a layout and 117 inside a block definition, so walking layouts
    alone reports them as effectively empty drawings.

    Entities inside a block *definition* have real, persistent handles --
    unlike the temporary copies returned by `insert.virtual_entities()`, which
    have `handle = None`. So the whole identity chain still holds for them and
    they remain clickable and commentable.
    """
    for layout in doc.layouts:
        yield layout.name, layout
    for name in _renderable_blocks(doc):
        yield block_layout_name(name), doc.blocks.get(name)


def iter_entities(doc: Drawing, drawing_id: str) -> Iterator[EntityDoc]:
    """Yield one `EntityDoc` per top-level entity across *all* layouts.

    Modelspace is one layout among several. Files whose content lives only in
    a paperspace layout would otherwise be reported as empty.
    """
    cache = bbox.Cache()
    seen: set[str] = set()

    for layout_name, container in _containers(doc):
        for entity in container:
            handle = entity.dxf.get("handle")
            if not handle:
                # Handle-less entities cannot be selected, commented on, or
                # linked to an SVG group, so they have no place in the store.
                continue
            if handle in seen:
                continue
            seen.add(handle)

            box, bbox_centre = _entity_bbox(entity, cache)
            length, area = _measure(entity)
            dxftype = entity.dxftype()
            extra: dict[str, Any] = {}
            if dxftype in RING_TYPES:
                extra.update(_ring_fields(entity))
            if dxftype in RING_TYPES or dxftype in CURVE_TYPES:
                extra.update(_path_point_fields(entity))
            if dxftype in ANCHOR_TYPES:
                extra.update(_anchor_fields(entity))
                extra["text_style"] = _text_style_of(entity)
            if dxftype in PATH_TYPES:
                extra.update(_ends_fields(entity))
            yield EntityDoc(
                _id=f"{drawing_id}:{handle}",
                drawing_id=drawing_id,
                handle=handle,
                type=dxftype,
                layer=entity.dxf.get("layer", "0"),
                layout=layout_name,
                block_name=entity.dxf.get("name") if dxftype == "INSERT" else None,
                text=_text_of(entity),
                attribs=_attribs_of(entity),
                bbox=box,
                bbox_centre=bbox_centre,
                length=length,
                area=area,
                **extra,
            )


def _collect_xrefs(doc: Drawing) -> list[dict[str, Any]]:
    """Every external file this drawing points at, recorded even if missing.

    Three kinds, and all of them make AutoCAD prompt the recipient:

    - **DWG xrefs** — blocks flagged as external references;
    - **underlay definitions** (PDF/DWF/DGN) — Janadriyah attaches two PDFs;
    - **image definitions** (raster attachments) — Janadriyah attaches four
      JPGs, all pointing at the original author's own disk
      (``C:/Users/DELL/Desktop/...``).

    Those six are exactly TrueView's "6 reference files not found" prompt on
    this file. Cataloguing them lets our UI explain that prompt instead of
    leaving the user to wonder whether our export broke something — it did
    not; the attachments were never distributed with the drawing.
    """
    xrefs: list[dict[str, Any]] = []
    for block_record in doc.block_records:
        try:
            block = doc.blocks.get(block_record.dxf.name)
        except Exception:
            continue
        if block is None:
            continue
        block_def = block.block
        if block_def is None:
            continue
        # bit 4 = external reference, bit 8 = xref overlay
        flags = int(block_def.dxf.get("flags", 0))
        if flags & 0b1100:
            xrefs.append(
                {
                    "block_name": block_def.dxf.get("name", ""),
                    "path": block_def.dxf.get("xref_path", ""),
                    "resolved": False,
                    "kind": "dwg-xref",
                }
            )

    # Underlay and raster definitions live in the OBJECTS section, not in the
    # block table, so the loop above never sees them.
    definition_kinds = {
        "PDFDEFINITION": "pdf-underlay",
        "DWFDEFINITION": "dwf-underlay",
        "DGNDEFINITION": "dgn-underlay",
        "UNDERLAYDEFINITION": "underlay",
        "IMAGEDEF": "image",
    }
    try:
        for obj in doc.objects:
            kind = definition_kinds.get(obj.dxftype())
            if kind is None:
                continue
            xrefs.append(
                {
                    "block_name": obj.dxftype(),
                    "path": obj.dxf.get("filename", ""),
                    "resolved": False,
                    "kind": kind,
                }
            )
    except Exception as exc:  # noqa: BLE001 - cataloguing must not sink ingest
        log.warning("reference definition scan failed: %s", exc)
    return xrefs


# --- Drawing tables (UPLIFT-11) ---------------------------------------------
#
# Everything below reads the drawing's DICTIONARY -- the layer, style,
# linetype, dim style, header and page-setup tables -- not its entities. It is
# separated from the entity block above deliberately: the two answer different
# questions ("what objects are in here" vs "how is this drawing set up"), and
# before this spec only the first was stored.
#
# One rule holds across the whole block: a value that is not in the file
# becomes `None`, never its DXF default. Writing the default means inventing
# an answer to the question "does this drawing state it", and that is exactly
# the question being asked (G8).


class _Missing:
    """Sentinel: read and genuinely absent, not 'not read yet'."""


_MISSING = _Missing()

#: The header variables that are stored. Standard DXF names, not constants
#: specific to one drawing (G1): chosen because each of them changes the
#: meaning of another number in this document -- units, linetype scale,
#: dimension scale, precision, lineweight.
HEADER_VARS: tuple[str, ...] = (
    "$ACADVER",
    "$INSUNITS",
    "$MEASUREMENT",
    "$LTSCALE",
    "$DIMSCALE",
    "$LUNITS",
    "$LUPREC",
    "$AUNITS",
    "$CELWEIGHT",
)

#: `plot_paper_units` on the LAYOUT/PLOTSETTINGS object. PAPER units, which
#: have nothing to do with `$INSUNITS`: this A0 sheet is 841 x 1189 mm while
#: its model drawing is in metres. Equating the two is the root of D-076 --
#: 1,867 paper border lines were once reported as "2992.231252 m".
PAPER_UNIT_NAMES: dict[int, str] = {0: "inch", 1: "mm", 2: "pixel"}


def _layer_description(layer: Any) -> str | None:
    """The layer description, or None.

    Not a DXF attribute: it lives in the layer's extension dictionary
    (`AcAecLayerStandard`), so ezdxf presents it as a property that can throw
    on a layer whose extension dict is damaged. 261 audit errors in the
    largest file already prove that is not an edge case (G8).
    """
    try:
        text = (layer.description or "").strip()
    except Exception:  # noqa: BLE001 - one broken layer must not sink ingest
        return None
    return text or None


def _viewport_freezes(doc: Drawing) -> dict[str, list[str]]:
    """Layer -> the handles of the VIEWPORTs that freeze it.

    Read from the VIEWPORT entities, not from the layer table: a per-viewport
    freeze is a property of the VIEWPORT, and a layer does not know who freezes
    it. The direction is reversed here so that the layer table can carry it.

    Empty is a valid answer and often the right one; it is NOT the same as
    "not checked", and that is why every layer still carries this key with an
    empty list.
    """
    out: dict[str, list[str]] = {}
    for layout in doc.layouts:
        try:
            entities = list(layout)
        except Exception as exc:  # noqa: BLE001
            log.warning("viewport scan failed on layout %s: %s", layout.name, exc)
            continue
        for entity in entities:
            if entity.dxftype() != "VIEWPORT":
                continue
            handle = entity.dxf.get("handle")
            try:
                frozen = list(entity.frozen_layers)
            except Exception as exc:  # noqa: BLE001
                log.warning("frozen_layers unreadable on viewport %s: %s", handle, exc)
                continue
            for name in frozen:
                out.setdefault(str(name), []).append(str(handle))
    return out


def _layer_table(doc: Drawing, counts_by_layer: dict[str, int]) -> list[dict[str, Any]]:
    """The complete layer table -- eleven fields, not five.

    The old five fields could not answer the questions that make this table
    useful. The 23 August analysis session read color, linetype, lineweight,
    on/off and plot to prove that schools and houses CANNOT be told apart by
    their appearance -- and that sameness was the finding. Without `linetype`
    and `lineweight` stored, that finding cannot be repeated.

    `plot` and `locked` are stored because both explain why something does not
    appear on paper or cannot be selected, and until now that could only be
    answered by opening the original file.

    `color` uses the ezdxf property, not the raw `dxf.color`. DXF encodes "the
    layer is off" as a NEGATIVE colour, so a raw reading reports −20 for a
    layer whose colour is 20 and which is switched off. ACI has no negative
    colours: it is a number that lies, and it lies on exactly the four layers
    of this file that are off. Being off is already said by `off`; the colour
    stays the colour.
    """
    freezes = _viewport_freezes(doc)
    out: list[dict[str, Any]] = []
    for layer in doc.layers:
        name = layer.dxf.name
        out.append(
            {
                "name": name,
                "color": abs(int(layer.dxf.get("color", 7))),
                "frozen": bool(layer.is_frozen()),
                "off": not bool(layer.is_on()),
                "entity_count": counts_by_layer.get(name, 0),
                "linetype": layer.dxf.get("linetype", None),
                "lineweight": int(layer.dxf.get("lineweight", -3)),
                "plot": bool(layer.dxf.get("plot", 1)),
                "locked": bool(layer.is_locked()),
                "description": _layer_description(layer),
                "frozen_in_viewports": freezes.get(name, []),
            }
        )
    return out


def _text_styles(doc: Drawing) -> list[dict[str, Any]]:
    """The text style table, with its font files.

    This is the single piece of evidence behind UPLIFT-04: the Arabic text in
    this file is drawn through an SHX font that stores Latin bytes, and the
    only way to know a string must be read that way is that its style points
    at that font file. The filename is stored AS IT IS, including the drawing
    author's own absolute path: `C:/ACAD/ESA/X-ARAB1b.SHX` is the statement
    that the font once existed on someone's disk, and normalising it to a
    basename erases that.
    """
    out: list[dict[str, Any]] = []
    for style in doc.styles:
        out.append(
            {
                "name": style.dxf.get("name", ""),
                "font": style.dxf.get("font", "") or None,
                "bigfont": style.dxf.get("bigfont", "") or None,
            }
        )
    return out


def _pattern_length(linetype: Any) -> float | None:
    """The total pattern length of one linetype, or None.

    Not a DXF attribute: ezdxf keeps it inside `pattern_tags` as a group 40
    tag. `Continuous` and its kin hold 0.0 there -- a zero that MEANS
    something (there is no pattern), not a zero meaning unreadable, so the two
    must stay distinguishable and the unreadable one becomes None.
    """
    try:
        tags = linetype.pattern_tags
    except Exception:  # noqa: BLE001
        return None
    try:
        for code, value in tags.tags:
            if code == 40:
                return float(value)
    except Exception:  # noqa: BLE001
        return None
    return None


def _linetype_table(doc: Drawing) -> list[dict[str, Any]]:
    """The linetype table: name, description, pattern length.

    `pattern_length` carries drawing units (G2) -- 2.0 on a drawing in metres
    and 2.0 on a drawing in inches are two different lengths. It is not
    normalised here; the units are handed to the reader by `_unit_names`.
    """
    out: list[dict[str, Any]] = []
    for linetype in doc.linetypes:
        out.append(
            {
                "name": linetype.dxf.name,
                "description": linetype.dxf.get("description", "") or None,
                "pattern_length": _pattern_length(linetype),
            }
        )
    return out


def _dim_styles(doc: Drawing) -> list[dict[str, Any]]:
    """The dimension style table, each with its own `$DIMSCALE`.

    `scale` is None when that style does not state a `dimscale`. DXF does have
    a default of 1.0, but writing it here would make "this style sets its
    scale" and "this style inherits the default" look the same -- and what is
    being compared is precisely the style that sets it (moh = 2.0) against the
    one that does not.
    """
    out: list[dict[str, Any]] = []
    for style in doc.dimstyles:
        raw = style.dxf.get("dimscale", _MISSING)
        out.append(
            {
                "name": style.dxf.name,
                "scale": None if isinstance(raw, _Missing) else float(raw),
            }
        )
    return out


def _declared_crs(doc: Drawing) -> dict[str, Any] | None:
    """What the FILE says its coordinate system is, or `None` if it says
    nothing.

    A GEODATA object is the one place a DXF can state this, and none of the
    drawings in the store today has one — Janadriyah included, which is why a
    zone had to be inferred for it and why every lat/long it produces carries
    a warning. So this reads a path that is currently always empty, and that
    is the point: the next drawing to arrive may well carry one, and it must
    then need no config file, no override, and no code change.

    Three outcomes, and the middle one is the reason this returns a dict
    rather than an int:

    - no GEODATA at all → `None`. Nothing is claimed.
    - GEODATA present but its coordinate system cannot be read → a dict with
      `epsg: None` and `unreadable` saying why. NOT `None`: "this file says
      nothing" and "this file says something I could not read" are different
      facts, and only the second one is a lead worth following (G8).
    - GEODATA read → `epsg` and `xy_ordering`.

    `xy_ordering` is stored because it is a correctness fact, not trivia:
    ezdxf reports `False` when the declared axis order is northing-first, and
    a northing-first CRS fed to a converter that expects easting-first
    produces coordinates that look ordinary and are somewhere else entirely.

    Nothing here decides whether the declared system can be USED — that needs
    the converter, which lives in `landuse/crs.py`. Extraction records what
    the file said; the store decides what to do about it.
    """
    try:
        geodata = doc.modelspace().get_geodata()
    except Exception as exc:  # pragma: no cover - depends on the file
        log.debug("could not read GEODATA: %s", exc)
        return None
    if geodata is None:
        return None

    out: dict[str, Any] = {
        "source": "GEODATA object in model space",
        "epsg": None,
        "xy_ordering": None,
        "unreadable": None,
    }
    try:
        out["coordinate_type"] = int(geodata.dxf.coordinate_type)
    except Exception:
        out["coordinate_type"] = None

    try:
        epsg, xy_ordering = geodata.get_crs()
    except Exception as exc:
        # Every failure mode ezdxf has here is a sentence worth keeping: no
        # EPSG alias, unparseable XML, an axis abbreviation nobody recognises.
        # Storing the class name alongside it means the reason survives into a
        # response without the file having to be opened again.
        out["unreadable"] = f"{type(exc).__name__}: {exc}"
        return out

    out["epsg"] = int(epsg)
    out["xy_ordering"] = bool(xy_ordering)
    return out


def _header_vars(doc: Drawing) -> dict[str, Any]:
    """The selected header variables, with the absent ones valued None.

    The keys are always complete. A variable missing from the document is
    information -- three of the sixteen drawings do not state their units at
    all -- and dropping the key makes "not stated" indistinguishable from
    "never read" (G2, G8).
    """
    out: dict[str, Any] = {}
    for var in HEADER_VARS:
        try:
            value = doc.header.get(var, _MISSING)
        except Exception:  # noqa: BLE001
            value = _MISSING
        out[var] = None if isinstance(value, _Missing) else value
    return out


def _page_setup(layout: Any) -> dict[str, Any]:
    """One layout's page settings: paper size, its units, plot scale, device.

    UPLIFT-11 writes that this must be read "through the page setup object,
    NOT the LAYOUT attributes directly". Measured straight from the file, the
    truth is the opposite: in DXF, LAYOUT INHERITS PLOTSETTINGS, so
    `paper_size`, `paper_width`, `plot_paper_units`,
    `scale_numerator/denominator` and `plot_configuration_file` are on the
    LAYOUT object itself. All 48 of those attributes are present on all three
    Janadriyah layouts. The separate PLOTSETTINGS objects that are also in the
    file (six of them) are NAMED page setups a user can apply -- a different
    thing, not this layout's active settings.

    So what is implemented is its Definition of Done, which is conditional by
    design: read from LAYOUT, and if it is not there, say what is not there --
    do not fail silently the way today does. This divergence is recorded in
    PROGRESS.md, not patched over in silence.

    `paper_units` comes from `plot_paper_units`, NEVER from `$INSUNITS`. The
    two are unrelated: this sheet is 841 x 1189 mm while its model is in
    metres. Equating them is D-076.
    """
    unavailable = {
        "available": False,
        "why": (
            "this layout carries no plot settings; the LAYOUT object exposed "
            "none of paper_size / paper_width / plot_paper_units"
        ),
        "paper_size_name": None,
        "paper_width": None,
        "paper_height": None,
        "paper_units": None,
        "plot_scale": None,
        "plot_scale_text": None,
        "device": None,
        "page_setup_name": None,
        "source": None,
    }

    try:
        dxf_layout = layout.dxf_layout
    except Exception as exc:  # noqa: BLE001
        return {**unavailable, "why": f"LAYOUT object unreadable: {exc}"}
    if dxf_layout is None:
        return unavailable

    def attr(name: str) -> Any:
        try:
            value = dxf_layout.dxf.get(name, _MISSING)
        except Exception:  # noqa: BLE001
            return None
        return None if isinstance(value, _Missing) else value

    size_name = attr("paper_size")
    width = attr("paper_width")
    height = attr("paper_height")
    units_code = attr("plot_paper_units")
    if size_name is None and width is None and units_code is None:
        return unavailable

    numerator = attr("scale_numerator")
    denominator = attr("scale_denominator")
    scale = None
    scale_text = None
    if numerator is not None and denominator:
        scale = {"numerator": float(numerator), "denominator": float(denominator)}
        scale_text = f"{float(numerator):g}:{float(denominator):g}"

    return {
        "available": True,
        "why": None,
        "paper_size_name": size_name or None,
        "paper_width": None if width is None else float(width),
        "paper_height": None if height is None else float(height),
        "paper_units": (
            None if units_code is None
            else PAPER_UNIT_NAMES.get(int(units_code))
        ),
        "paper_units_code": None if units_code is None else int(units_code),
        "plot_scale": scale,
        "plot_scale_text": scale_text,
        "device": attr("plot_configuration_file") or None,
        "page_setup_name": attr("page_setup_name") or None,
        "source": "LAYOUT object (inherits PLOTSETTINGS)",
    }



def expanded_entity_counts(
    doc: Drawing, counts_by_layout: Mapping[str, int]
) -> dict[str, int]:
    """How many entities each layout would DRAW, with its INSERTs expanded.

    A layout's own entity count is what it contains, not what a renderer has
    to produce. Sedra's model space holds 19 INSERTs and draws roughly 990,000
    objects, and rendering it is what took the ingest past a 15.47 GiB ceiling
    and got the process killed. A cost that is three orders of magnitude off
    cannot be used to decide whether the work is affordable, so it is measured
    here instead of guessed at.

    Counted from the block table rather than by expanding anything, so this is
    arithmetic over numbers already in hand and costs nothing on any drawing.
    Recursion is memoised and depth-capped; a cycle contributes its own size
    once rather than looping.
    """
    block_sizes: dict[str, int] = {}

    def size_of(block_name: str, depth: int, seen: frozenset[str]) -> int:
        if depth > MAX_PLACEMENT_DEPTH or block_name in seen:
            return 0
        cached = block_sizes.get(block_name)
        if cached is not None:
            return cached
        block = doc.blocks.get(block_name) if block_name in doc.blocks else None
        if block is None:
            return 0
        total = 0
        for entity in block:
            total += 1
            if entity.dxftype() == "INSERT":
                child = entity.dxf.get("name")
                if child:
                    total += size_of(str(child), depth + 1, seen | {block_name})
        block_sizes[block_name] = total
        return total

    expanded: dict[str, int] = {}
    for name, container in _containers(doc):
        if container is None:
            expanded[name] = counts_by_layout.get(name, 0)
            continue
        total = 0
        try:
            for entity in container:
                total += 1
                if entity.dxftype() == "INSERT":
                    child = entity.dxf.get("name")
                    if child:
                        total += size_of(str(child), 1, frozenset())
        except Exception:  # noqa: BLE001 - a container we cannot walk
            total = counts_by_layout.get(name, 0)
        expanded[name] = total
    return expanded

def _layout_summaries(
    doc: Drawing,
    counts_by_layout: dict[str, int],
    expanded: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Per-layout entity counts -- the cheapest guide to where content lives.

    Counts come from what was actually extracted, not from a separate walk.
    Counting the container independently let a layout advertise entities that
    were never stored (handle-less ones are skipped), and that number drives
    both what gets rendered and which layout the viewer opens by default.

    Since UPLIFT-11 every REAL layout also carries `page_setup`. Block
    pseudo-layouts do not: a block definition is not printed to paper, and
    giving it that key would invent a question that does not apply to it.
    """
    page_setups = {}
    for layout in doc.layouts:
        page_setups[layout.name] = _page_setup(layout)

    out: list[dict[str, Any]] = []
    for name, _container in _containers(doc):
        summary: dict[str, Any] = {
            "name": name,
            "entity_count": counts_by_layout.get(name, 0),
            # What a renderer would actually have to draw. See
            # `expanded_entity_counts`: on Sedra this is 19 against ~990,000.
            "expanded_entity_count": (expanded or {}).get(
                name, counts_by_layout.get(name, 0)
            ),
            "is_modelspace": name.lower() == "model",
            "is_block": is_block_layout(name),
        }
        if not is_block_layout(name):
            summary["page_setup"] = page_setups.get(
                name,
                {
                    "available": False,
                    "why": "no LAYOUT object was found for this layout name",
                    "paper_size_name": None,
                    "paper_width": None,
                    "paper_height": None,
                    "paper_units": None,
                    "plot_scale": None,
                    "plot_scale_text": None,
                    "device": None,
                    "page_setup_name": None,
                    "source": None,
                },
            )
        out.append(summary)
    return out


def extract(path: Path, source_path: str | None = None) -> tuple[DrawingDoc, list[EntityDoc]]:
    """Extract one DXF file into its canonical drawing + entity documents.

    Args:
        path: local path to a `.dxf` file.
        source_path: logical origin recorded on the drawing document (e.g. the
            original `.dwg` path). Defaults to `path`.

    Raises:
        ExtractionError: if the file cannot be opened as a DXF at all.
    """
    started = time.perf_counter()
    # The identity hash comes from the ORIGINAL file, never the converted
    # DXF: converter output is not byte-deterministic (ODA stamps a
    # conversion time and fingerprint GUID into the DXF), so hashing it gave
    # the same DWG a fresh drawing_id on every re-ingest — orphaning its
    # comments each time (measured: Janadriyah forked 33e7… → 16be… on a
    # forced re-render). For a plain DXF ingest source_path IS path, so
    # nothing changes there.
    id_source = Path(source_path) if source_path else path
    drawing_id = compute_drawing_id(id_source if id_source.is_file() else path)
    doc, audit_errors, audit_fixes = load_document(path)
    warnings: list[str] = []
    if audit_errors:
        warnings.append(
            f"{audit_errors} audit errors were recovered while opening the file"
        )

    entities = list(iter_entities(doc, drawing_id))

    counts_by_type: dict[str, int] = {}
    counts_by_layer: dict[str, int] = {}
    counts_by_layout: dict[str, int] = {}
    for ent in entities:
        counts_by_type[ent.type] = counts_by_type.get(ent.type, 0) + 1
        counts_by_layer[ent.layer] = counts_by_layer.get(ent.layer, 0) + 1
        counts_by_layout[ent.layout] = counts_by_layout.get(ent.layout, 0) + 1

    # Whole-drawing extents: computed from modelspace geometry, with the
    # header values deliberately ignored.
    extents: dict[str, list[float]] | None = None
    try:
        whole = bbox.extents(doc.modelspace(), fast=_BBOX_FAST)
        if whole.has_data and all(
            _finite(v) for v in (whole.extmin.x, whole.extmin.y, whole.extmax.x, whole.extmax.y)
        ):
            extents = {
                "min": [whole.extmin.x, whole.extmin.y, whole.extmin.z],
                "max": [whole.extmax.x, whole.extmax.y, whole.extmax.z],
            }
    except Exception as exc:
        log.warning("extents computation failed for %s: %s", path.name, exc)
        warnings.append(f"extents could not be computed: {exc}")

    if extents is None:
        warnings.append(
            "modelspace has no computable extents (content may live in a "
            "paperspace layout or only in block definitions)"
        )

    units_code = int(doc.header.get("$INSUNITS", 0) or 0)
    blocks = sorted(b.name for b in doc.blocks if not b.name.startswith("*"))

    layers = _layer_table(doc, counts_by_layer)
    text_styles = _text_styles(doc)
    linetypes = _linetype_table(doc)
    dim_styles = _dim_styles(doc)
    header = _header_vars(doc)

    # An SHX style whose font file did not travel with the drawing cannot be
    # read by anyone -- not just by us; DWG TrueView shows a "Missing SHX
    # Files" prompt on the same file. It is recorded as a warning so that the
    # readiness report (G9) can name it, and it is NOT fixed by guessing the
    # glyphs. There is one way out: ask the drawing's author for the font file
    # (T-11, FIDELITY.md).
    shx_styles = [
        s["name"]
        for s in text_styles
        if (s["font"] or "").lower().endswith(".shx")
        or (s["bigfont"] or "").lower().endswith(".shx")
    ]
    if shx_styles:
        warnings.append(
            f"{len(shx_styles)} text styles point at SHX font files that are "
            "not shipped with the drawing; text drawn in them reads as the "
            "bytes stored, not as the glyphs intended"
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    drawing = DrawingDoc(
        _id=drawing_id,
        source_path=source_path or str(path),
        original_filename=path.name,
        dxf_version=doc.dxfversion,
        acad_release=doc.acad_release,
        units_code=units_code,
        units_name=UNIT_NAMES.get(units_code, f"unknown({units_code})"),
        extents=extents,
        layers=layers,
        blocks=blocks,
        layouts=_layout_summaries(
            doc, counts_by_layout, expanded_entity_counts(doc, counts_by_layout)
        ),
        text_styles=text_styles,
        linetypes=linetypes,
        dim_styles=dim_styles,
        header=header,
        declared_crs=_declared_crs(doc),
        block_placement=block_placements(doc),
        entity_count=len(entities),
        counts_by_type=counts_by_type,
        counts_by_layer=counts_by_layer,
        xrefs=_collect_xrefs(doc),
        file_bytes=path.stat().st_size,
        audit_errors=audit_errors,
        audit_fixes=audit_fixes,
        extract_ms=elapsed_ms,
        warnings=warnings,
    )
    log.info(
        "extracted drawing",
        extra={
            "drawing_id": drawing_id,
            "dxf_file": path.name,
            "entities": len(entities),
            "layers": len(layers),
            "blocks": len(blocks),
            "elapsed_ms": elapsed_ms,
        },
    )
    return drawing, entities
