"""PROVISIONAL — vertex geometry over HTTP, for the deck.gl map views.

    STATUS      Written by the frontend to unblock the map work. Not yet
                reviewed or owned by the backend team.
    SPEC        docs/DECKGL-GEOMETRY-CONTRACT.md — read that first; it states
                the contract, the measurements behind each decision, and the
                open questions this file deliberately does not answer.
    FOOTPRINT   This module, plus ONE `include_router` line at the bottom of
                `main.py`. Nothing else in cad-api is touched.

Why a separate module rather than another route in `main.py`: reviewing this
is reading one file, and rejecting it is deleting one file plus one line. The
backend team is expected to fold it into the existing route families once the
shape is agreed — at which point this file should stop existing rather than
become a second home for geometry.

What it does, in one sentence: it is a **projection over documents Mongo
already holds**. No ezdxf, no re-parsing of a 136 MB DXF, no new collection,
no write of any kind. `store_geometry.py` and `INGEST_VERSION` 6 already store
`ring` and `path_points`; §3.2 of the architecture note found that nothing
exposes them over HTTP, and this is that exposure.

Three constraints shaped the implementation, and all three are load-bearing:

**MongoDB runs with `notablescan`.** A query without an indexed plan does not
return slowly, it *fails* — `NoQueryExecutionPlans`. Every find here therefore
carries `drawing_id` and `layout`, which is the `drawing_id_1_layout_1` index.
This is why `layout` is required rather than inferred: an unscoped geometry
query is not a slow query in this deployment, it is a broken one.

**Reprojection is a display concession, not a measurement.** `crs.py` refuses
to reproject geometry on purpose — reprojecting a ring changes its side
lengths and its area, and every area figure in this project is computed in
drawing coordinates. `frame=lnglat` therefore reprojects **vertices for
drawing only**, and every `area` and `perimeter` in the response stays in
drawing units, carrying `area_unit` to say so. A consumer that measures a
reprojected ring is measuring the wrong thing, and the response says so in
`frame_basis` rather than leaving it to be discovered.

**Coverage is narrow, and silence about that would read as absence.** Rings
exist for closed LWPOLYLINEs. INSERTs, ARCs, CIRCLEs, HATCHes and SOLIDs carry
no vertices at all — 42 INSERTs on Janadriyah's Model are 42 documents with a
bounding box and nothing to draw. `geometry_coverage` reports that per type,
so a caller can tell "this layout has no buildings" from "this endpoint cannot
give you the buildings". §3.2 of the note calls the second half phase 5.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final, Iterable, Mapping, NamedTuple, NoReturn, Sequence

from fastapi import APIRouter, Query, Request, Response

from . import geo_view, geometry, mongo, store, store_landuse
from .extract import RING_TYPES
from . import landuse
from .landuse import crs as crs_math

log = logging.getLogger(__name__)


def _scope_for(layout: str) -> dict[str, Any]:
    """The shared scope rule, so the census counts what the feed draws."""
    from . import geo_h3  # noqa: PLC0415 -- deferred, one direction only

    return geo_h3.scope_match(layout)

router = APIRouter(tags=["map"])


#: The kinds of geometry this endpoint can return, and what each one is.
#:
#: `label` is off by default. Janadriyah's Model carries 5,518 TEXT, which is
#: nearly twice the geometry count — asked for unconditionally it would double
#: the payload to draw annotations most callers do not want.
KINDS: Final[tuple[str, ...]] = ("ring", "path", "segment", "label")
DEFAULT_KINDS: Final[str] = "ring,path,segment"

#: Measured: 2,885 rings / 20,212 vertices on Janadriyah's Model fetch in
#: ~950 ms and project in ~90 ms. The cap is set well above that so the
#: densest layout in the corpus arrives in one call, because a map that
#: paginates draws a partial city and looks like a rendering fault.
DEFAULT_LIMIT: Final[int] = 20_000
MAX_LIMIT: Final[int] = 60_000

#: Fields read out of `autocad_entities`. Listed rather than fetching whole
#: documents: `attribs` and `h3_cells` are large and irrelevant here, and a
#: projection is the difference between a 12 MB and a 3 MB read.
_PROJECTION: Final[dict[str, int]] = {
    "handle": 1,
    "type": 1,
    "layer": 1,
    "block_name": 1,
    "text": 1,
    "ring": 1,
    "ring_status": 1,
    "ring_orientation": 1,
    "ring_vertex_count": 1,
    "path_points": 1,
    "path_bulges": 1,
    "path_points_basis": 1,
    "polygon_centroid": 1,
    "bbox": 1,
    "bbox_centre": 1,
    "anchor_point": 1,
    "anchor_basis": 1,
    "ends": 1,
    "ends_basis": 1,
    "area": 1,
    "length": 1,
    "perimeter_from_ring": 1,
    "shape_key": 1,
}

#: Types whose `anchor_point` is worth returning as a label.
_LABEL_TYPES: Final[frozenset[str]] = frozenset({"TEXT", "MTEXT", "ATTDEF"})

#: Types whose whole geometry is in `ends`.
#:
#: LINE is the exact case and the reason this kind exists: `ends` is
#: `LINE.start`/`LINE.end` in WCS, which IS the line. 16,011 of them across the
#: corpus were stored and unreachable, 620 on Janadriyah's Model — 513 of those
#: on the road centreline layer, which is why that network drew as 21% of
#: itself.
#:
#: ARC WAS the approximate case, and from ingest version 8 it is not. The
#: extractor keeps the centre, radius and both angles it already had in hand
#: and flattens the curve to points within `geometry.ARC_CHORD_TOLERANCE`, so
#: an ARC arrives here carrying `path_points` and is drawn as the curve it is.
#: `_kind_of` picks `path` before `segment`, so `is_chord` goes false for it
#: on its own.
#:
#: An ARC ingested at version 7 or earlier still has only its two endpoints
#: and is still reported as a chord. That is why the flag stays rather than
#: being deleted: the store holds both, and a reader must be able to tell
#: which one it has.
_SEGMENT_TYPES: Final[frozenset[str]] = frozenset({"LINE", "ARC"})

#: `ring_status` when the extractor REFUSED to write a ring.
#:
#: A closed polyline containing an arc span. It has no `ring`, no
#: `path_points`, and a two-point `ring_diagnostic` — so it is completely
#: invisible to this endpoint while its type, LWPOLYLINE, is one that mostly
#: does have outlines. That combination is why it needs counting by name:
#: `types_with_no_outline` cannot report it, and 1,291 shapes corpus-wide were
#: silently absent from a coverage report that looked complete.
_RING_REFUSED: Final[str] = "bulge"


def _refuse(code: str, message: str, hint: str, status: int = 400) -> NoReturn:
    """Raise `main.ApiError`, imported at call time.

    Deferred on purpose. `main` imports this module to mount the router, so a
    module-level import here would be a cycle. Inside a function body the
    import runs after `main` has finished defining `ApiError`, and the error
    then travels through the handler already registered there — so a failure
    from this route reads exactly like a failure from any other route, which
    is the whole point of the `{error, message, hint}` taxonomy.
    """
    from .main import ApiError

    raise ApiError(code, message, hint, status)


#: Fields of the drawing document this module never reads, and which are most
#: of its weight. Measured on Sedra: `block_placement` 64.2 KB, `blocks`
#: 37.0 KB, `counts_by_layer` 31.2 KB and the style tables 15.3 KB, out of
#: 387.5 KB. `layers` (147.8 KB) and `layouts` (90.6 KB) ARE read -- by
#: `_layers_off` and by the layout check -- so they stay.
_DRAWING_UNUSED_HERE: Final[dict[str, int]] = {
    "block_placement": 0,
    "blocks": 0,
    "counts_by_layer": 0,
    "linetypes": 0,
    "text_styles": 0,
    "dim_styles": 0,
}


def _require_drawing(drawing_id: str) -> Mapping[str, Any]:
    """The drawing document, or a refusal naming where to get a valid id.

    Projected: this route reads a handful of fields from a document that is
    387 KB on Sedra, and fetching the rest cost about a second of every
    request for nothing.
    """
    doc = mongo.coll(mongo.COLL_DRAWINGS).find_one(
        {"_id": drawing_id}, _DRAWING_UNUSED_HERE
    )
    if not doc:
        _refuse(
            "DRAWING_NOT_FOUND",
            f"No drawing with id {drawing_id!r}.",
            "GET /drawings lists every ingested drawing with its id.",
            status=404,
        )
    return doc


def _require_layout(drawing: Mapping[str, Any], layout: str) -> None:
    """Refuse a layout this drawing does not have, and name the ones it does.

    A typo'd layout would otherwise return an empty feature list, which reads
    as "this drawing has no geometry" — the single most misleading answer this
    endpoint could give.
    """
    names = [str(l.get("name")) for l in (drawing.get("layouts") or [])]
    if layout in names:
        return
    shown = ", ".join(repr(n) for n in names[:12])
    more = f" (+{len(names) - 12} more)" if len(names) > 12 else ""
    _refuse(
        "LAYOUT_NOT_FOUND",
        f"Drawing {drawing.get('_id')!r} has no layout named {layout!r}.",
        f"Layouts on this drawing: {shown}{more}. "
        "Geometry is stored under the layout that OWNS the entities — for a "
        "drawing whose sheets project model space through a viewport that is "
        "'Model', not the sheet name.",
        status=404,
    )


def _require_kinds(raw: str) -> tuple[str, ...]:
    asked = tuple(k.strip() for k in raw.split(",") if k.strip())
    unknown = [k for k in asked if k not in KINDS]
    if unknown or not asked:
        _refuse(
            "BAD_KINDS",
            f"kinds={raw!r} is not a list of known geometry kinds.",
            f"Use a comma-separated subset of {','.join(KINDS)}. "
            "'ring' is a closed polygon, 'path' an open polyline, 'label' the "
            "anchor point of a piece of text.",
        )
    return asked


def _require_frame(raw: str, *, crs: crs_math.Crs | None) -> str:
    """Resolve `frame=auto`, and refuse `lnglat` when nothing can project it.

    The refusal is the important half. A drawing with no `crs:` block asked for
    lng/lat could be answered with raw eastings dressed up as degrees, and
    those look like valid coordinates — they would put Janadriyah in the Gulf
    of Guinea rather than fail. `crs.py` refuses to guess a zone for the same
    reason, and this refusal is that one, one layer up.
    """
    if raw not in ("auto", "world", "lnglat"):
        _refuse(
            "BAD_FRAME",
            f"frame={raw!r} is not a coordinate frame this endpoint knows.",
            "Use frame=world (the drawing's own coordinates), frame=lnglat "
            "(WGS 84 degrees, for a basemap), or frame=auto to get lnglat "
            "where the drawing has a CRS and world where it does not.",
        )
    if raw == "auto":
        return "lnglat" if crs is not None else "world"
    if raw == "lnglat" and crs is None:
        _refuse(
            "CRS_NOT_CONFIGURED",
            "This drawing has no CRS, so its coordinates cannot be turned "
            "into longitude and latitude.",
            "Use frame=world and draw it in its own coordinates without a "
            "basemap. To place it on a map, add a `crs:` block to "
            "cad_api/app/landuse/<drawing_id>.yaml — and note that a zone "
            "must be established rather than guessed: the wrong zone still "
            "produces coordinates that look valid.",
            status=409,
        )
    return raw


#: A viewport, in the frame the answer is expressed in.
_BBOX_ORDER: Final[str] = "minX,minY,maxX,maxY"


def _require_bbox(raw: str | None, *, frame: str) -> tuple[float, float, float, float] | None:
    """A `bbox=` parameter, or `None` when there is none.

    **The box is in the frame the answer comes back in**, which is the one
    place this deliberately differs from `/geo`. That route is only ever
    georeferenced, so its box is always degrees. This one answers in drawing
    coordinates for the seventeen drawings that have no CRS, and a box that
    had to be in degrees would be unusable for exactly the drawings that need
    it most. `viewport.frame` says which it was read as, so a client is never
    left guessing.

    Malformed is a refusal, not an empty result. A box that silently parsed to
    nothing would return an empty payload that looks exactly like a viewport
    over empty ground.
    """
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        _refuse(
            "BAD_BBOX",
            f"bbox needs four numbers, got {len(parts)}.",
            f"Pass {_BBOX_ORDER} — "
            + (
                "west,south,east,north in degrees, because this answer is in "
                "lng/lat."
                if frame == "lnglat"
                else "in the drawing's own units, because this answer is in "
                "drawing coordinates."
            ),
        )
    try:
        min_x, min_y, max_x, max_y = (float(p) for p in parts)
    except ValueError:
        _refuse(
            "BAD_BBOX",
            f"bbox={raw!r} is not four numbers.",
            f"Pass {_BBOX_ORDER} as plain decimals.",
        )
    if not (max_x > min_x and max_y > min_y):
        _refuse(
            "BAD_BBOX",
            "bbox has no area: its maximum must be greater than its minimum "
            "on both axes.",
            f"Pass {_BBOX_ORDER}. A zero-width box is not a viewport showing "
            "nothing, it is a box that cannot be drawn.",
        )
    return (min_x, min_y, max_x, max_y)


def _extent_in_frame(
    row: Mapping[str, Any],
    kind: str,
    *,
    frame: str,
    crs: crs_math.Crs | None,
) -> tuple[float, float, float, float] | None:
    """One row's extent in the frame the answer is in, for a viewport test.

    Read from the stored `bbox` where there is one, because it is already
    computed and is what every other route measures extents with. Where there
    is not, it is derived from the row's own vertices rather than the row
    being dropped.

    For lng/lat the four corners are projected and their min/max taken. That
    is a SUPERSET of the true projected extent — the projection turns an
    axis-aligned rectangle into a slightly rotated quadrilateral — and the
    direction of that error is the safe one: a viewport may keep something
    marginally outside it, and will never drop something inside it.
    """
    box = row.get("bbox") or {}
    low, high = box.get("min"), box.get("max")
    if low and high:
        corners = [
            [float(low[0]), float(low[1])],
            [float(high[0]), float(low[1])],
            [float(high[0]), float(high[1])],
            [float(low[0]), float(high[1])],
        ]
    else:
        corners = _coordinates(row, kind)
    if not corners:
        return None
    if frame == "lnglat":
        if crs is None:  # pragma: no cover - the frame guard rules this out
            return None
        corners = _project(corners, crs)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _meets(
    extent: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float],
) -> bool:
    """Whether an extent overlaps a viewport at all.

    Overlap, not containment. A road that crosses the view is in the view; a
    parcel larger than the box contains it and is in it too. Requiring
    containment would empty the screen at exactly the zoom where somebody is
    looking closely at one thing.
    """
    return (
        extent[0] <= bbox[2]
        and extent[2] >= bbox[0]
        and extent[1] <= bbox[3]
        and extent[3] >= bbox[1]
    )


class LayerFilter(NamedTuple):
    """Which layers to draw, as the caller expressed it.

    Inclusion AND exclusion, in the one parameter, because the two questions
    have very different sizes on a real drawing. Isolating three layers is a
    short list. Hiding ONE layer out of 215 -- which is the interesting case,
    since Sedra's `EXT-BASE` is 440,340 of the scope's 610,944 objects and
    crowds everything else out of any capped fetch -- would need the other 214
    names spelled out. Measured on Sedra's layer names, url-encoded, that is
    about 12 KB of query string, past what the server will accept on a request
    line. `!NAME` says the same thing in one term.
    """

    include: list[str]
    exclude: list[str]


def _layer_filter(raw: str | None) -> LayerFilter | None:
    """`A,B` draws only A and B; `!A` draws everything except A.

    A name that really begins with `!` can be escaped as `!!`. That is not
    hypothetical tidiness -- AutoCAD layer names permit almost anything, and a
    filter that could not name such a layer would silently draw the wrong
    thing rather than fail.
    """
    if raw is None:
        return None
    include: list[str] = []
    exclude: list[str] = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        if token.startswith("!!"):
            include.append(token[1:])
        elif token.startswith("!"):
            exclude.append(token[1:])
        else:
            include.append(token)
    if not include and not exclude:
        return None
    return LayerFilter(include=include, exclude=exclude)



#: `total_matches` for one (drawing, scope, filter), remembered.
#:
#: The total means the whole layout, so it does not change from page to page,
#: and counting it does not belong on every page: measured, `count_documents`
#: over 610,944 matching rows is 2.2 s, which was most of what a paged
#: request cost once the page itself came from an index. Keyed by the
#: drawing's own precondition token, so a re-ingest or a re-index invalidates
#: it rather than serving a stale count for the previous version of a drawing.
_TOTALS: dict[tuple[Any, ...], int] = {}
_TOTALS_MAX = 64


def _total_matches(entities: Any, query: Mapping[str, Any], token: Any) -> int:
    """The whole layout's match count, counted once per version."""
    key = (token, json.dumps(query, sort_keys=True, default=str))
    remembered = _TOTALS.get(key)
    if remembered is not None:
        return remembered
    total = entities.count_documents(dict(query))
    if len(_TOTALS) >= _TOTALS_MAX:
        # Bounded rather than unbounded: this is a convenience, not a store,
        # and a process that never forgets a drawing it served once is a leak.
        _TOTALS.clear()
    _TOTALS[key] = total
    return total

#: Layer lists per (version, scope). Same bounded-cache reasoning as
#: `_TOTALS`: measured at 2.38 s over Sedra's 679,897 in-scope rows and 0.08 s
#: over Janadriyah's 20,334, which is worth paying once and not per page.
_LAYERS: dict[tuple[Any, str], list[dict[str, Any]]] = {}
_LAYERS_MAX = 32


def layers_in_scope(
    entities: Any, query: Mapping[str, Any], token: Any
) -> list[dict[str, Any]]:
    """Every layer PRESENT IN THE SCOPE, with how many objects it holds there.

    The membership question, answered from the store rather than from whatever
    pages a client happens to have fetched. A panel that built its list from
    fetched features grew new rows as paging proceeded, which reads as the
    drawing changing under the user; and it labelled the drawing-wide count as
    the layer's total, so `XR_BUILDING FOOTPRINT` showed "of 22,943" when
    22,440 of those are in the map's scope and 503 are not.

    Both facts are needed and they are different: this is the denominator, and
    the numerator is counted from the features actually on screen.
    """
    key = (token, json.dumps(dict(query), sort_keys=True, default=str))
    remembered = _LAYERS.get(key)
    if remembered is not None:
        return remembered
    rows = list(
        entities.aggregate(
            [
                {"$match": dict(query)},
                {"$group": {"_id": "$layer", "n": {"$sum": 1}}},
                {"$sort": {"n": -1, "_id": 1}},
            ],
            allowDiskUse=True,
        )
    )
    out = [
        {"name": str(r["_id"]) if r["_id"] is not None else "", "count": int(r["n"])}
        for r in rows
    ]
    if len(_LAYERS) >= _LAYERS_MAX:
        _LAYERS.clear()
    _LAYERS[key] = out
    return out


def scope_layer_counts(
    entities: Any, drawing_id: str, layout: str, kinds: Sequence[str]
) -> list[dict[str, Any]]:
    """The scope's per-layer counts, for the layers route AND for totals.

    One owner so the two share a cache entry and can never disagree. The
    filter is deliberately the UNFILTERED one: this describes the scope, and a
    caller narrowing to some layers is asking a question about that scope, not
    redefining it.
    """
    return layers_in_scope(
        entities,
        _mongo_filter(drawing_id, layout, kinds, None),
        geo_view.etag_for(
            drawing_id,
            **geo_view.preconditions(drawing_id),
            params={
                "route": "geometry/layers",
                "layout": layout,
                "kinds": ",".join(kinds),
            },
        ),
    )


def keeps_layer(name: str, chosen: "LayerFilter | None") -> bool:
    """Whether `name` survives the filter, by the same rule Mongo applies."""
    if chosen is None:
        return True
    if chosen.include and name not in chosen.include:
        return False
    return name not in chosen.exclude


def _mongo_filter(
    drawing_id: str, layout: str, kinds: Sequence[str], layers: LayerFilter | None
) -> dict[str, Any]:
    """The query, always carrying the `drawing_id + layout` index prefix.

    See the module docstring on `notablescan`: the `$or` narrows an already
    indexed scan, and would fail outright as a query on its own.
    """
    # Each clause pairs its geometry field with the entity TYPES that can
    # carry it, because `{"$ne": None}` alone is not something an index can
    # answer: finding 500 rings meant fetching about 21,700 documents and
    # testing each one, measured at 4.9 s on a 990,790 row drawing. `type` is
    # indexed with `drawing_id`, and the extractor only ever writes `ring` and
    # `path_points` for RING_TYPES (see `extract.iter_entities`), so the pair
    # is a narrowing that cannot change the answer. `segment` and `label`
    # already did this; `ring` and `path` did not.
    wanted: list[dict[str, Any]] = []
    if "ring" in kinds:
        wanted.append({"ring": {"$ne": None}, "type": {"$in": sorted(RING_TYPES)}})
    if "path" in kinds:
        wanted.append(
            {"path_points": {"$ne": None}, "type": {"$in": sorted(RING_TYPES)}}
        )
    if "segment" in kinds:
        wanted.append(
            {"ends": {"$ne": None}, "type": {"$in": sorted(_SEGMENT_TYPES)}}
        )
    if "label" in kinds:
        wanted.append(
            {"anchor_point": {"$ne": None}, "type": {"$in": sorted(_LABEL_TYPES)}}
        )
    from . import geo_h3  # noqa: PLC0415 -- deferred, one direction only

    # `$and` rather than merging: the scope rule may itself be an `$or`, and
    # `wanted` already uses one. Two `$or` keys in one dict is a silent
    # overwrite, and the kind that drops half a query rather than failing.
    clauses: list[dict[str, Any]] = [geo_h3.scope_match(layout)]
    if wanted:
        clauses.append({"$or": wanted})
    query: dict[str, Any] = {"drawing_id": drawing_id, "$and": clauses}
    if layers:
        clause: dict[str, Any] = {}
        if layers.include:
            clause["$in"] = layers.include
        if layers.exclude:
            clause["$nin"] = layers.exclude
        # Both together are an intersection and mean what they look like:
        # "these layers, minus those". `drawing_id` still leads the index, so
        # this narrows an indexed scan rather than becoming one of its own --
        # the same reasoning as the `$or` in the module docstring.
        query["layer"] = clause
    return query


def _kind_of(row: Mapping[str, Any], kinds: Sequence[str]) -> str | None:
    """Which kind this document answers as, in a fixed order of preference.

    A closed polyline carries `ring` and never `path_points` — the extractor
    writes one or the other, not both — so the order only decides what a
    future document carrying both would be reported as, and 'a closed shape'
    is the more useful of the two answers.
    """
    if "ring" in kinds and row.get("ring"):
        return "ring"
    if "path" in kinds and row.get("path_points"):
        return "path"
    if (
        "segment" in kinds
        and row.get("ends")
        and row.get("type") in _SEGMENT_TYPES
    ):
        return "segment"
    if "label" in kinds and row.get("anchor_point"):
        return "label"
    return None


def _coordinates(row: Mapping[str, Any], kind: str) -> list[list[float]]:
    if kind == "ring":
        return [[float(p[0]), float(p[1])] for p in row["ring"]]
    if kind == "path":
        return [[float(p[0]), float(p[1])] for p in row["path_points"]]
    if kind == "segment":
        return [[float(p[0]), float(p[1])] for p in row["ends"]]
    anchor = row["anchor_point"]
    return [[float(anchor[0]), float(anchor[1])]]


def _for_zoom(
    coords: list[list[float]], kind: str, zoom: float | None
) -> list[list[float]]:
    """The same shape, carrying only the vertices this zoom can resolve.

    Version 8 stored the curve instead of its chord, which is right and is not
    undone here -- but it made features fatter: measured on Atlas, 875 bytes
    per feature on Janadriyah and 1,221 on Sedra against roughly 690 before.
    A curve seen from far away does not need thirty points to look like a
    curve, so the vertex count follows the camera instead of the store.

    Applied to what is SENT, never to what is stored. The store keeps the full
    curve and every other answer is computed from it; a coarser view is a
    view, not a loss.

    Only for two-point-or-more line work, and only where a `zoom` was given.
    A `segment` is two points and cannot be reduced; a `label` is one.
    """
    if zoom is None or kind not in ("ring", "path") or len(coords) < 3:
        return coords
    # The feature's own latitude, so a drawing spanning degrees is not
    # simplified against a single distant reference.
    tolerance = geometry.degrees_per_pixel(zoom, coords[0][1]) * geometry.SIMPLIFY_PIXELS
    kept = geometry.simplify([(c[0], c[1]) for c in coords], tolerance)
    return [[x, y] for x, y in kept]


def _project(points: Iterable[Sequence[float]], crs: crs_math.Crs) -> list[list[float]]:
    """Vertices as `[lon, lat]`.

    **Longitude first.** `crs.to_lat_lon()` returns `(lat, lon)` and says so
    loudly, because that order gets swapped in every project that has used
    both. This function swaps it exactly once, here, because GeoJSON and
    deck.gl both want `[lon, lat]` and a consumer that has to remember which
    way round a payload is will eventually remember wrong.
    """
    out: list[list[float]] = []
    for p in points:
        lat, lon = crs.to_lat_lon(float(p[0]), float(p[1]))
        out.append([lon, lat])
    return out


def _bounds(points: Iterable[Sequence[float]]) -> list[list[float]] | None:
    """[[min x, min y], [max x, max y]] over every vertex, or None if empty."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    seen = False
    for x, y in points:
        seen = True
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    if not seen:
        return None
    return [[min_x, min_y], [max_x, max_y]]


#: Coverage per (version, drawing, layout). Same bounded-cache reasoning as
#: `_TOTALS` and `_LAYERS`, and the largest of the three by far.
_COVERAGE: dict[tuple[Any, str, str], dict[str, Any]] = {}
_COVERAGE_MAX = 32


#: Content boxes per (version, drawing, layout). Same bounded-cache reasoning
#: as `_COVERAGE`: it is a summary of the drawing, not of the request.
_CONTENT: dict[tuple[Any, str, str], dict[str, Any] | None] = {}
_CONTENT_MAX = 32


def content_bounds(drawing_id: str, layout: str, token: Any = None) -> dict[str, Any] | None:
    """Where the drawing IS, as opposed to where its extreme corners are.

    Built from the H3 cells already stored, so it needs no new machinery: the
    smallest set of coarse cells holding `CONTENT_SHARE` of the entities in
    scope, and the lng/lat box those cells cover.

    Measured on Atlas: Janadriyah's estate is 10 of its 27 coarse cells, and
    the 17 dropped ones are the `NBHD *` boundaries that are real, correctly
    placed, and up to 12 km away -- which had been shrinking the thing a
    person came to look at into a few pixels. Sedra is 75 of 141.

    Cached, because it is one aggregation over every in-scope entity: 3.02 s
    on Sedra, 0.27 s on Janadriyah. Computed once per drawing and scope, never
    per request.
    """
    from . import geo_h3  # noqa: PLC0415 -- deferred, one direction only

    if token is not None:
        if (token, drawing_id, layout) in _CONTENT:
            return _CONTENT[(token, drawing_id, layout)]
    rows = list(
        mongo.coll(mongo.COLL_ENTITIES).aggregate(
            [
                {
                    "$match": {
                        "drawing_id": drawing_id,
                        **_scope_for(layout),
                        "h3_coarse": {"$ne": None},
                    }
                },
                {"$group": {"_id": "$h3_coarse", "n": {"$sum": 1}}},
            ]
        )
    )
    counts = {str(r["_id"]): int(r["n"]) for r in rows if r.get("_id")}
    out: dict[str, Any] | None = None
    if counts:
        kept = geo_h3.cells_holding_most(counts)
        box = geo_h3.bounds_of_cells(kept)
        if box is not None:
            held = sum(counts[c] for c in kept)
            total = sum(counts.values())
            out = {
                "lnglat": box,
                "basis": "h3_content_cells",
                "cells_used": len(kept),
                "cells_total": len(counts),
                "entities_inside": held,
                "entities_total": total,
                "share": round(held / total, 4) if total else None,
                "note": (
                    f"the {len(kept)} of {len(counts)} coarse cells holding "
                    f"{held:,} of {total:,} placed objects. Objects outside it "
                    "are real and correctly placed -- a site boundary or key "
                    "plan parked far from the work -- and are still drawn; "
                    "they simply do not decide the camera"
                ),
            }
    if token is not None:
        if len(_CONTENT) >= _CONTENT_MAX:
            _CONTENT.clear()
        _CONTENT[(token, drawing_id, layout)] = out
    return out


def _coverage_token(drawing_id: str, layout: str) -> Any:
    """A token that changes with the DRAWING, never with the request.

    The first attempt passed the request's own ETag, which carries `limit` and
    `offset` -- so every page minted a fresh key and the cache never hit once.
    Coverage depends on the drawing and the layout and on nothing else in the
    request, and its key has to say exactly that.
    """
    # `ingested_at` as well as the version facts. A `--force` re-ingest lands
    # new entities under the SAME `INGEST_VERSION`, so a token built from the
    # version alone would keep serving the previous drawing's coverage until
    # the process restarted. The timestamp changes on every ingest, which is
    # exactly the event that can change these counts.
    version, ingested_at = store.version_facts_of(drawing_id)
    facts = dict(geo_view.preconditions(drawing_id))
    facts["ingest_version"] = version
    return geo_view.etag_for(
        drawing_id,
        **facts,
        params={
            "route": "geometry/coverage",
            "layout": layout,
            "ingested_at": str(ingested_at),
        },
    )


def _coverage(drawing_id: str, layout: str, token: Any = None) -> dict[str, Any]:
    """Per type: how many entities there are, and how many carry geometry.

    This is the honest half of the response. Without it a map showing plots
    and no buildings is indistinguishable from a site with no buildings, and
    the difference is the whole of phase 5.

    CACHED, because it is a property of the drawing and the layout and of
    nothing else in the request. It is a `$group` over every entity in the
    layout -- 679,897 rows in Sedra's model space -- and it ran on EVERY page.
    Measured on Atlas it is 4.69 s there and 0.28 s on Janadriyah, which is
    the whole of the fixed cost a request pays before it counts anything: a
    one-feature request took 4.4-4.8 s and a 4,000-feature request 8.6-11.2 s,
    and the difference between those two is the only part that depended on the
    answer. The client's six-page accumulation paid it six times, about 27
    seconds of repeating the same aggregation.

    This is the same lesson as the 2.2 s `count_documents` cached above, one
    order of magnitude larger and still present until now.
    """
    if token is not None:
        remembered = _COVERAGE.get((token, drawing_id, layout))
        if remembered is not None:
            return remembered
    pipeline = [
        {"$match": {"drawing_id": drawing_id, **_scope_for(layout)}},
        {
            "$group": {
                "_id": "$type",
                "entities": {"$sum": 1},
                "rings": {"$sum": {"$cond": [{"$ifNull": ["$ring", False]}, 1, 0]}},
                "paths": {
                    "$sum": {"$cond": [{"$ifNull": ["$path_points", False]}, 1, 0]}
                },
                "anchors": {
                    "$sum": {"$cond": [{"$ifNull": ["$anchor_point", False]}, 1, 0]}
                },
                "segments": {
                    "$sum": {"$cond": [{"$ifNull": ["$ends", False]}, 1, 0]}
                },
                "ring_refused": {
                    "$sum": {
                        "$cond": [{"$eq": ["$ring_status", _RING_REFUSED]}, 1, 0]
                    }
                },
            }
        },
        {"$sort": {"entities": -1}},
    ]
    rows = list(mongo.coll(mongo.COLL_ENTITIES).aggregate(pipeline))
    by_type = [
        {
            "type": r["_id"],
            "entities": r["entities"],
            "rings": r["rings"],
            "paths": r["paths"],
            # `ends` on an LWPOLYLINE duplicates its `path_points`, so it is
            # only a NEW outline where nothing else supplies one.
            "segments": (
                r["segments"] if r["_id"] in _SEGMENT_TYPES else 0
            ),
            "label_anchors": r["anchors"],
            "ring_refused": r["ring_refused"],
            "with_outline": (
                r["rings"]
                + r["paths"]
                + (r["segments"] if r["_id"] in _SEGMENT_TYPES else 0)
            ),
        }
        for r in rows
    ]
    entities = sum(r["entities"] for r in by_type)
    with_outline = sum(r["with_outline"] for r in by_type)
    refused = sum(r["ring_refused"] for r in by_type)
    missing = [r for r in by_type if r["with_outline"] == 0 and r["entities"] > 0]
    out = {
        "entities_in_layout": entities,
        "with_outline": with_outline,
        "without_outline": entities - with_outline,
        "by_type": by_type,
        "types_with_no_outline": [r["type"] for r in missing],
        # Counted separately because nothing else can see it. These are CLOSED
        # polylines containing an arc span, for which the extractor refuses to
        # write a `ring` at all — so they are invisible under a type,
        # LWPOLYLINE, that mostly does have outlines, and
        # `types_with_no_outline` can never name them.
        "rings_refused": refused,
        "rings_refused_reason": (
            "closed polylines containing an arc span. The extractor writes no "
            "`ring` for these and no `path_points` either, so nothing can draw "
            "them — they are not merely approximated, they are absent."
            if refused
            else None
        ),
        "note": (
            "Outlines come from closed and open LWPOLYLINEs, plus LINE and ARC. "
            "A LINE's endpoints ARE the line. An ARC ingested at version 8 or "
            "later carries its flattened curve and is drawn as one; an older "
            "one carries only endpoints and is flagged `is_chord` per feature. "
            "INSERT, CIRCLE, HATCH and SOLID "
            "carry a bounding box and no vertices, by the same design decision "
            "that makes a block placement one commentable object. Drawing "
            "those needs the ezdxf recorder pass described in "
            "docs/DECKGL-GEOMETRY-CONTRACT.md, which this endpoint "
            "deliberately does not do."
        ),
    }
    if token is not None:
        if len(_COVERAGE) >= _COVERAGE_MAX:
            # Bounded rather than unbounded: a convenience, not a store.
            _COVERAGE.clear()
        _COVERAGE[(token, drawing_id, layout)] = out
    return out


def _layers_off(drawing: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Layers this drawing keeps switched OFF or FROZEN, from its layer table.

    Returned with the geometry because the geometry is the only place a map
    consumer can learn it. The rendered SVG publishes the same fact in its
    `data-off-layers` attribute — but only for the layout it rendered, and a
    paper sheet whose viewport already excludes those layers reports none.
    A caller drawing model space while looking at a sheet therefore has no way
    to know, and draws eleven layers AutoCAD switches off.

    That is not hypothetical: on Janadriyah it is `00_Prop - Road - CL_` and
    every `NBHD *` boundary, which is 836 objects scattered up to 12 km from
    the estate. They are real, correctly placed, and off.

    The layer table is the right source because it is a property of the
    DRAWING, not of any one layout, so this answer does not change with the
    layout the caller happens to be showing.
    """
    return [
        {
            "name": str(layer.get("name")),
            "off": bool(layer.get("off")),
            "frozen": bool(layer.get("frozen")),
        }
        for layer in (drawing.get("layers") or [])
        if layer.get("off") or layer.get("frozen")
    ]


def _placement(
    drawing: Mapping[str, Any], crs: crs_math.Crs | None, frame: str
) -> dict[str, Any]:
    """Whether this drawing may be put on a map, and what to say if not.

    Three separate conditions, reported separately rather than collapsed into
    one boolean: a caller that knows *which* one failed can offer the right
    next step, and "cannot be placed" on its own sends people to the wrong
    one.
    """
    units_name = drawing.get("units_name")
    units_code = drawing.get("units_code")
    metric = units_name == "m"
    reasons: list[str] = []
    if crs is None:
        reasons.append(
            "the drawing has no `crs:` block, so nothing establishes where in "
            "the world its coordinates sit"
        )
    if not metric:
        reasons.append(
            f"the drawing's units are {units_name or 'undeclared'} "
            f"(code {units_code}), and the UTM inverse in crs.py reads its "
            "input as metres"
        )
    return {
        "can_georeference": crs is not None and metric,
        "why_not": reasons or None,
        "units_name": units_name,
        "units_code": units_code,
        "units_declared_in_file": bool(units_name),
        "frame_in_use": frame,
        "fallback": (
            "frame=world draws the geometry in the drawing's own coordinates "
            "with no basemap. That is a real view of the drawing; what it "
            "cannot do is say where on Earth it is."
        ),
    }


@router.get("/drawings/{drawing_id}/geometry/layers")
def drawing_geometry_layers(
    drawing_id: str,
    layout: str = Query(
        ...,
        description=(
            "The layout the entities are STORED under, exactly as "
            "`/geometry` takes it."
        ),
    ),
    kinds: str = Query(
        DEFAULT_KINDS,
        description=(
            "Comma-separated subset of ring,path,label. The same default as "
            "`/geometry`, so the list describes the same population the map "
            "is drawing from."
        ),
    ),
) -> dict[str, Any]:
    """Every layer present in this SCOPE, with its total there.

    Separate from `/geometry` on purpose. It is one aggregation over the whole
    scope -- 2.38 s across Sedra's 679,897 in-scope rows, 0.08 s across
    Janadriyah's 20,334 -- so putting it on every page would pay it per page
    for an answer that does not change between them. Cached per version and
    scope; a client asks once.

    It exists because the layer panel had no scope-stable source for its list.
    It built one from the features it had fetched, so rows appeared as paging
    proceeded, and it took the layer's DRAWING-wide count as the total, which
    is a different number: `XR_BUILDING FOOTPRINT` holds 22,943 in the file
    and 22,440 in the map's scope.
    """
    _require_drawing(drawing_id)
    entities = mongo.coll(mongo.COLL_ENTITIES)
    rows = scope_layer_counts(entities, drawing_id, layout, _require_kinds(kinds))
    return {
        "drawing_id": drawing_id,
        "layout": layout,
        "kinds": kinds,
        "layers": rows,
        "total": sum(r["count"] for r in rows),
        # What the list is OF, in the words a panel can put on screen. The
        # scope is not the sheet the layout picker names: since A2.1 the map
        # draws the drawing's world-placed geometry, and a list that silently
        # described something else is how the picker came to look broken.
        "scope": "world_placed",
        "scope_label": "the drawing's world-placed geometry",
        "scope_note": (
            "these are the layers the MAP can draw, which is model space plus "
            "every block the drawing places at identity. It is not the layer "
            "list of the sheet named in the layout picker"
        ),
        "membership_is": (
            "every layer present in this scope, counted in the store. It does "
            "not change as the client pages through the geometry"
        ),
    }


@router.get("/drawings/{drawing_id}/geometry")
def drawing_geometry(
    drawing_id: str,
    layout: str = Query(
        ...,
        description=(
            "Required. The layout the entities are STORED under — 'Model' for "
            "a drawing whose sheets project model space through a viewport. "
            "Required rather than inferred because MongoDB runs with "
            "notablescan here: an unscoped geometry query does not return "
            "slowly, it fails."
        ),
    ),
    frame: str = Query(
        "auto",
        description="'auto', 'world' (drawing coordinates) or 'lnglat' (WGS 84 degrees).",
    ),
    kinds: str = Query(
        DEFAULT_KINDS,
        description="Comma-separated subset of ring,path,label. 'label' is off by default.",
    ),
    layers: str | None = Query(
        None, description="Comma-separated layer names. Omit for every layer."
    ),
    have: str | None = Query(
        None,
        description=(
            "Comma-separated coarse H3 cells the caller ALREADY holds. The "
            "answer is restricted to the cells in `bbox` MINUS these, so a pan "
            "asks only for what is newly in view instead of re-fetching what "
            "is already painted. The caller sends what it has rather than what "
            "it wants because deciding which cells a box covers needs H3, and "
            "that lives here. Ignored without a `bbox`. `cells_served` in the "
            "response says what came back, which is what to accumulate."
        ),
    ),
    zoom: float | None = Query(
        None,
        ge=0,
        le=24,
        description=(
            "The map's current Web Mercator zoom. Given, vertices are dropped "
            "where they are closer together than the screen can show -- under "
            "a pixel of error, so the reduction cannot be seen. Omitted, every "
            "stored vertex is sent. Only affects `frame=lnglat`."
        ),
    ),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    bbox: str | None = Query(
        None,
        description=(
            "Restrict `features` to a viewport, as `minX,minY,maxX,maxY` IN "
            "THE FRAME THIS ANSWER COMES BACK IN — degrees when that is "
            "lnglat, the drawing's own units when it is world. `total_matches` "
            "is NOT restricted: it stays the whole layout, so a filtered view "
            "can never be read as the drawing. See `viewport` in the response."
        ),
    ),
    request: Request = None,  # type: ignore[assignment]
    response: Response = None,  # type: ignore[assignment]
) -> Any:
    """Vertex geometry for one layout, ready to hand to a deck.gl layer.

    The response is deliberately one object rather than GeoJSON. GeoJSON has
    nowhere to put the three things that decide whether the picture may be
    believed — which frame the numbers are in, whether the CRS was *declared*
    or *inferred*, and what fraction of the layout carries no geometry at all
    — and a `FeatureCollection` with those bolted into `properties` is a
    convention nobody else reads anyway. Every field here is already the shape
    the rest of cad-api uses: `total_matches`/`truncated` on the list, a
    `basis` on anything computed, and `None` plus a reason rather than a zero.

    Ordering is by handle, so `offset` paginates a stable list rather than
    whatever order the index happened to return.
    """
    drawing = _require_drawing(drawing_id)
    # ONCE per request. Each call is two projections against Atlas -- 0.44 s
    # measured -- and it was being recomputed for the coverage cache, the
    # content box and the total, so the request paid it three times for an
    # answer that cannot differ between them.
    version_token = _coverage_token(drawing_id, layout)
    _require_layout(drawing, layout)
    wanted_kinds = _require_kinds(kinds)
    # This route is also called as a plain Python function -- several suites
    # do exactly that rather than going through the ASGI stack -- and then an
    # unpassed default arrives as FastAPI's own `Query(...)` object instead of
    # the value it wraps. A number is the only thing the simplifier can use,
    # so anything else means "not asked for". The `/geo` route learned this
    # same lesson with its `dimension` parameter; the shape repeats because
    # the calling convention does.
    zoom = zoom if isinstance(zoom, (int, float)) and not isinstance(zoom, bool) else None
    held_cells: set[str] = set()
    if isinstance(have, str) and have.strip():
        held_cells = {c for c in (x.strip() for x in have.split(",")) if c}
    nothing_new = False

    layer_filter = _layer_filter(layers)

    try:
        config = landuse.for_drawing(drawing_id)
    except landuse.ConfigError as exc:
        # Re-raised through the existing land-use handler's taxonomy rather
        # than swallowed: a config this endpoint cannot read is the same fault
        # `/land-use` would report, and reporting it twice differently is how
        # two answers to one question start.
        raise store_landuse.LandUseConfigInvalid(
            exc.code, exc.message, exc.hint
        ) from exc
    crs = config.crs if config else None
    frame_in_use = _require_frame(frame, crs=crs)
    view = _require_bbox(bbox, frame=frame_in_use)

    # The cache key, computed BEFORE the read rather than from its result.
    # That ordering is the whole value of it: a client that already holds this
    # answer gets a 304 having cost two small documents, instead of paying for
    # 20,334 rings to be read and reprojected so the bytes can be thrown away.
    #
    # `geo_view`'s implementation rather than a second one. The header parsing
    # in particular is not worth writing twice — `If-None-Match` is a list,
    # may be `*`, and may carry `W/` prefixes, and a second version that
    # handled one client would silently stop working for the next.
    # `stored_resolution` comes along in `preconditions` and is irrelevant to
    # geometry; carrying it can only invalidate this cache slightly more often
    # than strictly needed, which is the safe direction.
    tag = geo_view.etag_for(
        drawing_id,
        **geo_view.preconditions(drawing_id),
        params={
            "route": "geometry",
            "layout": layout,
            "frame": frame_in_use,
            "kinds": ",".join(wanted_kinds),
            # Both halves, and normalised, so `A,B` and `B,A` are one cache
            # entry while `A` and `!A` can never collide into one.
            "layers": (
                {
                    "include": sorted(layer_filter.include),
                    "exclude": sorted(layer_filter.exclude),
                }
                if layer_filter
                else None
            ),
            "limit": limit,
            "offset": offset,
            # In the ETag because they change the body. Same rule the
            # `dimension` parameter had to learn.
            "zoom": zoom,
            "have": sorted(held_cells) if held_cells else None,
            "bbox": list(view) if view else None,
        },
    )
    if request is not None and geo_view.etag_matches(
        request.headers.get("if-none-match"), tag
    ):
        return Response(
            status_code=304, headers={"ETag": tag, "Cache-Control": "no-cache"}
        )
    if response is not None:
        response.headers["ETag"] = tag
        # `no-cache` does NOT mean "do not cache" — it means "cache this, but
        # revalidate before reusing it". Without it a browser stores nothing
        # and never sends `If-None-Match`, so the ETag above is computed,
        # served, and ignored. Measured before this line existed: returning to
        # a drawing already visited this session refetched 6.3 MB and got a
        # 200 every time.
        response.headers["Cache-Control"] = "no-cache"

    entities = mongo.coll(mongo.COLL_ENTITIES)
    query = _mongo_filter(drawing_id, layout, wanted_kinds, layer_filter)

    # `_id` is `f"{drawing_id}:{handle}"` and `drawing_id` is fixed by the
    # query, so ordering by `_id` is the SAME order as ordering by handle --
    # and it is the primary key, so the server does it from an index instead
    # of materialising the whole layout to sort it in this process.
    # A viewport, expressed as the coarse cells covering it. This is what
    # makes "what is on screen" an indexed question: the box is already in
    # degrees, H3 turns degrees into cells with no projection, and every
    # entity carries the coarse cell its fine cell belongs to. Only for
    # lng/lat answers -- a drawing with no CRS has no cells to match.
    viewport_cells: list[str] | None = None
    scoped = query
    if view is not None and frame_in_use == "lnglat":
        from . import geo_h3 as _gh  # noqa: PLC0415 -- deferred, one direction

        viewport_cells = _gh.cells_for_viewport(view[0], view[1], view[2], view[3])
        if held_cells and viewport_cells:
            # Only what is NEWLY in view. Subtraction, never addition: the
            # caller's list can shrink the answer and can never widen it past
            # the box it arrived with.
            fresh = [c for c in viewport_cells if c not in held_cells]
            if not fresh:
                # Everything on screen is already painted. An empty page is
                # the correct answer and costs a comparison instead of a
                # query -- but `total_matches` still describes the layout, so
                # it cannot be read as "the drawing has nothing here".
                nothing_new = True
            viewport_cells = fresh
        # ONLY where the drawing carries the field the filter matches on.
        #
        # `h3_coarse` is written at indexing time. A drawing indexed before it
        # existed has `h3_cell` and no `h3_coarse`, and an `$in` against a
        # field that is absent everywhere matches NOTHING -- so a viewport
        # request would answer a perfectly good drawing with an empty map and
        # no reason given. Falling back to the whole layout is slower and
        # right, and `viewport_applied` below says which happened.
        if viewport_cells and not entities.count_documents(
            {**query, "h3_coarse": {"$ne": None}}, limit=1
        ):
            viewport_cells = None
        if viewport_cells:
            # `scoped` narrows to the view; `query` stays the whole layout,
            # because `total_matches` means the layout and always has. Two
            # names rather than one mutated dict, so a page can never be
            # counted as the drawing.
            scoped = {**query, "h3_coarse": {"$in": viewport_cells}}

    if nothing_new:
        # Every cell on screen is already in the caller's hands. Answering
        # with an empty page costs one comparison instead of a query, and it
        # is the honest answer -- but `total_matches` still describes the
        # LAYOUT, so nobody can read this as "the drawing has nothing here".
        page = []
        rows = []
        in_view = None
        total = _total_matches(entities, query, version_token)
    elif view is None or viewport_cells:
        # The ordinary case, and the one the viewer actually sends. Ask the
        # database for the page rather than for the drawing.
        #
        # This used to fetch every matching row, sort all of them here, and
        # slice at the very end, so `limit` bought a smaller RESPONSE and no
        # less work. Measured on a 679,878 row drawing: 25 s for limit=4000
        # against 32 s unbounded -- a tenth of the payload for a thirtieth of
        # the saving. `total_matches` stays the whole layout, counted from the
        # same filter, because a page must never be readable as the drawing.
        if layer_filter is not None:
            # SUMMED, not counted. `count_documents` over a `$nin` is 22 s on
            # this drawing -- the `find` itself is 67 ms -- and every distinct
            # layer selection is a new cache key, so the user would pay it
            # again on every checkbox. The scope's per-layer counts are
            # already cached and this total is exactly their sum over the
            # layers the filter keeps, so it is the same number for none of
            # the cost.
            total = sum(
                row["count"]
                for row in scope_layer_counts(
                    entities, drawing_id, layout, wanted_kinds
                )
                if keeps_layer(row["name"], layer_filter)
            )
        else:
            # The version token, NOT the request's ETag. The count depends on
            # the drawing, the scope and the filter and never on `limit` or
            # `offset`, but the ETag carries both -- so keying on it minted a
            # fresh entry per page and this 2.2 s count never hit its cache
            # once. Exactly the mistake `_coverage` had just been fixed for,
            # one layer down.
            total = _total_matches(
                entities, query, version_token
            )
        page = list(
            entities.find(scoped, _PROJECTION)
            .sort("_id", 1)
            .skip(offset)
            .limit(limit)
        )
        # Deliberately NOT counted for a viewport. Counting what the view
        # holds is a second `count_documents`, it is 2.2 s on this drawing,
        # and every pan is a different box so it can never be cached usefully
        # -- the whole saving of the indexed query went straight back into
        # counting. `None` means "not counted", and `next_offset` below is
        # derived from whether the page came back full instead, which is the
        # only thing the caller needs it for.
        in_view = None if viewport_cells else total
        rows = page
    else:
        # No cells to match against -- a drawing without a coordinate system,
        # or a box H3 could not cover. Falls back to the scan this route has
        # always done, which is correct and slow rather than fast and wrong.
        rows = list(entities.find(query, _PROJECTION).sort("_id", 1))
        total = len(rows)

    # The viewport narrows what is PAGINATED, so `limit` spends itself on what
    # is in view rather than on whatever sorted first. `total` above is left
    # as the whole layout, which is what `total_matches` has always meant and
    # what stops a filtered view being read as the drawing.
    if view is not None and not viewport_cells:
        kept = []
        for row in rows:
            kind = _kind_of(row, wanted_kinds)
            if kind is None:
                continue
            extent = _extent_in_frame(row, kind, frame=frame_in_use, crs=crs)
            if extent is None or _meets(extent, view):
                # An object with no extent at all is kept rather than dropped:
                # it cannot be shown to be outside the box, and dropping it
                # would be a claim this endpoint cannot support.
                kept.append(row)
        rows = kept
        in_view = len(rows)
        page = rows[offset : offset + limit]

    land_use = store_landuse.classify_layers(
        drawing_id, [str(r.get("layer") or "") for r in page]
    )
    area_unit = "m2" if drawing.get("units_name") == "m" else None

    features: list[dict[str, Any]] = []
    world_points: list[list[float]] = []
    counts = {k: 0 for k in KINDS}
    for row in page:
        kind = _kind_of(row, wanted_kinds)
        if kind is None:
            continue
        world = _coordinates(row, kind)
        world_points.extend(world)
        counts[kind] += 1
        centre = row.get("polygon_centroid") or row.get("bbox_centre")
        classification = land_use.get(str(row.get("layer") or ""), {})
        features.append(
            {
                "handle": row.get("handle"),
                "type": row.get("type"),
                "layer": row.get("layer"),
                "block_name": row.get("block_name"),
                "kind": kind,
                "closed": kind == "ring",
                "coordinates": (
                    # Simplified AFTER projection: the tolerance is in
                    # degrees, and converting it back into drawing units to
                    # apply it earlier would add an approximation to remove
                    # one. The world frame is left alone -- "how many points
                    # does this need" is a screen question and that frame has
                    # no screen.
                    _for_zoom(_project(world, crs), kind, zoom)
                    if frame_in_use == "lnglat"
                    else world
                ),
                "centroid": (
                    (_project([centre], crs)[0] if frame_in_use == "lnglat" else
                     [float(centre[0]), float(centre[1])])
                    if centre
                    else None
                ),
                "centroid_basis": (
                    "polygon_centroid"
                    if row.get("polygon_centroid")
                    else ("bbox_centre" if row.get("bbox_centre") else None)
                ),
                "text": row.get("text"),
                # Measured in DRAWING coordinates, whatever `frame` says. See
                # the module docstring: a reprojected ring has a different
                # area, and this is the number the rest of the system quotes.
                "area": row.get("area"),
                "perimeter": row.get("perimeter_from_ring"),
                "length": row.get("length"),
                "ring_status": row.get("ring_status"),
                "ring_orientation": row.get("ring_orientation"),
                "vertex_count": len(world),
                # Arc-ness per segment of an open path. Passed through
                # untouched and NOT flattened: a consumer that draws straight
                # segments between these points is drawing chords, and one
                # that silently received no bulges could not know that.
                "bulges": row.get("path_bulges"),
                "has_bulge": bool(
                    row.get("path_bulges") and any(row["path_bulges"])
                ),
                # True when these two points are an ARC's endpoints and
                # nothing better was stored -- an ingest older than version 8.
                # The line between them is a chord, and a consumer drawing it
                # unknowingly would run a road straight through the inside of
                # a bend. From version 8 the ARC carries `path_points` and is
                # kind `path`, so this is false without a special case.
                "is_chord": kind == "segment" and row.get("type") == "ARC",
                "ends_basis": row.get("ends_basis"),
                "shape_key": row.get("shape_key"),
                "land_use": classification.get("land_use"),
                "land_use_subtype": classification.get("land_use_subtype"),
                "land_use_role": classification.get("land_use_role"),
                "land_use_verified": classification.get("land_use_verified"),
            }
        )

    world_bounds = _bounds(world_points)
    origin_world = (
        [
            (world_bounds[0][0] + world_bounds[1][0]) / 2.0,
            (world_bounds[0][1] + world_bounds[1][1]) / 2.0,
        ]
        if world_bounds
        else None
    )

    return {
        "drawing_id": drawing_id,
        "layout": layout,
        "frame": frame_in_use,
        "frame_basis": (
            "WGS 84 degrees, [lon, lat], one inverse UTM per vertex. Display "
            "only: reprojecting a ring changes its side lengths and its area, "
            "so every `area`, `perimeter` and `length` below stays in drawing "
            "coordinates and is NOT measurable off these vertices."
            if frame_in_use == "lnglat"
            else "The drawing's own coordinates, [x, y], exactly as stored. "
            "Nothing places them on Earth; draw them without a basemap."
        ),
        "crs": (crs.as_dict() if crs else crs_math.unknown_crs(
            "this drawing has no `crs:` block",
            "add one to cad_api/app/landuse/<drawing_id>.yaml, established "
            "rather than guessed",
        )),
        "crs_caveat": crs_math.caveat(crs) if crs else None,
        "crs_confirmed_by": crs_math.confirmed_by(crs) if crs else None,
        "placement": _placement(drawing, crs, frame_in_use),
        "units": {
            "name": drawing.get("units_name"),
            "code": drawing.get("units_code"),
            "area_unit": area_unit,
            "area_unit_reason": (
                None
                if area_unit
                else "areas are in the square of an undeclared unit, so no name is written for them"
            ),
        },
        "origin": (
            {
                "world": origin_world,
                "lnglat": _project([origin_world], crs)[0] if crs and origin_world else None,
                "basis": (
                    "the centre of this response's own vertices, offered so a "
                    "caller using deck.gl's METER_OFFSETS has a site origin. "
                    "Note the trade: METER_OFFSETS treats drawing metres as "
                    "true metres, which ignores UTM grid convergence — about "
                    "0.8 degrees at Janadriyah, or 28 m of rotation across a "
                    "2 km site. frame=lnglat has no such error."
                ),
            }
            if origin_world
            else None
        ),
        # Where the drawing IS, beside where its corners are. The client's
        # DEFAULT camera uses `content`; "Fit all" uses `world`/`lnglat`. Both
        # states name themselves on screen, per the naming rule in section 2.
        "content_bounds": content_bounds(
            drawing_id, layout, version_token
        ),
        "bounds": {
            "world": world_bounds,
            "lnglat": (
                [
                    _project([world_bounds[0]], crs)[0],
                    _project([world_bounds[1]], crs)[0],
                ]
                if crs and world_bounds
                else None
            ),
        },
        "features": features,
        "counts": counts,
        "total_matches": total,
        "returned": len(features),
        "offset": offset,
        # Paging is over what the viewport kept, so these describe the list
        # the caller is actually walking. Without a viewport `in_view` IS
        # `total` and every one of these means exactly what it always did.
        "truncated": (
            len(page) == limit if in_view is None else offset + len(page) < in_view
        ),
        # A full page means there may be more; a short one means there is not.
        # When the view was counted this is the same answer, and when it was
        # not it is still correct.
        "next_offset": (
            (offset + limit)
            if (len(page) == limit if in_view is None else offset + len(page) < in_view)
            else None
        ),
        # One boolean the caller can read, rather than three fields to
        # combine. `/geo` answers the same question the same way, and a client
        # that had to work it out from `frame` and `crs.known` would be a
        # second opinion that could disagree with the one every other route
        # gives.
        "georeferenced": crs is not None,
        "georeferenced_reason": (
            None
            if crs is not None
            else (
                "this drawing declares no coordinate system and has no CRS "
                "config, so it has no position on the earth. Its geometry is "
                "returned in its own coordinates, which is a real view of the "
                "drawing and makes no claim about where on Earth it is"
            )
        ),
        "viewport": (
            None
            if view is None
            else {
                "bbox": list(view),
                "bbox_order": (
                    "west,south,east,north in degrees"
                    if frame_in_use == "lnglat"
                    else "minX,minY,maxX,maxY in the drawing's own units"
                ),
                "frame": frame_in_use,
                # What this answer actually covers, so the caller can
                # accumulate it and send it back as `have` next time. Echoed
                # rather than left to be inferred: deciding which cells a box
                # covers needs H3, and a client that guessed would eventually
                # guess differently from the server.
                "cells_served": sorted(viewport_cells or []),
                "cells_already_held": sorted(held_cells),
                "nothing_new": nothing_new,
                # Not counted for a viewport: see `in_view` above. Said as
                # null with a reason rather than as a number that was never
                # measured.
                "matches_in_view": in_view,
                "matches_outside": None if in_view is None else total - in_view,
                "matches_in_view_note": (
                    None
                    if in_view is not None
                    else "not counted: counting what a viewport holds costs a "
                    "second full count on every pan, and the page itself "
                    "already says whether more is available"
                ),
                "note": (
                    "`features` is restricted to this box and so is the paging "
                    "over it; `total_matches` is NOT — it stays the whole "
                    "layout, so a viewport can never be mistaken for the "
                    "drawing"
                ),
                "kept_when": (
                    "the object's extent OVERLAPS the box — not is contained "
                    "by it. A road crossing the view is in the view, and a "
                    "parcel larger than the box contains it and is in it too"
                ),
                "measured_on": (
                    "the object's stored bounding box where it has one, and "
                    "its own vertices where it does not. In lng/lat the box's "
                    "four corners are projected and their extremes taken, "
                    "which is a superset of the true projected extent: this "
                    "may keep something marginally outside the view and will "
                    "never drop something inside it"
                ),
                "empty_is_an_answer": (
                    "nothing here means nothing of these kinds in this part of "
                    "the drawing. It is not an error and not a failed query; "
                    "`total_matches` says what the layout holds elsewhere"
                ),
            }
        ),
        # Not filtered out of `features` — hidden is a VIEW decision, and an
        # endpoint that silently dropped them would make an object
        # unreachable rather than unshown. The caller is told, and decides.
        "layers_off": _layers_off(drawing),
        "layers_off_basis": (
            "layers this DRAWING keeps switched off or frozen, from its layer "
            "table. Their features are included above and should start "
            "hidden, matching AutoCAD. The rendered SVG carries the same list "
            "in `data-off-layers`, but only for the layout it rendered — a "
            "paper sheet reports none, because its viewport already excludes "
            "them."
        ),
        "geometry_coverage": _coverage(drawing_id, layout, version_token),
        "hint": (
            "Layers are classified by the land-use config, not by this "
            "endpoint. Read `land_use_role`: 'parcel' is a plot, 'network' a "
            "road centreline whose native measure is length, and null means "
            "no config decides this layer — which is not the same as unknown."
        ),
    }
