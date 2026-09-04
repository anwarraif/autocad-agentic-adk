"""The map payload: cells to colour and parcels to draw.

Answers `GET /drawings/{id}/geo?res=N` against the contract drafted in
`FRONTEND-MAP-BRIEF-AMEER.md`: a list of cells carrying counts and areas per
land use, and a GeoJSON FeatureCollection of the parcels behind them. Both
come out of the cells written in G2, so this module computes no geometry of
its own — it groups, classifies and shapes.

Three decisions in here are worth reading before the code.

**A parcel is counted in ONE cell, its representative one.** A parcel covers
several cells; counting it in each would make the totals depend on the
resolution, and the first thing anybody does with this payload is compare
`?res=9` against `?res=13`. Counting the representative cell makes the two
agree exactly, which is what the acceptance check asks for. The cost is that a
parcel straddling a cell boundary is attributed wholly to one side, and the
response says so rather than leaving it to be discovered.

**Coarser cells are reached with `cell_to_parent`, never by re-indexing.** The
two routes disagree for anything near a cell boundary — the mosque is one such
point at resolution 7 — so mixing them would make totals differ by a handful
of parcels with nothing to explain it. See `tests/test_geo_h3.py`.

**Aggregates are never compacted.** `compact_cells` merges children into a
parent, and a count cannot survive that merge: seven parcels in seven children
are not seven parcels in the parent, they are a number the reader can no
longer attribute. `?compact=true` therefore compacts the COVERAGE — the set of
cells the parcels occupy, which is a set and merges honestly — and leaves the
counted cells alone. The silhouette is what compaction is for.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Iterable, Mapping, Sequence

from . import evidence as ev
from . import landuse, store_landuse
from .store_landuse import GEOJSON_DECIMALS
from .extract import RING_TYPES
from .mongo import COLL_ENTITIES, coll

log = logging.getLogger(__name__)

try:  # pragma: no cover - the dependency is pinned
    import h3
except ImportError:  # pragma: no cover
    h3 = None  # type: ignore[assignment]


#: Parcel features returned before the collection is truncated.
#:
#: The reference drawing has 2,885 of them and they serialise to about 1.4 MB,
#: which a browser handles and a chat response does not. The cap is stated in
#: the response together with how many were left out, so a map that is missing
#: parcels can tell that it is missing them (G7).
MAX_PARCEL_FEATURES = 5_000

#: Cells returned in the aggregate list before it is truncated. At resolution
#: 9 the reference drawing produces 26; at 13 it produces 2,721. The cap only
#: bites on a drawing far larger than anything here, and it is stated for the
#: same reason.
MAX_CELLS = 20_000

#: Handles one export may be asked for by name.
#:
#: An agent answer names objects; the largest one measured named 2,380 of
#: them, and the chat's own candidate cap is 2,000. This sits above both so
#: that the limit belongs to the store rather than to whoever is calling, and
#: an export that hits it says so rather than quietly shortening the answer.
MAX_EXPORT_HANDLES = 4_000

#: Rings drawn around an answer's own cells when a catchment is asked for.
#: The same ceiling `h3_vicinity` uses, and for the same reason: `grid_disk`
#: grows as 3k²+3k+1.
MAX_EXPORT_RINGS = 25

#: The area unit this payload is allowed to publish. `area_m2` is a promise
#: about the number's unit, and a drawing in inches would break that promise
#: while looking identical. There is no conversion here: the areas come from
#: the drawing and the drawing says what they are.
AREA_UNIT = "m2"


class GeoUnavailable(Exception):
    """The drawing cannot be put on a map, with the reason why.

    A fact about the drawing, not a fault in the request: the caller turns it
    into an explicit `georeferenced: false` answer rather than an error.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class GeoNotIndexed(Exception):
    """The drawing can be mapped, but nothing in it has been indexed yet.

    A different state from `GeoUnavailable`, and the difference is the whole
    reason this class exists. A drawing with no coordinate system has no
    position and never will until somebody gives it one. A drawing WITH a
    coordinate system and no cells is one command away from working — and
    serving it as an ordinary empty answer says "this site has nothing in it",
    which is a lie about the drawing rather than a fact about the index.

    It is not hypothetical. An external re-ingest of the reference drawing
    stripped all 46,754 entities of their `h3_*` fields mid-campaign —
    `store.replace_entities` writes with `ReplaceOne` — and the endpoints went
    on answering 200 with an empty cell list and totals of zero.
    """

    def __init__(self, reason: str, how_to_fix: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.how_to_fix = how_to_fix


class GeoBadRequest(Exception):
    """The request asked for something this store cannot honestly produce.

    Carries the same three fields as the API's own errors — a code, what is
    wrong, and what to do instead — so the route can hand it straight over
    without inventing a hint of its own.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


@dataclass(frozen=True)
class BBox:
    """A viewport, in degrees, west/south/east/north.

    The order is the one GeoJSON and every mapping library uses — the same
    order `[min_lon, min_lat, max_lon, max_lat]` that a `bbox` member takes in
    RFC 7946 — so that a caller who has a viewport already can pass it
    through without transposing it. Longitude first, exactly as the
    coordinates in this API's own GeoJSON, and for the same reason: the pair
    that gets swapped is the pair whose two conventions live in one codebase.
    """

    west: float
    south: float
    east: float
    north: float

    def contains(self, lon: float, lat: float) -> bool:
        return self.west <= lon <= self.east and self.south <= lat <= self.north


def parse_bbox(raw: str | None) -> BBox | None:
    """A `bbox=` parameter, or `None` when there is none.

    Refuses rather than repairs. A reversed box — north and south the wrong
    way round — describes no ground at all, and quietly swapping the two would
    answer a question nobody asked with an answer that looks perfectly
    ordinary: an empty viewport reads exactly like an empty part of the site.
    So it comes back as a 400 naming what is wrong, and the caller fixes their
    viewport instead of wondering where the parcels went.
    """
    if raw is None or not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 4:
        raise GeoBadRequest(
            "BBOX_MALFORMED",
            f"bbox={raw!r} has {len(parts)} value(s); it takes four.",
            "bbox=west,south,east,north in degrees — the same order a GeoJSON "
            "`bbox` uses, longitude first. Example for this drawing: "
            "bbox=46.89,24.85,46.90,24.86",
        )
    try:
        west, south, east, north = (float(p) for p in parts)
    except ValueError:
        raise GeoBadRequest(
            "BBOX_MALFORMED",
            f"bbox={raw!r} contains something that is not a number.",
            "Four decimal degrees: west,south,east,north.",
        ) from None

    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise GeoBadRequest(
            "BBOX_OUT_OF_RANGE",
            f"bbox longitudes {west} and {east} are outside -180..180.",
            "Longitude first: bbox=west,south,east,north. A pair that looks "
            "like a latitude in the first slot is usually a transposed box.",
        )
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise GeoBadRequest(
            "BBOX_OUT_OF_RANGE",
            f"bbox latitudes {south} and {north} are outside -90..90.",
            "Longitude first: bbox=west,south,east,north.",
        )
    if west > east or south > north:
        raise GeoBadRequest(
            "BBOX_REVERSED",
            f"bbox={raw!r} has its corners the wrong way round.",
            "west must be <= east and south <= north. Swapping them here "
            "would answer a different question and return an empty viewport, "
            "which reads exactly like an empty part of the drawing.",
        )
    return BBox(west=west, south=south, east=east, north=north)


#: Everything an answer from this route depends on, in the order it is hashed.
#:
#: The plan's premise is that `drawing_id` is a content hash, so the response
#: cannot change until the file does. That is true of the FILE and not quite
#: true of the ANSWER, and the difference is what this tuple exists for: the
#: land use config decides which layers are parcels and what they are called,
#: and the stored H3 resolution decides which cells there are to group. Either
#: can change while the drawing does not, and an ETag that ignored them would
#: serve a stale body with a confident 304 — the one cache failure nobody
#: reports, because everything looks like it is working.
ETAG_INPUTS = (
    "drawing_id",
    "config_version",
    "stored_resolution",
    "ingest_version",
    "params",
)


def etag_for(
    drawing_id: str,
    *,
    config_version: Any,
    stored_resolution: int,
    ingest_version: Any,
    params: Mapping[str, Any],
) -> str:
    """A strong ETag for one `/geo` answer.

    Strong, not weak: the bytes really are identical for identical inputs.
    Nothing in this route reads a clock, a random number or anything outside
    the store, which is the property that makes a cache safe here and is
    tested rather than assumed.

    The parameters are normalised before hashing — sorted, and rendered
    through `json.dumps` with sorted keys — so that `?res=9&parcels=false` and
    `?parcels=false&res=9` are one cache entry rather than two. A cache key
    that depends on the order somebody typed things is a cache that mostly
    misses.
    """
    payload = json.dumps(
        {
            "drawing_id": drawing_id,
            "config_version": config_version,
            "stored_resolution": stored_resolution,
            "ingest_version": ingest_version,
            "params": {k: params[k] for k in sorted(params)},
        },
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f'"{digest}"'


def etag_matches(if_none_match: str | None, tag: str) -> bool:
    """Whether a conditional request may be answered 304.

    `If-None-Match` is a LIST, and it may be `*`, and each entry may carry a
    `W/` prefix. Comparing the raw header against one tag works for exactly
    one client and silently stops working for the next — a browser sending two
    tags would simply never get a 304, and nothing would look broken.
    """
    if not if_none_match:
        return False
    candidates = [c.strip() for c in if_none_match.split(",")]
    if "*" in candidates:
        return True
    bare = tag.strip().lstrip("W/")
    return any(c.lstrip("W/") == bare for c in candidates if c)


def preconditions(drawing_id: str) -> dict[str, Any]:
    """The three store facts an ETag is built from, read cheaply.

    Two small documents and a config file, none of them the aggregation the
    route is here to avoid. That is the point of a conditional request: a
    client that already has the answer must not pay for it to be computed
    again.
    """
    from . import store  # noqa: PLC0415 -- deferred, see store_landuse's header

    try:
        config = landuse.for_drawing(drawing_id)
    except landuse.ConfigError:
        # A config that cannot be read still produces a deterministic answer —
        # a refusal — and the refusal is cacheable. Its key is the fact that
        # it could not be read.
        config = None
    # The VERSION only. This used to fetch the whole drawing document for one
    # field, and on Sedra that document is 441 KB: measured on Atlas, 2.74 s
    # against 0.22 s for the projection. `preconditions` runs on every request
    # -- twice, since the coverage cache needs a version token of its own --
    # so it was most of the fixed cost a request paid before it counted
    # anything.
    return {
        "config_version": (config.version if config else None),
        "stored_resolution": (
            config.h3_resolution if config else landuse.DEFAULT_H3_RESOLUTION
        ),
        "ingest_version": store.ingest_version_of(drawing_id),
    }


#: The dimension a drawing is coloured and counted by when no land-use config
#: classifies any of its layers. `layer` is chosen because EVERY entity has one
#: -- it is the one dimension no drawing can lack.
FALLBACK_DIMENSION: Final[str] = "layer"

#: The share of in-scope shapes a dimension must explain before it is used by
#: DEFAULT. Below this the view is not a map of the drawing -- it is a map of
#: the handful of things somebody has got round to classifying.
#:
#: Measured, the two cases are four orders of magnitude apart and no threshold
#: between them is doing real work: Janadriyah's config explains 2,576 of
#: 3,432 shapes (75%), Sedra's explains 4 of 43,680 (0.009%). It is written as
#: a readability floor rather than tuned, and an explicit `?dimension=` always
#: overrides it, so a human's config is deferred to whenever a human asks.
READABLE_COVERAGE: Final[float] = 0.01


def _as_layer_dimension(layer: str) -> landuse.LandUse:
    """One layer, presented as its own bucket.

    Not a guess about what the layer MEANS. The use is the layer's own name,
    the config layer is `no_config`, and `is_parcel` is true only so that the
    same loop can count it -- the response says `dimension: layer`, so nothing
    downstream can read these as land uses.
    """
    return landuse.LandUse(
        layer=layer,
        use=layer,
        subtype=None,
        role=landuse.PARCEL_ROLE,
        config_layer=ev.ConfigLayer.NO_CONFIG,
        config_version=None,
        note=(
            "counted by layer: no land-use config classifies this drawing, so "
            "the view falls back to a dimension the drawing does have"
        ),
        not_established="land use",
        how_to_verify=(
            "accept this drawing's land-use draft; the view then colours by "
            "land use with no code change"
        ),
    )


def _classification(drawing_id: str) -> dict[str, landuse.LandUse]:
    """Layer name -> land use, for every layer this drawing classifies.

    Read once per request rather than per entity: 2,885 parcels over 292
    layers would otherwise be 2,885 config lookups for 292 distinct answers.
    """
    config = landuse.for_drawing(drawing_id)
    pattern_set = landuse.patterns()
    out: dict[str, landuse.LandUse] = {}
    for row in coll(COLL_ENTITIES).distinct("layer", {"drawing_id": drawing_id}):
        name = str(row or "")
        if not name:
            continue
        hit = landuse.classify(
            drawing_id, name, config=config, pattern_set=pattern_set
        )
        if hit is not None:
            out[name] = hit
    return out


def _feature(
    crs: Any, row: Mapping[str, Any], land_use: landuse.LandUse, area: float | None
) -> dict[str, Any] | None:
    """One parcel as a GeoJSON Feature.

    `handle` is on every feature and is the field the brief guarantees will
    survive into the final contract: it is the same id the agent quotes in
    chat, so marking an answer on the map is answer handles -> their features.
    """
    ring = store_landuse._ring_as_geojson(crs, row.get("ring") or [])
    if ring is None:
        return None
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "handle": row.get("handle"),
            "layer": row.get("layer"),
            "land_use": land_use.use,
            "land_use_subtype": land_use.subtype,
            "land_use_verified": land_use.provenance().verified,
            "area_m2": area,
            "cell": row.get("h3_cell"),
        },
    }


def _hierarchy(stored_cells: set[str], stored_res: int) -> dict[str, Any] | None:
    """The ladder of H3 levels this site occupies, from one hexagon down.

    Harsh, 1 September 2026: *"H3 is a hierarchical clustering... yesterday
    what you showed is one single hierarchy. The top will be capped to the
    maximum hexagon which covers the whole project."* This is that cap, and
    every rung under it, answered from the drawing rather than asserted.

    Derived, never stored. `cell_to_parent` over the cells already indexed is
    the same roll-up the counts use, so a level here can never disagree with
    the counts served at that level -- which it could if this were a second
    index built by a second pass.

    Walked from the stored resolution UP, each level's parents taken from the
    level below rather than from the stored set. That is not only cheaper --
    2,546 cells collapse to 1,957 and then to 645 within two steps -- it is
    also the statement being made: each rung is the one above it aggregated.

    `res_cover` is the FINEST resolution at which the whole site is a single
    hexagon: the top of the ladder in Harsh's sense, and the coarsest level
    worth offering, since everything above it is the same one hexagon getting
    bigger. It is None for a site that straddles two cells even at res 0, in
    which case the ladder simply starts there -- a site with no single-hexagon
    cover is a fact to report, not a reason to refuse the rest.
    """
    if not stored_cells:
        return None

    counts: dict[int, int] = {stored_res: len(stored_cells)}
    current = stored_cells
    for res in range(stored_res - 1, -1, -1):
        current = {h3.cell_to_parent(cell, res) for cell in current}
        counts[res] = len(current)

    singles = [res for res, n in sorted(counts.items()) if n == 1]
    res_cover = max(singles) if singles else None
    start = res_cover if res_cover is not None else 0
    return {
        "res_cover": res_cover,
        "res_stored": stored_res,
        "levels": [
            {"res": res, "cells": counts[res]}
            for res in range(start, stored_res + 1)
        ],
        "derived_from": (
            "cell_to_parent over the cells this drawing is indexed into. No "
            "level here is stored, and none is a re-measurement: a coarser "
            "rung is the finer one aggregated, which is why the land-use "
            "totals are identical at every level"
        ),
        "cap_note": (
            "res_cover is the finest resolution at which one hexagon covers "
            "the whole site. Coarser than that is the same single hexagon "
            "getting larger, so it is the top of the ladder"
            if res_cover is not None
            else "this site has no single-hexagon cover at any resolution: it "
            "straddles a cell boundary even at res 0, so the ladder starts "
            "there"
        ),
    }


def _covers_any(cell: str, fine_cells: Iterable[str]) -> bool:
    """Whether an aggregated cell contains any of these finer ones.

    The aggregate list is at the requested resolution and the parcels name
    theirs at the stored one, so they are compared by ancestry rather than by
    equality — `cell_to_parent`, the one direction this campaign derives in.
    """
    target = h3.get_resolution(cell)
    for fine in fine_cells:
        try:
            if h3.cell_to_parent(fine, target) == cell:
                return True
        except Exception:  # pragma: no cover - a cell coarser than the parcel's
            continue
    return False


def _cell_in_bbox(cell: str, bbox: BBox) -> bool:
    """Whether a cell meets the viewport.

    Centre first, because most cells are decided by it and `cell_to_boundary`
    costs six coordinate conversions. The third test is for the case the first
    two miss: a cell LARGER than the box — resolution 9 cells are 105,000 m2 —
    contains it entirely, has no vertex inside it, and would vanish from a
    viewport it completely covers.
    """
    lat, lon = h3.cell_to_latlng(cell)
    if bbox.contains(lon, lat):
        return True
    boundary = h3.cell_to_boundary(cell)
    if any(bbox.contains(v_lon, v_lat) for v_lat, v_lon in boundary):
        return True
    lats = [v[0] for v in boundary]
    lons = [v[1] for v in boundary]
    mid_lon = (bbox.west + bbox.east) / 2
    mid_lat = (bbox.south + bbox.north) / 2
    return min(lons) <= mid_lon <= max(lons) and min(lats) <= mid_lat <= max(lats)


def _feature_in_bbox(feature: Mapping[str, Any], bbox: BBox) -> bool:
    """Whether a parcel's own extent meets the viewport.

    The parcel's extent, not its cell's. A viewport at high zoom should not be
    snapped to whatever resolution the cells happen to be aggregated at.
    """
    ring = feature["geometry"]["coordinates"][0]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return (
        min(lons) <= bbox.east
        and max(lons) >= bbox.west
        and min(lats) <= bbox.north
        and max(lats) >= bbox.south
    )


def geo(
    drawing_id: str,
    *,
    res: int | None = None,
    layout: str | None = None,
    parcels: bool = True,
    compact: bool = False,
    dimension: str | None = None,
    limit: int = MAX_PARCEL_FEATURES,
    bbox: BBox | None = None,
    outline: bool = False,
    highlight: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Cells and parcels for one drawing at one resolution.

    Raises `GeoUnavailable` when the drawing has no cells to give — the caller
    turns that into an answer that says so, never into an empty success.
    """
    if h3 is None:  # pragma: no cover
        raise GeoUnavailable("the h3 package is not installed in this image")

    try:
        config = landuse.for_drawing(drawing_id)
    except landuse.ConfigError as exc:
        raise GeoUnavailable(
            f"this drawing's config cannot be read: {exc.message}"
        ) from exc

    choice = store_landuse.crs_for_drawing(drawing_id, config=config)
    stored_res = config.h3_resolution if config else landuse.DEFAULT_H3_RESOLUTION
    if choice.crs is None:
        raise GeoUnavailable(
            choice.note
            or (
                "this drawing is not georeferenced: it declares no coordinate "
                "system and has no CRS config, so it has no cells and cannot "
                "be put on a map"
            )
        )

    if res is not None and res > stored_res:
        # Checked before any work is done, and refused rather than served.
        # `cell_to_children` would happily return seven cells where the
        # drawing supports one piece of evidence, and seven cells look like
        # more detail rather than like the same fact restated.
        raise GeoBadRequest(
            "H3_RESOLUTION_FINER_THAN_STORED",
            f"res={res} is finer than resolution {stored_res}, which is what "
            "this drawing's cells are stored at.",
            f"Ask for {stored_res} or coarser. Splitting a stored cell into "
            "children would return more cells without returning more evidence.",
        )

    target = stored_res if res is None else res
    scope_layout = layout or "Model"

    # What counts as being in scope. A drawing that keeps its content inside a
    # bound xref stores that content under a `[block] ...` layout while its
    # coordinates are ordinary positions in the drawing, and the index says so
    # per entity with `h3_world_placed` (see `geo_h3.cells_for_entity`). Model
    # scope therefore means "on the Model layout, OR measured to be in the
    # world", because otherwise a fully indexed drawing answers with an empty
    # map and nothing on screen explains why.
    #
    # Asked only for model scope: a request for one sheet means that sheet.
    from . import geo_h3 as _geo_h3  # noqa: PLC0415 -- deferred, one direction only

    scope_match: dict[str, Any] = _geo_h3.scope_match(scope_layout)

    # Is there an index at all? One `find_one` against the compound index,
    # asked before the aggregation rather than inferred from its emptiness —
    # "no cells came back" has two causes and only one of them is a fact about
    # the drawing.
    if not coll(COLL_ENTITIES).find_one(
        {"drawing_id": drawing_id, **scope_match, "h3_cell": {"$ne": None}},
        {"_id": 1},
    ):
        # Two reasons a layout has no cells, and they need opposite advice.
        # A sheet or a block definition can NEVER have them — their
        # coordinates are page geometry and block-local frames — so telling
        # someone to run the backfill would send them to a command that
        # changes nothing. Model space with no cells is the one the backfill
        # fixes.
        from . import store  # noqa: PLC0415 -- deferred, one direction only

        space = store._space_of(scope_layout)
        if space != "model":
            raise GeoNotIndexed(
                f"layout {scope_layout!r} is {space} space, whose coordinates "
                "are "
                + (
                    "page geometry"
                    if space == "paper"
                    else "the block's own frame"
                )
                + " rather than positions on the earth. It has no cells and "
                "never will; this is not a missing index",
                f"ask for the model layout instead: ?layout=Model",
            )
        raise GeoNotIndexed(
            f"this drawing is georeferenced but no entity in layout "
            f"{scope_layout!r} has an H3 cell, so there is nothing to map yet. "
            "That is an index that has not been built or has been removed — "
            "re-ingesting a drawing strips these fields — and NOT a drawing "
            "with nothing in it",
            "python scripts/backfill_h3.py --drawing " + drawing_id,
        )

    from . import store  # noqa: PLC0415 -- deferred, see store_landuse's header

    drawing = store.get_drawing(drawing_id) or {}
    units = store._unit_names(drawing, scope_layout)
    area_unit = units.get("area_unit")
    area_publishable = area_unit == AREA_UNIT

    classified = _classification(drawing_id)
    parcel_layers = [name for name, use in classified.items() if use.is_parcel]

    # NO VIEW GOES BLANK BECAUSE AN OPTIONAL CONFIG IS ABSENT.
    #
    # This used to widen the query when no layer was a parcel, and then reject
    # every row it fetched -- `classified` was empty, so `land_use is None`
    # was true for all 990,790 of them. Sedra holds 106,580 distinct cells and
    # the view drew ONE hexagon: the index was full and the measure was wrong
    # for the drawing.
    #
    # The dimension is now chosen from what the drawing actually contains, and
    # NAMED in the response so the client renders what it was told rather than
    # assuming parcels. A drawing that has a config is untouched by this.
    #
    # A caller may ASK for a dimension. That is how the map stops being blank
    # on a drawing whose config classifies four shapes out of 5,773 without
    # any code overriding a decision a human made: the human's config still
    # wins by default, and the alternative is offered rather than imposed.
    if dimension not in (None, "land_use", FALLBACK_DIMENSION):
        raise GeoBadRequest(
            "UNKNOWN_DIMENSION",
            f"dimension must be 'land_use' or '{FALLBACK_DIMENSION}', "
            f"not {dimension!r}",
            "omit it to let the drawing decide, or name one of the two the "
            "response's `dimension.alternatives` lists",
        )
    #: Every shape the map COULD colour, whatever the dimension. The
    #: denominator that turns "4 parcels" into "4 of 43,680".
    #:
    #: EXACTLY the population the loop below counts from -- same scope, same
    #: types, same cell requirement, and deliberately NO `ring` clause. A
    #: first version added one, and printed "43,680 of 5,773": the numerator
    #: was ring-TYPED rows and the denominator was rows carrying a ring array,
    #: which are different questions wearing one fraction. A denominator that
    #: does not answer the numerator's question is worse than none.
    shapes_in_scope = coll(COLL_ENTITIES).count_documents(
        {
            "drawing_id": drawing_id,
            **scope_match,
            "type": {"$in": sorted(RING_TYPES)},
            "h3_cell": {"$ne": None},
        }
    )

    if dimension is not None:
        by_layer = dimension == FALLBACK_DIMENSION
    elif not parcel_layers:
        by_layer = True
    else:
        # A config that classifies four shapes out of 43,680 is not wrong, and
        # nothing here overrides it -- `?dimension=land_use` still returns it
        # exactly. It is just not what this drawing should OPEN as.
        classifiable = coll(COLL_ENTITIES).count_documents(
            {
                "drawing_id": drawing_id,
                **scope_match,
                "type": {"$in": sorted(RING_TYPES)},
                "h3_cell": {"$ne": None},
                "layer": {"$in": parcel_layers},
            }
        )
        by_layer = (
            shapes_in_scope > 0
            and classifiable / shapes_in_scope < READABLE_COVERAGE
        )

    #: Every handle whose cell should be flagged. Upper-cased once here so
    #: the comparison in the loop is a set membership and not a per-row
    #: normalisation over thousands of documents.
    wanted = {h.strip().upper() for h in (highlight or []) if str(h).strip()}

    cells: dict[str, dict[str, Any]] = {}
    features: list[dict[str, Any]] = []
    coverage: set[str] = set()
    #: The cells as INDEXED, before any roll-up. The hierarchy is derived from
    #: these rather than from `cells`, which is already keyed at the requested
    #: resolution -- a ladder built from a rolled-up set would report the
    #: request back to the caller instead of describing the drawing.
    stored_cells: set[str] = set()
    #: Cells holding at least one of `highlight`'s objects, at `target`.
    highlighted: set[str] = set()
    totals_count: dict[str, int] = {}
    totals_area: dict[str, float] = {}
    counted = 0
    without_cell = 0
    without_ring = 0
    features_dropped = 0

    # Inclusion only. A projection that mixes 1 and 0 is rejected by MongoDB
    # for every field except `_id`, so the optional fields are added rather
    # than switched off -- and they are optional because they are the two big
    # ones: a ring carries up to 4,096 vertices and a covering up to 20,000
    # cells, over thousands of documents.
    projection: dict[str, Any] = {
        "handle": 1,
        "layer": 1,
        "area": 1,
        "type": 1,
        "h3_cell": 1,
    }
    if compact or outline:
        projection["h3_cells"] = 1
    if parcels:
        projection["ring"] = 1
    query = {
        # drawing_id first, then the indexed layout. `layer` and `type` narrow
        # what the index already selected.
        "drawing_id": drawing_id,
        **scope_match,
        "layer": {"$in": parcel_layers},
        # A parcel is a polygon. Without this the count is of everything drawn
        # ON a parcel layer, which on the reference drawing means 15 HATCHes
        # turning 4 commercial parcels into 19 — and `land_use_summary`, which
        # already excludes them, would be answering the same question with a
        # different number. Two answers to one question is the defect; which
        # of them is right is secondary.
        "type": {"$in": sorted(RING_TYPES)},
    }
    if by_layer or not parcel_layers:
        # Every layer, because every layer IS the dimension.
        query.pop("layer")


    for row in coll(COLL_ENTITIES).find(query, projection).batch_size(1000):
        row_layer = str(row.get("layer") or "")
        land_use = (
            _as_layer_dimension(row_layer)
            if by_layer
            else classified.get(row_layer)
        )
        if land_use is None or not land_use.is_parcel:
            continue
        if str(row.get("type") or "") not in RING_TYPES:
            continue
        cell = row.get("h3_cell")
        if not cell:
            without_cell += 1
            continue

        counted += 1
        stored_cells.add(cell)
        parent = h3.cell_to_parent(cell, target) if target != stored_res else cell
        if wanted and str(row.get("handle") or "").upper() in wanted:
            highlighted.add(parent)
        bucket = cells.setdefault(
            parent, {"cell": parent, "counts": {}, "area_m2": {}}
        )
        bucket["counts"][land_use.use] = bucket["counts"].get(land_use.use, 0) + 1
        totals_count[land_use.use] = totals_count.get(land_use.use, 0) + 1

        area = row.get("area")
        if area_publishable and isinstance(area, (int, float)):
            bucket["area_m2"][land_use.use] = round(
                bucket["area_m2"].get(land_use.use, 0.0) + float(area), 3
            )
            totals_area[land_use.use] = round(
                totals_area.get(land_use.use, 0.0) + float(area), 3
            )

        if compact or outline:
            coverage.update(row.get("h3_cells") or [cell])

        if parcels:
            if len(features) >= limit:
                features_dropped += 1
                continue
            feature = _feature(
                choice.crs,
                row,
                land_use,
                float(area) if area_publishable and isinstance(area, (int, float)) else None,
            )
            if feature is None:
                without_ring += 1
            else:
                features.append(feature)

    cell_list = sorted(cells.values(), key=lambda c: c["cell"])

    # `highlighted` on EVERY cell rather than only on the lit ones, and only
    # when a highlight was asked for. Present-or-absent is decided by the
    # caller's own parameter, so nobody has to guard against a key that comes
    # and goes with the data; within a response that has it, `false` is an
    # answer rather than a gap.
    if wanted:
        for entry in cell_list:
            entry["highlighted"] = entry["cell"] in highlighted

    cells_dropped = max(0, len(cell_list) - MAX_CELLS)
    if cells_dropped:
        cell_list = cell_list[:MAX_CELLS]

    # The viewport, applied LAST and to the drawn things only.
    #
    # `totals` is deliberately left alone. A filtered view that also narrowed
    # its totals would be a picture of the whole site as far as any reader
    # could tell — "2,380 houses" printed beside four hexagons, with nothing
    # saying the number came from elsewhere on the screen. So the totals stay
    # site-wide, the viewport reports its own counts separately, and the note
    # says which is which.
    viewport: dict[str, Any] | None = None
    if bbox is not None:
        before_cells, before_parcels = len(cell_list), len(features)
        cell_list = [c for c in cell_list if _cell_in_bbox(c["cell"], bbox)]
        features = [f for f in features if _feature_in_bbox(f, bbox)]
        in_view: dict[str, int] = {}
        for cell in cell_list:
            for use, count in cell["counts"].items():
                in_view[use] = in_view.get(use, 0) + count
        viewport = {
            "bbox": [bbox.west, bbox.south, bbox.east, bbox.north],
            "bbox_order": "west,south,east,north in degrees",
            "cells_in_view": len(cell_list),
            "cells_outside": before_cells - len(cell_list),
            "parcels_drawn_in_view": len(features),
            "parcels_outside": before_parcels - len(features),
            "counts_in_view": dict(sorted(in_view.items())),
            "note": (
                "`cells` and `parcels` are restricted to this box; `totals` "
                "is NOT — it stays the whole drawing, so a viewport can never "
                "be mistaken for the site. `counts_in_view` is the sum over "
                "the cells kept, and a kept cell is counted whole even where "
                "it hangs over the edge of the box"
            ),
            "empty_is_an_answer": (
                "no cells here means no parcels in this part of the drawing. "
                "It is not an error and not a failed query; `totals` says what "
                "the drawing holds elsewhere"
            ),
            "cells_kept_when": (
                "the cell's boundary meets the box — its centre inside, or any "
                "of its six vertices, or the box's own centre inside the cell"
            ),
            "parcels_kept_when": (
                "the parcel's own lat/long extent meets the box. Judged on the "
                "parcel rather than on its cell, so that a viewport at high "
                "zoom is not snapped to the aggregation resolution"
            ),
        }

    body: dict[str, Any] = {
        "drawing_id": drawing_id,
        "georeferenced": True,
        "indexed": True,
        "resolution": target,
        "stored_resolution": stored_res,
        "layout": scope_layout,
        "epsg": choice.crs.epsg,
        "crs_source_layer": choice.layer,
        "crs_caveat": store_landuse.crs_math.caveat(choice.crs),
        "crs_declaration_note": choice.note,
        # THE CONTRACT: the backend declares what it measured, the frontend
        # renders what it was told. The client must not decide what these
        # numbers mean -- that is the second-opinion mistake moved across the
        # API boundary.
        "dimension": {
            "name": FALLBACK_DIMENSION if by_layer else "land_use",
            "label": "Layer" if by_layer else "Land use",
            "measure": "parcels",
            "measure_label": (
                "closed shapes per layer" if by_layer else "parcels per land use"
            ),
            # What the dimension EXPLAINS, said as a number rather than left
            # for the reader to infer from one hexagon. Sedra's config
            # classifies 4 shapes out of 5,773 in scope; that is not a broken
            # map and it is not a full one either, and only the fraction says
            # which.
            "covers": counted,
            "of_shapes_in_scope": shapes_in_scope,
            "leaves_unexplained": max(0, shapes_in_scope - counted),
            "alternatives": [
                {
                    "name": FALLBACK_DIMENSION if not by_layer else "land_use",
                    "label": "Layer" if not by_layer else "Land use",
                    "ask_for_it": (
                        f"?dimension="
                        f"{FALLBACK_DIMENSION if not by_layer else 'land_use'}"
                    ),
                    "available": True if not by_layer else bool(parcel_layers),
                }
            ],
            "requested": dimension,
            "chosen_because": (
                "no land-use config classifies any layer of this drawing, so "
                "the view falls back to a dimension the drawing does have. "
                "Accepting the land-use draft switches it to land use with no "
                "code change"
                if by_layer
                else "this drawing has a land-use config and its layers "
                "classify, which is the richest dimension available"
            ),
            "is_fallback": by_layer,
        },
        "cells": cell_list,
        "hierarchy": _hierarchy(stored_cells, stored_res),
        "parcels": {"type": "FeatureCollection", "features": features},
        "totals": {
            "parcels": counted,
            "counts": dict(sorted(totals_count.items())),
            "area_m2": dict(sorted(totals_area.items())) if area_publishable else None,
        },
        "area_unit": area_unit,
        "area_note": (
            None
            if area_publishable
            else (
                f"areas are withheld: this drawing's areas are in "
                f"{area_unit or 'units it does not state'}, and `area_m2` is a "
                "promise about the unit. Nothing here converts them"
            )
        ),
        "counts_scope": (
            "one count per parcel, placed in the single cell that represents "
            "it. A parcel covers several cells; counting it in each would make "
            "the total depend on the resolution, and a parcel that straddles a "
            "boundary is attributed wholly to one side. A parcel is a polygon "
            "on a layer this drawing classifies with role=parcel: a HATCH "
            "drawn over one is not a second parcel, which is the same rule "
            "land_use_summary counts by"
        ),
        "aggregation": (
            f"cells are stored at resolution {stored_res} and rolled up to "
            f"{target} with cell_to_parent. Re-indexing the points at "
            f"{target} would give a different answer for anything near a cell "
            "boundary"
        ),
        "viewport": viewport,
        "limits": {
            "max_parcel_features": limit,
            "parcels_omitted": features_dropped,
            "max_cells": MAX_CELLS,
            "cells_omitted": cells_dropped,
        },
        "excluded": {
            "parcels_without_a_cell": without_cell,
            "parcels_without_a_usable_ring": without_ring,
            "note": (
                "parcels counted in `cells` but not drawn in `parcels`: their "
                "ring is one this store will not vouch for. `GET "
                "/drawings/{id}/entities/{handle}` says which of the five "
                "reasons applies"
            ),
        },
    }

    # What the highlight actually found, on the same three-bucket rule the
    # export uses: asked for, real objects, and tokens that were never
    # objects at all. An answer's prose carries counts and areas that are
    # hex-shaped by accident, and calling those "parcels we could not find"
    # invents a problem out of a table cell.
    if wanted:
        real = _entities_among(drawing_id, wanted)
        body["highlight"] = {
            "asked": sorted(wanted),
            "cells": sorted(highlighted),
            "cells_lit": len(highlighted),
            "handles_not_objects": len(wanted - real) or None,
            "note": (
                f"{len(wanted)} token(s) asked for; {len(real)} are objects in "
                f"this drawing and their cells at resolution {target} are "
                "flagged `highlighted: true`. Only a PARCEL lights a cell: an "
                "answer naming a road or a text object lights none, because "
                "the cell layer never held one for it"
                + (
                    f". The remaining {len(wanted - real)} are not objects in "
                    "this drawing at all"
                    if wanted - real
                    else ""
                )
            ),
        }
    else:
        body["highlight"] = None

    if outline:
        # The site's own boundary, dissolved out of the cells the parcels
        # occupy. A map that wants "where is this development" draws this one
        # shape instead of 25,627 hexagons, and gets the same ground.
        from . import geo_h3  # noqa: PLC0415 -- deferred, one direction only

        body["coverage_outline"] = geo_h3.outline_of(coverage)
        body["coverage_outline_note"] = (
            "the boundary of the ground the parcels' cells cover, as one "
            "MultiPolygon. Its edges are hexagon edges, so it is as "
            "approximate as the cells are — it is not the site's surveyed "
            "boundary and must not be quoted as one"
        )

    if compact:
        packed = sorted(h3.compact_cells(coverage)) if coverage else []
        body["coverage_compact"] = packed
        body["coverage_note"] = (
            f"{len(packed):,} cells covering the same ground as "
            f"{len(coverage):,} at resolution {stored_res}. A SET, with no "
            "counts: compaction merges children into a parent and a count "
            "cannot survive that merge, so the counted cells above are never "
            "compacted"
        )

    return body


def resolutions_available(stored: int) -> Iterable[int]:
    """Resolutions this store can answer for, coarsest first.

    Never finer than what is stored. A finer cell is not a better answer
    derived from a coarser one; it is an invention, and `cell_to_children`
    would happily produce seven of them.
    """
    return range(0, stored + 1)


#: A colour per land use, for the one convention every GeoJSON viewer reads.
#:
#: Keyed by LAND USE, which is this repo's own vocabulary and applies to any
#: drawing (`landuse.USES`) — not by layer name, which would be one drawing's
#: trait and is what G1 forbids. A use with no entry here falls back to the
#: neutral colour rather than to a random one, so a vocabulary that grows does
#: not silently start colouring two things the same.
#:
#: The names are simplestyle-spec: geojson.io, GitHub's renderer and most
#: Leaflet setups read `fill`, `stroke` and `fill-opacity` off a feature's
#: properties. Nothing here is a map style file; it travels inside the data
#: because the file has to render for somebody who was sent nothing else.
LAND_USE_COLOURS: Mapping[str, str] = MappingProxyType(
    {
        "residential": "#e8a33d",
        "education": "#3d7fe8",
        "religious": "#7d5be8",
        "commercial": "#e85b8a",
        "community": "#e8d23d",
        "open_space": "#4caf50",
        "utility": "#9e9e9e",
        "parking": "#795548",
        "industrial": "#8d6e63",
        "hospitality": "#26a69a",
        "sports": "#66bb6a",
        "cemetery": "#78909c",
        "road": "#607d8b",
    }
)

#: What an unrecognised use is drawn in. Deliberately drab: a use nobody has
#: chosen a colour for should look unchosen rather than look like a category.
UNKNOWN_COLOUR = "#b0bec5"

#: The cell boundaries, drawn as an unfilled overlay. They sit on top of the
#: parcels and must not hide them.
CELL_COLOUR = "#38bdf8"

#: The catchment ring around an answer's objects. Faintly filled rather than
#: hollow, because it is the one shape in the file that means "this area"
#: rather than "this object" — and it is drawn under everything else.
CATCHMENT_COLOUR = "#f472b6"


def _style(
    colour: str,
    *,
    fill_opacity: float,
    stroke_width: float = 1.0,
    stroke_opacity: float = 1.0,
) -> dict[str, Any]:
    """simplestyle properties, spelled out once.

    Written as one helper rather than inline so that a change of convention —
    if a viewer ever wants `marker-color` or `stroke-opacity` too — happens in
    one place instead of in every feature builder.
    """
    return {
        "fill": colour,
        "fill-opacity": fill_opacity,
        "stroke": colour,
        "stroke-width": stroke_width,
        "stroke-opacity": stroke_opacity,
    }


def _entities_among(drawing_id: str, wanted: Iterable[str]) -> set[str]:
    """Which of these tokens are objects in this drawing at all.

    The export is handed whatever an answer NAMED, and an answer is prose: a
    table of areas and counts contributes "2761", "32912" and "109", all of
    which are hex-shaped and none of which is an object. Without this lookup
    the file could only say "71 asked for, 9 drawn" and had to describe the
    other 62 as objects it failed to draw — inventing 62 missing parcels out
    of table cells, which is the one thing the answer-marking rules already
    forbid.

    Queried by `_id`, the primary key, so it needs no index of its own and
    stays fast whatever the drawing's size — the same lookup shape
    `store.describe_handles` uses. Imported inside the function because
    `store` imports this module's neighbours at load time.
    """
    from app import store

    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        return set()
    rows = store.coll(store.COLL_ENTITIES).find(
        {"_id": {"$in": [f"{drawing_id}:{h}" for h in wanted]}},
        {"handle": 1},
    )
    return {str(row["handle"]).upper() for row in rows}


def export_geojson(
    drawing_id: str,
    *,
    res: int | None = None,
    layout: str | None = None,
    bbox: BBox | None = None,
    cells: bool = True,
    parcels: bool = True,
    limit: int = MAX_PARCEL_FEATURES,
    handles: Sequence[str] | None = None,
    rings: int = 0,
    context: bool = False,
) -> dict[str, Any]:
    """One FeatureCollection a person can drag into geojson.io.

    Everything this campaign can show, in the one file format every map tool
    opens: the parcels coloured by what they are, the hexagon cells as an
    overlay, and — on the collection itself and on every feature — the
    sentence saying the coordinate system was inferred.

    That last part is the reason this is not just `/geo` with different keys.
    A file gets forwarded. It arrives with no route, no documentation and
    nobody to ask, and the caveat has to be inside it or it does not exist.

    Deterministic: the same request produces byte-identical output, because
    the features come out of the same ordered aggregation `/geo` uses and
    nothing here reads a clock or a random number. That is what lets the ETag
    from the previous wave apply to this route unchanged.
    """
    body = geo(
        drawing_id,
        res=res,
        layout=layout,
        parcels=parcels,
        compact=False,
        limit=limit,
        bbox=bbox,
        outline=False,
    )

    # An answer's own objects, and nothing else.
    #
    # Filtered here rather than queried separately so that a file exported
    # from an answer is a SUBSET of the file exported from the drawing —
    # same colours, same properties, same caveat, same counting rule. Two
    # code paths would eventually give two answers to "what is this parcel",
    # and the one nobody exports by hand would be the one that drifted.
    wanted = {h.strip().upper() for h in (handles or []) if str(h).strip()}
    # Defined before the branch so the collection's properties can be built
    # unconditionally: a key that exists only on the filtered path is a key
    # every reader has to guard against.
    real: set[str] = set()
    undrawable: set[str] = set()
    not_objects: set[str] = set()
    answer_cells: set[str] = set()
    # The whole site's grid, held before the answer filter narrows it. This is
    # what `context` draws: the answer is only legible as "here and not
    # there", and a file containing four hexagons and nothing else cannot say
    # the "not there" part.
    site_cells: list[dict[str, Any]] = list(body["cells"])
    if wanted:
        kept = []
        for feature in body["parcels"]["features"]:
            if str(feature["properties"].get("handle") or "").upper() in wanted:
                kept.append(feature)
                if feature["properties"].get("cell"):
                    answer_cells.add(feature["properties"]["cell"])
        body["parcels"]["features"] = kept
        # The cell overlay follows the objects: an answer's file should show
        # the cells its objects are in, not the whole drawing's grid.
        body["cells"] = [c for c in body["cells"] if _covers_any(c["cell"], answer_cells)]
        # Three buckets, not two. `drawn` is what this file contains; `real`
        # is what the drawing actually holds; the difference between them is
        # a genuine exclusion and gets named, while everything outside `real`
        # was never an object reference and gets counted without being
        # dignified as a missing parcel.
        drawn = {str(f["properties"].get("handle") or "").upper() for f in kept}
        real = _entities_among(drawing_id, wanted)
        undrawable = real - drawn
        not_objects = wanted - real

    caveat = body.get("crs_caveat")
    provenance = {
        "drawing_id": drawing_id,
        "layout": body.get("layout"),
        "epsg": body.get("epsg"),
        "crs_source_layer": body.get("crs_source_layer"),
        "caveat": caveat,
        "generated_by": "cad-api GET /drawings/{id}/geo/export.geojson",
    }

    features: list[dict[str, Any]] = []

    if parcels:
        for feature in body["parcels"]["features"]:
            use = str(feature["properties"].get("land_use") or "")
            colour = LAND_USE_COLOURS.get(use, UNKNOWN_COLOUR)
            features.append(
                {
                    "type": "Feature",
                    "geometry": feature["geometry"],
                    "properties": {
                        # `kind` first, because it is what a reader filters
                        # on. Both kinds carry a `cell` — a parcel names the
                        # cell it is counted in — so "has a cell key" cannot
                        # tell them apart, and counting on that was this
                        # function's first bug: 3,190 features reported as
                        # 3,190 cells and 2,545 parcels at the same time.
                        "kind": "parcel",
                        **feature["properties"],
                        **_style(colour, fill_opacity=0.55),
                        "title": f"{use or 'unclassified'} · {feature['properties'].get('handle')}",
                        **provenance,
                    },
                }
            )

    # The grid the answer sits IN, faint and underneath, when asked for.
    #
    # Only alongside an answer: without one, `cells` already IS the whole
    # site, and emitting it twice under two names would be the same hexagons
    # claiming to be two different things.
    #
    # Faint by stroke opacity rather than by a paler colour. A paler blue is
    # a different value on the same ramp the counted cells use, and would
    # read as "fewer parcels here"; a washed-out version of the SAME colour
    # reads as what it is, the same grid turned down.
    if cells and context and wanted:
        answered_ids = {c["cell"] for c in body["cells"]}
        for cell in site_cells:
            if cell["cell"] in answered_ids:
                continue
            ring = [
                [round(lon, GEOJSON_DECIMALS), round(lat, GEOJSON_DECIMALS)]
                for lat, lon in h3.cell_to_boundary(cell["cell"])
            ]
            ring.append(list(ring[0]))
            total = sum(cell["counts"].values())
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "kind": "cell_context",
                        "cell": cell["cell"],
                        "resolution": body["resolution"],
                        "parcels": total,
                        "counts": cell["counts"],
                        "area_m2": cell.get("area_m2"),
                        # Tuned by eye against the rendered file, not
                        # picked from a palette. At 0.5px and 0.25 opacity
                        # the grid was present in the data and invisible on
                        # the map, which is the same as not exporting it:
                        # the answer needs something to be read AGAINST.
                        **_style(
                            CELL_COLOUR,
                            fill_opacity=0.0,
                            stroke_width=1.0,
                            stroke_opacity=0.5,
                        ),
                        "title": f"cell {cell['cell']} · {total} parcel(s) · context",
                        "context_note": (
                            "part of the site's grid, drawn so the answer's "
                            "own cells have something to stand out against. "
                            "It holds no object from this answer"
                        ),
                        **provenance,
                    },
                }
            )

    if cells:
        for cell in body["cells"]:
            ring = [
                [round(lon, GEOJSON_DECIMALS), round(lat, GEOJSON_DECIMALS)]
                for lat, lon in h3.cell_to_boundary(cell["cell"])
            ]
            ring.append(list(ring[0]))
            total = sum(cell["counts"].values())
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "kind": "cell",
                        "cell": cell["cell"],
                        "resolution": body["resolution"],
                        "parcels": total,
                        "counts": cell["counts"],
                        "area_m2": cell.get("area_m2"),
                        # Unfilled: the cells are a grid drawn OVER the
                        # parcels, and a filled hexagon hides the thing it is
                        # meant to be counting.
                        # Stronger when there is a context grid under it:
                        # the same 1.5px stroke that reads as "the grid" on
                        # its own has to read as "these ones" once the rest
                        # of the site is on the page with it.
                        **_style(
                            CELL_COLOUR,
                            fill_opacity=0.18 if (context and wanted) else 0.0,
                            stroke_width=2.5 if (context and wanted) else 1.5,
                        ),
                        "title": f"cell {cell['cell']} · {total} parcel(s)",
                        **provenance,
                    },
                }
            )

    # The catchment, when one was asked for. Drawn around the answer's own
    # cells with the same `grid_disk` the vicinity recipe uses and dissolved
    # by the same `outline_of`, so the shape in this file is the shape that
    # recipe counted inside — not a second idea of what "near" means.
    if rings and answer_cells:
        from . import geo_h3  # noqa: PLC0415 -- deferred, one direction only

        target_res = body["resolution"]
        own = {h3.cell_to_parent(c, target_res) for c in answer_cells}
        disk: set[str] = set()
        for cell in own:
            disk.update(h3.grid_disk(cell, rings))
        outline = geo_h3.outline_of(disk)
        if outline is not None:
            edge = h3.average_hexagon_edge_length(target_res, unit="m")
            reach = round(rings * (3 ** 0.5) * edge, 1)
            outline["properties"].update(
                {
                    "kind": "catchment",
                    "rings": rings,
                    "resolution": target_res,
                    "reach_centre_to_centre_m": reach,
                    "reach_to_disk_corner_m": round(reach + edge, 1),
                    **_style(CATCHMENT_COLOUR, fill_opacity=0.12, stroke_width=2.0),
                    "title": f"catchment · {rings} ring(s) ≈ {reach} m",
                    **provenance,
                }
            )
            features.append(outline)

    drawn_parcels = sum(1 for f in features if f["properties"]["kind"] == "parcel")

    return {
        "type": "FeatureCollection",
        # The same two flags the refusal shapes carry, in the same place, so a
        # reader tests one field rather than branching on whether the file
        # succeeded. A key that appears only when something is wrong is a key
        # nobody writes a branch for.
        "georeferenced": True,
        "indexed": True,
        # Not part of the GeoJSON spec and harmless to every reader of it:
        # unknown members on a FeatureCollection are ignored, and a file that
        # arrives with no context is worth more than a file that is pure.
        "properties": {
            **provenance,
            "resolution": body["resolution"],
            "stored_resolution": body["stored_resolution"],
            "answer_handles": sorted(wanted) if wanted else None,
            # Named, by name, because they are real objects this file does
            # not draw. The tokens that were never objects get a count and no
            # list: printing "2761" as a missing parcel would invent a
            # problem out of a table cell.
            "answer_handles_undrawable": sorted(undrawable) or None,
            "answer_handles_not_objects": len(not_objects) or None,
            "answer_handles_note": (
                None
                if not wanted
                else (
                    "this file is one answer's objects, not the drawing. "
                    f"{len(wanted)} token(s) were asked for; {len(real)} are "
                    f"objects in this drawing and {drawn_parcels} of those had "
                    "a boundary this store will vouch for. "
                    + (
                        f"{len(undrawable)} real object(s) are named in "
                        "`answer_handles_undrawable` and not drawn here. "
                        if undrawable
                        else ""
                    )
                    + (
                        f"The remaining {len(not_objects)} are not objects in "
                        "this drawing at all — an answer's prose contains "
                        "counts and areas that are hex-shaped by accident. "
                        if not_objects
                        else ""
                    )
                    + "`totals_whole_drawing` is still the whole drawing"
                )
            ),
            "parcels_in_file": drawn_parcels,
            "cells_in_file": sum(
                1 for f in features if f["properties"]["kind"] == "cell"
            ),
            # Counted separately and named separately. A reader filtering on
            # `kind` needs to know the two are not the same thing, and a
            # single "cells" number that silently included the context would
            # make an answer covering four hexagons look like one covering
            # two thousand.
            "context_cells_in_file": sum(
                1 for f in features if f["properties"]["kind"] == "cell_context"
            )
            or None,
            "context_note": (
                "`kind: \"cell_context\"` is the rest of the site's grid, "
                "drawn faint so the answer's own cells read as a cluster on "
                "it. Those carry `kind: \"cell\"` and a stronger style. "
                "Filter on `kind` to separate them"
                if (context and wanted)
                else None
            ),
            "totals_whole_drawing": body["totals"],
            "viewport": body.get("viewport"),
            "note": (
                "colours are simplestyle properties on each feature, so this "
                "renders without a style file. `totals_whole_drawing` counts "
                "the WHOLE drawing even when a viewport was applied — the "
                "features may be a part of it"
            ),
            "not_a_survey": (
                "every coordinate here was reprojected from drawing "
                "coordinates through a coordinate system that "
                + (
                    "the file states"
                    if body.get("crs_source_layer") == "declared_in_file"
                    else "was INFERRED, because the file does not state one"
                )
                + ". Areas and lengths must be read from the drawing, not "
                "measured off these shapes"
            ),
        },
        "features": features,
    }

