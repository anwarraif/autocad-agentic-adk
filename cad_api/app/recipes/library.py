"""Eight analysis recipes, all of them derived from real computations.

Owner: subagent R — UPLIFT-09.

Every recipe here **rides on** functions that already exist in
`store_spatial`, `store_geometry`, `store_stats`, and `store_landuse`. Not one
of them rewrites geometry, and that is not a saving in lines but a condition
of correctness: a second ray-caster in this repo would answer differently from
the first one on the same polygon, and there is nothing in either response
that shows which one was used.

That is why `Recipe.built_on` names the functions that are really called, and
it is published in every response.

## Layer names never live in this file (G1)

Not one typology code, plot-number layer name, or plot size range is written
in this file — and the G1 grep is run in the tests, not relied on as a habit.
What names the layers is the **caller** or **that drawing's land use config**
(UPLIFT-02), and a recipe that derives them from the config says which layers
it used. A default inside a `.py` would make this module a solution specific
to one drawing on the day the 17th drawing arrives.

## Three departures from the parameter table in the spec, and why

1.  **`size_module` does not accept `tolerance`.** `shape_key` is computed at
    ingest and `_by_shape_key` reads it as it is; a `tolerance` here would not
    touch the shape grouping at all. A parameter that is accepted and then
    ignored is worse than a parameter that does not exist: the first one
    promises a control it does not have.
2.  **`cross_check_classification` accepts `edge_range` and `area_range`, not
    a single `shape_test`.** A single `shape_test` would have to be
    interpreted, and interpreting a test handed over by the caller is one step
    away from running it — exactly what is refused at the head of
    `registry.py`.
3.  **`coverage_gap` accepts `parcel_use`.** Without it, the only way to say
    "residential parcels" is to type all thirteen of their layer names, and a
    list of layer names typed at the call site is a drawing-specific constant
    that has moved, not one that has gone.
"""

from __future__ import annotations

import math
from typing import Any, Final, Mapping, Sequence

from .. import evidence as ev
from .. import landuse
from .. import store_geometry as geometry
from .. import store_landuse as landuse_store
from .. import store_spatial as spatial
from .. import store_stats as stats
from ..extract import RING_TYPES
from ..mongo import COLL_ENTITIES, coll
from .registry import (
    COMPUTED_PROVENANCE,
    MAX_ROWS_RETURNED,
    Param,
    Recipe,
    RecipeRefused,
    refuse_oversize,
    register,
    scope_for,
)

# --- stated limits (G7) ------------------------------------------------------
#
# A number that does not belong to this module is taken from its owner, not
# copied. A copied limit would differ from the limit that really applies on
# the first day one of them is changed, and its refusal would name a number
# that did not refuse it.

#: The `edge` mode limit in `store_spatial`. It is what really binds
#: `adjacency`, and the message names that number as it is.
MAX_ADJACENCY_ITEMS: Final[int] = spatial.MAX_ITEMS["edge"]

#: Inner polygons and outer polygons for `containment`. The outer limit is far
#: smaller because every outer polygon holds its whole ring in memory.
MAX_CONTAINMENT_INNER: Final[int] = 25_000
MAX_CONTAINMENT_OUTER: Final[int] = 2_000

#: How many number texts `coverage_gap` may count before it refuses. Aligned
#: with `store_spatial.MAX_CANDIDATES`, and taken from there.
MAX_NUMBER_ENTITIES: Final[int] = spatial.MAX_CANDIDATES

#: A separate limit for `distinct`, far tighter: its result is ONE MongoDB
#: document, and a document stops at 16 MB.
MAX_DISTINCT_VALUES: Final[int] = 50_000


def _entities():
    """This file's only door to MongoDB, and it goes through `coll()`."""
    return coll(COLL_ENTITIES)


def _ring_type_filter() -> dict[str, Any]:
    return {"$in": sorted(RING_TYPES)}


# --- layers from config, not from code ---------------------------------------


def _layers_from_config(
    drawing_id: str, *, role: str = "parcel", use: str | None = None
) -> tuple[tuple[str, ...], str]:
    """The layers THIS drawing's config names, plus a sentence naming where they came from.

    This is the only legitimate way for a recipe to know which layers are
    "parcel" (G4, G10). The classification is re-read per drawing, per call; a
    layer name that means residential in this drawing does not necessarily
    mean residential in another drawing.

    It REFUSES — rather than guessing — when the config does not exist yet.
    The refusal is actionable in two directions: create the config, or name
    the layers directly in the parameter. A guess here would read as a fact in
    every number that followed it.
    """
    try:
        config = landuse.for_drawing(drawing_id)
    except landuse.ConfigError as exc:
        raise landuse_store.LandUseConfigInvalid(
            exc.code, exc.message, exc.hint
        ) from exc

    if config is None:
        raise RecipeRefused(
            "RECIPE_NO_CONFIG",
            f"drawing {drawing_id!r} has no land use config yet, so the "
            f"layers with role {role!r} cannot be derived.",
            "Name the layers directly in the parameter, or create "
            f"`{landuse.config_path(drawing_id).name}`. Guessing which layers "
            "are parcels would make every number after it read as a fact when "
            "it is a guess.",
        )

    names = tuple(
        sorted(
            name
            for name, entry in config.layers.items()
            if entry.role == role and (use is None or entry.use == use)
        )
    )
    if not names:
        raise RecipeRefused(
            "RECIPE_NO_CONFIG",
            f"config {config.path} names no layer at all with role "
            f"{role!r}" + (f" and land use {use!r}" if use else "") + ".",
            "Name the layers directly in the parameter, or add that layer to "
            "the config through a merge request. Zero layers is a finding, "
            "not a reason to use the whole drawing instead.",
        )
    where = f"land use config {config.path} (role={role}"
    where += f", use={use})" if use else ")"
    return names, where


def _resolve_layers(
    drawing_id: str,
    given: Sequence[str] | None,
    *,
    role: str = "parcel",
    use: str | None = None,
) -> tuple[tuple[str, ...], str]:
    if given:
        return tuple(given), "the list of layers handed over by the caller"
    return _layers_from_config(drawing_id, role=role, use=use)


# --- evidence used by several recipes ----------------------------------------


def _topology_observation(detail: str, *, locator: str, layout: str | None) -> ev.Observation:
    """An observation from coordinates. Its corpus is `coordinates`, which NEVER states.

    That is not a weakness to be worked around. A polygon that coincides with
    another polygon implies that the two are copies; the file itself says
    nothing about that, and an answer that reads "stated by the file" for a
    geometric conclusion is a grade rise in disguise.
    """
    return ev.Observation(
        origin=ev.Origin.TOPOLOGY,
        detail=detail,
        locator=locator,
        layout=layout,
    )


def _absence_observation(detail: str, *, locator: str, layout: str | None) -> ev.Observation:
    return ev.Observation(
        origin=ev.Origin.ABSENCE,
        detail=detail,
        locator=locator,
        layout=layout,
    )


def _computed_evidence(
    claim: str,
    *,
    drawing_id: str,
    scope: ev.Scope,
    observations: Sequence[ev.Observation],
    not_established: str,
    how_to_verify: str,
) -> ev.Evidence:
    """Evidence for a claim born from computation, not from vocabulary.

    `tokens` is always empty, and that is a decision rather than an oversight:
    there is no word that, when it appears in a file string, makes that file
    STATE that 69 plot numbers have no polygon. This claim can therefore never
    be `stated`, and that ceiling is right.
    """
    return ev.Evidence.of(
        ev.Claim(value=claim, tokens=()),
        drawing_id=drawing_id,
        scope=scope,
        provenance=COMPUTED_PROVENANCE,
        observations=tuple(observations),
        not_established=not_established,
        how_to_verify=how_to_verify,
        ceiling=ev.Grade.INFERRED,
    )


# =============================================================================
# 1. parcel_inventory
# =============================================================================


def _run_parcel_inventory(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    role: str,
    min_area: float | None,
    max_area: float | None,
) -> dict[str, Any]:
    summary = landuse_store.land_use_summary(drawing_id, layout)
    if not summary:
        raise RecipeRefused(
            "RECIPE_NO_CONFIG",
            f"drawing {drawing_id!r} is not in the store.",
            "Check the drawing_id. A misspelled id returns a zero that cannot "
            "be told apart from a correct zero.",
        )

    body: dict[str, Any] = dict(summary)
    wanted = (role or "parcel").strip().lower()
    if wanted == "parcel":
        # `roles_not_counted` stays in the response — removed, it would make
        # hatches and labels read as "not there", when both are there and are
        # deliberately not counted as plots.
        body["role_filter"] = "parcel"
    elif wanted == "all":
        body["role_filter"] = "all"
    else:
        rows = [r for r in body.get("roles_not_counted", []) if r.get("role") == wanted]
        body["roles_not_counted"] = rows
        body["uses"] = []
        body["role_filter"] = wanted

    body["role_filter_note"] = (
        "role=parcel is the only one that produces a plot count. Hatches "
        "(overlay), labels (annotation), and block boundaries (structure) are "
        "reported separately and not added in — adding them counts the same "
        "area twice."
    )

    if min_area is None and max_area is None:
        body["size_band"] = None
        body["size_band_note"] = (
            "no size filter was asked for, so every row in `uses` covers all "
            "of its parcels"
        )
        return body

    lo = -math.inf if min_area is None else float(min_area)
    hi = math.inf if max_area is None else float(max_area)
    bands: list[dict[str, Any]] = []
    for row in body.get("uses", []):
        layers = [r["layer"] for r in row.get("by_layer", [])]
        if not layers:
            continue
        raw = stats.values_for(
            drawing_id,
            measure="area",
            layout_name=layout or "",
            layers=layers,
        )
        values = [v for name in raw["layers_present"] for v in raw["values"].get(name, [])]
        inside = [v for v in values if lo <= v <= hi]
        bands.append(
            {
                "use": row["use"],
                "parcels": row["parcels"],
                "parcels_measured": len(values),
                "parcels_in_band": len(inside),
                "parcels_outside_band": len(values) - len(inside),
                # A parcel with no measured area is neither inside the band NOR
                # outside it: it was not checked at all, and putting it on
                # either side is a wrong answer, not an empty one.
                "parcels_not_measured": row["parcels"] - len(values),
                "band_area": ev.total(
                    [
                        ev.Quantity.measured(
                            basis=f"{row['use']} parcels inside the band",
                            method="Σ area of the parcels whose area falls inside the band",
                            value=v,
                            unit=units.get("area_unit"),
                            unit_reason=(
                                None
                                if units.get("area_unit")
                                else "this drawing does not declare units"
                            ),
                        )
                        for v in inside
                    ],
                    basis=f"Σ area of the {row['use']} parcels inside the band",
                    method="Σ area of the parcels whose area falls inside the band",
                ).as_dict(),
            }
        )

    body["size_band"] = {
        "min_area": min_area,
        "max_area": max_area,
        "unit": units.get("area_unit"),
        "unit_reason": (
            None if units.get("area_unit") else "this drawing does not declare units"
        ),
        "by_use": bands,
    }
    body["size_band_note"] = (
        "this band FILTERS, it does not recount. The numbers in `uses` remain "
        "the whole population; `parcels_in_band` counts how many of them have "
        "an area that falls inside the band. Parcels whose area is not "
        "measured are counted separately in `parcels_not_measured` and are "
        "not treated as zero."
    )
    return body


register(
    Recipe(
        name="parcel_inventory",
        answers="which parcels this drawing contains, how many, and how large",
        when_to_use=(
            "an opening question about what a drawing contains — 'how many "
            "houses are there', 'is there a school', 'what is the "
            "composition'. Not for finding a single object and not for "
            "totalling one layer."
        ),
        params=(
            Param(
                "role",
                "text",
                "the layer role that is counted: `parcel` (the default, and "
                "the only one that produces a plot count), `all`, or one of "
                "the non-parcel roles to see it on its own.",
                default="parcel",
            ),
            Param(
                "min_area",
                "number",
                "the smallest area that enters the band, in DRAWING units. It "
                "filters, it does not recount.",
            ),
            Param(
                "max_area",
                "number",
                "the largest area that enters the band, in DRAWING units.",
            ),
        ),
        returns=(
            "uses[].parcels",
            "uses[].total_area",
            "uses[].evidence",
            "roles_not_counted",
            "empty_but_declared",
            "size_band",
        ),
        built_on=("store_landuse.land_use_summary", "store_stats.values_for"),
        limits={
            "layer_rows_per_use": landuse_store.MAX_LAYER_ROWS,
            "rows_scanned_for_band": stats.MAX_ROWS,
        },
        run=_run_parcel_inventory,
    )
)


# =============================================================================
# 2. label_coverage
# =============================================================================


def _run_label_coverage(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    target_layers: Sequence[str] | None,
    label_layer: str | None,
    label_type: str,
) -> dict[str, Any]:
    layers, where = _resolve_layers(drawing_id, target_layers, role="parcel")
    body = dict(
        spatial.join_labels(
            drawing_id,
            layout=layout or "",
            target_layer=list(layers),
            label_layer=label_layer,
            label_type=label_type,
            limit=MAX_ROWS_RETURNED,
        )
    )
    total = body["total_targets"]
    with_label = body["targets_with_label"]
    body["target_layers"] = list(layers)
    body["target_layers_source"] = where
    body["coverage_fraction"] = (with_label / total) if total else None
    body["coverage_fraction_note"] = (
        "the numerator is polygons that CONTAIN text, not texts that were "
        "found. Counting labels as a parcel count is the mistake that makes "
        "one parcel with two labels count as two parcels."
    ) if total else "no target polygons, so there is no fraction to compute"
    body["without_label_is_not_without_parcel"] = (
        f"{body['targets_without_label']} polygons contain no text inside "
        "their boundary. That is the absence of a LABEL, not the absence of a "
        "parcel — and a polygon skipped because its ring is not closed does "
        "not appear in this number at all; it is counted separately, per "
        "cause."
    )
    return body


register(
    Recipe(
        name="label_coverage",
        answers="how many parcels have a label inside them, and how many do not",
        when_to_use=(
            "when the question is 'are all the plots numbered', 'how many are "
            "not labelled yet', or when a label count is about to be used as "
            "a parcel count — where it is almost always wrong."
        ),
        params=(
            Param(
                "target_layers",
                "layers",
                "the polygon layers that are checked. Left empty, they are "
                "derived from this drawing's land use config (role=parcel) "
                "and the response names those layers.",
            ),
            Param(
                "label_layer",
                "layer",
                "restrict the text considered to a single layer. Left empty, "
                "text from any layer takes part.",
            ),
            Param(
                "label_type",
                "text",
                "`TEXT`, `MTEXT`, `INSERT`, or `ANY`.",
                default="TEXT",
            ),
        ),
        returns=(
            "total_targets",
            "targets_with_label",
            "targets_without_label",
            "coverage_fraction",
            "boundary_cases",
            "skipped_bulge/open/degenerate/oversize",
        ),
        built_on=("store_spatial.join_labels",),
        limits={
            "targets": spatial.MAX_TARGETS,
            "label_candidates": spatial.MAX_CANDIDATES,
            "pairs_after_prefilter": spatial.PAIRS_BUDGET,
            "rows_returned": MAX_ROWS_RETURNED,
        },
        run=_run_label_coverage,
    )
)


# =============================================================================
# 3. size_module
# =============================================================================


def _run_size_module(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    layers: Sequence[str] | None,
    top: float | None,
) -> dict[str, Any]:
    limit = int(top) if top else 20
    limit = max(1, min(limit, MAX_ROWS_RETURNED))
    body = dict(
        geometry.shape_fingerprint(
            drawing_id,
            layout=layout or "",
            units=units,
            layers=list(layers) if layers else None,
            limit=limit,
        )
    )
    body["layers_requested"] = list(layers) if layers else None
    body["module_note"] = (
        "the plot module is read from the geometry itself, without reading a "
        "single layer name. That is what makes this reading survive into the "
        "next drawing: another contractor uses different layer conventions, "
        "and a 12x25 plot is still a 12x25 plot. `modal_edge_lengths` is a "
        "DESCRIPTION, not a filter — a drawing with no module at all returns "
        "groups of one member, and that is an answer."
    )
    body["no_tolerance_note"] = (
        "this recipe does not accept `tolerance`. `shape_key` is computed at "
        "ingest and read as it is, so a tolerance here would not touch the "
        "shape grouping at all — and a parameter that is accepted and then "
        "ignored promises a control it does not have."
    )
    return body


register(
    Recipe(
        name="size_module",
        answers="which plot modules this drawing uses, read from its own geometry",
        when_to_use=(
            "'what size are the plots', 'how many size types are there', or "
            "when a layer-name-based classification needs to be compared with "
            "the shape that is actually drawn."
        ),
        params=(
            Param(
                "layers",
                "layers",
                "restrict to specific layers. Left empty, every polygon with "
                "a complete ring on that layout takes part — and that is "
                "indeed how a module is read without relying on layer names.",
            ),
            Param(
                "top",
                "number",
                "how many of the top shape groups are broken out. The rest "
                "are still counted in `distinct_shapes`.",
                default=20,
            ),
        ),
        returns=("groups[].count", "groups[].modal_edge_lengths", "distinct_shapes", "skipped"),
        built_on=("store_geometry.shape_fingerprint",),
        limits={
            "target_polygons": geometry.MAX_TARGET_POLYGONS,
            "rows_returned": MAX_ROWS_RETURNED,
        },
        run=_run_size_module,
    )
)


# =============================================================================
# 4. adjacency
# =============================================================================


def _run_adjacency(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    layers: Sequence[str] | None,
    handles: Sequence[str] | None,
    gap_max: float | None,
) -> dict[str, Any]:
    if not layers and not handles:
        raise RecipeRefused(
            "RECIPE_PARAM_REQUIRED",
            "recipe 'adjacency' needs `layers` or `handles`.",
            "Name which parcels are being compared. A matrix over the whole "
            "drawing is not a broader answer — it is an answer nobody can "
            "read, and the `edge` mode limit "
            f"({MAX_ADJACENCY_ITEMS} items) would refuse it.",
        )
    body = dict(
        spatial.distance_matrix(
            drawing_id,
            layout=layout or "",
            handles=list(handles) if handles else None,
            layers=list(layers) if layers else None,
            mode="edge",
            cluster_threshold=gap_max,
        )
    )
    body["adjacency_note"] = (
        "an edge distance of 0 means the two polygons touch. This is not an "
        "adjacency graph over the whole drawing: `edge` mode costs O(n·m) per "
        f"pair of EDGES and is therefore limited to {MAX_ADJACENCY_ITEMS} items. "
        "For 'how many are around each object' on a large population, the one "
        "that answers is proximity_count, whose output is O(n)."
    )
    if gap_max is None:
        body["cluster_note"] = (
            "`gap_max` was not given, so no clusters were formed. An "
            "adjacency threshold is a project's planning number, not a "
            "property of a CAD drawing, so it has no default."
        )
    return body


register(
    Recipe(
        name="adjacency",
        answers="which parcels are next to which, measured edge to edge",
        when_to_use=(
            "'what is this one next to', 'how many separate clusters are "
            "there'. Only for small sets that have already been named; for a "
            "large population use proximity_count."
        ),
        params=(
            Param("layers", "layers", "every parcel on these layers."),
            Param("handles", "layers", "or the parcels named one by one."),
            Param(
                "gap_max",
                "number",
                "the largest gap still counted as adjacent, in DRAWING units. "
                "Without it no clusters are formed, and that is said out loud "
                "— the threshold is a planning number, not a drawing's trait.",
            ),
        ),
        returns=("matrix", "nearest", "clusters", "not_measured", "skipped_*"),
        built_on=("store_spatial.distance_matrix",),
        limits={
            "items": MAX_ADJACENCY_ITEMS,
            "edge_pairs": spatial.MAX_EDGE_PAIRS,
        },
        carries_meaning=False,
        run=_run_adjacency,
    )
)


# =============================================================================
# 5. containment
# =============================================================================


def _run_containment(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    inner_layers: Sequence[str] | None,
    outer_layer: str,
) -> dict[str, Any]:
    inner_names, where = _resolve_layers(drawing_id, inner_layers, role="parcel")

    entities = _entities()
    outer_query = {
        "drawing_id": drawing_id,
        "layout": layout,
        "layer": outer_layer,
        "ring_status": "complete",
    }
    outer_total = entities.count_documents(dict(outer_query))
    refuse_oversize(
        what=f"outer polygons on layer {outer_layer!r}",
        size=outer_total,
        limit=MAX_CONTAINMENT_OUTER,
        hint=(
            "Every outer polygon holds its whole ring in memory for as long "
            "as the test runs. Narrow `outer_layer`, or turn the question "
            "around and use label_coverage if what is really being looked for "
            "is text inside polygons."
        ),
    )

    inner_query = {
        "drawing_id": drawing_id,
        "layout": layout,
        "layer": {"$in": list(inner_names)},
        "ring_status": "complete",
    }
    inner_total = entities.count_documents(dict(inner_query))
    refuse_oversize(
        what="inner polygons to be tested",
        size=inner_total,
        limit=MAX_CONTAINMENT_INNER,
        hint="Narrow `inner_layers`.",
    )

    outer = list(
        entities.find(
            dict(outer_query),
            {"handle": 1, "layer": 1, "ring": 1, "ring_origin": 1, "area": 1, "_id": 0},
        ).sort([("handle", 1)])
    )
    # Inner polygons are handed to the point-in-polygon engine as POINTS, and
    # the point is `polygon_centroid` — not the centre of the bounding box,
    # which on a rotated grid misses by several metres with no sign of it at
    # all in the response. A polygon without a centroid is skipped AND counted.
    inner_docs = list(
        entities.find(
            dict(inner_query),
            {"handle": 1, "layer": 1, "polygon_centroid": 1, "area": 1, "_id": 0},
        ).sort([("layer", 1), ("handle", 1)])
    )
    points: list[dict[str, Any]] = []
    no_centroid = 0
    for doc in inner_docs:
        centre = doc.get("polygon_centroid")
        if not centre or len(centre) < 2:
            no_centroid += 1
            continue
        points.append(
            {
                "handle": doc.get("handle"),
                "layer": doc.get("layer"),
                "type": "polygon_centroid",
                "text": None,
                "anchor_point": [float(centre[0]), float(centre[1])],
                "anchor_basis": "polygon_centroid",
            }
        )

    # `spatial._join` is used as it is — the grid prefilter, the pair budget,
    # the boundary rule, and its edge-case counts have all been reviewed and
    # tested. Copying its logic here would put a SECOND ray-caster in this
    # repo, and two ray-casters would answer differently on the same polygon
    # with no sign of it at all in either response.
    joined = spatial._join(outer, points, budget=spatial.PAIRS_BUDGET)

    placed = sum(row["label_count"] for row in joined["rows"])
    shared = joined["labels_matched_more_than_one_target"]
    with_inner = sum(1 for row in joined["rows"] if row["label_count"])

    scope_note = (
        f"layout {layout!r}; {outer_total} outer polygons on layer "
        f"{outer_layer!r} and {len(points)} inner polygons from "
        f"{len(inner_names)} layers, all of them `ring_status == 'complete'`; "
        "test point `polygon_centroid`"
    )
    scope = scope_for(layout=layout, units=units, note=scope_note)

    rows = [
        {
            "outer_handle": row["target_handle"],
            "outer_layer": row["target_layer"],
            "inner_count": row["label_count"],
            "inner": [
                {"handle": m["handle"], "layer": m["layer"], "where": m["where"]}
                for m in row["labels"][:20]
            ],
            "inner_truncated": row["label_count"] > 20,
        }
        for row in joined["rows"][:MAX_ROWS_RETURNED]
    ]

    observations = [
        _topology_observation(
            f"{placed} inner-polygon centroids fall inside the ring of a "
            f"polygon on layer {outer_layer!r}",
            locator=outer_layer,
            layout=layout,
        )
    ]
    if no_centroid:
        observations.append(
            _absence_observation(
                f"{no_centroid} inner polygons have no `polygon_centroid` and "
                "could not be tested at all",
                locator=",".join(inner_names[:3]),
                layout=layout,
            )
        )

    evidence = _computed_evidence(
        f"{with_inner} of {outer_total} polygons on layer {outer_layer!r} "
        f"contain the centroid of {placed} inner polygons",
        drawing_id=drawing_id,
        scope=scope,
        observations=observations,
        not_established=(
            "this is a CENTROID-INSIDE test, not a full containment test. An "
            "inner polygon that sticks out past its outer boundary still "
            "counts as 'inside' as long as its centroid is inside, and a "
            "concave outer polygon can contain a centroid without containing "
            "the parcel"
        ),
        how_to_verify=(
            "compare areas: Σ of the inner polygon areas of a block exceeding "
            "the area of that block itself is a sign that some of them stick "
            "out. A full containment test — every vertex, not one point — "
            "does not exist yet, and could be added as a recipe"
        ),
    )

    body: dict[str, Any] = {
        "scope_note": scope_note,
        "outer_layer": outer_layer,
        "inner_layers": list(inner_names),
        "inner_layers_source": where,
        "method": "point-in-polygon (ray casting) on the inner polygons' `polygon_centroid`",
        "point_used": "the inner polygon's polygon_centroid, not the centre of its bounding box",
        "outer_total": outer_total,
        "outer_with_inner": with_inner,
        "outer_without_inner": outer_total - with_inner,
        "inner_total": inner_total,
        "inner_placed": placed,
        "inner_matched_more_than_one_outer": shared,
        "inner_without_outer": (
            inner_total - placed
            if shared == 0
            else None
        ),
        "inner_without_outer_note": (
            None
            if shared == 0
            else (
                f"{shared} inner polygons fall inside more than one outer "
                "polygon, so the number of placements is not the number of "
                "polygons and the difference is WITHHELD rather than guessed. "
                "Overlapping outer polygons are a finding of their own"
            )
        ),
        "inner_without_centroid": no_centroid,
        "inner_without_centroid_note": (
            "polygons with a closed ring but no `polygon_centroid` — their "
            "signed area is zero, their ring folds back on itself. They are "
            "skipped AND counted; the centre of the bounding box is never a "
            "fallback"
        ),
        "boundary_cases": joined["boundary_cases"],
        "boundary_note": (
            "a centroid that falls exactly on the line is reported as inside "
            "AND counted here"
        ),
        "pairs_after_prefilter": joined["pairs_after_prefilter"],
        "pairs_tested": joined["pairs_tested"],
        "returned": len(rows),
        "truncated": len(rows) < len(joined["rows"]),
        "rows": rows,
    }
    return ev.attach(body, evidence)


register(
    Recipe(
        name="containment",
        answers="which parcels lie inside which block",
        when_to_use=(
            "'which block is this plot in', 'how many plots are in this "
            "block', or when a per-block count is about to be used and no "
            "field in the file states it."
        ),
        params=(
            Param(
                "inner_layers",
                "layers",
                "the polygon layers whose parent is being looked for. Left "
                "empty, derived from this drawing's land use config "
                "(role=parcel).",
            ),
            Param(
                "outer_layer",
                "layer",
                "the single polygon layer that acts as the container — a "
                "block boundary, a district boundary, whatever this drawing "
                "uses.",
                required=True,
            ),
        ),
        returns=(
            "outer_with_inner",
            "outer_without_inner",
            "inner_placed",
            "inner_without_outer",
            "boundary_cases",
            "rows[].inner",
        ),
        built_on=("store_spatial._join", "store_spatial.PAIRS_BUDGET"),
        limits={
            "outer_polygons": MAX_CONTAINMENT_OUTER,
            "inner_polygons": MAX_CONTAINMENT_INNER,
            "pairs_after_prefilter": spatial.PAIRS_BUDGET,
            "rows_returned": MAX_ROWS_RETURNED,
        },
        run=_run_containment,
    )
)


# =============================================================================
# 6. coverage_gap
# =============================================================================


def _run_coverage_gap(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    number_layer: str,
    parcel_layers: Sequence[str] | None,
    parcel_use: str | None,
) -> dict[str, Any]:
    names, where = _resolve_layers(
        drawing_id, parcel_layers, role="parcel", use=parcel_use
    )

    entities = _entities()
    # The filter is prefixed by `drawing_id` + `layout`, which is exactly the
    # `(drawing_id, layout)` index. This cluster runs with `notablescan`: a
    # query without an indexed plan does not slow down, it FAILS.
    number_query: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "layer": number_layer,
        "text": {"$nin": [None, ""]},
    }
    numbers_total = entities.count_documents(dict(number_query))
    refuse_oversize(
        what=f"texts on layer {number_layer!r}",
        size=numbers_total,
        limit=MAX_NUMBER_ENTITIES,
        hint="Narrow the `layout`, or name a more specific number layer.",
    )
    # `distinct` returns ONE document holding all of its values, and a MongoDB
    # document stops at 16 MB. Hence a separate limit, tighter than the count
    # limit: above this the number is WITHHELD together with its reason, not
    # reported as zero.
    if numbers_total <= MAX_DISTINCT_VALUES:
        numbers_distinct: int | None = len(entities.distinct("text", dict(number_query)))
        distinct_note = None
    else:
        numbers_distinct = None
        distinct_note = (
            f"{numbers_total} texts is above the limit of {MAX_DISTINCT_VALUES} "
            "for collecting distinct values; the number is withheld, and "
            "withheld is not zero"
        )

    parcel_query = {
        "drawing_id": drawing_id,
        "layout": layout,
        "layer": {"$in": list(names)},
        "type": _ring_type_filter(),
    }
    parcels_total = entities.count_documents(dict(parcel_query))

    joined = spatial.join_labels(
        drawing_id,
        layout=layout or "",
        target_layer=list(names),
        label_layer=number_layer,
        label_type="ANY",
        limit=1,
    )
    placed = joined["label_count"]
    shared = joined["labels_matched_more_than_one_target"]

    gap = numbers_total - parcels_total
    numbers_without_parcel = (numbers_total - placed) if shared == 0 else None

    scope_note = (
        f"layout {layout!r}; number = a text-bearing entity on layer "
        f"{number_layer!r}; parcel = {'/'.join(sorted(RING_TYPES))} on "
        f"{len(names)} layers from {where}"
    )
    scope = scope_for(layout=layout, units=units, note=scope_note)

    observations = [
        _topology_observation(
            f"{numbers_total} text-bearing entities on layer {number_layer!r} "
            f"on layout {layout!r}"
            + (
                f", {numbers_distinct} of them with distinct values"
                if numbers_distinct is not None
                else ""
            ),
            locator=number_layer,
            layout=layout,
        ),
        _topology_observation(
            f"{parcels_total} polygons on {len(names)} parcel layers on the same layout",
            locator=",".join(names[:3]),
            layout=layout,
        ),
        _absence_observation(
            f"{abs(gap)} "
            + ("numbers without a polygon" if gap > 0 else "polygons without a number")
            + " after the two counts were compared",
            locator=number_layer,
            layout=layout,
        ),
    ]

    evidence = _computed_evidence(
        f"{numbers_total} numbers and {parcels_total} parcel polygons in this "
        f"scope, a difference of {abs(gap)}",
        drawing_id=drawing_id,
        scope=scope,
        observations=observations,
        not_established=(
            "the CAUSE of this difference is not established by this drawing. "
            "The number is an observation, not an explanation: a number "
            "without a polygon can mean the plot was drawn on another layer, "
            "the plot has not been drawn yet, or the number is left over from "
            "an earlier revision, and the file does not say which"
        ),
        how_to_verify=(
            "run it again with a wider `parcel_layers` — if the difference "
            "shrinks, those numbers really do sit on parcels whose layers "
            "have not been named. The rest needs the plot list from the "
            "drawing's owner; the DXF file does not carry it"
        ),
    )

    body: dict[str, Any] = {
        "scope_note": scope_note,
        "number_layer": number_layer,
        "parcel_layers": list(names),
        "parcel_layers_source": where,
        "numbers_total": numbers_total,
        "numbers_distinct": numbers_distinct,
        "numbers_distinct_note": distinct_note,
        "numbers_repeated": (
            None if numbers_distinct is None else numbers_total - numbers_distinct
        ),
        "parcels_total": parcels_total,
        "gap": gap,
        "gap_direction": (
            "more numbers than polygons"
            if gap > 0
            else ("more polygons than numbers" if gap < 0 else "balanced")
        ),
        # This is the sentence that has never been said to anyone, and the
        # reason a drawing containing 2,380 parcels was once answered with a
        # larger number: those two counts count different things, and only one
        # of them is a house.
        "what_each_number_counts": {
            "numbers_total": (
                f"TEXT entities on layer {number_layer!r}. This is a LABEL "
                "count, not a parcel count: a label can exist without its "
                "polygon, and using it to answer 'how many houses are there' "
                "overstates the answer by exactly the difference below"
            ),
            "parcels_total": (
                "polygons on the layers whose config declares role=parcel. "
                "THIS is what answers 'how many plots are there'"
            ),
        },
        "answer_to_how_many": parcels_total,
        "answer_to_how_many_basis": landuse_store.PARCEL_BASIS,
        "join": {
            "targets_tested": joined["total_targets"],
            "targets_with_number": joined["targets_with_label"],
            "targets_without_number": joined["targets_without_label"],
            "numbers_placed": placed,
            "numbers_matched_more_than_one_parcel": shared,
            "boundary_cases": joined["boundary_cases"],
            "skipped_targets_total": joined["skipped_targets_total"],
            "method": joined["method"],
            "point_used": joined["point_used"],
        },
        "numbers_without_parcel": numbers_without_parcel,
        "numbers_without_parcel_note": (
            "numbers whose insertion point does not fall inside any parcel "
            "polygon in this scope"
            if shared == 0
            else (
                f"WITHHELD: {shared} numbers fall inside more than one "
                "polygon, so the number of placements is not the number of "
                "numbers. A withheld number is not zero"
            )
        ),
        "parcels_without_number": joined["targets_without_label"],
        "is_an_observation_not_an_explanation": (
            "this difference is an observation, not an explanation: it was "
            "OBSERVED, and its cause is not established by this drawing. Do "
            "not present it as a defect finding before the plot list from the "
            "drawing's owner confirms it"
        ),
        "sanity": {
            "targets_tested_vs_parcels_total": joined["total_targets"] - parcels_total,
            "note": (
                "a difference here means some parcel polygons have no closed "
                "ring and could not be tested; the count per cause is in "
                "`join.skipped_targets_total`. Zero means every parcel was "
                "tested"
            ),
        },
    }
    return ev.attach(body, evidence)


register(
    Recipe(
        name="coverage_gap",
        answers="plot numbers without a polygon, and polygons without a plot number",
        when_to_use=(
            "'how many houses are there', 'do the numbers and the plots "
            "match', or any time two counts about the same thing are not the "
            "same. It is also what holds back the most expensive mistake: "
            "answering the number of houses with the number of text entities "
            "on the plot-number layer."
        ),
        params=(
            Param(
                "number_layer",
                "layer",
                "the layer that carries the plot numbers. It has no default: "
                "a layer name is one drawing's trait, and a default inside "
                "the code would be wrong on the next drawing (G1).",
                required=True,
            ),
            Param(
                "parcel_layers",
                "layers",
                "the parcel polygon layers. Left empty, derived from this "
                "drawing's land use config.",
            ),
            Param(
                "parcel_use",
                "text",
                "when `parcel_layers` is left empty, restrict what is derived "
                "to a single land use — e.g. `residential`. Without it, EVERY "
                "layer with role parcel takes part, including schools, "
                "mosques, and parks.",
            ),
        ),
        returns=(
            "numbers_total",
            "numbers_distinct",
            "parcels_total",
            "gap",
            "numbers_without_parcel",
            "parcels_without_number",
            "what_each_number_counts",
        ),
        built_on=("store_spatial.join_labels", "autocad_entities.count_documents"),
        limits={
            "number_entities": MAX_NUMBER_ENTITIES,
            "targets": spatial.MAX_TARGETS,
            "pairs_after_prefilter": spatial.PAIRS_BUDGET,
        },
        run=_run_coverage_gap,
    )
)


# =============================================================================
# 7. outliers
# =============================================================================


_FIELDS: Final[tuple[str, ...]] = ("area", "length")


def _run_outliers(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    layers: Sequence[str] | None,
    field: str,
    z: float | None,
) -> dict[str, Any]:
    measure = (field or "area").strip().lower()
    if measure not in _FIELDS:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"field {field!r} is neither `area` nor `length`.",
            "Both are fields that are really stored per entity. Other axes "
            "are not indexed yet, and this cluster refuses queries without an "
            "indexed plan.",
        )
    names, where = _resolve_layers(drawing_id, layers, role="parcel")
    threshold = 2.0 if z is None else float(z)
    if threshold <= 0:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"z has the value {z!r}; it must be greater than zero.",
            "z is how many standard deviations from the mean. A zero or "
            "negative value would flag the whole population as deviating.",
        )

    raw = stats.values_for(
        drawing_id,
        measure=measure,
        layout_name=layout or "",
        layers=list(names),
    )
    pooled = [v for name in raw["layers_present"] for v in raw["values"].get(name, [])]
    described = stats.describe(
        pooled,
        measure=measure,
        units=units,
        population=f"{len(names)} parcel layers on layout {layout!r}",
        matched_count=raw["total_matched"],
    )

    mean_block = described.get("mean") or {}
    sd_block = described.get("sd") or {}
    mean = mean_block.get("value")
    sd = sd_block.get("value")

    scope_note = (
        f"layout {layout!r}; {len(pooled)} of {raw['total_matched']} objects "
        f"measured on {len(names)} layers from {where}; deviating = |x − "
        f"mean| > {threshold} × standard deviation"
    )

    body: dict[str, Any] = {
        "scope_note": scope_note,
        "field": measure,
        "layers": list(names),
        "layers_source": where,
        "z": threshold,
        "stats": described,
        "population": raw["total_matched"],
        "measured": len(pooled),
        "not_measured": (
            f"{raw['total_matched'] - len(pooled)} objects match this filter "
            f"but carry no {measure}; they are not counted as zero and cannot "
            "deviate. 'Deviating' here is a POSITION within the distribution, "
            "not a defect: a corner parcel that really is larger will show up "
            "here, and that is correct."
        ),
    }

    if mean is None or sd is None or sd == 0:
        body["outliers"] = []
        body["outliers_total"] = 0
        body["band"] = None
        body["why_empty"] = (
            "this distribution has no usable standard deviation: "
            + (
                "the population holds fewer than two values"
                if len(pooled) < 2
                else "every measured value is identical, so nothing deviates"
            )
        )
        return body

    lo = mean - threshold * sd
    hi = mean + threshold * sd
    base = {
        "drawing_id": drawing_id,
        "layout": layout,
        "layer": {"$in": list(names)},
    }
    # TWO queries, not one `$or`. Both are prefixed by `drawing_id` +
    # `layout`, which is exactly the `(drawing_id, layout)` index; a top-level
    # `$or` may be split by the planner into branches without an index, and
    # this cluster runs with `notablescan` — a query without an indexed plan
    # does not slow down, it FAILS, on the request path.
    projection = {"handle": 1, "layer": 1, "type": 1, measure: 1, "_id": 0}
    found: list[dict[str, Any]] = []
    for bound in ({"$lt": lo}, {"$gt": hi}):
        found.extend(
            _entities()
            .find({**base, measure: bound}, projection)
            .sort([("layer", 1), ("handle", 1)])
            .limit(MAX_ROWS_RETURNED + 1)
        )
    found.sort(key=lambda d: (str(d.get("layer")), str(d.get("handle"))))
    rows = [
        {
            "handle": doc.get("handle"),
            "layer": doc.get("layer"),
            "type": doc.get("type"),
            "value": doc.get(measure),
            "unit": units.get("area_unit" if measure == "area" else "length_unit"),
            "z_score": (float(doc.get(measure)) - mean) / sd,
            "direction": "large" if float(doc.get(measure)) > hi else "small",
        }
        for doc in found[:MAX_ROWS_RETURNED]
    ]
    body["band"] = {
        "low": lo,
        "high": hi,
        "unit": units.get("area_unit" if measure == "area" else "length_unit"),
        "unit_reason": (
            None
            if units.get("area_unit" if measure == "area" else "length_unit")
            else "this drawing does not declare units"
        ),
        "method": f"mean ± {threshold} × POPULATION standard deviation (ddof=0)",
    }
    body["outliers"] = rows
    body["outliers_total"] = len(found) if len(found) <= MAX_ROWS_RETURNED else None
    body["outliers_truncated"] = len(found) > MAX_ROWS_RETURNED
    return body


register(
    Recipe(
        name="outliers",
        answers="which parcels deviate from the distribution of their own type",
        when_to_use=(
            "'is there anything odd', 'which plot is the largest/smallest', "
            "or when a mean is about to be quoted and it needs to be known "
            "whether a handful of values are driving it."
        ),
        params=(
            Param(
                "layers",
                "layers",
                "the population that is examined. Left empty, derived from "
                "this drawing's land use config (role=parcel) — and that "
                "mixes schools in with houses, so name the layers if one type "
                "is meant.",
            ),
            Param("field", "text", "`area` or `length`.", default="area"),
            Param(
                "z",
                "number",
                "how many standard deviations from the mean before a value is "
                "called deviating.",
                default=2.0,
            ),
        ),
        returns=("stats", "band", "outliers[].handle", "outliers[].z_score", "not_measured"),
        built_on=("store_stats.values_for", "store_stats.describe"),
        limits={"rows_scanned": stats.MAX_ROWS, "rows_returned": MAX_ROWS_RETURNED},
        carries_meaning=False,
        run=_run_outliers,
    )
)


# =============================================================================
# 8. cross_check_classification
# =============================================================================


def _run_cross_check(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    layers: Sequence[str] | None,
    edge_range: tuple[float, float] | None,
    area_range: tuple[float, float] | None,
) -> dict[str, Any]:
    if edge_range is None or area_range is None:
        raise RecipeRefused(
            "RECIPE_PARAM_REQUIRED",
            "recipe 'cross_check_classification' needs `edge_range` and `area_range`.",
            "Both are traits of ONE drawing — the size of its plot module — "
            "and a default inside the code would make this sweep a solution "
            "specific to that drawing (G1). Run `size_module` first if the "
            "ranges are not known yet; it reads them from the geometry "
            "without a single layer name.",
        )
    names, where = _resolve_layers(drawing_id, layers, role="parcel")
    body = dict(
        geometry.cross_check_layers(
            drawing_id,
            layout=layout or "",
            units=units,
            coded_layers=list(names),
            edge_range=edge_range,
            area_range=area_range,
            limit=MAX_ROWS_RETURNED,
        )
    )
    body["coded_layers_source"] = where
    body["disagreement_is_not_an_error"] = (
        "a polygon that passes the shape test outside the coded layers is not "
        "a classification error by itself: plot shapes do repeat, and a "
        "pocket park the size of a plot is still a park. What this sweep "
        "answers is HOW FAR the two readings agree, not which reading is "
        "right."
    )
    return body


register(
    Recipe(
        name="cross_check_classification",
        answers="whether the layer-name-based classification matches the shape that is drawn",
        when_to_use=(
            "before quoting a plot count as a fact, and any time a "
            "classification stands on layer names alone. It is a second "
            "method that does not read layer names at all."
        ),
        params=(
            Param(
                "layers",
                "layers",
                "the coded layers whose classification is checked. Left "
                "empty, derived from this drawing's land use config "
                "(role=parcel).",
            ),
            # Both are MANDATORY, but not through `required=True`: a refusal
            # born from the parameter engine talks about one parameter, and
            # what is needed here is a single message that names both at once
            # together with the recipe that reads the ranges out of the geometry.
            Param(
                "edge_range",
                "range",
                "the edge lengths considered to pass the plot module test, "
                "`[min, max]` in DRAWING units. MANDATORY — run `size_module` "
                "first if the ranges are not known yet.",
            ),
            Param(
                "area_range",
                "range",
                "the area considered to pass the plot module test, "
                "`[min, max]` in DRAWING units squared. MANDATORY.",
            ),
        ),
        returns=(
            "coded_total",
            "coded_passing_shape_test",
            "coded_failing_shape_test",
            "shape_matches_outside_coded_layers",
            "outside_by_layer",
            "agreement_fraction",
        ),
        built_on=("store_geometry.cross_check_layers",),
        limits={
            "target_polygons": geometry.MAX_TARGET_POLYGONS,
            "rows_returned": MAX_ROWS_RETURNED,
        },
        run=_run_cross_check,
    )
)
