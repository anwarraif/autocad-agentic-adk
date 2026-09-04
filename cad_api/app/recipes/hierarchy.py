"""Road hierarchy: how far apart the junctions sit, and how wide the corridor is.

The RCDC poster lists "road hierarchy ROW/junctions/cul-de-sac" as a Spatial
Analysis element. Three of those four words are already measurable from what
ingest stores, and this recipe measures them. The fourth, a road's CLASS, is
not here and is not guessed: nothing in a masterplan DXF states that a strip
of tarmac is a collector rather than a local street, and a hierarchy invented
from widths would be a classification wearing a measurement's clothes.

What it does measure, and how each figure is grounded:

**Junction spacing.** A path whose two ends both land on junction nodes is one
segment between junctions, and its stored `length` is that spacing. The
distribution of those lengths is what a reviewer means by block length. Paths
that dead-end, or that end where nothing else meets them, are not spacings and
are excluded and counted.

**Cul-de-sacs.** These are the dead ends `junction_census` already brackets,
carried through unchanged rather than recounted. The bracket is kept: a
T-junction whose through-road is a curve the store could not resolve is a
node that MIGHT be a dead end, and collapsing the bracket to one number is the
mistake `junction_census` exists to avoid.

**Corridor width.** Only where the Phase-8 corridor signal actually identifies
a corridor, and never from a layer name. For a long thin ring, twice the area
over the perimeter is its mean width, and both inputs are stored per entity.
The assumption that the ring IS long and thin is stated with the answer along
with the implied length, so a reader can see when it does not hold.

Zero new tools: this rides `run_analysis` like every other recipe. The whole
network preparation, from choosing the road layers to clustering the nodes, is
`junctions.prepare_network` unchanged, so this recipe and `junction_census`
can never disagree about which copy of the geometry they described or at what
tolerance.
"""

from __future__ import annotations

import statistics
from typing import Any, Final, Mapping, Sequence

from ..mongo import COLL_ENTITIES, coll
from . import corridor as corridor_mod
from . import junctions as jn
from .registry import Param, Recipe, RecipeRefused, register

#: The lowest branch degree that makes a node a junction rather than a bend.
#: Two paths meeting end to end is one road drawn in two pieces, not a place
#: where a driver can turn, and counting it would halve every spacing.
JUNCTION_DEGREE: Final[int] = 3

#: Rings sampled when measuring corridor width. The corridor layer on the
#: reference drawing carries a few hundred; this sits above that and exists so
#: a pathological layer refuses instead of running unbounded.
MAX_CORRIDOR_RINGS: Final[int] = 20_000

#: Spacings listed individually. The statistics are always over every spacing;
#: it is only the row list that is capped, and the cap is stated.
MAX_SPACINGS_LISTED: Final[int] = 50


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    """The value at `fraction` through a sorted list, linearly interpolated.

    Written here rather than taken from `statistics.quantiles` because that
    function needs at least two points and raises on one, and a drawing with a
    single measurable spacing should report that one spacing rather than an
    exception.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 3)


def _spacing_stats(values: Sequence[float], unit: str | None) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "unit": unit,
            "why_none": (
                "no path on these layers has a junction at BOTH of its ends, "
                "so this drawing states no junction-to-junction spacing"
            ),
        }
    return {
        "count": len(values),
        "unit": unit,
        "min": round(min(values), 3),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "p75": _quantile(values, 0.75),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "total": round(sum(values), 3),
    }


def _corridor_width(
    drawing_id: str, layout: str | None, length_unit: str | None
) -> dict[str, Any]:
    """Width of the layer the corridor signal ranks first, or why not.

    The signal is `corridor.rank_region_layers`, unchanged: it ranks closed
    ring layers by how much of the network they contain and how few parcels,
    and it never reads a layer name. If it declines to rank, that reason is
    carried through instead of a number.
    """
    ranking = corridor_mod.rank_region_layers(drawing_id, layout)
    if not ranking.get("ranked"):
        return {
            "measured": False,
            "why_not": ranking.get("why_not")
            or "the corridor signal did not rank any layer on this layout",
            "signal": {"ranked": False},
        }

    best = ranking.get("best") or {}
    layer = best.get("layer") if isinstance(best, Mapping) else None
    if not layer:
        return {
            "measured": False,
            "why_not": "the corridor signal ranked no layer first",
            "signal": {"ranked": True},
        }

    rows = list(
        coll(COLL_ENTITIES)
        .find(
            {
                "drawing_id": drawing_id,
                "layout": layout,
                "layer": layer,
                "area": {"$ne": None},
                "perimeter_from_ring": {"$ne": None},
            },
            {"handle": 1, "area": 1, "perimeter_from_ring": 1, "_id": 0},
        )
        .limit(MAX_CORRIDOR_RINGS + 1)
    )
    if len(rows) > MAX_CORRIDOR_RINGS:
        raise RecipeRefused(
            "RECIPE_INPUT_TOO_LARGE",
            f"layer {layer!r} holds more than {MAX_CORRIDOR_RINGS} measurable "
            "rings on this layout.",
            "Name one layout, or narrow the drawing. A width taken from part "
            "of a layer would wear the name of the whole one.",
        )

    widths: list[float] = []
    lengths: list[float] = []
    for row in rows:
        area = row.get("area")
        perimeter = row.get("perimeter_from_ring")
        if not area or not perimeter or perimeter <= 0:
            continue
        # For a long thin strip, area is about width x length and perimeter
        # about twice their sum, so 2A/P is the mean width. The implied length
        # is published beside it precisely so a reader can see the shapes where
        # that stops being true: when width and length are close, it is not a
        # corridor and this number does not mean what it says.
        width = 2.0 * float(area) / float(perimeter)
        widths.append(width)
        if width > 0:
            lengths.append(float(area) / width)

    if not widths:
        return {
            "measured": False,
            "why_not": (
                f"layer {layer!r} ranked first for corridor behaviour but "
                "carries no ring with both an area and a perimeter stored, so "
                "there is nothing to take a width from"
            ),
            "layer": layer,
        }

    return {
        "measured": True,
        "layer": layer,
        "rings_measured": len(widths),
        "unit": length_unit,
        "width": {
            "min": round(min(widths), 3),
            "median": _quantile(widths, 0.5),
            "mean": round(statistics.fmean(widths), 3),
            "max": round(max(widths), 3),
        },
        "implied_length_median": _quantile(lengths, 0.5) if lengths else None,
        "method": (
            "twice the ring area divided by its perimeter, which is the mean "
            "width of a long thin strip. Both inputs are stored per entity by "
            "ingest and neither is re-derived here"
        ),
        "assumption": (
            "that the ring is long and thin. Compare the median width with "
            "the median implied length: where they are close the shape is not "
            "a corridor and this width does not describe it"
        ),
        "chosen_by": (
            "the corridor signal, which ranks closed ring layers by how much "
            "of the road network falls inside them and how few parcels do. It "
            "never reads a layer name, and it does not claim this layer IS a "
            "road"
        ),
        "signal": {
            "ranked": True,
            "network_inside": best.get("network_inside"),
            "parcels_inside": best.get("parcels_inside"),
            # The key is `corridor_score`, not `score`. Asking for the wrong
            # one published a null that read like a failed measurement when
            # the signal had in fact measured it.
            "corridor_score": best.get("corridor_score"),
            "runner_up": best.get("runner_up"),
            "verdict": best.get("verdict"),
        },
    }


def _run_road_hierarchy(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    layers: Sequence[str] | None,
    snap_tolerance: float | None,
    copy_index: float | None,
) -> dict[str, Any]:
    prepared = jn.prepare_network(
        drawing_id=drawing_id,
        layout=layout,
        units=units,
        layers=layers,
        snap_tolerance=snap_tolerance,
        copy_index=copy_index,
    )
    if prepared["stale_answer"] is not None:
        return prepared["stale_answer"]

    described = prepared["described"]
    result = prepared["result"]
    nodes = result["nodes"]
    histogram = {
        "at_least": result["at_least"],
        "at_most": result["at_most"],
    }

    # Which node each path ends at. `node.ending` holds path INDEXES, so this
    # is the same membership the census counted degrees from; nothing is
    # re-clustered and no second tolerance exists to disagree with the first.
    by_index: dict[int, list[Any]] = {}
    for node in nodes:
        for index in node.ending:
            by_index.setdefault(index, []).append(node)

    spacings: list[float] = []
    rows: list[dict[str, Any]] = []
    excluded = {"one_end_only": 0, "no_length_stored": 0, "not_between_junctions": 0}

    for path in described:
        attached = by_index.get(path.index, [])
        if len(attached) < 2:
            excluded["one_end_only"] += 1
            continue
        if not all(node.degree_min >= JUNCTION_DEGREE for node in attached[:2]):
            excluded["not_between_junctions"] += 1
            continue
        if path.length is None:
            excluded["no_length_stored"] += 1
            continue
        spacings.append(float(path.length))
        if len(rows) < MAX_SPACINGS_LISTED:
            rows.append(
                {"handle": path.handle, "layer": path.layer, "length": path.length}
            )

    length_unit = units.get("length_unit")
    dead_ends = {
        "at_least": histogram["at_least"].get("dead_ends", 0),
        "at_most": histogram["at_most"].get("dead_ends", 0),
    }

    return {
        "junction_spacing": {
            **_spacing_stats(spacings, length_unit),
            "measured_between": (
                f"paths whose BOTH ends sit on a node of branch degree "
                f"{JUNCTION_DEGREE} or more"
            ),
            "basis": (
                "each spacing is the path's own stored `length`, which follows "
                "the road as drawn rather than the straight line between its "
                "ends"
            ),
            "tolerance": {
                "snap_tolerance": prepared["tolerance"],
                "unit": length_unit,
                "note": (
                    "two path ends within this distance are one junction. It "
                    "is the value junction_census derived for this drawing, "
                    "not a second one chosen here, and it is what decides "
                    "whether a node exists at all"
                ),
            },
            "excluded": excluded,
            "rows": rows,
            "rows_capped_at": MAX_SPACINGS_LISTED,
        },
        "cul_de_sacs": {
            **dead_ends,
            "exact": (
                dead_ends["at_least"]
                if dead_ends["at_least"] == dead_ends["at_most"]
                else None
            ),
            "basis": (
                "the dead ends junction_census brackets, carried through "
                "unchanged. A bracket rather than one number because a node "
                "whose through-road is a curve the store could not resolve "
                "MIGHT be a dead end and might not"
            ),
        },
        "corridor": _corridor_width(drawing_id, layout, length_unit),
        "network": {
            "layers": list(prepared["names"]),
            "layers_source": prepared["layers_source"],
            "paths_in_this_copy": len(described),
            "nodes_total": len(nodes),
            "copies": len(prepared["summaries"]),
            "copy_described": prepared["chosen"],
            "copy_chosen_by": prepared["choice_basis"],
        },
        "not_measured": (
            "the CLASS of any road. Nothing in this drawing states that a "
            "strip is a collector rather than a local street, and a hierarchy "
            "inferred from width alone would be a classification wearing a "
            "measurement's clothes. Right of way is not measured either: the "
            "corridor width above is the width of a ring the corridor signal "
            "ranked first, which is the paved corridor as drawn and not a "
            "legal reservation"
        ),
        "scope_note": (
            f"layout {layout!r}; network layers chosen by "
            f"{prepared['layers_source']}; {len(described)} paths in the copy "
            f"described; lengths in {length_unit or 'no unit the drawing states'}"
        ),
    }


register(
    Recipe(
        name="road_hierarchy",
        answers=(
            "how far apart this drawing puts its junctions, how many "
            "cul-de-sacs it has, and how wide its corridor is where a "
            "corridor can be identified"
        ),
        when_to_use=(
            "'what is the block length here', 'how many cul-de-sacs', 'how "
            "wide are the roads'. It measures spacing along the paths as "
            "drawn and takes cul-de-sacs from junction_census unchanged. It "
            "does NOT classify roads: nothing in the drawing states a road "
            "class, and this recipe refuses to invent one."
        ),
        params=(
            Param(
                "layers",
                "layers",
                "the layers whose paths are measured. Left empty they are "
                "derived exactly as junction_census derives them, from the "
                "land use config and the Dossier geometric role, never from a "
                "layer name.",
            ),
            Param(
                "snap_tolerance",
                "number",
                "the distance within which two path ends count as the same "
                "junction, in DRAWING units. Left empty it is derived from "
                "this drawing by the same rule junction_census uses, and the "
                "value taken is published with the answer.",
                default=None,
            ),
            Param(
                "copy_index",
                "number",
                "which detached copy of the geometry to describe, when the "
                "layers hold the same thing drawn more than once. Left empty "
                "the live copy is identified by measurement and named.",
                default=None,
            ),
        ),
        returns=(
            "junction_spacing.count",
            "junction_spacing.median",
            "junction_spacing.p25",
            "junction_spacing.p75",
            "junction_spacing.excluded",
            "junction_spacing.tolerance",
            "cul_de_sacs.at_least",
            "cul_de_sacs.at_most",
            "corridor.measured",
            "corridor.width.median",
            "corridor.assumption",
            "network.copy_described",
            "not_measured",
        ),
        built_on=(
            "junctions.prepare_network",
            "junctions._run_census",
            "corridor.rank_region_layers",
            "recipes.topology._road_layers",
        ),
        limits={
            "path_entities": jn.MAX_PATH_ENTITIES,
            "endpoint_comparisons": jn.MAX_POINT_COMPARISONS,
            "corridor_rings": MAX_CORRIDOR_RINGS,
            "spacings_listed": MAX_SPACINGS_LISTED,
        },
        # It MEASURES and does not classify. `carries_meaning=False` is the
        # honest half of the UPLIFT-08 rule: this recipe says how far apart
        # the junctions are, never what kind of road runs between them, so it
        # owes a `not_measured` rather than an `evidence` block. Setting it
        # True and attaching a land use evidence block would dress a spacing
        # in a classification it never made.
        carries_meaning=False,
        run=_run_road_hierarchy,
    )
)
