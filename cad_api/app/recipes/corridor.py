"""The corridor signal: which region layer behaves like a road corridor.

Owner: the CORRIDOR-SIGNAL lane — DOSSIER Phase 8a.

This module is a **scoring helper**, not a recipe. It registers nothing, it is
not in the catalogue, and the only thing it publishes is a ranked list of
candidate layers with the two measurements that ranked them.

## Why it exists — five attempts that were measured, and what they proved

`frontage_check` run with its default parameters on the reference drawing
establishes NOTHING: every road layer it finds by geometric role is an open
path, open paths resolve only to bounding boxes, and a bounding box proves
neither frontage nor its absence. The response therefore offers candidate
layers of CLOSED rings to retry with — and until this module existed it ranked
them **by area**.

That ranking has a specific, measured failure. On the reference drawing the
largest closed-ring layer is the SITE BOUNDARY (6,837,897 m²). Every parcel
lies inside the site boundary, so a retry against it reports universal
frontage — a wrong answer wearing a confident label, which is worse than the
honest refusal it replaced. The layer that really carries the right of way on
that drawing is only the SECOND largest, and its name contains no road word at
all. So neither size nor name finds it.

## The signal, and it is geometric

> A road corridor **contains the road network and not the parcels.**
> A site boundary contains both. A parcel layer contains neither.

Two fractions per candidate layer, both measured from points this store already
holds:

* `network_inside` — the share of sampled `network`-role geometry whose
  representative point falls inside one of this layer's rings;
* `parcels_inside` — the same share for the parcel geometry.

and one score built from them:

    corridor_score = network_inside x (1 - parcels_inside)

A product rather than a difference, and that choice was measured too. With a
DIFFERENCE, a residential plot layer on the reference drawing (`network` 0.025,
`parcels` 0.000, difference +0.025) outranks the real corridor (`network` 0.442,
`parcels` 0.742, difference -0.300), because a difference rewards holding
nothing at all. The product does not: a layer that holds none of the network
scores zero however few parcels it holds, and a layer that holds every parcel
scores zero however much network it holds. Both zeros are the right answer.

## What this never does

* **It never reads a layer name** (G1). Not one name, token, prefix or
  typology code appears in this file. The corridor layer on the reference
  drawing carries no road word, so a name search finds nothing — which is
  exactly the dead end this replaces.
* **It never ranks by size.** Area travels on every row because a reader wants
  it, and it takes no part in the ordering.
* **It never claims a layer IS a road.** The strongest thing said here is
  "behaves like a corridor in this layout": a comparison between the candidates
  on ONE drawing, from geometry, with both fractions published so the reader
  can disagree. Meaning is still a human's call.

## It is a SAMPLE, and it says so

Testing every network entity against every ring of every candidate layer is
work this cannot afford inside a request. So a fixed number of points is taken
from each population, at evenly spaced positions in a deterministically sorted
list rather than as the first k rows — the first k sit wherever the drafter
happened to start, and would describe one corner of the site and call it the
layer. The sample size, the stride, and the population it came from travel with
every fraction, exactly as `dossier_backfill`'s containment sample does.

## Riding on, not rewriting

`store_spatial._SpatialIndex` is the prefilter, `store_spatial._ring_of` and
`_origin_of` read the stored ring in the frame it was stored in, and
`region.point_in_ring` is the point test — the same one `join_labels` uses, so
a point on a ring's edge is decided here exactly as it is decided there. A
second point-in-polygon in this repo would answer differently on the same ring
with nothing in either response to say which one ran.

## Honest degradation

No Dossier, no closed-ring layer, no `network`-role geometry, no parcel
geometry, or a population above a stated cap: in every one of those cases this
returns the candidates it CAN name with `corridor_score: null` and a reason, or
no candidates and a reason. It never falls back to ranking by size, because a
wrong ranking under a confident label is the defect this module was written to
remove.
"""

from __future__ import annotations

from typing import Any, Final, Mapping, Sequence

from .. import landuse
from .. import region
from .. import store_spatial as spatial
from ..mongo import COLL_ENTITIES, coll

Point = tuple[float, float]
Box = tuple[float, float, float, float]


# --- stated limits (G7) ------------------------------------------------------

#: Points taken from EACH population — the network sample and the parcel
#: sample. Measured 26 August 2026 on the reference drawing: 240 points against
#: 2,881 rings costs 2,760 point-in-ring tests and about 0.4 s, and the layer
#: ranked first is the same at 60, 120, 240, 480, 1,000 and 2,000 points. It is
#: NOT the same at 30: below roughly fifty points the sample is too coarse to
#: separate the top two, and that is a property of sampling rather than of this
#: score. It is a sample size chosen for cost, published with every fraction it
#: produced, and never presented as a census.
DEFAULT_SAMPLE_POINTS: Final[int] = 240

#: The largest sample a caller may ask for. A sample the size of the population
#: is a census wearing a sample's clothes and costs what a census costs.
MAX_SAMPLE_POINTS: Final[int] = 2_000

#: Closed rings that may enter one ranking. Above it the ranking is WITHHELD
#: with a reason rather than computed from a subset chosen by some tie-break
#: nobody asked for. The reference drawing brings 2,881, so this sits an order
#: of magnitude above the largest drawing in the store.
MAX_CANDIDATE_RINGS: Final[int] = 20_000

#: Candidates named in the answer. Five is what the caller before this module
#: returned, and the number is kept so that this is a change of ORDER and of
#: evidence, not a change of size.
MAX_CANDIDATES_RETURNED: Final[int] = 5

#: `dossier_read.profiles_for` stops at `MAX_PROFILE_LAYERS` names per call and
#: says so; layers are asked for in windows of that size so that a drawing with
#: eighty layers is profiled completely rather than silently to forty.
_PROFILE_FALLBACK_CHUNK: Final[int] = 40

#: The stored points that can represent an entity, best first. A closed ring
#: has a true centroid; everything else has the centre of its bounding box,
#: and a text carries its insertion point. Which one was used is COUNTED per
#: population and published, because a bounding-box centre is not on the
#: geometry it stands for: for a straight `LINE` it is the midpoint and lies on
#: the road, for an `ARC` or a bent polyline it does not.
POINT_FIELDS: Final[tuple[str, ...]] = (
    "polygon_centroid",
    "bbox_centre",
    "anchor_point",
)

#: A point exactly on a ring's edge counts as INSIDE. Chosen to agree with
#: `store_spatial._join`, which counts a boundary hit as a match: a road
#: centreline snapped to the edge of the corridor that carries it is normal
#: draughting, not an edge case, and dropping those points would under-state
#: exactly the layer this signal is looking for.
BOUNDARY_COUNTS_AS_INSIDE: Final[bool] = True


def _entities():
    """This module's only door to MongoDB, and it goes through `coll()`."""
    return coll(COLL_ENTITIES)


# =============================================================================
# Pure arithmetic
# =============================================================================


def corridor_score(
    network_inside: float | None, parcels_inside: float | None
) -> float | None:
    """`network_inside x (1 - parcels_inside)`, or None when either is absent.

    Absent is not zero (G8). A layer whose parcel fraction could not be
    measured — because this drawing carries no parcel geometry at all — has an
    UNKNOWN score, and an unknown score may not be ordered against a known one.

    The two zeros of this product are the two right answers:

    * `network_inside == 0` — the layer holds none of the road network. It is
      not a corridor whatever else is true of it.
    * `parcels_inside == 1` — the layer holds every parcel that was sampled.
      That is how a SITE BOUNDARY behaves, and it is the trap this whole
      module exists to avoid falling into.
    """
    if network_inside is None or parcels_inside is None:
        return None
    return float(network_inside) * (1.0 - float(parcels_inside))


def _fraction(inside: int, sampled: int) -> float | None:
    """`inside / sampled`, or None when nothing was sampled.

    Zero out of zero is not zero (G8): a population with no point in it was
    never asked the question, and a fraction of 0.0 would answer it.
    """
    if sampled <= 0:
        return None
    return inside / sampled


def _point_of(doc: Mapping[str, Any]) -> tuple[Point | None, str | None]:
    """The stored point that represents this entity, and WHICH field it was."""
    for field in POINT_FIELDS:
        value = doc.get(field)
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) >= 2
            and value[0] is not None
            and value[1] is not None
        ):
            try:
                return (float(value[0]), float(value[1])), field
            except (TypeError, ValueError):
                continue
    return None, None


def _spread(rows: Sequence[Mapping[str, Any]], size: int) -> tuple[list[Any], float]:
    """`size` rows taken at evenly spaced positions, so the sample SPANS.

    The first `size` rows sit wherever the drafter happened to start drawing,
    and five of them describe one corner of a site and call it the layer. The
    stride is returned so that it can be published with the fraction: a sample
    that does not admit how it was taken is a confident half-truth.

    The positions are computed rather than sliced, and that is not a detail.
    A slice `rows[::len(rows) // size]` collapses to `rows[::1]` as soon as the
    sample is more than half the population, which then truncates to the FIRST
    `size` rows — the exact failure this function exists to avoid, arriving
    silently at the sample sizes a caller would choose to be more careful.
    Measured on the reference drawing: at 1,000 points of a 1,574-row network
    population the sliced version sampled two thirds of one layer and read the
    winning fraction as 0.177, where an evenly spaced sample of the same size
    reads 0.445.
    """
    if size <= 0 or not rows:
        return [], 1.0
    if len(rows) <= size:
        return list(rows), 1.0
    stride = len(rows) / size
    return [rows[int(i * stride)] for i in range(size)], stride


# =============================================================================
# Reading what the store already holds
# =============================================================================


def _roles_by_layer(
    drawing_id: str, layout: str | None, document: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """`{layer: profile}` for every layer in this layout, through `dossier_read`.

    The role is read through `dossier_read.profiles_for` rather than off the raw
    Dossier document, so the storage shape stays that module's business — and
    the document this call already holds is HANDED to it, because reading the
    reference drawing's Dossier costs 0.3 s warm and this module has no business
    paying that twice.
    """
    try:  # pragma: no cover - the reader is optional everywhere else too
        from .. import dossier_read
    except Exception as exc:  # noqa: BLE001 -- guarded on purpose
        return {}, (
            f"dossier_read is not importable ({type(exc).__name__}), so no "
            "layer's geometric role could be read"
        )

    blocks = document.get("layers")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        return {}, "this Dossier records no layer buckets"
    names = sorted(
        {
            str(b.get("layer"))
            for b in blocks
            if isinstance(b, Mapping)
            and b.get("layer")
            and (layout is None or str(b.get("layout")) == str(layout))
        }
    )
    if not names:
        return {}, f"this Dossier records no layer bucket on layout {layout!r}"

    chunk = max(1, int(getattr(dossier_read, "MAX_PROFILE_LAYERS", _PROFILE_FALLBACK_CHUNK)))
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(names), chunk):
        window = names[start : start + chunk]
        try:
            profiles = dossier_read.profiles_for(
                drawing_id, window, layout=layout, dossier=document
            )
        except Exception as exc:  # noqa: BLE001
            return out, (
                f"the Dossier was read but profiling stopped after {len(out)} "
                f"layers ({type(exc).__name__})"
            )
        for name, profile in profiles.items():
            if isinstance(profile, Mapping) and profile.get("profiled"):
                out[str(name)] = dict(profile)
    return out, None


def _config_layers(drawing_id: str, role: str) -> tuple[tuple[str, ...], str | None]:
    """Layers this drawing's land use config gives `role`, or a reason it has none.

    G3: a drawing with no config is answered, not refused and not guessed at.
    """
    try:
        config = landuse.for_drawing(drawing_id)
    except Exception as exc:  # noqa: BLE001 -- a broken config must not be an outage
        return (), (
            f"this drawing's land use config could not be read "
            f"({type(exc).__name__})"
        )
    if config is None:
        return (), "this drawing has no land use config yet"
    names = tuple(
        sorted(name for name, entry in config.layers.items() if entry.role == role)
    )
    if not names:
        return (), f"this drawing's land use config names no layer with role {role!r}"
    return names, None


def _points_for(
    drawing_id: str,
    layout: str | None,
    layers: Sequence[str],
    *,
    rings_only: bool,
    sample_size: int,
) -> dict[str, Any]:
    """A spread sample of representative points from `layers`, with its census.

    `rings_only` narrows to entities whose ring closed cleanly. It is True for
    the parcel population — a parcel is a closed polygon and its centroid is
    the point that stands for it — and False for the network population, where
    an open path is the normal case and a bounding-box centre is all this store
    keeps.
    """
    if not layers:
        return {
            "points": [],
            "sampled": 0,
            "population": 0,
            "step": 1,
            "without_a_point": 0,
            "point_fields_used": {},
        }
    query: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "layer": {"$in": list(layers)},
    }
    if rings_only:
        query["ring_status"] = "complete"
    rows = list(
        _entities()
        .find(
            query,
            {
                "handle": 1,
                "layer": 1,
                "type": 1,
                "polygon_centroid": 1,
                "bbox_centre": 1,
                "anchor_point": 1,
                "_id": 0,
            },
        )
        # Sorted before it is sampled, so the same drawing gives the same
        # sample twice. An unordered sample would make this signal answer
        # differently on two identical calls.
        .sort([("layer", 1), ("handle", 1)])
    )
    sampled, step = _spread(rows, sample_size)

    points: list[tuple[Point, str]] = []
    fields: dict[str, int] = {}
    missing = 0
    for doc in sampled:
        point, field = _point_of(doc)
        if point is None or field is None:
            missing += 1
            continue
        points.append((point, str(doc.get("layer") or "")))
        fields[field] = fields.get(field, 0) + 1
    return {
        "points": points,
        "sampled": len(points),
        "population": len(rows),
        "step": round(step, 4),
        "without_a_point": missing,
        "point_fields_used": dict(sorted(fields.items())),
    }


def _candidate_rings(
    drawing_id: str, layout: str | None, layers: Sequence[str]
) -> tuple[list[dict[str, Any]], list[list[Point]], list[Box | None]]:
    """Every closed ring on the candidate layers, with its box.

    The ring is read through `store_spatial._ring_of` and its translation point
    through `_origin_of`, so that a point is tested against the ring in the same
    frame the ring's own centroid was computed in.
    """
    if not layers:
        return [], [], []
    rows = list(
        _entities()
        .find(
            {
                "drawing_id": drawing_id,
                "layout": layout,
                "layer": {"$in": list(layers)},
                "ring_status": "complete",
            },
            {
                "handle": 1,
                "layer": 1,
                "ring": 1,
                "ring_origin": 1,
                "_id": 0,
            },
        )
        .sort([("layer", 1), ("handle", 1)])
    )
    kept: list[dict[str, Any]] = []
    rings: list[list[Point]] = []
    boxes: list[Box | None] = []
    for doc in rows:
        ring = spatial._ring_of(doc)
        if len(ring) < 3:
            continue
        kept.append(doc)
        rings.append(ring)
        boxes.append(spatial._box_of(ring))
    return kept, rings, boxes


def _containment(
    points: Sequence[tuple[Point, str]],
    rows: Sequence[Mapping[str, Any]],
    rings: Sequence[Sequence[Point]],
    index: Any,
) -> tuple[list[tuple[str, frozenset[str]]], int, int]:
    """Which candidate LAYERS each of these points falls inside.

    Returns `(hits, ring_tests, points_inside_nothing)` where `hits` is one
    `(owning_layer, layers_containing_it)` pair per point. Kept per point
    rather than reduced to counts here, because a candidate must be able to
    exclude the points that came off ITS OWN layer from its own fraction, and a
    count that has already been summed cannot be un-summed.

    `points_inside_nothing` is published rather than dropped: on the reference
    drawing a third of the road network sits outside EVERY closed-ring layer,
    because that network is drawn three times and two of the copies are parked
    off the site. That fact is why no candidate's `network_inside` can reach
    1.0 there, and a reader who cannot see it would read 0.44 as a weak signal
    rather than as a nearly complete one over the geometry that is on the site.
    """
    hits: list[tuple[str, frozenset[str]]] = []
    tests = 0
    outside_everything = 0
    for (x, y), owner in points:
        seen: set[str] = set()
        for i in index.hits(x, y):
            tests += 1
            where = region.point_in_ring(
                x,
                y,
                rings[i],
                origin=spatial._origin_of(dict(rows[i]), rings[i]),
                eps_rel=spatial.BOUNDARY_EPS_REL,
            )
            if where == "outside":
                continue
            if where == "boundary" and not BOUNDARY_COUNTS_AS_INSIDE:
                continue
            seen.add(str(rows[i].get("layer") or ""))
        if not seen:
            outside_everything += 1
        hits.append((owner, frozenset(seen)))
    return hits, tests, outside_everything


# =============================================================================
# The ranking
# =============================================================================


def _withheld(
    reason: str,
    *,
    candidates: list[dict[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A ranking that did not happen, said out loud.

    The candidates it CAN name are still named — a layer of closed rings is a
    fact even when the signal could not be computed — in layer-name order, with
    `corridor_score: null` on every row. What never happens is a fallback to
    ranking by size: that ranking is the defect, and an unordered honest list
    beats a confidently ordered wrong one.
    """
    block: dict[str, Any] = {
        "ranked": False,
        "ranked_by": (
            "nothing — the corridor signal could not be computed, so these "
            "candidates are listed in layer-name order and NOT ranked"
        ),
        "why_not": reason,
        "best": None,
        "candidates": candidates or [],
        "never_ranked_by_size": (
            "the previous ranking was by total area, and on the reference "
            "drawing that put the SITE BOUNDARY first — every parcel is inside "
            "it, so a retry against it reports universal frontage. Size is not "
            "a fallback here; an unranked list is"
        ),
    }
    if extra:
        block.update(extra)
    return block


_ORDINAL_SUFFIX: Final[dict[int, str]] = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(rank: int) -> str:
    if 10 <= rank % 100 <= 20:
        return f"{rank}th"
    return f"{rank}{_ORDINAL_SUFFIX.get(rank % 10, 'th')}"


def _why(
    *,
    layer: str,
    rank: int,
    score: float | None,
    network_inside: float | None,
    parcels_inside: float | None,
    network_in: int,
    network_n: int,
    parcels_in: int,
    parcels_n: int,
    best_layer: str | None,
) -> str:
    """One sentence saying why this layer ranked where it did.

    The verdict alone is not publishable. A reader must be able to see the two
    measurements that produced the ordering and disagree with the conclusion
    while agreeing with the numbers.
    """
    head = (
        f"{layer!r} contains {network_in} of the {network_n} sampled "
        f"network points"
        + (f" ({network_inside:.1%})" if network_inside is not None else "")
        + f" and {parcels_in} of the {parcels_n} sampled parcel points"
        + (f" ({parcels_inside:.1%})" if parcels_inside is not None else "")
        + ". "
    )
    if score is None:
        return head + (
            "One of the two fractions could not be measured, so no score was "
            "computed for it — absent is not zero."
        )
    if network_inside == 0.0:
        return head + (
            "It holds none of the sampled road network, so nothing here "
            "suggests a corridor: score 0."
        )
    if parcels_inside is not None and parcels_inside >= 1.0:
        return head + (
            "It holds EVERY sampled parcel as well, which is how a site "
            "boundary behaves and not how a corridor behaves: score 0. This is "
            "the layer a ranking by size would have proposed."
        )
    body = (
        f"A corridor holds the network and not the parcels, so its score is "
        f"{network_inside:.3f} x (1 - {parcels_inside:.3f}) = {score:.4f}, "
        f"which ranks it {_ordinal(rank)}. "
    )
    if rank == 1:
        return head + body + (
            "That is the strongest corridor signal in this layout. It says the "
            "layer BEHAVES like a corridor here; it does not say the layer is "
            "a road."
        )
    return head + body + (
        f"A weaker corridor signal than {best_layer!r}."
        if best_layer
        else "A weaker corridor signal than the layer ranked first."
    )


def rank_region_layers(
    drawing_id: str,
    layout: str | None,
    *,
    exclude: Sequence[str] | None = None,
    sample_size: int = DEFAULT_SAMPLE_POINTS,
    limit: int = MAX_CANDIDATES_RETURNED,
) -> dict[str, Any]:
    """Rank this layout's closed-ring layers by how much they behave like a corridor.

    Never raises. Every failure — no Dossier, no closed-ring layer, no network
    geometry, no parcel geometry, a population above a stated cap — comes back
    as a block whose `ranked` is False and whose `why_not` says which one it
    was. A caller that cannot see the reason would have to invent one.

    Args:
        drawing_id: the drawing, as always a content hash.
        layout: one layout. `None` means every layout, which is what the
            Dossier reader means by it too.
        exclude: layers that must not be OFFERED as candidates — typically the
            ones the caller has already tested. They stay in the parcel
            population: a corridor is recognised by not containing the parcels,
            so removing the parcels would remove the evidence.
        sample_size: points taken from each population. Capped at
            `MAX_SAMPLE_POINTS` and published with every fraction.
        limit: candidates named in the answer.

    Returns:
        A block carrying `ranked`, `ranked_by`, `best`, `candidates` (each with
        its two fractions, the sample they came from, its score and one
        sentence saying why it ranked where it did), `sample`, `caps`, and the
        two standing statements: that no decision here comes from a layer name,
        and that nothing here claims a layer IS a road.
    """
    sample_size = max(1, min(int(sample_size), MAX_SAMPLE_POINTS))
    limit = max(1, int(limit))
    skip = {str(name).casefold() for name in (exclude or ())}

    caps = {
        "sample_points_per_population": sample_size,
        "sample_points_ceiling": MAX_SAMPLE_POINTS,
        "candidate_rings": MAX_CANDIDATE_RINGS,
        "candidates_returned": limit,
    }
    standing = {
        "score_definition": (
            "corridor_score = network_inside x (1 - parcels_inside), where each "
            "fraction is the share of a SAMPLE of that population whose stored "
            "representative point falls inside one of this layer's closed "
            "rings. Both fractions are dimensionless, so this behaves "
            "identically in metres, in inches, and in a drawing that declares "
            "no unit at all (G2)"
        ),
        "why_a_product_and_not_a_difference": (
            "a difference rewards a layer for holding nothing: measured on the "
            "reference drawing, a residential plot layer (network 0.025, "
            "parcels 0.000) outranks the real corridor (network 0.442, parcels "
            "0.742) on a difference. The product sends both wrong answers to "
            "zero — a layer holding none of the network, and a layer holding "
            "every parcel"
        ),
        "never_from_a_name": (
            "no decision in this ranking reads a layer NAME. The corridor layer "
            "on the reference drawing carries no road word at all, which is why "
            "a name search finds nothing; and template layers named after roads "
            "are sheet layout, which is why a name search finds the wrong thing"
        ),
        "never_ranked_by_size": (
            "area is published on every row because a reader wants it, and it "
            "takes NO part in the ordering. Ranked by size, the first candidate "
            "on the reference drawing is the site boundary, every parcel is "
            "inside it, and the retry reports universal frontage"
        ),
        "what_this_does_not_establish": (
            "that any of these layers IS a road corridor. This measures where "
            "geometry sits relative to geometry. A layer can behave like a "
            "corridor and be a landscape strip, a survey envelope, or a "
            "drafting artefact; naming it is a human's call"
        ),
        "caps": caps,
    }

    # --- the Dossier ---------------------------------------------------------
    try:  # pragma: no cover - the reader is optional everywhere else too
        from .. import dossier_read
    except Exception as exc:  # noqa: BLE001
        return _withheld(
            f"dossier_read is not importable ({type(exc).__name__})",
            extra=standing,
        )
    try:
        document = dossier_read.dossier_for(drawing_id)
    except Exception as exc:  # noqa: BLE001
        return _withheld(
            f"the Dossier store could not be read ({type(exc).__name__})",
            extra=standing,
        )
    if not document:
        return _withheld(
            "no Dossier has been built for this drawing, so no layer has a "
            "geometric role yet. That is 'not computed', not 'there are no "
            "corridors'",
            extra=standing,
        )

    profiles, role_note = _roles_by_layer(drawing_id, layout, document)
    if not profiles:
        return _withheld(
            role_note or "no layer in this layout could be profiled",
            extra=standing,
        )

    region_layers = sorted(
        name for name, p in profiles.items() if p.get("role") == "region"
    )
    candidate_layers = [
        name for name in region_layers if name.casefold() not in skip
    ]
    if not candidate_layers:
        return _withheld(
            (
                f"no layer on layout {layout!r} profiles as `region` (closed "
                "rings) outside the ones already tested"
            ),
            extra=standing,
        )

    def _row(name: str) -> dict[str, Any]:
        """The shape every candidate has, scored or not."""
        profile = profiles.get(name) or {}
        measure = profile.get("measure")
        measure = measure if isinstance(measure, Mapping) else {}
        return {
            "layer": name,
            "entities": profile.get("entities"),
            "area": measure.get("value"),
            "unit": measure.get("unit"),
            "corridor_score": None,
            "rank": None,
            "network_inside": None,
            "parcels_inside": None,
            "why": None,
        }

    unranked = [_row(name) for name in candidate_layers[:limit]]

    # --- the two populations -------------------------------------------------
    network_layers = sorted(
        name for name, p in profiles.items() if p.get("role") == "network"
    )
    config_network, config_network_note = _config_layers(
        drawing_id, landuse.NETWORK_ROLE
    )
    network_sources: list[str] = []
    if config_network:
        network_layers = sorted(set(network_layers) | set(config_network))
        network_sources.append(
            f"this drawing's land use config (role={landuse.NETWORK_ROLE})"
        )
    if any(p.get("role") == "network" for p in profiles.values()):
        network_sources.append(
            f"the Dossier's geometric role `network` on layout {layout!r}"
        )
    if not network_layers:
        return _withheld(
            (
                "no layer on this layout carries the `network` role, from this "
                "drawing's land use config or from the Dossier's geometric "
                "role, so there is no road network to ask the question about. "
                + (config_network_note or "")
            ).strip(),
            candidates=unranked,
            extra=standing,
        )

    # The parcel population takes the config's word where there is one, because
    # that is the same notion of "parcel" `frontage_check` itself resolves; and
    # falls back to the GEOMETRIC region role where there is not, so that a
    # drawing nobody has classified yet is still answered (G3, G4).
    config_parcels, config_parcel_note = _config_layers(
        drawing_id, landuse.PARCEL_ROLE
    )
    if config_parcels:
        parcel_layers = list(config_parcels)
        parcel_source = (
            f"this drawing's land use config (role={landuse.PARCEL_ROLE})"
        )
    else:
        parcel_layers = list(region_layers)
        parcel_source = (
            f"the Dossier's geometric role `region` on layout {layout!r}, "
            "because " + (config_parcel_note or "no config named the parcels")
        )

    # --- the geometry --------------------------------------------------------
    try:
        rows, rings, boxes = _candidate_rings(drawing_id, layout, candidate_layers)
        network = _points_for(
            drawing_id,
            layout,
            network_layers,
            rings_only=False,
            sample_size=sample_size,
        )
        parcels = _points_for(
            drawing_id,
            layout,
            parcel_layers,
            rings_only=True,
            sample_size=sample_size,
        )
    except Exception as exc:  # noqa: BLE001 -- a store failure is a reason, not a 500
        return _withheld(
            f"the geometry could not be read from the store ({type(exc).__name__})",
            candidates=unranked,
            extra=standing,
        )

    if not rows:
        return _withheld(
            (
                f"the {len(candidate_layers)} candidate layer(s) on layout "
                f"{layout!r} carry no ring that closed cleanly, so there is "
                "nothing for a point to be inside of"
            ),
            candidates=unranked,
            extra=standing,
        )
    if len(rows) > MAX_CANDIDATE_RINGS:
        return _withheld(
            (
                f"{len(rows)} closed rings are above the stated ceiling of "
                f"{MAX_CANDIDATE_RINGS} for one ranking. The ranking is "
                "withheld rather than computed over a subset chosen by a "
                "tie-break nobody asked for"
            ),
            candidates=unranked,
            extra=standing,
        )
    if not network["points"]:
        return _withheld(
            (
                f"the {len(network_layers)} network layer(s) on layout "
                f"{layout!r} carry no entity with a stored point, so the share "
                "of the network inside a candidate could not be measured"
            ),
            candidates=unranked,
            extra=standing,
        )
    if not parcels["points"]:
        return _withheld(
            (
                f"the {len(parcel_layers)} parcel layer(s) on layout {layout!r} "
                "carry no closed ring with a stored point, so the share of the "
                "parcels inside a candidate could not be measured. Without it "
                "a site boundary and a corridor are indistinguishable — both "
                "contain the network"
            ),
            candidates=unranked,
            extra=standing,
        )

    index = spatial._SpatialIndex(boxes)
    net_hits, net_tests, net_outside = _containment(
        network["points"], rows, rings, index
    )
    par_hits, par_tests, par_outside = _containment(
        parcels["points"], rows, rings, index
    )

    # A ring contains its own centroid, and that is arithmetic rather than
    # evidence. Parcel points drawn from the candidate layer itself are
    # therefore left out of THAT candidate's parcel fraction, on both sides of
    # it, and the denominator each candidate really used is published on its
    # row.
    scored: list[dict[str, Any]] = []
    for name in candidate_layers:
        network_n = len(net_hits)
        network_in = sum(1 for _owner, seen in net_hits if name in seen)
        own = [(owner, seen) for owner, seen in par_hits if owner != name]
        parcels_n = len(own)
        parcels_in = sum(1 for _owner, seen in own if name in seen)
        network_fraction = _fraction(network_in, network_n)
        parcels_fraction = _fraction(parcels_in, parcels_n)
        score = corridor_score(network_fraction, parcels_fraction)
        row = _row(name)
        row.update(
            {
                "corridor_score": None if score is None else round(score, 6),
                "network_inside": {
                    "fraction": (
                        None if network_fraction is None else round(network_fraction, 6)
                    ),
                    "points_inside": network_in,
                    "points_sampled": network_n,
                    "population": network["population"],
                    "sample_step": network["step"],
                    "basis": (
                        "the share of a sample of the network-role geometry "
                        "whose stored representative point falls inside one of "
                        "this layer's closed rings. A fraction of the SAMPLE, "
                        "not of the layer, and dimensionless"
                    ),
                },
                "parcels_inside": {
                    "fraction": (
                        None if parcels_fraction is None else round(parcels_fraction, 6)
                    ),
                    "points_inside": parcels_in,
                    "points_sampled": parcels_n,
                    "population": parcels["population"],
                    "sample_step": parcels["step"],
                    "own_points_excluded": len(par_hits) - parcels_n,
                    "basis": (
                        "the same measurement over the parcel geometry, with "
                        "this layer's OWN points left out: a ring contains its "
                        "own centroid, and that is arithmetic rather than "
                        "evidence"
                    ),
                },
            }
        )
        scored.append(row)

    # Ranked by score, and ties broken by layer NAME so that two identical
    # scores come back in the same order on every call. The tie-break is
    # alphabetical precisely because it must carry no meaning: a tie-break on
    # area would put size back into the ordering by the back door. A candidate
    # whose score could not be MEASURED sorts last on its own key rather than
    # being treated as a zero — absent is not zero (G8), and a zero is a
    # finding while an absence is a gap.
    scored.sort(
        key=lambda r: (
            r["corridor_score"] is None,
            -(r["corridor_score"] or 0.0),
            str(r["layer"]),
        )
    )
    measurable = [r for r in scored if r["corridor_score"] is not None]
    positive = [r for r in measurable if r["corridor_score"] > 0]
    best = positive[0] if positive else None
    best_layer = str(best["layer"]) if best else None
    for position, row in enumerate(scored, start=1):
        row["rank"] = position
        row["why"] = _why(
            layer=str(row["layer"]),
            rank=position,
            score=row["corridor_score"],
            network_inside=row["network_inside"]["fraction"],
            parcels_inside=row["parcels_inside"]["fraction"],
            network_in=row["network_inside"]["points_inside"],
            network_n=row["network_inside"]["points_sampled"],
            parcels_in=row["parcels_inside"]["points_inside"],
            parcels_n=row["parcels_inside"]["points_sampled"],
            best_layer=best_layer,
        )

    kept = scored[:limit]
    runner_up = kept[1]["corridor_score"] if len(kept) > 1 else None

    # The layer the OLD rule would have proposed, and what it actually scored.
    # Area appears here for one purpose — to show what ranking by it does — and
    # takes no part in the ordering above. A reader who is told only "we now
    # rank differently" has to take that on trust; a reader shown that the
    # biggest layer holds 100% of the parcels can check it.
    by_area = max(
        (r for r in scored if isinstance(r.get("area"), (int, float))),
        key=lambda r: float(r["area"]),
        default=None,
    )
    swallows_everything = [
        {
            "layer": r["layer"],
            "area": r["area"],
            "unit": r["unit"],
            "network_inside": r["network_inside"]["fraction"],
            "parcels_inside": r["parcels_inside"]["fraction"],
            "rank": r["rank"],
        }
        for r in scored
        if (r["parcels_inside"]["fraction"] or 0.0) >= 1.0
    ][:limit]

    block: dict[str, Any] = {
        # `ranked` is False when not one candidate could be scored: the rows
        # are still returned, with their fractions and their reasons, but
        # calling that an ordering would be claiming a comparison nobody made.
        "ranked": bool(measurable),
        "ranked_by": (
            (
                "corridor_score = network_inside x (1 - parcels_inside), "
                "highest first, ties broken by layer name; a candidate whose "
                "score could not be measured sorts last and is never treated "
                "as a zero"
            )
            if measurable
            else (
                "nothing — no candidate could be scored, so these rows are in "
                "layer-name order and are NOT ranked"
            )
        ),
        "why_not": (
            None
            if measurable
            else (
                "one of the two fractions was unmeasurable for every candidate "
                "on this layout"
            )
        ),
        "best": (
            {
                "layer": best_layer,
                "corridor_score": best["corridor_score"],
                "network_inside": best["network_inside"]["fraction"],
                "parcels_inside": best["parcels_inside"]["fraction"],
                "runner_up": kept[1]["layer"] if len(kept) > 1 else None,
                "runner_up_score": runner_up,
                "why": best["why"],
                "verdict": (
                    f"{best_layer!r} behaves more like a road corridor than any "
                    f"other closed-ring layer on layout {layout!r}: it holds "
                    "the most of the network among the layers that do not hold "
                    "all of the parcels. It is a COMPARISON on this drawing, "
                    "not a threshold and not a claim about what the layer is — "
                    "read the two fractions beside it before acting on it"
                ),
            }
            if best is not None and best_layer
            else None
        ),
        "no_corridor_signal": (
            None
            if best_layer
            else (
                (
                    "not one candidate layer scored above zero: every one of "
                    "them either holds none of the sampled road network or "
                    "holds every sampled parcel. On a drawing with no road "
                    "corridors that is the correct outcome, and it is stated "
                    "rather than filled in with the largest layer"
                )
                if measurable
                else (
                    "no candidate could be SCORED at all: one of the two "
                    "fractions was unmeasurable for every one of them — most "
                    "often because the only closed-ring geometry on this "
                    "layout is the candidate itself, so there is no parcel "
                    "geometry left to ask the second question about. That is "
                    "an absence, not a zero, and not a finding of 'no corridor'"
                )
            )
        ),
        "candidates": kept,
        "candidates_measurable": len(measurable),
        "candidates_scored": len(scored),
        "candidates_truncated": len(scored) > len(kept),
        "ranking_by_size_would_have_proposed": (
            {
                "layer": by_area["layer"],
                "area": by_area["area"],
                "unit": by_area["unit"],
                "corridor_score": by_area["corridor_score"],
                "rank_on_the_corridor_signal": by_area["rank"],
                "network_inside": by_area["network_inside"]["fraction"],
                "parcels_inside": by_area["parcels_inside"]["fraction"],
                "why_it_is_shown": (
                    "this is the layer the previous rule — largest closed-ring "
                    "layer first — would have offered. Its two fractions are "
                    "here so that the change of rule can be checked rather "
                    "than believed"
                ),
            }
            if by_area is not None
            else None
        ),
        "contains_every_sampled_parcel": {
            "count": sum(
                1
                for r in scored
                if (r["parcels_inside"]["fraction"] or 0.0) >= 1.0
            ),
            "rows": swallows_everything,
            "cap": limit,
            "why": (
                "a layer that contains every parcel that was sampled is a site "
                "boundary or an envelope, whatever it is called. Retrying a "
                "frontage check against one of these reports that every parcel "
                "has frontage — a wrong answer with a confident label. They "
                "score zero here BECAUSE they contain the parcels, not because "
                "they are large"
            ),
        },
        "network_layers": list(network_layers),
        "network_layers_source": " and ".join(network_sources) or "the Dossier",
        "parcel_layers": list(parcel_layers),
        "parcel_layers_source": parcel_source,
        "sample": {
            "network_points_sampled": network["sampled"],
            "network_population": network["population"],
            "network_sample_step": network["step"],
            "network_points_without_a_stored_point": network["without_a_point"],
            "network_point_fields_used": network["point_fields_used"],
            "network_points_inside_no_candidate": net_outside,
            "parcel_points_sampled": parcels["sampled"],
            "parcel_population": parcels["population"],
            "parcel_sample_step": parcels["step"],
            "parcel_points_without_a_stored_point": parcels["without_a_point"],
            "parcel_point_fields_used": parcels["point_fields_used"],
            "parcel_points_inside_no_candidate": par_outside,
            "rings_tested": len(rows),
            "point_in_ring_tests": net_tests + par_tests,
            "basis": (
                f"one network row every {network['step']:.4g} and one parcel "
                f"row every {parcels['step']:.4g}, at evenly spaced positions "
                "in a list sorted by layer and handle — so the sample spans "
                "the drawing rather than the corner the drafter started in, "
                "and so two identical calls take the same sample. A stride of "
                "1 means the population was taken whole"
            ),
            "it_is_a_sample": (
                "these fractions describe the sample, not the layer. A larger "
                "sample would move them; measured on the reference drawing it "
                "does not move the ORDER"
            ),
            "points_are_proxies": (
                "an entity is represented by one stored point: a closed ring by "
                "its polygon centroid, everything else by the centre of its "
                "bounding box. For a straight LINE that centre is the midpoint "
                "and lies on the geometry; for an ARC or a bent polyline it "
                "does not, so a road that curves can be counted outside a "
                "corridor it runs through. The counts of which field was used "
                "are published above"
            ),
            "outside_everything_note": (
                "`network_points_inside_no_candidate` is the share of the "
                "network that sits inside NO closed-ring layer at all. It caps "
                "what any candidate's `network_inside` can reach, and on the "
                "reference drawing it is large because that road network is "
                "drawn three times with two copies parked off the site"
            ),
        },
        "role_note": role_note,
    }
    block.update(standing)
    return block
