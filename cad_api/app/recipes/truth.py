"""Does the drawing tell the truth about itself? — DOSSIER Phase 6.

Owner: subagent TRUTH. Two recipes, `dimension_truth` and `drafting_hygiene`,
registered here the way `library.py` registers its eight: at module import,
never from a request.

## Why a printed number is worth checking at all

A dimension label is TEXT. It is **typed**, not measured. Nothing in a DXF file
binds the characters `25.00` to the line they sit beside; move the line, forget
the label, and the drawing goes on stating a number that is no longer true —
confidently, in a document somebody builds from. That is the same class of
defect as answering "0 road parcels": literally correct about the file, wrong
about the world.

`schedule_check` (`app/store_schedule.py`) is this recipe's older sibling and
the two must not be confused, so the difference is written down rather than
left to be inferred:

| | `schedule_check` | `dimension_truth` |
|---|---|---|
| the stated number comes from | a cell in a drawn `ACAD_TABLE` | free annotation text floating in the drawing |
| the measured number is | the **area** of a closed ring | a **length**: the nearest edge of a ring, or a path entity's own length |
| the label is tied to geometry by | the smallest polygon containing it that contains **no other schedule key** | containment **and** nearest-geometry-within-a-radius, with every candidate kept |
| ambiguity is | resolved — the schedule's own key anchors the choice | **reported**, never resolved, because there is no external key to anchor it |
| its tolerance is | 2 % relative, because a schedule is rounded to whole units | derived per label from the decimals that label prints, plus any `±` the label states itself |

The last row of that table is the important one. `schedule_check` may pick the
smallest containing polygon because the schedule supplies an independent
identifier: if the choice were wrong, the number would not be in the table at
all. Here there is no such anchor. A label sitting between two lines is
genuinely ambiguous, and choosing the closer one would produce a verdict that
looks exactly like a checked one.

## What this recipe does not do

* It does **not** classify text. `dossier_census.classify_text` decides what
  counts as a `dimension_value`, and it is called — not copied. A second
  classifier in this repo would disagree with the first one on some string and
  nothing in either response would show which had spoken.
* It does **not** rewrite geometry. Point-in-ring and point-to-segment come
  from `app.region`, exactly as `store_spatial` uses them; the grid prefilter
  is `store_spatial._SpatialIndex` itself. Importing a sibling's private
  helper follows the precedent set in `library.py`, which calls
  `store_spatial._join` for the same reason: two ray-casters would answer
  differently on the same polygon with no sign of it in either response.
* It does **not** convert units. A label reading `1.5 m` in a drawing whose
  header declares inches is reported as a unit mismatch and its verdict is
  withheld. A conversion here would decide, silently, that the label means what
  we assumed — and a wrong conversion produces confident agreement, which is
  worse than no answer.

## The honest hole, stated before anyone finds it

Content decides the class (rule G1), and by content a decimal number in a
legend is indistinguishable from a dimension label. On `lineweights.dxf` the
texts `0.00`, `0.05`, `0.13` are a lineweight key, not measurements, and this
recipe will happily associate them with the sample lines drawn beside them and
report a disagreement. That is not a bug to be tuned away with a layer-name
filter — it is the limit of reading content, and it is stated in
`not_established` on every response so a reader meets it before quoting a
number.
"""

from __future__ import annotations

import math
import re
from typing import Any, Final, Mapping, Sequence

from .. import dossier_census as census
from .. import evidence as ev
from .. import region
from .. import store_spatial as spatial
from .. import store_tables as tables
from ..mongo import COLL_DRAWINGS, COLL_ENTITIES, coll
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
# A number owned by another module is taken from it, never copied: a copied
# limit differs from the real one on the first day either is changed, and the
# refusal then names a number that did not refuse it.

#: Text rows read from one layout before classification. Taken from its owner.
MAX_TEXT_ROWS: Final[int] = spatial.MAX_CANDIDATES

#: Geometry rows that may become candidates. Taken from its owner.
MAX_GEOMETRY_ROWS: Final[int] = spatial.MAX_TARGETS

#: Dimension labels actually checked in one call. Far below the text limit,
#: because every label carries a spatial search and a row in the answer.
MAX_LABELS: Final[int] = 5_000

#: Rings whose vertices are loaded. Only the candidates that survived the
#: prefilter are loaded at all, so this binds the NEEDED set, not the layout:
#: every ring loaded is held in memory whole for as long as the check runs.
MAX_RINGS_LOADED: Final[int] = 2_000

#: Candidate pairs surviving the spatial prefilter. Taken from its owner, and
#: counted the same way: after the prefilter, never over the raw product.
PAIRS_BUDGET: Final[int] = spatial.PAIRS_BUDGET

#: Candidates listed per ambiguous label, and handles listed per hygiene
#: finding. Counts always cover everything; only the listing is cut, and every
#: cut says how much it left out.
MAX_CANDIDATES_LISTED: Final[int] = 10
MAX_HANDLES_LISTED: Final[int] = 20

#: The relative floor under every tolerance, and it is a fact about floating
#: point rather than about any drawing. `extract._length_and_area` records the
#: measured size of it: a shoelace over projected coordinates sums numbers
#: around 1e12 to produce a number around 1e2, leaving a relative error around
#: 4e-7. A band tighter than this reports arithmetic as a defect.
ARITHMETIC_NOISE_RELATIVE: Final[float] = 1e-6

#: The DXF default layer. Not a drawing-specific constant (G1): `0` is defined
#: by the file format itself and exists in every DXF ever written, exactly like
#: `extract.RING_TYPES`. It is a real defect in most drafting standards because
#: content on it cannot be frozen, coloured, or plotted as a group — and inside
#: a block definition it takes on the properties of wherever the block lands.
DEFAULT_LAYER: Final[str] = "0"


# --- printed values ----------------------------------------------------------
#
# What follows extracts the VALUE from a label. It deliberately does not decide
# whether a string is a dimension at all — `census.classify_text` decides that,
# and is called. Two private names are imported from that module rather than
# re-typed, following the precedent in `library.py`, which calls
# `store_spatial._join` instead of copying it: a second notion of "what a
# number looks like" would classify a string as a dimension and then fail to
# find its number, and the response would show a coverage hole with no cause.

_NUMBER: Final[re.Pattern[str]] = census._NUMBER_TOKEN
_PAIR: Final[re.Pattern[str]] = census._DIM_PAIR

#: A trailing unit token, built from the census's own unit alternatives so the
#: two cannot drift apart.
_TRAILING_UNIT: Final[re.Pattern[str]] = re.compile(
    rf"({census._UNIT})\s*$", re.IGNORECASE
)

#: Unit tokens that mean the label is not stating a length at all.
_AREA_UNITS: Final[frozenset[str]] = frozenset({"sqm", "m2"})
_ANGLE_UNITS: Final[frozenset[str]] = frozenset({"°", "deg"})

#: The marks a draughtsman writes before a stated tolerance.
_PLUS_MINUS: Final[tuple[str, ...]] = ("±", "+/-")


def printed_value(text: str) -> dict[str, Any]:
    """The number a label PRINTS, with everything needed to judge it.

    Returns a dict that always carries `value`, and carries `skip_reason` when
    the label states something this check cannot compare against a length. A
    skipped label is counted and named, never dropped (G8).

    `decimals` is the count of digits printed after the decimal point, and it
    is what sets the tolerance: `25.00` is typed to a hundredth, so a measured
    24.997 is the same number as far as the drawing was ever able to say, while
    24.00 is not.
    """
    raw = (text or "").strip()
    out: dict[str, Any] = {
        "text": raw,
        "value": None,
        "decimals": None,
        "unit": None,
        "stated_tolerance": None,
        "skip_reason": None,
    }
    if not raw:
        out["skip_reason"] = "the label is empty"
        return out

    if _PAIR.fullmatch(raw):
        out["skip_reason"] = (
            "this label states a SIZE (`w x h`), not one length. Comparing a "
            "pair against a single measured length would be comparing two "
            "different things, so it is counted here and not judged."
        )
        return out

    unit_match = _TRAILING_UNIT.search(raw)
    unit = unit_match.group(1).strip().casefold() if unit_match else None
    out["unit"] = unit
    if unit in _AREA_UNITS:
        out["skip_reason"] = (
            f"this label states an AREA (unit {unit!r}). Areas stated by a "
            "drawing are compared with drawn areas by `schedule_check`, which "
            "measures rings; this recipe measures lengths."
        )
        return out
    if unit in _ANGLE_UNITS:
        out["skip_reason"] = (
            f"this label states an ANGLE (unit {unit!r}), and no angle is "
            "stored per entity, so there is nothing to compare it with."
        )
        return out

    tokens = _NUMBER.findall(raw)
    if not tokens:
        out["skip_reason"] = "no number could be read out of this label"
        return out

    first = tokens[0]
    out["value"] = float(first.replace(",", ""))
    out["decimals"] = len(first.split(".", 1)[1]) if "." in first else 0

    if len(tokens) > 1 and any(mark in raw for mark in _PLUS_MINUS):
        # The drawing states its own tolerance. Using it is not leniency: it is
        # reading the label as its author wrote it.
        try:
            out["stated_tolerance"] = abs(float(tokens[1].replace(",", "")))
        except ValueError:  # pragma: no cover - the pattern already excludes it
            out["stated_tolerance"] = None
    return out


def tolerance_for(
    printed: Mapping[str, Any], measured: float, override: float | None
) -> dict[str, Any]:
    """The agreement band for one comparison, and every part it is made of.

    Derived, never hard-coded, and never in metres: each component is in the
    units of the numbers themselves, so this behaves identically in a drawing
    that is in inches and in one that declares no unit at all (G2).
    """
    if override is not None:
        return {
            "value": float(override),
            "basis": "handed over by the caller, in DRAWING units",
            "from_printed_decimals": None,
            "from_stated_tolerance": None,
            "from_arithmetic_noise": None,
        }
    decimals = printed.get("decimals")
    half_ulp = 0.5 * (10.0 ** -int(decimals)) if decimals is not None else None
    stated = printed.get("stated_tolerance")
    noise = ARITHMETIC_NOISE_RELATIVE * abs(float(measured))
    parts = [v for v in (half_ulp, stated, noise) if v is not None]
    return {
        "value": max(parts) if parts else noise,
        "basis": (
            "the widest of: half of the last decimal place the label prints "
            f"({half_ulp}), the tolerance the label states itself ({stated}), "
            f"and {ARITHMETIC_NOISE_RELATIVE} x the measured value for "
            f"floating-point noise ({noise}). In DRAWING units, like the two "
            "numbers it separates."
        ),
        "from_printed_decimals": half_ulp,
        "from_stated_tolerance": stated,
        "from_arithmetic_noise": noise,
    }


# --- geometry, all of it borrowed --------------------------------------------


def _ring_points(doc: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in (doc.get("ring") or [])]


def _box_of(doc: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    box = doc.get("bbox")
    if not isinstance(box, Mapping):
        return None
    lo, hi = box.get("min"), box.get("max")
    if not lo or not hi or len(lo) < 2 or len(hi) < 2:
        return None
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def _label_point(doc: Mapping[str, Any]) -> tuple[tuple[float, float] | None, str]:
    """Where a label sits, and which point that is.

    `anchor_point` first, always: `store_spatial.POINT_USED` is emphatic that
    the centre of a bounding box may never stand in for an insertion point, and
    on a rotated grid it misses by metres. But a `DIMENSION` is not an anchor
    type and carries no insertion point at all, and refusing to place it would
    silently drop the very entity this recipe exists for. So the box centre is
    used only where there is no insertion point to use, and the response says
    which of the two produced each row.
    """
    anchor = doc.get("anchor_point")
    if anchor and len(anchor) >= 2 and anchor[0] is not None and anchor[1] is not None:
        return (float(anchor[0]), float(anchor[1])), "anchor_point"
    centre = doc.get("bbox_centre")
    if centre and len(centre) >= 2 and centre[0] is not None and centre[1] is not None:
        return (
            (float(centre[0]), float(centre[1])),
            "bbox_centre (this entity type carries no insertion point)",
        )
    return None, "no insertion point and no bounding box centre are stored"


def _label_radius(doc: Mapping[str, Any]) -> tuple[float | None, str]:
    """How far from a label its geometry may sit, when the caller named no radius.

    The SHORTER side of the label's own bounding box — its character height —
    and nothing else. No multiplier, because a multiplier is a drafting
    convention smuggled into a `.py` (G1), and no drawing-wide number, because
    a dimension text is drawn at that dimension's own height and its box is the
    only scale in the file belonging to THAT label. It therefore behaves
    identically in metres, in inches, and in a file declaring no unit.

    The shorter side rather than the diagonal, and that was a correction rather
    than a first guess. The diagonal grows with the number of CHARACTERS: on
    the reference drawing's sheet, `68878.55` cast a net roughly eight times
    wider than `4.86` did, so the longer number reached more geometry and was
    reported ambiguous for a reason that had nothing to do with the drawing.
    The shorter side is the character height for any axis-aligned rotation, and
    a rotated label keeps the same radius as an upright one.

    The honest limit: an MTEXT wrapped over several lines has a box as tall as
    the paragraph, so its shorter side may be its width instead. The radius is
    published on every row precisely so that case can be seen.
    """
    box = _box_of(doc)
    if box is None:
        return None, (
            "no search radius was given and this label carries no bounding "
            "box, so no radius could be derived; only containment could place it"
        )
    shorter = min(box[2] - box[0], box[3] - box[1])
    if shorter <= 0.0:
        return None, (
            "no search radius was given and this label's bounding box is flat "
            "in one axis, so no character height could be derived from it"
        )
    return shorter, (
        "derived per label as the SHORTER side of its own bounding box — the "
        "character height — in DRAWING units. Not the diagonal: that grows "
        "with the number of characters, so a longer number would cast a wider "
        "net for a reason that has nothing to do with the drawing"
    )


def _distance_to(
    doc: Mapping[str, Any], x: float, y: float
) -> tuple[float | None, str]:
    """How far a label point is from one piece of geometry, and how exactly.

    Two answers of different quality, and the response never mixes them up.
    Where a ring is stored the distance is exact, measured against the polygon's
    own edges with `region._point_segment_distance` — the same function
    `store_spatial._ring_distance` uses. Where it is not, only a bounding box
    exists: **endpoint coordinates are not stored for path entities**, which is
    exactly the gap Phase 5c opens as a decision. A box lies at or inside the
    real geometry, so that distance is a LOWER bound and a diagonal line can be
    associated when it is really further away.
    """
    ring = _ring_points(doc)
    if len(ring) >= 3:
        n = len(ring)
        best = min(
            region._point_segment_distance(x, y, ring[i], ring[(i + 1) % n])
            for i in range(n)
        )
        return best, "exact: the shortest distance to the polygon's own edges"
    box = _box_of(doc)
    if box is None:
        return None, "no bounding box is stored for this entity"
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy), (
        "lower bound: the shortest distance to the entity's BOUNDING BOX. "
        "Endpoint coordinates are not stored for path entities, so a diagonal "
        "line reads as nearer than it is"
    )


#: The rule, in one line, published on every response.
REACH_DECIDES: Final[str] = (
    "reach decides, containment describes. A piece of geometry becomes a "
    "candidate when it lies within the label's search radius — and a polygon "
    "offers ONE candidate per distinct length among the edges that are within "
    "reach, because a label a character-height from one edge is naming that "
    "edge. A polygon that merely CONTAINS the label with no edge in reach is a "
    "sheet border or a site boundary, not the thing being dimensioned; it is "
    "listed under `contains_but_out_of_reach` and given no measure, which is "
    "the correction the corpus forced — on the reference drawing's sheet, the "
    "title border contained every label and made all 21 of them ambiguous."
)


def _candidates_for(
    doc: Mapping[str, Any],
    x: float,
    y: float,
    radius: float | None,
    *,
    contained: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, bool]:
    """What one piece of geometry offers this label.

    Returns `(candidates, out_of_reach, unmeasurable)`. A polygon may offer
    MORE than one candidate, and that is the decision that keeps this check
    able to fail: a label sitting in the corner of a 12 x 25 plot is within
    reach of a 12 edge AND a 25 edge, so it offers both — and the ambiguity
    machinery then reports it as ambiguous instead of quietly taking whichever
    of the two happened to be nearer. What is never done is offering every
    measure a shape has — perimeter, each edge, the area — and accepting
    whichever one matched, which is a check that can never find a lie because
    something always matches.
    """
    ring = _ring_points(doc)
    edges = doc.get("edge_lengths")
    route = "containment" if contained else "within_radius"
    identity = {
        "handle": doc.get("handle"),
        "layer": doc.get("layer"),
        "type": doc.get("type"),
        "route": route,
        "contains_the_label": contained,
    }

    if (
        len(ring) >= 3
        and isinstance(edges, (list, tuple))
        and len(edges) == len(ring)
    ):
        n = len(ring)
        per_edge = sorted(
            (
                region._point_segment_distance(x, y, ring[i], ring[(i + 1) % n]),
                float(edges[i]),
                i,
            )
            for i in range(n)
        )
        in_reach = (
            []
            if radius is None
            else [e for e in per_edge if e[0] <= radius]
        )
        if not in_reach:
            return (
                [],
                {
                    **identity,
                    "nearest_edge_distance": per_edge[0][0],
                    "nearest_edge_length": per_edge[0][1],
                    "why": (
                        "no edge of this polygon is within the label's search "
                        "radius"
                        + (
                            ""
                            if radius is not None
                            else ", and no radius could be derived for this label"
                        )
                    ),
                },
                False,
            )
        out: list[dict[str, Any]] = []
        seen: set[float] = set()
        for distance, length, index in in_reach:
            key = round(length, 9)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    **identity,
                    "distance": distance,
                    "distance_basis": "exact: the shortest distance to that edge",
                    "measured": length,
                    "measure_basis": (
                        f"the length of the polygon's nearest edge at this "
                        f"value (edge {index} of {n}, counted from the stored "
                        "ring)"
                    ),
                }
            )
        return out, None, False

    distance, distance_basis = _distance_to(doc, x, y)
    if distance is None:
        return [], None, False
    if radius is None or distance > radius:
        if contained:
            return (
                [],
                {
                    **identity,
                    "nearest_edge_distance": distance,
                    "nearest_edge_length": None,
                    "why": (
                        "this polygon contains the label but stores no usable "
                        "per-edge lengths, and its outline is not within the "
                        "label's search radius"
                    ),
                },
                False,
            )
        return [], None, False

    length = doc.get("length")
    if length is None:
        # Counted as unmeasurable, never as zero (G8).
        return [], None, True
    return (
        [
            {
                **identity,
                "distance": distance,
                "distance_basis": distance_basis,
                "measured": float(length),
                "measure_basis": (
                    "the polygon's stored length, which for a closed ring is "
                    "its PERIMETER — its per-edge lengths are not stored, so "
                    "no single edge could be offered"
                    if len(ring) >= 3
                    else "the entity's stored length"
                ),
            }
        ],
        None,
        False,
    )


# --- evidence ----------------------------------------------------------------


def _observation(origin: ev.Origin, detail: str, **kw: Any) -> ev.Observation:
    return ev.Observation(origin=origin, detail=detail, **kw)


def _computed_evidence(
    claim: str,
    *,
    drawing_id: str,
    scope: ev.Scope,
    observations: Sequence[ev.Observation],
    not_established: str,
    how_to_verify: str,
) -> ev.Evidence:
    """Evidence for a claim born from comparing two numbers.

    `tokens` is empty and stays empty. There is no word which, appearing in a
    file, makes that file STATE that six of its labels disagree with their
    geometry: the disagreement is computed, so the ceiling is `inferred` and
    that ceiling is right.
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


def _length_quantity(
    value: float | None,
    units: Mapping[str, Any],
    *,
    basis: str,
    method: str,
    withheld_reason: str = "not measurable for this entity",
) -> dict[str, Any]:
    """A length that carries its unit, and admits when the drawing states none (G2)."""
    if value is None:
        return ev.Quantity.withheld(
            basis=basis, method=method, reason=withheld_reason
        ).as_dict()
    return ev.Quantity.measured(
        basis=basis,
        method=method,
        value=float(value),
        unit=units.get("length_unit"),
        unit_reason=(
            None
            if units.get("length_unit")
            else units.get("why_no_unit")
            or "this drawing declares no units, so none is claimed"
        ),
    ).as_dict()


# =============================================================================
# 1. dimension_truth — the pure core
# =============================================================================


#: Said on every response. The limit of reading content, met before a number is
#: quoted rather than after.
CONTENT_IS_NOT_INTENT: Final[str] = (
    "a `dimension_value` is decided from the text CONTENT alone (rule G1), and "
    "by content a decimal number in a legend, a price list, or a coordinate "
    "table is indistinguishable from a dimension label. Every row here says "
    "which geometry it was matched to and how far away it sat, so a reading "
    "that is not a dimension at all can be recognised as such — it is not "
    "filtered out by layer name, because a layer name is one drawing's habit."
)

#: Said on every response that reports a disagreement.
WHICH_ONE_IS_WRONG: Final[str] = (
    "WHICH of the two is wrong is not established. What is measured is that "
    "the typed number and the drawn geometry do not agree within the stated "
    "band — not that the geometry is right. A label can be stale, mistyped, or "
    "copied from the label above it; an outline can be the thing that moved."
)


def dimension_labels(
    labels: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The rows the census would class `dimension_value`, in handle order.

    One function, called by both the check and the wrapper that decides which
    rings are worth loading. Two copies of this rule would drift, and the
    symptom would be a ring that was never fetched showing up as a label that
    could not be associated — a coverage hole with no visible cause.

    `classify_text` is called WITHOUT font information on purpose: the
    dimension branch is decided before the SHX branch is ever reached, so the
    verdict for this class is font-independent and identical to the census's.
    """
    out: list[dict[str, Any]] = []
    for row in labels:
        text = row.get("text")
        if text is None or not str(text).strip():
            continue
        if census.classify_text(str(text))[0] != census.KIND_DIMENSION:
            continue
        out.append(dict(row))
    out.sort(key=lambda r: str(r.get("handle") or ""))
    return out


def radius_of(
    row: Mapping[str, Any], search_radius: float | None
) -> tuple[float | None, str]:
    """The search radius for one label: the caller's, or derived from the label."""
    if search_radius is not None:
        return float(search_radius), "handed over by the caller, in DRAWING units"
    return _label_radius(row)


def check_labels(
    labels: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    search_radius: float | None = None,
    tolerance: float | None = None,
    show_all: bool = False,
    row_cap: int = MAX_ROWS_RETURNED,
) -> dict[str, Any]:
    """Compare printed dimension values against measured geometry. No MongoDB.

    Split out of the recipe wrapper for the reason `store_tables.py` gives for
    the same split: the test suite runs in a throwaway container with no
    database, and a check that has never been shown a real lie has not been
    tested. `labels` and `geometry` are Mongo documents exactly as they are
    stored, not intermediate objects, so a fixture proves the projection too.

    Raises:
        RecipeRefused: `RECIPE_INPUT_TOO_LARGE`, when more labels are handed
            over than `MAX_LABELS`, or when the pairs surviving the spatial
            prefilter exceed `PAIRS_BUDGET`. Both are checked BEFORE the work
            runs, so a limit limits work rather than reporting on work already
            done.
    """
    labels = list(labels)
    geometry = list(geometry)

    # The census is the authority on what a dimension label is, so it is asked,
    # and its own arithmetic is published beside this recipe's. If the two ever
    # disagree, the response shows it rather than hiding it behind one number.
    counted = census.annotation_census(labels)
    dimension_class = next(
        (c for c in counted["classes"] if c["kind"] == census.KIND_DIMENSION), None
    )

    chosen = dimension_labels(labels)

    refuse_oversize(
        what="dimension labels to check",
        size=len(chosen),
        limit=MAX_LABELS,
        hint=(
            "Narrow `label_layers`, or run one layout at a time. Every label "
            "costs a spatial search and a row in the answer, so a run this "
            "large is a different question from the one this recipe answers."
        ),
    )

    drawing_unit = units.get("length_unit")
    boxes = [_box_of(g) for g in geometry]

    # The prefilter index is built over boxes GROWN by the largest radius any
    # label will use, so `hits()` returns a superset of what each label can
    # reach and each label then filters by its own radius. `_SpatialIndex` is
    # used as it is: its equivalence to a full scan is already tested in
    # `test_spatial.py`, and an index that changes the answer is not an
    # optimisation but a bug.
    radii = [radius_of(row, search_radius) for row in chosen]
    grow = max([r for r, _ in radii if r is not None], default=0.0)
    grown = [
        None if b is None else (b[0] - grow, b[1] - grow, b[2] + grow, b[3] + grow)
        for b in boxes
    ]
    index = spatial._SpatialIndex(grown)

    points: list[tuple[tuple[float, float] | None, str]] = [
        _label_point(row) for row in chosen
    ]
    pairs = sum(
        len(index.hits(p[0], p[1])) for p, _ in points if p is not None
    )
    if pairs > PAIRS_BUDGET:
        raise RecipeRefused(
            "RECIPE_INPUT_TOO_LARGE",
            f"{pairs} label/geometry pairs survived the spatial prefilter, "
            f"above the budget of {PAIRS_BUDGET}. Counted after the prefilter, "
            f"not over {len(chosen)} x {len(geometry)} raw pairs.",
            "Narrow `geometry_layers`, or hand over a smaller `search_radius`. "
            "A radius wide enough to reach the whole drawing does not make the "
            "answer broader, it makes every label ambiguous.",
        )

    rows: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    unassociated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    without_point = 0
    agree = 0
    disagree = 0
    unit_mismatch = 0

    for row, (point, point_basis), (radius, radius_basis) in zip(
        chosen, points, radii
    ):
        handle = row.get("handle")
        printed = printed_value(str(row.get("text") or ""))
        base: dict[str, Any] = {
            "text_handle": handle,
            "text_layer": row.get("layer"),
            "text_type": row.get("type"),
            "printed_text": printed["text"],
            "printed": printed["value"],
            "printed_unit": printed["unit"],
        }

        if printed["skip_reason"] is not None:
            skipped.append({**base, "reason": printed["skip_reason"]})
            continue

        if printed["unit"] and drawing_unit and printed["unit"] != str(drawing_unit).casefold():
            unit_mismatch += 1
            skipped.append(
                {
                    **base,
                    "drawing_unit": drawing_unit,
                    "reason": (
                        f"the label prints {printed['unit']!r} while this "
                        f"drawing declares {drawing_unit!r}. No conversion is "
                        "applied and no verdict is given: a conversion would "
                        "decide silently that the label means what we assumed, "
                        "and a wrong one produces confident agreement."
                    ),
                }
            )
            continue

        if point is None:
            without_point += 1
            unassociated.append({**base, "reason": point_basis})
            continue

        x, y = point
        candidates: list[dict[str, Any]] = []
        out_of_reach: list[dict[str, Any]] = []
        without_measure = 0
        for i in index.hits(x, y):
            doc = geometry[i]
            ring = _ring_points(doc)
            contained = (
                region.point_in_ring(
                    x, y, ring, origin=spatial._origin_of(dict(doc), ring)
                )
                != "outside"
                if len(ring) >= 3
                else False
            )
            found, missed, unmeasurable = _candidates_for(
                doc, x, y, radius, contained=contained
            )
            candidates.extend(found)
            if missed is not None:
                out_of_reach.append(missed)
            without_measure += 1 if unmeasurable else 0

        candidates.sort(
            key=lambda c: (
                math.inf if c["distance"] is None else c["distance"],
                str(c["handle"] or ""),
                c["measured"],
            )
        )
        out_of_reach.sort(key=lambda c: str(c["handle"] or ""))

        if not candidates:
            unassociated.append(
                {
                    **base,
                    "point_basis": point_basis,
                    "search_radius": radius,
                    "search_radius_basis": radius_basis,
                    "geometry_without_a_measure": without_measure,
                    "contains_but_out_of_reach": out_of_reach[:MAX_CANDIDATES_LISTED],
                    "contains_but_out_of_reach_total": len(out_of_reach),
                    "reason": (
                        "no measurable geometry lies within this label's search "
                        "radius"
                        + (
                            ""
                            if radius is not None
                            else ", and no radius could be derived for it"
                        )
                        + (
                            ""
                            if not out_of_reach
                            else (
                                f"; {len(out_of_reach)} polygon(s) contain it "
                                "but have no edge in reach — they are listed, "
                                "so a person can still open them"
                            )
                        )
                    ),
                }
            )
            continue

        # Ambiguity that would not change the answer is not ambiguity. Several
        # candidates all measuring the same value within the band leave one
        # answer, and saying so is more useful than refusing.
        nearest = candidates[0]
        band = tolerance_for(printed, nearest["measured"], tolerance)
        spread = max(c["measured"] for c in candidates) - min(
            c["measured"] for c in candidates
        )
        if len(candidates) > 1 and spread > band["value"]:
            ambiguous.append(
                {
                    **base,
                    "point_basis": point_basis,
                    "search_radius": radius,
                    "search_radius_basis": radius_basis,
                    "candidates": candidates[:MAX_CANDIDATES_LISTED],
                    "candidates_total": len(candidates),
                    "candidates_omitted": max(
                        0, len(candidates) - MAX_CANDIDATES_LISTED
                    ),
                    "contains_but_out_of_reach": out_of_reach[:MAX_CANDIDATES_LISTED],
                    "contains_but_out_of_reach_total": len(out_of_reach),
                    "measured_spread": spread,
                    "tolerance": band,
                    "reason": (
                        f"{len(candidates)} pieces of geometry could be the one "
                        f"this label names, and they do not measure the same "
                        f"thing (their values span {spread}, wider than the "
                        f"band of {band['value']}). No verdict is given: "
                        "choosing the nearer one would produce a row that looks "
                        "exactly like a checked one."
                    ),
                }
            )
            continue

        measured = nearest["measured"]
        difference = measured - float(printed["value"])
        agrees = abs(difference) <= band["value"]
        agree += 1 if agrees else 0
        disagree += 0 if agrees else 1
        result = {
            **base,
            "measured": measured,
            "measured_quantity": _length_quantity(
                measured,
                units,
                basis=f"the length {nearest['handle']} offers to this label",
                method=nearest["measure_basis"],
            ),
            "difference": difference,
            "difference_ratio": (
                difference / float(printed["value"]) if printed["value"] else None
            ),
            "difference_percent": (
                100.0 * difference / float(printed["value"])
                if printed["value"]
                else None
            ),
            # Published without a threshold attached, deliberately. A row where
            # the two numbers are orders of magnitude apart is far more likely
            # to be a label that was never a dimension — a coordinate, a price,
            # a lineweight — than a drawing that is wrong by a factor of a
            # thousand. Naming a cut-off would turn that reading into a rule
            # and start hiding real defects behind it.
            "printed_over_measured": (
                float(printed["value"]) / measured if measured else None
            ),
            "agrees": agrees,
            "tolerance": band,
            "geometry_handle": nearest["handle"],
            "geometry_layer": nearest["layer"],
            "geometry_type": nearest["type"],
            "association_route": nearest["route"],
            "distance": _length_quantity(
                nearest["distance"],
                units,
                basis=f"from label {handle} to geometry {nearest['handle']}",
                method=nearest["distance_basis"],
                withheld_reason="this entity stores no geometry to measure from",
            ),
            "measure_basis": nearest["measure_basis"],
            "point_basis": point_basis,
            "search_radius": radius,
            "search_radius_basis": radius_basis,
            "contains_but_out_of_reach": out_of_reach[:MAX_CANDIDATES_LISTED],
            "contains_but_out_of_reach_total": len(out_of_reach),
            "candidates_total": len(candidates),
            "resolved_because": (
                "one candidate"
                if len(candidates) == 1
                else (
                    f"{len(candidates)} candidates, and every one of them "
                    f"measures the same value within the band ({spread} <= "
                    f"{band['value']}), so the answer does not depend on the "
                    "choice"
                )
            ),
            "geometry_without_a_measure": without_measure,
        }
        rows.append(result)

    checked = len(rows)
    total = len(chosen)
    listed = rows if show_all else [r for r in rows if not r["agrees"]]
    listed = sorted(
        listed, key=lambda r: -abs(r.get("difference_ratio") or 0.0)
    )[:row_cap]

    scope_note = (
        f"layout {layout!r}; {total} texts classed `dimension_value` by "
        f"dossier_census out of {counted['total']} text-bearing rows, checked "
        f"against {len(geometry)} measurable entities on the same layout; "
        "lengths in "
        + (str(drawing_unit) if drawing_unit else "DRAWING units, none declared")
    )
    scope = scope_for(layout=layout, units=units, note=scope_note)

    observations: list[ev.Observation] = []
    for r in [x for x in rows if not x["agrees"]][:3]:
        observations.append(
            _observation(
                ev.Origin.DIMENSION_TEXT,
                f"{r['text_type']} {r['text_handle']} on layer "
                f"{r['text_layer']!r} prints {r['printed_text']!r}",
                observed=str(r["printed_text"]),
                locator=str(r["text_handle"]),
                layout=layout,
            )
        )
        observations.append(
            _observation(
                ev.Origin.GEOMETRY,
                f"{r['geometry_type']} {r['geometry_handle']} on layer "
                f"{r['geometry_layer']!r} measures {r['measured']}",
                observed=f"{r['measured']}",
                locator=str(r["geometry_handle"]),
                layout=layout,
            )
        )
    if not observations and checked:
        first = rows[0]
        observations.append(
            _observation(
                ev.Origin.GEOMETRY,
                f"{checked} labels were compared and every one of them agrees "
                f"with its geometry, the first being {first['text_handle']}",
                observed=f"{first['measured']}",
                locator=str(first["geometry_handle"]),
                layout=layout,
            )
        )
    if len(unassociated) or not checked:
        observations.append(
            _observation(
                ev.Origin.ABSENCE,
                f"{len(unassociated)} of {total} dimension labels could not be "
                "tied to any geometry at all, so no verdict exists for them",
                locator=str(layout),
                layout=layout,
            )
        )

    evidence = _computed_evidence(
        (
            f"{disagree} of {checked} checked dimension labels disagree with "
            f"the geometry they label, on layout {layout!r}"
            if checked
            else (
                f"no dimension label on layout {layout!r} could be checked "
                f"against geometry ({total} were found)"
            )
        ),
        drawing_id=drawing_id,
        scope=scope,
        observations=observations,
        not_established=(
            WHICH_ONE_IS_WRONG
            + " "
            + CONTENT_IS_NOT_INTENT
            + " A DIMENSION whose text is the placeholder `<>` prints a number "
            "generated at draw time; the file does not store it, so no such "
            "entity can be checked here and none is counted as agreeing."
        ),
        how_to_verify=(
            "open the two handles in the drawing side by side — the label and "
            "the geometry are both reported per row precisely so this takes one "
            "click each. Where a row is ambiguous, decide which candidate the "
            "label belongs to and re-run with a smaller `search_radius` or a "
            "narrower `geometry_layers`."
        ),
    )

    body: dict[str, Any] = {
        "scope_note": scope_note,
        "labels_total": total,
        "labels_checked": checked,
        "labels_agree": agree,
        "labels_disagree": disagree,
        "labels_ambiguous": len(ambiguous),
        "labels_not_associated": len(unassociated),
        "labels_skipped": len(skipped),
        "labels_without_point": without_point,
        "labels_with_unit_mismatch": unit_mismatch,
        "coverage_fraction": (checked / total) if total else None,
        # The campaign's coverage arithmetic in miniature, published rather
        # than assumed: every label found is in exactly one bucket, and a
        # response where this does not hold has lost some of them.
        "coverage_arithmetic": {
            "equation": (
                "labels_total = labels_checked + labels_ambiguous + "
                "labels_not_associated + labels_skipped"
            ),
            "left": total,
            "right": checked + len(ambiguous) + len(unassociated) + len(skipped),
            "holds": total
            == checked + len(ambiguous) + len(unassociated) + len(skipped),
            "verdict_equation": "labels_checked = labels_agree + labels_disagree",
            "verdict_holds": checked == agree + disagree,
        },
        "coverage_note": (
            f"{checked} of {total} labels produced a verdict. The other "
            f"{total - checked} are counted per cause — {len(ambiguous)} "
            f"ambiguous, {len(unassociated)} tied to no geometry, "
            f"{len(skipped)} stating something that is not one length — and "
            "not one of them is counted as agreeing. This is a statement about "
            "the CHECK, not about the drawing: hiding it would overstate how "
            "much of the file was really examined."
            if total
            else "no text on this layout classes as a dimension value, so there "
            "is nothing to cover. That is a measured absence, not a clean bill"
        ),
        "geometry_candidates": len(geometry),
        "pairs_after_prefilter": pairs,
        "method": REACH_DECIDES,
        "route_note": (
            "every row names its `association_route`: `containment` when the "
            "label's point also falls inside the polygon it was matched to, "
            "`within_radius` when it sits beside it. Both had to be within "
            "reach to become a candidate at all. A label whose candidates "
            "measure different things is reported AMBIGUOUS and given no "
            "verdict — one candidate per distinct edge length means a label in "
            "the corner of a plot offers both edges and is refused, rather "
            "than quietly taking whichever was nearer."
        ),
        "point_used": (
            "the label's `anchor_point`, and only where the entity type carries "
            "none — a DIMENSION does not — the centre of its bounding box, "
            "named per row"
        ),
        "tolerance_basis": (
            "derived per row and published per row: the widest of half the last "
            "decimal place the label prints, the tolerance the label states "
            "itself after a ± mark, and "
            f"{ARITHMETIC_NOISE_RELATIVE} x the measured value for "
            "floating-point noise. Never a fixed number of metres: the band is "
            "in the units of the two numbers it separates, whatever those are, "
            "and a drawing is allowed to declare none (G2)."
        ),
        "content_is_not_intent": CONTENT_IS_NOT_INTENT,
        "which_one_is_wrong": WHICH_ONE_IS_WRONG,
        "census": {
            "source": "dossier_census.annotation_census",
            "rows_given": counted["total"],
            "classified": counted["classified"],
            "without_text": counted["without_text"]["count"],
            "dimension_value_count": (
                dimension_class["count"] if dimension_class else 0
            ),
            "dimension_value_samples": (
                dimension_class["samples"] if dimension_class else []
            ),
            "agrees_with_this_recipe": (
                (dimension_class["count"] if dimension_class else 0) == total
            ),
            "note": (
                "the census counts the population; this recipe checks it. The "
                "two numbers are published side by side so a drift between them "
                "is visible instead of being absorbed."
            ),
        },
        "rows": listed,
        "rows_ordering": (
            "disagreements first, widest RELATIVE difference at the top — the "
            "same ordering `schedule_check` uses. Read `printed_over_measured` "
            "before reading a row as a defect: two numbers orders of magnitude "
            "apart are usually a label that was never a dimension, not a "
            "drawing that is wrong by a factor of a thousand."
        ),
        "rows_showing": "disagreements only" if not show_all else "every checked label",
        "rows_returned": len(listed),
        "rows_omitted": max(
            0, len(rows if show_all else [r for r in rows if not r["agrees"]]) - len(listed)
        ),
        "ambiguous": ambiguous[:row_cap],
        "ambiguous_omitted": max(0, len(ambiguous) - row_cap),
        "not_associated": unassociated[:row_cap],
        "not_associated_omitted": max(0, len(unassociated) - row_cap),
        "skipped": skipped[:row_cap],
        "skipped_omitted": max(0, len(skipped) - row_cap),
        "limits_applied": {
            "labels": MAX_LABELS,
            "geometry_rows": MAX_GEOMETRY_ROWS,
            "rings_loaded": MAX_RINGS_LOADED,
            "pairs_after_prefilter": PAIRS_BUDGET,
            "rows_returned": row_cap,
            "candidates_listed_per_label": MAX_CANDIDATES_LISTED,
        },
    }
    return ev.attach(body, evidence)


# --- the wrapper that fetches ------------------------------------------------


def _entities():
    """This module's only door to MongoDB, and it goes through `coll()`."""
    return coll(COLL_ENTITIES)


def _fetch_labels(
    drawing_id: str, layout: str | None, layers: Sequence[str] | None
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "text": {"$nin": [None, ""]},
    }
    if layers:
        query["layer"] = {"$in": list(layers)}
    total = _entities().count_documents(dict(query))
    refuse_oversize(
        what="text-bearing entities on this layout",
        size=total,
        limit=MAX_TEXT_ROWS,
        hint="Narrow `label_layers`, or run one layout at a time.",
    )
    return list(
        _entities()
        .find(
            dict(query),
            {
                "handle": 1,
                "layer": 1,
                "layout": 1,
                "type": 1,
                "text": 1,
                "anchor_point": 1,
                "anchor_basis": 1,
                "bbox": 1,
                "bbox_centre": 1,
                "_id": 0,
            },
        )
        .sort([("handle", 1)])
    )


def _fetch_geometry(
    drawing_id: str, layout: str | None, layers: Sequence[str] | None
) -> tuple[list[dict[str, Any]], int]:
    """Measurable entities on one layout, WITHOUT their vertices.

    Two queries, never one top-level `$or`. Both are prefixed by `drawing_id` +
    `layout`, which is exactly the `(drawing_id, layout)` index; a top-level
    `$or` may be split by the planner into branches without an index, and this
    cluster runs with `notablescan` — a query without an indexed plan does not
    slow down, it FAILS, on the request path.

    The second query exists so that a closed ring whose length could not be
    computed — a bulged span, a reading that failed — is still counted rather
    than vanishing from the population. Counted, and reported.
    """
    base: dict[str, Any] = {"drawing_id": drawing_id, "layout": layout}
    if layers:
        base["layer"] = {"$in": list(layers)}
    projection = {
        "handle": 1,
        "layer": 1,
        "type": 1,
        "length": 1,
        "area": 1,
        "bbox": 1,
        "ring_status": 1,
        "ring_origin": 1,
        "edge_lengths": 1,
        "_id": 0,
    }

    found: dict[str, dict[str, Any]] = {}
    for query in (
        {**base, "length": {"$ne": None}},
        {**base, "ring_status": "complete", "length": None},
    ):
        for doc in _entities().find(query, projection):
            found[str(doc.get("handle"))] = doc

    refuse_oversize(
        what="measurable entities on this layout",
        size=len(found),
        limit=MAX_GEOMETRY_ROWS,
        hint=(
            "Narrow `geometry_layers`. A candidate set this large means every "
            "label is being compared against the whole drawing."
        ),
    )
    rings_without_length = sum(
        1
        for d in found.values()
        if d.get("ring_status") == "complete" and d.get("length") is None
    )
    rows = sorted(found.values(), key=lambda d: str(d.get("handle") or ""))
    return rows, rings_without_length


def _load_rings(
    drawing_id: str, layout: str | None, handles: Sequence[str]
) -> dict[str, list[list[float]]]:
    """The vertices of the rings that the prefilter actually needs.

    Fetched in a second pass, and only for the handles that survived, because a
    ring is the one field in this store that is large: loading every ring on a
    layout to answer a question about forty labels is the cost the prefilter
    exists to avoid.
    """
    wanted = list(dict.fromkeys(h for h in handles if h))
    if not wanted:
        return {}
    refuse_oversize(
        what="rings whose vertices must be loaded",
        size=len(wanted),
        limit=MAX_RINGS_LOADED,
        hint=(
            "Hand over a smaller `search_radius`, or narrow `geometry_layers`. "
            "Every ring loaded is held in memory whole for as long as the check "
            "runs."
        ),
    )
    out: dict[str, list[list[float]]] = {}
    for doc in _entities().find(
        {
            "drawing_id": drawing_id,
            "layout": layout,
            "handle": {"$in": wanted},
            "ring_status": "complete",
        },
        {"handle": 1, "ring": 1, "_id": 0},
    ):
        ring = doc.get("ring")
        if ring:
            out[str(doc.get("handle"))] = ring
    return out


def _run_dimension_truth(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    label_layers: Sequence[str] | None,
    geometry_layers: Sequence[str] | None,
    search_radius: float | None,
    tolerance: float | None,
    show_all: bool,
) -> dict[str, Any]:
    labels = _fetch_labels(drawing_id, layout, label_layers)
    geometry, rings_without_length = _fetch_geometry(
        drawing_id, layout, geometry_layers
    )

    # Which rings are worth loading is decided by the SAME selection and the
    # SAME prefilter the check itself will use — `dimension_labels` and
    # `radius_of` are shared with `check_labels` — so nothing is loaded that
    # could not have been reached, and nothing reachable is left unloaded.
    chosen = dimension_labels(labels)
    radii = [radius_of(row, search_radius) for row in chosen]
    grow = max([r for r, _ in radii if r is not None], default=0.0)
    grown = [
        None if b is None else (b[0] - grow, b[1] - grow, b[2] + grow, b[3] + grow)
        for b in (_box_of(g) for g in geometry)
    ]
    index = spatial._SpatialIndex(grown)

    needed: list[str] = []
    for row in chosen:
        point, _ = _label_point(row)
        if point is None:
            continue
        for i in index.hits(point[0], point[1]):
            if geometry[i].get("ring_status") == "complete":
                needed.append(str(geometry[i].get("handle")))

    rings = _load_rings(drawing_id, layout, needed)
    for doc in geometry:
        ring = rings.get(str(doc.get("handle")))
        if ring:
            doc["ring"] = ring

    body = check_labels(
        labels,
        geometry,
        drawing_id=drawing_id,
        layout=layout,
        units=units,
        search_radius=search_radius,
        tolerance=tolerance,
        show_all=bool(show_all),
    )
    body["label_layers"] = list(label_layers) if label_layers else None
    body["label_layers_note"] = (
        "no layer filter: what makes a text a dimension value is its CONTENT, "
        "and a layer name is one drawing's habit (G1)"
        if not label_layers
        else "restricted by the caller"
    )
    body["geometry_layers"] = list(geometry_layers) if geometry_layers else None
    body["rings_loaded"] = len(rings)
    body["rings_complete_without_length"] = rings_without_length
    body["rings_complete_without_length_note"] = (
        "closed rings whose length could not be computed — a bulged span, or a "
        "reading that failed. They are in the candidate set and can still offer "
        "an edge length; they are counted here so that a hole in the "
        "measurement is visible rather than silently absent (G8)."
    )
    return body


register(
    Recipe(
        name="dimension_truth",
        answers=(
            "whether the numbers this drawing PRINTS agree with the geometry "
            "it DRAWS"
        ),
        when_to_use=(
            "'do the dimension labels match what is drawn', 'is this drawing "
            "internally consistent', or before quoting a size that was read "
            "off a label rather than measured. Not for areas stated in a drawn "
            "table — that is `schedule_check`, which compares areas and has the "
            "schedule's own key to anchor its choice of polygon."
        ),
        params=(
            Param(
                "label_layers",
                "layers",
                "restrict the texts considered to these layers. Left empty, "
                "every text on the layout takes part — the CONTENT decides "
                "what is a dimension value, never the layer name (G1).",
            ),
            Param(
                "geometry_layers",
                "layers",
                "restrict the geometry a label may be tied to. Left empty, "
                "every measurable entity on the layout is a candidate, and a "
                "wide candidate set makes more labels ambiguous rather than "
                "fewer.",
            ),
            Param(
                "search_radius",
                "number",
                "how far from a label its geometry may sit, in DRAWING units. "
                "Left empty, it is derived PER LABEL as that label's own "
                "bounding-box diagonal, and the value used is published on "
                "every row.",
            ),
            Param(
                "tolerance",
                "number",
                "override the agreement band, in DRAWING units. Left empty, it "
                "is derived per row from the decimals the label prints and any "
                "± it states itself; the derivation is published on every row.",
            ),
            Param(
                "show_all",
                "flag",
                "list the agreeing rows too. By default only disagreements are "
                "listed; the counts always cover everything either way.",
                default=False,
            ),
        ),
        returns=(
            "labels_total",
            "labels_checked",
            "labels_agree",
            "labels_disagree",
            "labels_ambiguous",
            "labels_not_associated",
            "coverage_fraction",
            "rows[].printed",
            "rows[].measured",
            "rows[].difference",
            "rows[].text_handle",
            "rows[].geometry_handle",
            "rows[].tolerance",
            "ambiguous[].candidates",
        ),
        built_on=(
            "dossier_census.annotation_census",
            "dossier_census.classify_text",
            "store_spatial._SpatialIndex",
            "store_spatial._origin_of",
            "region.point_in_ring",
            "region._point_segment_distance",
        ),
        limits={
            "text_rows": MAX_TEXT_ROWS,
            "labels": MAX_LABELS,
            "geometry_rows": MAX_GEOMETRY_ROWS,
            "rings_loaded": MAX_RINGS_LOADED,
            "pairs_after_prefilter": PAIRS_BUDGET,
            "rows_returned": MAX_ROWS_RETURNED,
            "candidates_listed_per_label": MAX_CANDIDATES_LISTED,
        },
        run=_run_dimension_truth,
    )
)


# =============================================================================
# 2. drafting_hygiene — the checks a reviewer runs by habit
# =============================================================================


WHY_DEFAULT_LAYER: Final[str] = (
    "layer `0` is the DXF default. Content left on it cannot be frozen, "
    "coloured, or plotted as a group, so it is unmanageable by the means a "
    "drawing offers; and inside a block definition it takes on the properties "
    "of wherever the block is inserted, so the same geometry looks different in "
    "two places. Most drafting standards forbid it outright."
)

WHY_HIDDEN: Final[str] = (
    "content on a layer the layer table marks frozen or off is content that "
    "ships without being looked at. It is in the file, it is in the handover, "
    "and it is invisible on the sheet the reviewer approves — which is the one "
    "way a drawing can carry something nobody has ever seen."
)


def hidden_layers(drawing: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    """The layers this drawing's own table marks frozen or off. Pure.

    Returns `(rows, why_not_known)`. A drawing extracted before the layer table
    was stored returns `([], reason)` — and that is NOT the same answer as "no
    layer is hidden". Reporting the second when the first is true is exactly
    the "0 road parcels" defect in miniature (G8).
    """
    layers = drawing.get("layers")
    if layers is None:
        return [], (
            "this drawing was extracted before the layer table was stored "
            f"(needs ingest_version {tables.TABLES_INGEST_VERSION}); whether "
            "any layer is frozen or off is NOT KNOWN, which is not the same "
            "answer as none being hidden. Re-ingest it."
        )
    rows: list[dict[str, Any]] = []
    for entry in layers:
        frozen = bool(entry.get("frozen"))
        off = bool(entry.get("off"))
        viewports = list(entry.get("frozen_in_viewports") or [])
        if not (frozen or off or viewports):
            continue
        states = [s for s, on in (("frozen", frozen), ("off", off)) if on]
        if viewports:
            states.append(f"frozen in {len(viewports)} viewport(s)")
        rows.append(
            {
                "layer": entry.get("name"),
                "frozen": frozen,
                "off": off,
                "frozen_in_viewports": len(viewports),
                "states": states,
                "plot": entry.get("plot"),
                "entities_in_drawing": entry.get("entity_count"),
            }
        )
    rows.sort(key=lambda r: str(r["layer"] or ""))
    return rows, None


def text_scale_check(
    drawing: Mapping[str, Any], layout: str | None
) -> dict[str, Any]:
    """Text height against sheet scale — where it is derivable, and here it is not.

    The honest answer, given in full rather than approximated: **no entity in
    this store carries its text height.** `extract.EntityDoc` records the text
    STYLE name for a text entity and nothing about its size, so the height a
    reader would see on paper cannot be recovered from the store at all. What
    the file does hold is named below, so that the gap is a measured gap with a
    route out of it, not a shrug.

    Guessing from the bounding box was considered and refused: a box grows with
    the number of characters, wraps for MTEXT, and rotates with the text, so a
    height derived from it would be wrong by a factor nobody could see in the
    answer.
    """
    layouts = {
        str(l.get("name")): l for l in (drawing.get("layouts") or []) if l.get("name")
    }
    this = layouts.get(str(layout)) if layout else None
    page = (this or {}).get("page_setup") or {}
    header = drawing.get("header") or {}
    dim_styles = drawing.get("dim_styles")

    known: dict[str, Any] = {
        "layout": layout,
        "page_setup_available": bool(page.get("available")),
        "page_setup_why": page.get("why"),
        "paper_size_name": page.get("paper_size_name"),
        "paper_units": page.get("paper_units"),
        "plot_scale": page.get("plot_scale"),
        "plot_scale_text": page.get("plot_scale_text"),
        "header_dimscale": header.get("$DIMSCALE"),
        "header_luprec": header.get("$LUPREC"),
        "dim_style_scales": (
            None
            if dim_styles is None
            else [
                {"name": s.get("name"), "scale": s.get("scale")}
                for s in dim_styles
                if s.get("scale") is not None
            ]
        ),
    }
    return {
        "derivable": False,
        "why_not": (
            "text height is not stored. `extract.EntityDoc` records a text "
            "entity's STYLE name and its bounding box, and neither gives the "
            "drawn character height: a box grows with the number of characters, "
            "wraps for MTEXT, and rotates with the text. Deriving a height from "
            "it would be wrong by a factor that nothing in the answer would "
            "show."
        ),
        "what_is_known": known,
        "what_would_make_it_derivable": (
            "store the character height per text entity at ingest — `height` "
            "for TEXT/ATTRIB and `char_height` for MTEXT, both of which "
            "`app/export.py` already reads while rendering. With that field and "
            "the layout's plot scale above, the printed height follows "
            "arithmetically and this check becomes a real one."
        ),
        "not_a_pass": (
            "this is NOT a clean result for text height. Nothing was measured, "
            "so nothing passed; an unmeasurable check reported as zero findings "
            "is the defect this whole campaign exists to end (G8)."
        ),
    }


def hygiene_report(
    *,
    drawing_id: str,
    drawing: Mapping[str, Any],
    layout: str | None,
    units: Mapping[str, Any],
    default_layer: Mapping[str, Any],
    hidden: Sequence[Mapping[str, Any]],
    hidden_not_known: str | None,
    handle_cap: int = MAX_HANDLES_LISTED,
) -> dict[str, Any]:
    """Assemble the hygiene findings. Pure: it never touches MongoDB.

    `default_layer` and each row of `hidden` arrive already counted, in the
    shape `{"count", "by_type", "handles", "handles_omitted"}`, because the
    counting is one indexed query per layer and the judging is what is worth
    testing without a database.
    """
    hidden_rows = [dict(r) for r in hidden]
    hidden_with_content = [r for r in hidden_rows if int(r.get("count") or 0) > 0]
    hidden_total = sum(int(r.get("count") or 0) for r in hidden_with_content)
    on_default = int(default_layer.get("count") or 0)

    findings: list[dict[str, Any]] = [
        {
            "finding": "entities_on_default_layer",
            "layer": DEFAULT_LAYER,
            "what": (
                f"{on_default} entities sit on layer {DEFAULT_LAYER!r} in "
                f"layout {layout!r}"
            ),
            "count": on_default,
            "by_type": default_layer.get("by_type") or {},
            "handles": list(default_layer.get("handles") or [])[:handle_cap],
            "handles_omitted": int(default_layer.get("handles_omitted") or 0),
            "handle_cap": handle_cap,
            "why_it_matters": WHY_DEFAULT_LAYER,
            "clean": on_default == 0,
        },
        {
            "finding": "content_on_hidden_layers",
            "what": (
                f"{hidden_total} entities in layout {layout!r} sit on "
                f"{len(hidden_with_content)} layers this drawing's layer table "
                "marks frozen or off"
                if hidden_not_known is None
                else "not known"
            ),
            "count": None if hidden_not_known is not None else hidden_total,
            "count_withheld_why": hidden_not_known,
            "layers": [
                {
                    "layer": r.get("layer"),
                    "states": r.get("states"),
                    "plot": r.get("plot"),
                    "entities_in_this_layout": r.get("count"),
                    "entities_in_drawing": r.get("entities_in_drawing"),
                    "by_type": r.get("by_type") or {},
                    "handles": list(r.get("handles") or [])[:handle_cap],
                    "handles_omitted": int(r.get("handles_omitted") or 0),
                }
                for r in hidden_with_content
            ],
            "layers_hidden_but_empty_here": [
                r.get("layer")
                for r in hidden_rows
                if int(r.get("count") or 0) == 0
            ],
            "handle_cap": handle_cap,
            "why_it_matters": WHY_HIDDEN,
            "clean": None if hidden_not_known is not None else hidden_total == 0,
        },
        {
            "finding": "text_height_vs_sheet_scale",
            "what": "not derivable from this store",
            "count": None,
            "count_withheld_why": "text height is not stored per entity",
            "detail": text_scale_check(drawing, layout),
            "why_it_matters": (
                "text that is legible in model space can be unreadable at the "
                "sheet's plot scale, and a drawing whose notes cannot be read "
                "is a drawing whose notes were not read."
            ),
            "clean": None,
        },
    ]

    scope_note = (
        f"layout {layout!r}; the drafting checks a reviewer runs by habit — "
        f"content on the DXF default layer {DEFAULT_LAYER!r}, content on layers "
        "the drawing's own layer table marks frozen or off, and text height "
        "against sheet scale. Counts are per LAYOUT; the layer table's own "
        "`entity_count` covers the whole drawing including block definitions, "
        "and the two are reported side by side rather than mixed"
    )
    scope = scope_for(layout=layout, units=units, note=scope_note)

    observations: list[ev.Observation] = []
    if on_default:
        observations.append(
            _observation(
                ev.Origin.LAYER_NAME,
                f"{on_default} entities in layout {layout!r} carry layer "
                f"{DEFAULT_LAYER!r}",
                observed=DEFAULT_LAYER,
                locator=DEFAULT_LAYER,
                layout=layout,
            )
        )
    for row in hidden_with_content[:4]:
        observations.append(
            _observation(
                ev.Origin.LAYER_NAME,
                f"the layer table marks {row.get('layer')!r} "
                f"{', '.join(row.get('states') or [])}, and {row.get('count')} "
                f"entities in layout {layout!r} sit on it",
                observed=str(row.get("layer")),
                locator=str(row.get("layer")),
                layout=layout,
            )
        )
    observations.append(
        _observation(
            ev.Origin.ABSENCE,
            "text height against sheet scale could not be checked at all: no "
            "entity in this store carries its text height",
            locator=str(layout),
            layout=layout,
        )
    )

    raised = sum(1 for f in findings if f.get("clean") is False)
    evidence = _computed_evidence(
        (
            f"{raised} of the 3 drafting-hygiene checks raise a finding on "
            f"layout {layout!r}: {on_default} entities on layer "
            f"{DEFAULT_LAYER!r} and {hidden_total} on frozen or off layers"
        ),
        drawing_id=drawing_id,
        scope=scope,
        observations=observations,
        not_established=(
            "whether any of this is INTENDED. A construction layer parked on "
            "`0`, a reference underlay switched off on purpose, and a mistake "
            "look identical from inside the file; this reports what is there, "
            "not what it means. One of the three checks — text height against "
            "sheet scale — was not run at all, because the height is not "
            "stored, and it is reported as unmeasured rather than as clean."
        ),
        how_to_verify=(
            "open the handles listed here in the drawing. For the hidden "
            "layers, thaw or switch each one on and look at what appears: "
            "content that should have shipped is a drafting slip, content that "
            "should not have shipped is a handover problem."
        ),
    )

    body: dict[str, Any] = {
        "scope_note": scope_note,
        "findings": findings,
        "checks_run": 2,
        "checks_not_run": 1,
        "checks_total": 3,
        "checks_not_run_note": (
            "text height against sheet scale was NOT run — the height is not "
            "stored per entity. It is reported as unmeasured, and an "
            "unmeasurable check is never counted as a pass (G8)."
        ),
        "findings_raised": raised,
        "entities_on_default_layer": on_default,
        "entities_on_hidden_layers": (
            None if hidden_not_known is not None else hidden_total
        ),
        "hidden_layers_total": len(hidden_rows),
        "hidden_layers_with_content_here": len(hidden_with_content),
        "layer_table_available": hidden_not_known is None,
        "layer_table_why_not": hidden_not_known,
        "handle_cap": handle_cap,
        "handle_cap_note": (
            f"counts cover everything; only the handle listings stop at "
            f"{handle_cap}, and every one of them says how many it left out (G7)."
        ),
    }
    return ev.attach(body, evidence)


def _layer_facts(
    drawing_id: str, layout: str | None, layer: str, handle_cap: int
) -> dict[str, Any]:
    """Count, types and a capped handle list for one layer on one layout.

    `count_documents` first and a capped `find` second, so a layer holding
    35,841 entities costs one count and twenty documents rather than 35,841.
    """
    query = {"drawing_id": drawing_id, "layout": layout, "layer": layer}
    total = _entities().count_documents(dict(query))
    handles: list[str] = []
    by_type: dict[str, int] = {}
    for doc in (
        _entities()
        .find(dict(query), {"handle": 1, "type": 1, "_id": 0})
        .sort([("handle", 1)])
        .limit(handle_cap)
    ):
        handles.append(str(doc.get("handle")))
        kind = str(doc.get("type") or "(type missing)")
        by_type[kind] = by_type.get(kind, 0) + 1
    return {
        "layer": layer,
        "count": total,
        "by_type": dict(sorted(by_type.items())),
        "by_type_note": (
            "counted over the handles listed, not over the whole layer: the "
            "listing is capped and the type mix travels with it"
            if total > len(handles)
            else "counted over every entity on this layer in this layout"
        ),
        "handles": handles,
        "handles_omitted": max(0, total - len(handles)),
    }


def _run_drafting_hygiene(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    handle_cap: float | None,
) -> dict[str, Any]:
    cap = MAX_HANDLES_LISTED if handle_cap is None else int(handle_cap)
    if cap < 1:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"handle_cap has the value {handle_cap!r}; it must be at least 1.",
            "A finding with no handle at all cannot be opened by anyone, which "
            "is the one thing this recipe exists to make possible.",
        )
    # Stated, and it really binds (G7): asking for more is refused with a
    # suggestion rather than quietly served a shorter list.
    refuse_oversize(
        what="handles listed per finding",
        size=cap,
        limit=MAX_HANDLES_LISTED,
        hint=(
            "This listing exists so a person can click a few examples, not so "
            "a whole layer can be exported. The COUNT already covers "
            "everything; to see every handle on a layer, ask query_entities "
            "for that layer directly."
        ),
    )

    drawing = coll(COLL_DRAWINGS).find_one({"_id": drawing_id})
    if drawing is None:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"there is no drawing with id {drawing_id!r}.",
            "Check the drawing_id. A misspelled id returns a clean report that "
            "cannot be told apart from a genuinely clean drawing.",
        )

    default_layer = _layer_facts(drawing_id, layout, DEFAULT_LAYER, cap)
    rows, not_known = hidden_layers(drawing)
    hidden: list[dict[str, Any]] = []
    for row in rows:
        facts = _layer_facts(drawing_id, layout, str(row["layer"]), cap)
        hidden.append({**row, **facts})

    return hygiene_report(
        drawing_id=drawing_id,
        drawing=drawing,
        layout=layout,
        units=units,
        default_layer=default_layer,
        hidden=hidden,
        hidden_not_known=not_known,
        handle_cap=cap,
    )


register(
    Recipe(
        name="drafting_hygiene",
        answers=(
            "the drafting checks a reviewer runs by habit: content on layer "
            "`0`, content hidden on frozen or off layers, and text height "
            "against sheet scale"
        ),
        when_to_use=(
            "'is anything drawn on layer 0', 'is there hidden content in this "
            "file', 'will this pass a drafting review'. Cheap, and worth "
            "running before any question about what the drawing contains — "
            "content on a switched-off layer is content nobody has looked at."
        ),
        params=(
            Param(
                "handle_cap",
                "number",
                "how many handles are listed per finding, up to "
                f"{MAX_HANDLES_LISTED}. The counts always cover everything; "
                "only the listing is cut, and it says how much it left out.",
                default=MAX_HANDLES_LISTED,
            ),
        ),
        returns=(
            "findings[].finding",
            "findings[].count",
            "findings[].handles",
            "findings[].why_it_matters",
            "entities_on_default_layer",
            "entities_on_hidden_layers",
            "checks_not_run",
            "layer_table_available",
        ),
        built_on=(
            "store_tables.TABLES_INGEST_VERSION",
            "autocad_entities.count_documents",
            "autocad_drawings.find_one",
        ),
        limits={
            "handles_per_finding": MAX_HANDLES_LISTED,
            "queries_per_hidden_layer": 2,
        },
        run=_run_drafting_hygiene,
    )
)
