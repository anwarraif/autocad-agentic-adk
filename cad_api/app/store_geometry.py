"""find_duplicates + shape cross-check

Owner: subagent E — UPLIFT-07.

This file is deliberately split out of `store.py` so that two specs worked on
at the same time never touch the same line. The rules, and their reasons:

- **Do not import from `.store`.** `store.py` imports this module at its
  bottom; importing back would be circular. Take `coll` and friends straight
  from `.mongo`, and geometry from `.region`.
- **Do not add lines to `store.py` or `main.py`.** Both are shared property,
  and every line added there is one merge conflict. Functions here are reached
  as `store.store_geometry.<function>`.
- **Every response that carries MEANING must carry `evidence`** per
  `docs/UPLIFT-08-EVIDENCE.md` — use the helpers in `app/evidence.py`, do not
  assemble the dict yourself.
- **Every number carries its unit** (G2), and the unit is allowed to be
  absent. A withheld number becomes `None` + a reason, never `0`.

The `docs/UPLIFT-GENERALITY-RULES.md` G1-G10 checklist is run before every
commit in this file.

---

Four decisions that determine the shape of this entire file.

**One, data retrieval and computation are fully separated.** Every public
function here is two lines: one loads rows from Mongo, one calls a pure
`build_*`. The reason is not tidiness. The numbers this spec demands — 52
groups, 2,312 passing — are properties of the data that has been ingested, and
a test that fakes the Mongo driver will pass while the grouping is wrong. What
can be tested without a database is the arithmetic; what cannot be, is not
pretended to be.

**Two, `units` is handed over by the caller and has no default.**
`_unit_names` lives in `store.py`, and this file must not import it. Copying it
here would mean two places decide whether a sheet has units, and the second
would fall behind. So the pattern used is exactly the `evidence.Scope`
pattern: units pass as a parameter, as they come from
`store._unit_names(drawing, layout)`, and a response cannot be issued without
them.

**Three, `shape_key` is never compared across drawings.** It is
scale-sensitive but unit-blind: a 12x25 metre plot and a 12x25 inch component
produce exactly the same key. Every query here carries `drawing_id` as its
filter prefix — which is also a technical necessity, because this cluster runs
with `notablescan` and a query without an indexed plan FAILS, it does not slow
down.

**Four, this tool reports and does not judge.** A geometry duplicate is not a
defect; it can be a working copy, an overlay, or a deliberately repeated
shape. That is why `interpretation_note` is mandatory in every response,
`kind="shape"` carries `defect: false` explicitly, and the word "defect" never
appears in any output.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import evidence as ev
from . import region
from .mongo import COLL_ENTITIES, coll

#: Three kinds, deliberately kept apart because they answer three different
#: questions. Folding `shape` into `geometry` would report every plot from one
#: typology module as a duplicate — hundreds of false findings drowning out
#: dozens of real ones.
KINDS: tuple[str, ...] = ("geometry", "shape", "label")

#: Default tolerance, in DRAWING UNITS — not metres. The reason is the same as
#: for `geometry.SHAPE_KEY_QUANTUM`: a drawing in inches uses the same number
#: and therefore a stricter tolerance, so it fails in the safe direction. The
#: value is echoed in the response together with its unit, never promised by
#: the parameter's name.
DEFAULT_TOLERANCE = 0.01

#: Limit on targets loaded at once (G7). The largest polygon count in a single
#: layout across the 16 drawings that exist is 3,148; this limit is six times
#: that, and exceeding it yields a refusal with advice on narrowing — not a
#: silent truncation, which is a wrong answer with a right face.
MAX_TARGET_POLYGONS = 20_000

#: Limit on labels tested against polygons in `kind="label"` (G7).
MAX_LABELS = 60_000

#: Ring projection: only the fields actually used. A full `ring` per document
#: turns 3,148 rows into tens of MB without a single one being read — in the
#: `geometry` and `shape` kinds what gets used is `shape_key` and
#: `polygon_centroid`.
_RING_PROJECTION = {
    "handle": 1,
    "layer": 1,
    "layout": 1,
    "type": 1,
    "ring_status": 1,
    "shape_key": 1,
    "shape_key_basis": 1,
    "polygon_centroid": 1,
    "edge_lengths": 1,
    "area": 1,
    "bbox": 1,
}

#: `kind="label"` needs the ring itself for the point-in-polygon test.
_RING_PROJECTION_WITH_RING = {**_RING_PROJECTION, "ring": 1}

_LABEL_PROJECTION = {
    "handle": 1,
    "layer": 1,
    "layout": 1,
    "type": 1,
    "text": 1,
    "anchor_point": 1,
    "anchor_basis": 1,
}


class DuplicateRefused(Exception):
    """A request that could only be answered misleadingly.

    Its shape mirrors `store.MeasureRefused` (`code`/`message`/`hint`) so that
    `main.py` can reuse the existing `_refusal_handler`, and a refusal comes
    out as an actionable 400 — not a 500 that can only be stared at. It does
    not inherit from `MeasureRefused` precisely because this file must not
    import `store`; registering its handler is requested in
    `docs/PROGRESS.md`.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


# --- Ring census -------------------------------------------------------------


#: Every `ring_status` value other than `complete`, and the name of the field
#: that reports it. Written out in full so that a new status in `extract.py`
#: falls into a visible `skipped_other` rather than disappearing from the sum.
_SKIP_FIELDS: Mapping[str, str] = {
    "bulge": "skipped_bulge",
    "open": "skipped_open",
    "non_planar": "skipped_non_planar",
    "oversize": "skipped_oversize",
    "degenerate": "skipped_degenerate",
    "unreadable": "skipped_unreadable",
}


@dataclass(frozen=True, slots=True)
class RingCensus:
    """How many polygons can be targets, and how many were skipped per cause.

    A skipped target is not a target that did not match, and the difference
    between the two is the difference between "there are no duplicates here"
    and "1,298 polygons were never examined". That is why every key is present
    even when it is zero: an absent key reads as "not applicable", a zero reads
    as "examined, none found".
    """

    complete: int
    skipped: Mapping[str, int]

    @classmethod
    def from_counts(cls, counts: Mapping[str, int]) -> "RingCensus":
        skipped = {field: 0 for field in _SKIP_FIELDS.values()}
        skipped["skipped_other"] = 0
        for status, n in counts.items():
            if status == "complete":
                continue
            skipped[_SKIP_FIELDS.get(status, "skipped_other")] += int(n)
        return cls(complete=int(counts.get("complete", 0)), skipped=skipped)

    def as_dict(self) -> dict[str, int]:
        return dict(self.skipped)


# --- Validation --------------------------------------------------------------


def _require_layout(layout: str | None) -> str:
    if not isinstance(layout, str) or not layout.strip():
        raise DuplicateRefused(
            "LAYOUT_REQUIRED",
            "find_duplicates was called without a layout.",
            "Name the layout. Without it, model space geometry and sheet "
            "geometry that happen to share coordinates enter the same "
            "comparison, and two things that never meet get reported as "
            "coincident.",
        )
    return layout


def _require_kind(kind: str) -> str:
    if kind not in KINDS:
        raise DuplicateRefused(
            "UNKNOWN_KIND",
            f"duplicate kind {kind!r} is not recognised.",
            "Choose one of: " + ", ".join(KINDS) + ". The three answer "
            "different questions and are deliberately not merged.",
        )
    return kind


def _require_tolerance(tolerance: float) -> float:
    try:
        value = float(tolerance)
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value <= 0:
        raise DuplicateRefused(
            "TOLERANCE_NOT_POSITIVE",
            f"tolerance {tolerance!r} is not a positive number.",
            "Give a positive value in drawing units. A zero tolerance demands "
            "exact float equality over shoelace-derived centroids, which is "
            "not an answerable question; if you meant 'very strict', "
            f"use {DEFAULT_TOLERANCE}.",
        )
    return value


def refuse_if_oversize(total: int, *, layout: str) -> None:
    """The size limit is stated, not assumed (G7)."""
    if total > MAX_TARGET_POLYGONS:
        raise DuplicateRefused(
            "TOO_MANY_POLYGONS",
            f"{total} complete polygons in layout {layout!r}, above the limit "
            f"of {MAX_TARGET_POLYGONS}.",
            "Narrow it with `layers`, or run it per layer. This limit is "
            "stated so that the answer is not truncated silently: a duplicate "
            "list that has lost part of its contents reads exactly like a "
            "drawing that genuinely has no duplicates there.",
        )


# --- Units (G2) --------------------------------------------------------------


def _tolerance_units(units: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """The tolerance's unit name, or None together with the reason.

    Never guesses. 11 of the 16 drawings are in inches and 3 state no unit at
    all; a response that writes "m" because the drawing currently open happens
    to use metres is a wrong number wearing the clothes of a right one.
    """
    length = units.get("length_unit")
    if length:
        return length, None
    reason = units.get("why_no_unit") or (
        "this drawing does not state its unit, so the tolerance and distances "
        "in this response are unnamed drawing units. The numbers can still be "
        "compared with one another inside this drawing, and cannot be compared "
        "with another drawing."
    )
    return None, reason


# --- Pure grouping -----------------------------------------------------------


def _by_shape_key(rows: Sequence[Mapping[str, Any]]) -> tuple[
    dict[str, list[Mapping[str, Any]]], int
]:
    """Group by `shape_key`; also return how many have none.

    A ring without a `shape_key` is **skipped and counted**, not treated as
    unique. Treating it as unique would report "no duplicates" for polygons
    that were never examined at all.
    """
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    without = 0
    for row in rows:
        key = row.get("shape_key")
        if not key:
            without += 1
            continue
        groups[key].append(row)
    return dict(groups), without


def _centroid(row: Mapping[str, Any]) -> tuple[float, float] | None:
    c = row.get("polygon_centroid")
    if not c or len(c) < 2:
        return None
    try:
        return (float(c[0]), float(c[1]))
    except (TypeError, ValueError):
        return None


class _Union:
    """Union-find. A duplicate group is a CONNECTED component, not a pair:
    three stacked copies are one finding, not three."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, a: int) -> int:
        while self._parent[a] != a:
            self._parent[a] = self._parent[self._parent[a]]
            a = self._parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _cluster_coincident(
    rows: Sequence[Mapping[str, Any]], tolerance: float
) -> list[list[Mapping[str, Any]]]:
    """Group rows with the same `shape_key` that also sit in the same place.

    Two conditions, both from the spec: `polygon_centroid` distance <=
    tolerance AND area difference <= tolerance squared. The second looks
    redundant — two true copies have areas identical bit for bit — and it is:
    it is not a similarity test but a hash-collision guard. `shape_key` is a
    SHA-1 truncated to 16 hex digits; two different shapes that happen to meet
    at the same key and happen to sit close together would pass without this
    guard, and nothing in the response would show that it had happened.

    Pair search goes through tolerance-sized buckets rather than an
    all-against-all comparison: one plot module can have hundreds of members,
    and O(n^2) there is waste that grows precisely in the tidiest drawings.
    """
    if len(rows) < 2:
        return []

    points: list[tuple[float, float] | None] = [_centroid(r) for r in rows]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, p in enumerate(points):
        if p is None:
            continue
        buckets[(math.floor(p[0] / tolerance), math.floor(p[1] / tolerance))].append(i)

    area_tol = tolerance * tolerance
    uf = _Union(len(rows))
    joined = False
    for (bx, by), members in buckets.items():
        neighbours: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbours.extend(buckets.get((bx + dx, by + dy), ()))
        for i in members:
            pi = points[i]
            ai = rows[i].get("area")
            for j in neighbours:
                if j <= i:
                    continue
                pj = points[j]
                if pi is None or pj is None:
                    continue
                if math.dist(pi, pj) > tolerance:
                    continue
                aj = rows[j].get("area")
                if ai is not None and aj is not None:
                    if abs(float(ai) - float(aj)) > area_tol:
                        continue
                uf.union(i, j)
                joined = True

    if not joined:
        return []
    out: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for i, row in enumerate(rows):
        if points[i] is None:
            continue
        out[uf.find(i)].append(row)
    return [members for members in out.values() if len(members) > 1]


def _max_gap(members: Sequence[Mapping[str, Any]]) -> float | None:
    """The LARGEST centroid distance inside the group, not the first pair's.

    A group is a connected component, so A-B and B-C each below the tolerance
    can produce an A-C above it. Reporting the first pair hides that this group
    is wider than its tolerance.
    """
    pts = [p for p in (_centroid(m) for m in members) if p is not None]
    if len(pts) < 2:
        return None
    return max(
        math.dist(pts[i], pts[j])
        for i in range(len(pts))
        for j in range(i + 1, len(pts))
    )


def _modal_edge_lengths(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """The edge lengths that occur most often in this shape group.

    The mode and not the mean: a mean over two plot modules that happen to
    share a group produces a number no polygon actually has.
    """
    counter: Counter[tuple[float, ...]] = Counter()
    for row in rows:
        lens = row.get("edge_lengths")
        if not lens:
            continue
        counter[tuple(round(float(v), 6) for v in lens)] += 1
    if not counter:
        return []
    return list(counter.most_common(1)[0][0])


def _member(row: Mapping[str, Any], *, area_unit: str | None) -> dict[str, Any]:
    return {
        "handle": row.get("handle"),
        "layer": row.get("layer"),
        "area": row.get("area"),
        "area_units": area_unit if row.get("area") is not None else None,
    }


def _layer_counts(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter(str(r.get("layer")) for r in rows)
    return [
        {"layer": name, "count": n}
        for name, n in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


# --- Evidence ----------------------------------------------------------------


def _scope(layout: str, units: Mapping[str, Any], note: str) -> ev.Scope:
    return ev.Scope(
        layout=layout,
        # Block definitions are never included: the coordinates inside them are
        # rescaled by each INSERT, so two shapes that are "identical" there are
        # not necessarily identical anywhere this drawing is actually drawn.
        includes_block_definitions=False,
        units=units,
        note=note,
    )


#: The provenance of every claim in this file. `Origin.GEOMETRY` falls into
#: `Corpus.COORDINATES`, which is in `NEVER_SPEAKS` — so any evidence based on
#: shape alone will never rise above `inferred`, however many polygons agree.
#: That is not a limitation to be worked around; it is the right answer.
#: Coordinates infer; they do not state.
_GEOMETRY_PROVENANCE = ev.Provenance(
    config_layer=ev.ConfigLayer.NO_CONFIG,
    note=(
        "this finding uses no land-use config and does not use layer names as "
        "its basis; it is computed from coordinates alone"
    ),
)


def _geometry_evidence(
    *,
    claim_value: str,
    drawing_id: str,
    scope: ev.Scope,
    observations: Sequence[ev.Observation],
    contradictions: Sequence[ev.Contradiction] = (),
    not_established: str | None = None,
    how_to_verify: str | None = None,
) -> ev.Evidence:
    return ev.Evidence.of(
        # Tokens are empty, and deliberately so: there is no word inside the
        # file that could be searched for to prove the drawing STATES that two
        # polygons are copies. This claim is inferred from coordinates and will
        # never be more than that.
        ev.Claim(value=claim_value, tokens=()),
        drawing_id=drawing_id,
        scope=scope,
        provenance=_GEOMETRY_PROVENANCE,
        observations=observations,
        contradictions=contradictions,
        not_established=not_established,
        how_to_verify=how_to_verify,
    )


# --- Interpretation notes ----------------------------------------------------

#: Mandatory in every response. A tool that reports "52 duplicates" without
#: this sentence will be read as "52 defects", and there is nothing in the data
#: that supports that reading.
_INTERPRETATION: Mapping[str, str] = {
    "geometry": (
        "A geometry duplicate does NOT mean the data is broken. It can be a "
        "working copy, an overlay, or a repeated shape that was fully "
        "intended. This tool reports; the judgement belongs to the reader."
    ),
    "shape": (
        "A repeated shape is NOT a duplicate and is not a defect. One plot "
        "module used hundreds of times produces one large group here, and that "
        "is in fact the sign of a consistent drawing. What answers 'is there "
        "stacked geometry' is kind=geometry."
    ),
    "label": (
        "Repeated text inside one polygon is NOT automatically wrong. It can "
        "be a plot number deliberately repeated at two corners, or annotations "
        "that happen to read the same. This tool reports; the judgement "
        "belongs to the reader."
    ),
}


# --- Response: shared envelope ----------------------------------------------


def _envelope(
    *,
    drawing_id: str,
    layout: str,
    kind: str,
    tolerance: float | None,
    units: Mapping[str, Any],
    ring_census: RingCensus,
    layers: Sequence[str] | None,
    limit: int,
) -> dict[str, Any]:
    unit_name, unit_reason = _tolerance_units(units)
    area_unit = units.get("area_unit")
    body: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "kind": kind,
        "layers": list(layers) if layers else None,
        "limit": limit,
        "targets_considered": ring_census.complete,
        "skipped": dict(ring_census.as_dict()),
        "interpretation_note": _INTERPRETATION[kind],
    }
    if tolerance is not None:
        body["tolerance"] = tolerance
        body["tolerance_units"] = unit_name
        body["tolerance_units_note"] = unit_reason
        body["area_tolerance"] = tolerance * tolerance
        body["area_tolerance_units"] = area_unit
    return body


def _scope_sentence(
    layout: str,
    units: Mapping[str, Any],
    kind: str,
    tolerance: float | None,
    layers: Sequence[str] | None,
) -> str:
    unit_name, _ = _tolerance_units(units)
    unit_text = unit_name or "unstated drawing units"
    bits = [
        f"layout {layout!r}",
        "without block definitions",
        f"kind {kind}",
        f"only ring_status=complete polygons, measured in {unit_text}",
    ]
    if tolerance is not None:
        bits.append(f"tolerance {tolerance:g} {unit_text}")
    if layers:
        bits.append("layers " + ", ".join(layers[:6]) + ("…" if len(layers) > 6 else ""))
    return "; ".join(bits) + "."


# --- Response: find_duplicates -----------------------------------------------


def build_duplicates(
    *,
    drawing_id: str,
    layout: str | None,
    rows: Sequence[Mapping[str, Any]],
    ring_census: RingCensus,
    kind: str = "geometry",
    tolerance: float = DEFAULT_TOLERANCE,
    units: Mapping[str, Any],
    layers: Sequence[str] | None = None,
    limit: int = 50,
    label_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The whole `find_duplicates` computation, without one database call.

    Split out so that the shape of the response — the tolerance echo, the
    interpretation note, the census of what was skipped, the evidence contract
    — can be tested on a machine that has no Mongo, and so that the acceptance
    numbers that genuinely need live data do not get mixed together with the
    arithmetic that does not.
    """
    layout = _require_layout(layout)
    kind = _require_kind(kind)
    tolerance = _require_tolerance(tolerance)

    body = _envelope(
        drawing_id=drawing_id, layout=layout, kind=kind, tolerance=tolerance,
        units=units, ring_census=ring_census, layers=layers, limit=limit,
    )
    body["scope_note"] = _scope_sentence(layout, units, kind, tolerance, layers)
    area_unit = units.get("area_unit")

    if kind == "label":
        groups, extra_skipped = _label_groups(rows, label_rows or [], area_unit)
        claim = "repeated text inside a single polygon"
        body["defect"] = None
    else:
        by_key, without_key = _by_shape_key(rows)
        extra_skipped = {"skipped_no_shape_key": without_key}
        if kind == "shape":
            groups = _shape_groups(by_key, area_unit)
            claim = "identical shapes in different places"
            # Stated, not implied. A reader who sees "449 members" without
            # this line will read it as 449 problems.
            body["defect"] = False
        else:
            groups, no_centroid = _geometry_groups(
                by_key, tolerance, area_unit, body["tolerance_units"]
            )
            extra_skipped["skipped_no_centroid"] = no_centroid
            claim = "polygons stacked on one another with the same shape"
            body["defect"] = None

    body["skipped"].update(extra_skipped)

    # The neighbouring question, named, because the two are confusable by
    # wording and not by result.
    #
    # Measured: asked for "the total parcel area minus every square metre that
    # is claimed twice", the agent called this recipe eight times, layer by
    # layer, and subtracted 30,397.747 m2 of DUPLICATE COPIES. The ground
    # actually claimed by two different parcels is 29,241.258 m2, which
    # `overlap_scan` returns in one call. Both numbers are real, both are
    # about "twice", and the answer was well formed, cited its sources, and
    # was wrong -- nothing downstream could tell, because a wrong recipe
    # cleanly answered the question it was asked.
    #
    # So the distinction travels in the response rather than in an
    # instruction. It is a fact about what this recipe measures, true of every
    # drawing, and it is read at the moment the choice is being made.
    body["measures"] = (
        "the same shape stored more than once -- copies. Two entities are a "
        "group here because their geometry MATCHES, whatever they represent"
    )
    body["does_not_measure"] = (
        "ground claimed by two DIFFERENT parcels. Two neighbouring plots that "
        "cover the same square metres are not copies of each other and will "
        "not appear here, however much they overlap"
    )
    body["ask_instead"] = {
        "recipe": "overlap_scan",
        "when": (
            "the question is about double-counted GROUND -- area claimed "
            "twice, totals inflated by overlap, land counted in two land uses "
            "-- rather than about repeated geometry"
        ),
        "why_it_matters": (
            "subtracting this recipe's total from a land-use figure removes "
            "copies, not shared ground, and the two are different numbers on "
            "the same drawing"
        ),
    }

    total = len(groups)
    shown = groups[:limit]
    body["groups_total"] = total
    body["entities_involved"] = sum(g["member_count"] for g in groups)
    body["truncated"] = total > len(shown)
    body["groups"] = shown

    ev.attach(body, _duplicates_evidence(
        drawing_id=drawing_id, layout=layout, units=units, kind=kind,
        tolerance=tolerance, groups=groups, ring_census=ring_census,
        claim_value=claim, layers=layers,
    ))
    _attach_blind_spot(body, drawing_id=drawing_id, layout=layout)
    return body


#: How many layers the blind-spot block names before it truncates (G7).
MAX_BLIND_SPOT_LAYERS = 5


def _attach_blind_spot(
    body: dict[str, Any], *, drawing_id: str, layout: str | None
) -> None:
    """What this instrument cannot see, said out loud in its own answer.

    This function matches shapes by `shape_key`, and `shape_key` is null on
    OPEN geometry -- a LINE, an ARC, an unclosed polyline. So a layer holding
    the same road network drawn three times is, to this computation, perfectly
    clean. It does not report a false negative; it reports a true answer to a
    narrower question than the one that was asked.

    Measured: asked "is anything in this drawing drawn twice?", the agent
    called this tool and answered "62 groups, 125 entities, largest group 3
    polygons" -- literally correct, and it missed 1,233 entities of centreline
    drawn three times, the largest duplication in the file. The Dossier had
    recorded it since Phase 1. Nothing connected the two.

    So the finding this instrument is blind to travels WITH its answer, the
    same way `land_use_summary` carries a residual: an answer that cannot see
    something should say what it cannot see, next to what it can.
    """
    try:  # pragma: no cover - the reader is optional by design
        from . import dossier_read
    except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
        body["blind_spot"] = {
            "status": "not_computed",
            "why": f"the Dossier reader is unavailable ({type(exc).__name__})",
            "means": "this answer covers closed rings only, and nothing "
                     "checked open geometry",
        }
        return

    try:
        found = dossier_read.duplicate_scan(drawing_id, layout=layout)
    except AttributeError:
        found = None
    except Exception as exc:  # noqa: BLE001
        body["blind_spot"] = {
            "status": "not_computed",
            "why": f"{type(exc).__name__}: {exc}",
            "means": "this answer covers closed rings only",
        }
        return

    covers = (
        "this computation matches shapes by `shape_key`, which is null on open "
        "geometry (LINE, ARC, unclosed polyline). Its groups therefore describe "
        "CLOSED RINGS only."
    )
    if not found:
        body["blind_spot"] = {
            "status": "not_computed",
            "covers": covers,
            "means": (
                "no Dossier has been built for this drawing, so open geometry "
                "was NOT checked by anything. That is not the same answer as "
                "'no duplicates there' -- build it with "
                "`python scripts/dossier_backfill.py --drawing <id>`"
            ),
        }
        return

    layers = found.get("layers") or []
    body["blind_spot"] = {
        "status": "checked" if layers else "checked_clean",
        "covers": covers,
        "dossier_duplicate_layers": layers[:MAX_BLIND_SPOT_LAYERS],
        "dossier_duplicate_layers_total": len(layers),
        "truncated": len(layers) > MAX_BLIND_SPOT_LAYERS,
        "means": (
            "the Dossier examined open geometry separately and found "
            f"{len(layers)} layer(s) holding detached identical copies. Those "
            "copies are NOT in the groups above and a total measured over such "
            "a layer is multiplied by the number of copies."
            if layers
            else "the Dossier examined open geometry separately and found no "
                 "detached identical copies on this layout."
        ),
    }


def _geometry_groups(
    by_key: Mapping[str, Sequence[Mapping[str, Any]]],
    tolerance: float,
    area_unit: str | None,
    length_unit: str | None,
) -> tuple[list[dict[str, Any]], int]:
    groups: list[dict[str, Any]] = []
    no_centroid = 0
    for key, rows in by_key.items():
        missing = sum(1 for r in rows if _centroid(r) is None)
        no_centroid += missing
        for members in _cluster_coincident(list(rows), tolerance):
            gap = _max_gap(members)
            layers = sorted({str(m.get("layer")) for m in members})
            groups.append({
                "shape_key": key,
                "member_count": len(members),
                "members": [_member(m, area_unit=area_unit) for m in members],
                "centroid_gap": gap,
                "centroid_gap_units": None if gap is None else length_unit,
                "layers": layers,
                # A group wider than its tolerance is a chain, not a stack,
                # and its reader has a right to know the difference.
                "chained": gap is not None and gap > tolerance,
                "note": (
                    "copies across layers: " + " / ".join(layers)
                    if len(layers) > 1
                    else "copies within a single layer: " + layers[0]
                ),
            })
    groups.sort(key=lambda g: (-g["member_count"], g["shape_key"]))
    return groups, no_centroid


def _shape_groups(
    by_key: Mapping[str, Sequence[Mapping[str, Any]]],
    area_unit: str | None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for key, rows in by_key.items():
        if len(rows) < 2:
            continue
        groups.append({
            "shape_key": key,
            "member_count": len(rows),
            "members": [_member(m, area_unit=area_unit) for m in rows[:20]],
            "members_truncated": len(rows) > 20,
            "modal_edge_lengths": _modal_edge_lengths(rows),
            "layers": _layer_counts(rows),
        })
    groups.sort(key=lambda g: (-g["member_count"], g["shape_key"]))
    return groups


class _BboxGrid:
    """A single-use spatial index over polygon bounding boxes.

    It exists because a stated limit has to actually bind (G7). Without it,
    `kind="label"` tests every label against every polygon: with the limits
    that apply in this file, 20,000 polygons x 60,000 labels = 1.2 billion box
    comparisons. That number is forbidden by no written limit, and a limit that
    does not bind is a limit that is not there yet.

    The cell size is derived from the data — the mean bounding-box side —
    rather than from a constant, because a drawing in inches and a drawing in
    metres share no scale at all (G1, G2). A polygon whose box crosses more
    than `_MAX_CELLS_PER_BOX` cells is not put into any cell but into the
    `_wide` list, which is checked for every label: a single polygon as large
    as the whole drawing would fill every cell and turn this index into pure
    overhead.
    """

    #: How many cells one box may touch before it counts as wide.
    _MAX_CELLS_PER_BOX = 64

    def __init__(self, boxes: Iterable[region.Box]) -> None:
        items = list(boxes)
        self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._wide: list[int] = []
        if not items:
            self._size = 0.0
            return
        spans = [max(b[2] - b[0], b[3] - b[1]) for b in items]
        positive = [s for s in spans if s > 0]
        # The mean and not the maximum: a single district-boundary polygon
        # would set a cell as large as the drawing and void the index.
        self._size = (sum(positive) / len(positive)) if positive else 0.0
        if self._size <= 0:
            self._wide = list(range(len(items)))
            return
        for i, b in enumerate(items):
            x0, y0 = self._cell(b[0], b[1])
            x1, y1 = self._cell(b[2], b[3])
            if (x1 - x0 + 1) * (y1 - y0 + 1) > self._MAX_CELLS_PER_BOX:
                self._wide.append(i)
                continue
            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    self._cells[(cx, cy)].append(i)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / self._size), math.floor(y / self._size))

    def candidates(self, x: float, y: float) -> list[int]:
        """Indices of boxes that may contain this point. A superset, never a subset."""
        if self._size <= 0:
            return self._wide
        return self._cells.get(self._cell(x, y), []) + self._wide


def _label_groups(
    rings: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    area_unit: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Two or more texts with the same content inside a single polygon.

    The point test is `region.point_in_ring`, which returns THREE answers. A
    `boundary` point counts as inside and is reported separately: a label
    exactly on the line is a real state of affairs, and dropping it silently
    removes precisely the most ambiguous case.

    Seam note: point-in-polygon is a UPLIFT-03 deliverable (subagent C,
    `store_spatial.join_labels`). What is used here is its primitive in
    `region`, not a copy of C's function — the question really is different (C
    pairs labels to parcels; this groups labels PER parcel). When U03 lands,
    the two should best meet in one place; the request is in
    `docs/PROGRESS.md`.
    """
    skipped = {"skipped_no_anchor": 0, "skipped_no_ring": 0, "boundary_cases": 0}

    targets: list[tuple[Mapping[str, Any], list[tuple[float, float]], region.Box]] = []
    for row in rings:
        ring = row.get("ring")
        if not ring or len(ring) < 3:
            skipped["skipped_no_ring"] += 1
            continue
        pts = [(float(p[0]), float(p[1])) for p in ring]
        targets.append((row, pts, region.polygon_bbox(pts)))

    grid = _BboxGrid(t[2] for t in targets)

    inside: dict[
        tuple[Any, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = defaultdict(list)
    for label in labels:
        anchor = label.get("anchor_point")
        text = (label.get("text") or "").strip()
        if not anchor or len(anchor) < 2:
            skipped["skipped_no_anchor"] += 1
            continue
        if not text:
            continue
        x, y = float(anchor[0]), float(anchor[1])
        for i in grid.candidates(x, y):
            row, pts, box = targets[i]
            if not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
                continue
            verdict = region.point_in_ring(x, y, pts)
            if verdict == "outside":
                continue
            if verdict == "boundary":
                skipped["boundary_cases"] += 1
            inside[(row.get("handle"), text.casefold())].append((row, label))
            break

    groups: list[dict[str, Any]] = []
    for (_handle, folded), pairs in inside.items():
        if len(pairs) < 2:
            continue
        polygon = pairs[0][0]
        groups.append({
            "polygon_handle": polygon.get("handle"),
            "polygon_layer": polygon.get("layer"),
            "polygon_area": polygon.get("area"),
            "polygon_area_units": area_unit if polygon.get("area") is not None else None,
            "text": folded,
            "member_count": len(pairs),
            "members": [
                {
                    "handle": lab.get("handle"),
                    "layer": lab.get("layer"),
                    "type": lab.get("type"),
                    "text": lab.get("text"),
                }
                for _, lab in pairs
            ],
        })
    groups.sort(key=lambda g: (-g["member_count"], str(g["polygon_handle"])))
    return groups, skipped


def _duplicates_evidence(
    *,
    drawing_id: str,
    layout: str,
    units: Mapping[str, Any],
    kind: str,
    tolerance: float,
    groups: Sequence[Mapping[str, Any]],
    ring_census: RingCensus,
    claim_value: str,
    layers: Sequence[str] | None,
) -> ev.Evidence:
    scope = _scope(layout, units, _scope_sentence(layout, units, kind, tolerance, layers))
    unit_name, _ = _tolerance_units(units)
    unit_text = unit_name or "drawing units"

    if not groups:
        # An absence is a finding, and a finding that concludes by elimination
        # must name the surroundings it concluded from.
        observation = ev.Observation(
            origin=ev.Origin.ABSENCE,
            detail=(
                f"{ring_census.complete} complete polygons examined in layout "
                f"{layout!r} at tolerance {tolerance:g} {unit_text}; no group "
                "met the conditions"
            ),
            layout=layout,
        )
        return _geometry_evidence(
            claim_value=f"no {claim_value}",
            drawing_id=drawing_id,
            scope=scope,
            observations=(observation,),
            not_established=(
                f"{sum(ring_census.as_dict().values())} polygons were skipped "
                "because their ring is not complete, and never entered this "
                "comparison"
            ),
            how_to_verify=(
                "raise the tolerance, or inspect the bulged and open polygons "
                "listed under `skipped` — neither of them has a boundary that "
                "can be compared"
            ),
        )

    biggest = groups[0]
    observations = [
        ev.Observation(
            origin=ev.Origin.GEOMETRY,
            detail=(
                f"{len(groups)} groups found among {ring_census.complete} "
                f"complete polygons, at tolerance {tolerance:g} {unit_text}"
            ),
            layout=layout,
        ),
        ev.Observation(
            origin=ev.Origin.GEOMETRY,
            detail=(
                "the largest group has "
                f"{biggest.get('member_count')} member polygons"
            ),
            locator=str(
                biggest.get("shape_key") or biggest.get("polygon_handle") or ""
            ),
            layout=layout,
        ),
    ]
    return _geometry_evidence(
        claim_value=claim_value,
        drawing_id=drawing_id,
        scope=scope,
        observations=observations,
        not_established=(
            "whether these groups are deliberate. Stacked shapes can be "
            "working copies, overlays, or repetition that really was intended; "
            "geometry does not tell them apart"
        ),
        how_to_verify=(
            "compare against this drawing's land-use config, or ask whoever "
            "assembled the file whether the layers concerned really are copies"
        ),
    )


# --- Response: shape cross-check --------------------------------------------


def _passes_shape_test(
    row: Mapping[str, Any],
    edge_range: tuple[float, float],
    area_range: tuple[float, float],
) -> bool:
    """Whether this polygon matches the shape test HANDED OVER by the caller.

    The ranges are parameters and have no default, and that is not an
    oversight: "one edge 24-26 with area 150-650" is a trait of one drawing,
    and a default inside a `.py` turns this module into a special-purpose
    solution for that drawing (G1, G6). The 17th drawing arrives with a
    different module, or with no module at all, and must still be sweepable.
    """
    area = row.get("area")
    if area is None:
        return False
    if not (area_range[0] <= float(area) <= area_range[1]):
        return False
    lens = row.get("edge_lengths") or []
    return any(edge_range[0] <= float(v) <= edge_range[1] for v in lens)


def build_cross_check(
    *,
    drawing_id: str,
    layout: str | None,
    rows: Sequence[Mapping[str, Any]],
    ring_census: RingCensus,
    units: Mapping[str, Any],
    coded_layers: Sequence[str],
    edge_range: tuple[float, float],
    area_range: tuple[float, float],
    limit: int = 50,
) -> dict[str, Any]:
    """A shape sweep that uses no layer names, compared with one that does.

    This is what lifts an answer from a claim to a finding — but not by raising
    its evidence grade, and that is deliberate. Shape is `Corpus.COORDINATES`,
    which never STATES; agreement between shape and layer names therefore
    remains a single stating corpus, and the grade does not rise. What does
    change is the opposite direction: every polygon on a coded layer that FAILS
    the shape test is recorded as a `Contradiction`, and a contradiction
    LOWERS. This asymmetry belongs to UPLIFT-08, it is not this file's
    decision.
    """
    layout = _require_layout(layout)
    coded = {str(name) for name in coded_layers}

    # One pass, one shape-test evaluation per row. The previous version
    # filtered `failing` as `r not in passing`, which compares dict by dict:
    # quadratic on 2,380 rows, and wrong if two polygons happen to have
    # identical projections.
    in_coded: list[Mapping[str, Any]] = []
    passing: list[Mapping[str, Any]] = []
    failing: list[Mapping[str, Any]] = []
    outside: list[Mapping[str, Any]] = []
    for row in rows:
        fits = _passes_shape_test(row, edge_range, area_range)
        if str(row.get("layer")) in coded:
            in_coded.append(row)
            (passing if fits else failing).append(row)
        elif fits:
            outside.append(row)

    unit_name, unit_reason = _tolerance_units(units)
    scope_note = _scope_sentence(layout, units, "shape", None, sorted(coded))
    body: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "kind": "cross_check",
        "scope_note": scope_note,
        "targets_considered": ring_census.complete,
        "skipped": dict(ring_census.as_dict()),
        "coded_layers": sorted(coded),
        "shape_test": {
            "edge_length_range": list(edge_range),
            "area_range": list(area_range),
            "length_units": unit_name,
            "area_units": units.get("area_unit"),
            "units_note": unit_reason,
            "method": (
                "at least one edge length inside the range, AND the area "
                "inside the range; layer names are not used at all"
            ),
        },
        "coded_total": len(in_coded),
        "coded_passing_shape_test": len(passing),
        "coded_failing_shape_test": len(failing),
        "failing_by_layer": _layer_counts(failing),
        "shape_matches_outside_coded_layers": len(outside),
        "outside_by_layer": _layer_counts(outside),
        "outside_sample": [
            {"handle": r.get("handle"), "layer": r.get("layer"), "area": r.get("area")}
            for r in outside[:limit]
        ],
        "outside_sample_truncated": len(outside) > limit,
        "agreement_fraction": (len(passing) / len(in_coded)) if in_coded else None,
        "agreement_fraction_note": (
            None
            if in_coded
            else "there is no polygon on any of the coded layers named, so "
                 "there is no denominator; this is not zero agreement"
        ),
        "interpretation_note": (
            "Failing the shape test does NOT mean the classification is wrong. "
            "Corner plots and end plots genuinely are not rectangular, and are "
            "still plots. What is reported here is where two methods that "
            "share no assumptions DO NOT meet — not which one is right."
        ),
    }

    scope = _scope(layout, units, scope_note)
    observations = [
        ev.Observation(
            origin=ev.Origin.GEOMETRY,
            detail=(
                f"{len(passing)} of {len(in_coded)} polygons on coded layers "
                "pass the shape test, which uses no layer names"
            ),
            layout=layout,
        ),
        ev.Observation(
            origin=ev.Origin.GEOMETRY,
            detail=(
                f"{len(outside)} polygons of the same shape sit outside the "
                "coded layers"
            ),
            layout=layout,
        ),
    ]
    contradictions = [
        ev.Contradiction(
            detail=(
                f"{n['count']} polygons on layer {n['layer']!r} are classified "
                "through their layer name but do not pass the shape test"
            ),
            observation=ev.Observation(
                origin=ev.Origin.GEOMETRY,
                detail=(
                    f"layer {n['layer']!r}: {n['count']} polygons fail the "
                    "shape test"
                ),
                locator=str(n["layer"]),
                layout=layout,
            ),
        )
        for n in _layer_counts(failing)[:limit]
    ]

    ev.attach(body, _geometry_evidence(
        claim_value="agreement between layer-name classification and the shape test",
        drawing_id=drawing_id,
        scope=scope,
        observations=observations,
        contradictions=contradictions,
        not_established=(
            "whether the polygons that fail the shape test are misclassified. "
            "The shape test describes; it is not a filter that rejects the rest"
        ),
        how_to_verify=(
            "open the polygons listed under `outside_sample` in the viewer, "
            "and compare them against this drawing's land-use config"
        ),
    ))
    return body


def build_shape_fingerprint(
    *,
    drawing_id: str,
    layout: str | None,
    rows: Sequence[Mapping[str, Any]],
    ring_census: RingCensus,
    units: Mapping[str, Any],
    layers: Sequence[str] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """`group_by="shape"`: how many polygons per `shape_key`, and modal edges.

    Answers "which plot modules does this drawing use" without touching layer
    names at all. Groups with a single member are included, unlike in
    `kind="shape"`: the question here is the distribution of shapes, and a
    distribution that discards its tail is not a distribution.
    """
    layout = _require_layout(layout)
    by_key, without_key = _by_shape_key(rows)
    unit_name, unit_reason = _tolerance_units(units)

    groups = [
        {
            "shape_key": key,
            "count": len(members),
            "modal_edge_lengths": _modal_edge_lengths(members),
            "modal_edge_length_units": unit_name,
            "modal_edge_length_units_note": unit_reason,
            "layers": _layer_counts(members),
        }
        for key, members in by_key.items()
    ]
    groups.sort(key=lambda g: (-g["count"], g["shape_key"]))

    scope_note = _scope_sentence(layout, units, "shape", None, layers)
    body: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "group_by": "shape",
        "scope_note": scope_note,
        "targets_considered": ring_census.complete,
        "skipped": {**ring_census.as_dict(), "skipped_no_shape_key": without_key},
        "distinct_shapes": len(groups),
        "limit": limit,
        "truncated": len(groups) > limit,
        "groups": groups[:limit],
        "shape_key_basis": next(
            (r["shape_key_basis"] for r in rows if r.get("shape_key_basis")),
            None,
        ),
        "interpretation_note": _INTERPRETATION["shape"],
        "cross_drawing_note": (
            "shape_key is meaningful only WITHIN this drawing. It is "
            "scale-sensitive but unit-blind: a 12x25 metre shape and a 12x25 "
            "inch shape produce exactly the same key, so this key must not be "
            "compared against another drawing (G10)."
        ),
    }

    ev.attach(body, _geometry_evidence(
        claim_value="the distribution of polygon shapes, read without layer names",
        drawing_id=drawing_id,
        scope=_scope(layout, units, scope_note),
        observations=(
            ev.Observation(
                origin=ev.Origin.GEOMETRY,
                detail=(
                    f"{len(groups)} distinct shapes among "
                    f"{ring_census.complete} complete polygons"
                ),
                layout=layout,
            ),
        ),
        not_established=(
            "what each shape means. A group is a repeated shape, not a "
            "typology; what names it is the config, not the geometry"
        ),
        how_to_verify=(
            "compare the largest groups against this drawing's land-use "
            "config, or against the typology table on its sheet"
        ),
    ))
    return body


# --- Data retrieval ----------------------------------------------------------


def _ring_census(
    drawing_id: str, layout: str, layers: Sequence[str] | None
) -> RingCensus:
    """How many polygons per `ring_status`, one indexed aggregation.

    Its filter prefix is `drawing_id` + `layout`, which is exactly the
    `(drawing_id, layout)` index. This cluster runs with `notablescan`: a query
    without an indexed plan does not slow down, it FAILS.
    """
    match: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "ring_status": {"$exists": True},
    }
    if layers:
        match["layer"] = {"$in": list(layers)}
    rows = coll(COLL_ENTITIES).aggregate([
        {"$match": match},
        {"$group": {"_id": "$ring_status", "n": {"$sum": 1}}},
    ])
    return RingCensus.from_counts({str(r["_id"]): int(r["n"]) for r in rows})


def _load_rings(
    drawing_id: str,
    layout: str,
    layers: Sequence[str] | None,
    *,
    with_ring: bool = False,
) -> list[dict[str, Any]]:
    """Polygons that legitimately become targets. One filter form, and only one.

    `{"drawing_id", "layout", "ring_status": "complete"}` — not
    `if doc.get("ring")`. The second is correct today precisely because `ring`
    is already null for the other statuses, but the one that STATES the intent
    is `ring_status`, and that is what the next person reads.
    """
    query: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "ring_status": "complete",
    }
    if layers:
        query["layer"] = {"$in": list(layers)}
    projection = _RING_PROJECTION_WITH_RING if with_ring else _RING_PROJECTION
    return list(coll(COLL_ENTITIES).find(query, projection).sort([("handle", 1)]))


def _load_labels(
    drawing_id: str, layout: str, layers: Sequence[str] | None
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "anchor_point": {"$ne": None},
        "text": {"$nin": [None, ""]},
    }
    if layers:
        query["layer"] = {"$in": list(layers)}
    return list(
        coll(COLL_ENTITIES).find(query, _LABEL_PROJECTION)
        .sort([("handle", 1)])
        .limit(MAX_LABELS)
    )


# --- Public surface ----------------------------------------------------------


def find_duplicates(
    drawing_id: str,
    *,
    layout: str,
    units: Mapping[str, Any],
    layers: Sequence[str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    kind: str = "geometry",
    limit: int = 50,
) -> dict[str, Any]:
    """Duplicated geometry: in the same place, the same shape, or the same text.

    Args:
        drawing_id: mandatory, and must be the prefix of every filter.
            `shape_key` is unit-blind, so comparing it across drawings matches
            a 12x25 metre plot with a 12x25 inch component (G10) — and this
            cluster refuses queries without an index, so it is also a technical
            necessity.
        layout: mandatory. Without it, sheet geometry and model space geometry
            that share coordinates enter the same comparison.
        units: the output of `store._unit_names(drawing, layout)`, as it comes.
            This file must not import `store` (see the file header), and
            copying its logic would mean two places decide whether a sheet has
            units.
        tolerance: in drawing units. Its unit is echoed in the response, not
            promised by the parameter's name.
        kind: one of `KINDS`.
    """
    layout = _require_layout(layout)
    kind = _require_kind(kind)
    tolerance = _require_tolerance(tolerance)

    census = _ring_census(drawing_id, layout, layers)
    refuse_if_oversize(census.complete, layout=layout)

    rows = _load_rings(drawing_id, layout, layers, with_ring=(kind == "label"))
    label_rows = _load_labels(drawing_id, layout, None) if kind == "label" else None
    return build_duplicates(
        drawing_id=drawing_id, layout=layout, rows=rows, ring_census=census,
        kind=kind, tolerance=tolerance, units=units, layers=layers, limit=limit,
        label_rows=label_rows,
    )


def shape_fingerprint(
    drawing_id: str,
    *,
    layout: str,
    units: Mapping[str, Any],
    layers: Sequence[str] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """A second, independent method: shape distribution, without layer names."""
    layout = _require_layout(layout)
    census = _ring_census(drawing_id, layout, layers)
    refuse_if_oversize(census.complete, layout=layout)
    rows = _load_rings(drawing_id, layout, layers)
    return build_shape_fingerprint(
        drawing_id=drawing_id, layout=layout, rows=rows, ring_census=census,
        units=units, layers=layers, limit=limit,
    )


def cross_check_layers(
    drawing_id: str,
    *,
    layout: str,
    units: Mapping[str, Any],
    coded_layers: Sequence[str],
    edge_range: tuple[float, float],
    area_range: tuple[float, float],
    limit: int = 50,
) -> dict[str, Any]:
    """Layer-name classification compared with a layer-blind shape test.

    `coded_layers`, `edge_range` and `area_range` deliberately have no default:
    they are traits of ONE drawing, and a default inside a `.py` turns this
    module into a special-purpose solution for that drawing (G1). Until the
    UPLIFT-02 land-use config lands, the caller is the one who hands them over.
    """
    layout = _require_layout(layout)
    census = _ring_census(drawing_id, layout, None)
    refuse_if_oversize(census.complete, layout=layout)
    rows = _load_rings(drawing_id, layout, None)
    return build_cross_check(
        drawing_id=drawing_id, layout=layout, rows=rows, ring_census=census,
        units=units, coded_layers=coded_layers, edge_range=edge_range,
        area_range=area_range, limit=limit,
    )
