"""Dossier lane 3 — anomalies for one layer x layout.

Pure functions over already-fetched entity rows. No Mongo, no ezdxf, no
FastAPI, no I/O of any kind: lists of dicts in, list of dicts out. That is
what lets lane 1 read the store exactly once and lets this file be tested
without a database.

Why this module exists at all
-----------------------------
The headline anomaly in this project's reference drawing is duplicated
geometry, and the obvious instrument cannot see it. Phase 0 measured it:
``shape_key`` is populated for closed rings and is ``None`` for open geometry
(LINE, ARC, open LWPOLYLINE), so the existing ``shape_fingerprint`` /
``find_duplicates`` machinery returns nothing at all for a road-centreline
layer that is in fact drawn three times. The replacement instrument, proven
read-only against the live store on 25 August 2026:

    partition the layer's entities into clusters separated by a void wider
    than any real feature on that layer, then compare clusters by
    (entity count, bbox width, bbox height). Clusters agreeing on all three
    are copies.

    layer                    entities  clusters  per cluster            verdict
    00_Prop - Road - CL_        1,233     3      411, 1778.387x1585.99  copies
    ROW                            91     1      1781.366x1590.138      single
    VL4                           195     1      1767.003x1501.938      single

Both instruments are run here and **every finding names the one that produced
it**, because their strengths differ and a reader must know which one spoke:

* ``shape_key`` — exact, translation/rotation/winding invariant, but blind
  wherever the key is null, which is all open geometry.
* spatial clustering — sees open geometry, but works on bulk shape (count and
  bounding box) rather than on the geometry itself, so it is evidence rather
  than proof.

Rules this file is written to (``docs/UPLIFT-GENERALITY-RULES.md``)
------------------------------------------------------------------
* **G1** — nothing is decided from a layer NAME. ``layer`` is carried into
  the output so a person can check the finding; it never enters a branch.
* **G2** — the gap threshold is DERIVED from the layer's own geometry and
  STATED in the response. It is never a constant in drawing units: a void
  that means "detached copy" in metres means nothing in inches, and 11 of the
  18 drawings in this store are in inches while 3 declare no unit at all.
  When the caller supplies no unit, distances are labelled "drawing units"
  and the reason travels with them — never a defaulted metre.
* **G6** — no assumption about shape, axis, or module size. Bounding boxes
  are compared to each other, never to a remembered size.
* **G7** — every cap is stated, and anything omitted is counted in the
  response next to what was kept.
* **G8** — a missing datum is null plus a reason, never zero. An entity that
  cannot be placed is counted and named, not dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence as _AbcSequence
from typing import Any, Callable, NamedTuple, Sequence

__all__ = [
    "anomalies",
    "cluster_scan",
    "KINDS",
    "DUPLICATE_MATCH_REL_TOL",
    "MIN_CLUSTER_ENTITIES",
    "MAX_CLUSTERS_REPORTED",
    "MAX_GROUPS_REPORTED",
    "MAX_KEYS_REPORTED",
    "MAX_HANDLES_LISTED",
    "MAX_ROWS_CLUSTERED",
    "MAX_PAIRWISE_CLUSTERS",
]

#: The only `kind` values this lane may emit. Fixed by the interface contract
#: in docs/DOSSIER-01-DESIGN.md; a fourth kind would be a contract change.
KINDS = ("duplicate_clusters", "declared_but_empty", "unreadable")

#: Two clusters count as the same size when both bounding-box dimensions agree
#: to within this RELATIVE tolerance. Relative, not absolute, and therefore
#: carrying no unit at all (G2): it behaves identically in metres, in inches,
#: and in a drawing that declares nothing. At the reference layer's scale
#: (~1778 units) it works out at 0.0018 units, which is why Phase 0 could
#: report the three road clusters as "identical to 3 dp".
DUPLICATE_MATCH_REL_TOL = 1e-6

#: A cluster smaller than this has no internal structure, so two of them
#: agreeing on a bounding box is arithmetic, not evidence. Raising this makes
#: the instrument stricter; lowering it below 2 makes every isolated point a
#: candidate copy of every other isolated point.
MIN_CLUSTER_ENTITIES = 2

#: Reporting caps (G7). Everything above a cap is counted, never dropped in
#: silence: each capped list is published beside its `*_omitted` count.
MAX_CLUSTERS_REPORTED = 50
MAX_GROUPS_REPORTED = 20
MAX_KEYS_REPORTED = 20
MAX_HANDLES_LISTED = 12

#: Above this many rows the cluster instrument declines rather than running:
#: the partition sorts and re-sorts its input, and a layer this size means the
#: caller is asking a different question. The largest drawing in the store
#: holds 46,754 entities in total, so this ceiling has never been reached; it
#: exists so that the day it is, the response says so.
MAX_ROWS_CLUSTERED = 200_000

#: Inter-cluster separation is an all-pairs measurement. Above this many
#: clusters it is skipped and the omission is stated instead of being timed.
MAX_PAIRWISE_CLUSTERS = 500


# ---------------------------------------------------------------------------
# reading rows
# ---------------------------------------------------------------------------


class _Placed(NamedTuple):
    """One entity reduced to what both instruments need."""

    idx: int
    handle: str | None
    etype: str | None
    lo_x: float
    lo_y: float
    hi_x: float
    hi_y: float
    length: float | None


def _num(value: Any) -> float | None:
    """A finite float, or None. Booleans are not numbers here."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _xy(seq: Any) -> tuple[float, float] | None:
    """The first two finite coordinates of a point-like sequence, or None."""
    if isinstance(seq, (str, bytes)) or not isinstance(seq, _AbcSequence):
        return None
    if len(seq) < 2:
        return None
    x, y = _num(seq[0]), _num(seq[1])
    if x is None or y is None:
        return None
    return x, y


class _Readout(NamedTuple):
    placed: list[_Placed]
    counts: dict[str, int]
    named: dict[str, list[str]]


def _read_rows(rows: Sequence[Any]) -> _Readout:
    """Split rows into placeable entities and the defects that stopped them.

    Nothing is discarded quietly. Every row that does not become a `_Placed`
    is counted under the reason it failed and, where it can be named at all,
    its handle is kept so a person can go and look at it.
    """
    placed: list[_Placed] = []
    counts = {
        "rows": len(rows),
        "not_a_record": 0,
        "missing_bbox": 0,
        "malformed_bbox": 0,
        "reversed_bbox": 0,
        "unnameable": 0,
        "untyped": 0,
        "no_measure": 0,
    }
    # The names are capped at collection time, not at report time: a layer of
    # 200,000 broken rows must not build a 200,000-entry list to then throw
    # most of it away. The COUNTS above stay complete, and the report says how
    # many names it is not showing.
    named: dict[str, list[str]] = {
        "missing_bbox": [],
        "malformed_bbox": [],
        "reversed_bbox": [],
        "untyped": [],
    }

    def remember(bucket: str, label: str) -> None:
        if len(named[bucket]) < MAX_HANDLES_LISTED:
            named[bucket].append(label)

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            counts["not_a_record"] += 1
            counts["unnameable"] += 1
            continue

        raw_handle = row.get("handle")
        handle = str(raw_handle) if isinstance(raw_handle, (str, int)) else None
        if not handle:
            handle = None
            counts["unnameable"] += 1
        label = handle or f"row#{idx}"

        raw_type = row.get("type")
        etype = str(raw_type) if isinstance(raw_type, str) and raw_type else None
        if etype is None:
            counts["untyped"] += 1
            remember("untyped", label)

        if _num(row.get("length")) is None and _num(row.get("area")) is None:
            counts["no_measure"] += 1

        box = row.get("bbox")
        if box is None:
            counts["missing_bbox"] += 1
            remember("missing_bbox", label)
            continue
        if not isinstance(box, dict):
            counts["malformed_bbox"] += 1
            remember("malformed_bbox", label)
            continue
        lo = _xy(box.get("min"))
        hi = _xy(box.get("max"))
        if lo is None or hi is None:
            counts["malformed_bbox"] += 1
            remember("malformed_bbox", label)
            continue

        if lo[0] > hi[0] or lo[1] > hi[1]:
            counts["reversed_bbox"] += 1
            remember("reversed_bbox", label)
        lo_x, hi_x = (lo[0], hi[0]) if lo[0] <= hi[0] else (hi[0], lo[0])
        lo_y, hi_y = (lo[1], hi[1]) if lo[1] <= hi[1] else (hi[1], lo[1])

        placed.append(
            _Placed(
                idx=idx,
                handle=handle,
                etype=etype,
                lo_x=lo_x,
                lo_y=lo_y,
                hi_x=hi_x,
                hi_y=hi_y,
                length=_num(row.get("length")),
            )
        )

    counts["placed"] = len(placed)
    counts["unplaceable"] = (
        counts["missing_bbox"] + counts["malformed_bbox"] + counts["not_a_record"]
    )
    return _Readout(placed=placed, counts=counts, named=named)


# ---------------------------------------------------------------------------
# the gap threshold — derived, never a constant in drawing units
# ---------------------------------------------------------------------------


def _derive_threshold(placed: Sequence[_Placed]) -> tuple[float, _Placed | None]:
    """The widest single feature on this layer, in either direction.

    This is the whole basis of the instrument, and it is a measurement of the
    layer itself rather than a number chosen by a programmer. A void wider
    than the widest feature on the layer cannot lie *inside* a feature, so it
    separates one body of geometry from another.

    It fails in the safe direction. One oversized entity — a sheet border, a
    single polyline drawn across the whole estate — raises the threshold and
    merges bodies that a smaller threshold would have split. That is a missed
    finding, not an invented one, and the number and the entity that set it
    are both published so the miss is visible rather than mysterious.
    """
    widest = 0.0
    source: _Placed | None = None
    for p in placed:
        extent = max(p.hi_x - p.lo_x, p.hi_y - p.lo_y)
        if extent > widest:
            widest, source = extent, p
    return widest, source


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------


def _split_axis(
    members: Sequence[_Placed],
    lo_of: Callable[[_Placed], float],
    hi_of: Callable[[_Placed], float],
    threshold: float,
) -> tuple[list[list[_Placed]], float]:
    """Merge the members' 1-D intervals; cut wherever the void is too wide.

    Intervals, not points: an entity that spans a whole estate is one interval
    covering it, so a long feature closes the space it crosses instead of
    reporting a void at every step along it.

    Returns the groups and the widest void that was *kept* (i.e. judged to be
    inside one body), which is the margin a reader needs to judge the cut.
    """
    order = sorted(members, key=lambda p: (lo_of(p), hi_of(p), p.idx))
    groups: list[list[_Placed]] = []
    current = [order[0]]
    current_hi = hi_of(order[0])
    widest_kept = 0.0
    for item in order[1:]:
        gap = lo_of(item) - current_hi
        if gap > threshold:
            groups.append(current)
            current = [item]
            current_hi = hi_of(item)
            continue
        if gap > widest_kept:
            widest_kept = gap
        current.append(item)
        current_hi = max(current_hi, hi_of(item))
    groups.append(current)
    return groups, widest_kept


def _partition(
    placed: Sequence[_Placed], threshold: float
) -> tuple[list[list[_Placed]], float]:
    """Alternate x and y interval cuts until neither axis cuts any further.

    Iterative rather than recursive on purpose: a pathological layer that
    peels one entity off at a time would blow a recursion limit, and a
    RecursionError is not an answer.
    """
    stack: list[list[_Placed]] = [list(placed)]
    final: list[list[_Placed]] = []
    widest_internal = 0.0
    while stack:
        group = stack.pop()
        by_x, void_x = _split_axis(
            group, lambda p: p.lo_x, lambda p: p.hi_x, threshold
        )
        if len(by_x) > 1:
            stack.extend(by_x)
            continue
        by_y, void_y = _split_axis(
            group, lambda p: p.lo_y, lambda p: p.hi_y, threshold
        )
        if len(by_y) > 1:
            stack.extend(by_y)
            continue
        final.append(group)
        widest_internal = max(widest_internal, void_x, void_y)
    final.sort(key=lambda g: (min(p.lo_x for p in g), min(p.lo_y for p in g)))
    return final, widest_internal


def _summarise(group: Sequence[_Placed], rank: int) -> dict[str, Any]:
    lo_x = min(p.lo_x for p in group)
    lo_y = min(p.lo_y for p in group)
    hi_x = max(p.hi_x for p in group)
    hi_y = max(p.hi_y for p in group)
    lengths = [p.length for p in group if p.length is not None]
    ordered = sorted(group, key=lambda p: p.idx)
    handles = [p.handle for p in ordered if p.handle][:MAX_HANDLES_LISTED]
    return {
        "cluster": rank,
        "entities": len(group),
        "bbox_min": [lo_x, lo_y],
        "bbox_max": [hi_x, hi_y],
        "bbox_width": hi_x - lo_x,
        "bbox_height": hi_y - lo_y,
        "length_total": sum(lengths) if lengths else None,
        "length_measured": len(lengths),
        "length_unmeasured": len(group) - len(lengths),
        "example_handles": handles,
        "handles_omitted": max(0, len(group) - len(handles)),
    }


def _separation(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Void between two cluster boxes, on the axis that separates them most.

    This mirrors the cut rule exactly: the partition splits on a per-axis
    void, so the separation that matters is the larger of the two axis voids.
    """
    a_lo, a_hi = a["bbox_min"], a["bbox_max"]
    b_lo, b_hi = b["bbox_min"], b["bbox_max"]
    gap_x = max(b_lo[0] - a_hi[0], a_lo[0] - b_hi[0])
    gap_y = max(b_lo[1] - a_hi[1], a_lo[1] - b_hi[1])
    return max(0.0, gap_x, gap_y)


# ---------------------------------------------------------------------------
# comparing clusters
# ---------------------------------------------------------------------------


def _close(a: float, b: float, rel: float = DUPLICATE_MATCH_REL_TOL) -> bool:
    """Equality with a purely relative tolerance — no unit, no absolute floor."""
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return True
    return abs(a - b) <= rel * scale


def _same_size(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Three gates, and every one of them must hold.

    The entity counts must be EQUAL — integers, no tolerance, because a copy
    that lost an entity is not a copy. Both bounding-box dimensions must agree
    within the relative tolerance. This is the false-positive guard: two
    detached bodies of the same width but different height, or the same box
    with different populations, are neighbours, not copies.
    """
    return (
        a["entities"] == b["entities"]
        and _close(a["bbox_width"], b["bbox_width"])
        and _close(a["bbox_height"], b["bbox_height"])
    )


def _eligible(cluster: dict[str, Any]) -> bool:
    """A cluster that can carry the claim at all.

    Degenerate boxes and lone entities are excluded before comparison: a page
    full of zero-extent points would otherwise report itself as hundreds of
    mutual copies, which is the loudest false positive this instrument can
    produce.
    """
    if cluster["entities"] < MIN_CLUSTER_ENTITIES:
        return False
    return cluster["bbox_width"] > 0.0 or cluster["bbox_height"] > 0.0


def _group_by_size(clusters: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Greedy grouping of clusters that agree on all three signature fields."""
    candidates = [c for c in clusters if _eligible(c)]
    candidates.sort(key=lambda c: (-c["entities"], c["bbox_min"][0], c["bbox_min"][1]))
    groups: list[list[dict[str, Any]]] = []
    for cluster in candidates:
        for group in groups:
            if _same_size(group[0], cluster):
                group.append(cluster)
                break
        else:
            groups.append([cluster])
    return [g for g in groups if len(g) >= 2]


def _length_agreement(group: Sequence[dict[str, Any]]) -> tuple[bool | None, str]:
    """Corroboration from a second measurement, reported but never decisive.

    Σ length is an independent reading of the same claim — Phase 0's three
    road clusters agree at 27,987.522 each — but a length is absent on plenty
    of entities, so it is published as a separate verdict rather than folded
    into the match. A reader can see both and weigh them.
    """
    totals = [c["length_total"] for c in group]
    if any(t is None for t in totals):
        return None, "not every cluster in this group carries a length"
    first = totals[0]
    if all(_close(first, t) for t in totals[1:]):
        return True, (
            "the clusters' length totals agree within "
            f"{DUPLICATE_MATCH_REL_TOL:g} relative, independently of the bbox test"
        )
    return False, (
        "the clusters are the same size but their length totals differ — the "
        "bounding boxes match while the geometry inside them may not"
    )


# ---------------------------------------------------------------------------
# unit labelling (G2)
# ---------------------------------------------------------------------------


def _unit_block(unit: str | None) -> dict[str, Any]:
    if unit:
        return {
            "unit": unit,
            "unit_label": unit,
            "unit_basis": "unit supplied by the caller from the drawing's own header",
        }
    return {
        "unit": None,
        "unit_label": "drawing units",
        "unit_basis": (
            "no unit was supplied for this drawing, so every distance here is in "
            "the file's own drawing units. It must not be read as metres: 11 of "
            "the 18 drawings in this store are in inches and 3 declare nothing."
        ),
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


# ---------------------------------------------------------------------------
# the cluster instrument, in full
# ---------------------------------------------------------------------------


def cluster_scan(
    rows: Sequence[Any], *, layer: str = "", unit: str | None = None
) -> dict[str, Any]:
    """The whole cluster instrument, including the times it declines to run.

    `anomalies()` publishes only contract-shaped findings, so this is where a
    caller (or a person reading a Dossier and wondering why a layer is silent)
    can see the readings that did not become a finding: the derived threshold,
    every cluster, the voids inside and between them, and the reason the scan
    was skipped when it was.
    """
    rows = list(rows)
    return _scan(rows, _read_rows(rows), layer=layer, unit=unit)


def _scan(
    rows: Sequence[Any],
    readout: _Readout,
    *,
    layer: str,
    unit: str | None,
) -> dict[str, Any]:
    """`cluster_scan` over a readout that has already been taken.

    Split out so `anomalies()` reads each row exactly once: it needs the same
    readout for the `unreadable` finding, and a 46,754-entity drawing should
    not be walked twice per layer to produce one document.
    """
    report: dict[str, Any] = {
        "layer": layer,
        "rows": len(rows),
        "placed": len(readout.placed),
        "unplaceable": readout.counts["unplaceable"],
        "gap_threshold": None,
        "gap_basis": "",
        "threshold_source": None,
        "clusters": [],
        "clusters_total": 0,
        "clusters_listed": 0,
        "clusters_omitted": 0,
        "duplicate_groups": [],
        "largest_void_inside_a_cluster": None,
        "smallest_void_between_clusters": None,
        "skipped": None,
        **_unit_block(unit),
    }
    label = report["unit_label"]

    if len(rows) > MAX_ROWS_CLUSTERED:
        report["skipped"] = (
            f"{len(rows)} rows exceeds the stated cluster ceiling of "
            f"{MAX_ROWS_CLUSTERED}; no cluster reading was taken for this layer"
        )
        return report
    if not readout.placed:
        report["skipped"] = (
            f"none of the {len(rows)} rows on this layer could be placed, so "
            "the layer cannot be partitioned at all; see the `unreadable` "
            "finding for what stopped each one"
        )
        return report

    threshold, source = _derive_threshold(readout.placed)
    report["gap_threshold"] = threshold
    report["threshold_source"] = (
        None
        if source is None
        else {
            "handle": source.handle,
            "type": source.etype,
            "extent": max(source.hi_x - source.lo_x, source.hi_y - source.lo_y),
        }
    )
    report["gap_basis"] = (
        f"the widest single feature on this layer measures {_fmt(threshold)} "
        f"{label}"
        + (f" (entity {source.handle}, {source.etype})" if source else "")
        + ". A void wider than that cannot lie inside one feature, so it "
        "separates one body of geometry from another. Derived from this "
        "layer's own geometry, not a constant: the same code on a drawing in "
        "inches derives an inch-sized threshold."
    )

    if threshold <= 0.0:
        report["skipped"] = (
            "every entity on this layer has a zero-extent bounding box, so no "
            "void can be called wider than a feature and the layer cannot be "
            "partitioned; no duplicate claim is made from clustering here"
        )
        return report

    groups, widest_internal = _partition(readout.placed, threshold)
    clusters = [_summarise(g, i) for i, g in enumerate(groups)]
    report["clusters_total"] = len(clusters)
    report["clusters"] = clusters[:MAX_CLUSTERS_REPORTED]
    report["clusters_listed"] = len(report["clusters"])
    report["clusters_omitted"] = len(clusters) - len(report["clusters"])
    report["largest_void_inside_a_cluster"] = widest_internal

    if len(clusters) < 2:
        report["smallest_void_between_clusters"] = None
    elif len(clusters) > MAX_PAIRWISE_CLUSTERS:
        report["smallest_void_between_clusters"] = None
        report["separation_note"] = (
            f"{len(clusters)} clusters exceeds the stated all-pairs ceiling of "
            f"{MAX_PAIRWISE_CLUSTERS}; inter-cluster separation was not measured"
        )
    else:
        report["smallest_void_between_clusters"] = min(
            _separation(a, b)
            for i, a in enumerate(clusters)
            for b in clusters[i + 1 :]
        )

    matched = _group_by_size(clusters)
    for group in matched:
        agrees, agreement_basis = _length_agreement(group)
        shown = group[:MAX_CLUSTERS_REPORTED]
        report["duplicate_groups"].append(
            {
                "members": len(group),
                "members_listed": len(shown),
                "members_omitted": len(group) - len(shown),
                "entities_each": group[0]["entities"],
                "bbox_width": group[0]["bbox_width"],
                "bbox_height": group[0]["bbox_height"],
                "entities_involved": sum(c["entities"] for c in group),
                "length_total_each": group[0]["length_total"],
                "length_agrees": agrees,
                "length_basis": agreement_basis,
                "clusters": [
                    {
                        "cluster": c["cluster"],
                        "entities": c["entities"],
                        "bbox_min": c["bbox_min"],
                        "bbox_width": c["bbox_width"],
                        "bbox_height": c["bbox_height"],
                        "length_total": c["length_total"],
                        "example_handles": c["example_handles"],
                        "handles_omitted": c["handles_omitted"],
                    }
                    for c in shown
                ],
            }
        )
    report["duplicate_groups"].sort(
        key=lambda g: (-g["entities_involved"], -g["members"])
    )
    return report


# ---------------------------------------------------------------------------
# the exact instrument: shape_key
# ---------------------------------------------------------------------------


def _keyed_scan(rows: Sequence[Any]) -> dict[str, Any]:
    """Group rows by `shape_key`. Exact where it exists, absent where it does not."""
    buckets: dict[str, list[str]] = {}
    keyed = 0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        key = row.get("shape_key")
        if not isinstance(key, str) or not key:
            continue
        keyed += 1
        handle = row.get("handle")
        nameable = isinstance(handle, (str, int)) and handle
        buckets.setdefault(key, []).append(
            str(handle) if nameable else f"row#{idx}"
        )
    repeated = sorted(
        ((k, v) for k, v in buckets.items() if len(v) > 1),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    return {
        "keyed_rows": keyed,
        "unkeyed_rows": len(rows) - keyed,
        "distinct_keys": len(buckets),
        "repeated": repeated,
    }


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


def anomalies(
    rows: list[dict], *, layer: str, unit: str | None = None
) -> list[dict]:
    """Duplicate clusters and other oddities for one layer x layout.

    Args:
        rows: entity dicts for ONE layer x layout, as they come out of the
            store. Read for `handle`, `type`, `bbox`, `length`, `area` and
            `shape_key`; every one of those may be absent, and an absence is
            reported rather than assumed away. The list is never mutated.
        layer: the layer's name. Carried into the output so a reader can check
            a finding against the file. It never enters a decision (G1) —
            passing a different name changes the prose and nothing else.
        unit: OPTIONAL, and additive to the contract signature: callers using
            `anomalies(rows, layer=...)` are unaffected. When the caller knows
            what the drawing declares, distances are labelled with it; when it
            is None — the default, and the honest state for the 3 drawings in
            this store that declare nothing — they are labelled "drawing
            units" and the reason travels with them (G2). No path in this
            module ever writes a unit that was not handed to it.

    Returns:
        A list of findings, `[]` when the layer is unremarkable — never None.
        Each finding is `{"kind", "detail", "evidence", "counts"}` with `kind`
        one of `KINDS`. `evidence["instrument"]` names what produced the
        finding, and `detail` names it again in prose, because `shape_key` and
        spatial clustering see different things and a reader must know which
        one spoke.
    """
    rows = list(rows or [])
    unit_block = _unit_block(unit)

    if not rows:
        return [
            {
                "kind": "declared_but_empty",
                "detail": (
                    f"Layer {layer!r} is declared by the drawing but holds no "
                    "entities in this scope. A query against it returns "
                    "nothing, and nothing is not the same answer as 'there "
                    "are none' — that reading has already been made once in "
                    "this project and was wrong."
                ),
                "evidence": {
                    "instrument": "row_count",
                    "instrument_basis": (
                        "the layer was declared to this function and no rows "
                        "were supplied for it"
                    ),
                    "layer": layer,
                    "rows_received": 0,
                },
                "counts": {"entities": 0},
            }
        ]

    findings: list[dict] = []
    readout = _read_rows(rows)
    keyed = _keyed_scan(rows)

    # --- instrument 1: shape_key, exact where it exists ---------------------
    if keyed["repeated"]:
        listed = keyed["repeated"][:MAX_KEYS_REPORTED]
        involved = sum(len(v) for _, v in keyed["repeated"])
        biggest = keyed["repeated"][0]
        findings.append(
            {
                "kind": "duplicate_clusters",
                "detail": (
                    f"{len(keyed['repeated'])} shape fingerprints on layer "
                    f"{layer!r} are each carried by more than one entity "
                    f"({involved} entities in total; the largest group has "
                    f"{len(biggest[1])}). Instrument: shape_key — exact, and "
                    "invariant to translation, rotation and winding, but "
                    "populated only for closed rings: "
                    f"{keyed['keyed_rows']} of {len(rows)} rows on this layer "
                    "carry one, so it says nothing about the other "
                    f"{keyed['unkeyed_rows']}."
                ),
                "evidence": {
                    "instrument": "shape_key",
                    "instrument_basis": (
                        "identical shape fingerprints. Exact within one "
                        "drawing, and blind wherever the key is null — which "
                        "is all open geometry (LINE, ARC, open LWPOLYLINE)."
                    ),
                    "layer": layer,
                    "keyed_rows": keyed["keyed_rows"],
                    "unkeyed_rows": keyed["unkeyed_rows"],
                    "distinct_keys": keyed["distinct_keys"],
                    "repeated_keys": [
                        {
                            "shape_key": key,
                            "entities": len(handles),
                            "example_handles": handles[:MAX_HANDLES_LISTED],
                            "handles_omitted": max(
                                0, len(handles) - MAX_HANDLES_LISTED
                            ),
                        }
                        for key, handles in listed
                    ],
                    "keys_listed": len(listed),
                    "keys_omitted": len(keyed["repeated"]) - len(listed),
                    "keys_cap": MAX_KEYS_REPORTED,
                },
                "counts": {
                    "repeated_keys": len(keyed["repeated"]),
                    "entities_involved": involved,
                    "keyed_rows": keyed["keyed_rows"],
                    "unkeyed_rows": keyed["unkeyed_rows"],
                },
            }
        )

    # --- instrument 2: spatial clustering, for the geometry with no key -----
    scan = _scan(rows, readout, layer=layer, unit=unit)
    if scan["duplicate_groups"]:
        listed = scan["duplicate_groups"][:MAX_GROUPS_REPORTED]
        top = scan["duplicate_groups"][0]
        involved = sum(g["entities_involved"] for g in scan["duplicate_groups"])
        label = unit_block["unit_label"]
        findings.append(
            {
                "kind": "duplicate_clusters",
                "detail": (
                    f"{top['members']} spatially detached clusters on layer "
                    f"{layer!r} hold {top['entities_each']} entities each "
                    f"inside a bounding box of {_fmt(top['bbox_width'])} x "
                    f"{_fmt(top['bbox_height'])} {label}. Agreement on all "
                    "three is what makes them copies of one another rather "
                    "than neighbours. Instrument: spatial clustering — the "
                    "exact instrument is blind here, because shape_key is "
                    f"null on {keyed['unkeyed_rows']} of {len(rows)} rows on "
                    "this layer."
                ),
                "evidence": {
                    "instrument": "spatial_cluster",
                    "instrument_basis": (
                        "the layer was partitioned at voids wider than its own "
                        "widest feature, then the resulting clusters were "
                        "compared by (entity count, bbox width, bbox height). "
                        "This reads bulk shape, not geometry: it is evidence "
                        "of copying, and shape_key is proof where it exists."
                    ),
                    "layer": layer,
                    "unit": unit_block["unit"],
                    "unit_label": unit_block["unit_label"],
                    "unit_basis": unit_block["unit_basis"],
                    "gap_threshold": scan["gap_threshold"],
                    "gap_basis": scan["gap_basis"],
                    "threshold_source": scan["threshold_source"],
                    "largest_void_inside_a_cluster": scan[
                        "largest_void_inside_a_cluster"
                    ],
                    "smallest_void_between_clusters": scan[
                        "smallest_void_between_clusters"
                    ],
                    "clusters_total": scan["clusters_total"],
                    "clusters_listed": scan["clusters_listed"],
                    "clusters_omitted": scan["clusters_omitted"],
                    "clusters_cap": MAX_CLUSTERS_REPORTED,
                    "match_tolerance_relative": DUPLICATE_MATCH_REL_TOL,
                    "match_basis": (
                        "entity counts must be equal exactly; both bounding-box "
                        f"dimensions must agree within {DUPLICATE_MATCH_REL_TOL:g} "
                        "relative. The tolerance is relative and therefore "
                        "unit-free — identical behaviour in metres, in inches, "
                        "and in a drawing that declares nothing. A cluster of "
                        f"fewer than {MIN_CLUSTER_ENTITIES} entities, or one "
                        "with a zero-by-zero box, is excluded before comparison."
                    ),
                    "keyed_rows": keyed["keyed_rows"],
                    "unkeyed_rows": keyed["unkeyed_rows"],
                    "groups": listed,
                    "groups_listed": len(listed),
                    "groups_omitted": len(scan["duplicate_groups"]) - len(listed),
                    "groups_cap": MAX_GROUPS_REPORTED,
                    "entities_not_placed": scan["unplaceable"],
                },
                "counts": {
                    "duplicate_groups": len(scan["duplicate_groups"]),
                    "clusters": scan["clusters_total"],
                    "entities_involved": involved,
                    "entities_clustered": scan["placed"],
                    "entities_unplaceable": scan["unplaceable"],
                },
            }
        )

    # --- what could not be read at all --------------------------------------
    counts = readout.counts
    broken = (
        counts["unplaceable"] + counts["reversed_bbox"] + counts["unnameable"]
    )
    if broken:
        named: list[str] = []
        for bucket in ("missing_bbox", "malformed_bbox", "reversed_bbox", "untyped"):
            named.extend(readout.named[bucket])
        listed_names = named[:MAX_HANDLES_LISTED]
        # Counted from the FULL totals, not from the already-capped name list,
        # so the omission figure is the real one (G7).
        nameable = (
            counts["missing_bbox"]
            + counts["malformed_bbox"]
            + counts["reversed_bbox"]
            + counts["untyped"]
        )
        findings.append(
            {
                "kind": "unreadable",
                "detail": (
                    f"{counts['unplaceable']} of {counts['rows']} entities on "
                    f"layer {layer!r} could not be placed and were therefore "
                    "excluded from the cluster instrument; "
                    f"{counts['reversed_bbox']} carry a reversed bounding box "
                    f"and {counts['unnameable']} carry no handle at all. They "
                    "are counted and named here rather than dropped, so the "
                    "layer's totals still reconcile. Entities that carry "
                    "neither a length nor an area are counted too "
                    f"({counts['no_measure']}) — normal for TEXT and INSERT, "
                    "and not a defect on its own."
                ),
                "evidence": {
                    "instrument": "row_readout",
                    "instrument_basis": (
                        "each row was read for a handle, a type, a usable "
                        "bounding box and a measurement; every failure is "
                        "counted under the reason it failed"
                    ),
                    "layer": layer,
                    "named": listed_names,
                    "names_listed": len(listed_names),
                    "names_omitted": max(0, nameable - len(listed_names)),
                    "names_cap": MAX_HANDLES_LISTED,
                    "breakdown": {
                        "missing_bbox": counts["missing_bbox"],
                        "malformed_bbox": counts["malformed_bbox"],
                        "reversed_bbox": counts["reversed_bbox"],
                        "not_a_record": counts["not_a_record"],
                        "unnameable": counts["unnameable"],
                        "untyped": counts["untyped"],
                        "no_length_or_area": counts["no_measure"],
                    },
                    "note": (
                        "measurement drops are lane 2's `unmeasured_entities`; "
                        "`no_length_or_area` is published here only so the "
                        "layer's arithmetic can be checked, and it never "
                        "raises this finding on its own"
                    ),
                },
                "counts": {
                    "rows": counts["rows"],
                    "placed": counts["placed"],
                    "unplaceable": counts["unplaceable"],
                    "reversed_bbox": counts["reversed_bbox"],
                    "unnameable": counts["unnameable"],
                    "untyped": counts["untyped"],
                    "no_length_or_area": counts["no_measure"],
                },
            }
        )

    return findings
