"""DOSSIER lane 2 — length totals and connected-chain counting for one layer.

Pure arithmetic over rows that have already been fetched. No Mongo, no ezdxf,
no FastAPI: a list of dicts goes in, a plain dict comes out. That is what lets
this file be tested against fixtures and lets the store be read exactly once,
by lane 1's assembler.

Two numbers are produced, and they are not equally trustworthy. Saying so in
the response is the entire point of this module.

**Σ length is a measurement.** Every entity that carries a `length` is in the
sum; every entity that does not is COUNTED and named, never treated as zero
(rule G8). When no entity carries a length at all, `length_total` is `None`
with a reason — a layer of TEXT has no length, and `0.0` would be a claim that
it was measured and found to be nothing. When some entities carry no length the
sum is published as a FLOOR, not a total, which is the same asymmetry
`store.measure` already enforces with `sum_measured_only`.

**The chain count is an inference, and a weak one.** Union-find over segment
endpoints is the right instrument, but the rows this project stores carry no
endpoint coordinates — only `bbox`, `bbox_centre`, `length` and `type`. What
can honestly be recovered from a bounding box is:

* a **LINE** with a degenerate box (zero width or zero height) — its two
  distinct corners ARE its endpoints. Exact.
* a **LINE** with a real box — its endpoints are one of the box's two
  diagonals, and which one is not recorded anywhere. Offering all four corners
  makes the candidate set a strict SUPERSET of the truth: no real join can be
  missed, and some joins that are not real can be invented. A component count
  built from a superset of edges can only be too LOW, so for an all-LINE layer
  the answer is a stated lower bound.
* an **ARC**, an open **LWPOLYLINE**, a **SPLINE** — the ends need not sit on a
  corner at all, and a polyline that doubles back does not even reach its own
  box edge with its ends. Corners are a proxy, joins can be both missed and
  invented, and the number has no guaranteed direction.
* a **CIRCLE**, or anything with `ring_status == "complete"` — a closed curve
  has no free ends. It is its own component and cannot be joined to anything,
  including the spur road that really does meet it.

So the response carries `chains_bound` — `exact`, `lower_bound`, `approximate`
or `not_computed` — and `endpoint_quality`, the population behind that verdict.
Reporting "3 chains" without them would be inventing precision the stored data
cannot support, which is the exact failure this campaign exists to end.

**The tolerance decides the answer, so it is published.** Raise it and chains
merge; lower it and they split. `snap_tolerance` is therefore a field of the
response with `tolerance_basis` saying where it came from, and it is DERIVED
from the data — a fraction of the median segment length — because 0.05 is a
sane snap in metres and nonsense in inches, and 11 of the 18 drawings in this
store are in inches while 3 declare no unit at all (rule G2). No number in
drawing units is hard-coded here (rule G1). `tolerance_sensitivity` re-runs the
count at half and double so a reader can see for themselves how fragile it is.

Deliberately NOT done: an ARC's sweep can in principle be solved from its
stored `length` together with its bbox diagonal read as a chord, which would
place its endpoints — but only under the assumption that the arc does not cross
an axis extreme, and nothing stored can confirm that assumption. A number that
is right when an unverifiable premise holds is the shape of defect this file
exists to avoid. Open polylines were also left alone: `extract._ring_fields`
stores no vertex list for `ring_status == "open"`, and the diagnostic list it
does keep for bulged and oversize rings is truncated at the tail and does not
record whether the shape was closed, so its first and last entries are not
reliably endpoints.

Nothing here reads a layer name. This function is never told one (rule G1).
"""

from __future__ import annotations

import math
from typing import Any

#: Snap tolerance as a fraction of the median segment length. Dimensionless on
#: purpose: it means the same thing in metres, in inches, and in a drawing that
#: declares no unit. 0.1% of a median 70 m road segment is 7 cm — wide enough
#: for the rounding a CAD file accumulates, far narrower than any real gap.
SNAP_FRACTION = 0.001

#: Floor on the tolerance, relative to the largest coordinate magnitude seen.
#: Projected coordinates run to ~7e5, where a float64 ULP is ~1e-10; two ends
#: written by different code paths can differ in the last bits without being
#: separate points. Without this floor a layer whose entities all have zero
#: length would snap on exact equality alone.
FLOAT_NOISE_FACTOR = 1e-9

#: Stated limits (rule G7). Over them the answer is `None` with a reason and a
#: suggestion — never a silently truncated one. Σ length has no cap: it is a
#: single linear pass and there is no size at which it stops being exact.
MAX_CHAIN_ENTITIES = 200_000
MAX_POINT_COMPARISONS = 5_000_000
MAX_CHAIN_SIZES_REPORTED = 5

#: Factors the count is re-run at, so the reader can see the tolerance's grip.
SENSITIVITY_FACTORS = (0.5, 2.0)

#: Types whose geometry is closed by definition, so it has no free ends. Kept
#: deliberately short: SPLINE and ELLIPSE may be closed or open and nothing
#: stored says which, so they are treated as open-with-unknown-ends rather than
#: guessed at.
_ALWAYS_CLOSED_TYPES = frozenset({"CIRCLE"})

#: Types that are not path geometry at all. A network layer regularly carries a
#: stray label or block, and a TEXT does not have endpoints as a CATEGORY --
#: handing union-find its four bbox corners would let a label glue two roads
#: into one chain, and counting it as a component would report a label as a
#: road. Both are wrong, so these are excluded from the chain population and
#: the excluded count is published. The list is deliberately a DENY list: an
#: unrecognised type falls through to "path with ends we cannot locate", which
#: keeps it visible instead of quietly dropping it.
_NOT_PATH_TYPES = frozenset({
    "TEXT", "MTEXT", "ATTDEF", "ATTRIB", "INSERT", "POINT", "DIMENSION",
    "HATCH", "SOLID", "3DSOLID", "REGION", "IMAGE", "WIPEOUT", "VIEWPORT",
    "MESH", "BODY",
})


# --------------------------------------------------------------------------
# small readers, all of them refusing to guess
# --------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """A finite float, or None. `True` is not 1.0 and NaN is not a measurement."""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def _corners(bbox: Any) -> list[tuple[float, float]] | None:
    """The four corners of a stored bbox, deduplicated, or None if unusable."""
    if not isinstance(bbox, dict):
        return None
    lo, hi = bbox.get("min"), bbox.get("max")
    if not isinstance(lo, (list, tuple)) or not isinstance(hi, (list, tuple)):
        return None
    if len(lo) < 2 or len(hi) < 2:
        return None
    xs = [_number(lo[0]), _number(hi[0])]
    ys = [_number(lo[1]), _number(hi[1])]
    if any(v is None for v in xs + ys):
        return None
    x0, x1 = min(xs), max(xs)  # type: ignore[type-var]
    y0, y1 = min(ys), max(ys)  # type: ignore[type-var]
    seen: dict[tuple[float, float], None] = {}
    for pt in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        seen[pt] = None
    return list(seen)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _sig(value: float, digits: int = 9) -> float:
    """Round to significant figures, so the published tolerance is the one used.

    Rounding to a fixed number of decimals would flatten a 1e-12 tolerance to
    0.0 and make the response describe exact-match snapping that did not
    happen. The rounded value is what the union-find actually runs on.
    """
    if value == 0.0 or not math.isfinite(value):
        return 0.0
    return round(value, -int(math.floor(math.log10(abs(value)))) + (digits - 1))


def _fmt(value: float) -> str:
    return f"{value:.6g}"


# --------------------------------------------------------------------------
# endpoints, and how much of an endpoint each one really is
# --------------------------------------------------------------------------

_EXACT = "endpoints_exact"
_BOUNDED = "endpoints_bounded"
_UNKNOWN = "endpoints_unknown"
_CLOSED = "closed_no_ends"
_NOT_PATH = "not_path_geometry"
_NO_BBOX = "no_bbox"

#: Qualities whose entities take part in the chain count at all.
_IN_CHAINS = (_EXACT, _BOUNDED, _UNKNOWN, _CLOSED)


def _endpoint_candidates(
    row: dict, corners: list[tuple[float, float]] | None
) -> tuple[str, list[tuple[float, float]]]:
    """Candidate endpoints for one row, with an honest quality label.

    The label is the whole value of this function. The points it returns are
    fed to union-find either way; what changes between a LINE and an open
    polyline is how much the resulting number is worth, and only a label
    carries that downstream.
    """
    dxftype = row.get("type")
    dxftype = dxftype.upper() if isinstance(dxftype, str) else ""

    # Not path geometry: no ends to snap, and not a chain either.
    if dxftype in _NOT_PATH_TYPES:
        return _NOT_PATH, []

    if corners is None:
        return _NO_BBOX, []

    # A closed curve has no free ends at all. It is located, so it counts as a
    # component of its own; it offers no points, so it can never join.
    if dxftype in _ALWAYS_CLOSED_TYPES or row.get("ring_status") == "complete":
        return _CLOSED, []

    if dxftype == "LINE":
        # Degenerate box: horizontal, vertical or zero-length. The distinct
        # corners are the endpoints, with nothing inferred.
        if len(corners) <= 2:
            return _EXACT, corners
        # Real box: the endpoints are one diagonal of it and the file does not
        # record which. All four corners is the union of both hypotheses — a
        # superset of the truth, so no real join is lost.
        return _BOUNDED, corners

    # ARC, LWPOLYLINE, POLYLINE, SPLINE, anything else: the ends need not be at
    # a corner, and for a path that doubles back need not even touch the box
    # edge. Corners are a proxy and are labelled as one.
    return _UNKNOWN, corners


def _chains_at(
    points: list[tuple[int, float, float]],
    placed: int,
    tolerance: float,
    max_pairs: int,
) -> tuple[int, list[int]] | None:
    """Union-find over candidate points; returns (components, sizes) or None.

    None means the pairwise-comparison cap was hit — the points are packed more
    densely than the cap allows to scan. That is reported, never truncated.
    """
    parent = list(range(placed))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if tolerance > 0.0:
        grid: dict[tuple[int, int], list[int]] = {}
        for idx, (_gid, x, y) in enumerate(points):
            cell = (math.floor(x / tolerance), math.floor(y / tolerance))
            grid.setdefault(cell, []).append(idx)
        tol2 = tolerance * tolerance
        comparisons = 0
        for idx, (gid, x, y) in enumerate(points):
            cx = math.floor(x / tolerance)
            cy = math.floor(y / tolerance)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for jdx in grid.get((cx + dx, cy + dy), ()):
                        # Each unordered pair is reachable from both ends of it;
                        # this keeps exactly one visit.
                        if jdx <= idx:
                            continue
                        comparisons += 1
                        if comparisons > max_pairs:
                            return None
                        gid2, x2, y2 = points[jdx]
                        if gid2 == gid:
                            continue
                        if (x - x2) ** 2 + (y - y2) ** 2 <= tol2:
                            union(gid, gid2)
    else:
        # No scale could be derived from the data, so only exact coincidence
        # counts. This SPLITS rather than merges, which makes the count a
        # ceiling; `tolerance_basis` says so.
        first_at: dict[tuple[float, float], int] = {}
        for gid, x, y in points:
            key = (x, y)
            seen = first_at.get(key)
            if seen is None:
                first_at[key] = gid
            else:
                union(seen, gid)

    sizes: dict[int, int] = {}
    for i in range(placed):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    return len(sizes), sorted(sizes.values(), reverse=True)


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------


def network_measure(rows: list[dict], *, unit: str | None) -> dict:
    """Length total and connected-chain count for one layer x layout.

    rows: entity dicts carrying at least handle, type, length, bbox.
    Returns {"length_total", "measured_entities", "unmeasured_entities",
             "chains", "snap_tolerance", "tolerance_basis", "caveat"}
    and the evidence behind each of them.

    `length_total` is None when nothing carries a length -- never 0.0.
    `chains` is None when connectivity cannot be established at all; it is 0
    only for an empty input, where zero chains is arithmetic rather than a
    finding about geometry.

    `unit` is passed through untouched and may be None: 11 of the 18 drawings
    in this store are in inches and 3 declare nothing, so a bare number here
    would be a lie waiting to be quoted in metres (rule G2).
    """
    rows = list(rows or [])
    entities = len(rows)

    # ---- pass 1: length, and the population behind it --------------------
    lengths: list[float] = []
    positive_lengths: list[float] = []
    measured = 0
    zero_valued = 0
    malformed = 0
    closed_curve_entities = 0
    closed_curve_length = 0.0

    qualities: dict[str, int] = {
        _EXACT: 0, _BOUNDED: 0, _UNKNOWN: 0, _CLOSED: 0, _NOT_PATH: 0,
        _NO_BBOX: 0,
    }
    placed_points: list[tuple[int, float, float]] = []
    placed_count = 0
    diagonals: list[float] = []
    coord_magnitude = 0.0

    for row in rows:
        if not isinstance(row, dict):
            malformed += 1
            qualities[_NO_BBOX] += 1
            continue

        value = _number(row.get("length"))
        if value is not None:
            measured += 1
            lengths.append(value)
            if value == 0.0:
                zero_valued += 1
            else:
                positive_lengths.append(abs(value))

        corners = _corners(row.get("bbox"))
        quality, points = _endpoint_candidates(row, corners)
        qualities[quality] += 1

        if quality == _CLOSED and value is not None:
            closed_curve_entities += 1
            closed_curve_length += value

        span: float | None = None
        if corners is not None:
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            coord_magnitude = max(
                [coord_magnitude] + [abs(v) for v in xs] + [abs(v) for v in ys]
            )
            span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))

        if quality not in _IN_CHAINS:
            continue

        gid = placed_count
        placed_count += 1
        for x, y in points:
            placed_points.append((gid, x, y))

        # The fallback scale is taken from path geometry only. A network layer
        # carrying 500 text labels and 3 unmeasured lines would otherwise set
        # its snap tolerance from the size of the labels.
        if span:
            diagonals.append(span)

    unmeasured = entities - measured
    length_total = round(math.fsum(lengths), 6) if measured else None
    length_is_floor = bool(measured and unmeasured)

    # ---- the tolerance, derived from the data and then published ---------
    scale = _median(positive_lengths)
    scale_source = "the median non-zero entity length"
    scale_population = len(positive_lengths)
    if scale is None:
        scale = _median(diagonals)
        scale_source = "the median non-zero bounding-box diagonal"
        scale_population = len(diagonals)
    if scale is None:
        scale_source = "nothing in the rows carries a usable scale"
        scale_population = 0

    from_scale = SNAP_FRACTION * scale if scale else 0.0
    from_noise = FLOAT_NOISE_FACTOR * coord_magnitude
    tolerance = _sig(max(from_scale, from_noise))

    unit_label = unit.strip() if isinstance(unit, str) and unit.strip() else None
    unit_phrase = (
        f"in {unit_label}"
        if unit_label
        else "in drawing units (this drawing declares none)"
    )

    if entities == 0:
        tolerance_basis = (
            "No entities were supplied, so no scale could be derived and no "
            "snapping was done."
        )
    elif tolerance == 0.0:
        tolerance_basis = (
            "No snap tolerance could be derived: " + scale_source + ", and no "
            "coordinate has a magnitude to take float noise from. Endpoints "
            "were joined on exact coincidence only, which splits rather than "
            "merges, so the chain count is a ceiling and not a floor."
        )
    else:
        which = "the scale term" if from_scale >= from_noise else "the float-noise floor"
        tolerance_basis = (
            f"snap_tolerance = {_fmt(tolerance)} {unit_phrase}, the larger of "
            f"{SNAP_FRACTION:g} x {_fmt(scale or 0.0)} ({_fmt(from_scale)}, from "
            f"{scale_source} over {scale_population} entities) and "
            f"{FLOAT_NOISE_FACTOR:g} x {_fmt(coord_magnitude)} "
            f"({_fmt(from_noise)}, a float-noise floor taken from the largest "
            f"coordinate magnitude); {which} decided it. The fraction is "
            "dimensionless on purpose, so the tolerance follows the drawing "
            "instead of assuming metres. Two endpoints are joined when they "
            "are this far apart or closer. Raising it merges chains and "
            "lowering it splits them, which is why it is published rather "
            "than assumed."
        )

    # ---- pass 2: connectivity, at the tolerance and around it ------------
    caps_hit: list[str] = []
    chains: int | None
    chain_sizes: list[int] = []

    if entities == 0:
        chains = 0
        chains_basis = (
            "No entities were supplied. Zero chains is arithmetic over an "
            "empty set, not a finding about geometry."
        )
    elif placed_count == 0:
        chains = None
        chains_basis = (
            f"Not one of the {entities} rows is locatable path geometry -- no "
            f"usable bounding box ({qualities[_NO_BBOX]}), not path geometry "
            f"at all ({qualities[_NOT_PATH]}) -- so no connectivity can be "
            "computed. This is not zero chains."
        )
    elif placed_count > MAX_CHAIN_ENTITIES:
        chains = None
        caps_hit.append("max_entities_for_chains")
        chains_basis = (
            f"{placed_count} locatable entities exceed the stated cap of "
            f"{MAX_CHAIN_ENTITIES} for chain counting, so no count is given "
            "rather than a truncated one. Narrow the input by layout or by "
            "spatial extent and measure the parts. The length total above is "
            "unaffected: it is a single pass and has no cap."
        )
    else:
        result = _chains_at(
            placed_points, placed_count, tolerance, MAX_POINT_COMPARISONS
        )
        if result is None:
            chains = None
            caps_hit.append("max_point_comparisons_per_pass")
            chains_basis = (
                "The candidate endpoints are packed more densely than the "
                f"stated cap of {MAX_POINT_COMPARISONS} pairwise comparisons "
                "allows to scan, so no count is given rather than a partial "
                "one. Narrow the input, or re-run on a smaller extent."
            )
        else:
            chains, chain_sizes = result
            chains_basis = _describe_chains(qualities, placed_count, entities)

    sensitivity: list[dict[str, Any]] = []
    if chains is not None and placed_count:
        sensitivity.append(
            {"factor": 1.0, "snap_tolerance": tolerance, "chains": chains}
        )
        if tolerance > 0.0:
            for factor in SENSITIVITY_FACTORS:
                alt_tol = _sig(tolerance * factor)
                alt = _chains_at(
                    placed_points, placed_count, alt_tol, MAX_POINT_COMPARISONS
                )
                entry: dict[str, Any] = {
                    "factor": factor,
                    "snap_tolerance": alt_tol,
                    "chains": None if alt is None else alt[0],
                }
                if alt is None:
                    entry["note"] = (
                        "the cap on pairwise comparisons was hit at this "
                        "tolerance, so no count is given for it"
                    )
                sensitivity.append(entry)

    # ---- the verdict on how much the chain count is worth ----------------
    if chains is None:
        chains_bound = "not_computed"
    elif entities == 0:
        chains_bound = "exact"
    elif qualities[_UNKNOWN] or qualities[_CLOSED] or qualities[_NO_BBOX]:
        chains_bound = "approximate"
    elif qualities[_BOUNDED]:
        chains_bound = "lower_bound"
    else:
        chains_bound = "exact"

    sizes_top = chain_sizes[:MAX_CHAIN_SIZES_REPORTED]
    sizes_truncated = max(0, len(chain_sizes) - len(sizes_top))

    # ---- everything a reader would need to argue with the numbers --------
    caveats: list[str] = []
    if entities == 0:
        caveats.append(
            "No entities were supplied; there is nothing here to measure."
        )
    if malformed:
        caveats.append(
            f"Of {entities} rows, {malformed} are not entity documents at all; "
            "they were counted as unmeasured and unlocatable, not dropped."
        )
    if length_is_floor:
        caveats.append(
            "length_total is a FLOOR, not a total: it omits "
            f"{unmeasured} of {entities} entities that carry no length, and "
            "those were counted rather than zeroed."
        )
    if measured == 0 and entities:
        caveats.append(
            f"None of the {entities} entities carries a length, so length_total "
            "is null. That is not a measurement of zero."
        )
    if closed_curve_entities:
        caveats.append(
            f"Closed curves ({closed_curve_entities} of {entities}) contribute "
            f"{round(closed_curve_length, 6)} to length_total as CIRCUMFERENCE, "
            "which is not a run length. Having no free ends they also cannot "
            "join anything, so each is its own chain even where a path really "
            "does meet it."
        )
    if chains is not None and (qualities[_UNKNOWN] or qualities[_BOUNDED]):
        inferred = []
        if qualities[_UNKNOWN]:
            inferred.append(
                "ends a bounding box cannot locate at all "
                f"({qualities[_UNKNOWN]})"
            )
        if qualities[_BOUNDED]:
            inferred.append(
                "ends on one of two box diagonals with no record of which "
                f"({qualities[_BOUNDED]})"
            )
        caveats.append(
            "chains is inferred, not measured: the rows carry no endpoint "
            f"coordinates. Of {entities} entities -- "
            + "; ".join(inferred) + "."
        )
    if chains is not None and (qualities[_NO_BBOX] or qualities[_NOT_PATH]):
        excluded = []
        if qualities[_NO_BBOX]:
            excluded.append(f"no usable bounding box ({qualities[_NO_BBOX]})")
        if qualities[_NOT_PATH]:
            excluded.append(
                "not path geometry -- text, blocks, dimensions and the like "
                f"({qualities[_NOT_PATH]})"
            )
        caveats.append(
            f"The chain count covers {placed_count} of {entities} entities. "
            "Excluded: " + "; ".join(excluded) + "."
        )
    spread = {s["chains"] for s in sensitivity}
    if len(spread) > 1:
        detail = ", ".join(
            f"{s['chains']} at x{s['factor']:g}" for s in sensitivity
        )
        caveats.append(
            "The chain count moves with the tolerance: " + detail + "."
        )
    if unit_label is None and (length_total is not None or chains is not None):
        caveats.append(
            "This drawing declares no unit. Every number here is in drawing "
            "units and must not be labelled with one."
        )

    return {
        # what came in
        "entities": entities,
        "malformed_rows": malformed,
        "unit": unit_label,
        "unit_note": (
            f"the drawing declares its unit as {unit_label!r}; every length "
            "here is in that unit"
            if unit_label
            else "the drawing declares no unit; every number here is in "
            "drawing units and must not be labelled with one"
        ),
        # the measurement
        "length_total": length_total,
        "length_is_floor": length_is_floor,
        "length_note": _describe_length(measured, unmeasured, entities),
        "measured_entities": measured,
        "unmeasured_entities": unmeasured,
        "zero_length_entities": zero_valued,
        "closed_curve_entities": closed_curve_entities,
        "closed_curve_length": (
            round(closed_curve_length, 6) if closed_curve_entities else None
        ),
        # the inference
        "chains": chains,
        "chains_basis": chains_basis,
        "chains_bound": chains_bound,
        "chained_entities": placed_count,
        "unplaced_entities": qualities[_NO_BBOX],
        "non_path_entities": qualities[_NOT_PATH],
        "largest_chain": chain_sizes[0] if chain_sizes else None,
        "singleton_chains": sum(1 for s in chain_sizes if s == 1),
        "chain_sizes_top": sizes_top,
        "chain_sizes_truncated": sizes_truncated,
        "endpoint_quality": dict(qualities),
        # the knob that decided the inference
        "snap_tolerance": tolerance if entities else None,
        "tolerance_basis": tolerance_basis,
        "tolerance_sensitivity": sensitivity,
        # the stated limits
        "caps": {
            "max_entities_for_chains": MAX_CHAIN_ENTITIES,
            "max_point_comparisons_per_pass": MAX_POINT_COMPARISONS,
            "max_chain_sizes_reported": MAX_CHAIN_SIZES_REPORTED,
            "max_entities_for_length": None,
            "hit": caps_hit,
        },
        "caveat": " ".join(caveats),
        "caveats": caveats,
    }


def _describe_length(measured: int, unmeasured: int, entities: int) -> str:
    if entities == 0:
        return "No entities were supplied, so there is nothing to sum."
    if measured == 0:
        return (
            f"None of the {entities} entities carries a length, so there is no "
            "total to report. Null, not zero: a layer of text has no length, "
            "and zero would claim it was measured and found to be nothing."
        )
    if unmeasured == 0:
        return f"Summed over all {entities} entities; none was skipped."
    return (
        f"Summed over {measured} of {entities} entities. The gap is "
        f"{unmeasured} without a stored length, counted rather than zeroed, so "
        "this figure is a floor and must not be quoted as the layer's total."
    )


def _describe_chains(qualities: dict[str, int], placed: int, entities: int) -> str:
    """Say, in one paragraph, exactly what the chain number rests on."""
    head = (
        "Union-find over candidate endpoints: two entities are joined when a "
        "candidate point of one lies within snap_tolerance of a candidate "
        "point of the other, and a chain is a connected component of that "
        f"relation over {placed} of {entities} entities. The stored rows carry "
        "no endpoint coordinates, so the candidates are inferred from each "
        "entity's bounding box"
    )
    # Written as a legend -- "description (count)" -- so it reads correctly at
    # a count of one as well as at a count of eight hundred.
    legend = (
        (_EXACT, "LINE with a degenerate box, whose two corners ARE its "
                 "endpoints, so they are exact"),
        (_BOUNDED, "LINE with a real box, whose endpoints are one of the box's "
                   "two diagonals with no record of which, so all four corners "
                   "are offered -- a superset of the truth, which can invent a "
                   "join but can never miss one"),
        (_UNKNOWN, "ARC, polyline or other path whose ends a bounding box does "
                   "not identify at all, so its corners are a proxy and joins "
                   "can be both missed and invented"),
        (_CLOSED, "closed curve with no free ends, so it is its own component "
                  "and cannot be joined to a path that really touches it"),
        (_NOT_PATH, "not path geometry (text, blocks, dimensions and the like), "
                    "excluded so that a label cannot glue two chains together "
                    "and is not itself counted as one"),
        (_NO_BBOX, "no usable bounding box, excluded from the count entirely"),
    )
    parts = [
        f"{text} ({qualities[key]})" for key, text in legend if qualities[key]
    ]
    body = head + ": " + "; ".join(parts) + "."
    if qualities[_UNKNOWN] or qualities[_CLOSED] or qualities[_NO_BBOX]:
        tail = (
            " The count is therefore an APPROXIMATION with no guaranteed "
            "direction, and must be quoted as one."
        )
    elif qualities[_BOUNDED]:
        tail = (
            " Every candidate set is a superset of the real endpoints, so the "
            "count can only be too low: it is a LOWER BOUND on the true number "
            "of chains at this tolerance."
        )
    else:
        tail = " Every endpoint is exact, so the count is exact at this tolerance."
    return body + tail
