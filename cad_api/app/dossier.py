"""The Drawing Dossier -- profiler core and assembler.

Owner: lane 1 of DOSSIER Phase 1 (`docs/DOSSIER-01-DESIGN.md`).

This module answers one question per (layer x layout) bucket -- *what is this,
geometrically, and how much of it is there* -- and then assembles those answers
into the one document per drawing described in the design doc.

Three properties are structural, not stylistic, and none of them may be traded
away for convenience:

1.  **Pure over already-fetched rows.** Nothing here opens a database, reads a
    DXF, or imports `app.main`. `profile_rows` takes a list of dicts and returns
    a dict. That is what lets the whole profiler be tested with fixtures, run
    inside an ingest, a backfill or an MCP tool, and read the store exactly
    once. There is a test that greps this file's own source to keep it true.

2.  **Role is geometry, never a name** (rule G1). The decision reads `type` and
    `ring_status` and nothing else. A layer called after a road that holds 900
    closed rings profiles as `region`, and it is *supposed* to -- that
    disagreement is a finding, not a bug. The dominance threshold is the named
    constant `ROLE_DOMINANCE`, its value is printed inside every `role_basis`,
    and the full share of every family travels in `role_shares` so a reader can
    see how close the call was instead of trusting the verdict.

3.  **An absent measurement is `None` and a reason, never `0`** (rules G2, G8).
    42 of the reference road layer's 1,233 entities carry no length. They are
    counted, named in `unmeasured_entities`, and reported beside the total. A
    sum over nothing is `None`, because 0.0 is a claim and "nothing was
    measurable" is a different claim. No number is ever stamped "m" because the
    drawing that happened to be open used metres.

The coverage invariant is the campaign's whole claim:

    sum(block["entities"] for every layer x layout) == drawing.entity_total

It is computed in `build_dossier`, published in the `coverage` block, and it
**cannot pass by accident**: an unknown total, a caller total that disagrees
with the drawing's own, and a bucket that went missing all end at
`complete: false`. `assert_coverage()` turns that into an exception for the
parametric test and for the backfill.

Lanes 2, 3 and 4 are reached through guarded imports exactly as the interface
contract says. A missing sibling produces `"not_computed": "<why>"` in its part
of the document; it never raises, and it never quietly becomes a zero either.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

# --- The guarded seams to the sibling lanes ----------------------------------
#
# `except Exception` and not `except ImportError` on purpose: a sibling that is
# ABSENT and a sibling that is HALF-WRITTEN are the same situation for this
# module -- the Dossier still has to build -- and the difference between them is
# worth reporting rather than swallowing. The reason string travels into the
# document, so "lane 2 has a SyntaxError" reads differently from "lane 2 has not
# been written", which is the point.

try:  # lane 2 -- Sigma length and connected chains
    from . import dossier_network as _NETWORK

    _NETWORK_WHY: str | None = None
except Exception as exc:  # pragma: no cover - depends on sibling lane presence
    _NETWORK = None  # type: ignore[assignment]
    _NETWORK_WHY = f"{type(exc).__name__}: {exc}"

try:  # lane 3 -- duplicate clusters and other oddities
    from . import dossier_anomaly as _ANOMALY

    _ANOMALY_WHY: str | None = None
except Exception as exc:  # pragma: no cover - depends on sibling lane presence
    _ANOMALY = None  # type: ignore[assignment]
    _ANOMALY_WHY = f"{type(exc).__name__}: {exc}"

try:  # lane 4 -- block-name and annotation censuses
    from . import dossier_census as _CENSUS

    _CENSUS_WHY: str | None = None
except Exception as exc:  # pragma: no cover - depends on sibling lane presence
    _CENSUS = None  # type: ignore[assignment]
    _CENSUS_WHY = f"{type(exc).__name__}: {exc}"


# --- Stated constants (G1, G7) -----------------------------------------------

#: Schema version of the document this module emits. Bumped when a key changes
#: meaning, never when a key is added -- enrichment is additive by contract.
DOSSIER_VERSION: int = 1

#: The share of a bucket one type family must hold before it names the bucket's
#: role. Stated here, echoed in every `role_basis`, and never written as a bare
#: number inside a branch.
#:
#: Why 0.6 and not 0.5. Two reasons, and the second one is the real one:
#:
#: * A simple majority describes a bucket badly. At 0.51 the winning family is
#:   outnumbered by everything else put together, and calling that bucket
#:   `region` hides half of what is on it. Three of every five is a claim worth
#:   printing; one of every two is not.
#: * **Any threshold at or below 0.5 admits ties, and a tie has no honest
#:   winner.** Two families at exactly 0.5 would both qualify and the role would
#:   fall to whatever the tie-break happened to be -- an arbitrary answer
#:   wearing the same clothes as a measured one. Above 0.5 at most one family
#:   can ever qualify, so the decision is unique by arithmetic rather than by
#:   convention.
#:
#: Buckets that do not reach it are `mixed`, and `mixed` says which family led
#: and by how much rather than pretending the question was unanswerable.
ROLE_DOMINANCE: float = 0.60

#: Types reported by name in `types` before truncation begins (G7). What is
#: dropped is counted and stated in `types_truncation`; `entities` is never
#: computed from this map, so truncation can never move the coverage figure.
MAX_TYPES_REPORTED: int = 40

#: Items listed by name in the file-level fact lists (xrefs, text styles) before
#: truncation begins (G7). Same rule: the totals are counted in full, only the
#: naming is capped.
MAX_FACT_ITEMS: int = 25

#: Published numbers are rounded here, matching `store.sum_measured_only`.
#: Precision below this is float noise, not measurement.
OUTPUT_DECIMALS: int = 6

#: Polyline types that can carry a ring. A ring that closed cleanly
#: (`ring_status == "complete"`) is region geometry; the same type left open is
#: network geometry. The type alone decides nothing.
RING_TYPES = frozenset({"LWPOLYLINE", "POLYLINE"})

#: Types whose geometry is a path with a run length. `CIRCLE` and `ELLIPSE` are
#: here deliberately although they are closed: they carry no stored ring and no
#: area, and the existing `measure` already sums their circumference into a
#: layer length. Keeping them in this family is what lets the caveat that
#: travels with that total -- a circumference is not a run length -- be stated
#: instead of quietly being true.
PATH_TYPES = frozenset(
    {
        "LINE",
        "ARC",
        "CIRCLE",
        "ELLIPSE",
        "LWPOLYLINE",
        "POLYLINE",
        "SPLINE",
        "HELIX",
        "RAY",
        "XLINE",
    }
)

#: Types that mark a position. `INSERT` is the one the design doc names; `POINT`
#: is here because a POINT is a point by definition and pushing a layer of them
#: into `mixed` would lose that fact for no gain. Which types were counted is
#: named in `role_basis`, so the choice is inspectable rather than implied.
POINT_TYPES = frozenset({"INSERT", "POINT"})

#: Types that carry human-readable content rather than geometry. Every DXF
#: dimension variant begins with `DIMENSION`, so the prefix is tested as well.
ANNOTATION_TYPES = frozenset(
    {
        "TEXT",
        "MTEXT",
        "ATTRIB",
        "ATTDEF",
        "LEADER",
        "MLEADER",
        "MULTILEADER",
        "DIMENSION",
        "ARC_DIMENSION",
        "TOLERANCE",
    }
)

#: The four families that can name a role. `other` is counted and reported like
#: the rest but can never win: there is no role called "other" in the design,
#: and inventing one would turn "we did not recognise this" into a description.
ROLE_FAMILIES: tuple[str, ...] = ("region", "network", "points", "annotation")

#: The fifth bucket. HATCH, SOLID, 3DSOLID, VIEWPORT, IMAGE, WIPEOUT and
#: everything else land here. It is not a waste bin -- it is counted, it drags
#: the winning share down exactly as it should, and a bucket that is mostly
#: `other` is honestly reported as `mixed`.
OTHER_FAMILY: str = "other"

#: Label given to a row that carries no `type` at all. It is still counted: a
#: row we cannot classify is a row that exists (G8).
UNKNOWN_TYPE: str = "(type missing)"

_MIXED: str = "mixed"


class CoverageError(AssertionError):
    """The coverage invariant did not hold, with the arithmetic attached.

    Derived from `AssertionError` so that a test that forgets `pytest.raises`
    still fails rather than passing on a truthy exception object, and so a
    backfill that lets it escape stops rather than storing a document that
    claims a completeness it does not have.
    """

    def __init__(
        self,
        message: str,
        *,
        accounted: int | None = None,
        total: int | None = None,
        difference: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.accounted = accounted
        self.total = total
        self.difference = difference


# --- Small helpers ------------------------------------------------------------


def _field(row: Any, key: str) -> Any:
    """One field of one row, tolerating a row that is not a mapping.

    A malformed row is counted, never dropped: `entities` is `len(rows)` and
    nothing below is allowed to change that.
    """
    if isinstance(row, Mapping):
        return row.get(key)
    return None


def _number(value: Any) -> float | None:
    """A real, finite number, or None. `bool` is an `int` and is not a measure."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _round(value: float) -> float:
    return round(float(value), OUTPUT_DECIMALS)


def _type_of(row: Any) -> str:
    raw = _field(row, "type")
    if raw is None or raw == "":
        return UNKNOWN_TYPE
    return str(raw)


def _pct(share: float) -> float:
    return round(share * 100.0, 1)


_NO_UNIT_REASON = (
    "The caller stated no unit for this layer x layout, so none is claimed. "
    "$INSUNITS is a drawing-header field describing model space only and DXF "
    "carries no per-layout unit, so a sheet or a block definition gets no unit "
    "rather than the drawing's."
)

_COUNT_UNIT_REASON = (
    "A count of objects is not a length or an area, so it carries no drawing "
    "unit."
)


def _area_unit(unit: str | None) -> str | None:
    """The area unit derived from a length unit, or None.

    Follows `store._unit_names`: the area unit is the length unit with a `2`
    appended. When there is no length unit there is no area unit -- squaring an
    unknown does not produce a known.
    """
    if not unit:
        return None
    return f"{unit}2"


def _unit_pair(
    unit: Any, unit_reason: str | None
) -> tuple[str | None, str | None]:
    """Normalise a caller's unit into `(unit, reason_when_absent)`.

    Accepts either a plain name (`"m"`) or a whole `store._unit_names()` dict,
    because the integrator already holds one of those per layout and copying
    fields out of it by hand is how a sheet ends up wearing model space's unit.
    """
    if isinstance(unit, Mapping):
        name = unit.get("length_unit")
        reason = unit.get("why_no_unit") or unit_reason
        if name:
            return str(name), None
        return None, str(reason or _NO_UNIT_REASON)
    if unit:
        return str(unit), None
    return None, str(unit_reason or _NO_UNIT_REASON)


def _accepts(func: Any, name: str) -> bool:
    """Whether `func` will take a keyword called `name`.

    The interface contract fixes the minimum signature of each lane; a lane is
    free to accept MORE. A unit is one of those: a duplicate cluster measured
    "1778.387 x 1585.99" means nothing without it. Asking the signature rather
    than passing hopefully and catching TypeError keeps a genuine failure
    inside the lane distinguishable from a keyword it never wanted.
    """
    try:
        import inspect

        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):  # a builtin or an unreadable callable
        return False
    if name in parameters:
        return True
    return any(
        p.kind is p.VAR_KEYWORD for p in parameters.values()
    )


def _lane_call(
    module: Any,
    module_name: str,
    func_name: str,
    why_missing: str | None,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, str | None]:
    """Call a sibling lane, returning `(result, not_computed_reason)`.

    Never raises. A lane that is missing, that has not written this function
    yet, or that blew up on this particular bucket all produce a reason string
    that is published in the document. The Dossier is still built, the hole is
    named, and nothing anywhere becomes a zero because a lane was unavailable.
    """
    if module is None:
        detail = f" ({why_missing})" if why_missing else ""
        return None, f"{module_name} not available{detail}"
    func = getattr(module, func_name, None)
    if not callable(func):
        return None, (
            f"{module_name}.{func_name} not available (the module exists but "
            "does not define it)"
        )
    try:
        return func(*args, **kwargs), None
    except Exception as exc:  # a sibling lane must not break the assembler
        return None, f"{module_name}.{func_name} raised {type(exc).__name__}: {exc}"


# --- Role: decided from geometry, never from a name ---------------------------


def classify_entity(row: Any) -> str:
    """Which type family one entity belongs to. Reads geometry only.

    The order below is the decision, and it is exclusive -- every entity lands
    in exactly one family, which is what makes the family counts add up to the
    bucket and the bucket add up to the drawing.

    1.  A polyline whose ring closed cleanly is `region`. `ring_status` is the
        gate, not the type: the same LWPOLYLINE left open is a path.
    2.  A block reference or a point marker is `points`.
    3.  Text, dimensions and leaders are `annotation`.
    4.  Any remaining path type is `network` -- **whether or not it carries a
        length**. The design doc's shorthand for this family is "open paths
        with a length", but membership here is decided by geometry alone and
        measurability is reported separately, because an entity that could not
        be measured is still an open path. Folding the 42 unmeasured entities
        of the reference road layer out of the family would make the measure
        block read "1,191 of 1,191" and the hole would vanish from the account
        -- which is the exact failure this campaign exists to end (G8).
    5.  Everything else is `other`: hatches, solids, viewports, images, and any
        type nobody has met yet. Counted, never discarded.
    """
    dxftype = _type_of(row)
    if dxftype in RING_TYPES and _field(row, "ring_status") == "complete":
        return "region"
    if dxftype in POINT_TYPES:
        return "points"
    if dxftype in ANNOTATION_TYPES or dxftype.startswith("DIMENSION"):
        return "annotation"
    if dxftype in PATH_TYPES:
        return "network"
    return OTHER_FAMILY


def family_counts(rows: Sequence[Any]) -> dict[str, int]:
    """Entities per type family, every family present even at zero.

    Always five keys, so a reader never has to tell "this family is absent"
    apart from "this family was not looked for".
    """
    counts = {name: 0 for name in ROLE_FAMILIES}
    counts[OTHER_FAMILY] = 0
    for row in rows:
        counts[classify_entity(row)] += 1
    return counts


def decide_role(
    counts: Mapping[str, int], total: int
) -> tuple[str | None, str, dict[str, float]]:
    """`(role, role_basis, role_shares)` from family counts alone.

    `role` is None only for an empty bucket -- absence answered with `None` and
    a reason rather than with a role that was never observed (G8). Otherwise it
    is one of the four named families or `mixed`.

    The returned sentence always names the threshold, the winning share and the
    runner-up, so the reader can see a 61% call and a 99% call are not the same
    call even though they produce the same word.
    """
    tail = (
        "Role is decided from DXF type and ring status alone; the layer name is "
        "never read (rule G1)."
    )
    if total <= 0:
        return None, (
            "This layer x layout holds 0 entities, so no geometric role can be "
            f"decided and none is claimed. {tail}"
        ), {}

    shares = {name: count / total for name, count in counts.items()}
    ordered = sorted(shares.items(), key=lambda kv: (-kv[1], kv[0]))
    leader, leader_share = ordered[0]
    runner, runner_share = ordered[1] if len(ordered) > 1 else ("none", 0.0)
    published = {name: round(share, 6) for name, share in shares.items()}
    threshold_text = (
        f"the stated dominance threshold of {_pct(ROLE_DOMINANCE)}% "
        f"(ROLE_DOMINANCE={ROLE_DOMINANCE})"
    )

    if leader in ROLE_FAMILIES and leader_share >= ROLE_DOMINANCE:
        basis = (
            f"{leader} holds {counts[leader]} of {total} entities "
            f"({_pct(leader_share)}%), at or above {threshold_text}; the "
            f"runner-up is {runner} at {_pct(runner_share)}%. "
            f"{_family_membership_note(leader)} {tail}"
        )
        return leader, basis, published

    basis = (
        f"No type family reaches {threshold_text}: the largest is {leader} at "
        f"{counts[leader]} of {total} entities ({_pct(leader_share)}%), "
        f"followed by {runner} at {_pct(runner_share)}%. This bucket is "
        f"reported as {_MIXED} and carries a count only, because no single "
        f"native measure describes it. {tail}"
    )
    return _MIXED, basis, published


def _family_membership_note(family: str) -> str:
    """One clause naming what was counted into the winning family."""
    if family == "region":
        return (
            "Counted as region: LWPOLYLINE/POLYLINE whose ring_status is "
            "'complete'."
        )
    if family == "network":
        return (
            "Counted as network: path geometry that is not a closed ring "
            "(LINE, ARC, CIRCLE, ELLIPSE, SPLINE and open polylines), whether "
            "or not it carries a stored length."
        )
    if family == "points":
        return "Counted as points: INSERT and POINT."
    if family == "annotation":
        return (
            "Counted as annotation: TEXT, MTEXT, ATTRIB, ATTDEF, leaders and "
            "every DIMENSION variant."
        )
    return ""


# --- The type map, with its cap stated (G7) -----------------------------------


def type_mix(rows: Sequence[Any]) -> tuple[dict[str, int], dict[str, Any]]:
    """`(types, types_truncation)` -- counts per DXF type, largest first.

    Truncation is always reported, with `types_dropped: 0` on the ordinary
    path, so a consumer never has to branch on whether the key exists to learn
    whether it is looking at the whole picture.
    """
    counter = Counter(_type_of(row) for row in rows)
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = ordered[:MAX_TYPES_REPORTED]
    dropped = ordered[MAX_TYPES_REPORTED:]
    truncation = {
        "cap": MAX_TYPES_REPORTED,
        "types_present": len(ordered),
        "types_reported": len(kept),
        "types_dropped": len(dropped),
        "entities_dropped": sum(count for _, count in dropped),
        "basis": (
            "Types are listed largest first and capped at "
            f"{MAX_TYPES_REPORTED} (G7). The bucket's `entities` figure is "
            "len(rows) and is never derived from this map, so truncating it "
            "cannot move the coverage arithmetic."
        ),
    }
    return dict(kept), truncation


# --- Measures -----------------------------------------------------------------


def _count_measure(
    rows: Sequence[Any], *, role: str | None, note: str | None = None
) -> dict[str, Any]:
    """A bucket whose native measure is its own count.

    `unmeasured_entities` is 0 here and that is not a silent zero: counting
    cannot fail. The `basis` says so, which is the difference between a zero
    that was established and a zero that was assumed.
    """
    total = len(rows)
    block: dict[str, Any] = {
        "kind": "count",
        "value": total,
        "unit": None,
        "unit_reason": _COUNT_UNIT_REASON,
        "measured_entities": total,
        "unmeasured_entities": 0,
        "bucket_entities": total,
        "basis": (
            f"Every entity in this layer x layout is counted; counting cannot "
            f"fail, so nothing is unmeasured. Native measure for role "
            f"{role!r} is a count."
        ),
        "caveat": None,
    }
    if note:
        block["why_no_native_measure"] = note
    return block


def _area_measure(
    rows: Sequence[Any], *, unit: str | None, unit_reason: str | None
) -> dict[str, Any]:
    """Sigma area over a region bucket, with the hole in it counted.

    Summed over every row that carries a finite `area`, not only over the rows
    that classified as `region`: an entity that has a stored area contributes
    it, and restricting the sum to the winning family would drop real area on
    the floor of a bucket that is 80% rings. What the total does and does not
    cover is stated rather than implied.
    """
    values: list[float] = []
    negatives = 0
    for row in rows:
        area = _number(_field(row, "area"))
        if area is None:
            continue
        if area < 0:
            negatives += 1
        values.append(area)

    total = len(rows)
    measured = len(values)
    unmeasured = total - measured
    unit_name = _area_unit(unit)
    unit_why = None if unit_name else (unit_reason or _NO_UNIT_REASON)

    caveats = [
        f"{unmeasured} of {total} entities in this layer x layout carry no "
        "stored area and are counted here, not summed as zero."
    ] if unmeasured else []
    if negatives:
        caveats.append(
            f"{negatives} stored areas are negative (a ring wound clockwise); "
            "they are summed exactly as stored and are not silently made "
            "positive."
        )

    return {
        "kind": "area",
        "value": _round(math.fsum(values)) if measured else None,
        "unit": unit_name,
        "unit_reason": unit_why,
        "measured_entities": measured,
        "unmeasured_entities": unmeasured,
        "bucket_entities": total,
        "basis": (
            f"Exact summation (math.fsum) of the stored `area` field over "
            f"{measured} of {total} entities in this layer x layout. A bucket "
            "with nothing measurable reports None, never 0.0."
        ),
        "caveat": " ".join(caveats) or None,
    }


def _length_measure(
    rows: Sequence[Any], *, unit: str | None, unit_reason: str | None
) -> dict[str, Any]:
    """Sigma length and chain count, delegated to lane 2 behind the guard.

    Lane 2 is handed the **whole bucket**, not just the network family. That is
    the seam the design doc's own example fixes: 1,191 measured and 42
    unmeasured add up to the layer's 1,233, and slicing the family out first
    would silently change the denominator of a published number.
    """
    total = len(rows)
    unit_name = unit
    unit_why = None if unit_name else (unit_reason or _NO_UNIT_REASON)
    block: dict[str, Any] = {
        "kind": "length",
        "value": None,
        "unit": unit_name,
        "unit_reason": unit_why,
        "measured_entities": None,
        "unmeasured_entities": None,
        "bucket_entities": total,
        "chains": None,
        "snap_tolerance": None,
        "tolerance_basis": None,
        "basis": (
            "Sigma length and connected-chain count over every entity in this "
            "layer x layout, computed by dossier_network.network_measure "
            "(lane 2)."
        ),
        "caveat": None,
        "source": "dossier_network.network_measure",
    }

    result, why = _lane_call(
        _NETWORK, "dossier_network", "network_measure", _NETWORK_WHY,
        list(rows), unit=unit_name,
    )
    if why is not None:
        block["not_computed"] = why
        block["basis"] = (
            "No length was summed: lane 2 did not answer. The entity count "
            "above is still exact; the length is absent, not zero."
        )
        return block
    if not isinstance(result, Mapping):
        block["not_computed"] = (
            "dossier_network.network_measure returned "
            f"{type(result).__name__}, not a mapping"
        )
        return block

    block["value"] = result.get("length_total")
    block["measured_entities"] = result.get("measured_entities")
    block["unmeasured_entities"] = result.get("unmeasured_entities")
    block["chains"] = result.get("chains")
    block["snap_tolerance"] = result.get("snap_tolerance")
    block["tolerance_basis"] = result.get("tolerance_basis")
    block["caveat"] = _caveat_of(result)
    #: A sum taken over some of the entities is a FLOOR, not a total, and lane 2
    #: is the only thing that knows which it produced. Lifted to the top of the
    #: measure so a reader meets it beside the number rather than three keys
    #: down.
    block["value_is_floor"] = result.get("length_is_floor")
    #: Lane 2 answers with more than the contract's seven fields -- endpoint
    #: quality, tolerance sensitivity, closed-curve length, its own caps. None
    #: of it is dropped (G7): the flat fields above are the summary, this is
    #: everything lane 2 said.
    block["detail"] = dict(result)
    return block


def _caveat_of(result: Mapping[str, Any]) -> str | None:
    """One caveat string from a lane that may publish `caveat`, `caveats`, or both.

    An empty string is not a caveat; it is the absence of one, and it must not
    be published as though something had been said.
    """
    parts: list[str] = []
    single = result.get("caveat")
    if isinstance(single, str) and single.strip():
        parts.append(single.strip())
    many = result.get("caveats")
    if isinstance(many, (list, tuple)):
        for item in many:
            text = str(item).strip()
            if text and text not in parts:
                parts.append(text)
    return " ".join(parts) or None


# --- The per-bucket profile ---------------------------------------------------


def profile_rows(
    rows: Sequence[Any],
    *,
    layer: str,
    layout: str,
    unit: Any = None,
    unit_reason: str | None = None,
) -> dict[str, Any]:
    """Profile one (layer x layout) bucket. Pure; touches no database.

    `rows` are entity documents as stored -- `type`, `layer`, `layout`,
    `ring_status`, `area`, `length`, `bbox`, `block_name`, `text` and friends.
    They are read, never mutated.

    `unit` is the LENGTH unit for this bucket as the caller established it, and
    it is allowed to be absent. Pass the plain name (`"m"`), or the whole
    `store._unit_names()` dict for that layout and its `why_no_unit` reason is
    carried through. Passing nothing means "no unit is claimed" -- it never
    means metres (G2).

    Returns the layer block of the design doc's document shape: `entities`,
    `types` with its truncation stated, `role` + `role_basis` + `role_shares`,
    the native `measure` for the role, `anomalies` from lane 3 and `census`
    from lane 4, each behind its guard.
    """
    rows = list(rows)
    total = len(rows)
    unit_name, unit_why = _unit_pair(unit, unit_reason)

    types, truncation = type_mix(rows)
    counts = family_counts(rows)
    role, role_basis, shares = decide_role(counts, total)
    malformed = sum(1 for row in rows if not isinstance(row, Mapping))

    if role == "region":
        measure = _area_measure(rows, unit=unit_name, unit_reason=unit_why)
    elif role == "network":
        measure = _length_measure(rows, unit=unit_name, unit_reason=unit_why)
    elif role in ("points", "annotation"):
        measure = _count_measure(rows, role=role)
    elif role == _MIXED:
        measure = _count_measure(
            rows,
            role=role,
            note=(
                f"No type family reached {_pct(ROLE_DOMINANCE)}% of this "
                "bucket, so no single native measure would describe it. The "
                "family counts in `role_counts` say what the mix is."
            ),
        )
    else:  # empty bucket
        measure = {
            "kind": None,
            "value": None,
            "unit": None,
            "unit_reason": _COUNT_UNIT_REASON,
            "measured_entities": 0,
            "unmeasured_entities": 0,
            "bucket_entities": 0,
            "basis": "This layer x layout holds no entities.",
            "caveat": None,
        }

    block: dict[str, Any] = {
        "layer": layer,
        "layout": layout,
        "entities": total,
        "types": types,
        "types_truncation": truncation,
        "role": role,
        "role_basis": role_basis,
        "role_counts": counts,
        "role_shares": shares,
        "role_threshold": ROLE_DOMINANCE,
        "measure": measure,
        "anomalies": [],
        "anomalies_status": {"computed": False, "why": None},
        "census": _census_for(rows, role),
    }
    if malformed:
        block["malformed_rows"] = malformed
        block["malformed_rows_note"] = (
            f"{malformed} of {total} rows are not mappings. They are still "
            "counted in `entities` -- a row we cannot read is a row that "
            "exists -- and they classify as 'other'."
        )

    extra: dict[str, Any] = {}
    found = getattr(_ANOMALY, "anomalies", None)
    if found is not None and _accepts(found, "unit"):
        extra["unit"] = unit_name
    anomalies, why = _lane_call(
        _ANOMALY, "dossier_anomaly", "anomalies", _ANOMALY_WHY,
        rows, layer=layer, **extra,
    )
    if why is None and isinstance(anomalies, list):
        block["anomalies"] = anomalies
        block["anomalies_status"] = {"computed": True, "why": None}
    else:
        # An empty list here would read as "this layer is unremarkable", which
        # is a claim nobody made. The list stays empty and the status says the
        # question was not asked (G8).
        block["anomalies_status"] = {
            "computed": False,
            "why": why
            or (
                "dossier_anomaly.anomalies returned "
                f"{type(anomalies).__name__}, not a list"
            ),
            "not_computed": why,
        }
    return block


def _census_for(rows: Sequence[Any], role: str | None) -> dict[str, Any]:
    """Lane 4's census for the roles that have one, behind its guard."""
    if role == "points":
        func = "block_census"
    elif role == "annotation":
        func = "annotation_census"
    else:
        return {
            "applicable": False,
            "why": (
                "A census is computed for the `points` and `annotation` roles. "
                f"This bucket profiles as {role!r}."
            ),
        }

    result, why = _lane_call(
        _CENSUS, "dossier_census", func, _CENSUS_WHY, rows
    )
    if why is not None:
        return {"applicable": True, "not_computed": why, "source": f"dossier_census.{func}"}
    if not isinstance(result, Mapping):
        return {
            "applicable": True,
            "not_computed": (
                f"dossier_census.{func} returned {type(result).__name__}, not "
                "a mapping"
            ),
            "source": f"dossier_census.{func}",
        }
    out = dict(result)
    out["applicable"] = True
    out["source"] = f"dossier_census.{func}"
    return out


# --- The assembler ------------------------------------------------------------


def _buckets(
    rows_by_layer_layout: Any,
) -> list[tuple[tuple[str, str], list[Any]]]:
    """Normalise the caller's buckets into `[((layer, layout), rows), ...]`.

    Accepts a mapping keyed by `(layer, layout)` or an iterable of those pairs,
    because a Mongo aggregation naturally produces the second and a fixture
    naturally writes the first.
    """
    if isinstance(rows_by_layer_layout, Mapping):
        items: Iterable[Any] = rows_by_layer_layout.items()
    else:
        items = rows_by_layer_layout

    out: list[tuple[tuple[str, str], list[Any]]] = []
    for key, rows in items:
        if isinstance(key, str):
            layer, layout = key, ""
        else:
            layer, layout = (list(key) + ["", ""])[:2]
        out.append(((str(layer), str(layout)), list(rows)))
    return out


def build_dossier(
    drawing: Mapping[str, Any],
    rows_by_layer_layout: Any,
    *,
    entity_total: int | None,
    units_by_layout: Mapping[str, Any] | None = None,
    computed_at: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Assemble the whole Dossier for one drawing. Pure; touches no database.

    `drawing` is the `autocad_drawings` document. `rows_by_layer_layout` maps
    `(layer, layout)` to that bucket's entity rows -- every row in the drawing,
    partitioned, because the coverage invariant is arithmetic over a partition
    and a partition with a gap in it is what this whole document exists to
    detect.

    `entity_total` is the drawing's own entity count, passed explicitly so the
    comparison has two independent sides. When it is None the drawing
    document's `entity_count` is used and `total_basis` says so; when both
    exist and disagree the Dossier is **not** complete, because we do not know
    which of the two is right and picking one would be a guess wearing a
    measurement's clothes.

    `units_by_layout` gives the length unit per layout, either as a plain name
    or as a whole `store._unit_names()` dict. A layout that is not in the map
    gets no unit and a stated reason -- never the drawing's header unit, which
    describes model space only and produced this project's worst published
    number when it was stamped on a paper sheet.

    `strict=True` raises `CoverageError` instead of returning a document whose
    coverage failed. The default is False on purpose: a damaged drawing is
    normal (G8), and refusing to profile it would hide the whole file rather
    than the missing part of it. The failure is published loudly either way,
    and `assert_coverage()` is the hard gate for the backfill and the test.
    """
    drawing_id = drawing.get("_id") or drawing.get("drawing_id")
    if not drawing_id:
        raise ValueError(
            "build_dossier needs the drawing's id: a Dossier is keyed by "
            "drawing_id (a content hash) and one without a key cannot be "
            "stored or overwritten in place."
        )

    units = units_by_layout or {}
    blocks: list[dict[str, Any]] = []
    for (layer, layout), rows in _buckets(rows_by_layer_layout):
        supplied = units.get(layout)
        blocks.append(
            profile_rows(rows, layer=layer, layout=layout, unit=supplied)
        )
    # Deterministic order, so re-running the backfill overwrites a document
    # that is byte-identical when nothing changed.
    blocks.sort(key=lambda b: (-b["entities"], b["layer"], b["layout"]))

    coverage = _coverage(blocks, drawing, entity_total)
    document = {
        "_id": str(drawing_id),
        "drawing_id": str(drawing_id),
        "computed_at": computed_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dossier_version": DOSSIER_VERSION,
        "entity_total": coverage["total"],
        "layouts": _layouts_block(drawing, blocks),
        "layers": blocks,
        "file_facts": _file_facts(drawing, blocks),
        "coverage": coverage,
    }
    if strict:
        assert_coverage(document)
    return document


def _coverage(
    blocks: Sequence[Mapping[str, Any]],
    drawing: Mapping[str, Any],
    entity_total: int | None,
) -> dict[str, Any]:
    """The invariant, computed and published so it cannot pass by accident.

    Three separate ways this block refuses to say `complete: true`, and all
    three are real failures seen in this kind of code:

    * the buckets do not add up to the total -- a layer x layout went missing,
      or one was counted twice;
    * there is no total to compare against -- an unknown must never be filled
      in with `accounted`, which would make the invariant a tautology that
      passes on every drawing forever;
    * the caller's total and the drawing document's own count disagree -- the
      comparison has lost one of its two independent sides.
    """
    accounted = sum(int(block["entities"]) for block in blocks)
    declared = drawing.get("entity_count")
    declared = int(declared) if isinstance(declared, int) and not isinstance(
        declared, bool
    ) else None
    given = int(entity_total) if isinstance(entity_total, int) and not isinstance(
        entity_total, bool
    ) else None

    disagreement: str | None = None
    if given is not None and declared is not None and given != declared:
        disagreement = (
            f"The caller passed entity_total={given} while the drawing "
            f"document declares entity_count={declared}. The two sides of the "
            "comparison disagree, so completeness cannot be certified from "
            "either of them."
        )

    total = given if given is not None else declared
    if given is not None and declared is not None:
        basis = "entity_total passed by the caller, cross-checked against the drawing document's entity_count"
    elif given is not None:
        basis = "entity_total passed by the caller; the drawing document declares no entity_count to cross-check against"
    elif declared is not None:
        basis = "the drawing document's own entity_count; the caller passed no entity_total"
    else:
        basis = "no total is available from either the caller or the drawing document"

    difference = None if total is None else total - accounted
    complete = (
        total is not None and difference == 0 and disagreement is None
    )

    if total is None:
        verdict = (
            f"NOT COMPLETE: {accounted} entities are accounted for across "
            f"{len(blocks)} buckets, but the drawing states no total to check "
            "them against. Completeness is unknown, and unknown is not "
            "reported as true."
        )
    elif disagreement is not None:
        verdict = f"NOT COMPLETE: {disagreement} Buckets account for {accounted}."
    elif difference == 0:
        verdict = (
            f"COMPLETE: {accounted} entities across {len(blocks)} buckets "
            f"equals the drawing's own total of {total}. Every entity in this "
            "drawing is inside exactly one profiled bucket."
        )
    elif difference is not None and difference > 0:
        verdict = (
            f"NOT COMPLETE: the drawing holds {total} entities but the buckets "
            f"account for only {accounted}. {difference} entities are in the "
            "drawing and in no bucket -- a layer x layout is missing from this "
            "Dossier, and anything computed from it under-reports."
        )
    else:
        verdict = (
            f"NOT COMPLETE: the buckets account for {accounted} entities while "
            f"the drawing holds {total}. {-(difference or 0)} entities are "
            "counted more than once -- a bucket is duplicated, and anything "
            "computed from it over-reports."
        )

    return {
        "accounted": accounted,
        "total": total,
        "difference": difference,
        "complete": complete,
        "buckets": len(blocks),
        "total_basis": basis,
        "declared_entity_count": declared,
        "caller_entity_total": given,
        "disagreement": disagreement,
        "verdict": verdict,
        "basis": (
            "sum of `entities` over every (layer x layout) bucket, compared "
            "with the drawing's own entity total. File-level facts (xrefs, "
            "text styles, embedded documents) are counted in `file_facts` and "
            "are deliberately NOT added here: they are not entities, and "
            "adding them would break the one piece of arithmetic this "
            "document exists to prove."
        ),
    }


def assert_coverage(dossier: Mapping[str, Any]) -> None:
    """Raise `CoverageError` unless the coverage invariant held.

    This is the hard gate. `build_dossier` publishes the failure; this turns it
    into a stop, so a parametric test over every drawing in the store and a
    backfill loop both fail loudly rather than storing a document that quietly
    describes part of a file.
    """
    coverage = dossier.get("coverage")
    if not isinstance(coverage, Mapping):
        raise CoverageError(
            "This document carries no `coverage` block, so its completeness "
            "was never computed. A Dossier without the invariant is not a "
            "Dossier."
        )
    if coverage.get("complete") is True and coverage.get("difference") == 0:
        return
    raise CoverageError(
        f"coverage invariant failed for drawing {dossier.get('_id')!r}: "
        f"{coverage.get('verdict')}",
        accounted=coverage.get("accounted"),
        total=coverage.get("total"),
        difference=coverage.get("difference"),
    )


def _layouts_block(
    drawing: Mapping[str, Any], blocks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per layout: what the file declares, what was profiled, and whether they agree.

    Both numbers are published because they can differ, and when they do the
    difference is the finding. A layout the file never declared still appears
    here -- it holds entities, so leaving it out would be the coverage hole in
    a smaller shape.
    """
    declared: dict[str, Mapping[str, Any]] = {}
    for entry in drawing.get("layouts") or []:
        if isinstance(entry, Mapping) and entry.get("name") is not None:
            declared[str(entry["name"])] = entry

    profiled: Counter[str] = Counter()
    for block in blocks:
        profiled[str(block["layout"])] += int(block["entities"])

    out: list[dict[str, Any]] = []
    for name in sorted(set(declared) | set(profiled)):
        entry = declared.get(name)
        declared_count = entry.get("entity_count") if entry else None
        declared_count = (
            int(declared_count)
            if isinstance(declared_count, int) and not isinstance(declared_count, bool)
            else None
        )
        profiled_count = int(profiled.get(name, 0))
        out.append(
            {
                "name": name,
                "declared_entities": declared_count,
                "declared_entities_reason": None
                if entry is not None
                else (
                    "this layout is not in the drawing document's layout "
                    "table, so the file declares no count for it"
                ),
                "profiled_entities": profiled_count,
                "agrees": None
                if declared_count is None
                else declared_count == profiled_count,
                "is_modelspace": entry.get("is_modelspace") if entry else None,
                "is_block": entry.get("is_block") if entry else None,
            }
        )
    return out


def _capped(items: Sequence[Any]) -> tuple[list[Any], dict[str, Any]]:
    kept = list(items[:MAX_FACT_ITEMS])
    return kept, {
        "cap": MAX_FACT_ITEMS,
        "present": len(items),
        "reported": len(kept),
        "dropped": max(0, len(items) - len(kept)),
        "basis": (
            f"Named items are capped at {MAX_FACT_ITEMS} (G7). The totals "
            "beside this block are counted in full."
        ),
    }


def _file_facts(
    drawing: Mapping[str, Any], blocks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The file-level account: xrefs, text styles, layers, audit, units.

    These are counted so the Dossier spans the whole file rather than only its
    model-space geometry -- and they are kept out of the coverage sum on
    purpose. An xref is not an entity; adding one to `accounted` would make the
    invariant pass for the wrong reason.
    """
    xrefs = [x for x in (drawing.get("xrefs") or []) if isinstance(x, Mapping)]
    unresolved = [x for x in xrefs if not x.get("resolved")]
    xref_items, xref_trunc = _capped(
        [
            {
                "block_name": x.get("block_name"),
                "path": x.get("path"),
                "kind": x.get("kind"),
                "resolved": bool(x.get("resolved")),
            }
            for x in xrefs
        ]
    )

    styles = [s for s in (drawing.get("text_styles") or []) if isinstance(s, Mapping)]
    shx = [s for s in styles if str(s.get("font") or "").lower().endswith(".shx")]
    shx_items, shx_trunc = _capped(
        [{"name": s.get("name"), "font": s.get("font")} for s in shx]
    )

    declared_layers = [
        entry for entry in (drawing.get("layers") or []) if isinstance(entry, Mapping)
    ]
    declared_names = {str(e.get("name")) for e in declared_layers if e.get("name") is not None}
    profiled_names = {str(b["layer"]) for b in blocks}
    unprofiled = sorted(declared_names - profiled_names)
    undeclared = sorted(profiled_names - declared_names)
    unprofiled_items, unprofiled_trunc = _capped(unprofiled)
    undeclared_items, undeclared_trunc = _capped(undeclared)

    embedded = drawing.get("embedded_documents")
    if isinstance(embedded, Sequence) and not isinstance(embedded, (str, bytes)):
        embedded_block: dict[str, Any] = {
            "available": True,
            "total": len(embedded),
            "source": "the `embedded_documents` key on the drawing document",
        }
    else:
        embedded_block = {
            "available": False,
            "total": None,
            "why": (
                "Embedded documents and OLE objects are extracted from the DXF "
                "file itself (app/embedded.py) and are not fields on the "
                "drawing document, so this pure profiler cannot count them. "
                "The count is absent, not zero. Pass them in as "
                "`embedded_documents` on `drawing` to have them reported."
            ),
        }

    audit_errors = drawing.get("audit_errors")
    audit_fixes = drawing.get("audit_fixes")

    layouts_disagreeing = sum(
        1 for entry in _layouts_block(drawing, blocks) if entry["agrees"] is False
    )

    return {
        "xrefs": {
            "total": len(xrefs),
            "unresolved": len(unresolved),
            "items": xref_items,
            "truncation": xref_trunc,
        },
        "text_styles": {
            "total": len(styles),
            "shx_styles": len(shx),
            "shx_items": shx_items,
            "truncation": shx_trunc,
            "basis": (
                "A style counts as SHX when its font file name ends in `.shx`. "
                "SHX fonts are not distributed with the drawing, so text drawn "
                "in one is readable only through the SHX map."
            ),
        },
        "layers": {
            "declared": len(declared_names),
            "profiled": len(profiled_names),
            "declared_but_unprofiled": len(unprofiled),
            "declared_but_unprofiled_items": unprofiled_items,
            "declared_but_unprofiled_truncation": unprofiled_trunc,
            "profiled_but_undeclared": len(undeclared),
            "profiled_but_undeclared_items": undeclared_items,
            "profiled_but_undeclared_truncation": undeclared_trunc,
            "basis": (
                "A declared layer with no profiled bucket holds no entities in "
                "any layout. A profiled layer that the layer table never "
                "declared is the reverse, and both are worth seeing."
            ),
        },
        "layouts": {
            "declared": len(drawing.get("layouts") or []),
            "disagreeing": layouts_disagreeing,
            "basis": (
                "`disagreeing` counts layouts whose declared entity count and "
                "profiled entity count differ."
            ),
        },
        "embedded_documents": embedded_block,
        "audit": {
            "errors": audit_errors if isinstance(audit_errors, int) else None,
            "fixes": audit_fixes if isinstance(audit_fixes, int) else None,
            "basis": (
                "Recovered on open by ezdxf's auditor. A non-zero error count "
                "is normal for a converted file and does not invalidate what "
                "was read; it says the file arrived damaged."
            ),
        },
        "units": {
            "units_name": drawing.get("units_name"),
            "units_code": drawing.get("units_code"),
            "declared_in_file": bool(drawing.get("units_code")),
            "basis": (
                "$INSUNITS is a drawing-header field describing MODEL space. "
                "It is recorded here as a file fact and is NOT stamped on "
                "sheet or block-definition buckets; each bucket's unit is "
                "whatever the caller established for that layout, and absent "
                "when the caller established none."
            ),
        },
        "counted_here_not_in_coverage": (
            "Everything in `file_facts` is a file-level fact, not an entity. "
            "None of it is added to `coverage.accounted`."
        ),
    }
