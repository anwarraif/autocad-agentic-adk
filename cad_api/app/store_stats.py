"""mean/median/mode/sd in measure

Owner: subagent D — UPLIFT-06.

This file is deliberately kept apart from `store.py` so that two specs worked
on at the same time never touch the same line. The rules, and their reasons:

- **Do not import from `.store`.** `store.py` imports this module at its
  bottom; importing back would be circular. Take `coll` and its friends
  straight from `.mongo`, and geometry from `.region`.
- **Do not add lines to `store.py` or `main.py`.** Both are shared, and every
  line added there is one merge conflict. Functions here are reached as
  `store.store_stats.<function>`.
- **Every response that carries MEANING must carry `evidence`** per
  `docs/UPLIFT-08-EVIDENCE.md` — use the helpers in `app/evidence.py`, do not
  assemble the dict yourself.
- **Every number carries its unit** (G2), and a unit is allowed to be absent.
  A withheld number becomes `None` + a reason, never `0`.

The `docs/UPLIFT-GENERALITY-RULES.md` G1-G10 checklist is run before every
commit in this file.

---

One decision shapes this entire file:

    **The average is not the hard part. Saying what the average is OF is the
    hard part.**

Σ ÷ n can be written by anyone. What its reader cannot recompute are the three
things that make that number mean something, and all three must travel with it:

1.  **How many out of how many.** A HATCH carries no area. It is not an object
    of 0 m2, and an average that lets it in as a zero is dragged down by a
    number that is WRONG, not by one that is missing. So every statistic here
    is computed over measured values only, and the `basis` of each number names
    its denominator: "of 133 of 20,000".
2.  **Which population.** The mean area of "all polygons" mixes 300 m2 parcels
    with neighbourhood boundaries of thousands of m2, and the result describes
    no single object. Every cross-population aggregate says how many
    populations are mixed inside it.
3.  **At what rounding.** A continuous measurement has no mode until it is
    binned. The same plot area is stored as 300.0 and
    299.9953; without rounding every value is its own mode. So
    `mode_decimals` always travels in the response, and changing it changes
    the answer.

A dimensioned number never leaves bare: mean, median, mode, min, max,
sd, quartiles and Σ are all `evidence.Quantity`. Counts — `n`, `mode_count`,
`above_median` — deliberately are NOT, because `Quantity` is for a number that
can be wrong about its unit, and 449 objects cannot.
"""

from __future__ import annotations

import math
import statistics as st
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from . import evidence as ev
from .mongo import COLL_ENTITIES, coll

#: Decimals used to bin values before the mode is counted.
#:
#: A continuous measurement has no mode. A shoelace area computed over float
#: coordinates stores plots built to the same module as
#: 300.0 and 299.9953; counted raw, 512 plots have 512 modes and
#: `mode_count` is noise.
#:
#: The value is 2 and not 6 because 6 does not cover the real spread: on the
#: reference drawing, the mode at 6 decimals holds 265 of 512 and at 2 decimals
#: holds 449 — and 449 is the figure UPLIFT-06's own Definition of Done asks
#: for. 2 decimals also reproduces all thirteen `n mode` values in the spec's
#: table, and 485 objects above the overall median. Three independent pieces of
#: evidence, one rounding.
#:
#: A WARNING that cannot be removed: this is an ABSOLUTE rounding, so it
#: depends on the unit (G2). Two decimals of m2 is cm2; two decimals of
#: in2 is far finer. That is why it is a parameter, why its default is stated,
#: and why its value is echoed in every statistics block.
MODE_DECIMALS: int = 2

#: Below this the median and the mode are driven by a single value. Not a
#: sacred statistical threshold — a STATED one (G7), so the reader knows when
#: a median has stopped describing anything.
SMALL_POPULATION: int = 20

#: Cap on how many rows are pulled into memory to be computed over. Stated, not
#: assumed (G7): the median, the quartiles and the mode need the values
#: themselves and not only their count, so there is no aggregation route that
#: avoids this.
MAX_ROWS: int = 250_000

#: Stored numeric values are rounded to this before they are published, exactly
#: like `sum_measured_only` in `store.py`. Precision below this is float noise,
#: not measurement.
OUTPUT_DECIMALS: int = 6


class StatsRefused(Exception):
    """A refused statistics request, with a suggestion.

    Its shape mirrors `store.MeasureRefused` (`code`/`message`/`hint`) so the
    `@app.exception_handler` already in `main.py` can be reused without writing
    a new handler. See the change request in
    `docs/PROGRESS.md`.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


# --- Units -------------------------------------------------------------------


def _unit_of(units: Mapping[str, Any] | None, measure: str) -> tuple[str | None, str | None]:
    """The unit for `measure`, or None along with the reason there is none.

    `units` is handed over by the caller exactly as it came from
    `store._unit_names()`, precisely as `evidence.Scope` demands it and for the
    same reason: this module must not import `.store`, and a unit fetched by
    hand from somewhere else is a unit that can come from the wrong layout.
    """
    key = "length_unit" if measure == "length" else "area_unit"
    unit = (units or {}).get(key)
    if unit:
        return str(unit), None
    reason = (units or {}).get("why_no_unit") or (
        "This drawing declares no unit, so the numbers here are in drawing "
        "units and not in metres."
    )
    return None, str(reason)


def _unit_text(unit: str | None) -> str:
    return unit if unit else "drawing units (this file declares no unit)"


# --- Values ------------------------------------------------------------------


def _numbers(values: Iterable[Any]) -> list[float]:
    """Real numbers only. `bool` is an `int` in Python and is not a measure."""
    return [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]


def _round(value: float) -> float:
    return round(float(value), OUTPUT_DECIMALS)


def _quantise(value: float, decimals: int) -> float:
    return round(float(value), decimals)


# --- The core: one population, explained -------------------------------------


def describe(
    values: Sequence[Any],
    *,
    measure: str,
    units: Mapping[str, Any] | None,
    population: str,
    matched_count: int | None = None,
    mode_decimals: int = MODE_DECIMALS,
) -> dict[str, Any]:
    """One population's spread, with every number saying what it is a number of.

    Pure: it does not touch the database, so every rule here can be tested
    without an ingested drawing. `values` has already been filtered to MEASURED
    values; `matched_count` is how many objects actually matched the filter,
    and the difference between the two is what makes "the mean of 133 of 133"
    mean something different from "the mean of 133 of 20,000".

    What it does NOT do, deliberately: an unmeasurable object never enters as
    a 0. It is counted in `unmeasurable_count` and named in the `basis` of
    every number, because an area that is absent is not an area of zero.
    """
    nums = _numbers(values)
    n = len(nums)
    matched = int(matched_count) if matched_count is not None else n
    if matched < n:  # pragma: no cover - a mistaken caller
        matched = n
    unmeasurable = matched - n
    unit, unit_reason = _unit_of(units, measure)

    def _basis(label: str) -> str:
        return f"{label} of {n} of {matched} measured objects on {population}"

    def _held(label: str, method: str, reason: str) -> dict[str, Any]:
        return ev.Quantity.withheld(
            basis=_basis(label), method=method, reason=reason
        ).as_dict()

    def _got(label: str, method: str, value: float) -> dict[str, Any]:
        return ev.Quantity.measured(
            basis=_basis(label),
            method=method,
            value=_round(value),
            unit=unit,
            unit_reason=unit_reason,
        ).as_dict()

    mode_method = f"most frequent value after rounding to {mode_decimals} decimals"
    methods = {
        "mean": "Σ of the measured values ÷ how many there are",
        "median": "the middle value once sorted",
        "mode": mode_method,
        "min": "smallest measured value",
        "max": "largest measured value",
        "sd": "POPULATION standard deviation (ddof=0), not the sample one",
        "p25": "first quartile, linear interpolation (inclusive method)",
        "p75": "third quartile, linear interpolation (inclusive method)",
        "sum": "exact summation (math.fsum) over the measured values",
    }

    block: dict[str, Any] = {
        "n": n,
        "measured_count": n,
        "matched_count": matched,
        "unmeasurable_count": unmeasurable,
        "mode_decimals": mode_decimals,
        "mode_count": 0,
        "mode_ties": 0,
        "above_median": 0,
        "mode_note": None,
        "small_population_note": None,
        "distribution_note": None,
    }

    if n == 0:
        why = (
            f"{matched} objects matched this filter but not one of them "
            f"carries {measure}; zero is not the answer"
            if matched
            else "no object matched this filter"
        )
        for key, method in methods.items():
            block[key] = _held(key, method, why)
        block["mode_count"] = 0
        block["statement"] = (
            f"No {measure} can be summarised on {population}: {why}."
        )
        return block

    ordered = sorted(nums)
    mean = st.fmean(nums)
    median = st.median(nums)
    total = math.fsum(nums)

    block["mean"] = _got("mean", methods["mean"], mean)
    block["median"] = _got("median", methods["median"], median)
    block["min"] = _got("min", methods["min"], ordered[0])
    block["max"] = _got("max", methods["max"], ordered[-1])
    block["sum"] = _got("sum", methods["sum"], total)

    # sd and the quartiles need two values. `pstdev` over a single value returns
    # 0.0, which is correct by definition and misleading as an answer: it would
    # fire the "every value is identical" note for a population that has no
    # spread to talk about at all.
    if n >= 2:
        sd = st.pstdev(nums)
        quarts = st.quantiles(nums, n=4, method="inclusive")
        block["sd"] = _got("sd", methods["sd"], sd)
        block["p25"] = _got("p25", methods["p25"], quarts[0])
        block["p75"] = _got("p75", methods["p75"], quarts[2])
    else:
        sd = None
        too_few = "a spread needs at least two values; this population has one"
        for key in ("sd", "p25", "p75"):
            block[key] = _held(key, methods[key], too_few)

    # The mode, and the only rounding in this file that changes the answer.
    binned = [_quantise(v, mode_decimals) for v in nums]
    counts = Counter(binned)
    top = max(counts.values())
    if top < 2:
        block["mode"] = _held(
            "mode",
            mode_method,
            f"no value repeats at {mode_decimals} decimals; a "
            "mode with a single occurrence is not a typical value but the value "
            "that happened to be counted first",
        )
        block["mode_count"] = 0
        block["mode_note"] = (
            f"{n} values, not one of them repeats at {mode_decimals} decimals."
        )
        mode_value = None
    else:
        tied = sorted(v for v, c in counts.items() if c == top)
        mode_value = tied[0]
        block["mode"] = _got("mode", mode_method, mode_value)
        block["mode_count"] = int(top)
        block["mode_ties"] = len(tied)
        if len(tied) > 1:
            block["mode_note"] = (
                f"{len(tied)} values share the highest count ({top} times) "
                f"at {mode_decimals} decimals; the one reported is the "
                "smallest. A mode that shares its place is not a typical value."
            )

    # Above the median, counted at the SAME rounding as the mode. Counted raw,
    # the figure turns into a count of float noise: a value of 300.0000001
    # against a median of 300.0 counts as "above the design size" when it IS
    # the design size.
    qmedian = _quantise(median, mode_decimals)
    block["above_median"] = sum(1 for v in binned if v > qmedian)
    block["above_median_basis"] = (
        f"how many values are, rounded to {mode_decimals} decimals, larger "
        f"than this population's median rounded the same way ({qmedian})"
    )

    if n < SMALL_POPULATION:
        block["small_population_note"] = (
            f"n = {n}, below {SMALL_POPULATION}. The median and the mode over "
            "a population this small are driven by a single value; read both "
            "as an indication, not as a measurement."
        )

    block["distribution_note"] = _distribution_note(
        mean=mean,
        median=median,
        sd=sd,
        qmin=_quantise(ordered[0], mode_decimals),
        qmedian=qmedian,
        qmode=mode_value,
        mode_decimals=mode_decimals,
        unit=unit,
    )
    block["statement"] = _statement(
        block, measure=measure, population=population, unit=unit
    )
    return block


def _distribution_note(
    *,
    mean: float,
    median: float,
    sd: float | None,
    qmin: float,
    qmedian: float,
    qmode: float | None,
    mode_decimals: int,
    unit: str | None,
) -> str | None:
    """A sentence produced by a RULE, not by a model. First match wins.

    Two departures from the order the spec writes, both of them necessary:

    1.  `sd == 0` is checked BEFORE the floor. A population that is entirely
        identical also satisfies min = median = mode, so under the spec's order
        it would be told that its spread runs upward only — about a spread that
        does not exist.
    2.  The floor is compared at the same rounding as the mode, not at a
        tolerance of 1e-6. The real floor in this store is 299.9953
        against a median of 300.0: at 1e-6 not one of the 13 reference layers
        reads as floored, and the note the Definition of Done asks for on all
        thirteen would never be published. Comparing an already binned mode
        against an unbinned min also compares two different things.

    What is NOT here: a note for an ordinary spread. A note that always appears
    stops being read, and all that is left of it is its cost.
    """
    if sd is not None and sd == 0.0:
        return (
            "sd = 0: every value is identical. The mean is that value itself, "
            "and the median and the mode add nothing to it."
        )
    if qmode is not None and qmin == qmedian == qmode:
        where = f" {unit}" if unit else ""
        return (
            f"min = median = mode ({qmedian}{where}, compared at "
            f"{mode_decimals} decimals): the distribution is floored. It "
            "spreads upward only. The median describes the designed value; the "
            "mean sits higher because it is pulled by the upper tail."
        )
    if median > 0 and mean > median * 1.02:
        return (
            "the mean is pulled upward by a tail; the median better represents "
            "the typical value in this population."
        )
    return None


def _statement(
    block: Mapping[str, Any], *, measure: str, population: str, unit: str | None
) -> str:
    """One quotable sentence, carrying the number, its unit, and its denominator.

    Written here rather than left to the caller for the same reason as
    `store._measure_statement`: both wrong numbers this project ever shipped
    were composed as prose around a correct figure. Making an honest sentence
    the cheapest thing to quote works better than forbidding a model to
    paraphrase.
    """
    unit_text = _unit_text(unit)
    mean = block["mean"]["value"]
    median = block["median"]["value"]
    n = block["n"]
    matched = block["matched_count"]
    head = (
        f"Mean {measure} on {population}: {mean} {unit_text}, "
        f"over {n} of {matched} objects that matched and were measured. "
        f"Median {median} {unit_text}."
    )
    if block["unmeasurable_count"]:
        head += (
            f" {block['unmeasurable_count']} other objects matched but carry "
            f"no {measure}, and were NOT counted as zero."
        )
    mode = block["mode"]
    if mode["value"] is not None:
        head += (
            f" Mode {mode['value']} {unit_text} over {block['mode_count']} "
            f"objects (rounded to {block['mode_decimals']} decimals)."
        )
    for key in ("distribution_note", "small_population_note", "mode_note"):
        if block.get(key):
            head += " " + block[key]
    return head


# --- The database: raw values per population ---------------------------------


def values_for(
    drawing_id: str,
    *,
    measure: str,
    layout_name: str,
    layers: Sequence[str] | None = None,
    dxftype: str | None = None,
    block_name: str | None = None,
    group_field: str = "layer",
) -> dict[str, Any]:
    """Measured values per group, plus how many objects matched but were not
    measurable.

    `group_field` is the grouping axis (`layer` or `type`); `layers` still
    filters on `layer`, because the two are different axes and merging them is
    how `measure` once accepted a type as a layer name.

    Two queries, both prefixed by `drawing_id` because this cluster runs with
    `notablescan` and a query without an indexed plan FAILS rather than slows
    down. The values are streamed through `find` rather than collected with
    `$push` inside `$group`: `$push` puts the whole population into a single
    result document and hits the 16 MB limit on a large enough drawing, while a
    cursor does not.
    """
    if measure not in ("length", "area"):
        raise StatsRefused(
            "INVALID_MEASURE",
            f"measure must be 'length' or 'area', not {measure!r}.",
            "Use measure='area' for areas, 'length' for lengths.",
        )
    if group_field not in ("layer", "type"):
        raise StatsRefused(
            "INVALID_FIELD",
            f"group_field must be 'layer' or 'type', not {group_field!r}.",
            "Those are the axes that exist in the store; any other axis is not "
            "indexed, and this cluster refuses a query without an indexed plan.",
        )

    query: dict[str, Any] = {"drawing_id": drawing_id, "layout": layout_name}
    if layers:
        query["layer"] = {"$in": list(layers)}
    if dxftype:
        query["type"] = dxftype.upper()
    if block_name:
        query["block_name"] = block_name

    collection = coll(COLL_ENTITIES)
    matched: dict[str, int] = {}
    for row in collection.aggregate(
        [{"$match": query}, {"$group": {"_id": "$" + group_field, "n": {"$sum": 1}}}]
    ):
        matched[str(row["_id"])] = int(row["n"] or 0)

    total_matched = sum(matched.values())
    if total_matched > MAX_ROWS:
        raise StatsRefused(
            "TOO_MANY_ROWS",
            f"{total_matched} objects matched; the cap is {MAX_ROWS}. The "
            "median, the quartiles and the mode need the values one by one and "
            "not only their count, so no aggregation route avoids this cap.",
            "Narrow it down with layer, type, or layout, then try again.",
        )

    values: dict[str, list[float]] = {}
    numeric = dict(query)
    numeric[measure] = {"$type": "number"}
    for doc in collection.find(numeric, {group_field: 1, measure: 1, "_id": 0}):
        values.setdefault(str(doc.get(group_field)), []).append(float(doc[measure]))

    present = sorted(matched)
    # Only meaningful when what is grouped is the same as what is filtered. A
    # layer that was asked for and is not there is a finding; a layer asked for
    # while the grouping is per type is not the same question.
    absent = (
        [name for name in (layers or []) if name not in matched]
        if group_field == "layer"
        else []
    )
    return {
        "values": values,
        "matched": matched,
        "group_field": group_field,
        "layers_present": present,
        "layers_absent": absent,
        "total_matched": total_matched,
    }


# --- The 23 August meeting question: the mean per typology --------------------


def stats_by_layer(
    drawing_id: str,
    *,
    measure: str,
    layout_name: str,
    layers: Sequence[str],
    units: Mapping[str, Any] | None,
    dxftype: str | None = None,
    block_name: str | None = None,
    mode_decimals: int = MODE_DECIMALS,
) -> dict[str, Any]:
    """The spread per layer, and their pooled spread, in one call.

    This is the question that at the 23 August meeting was answered OUTSIDE the
    agent by opening the DXF directly: "what is the mean area per typology". It
    is one call and not thirteen because the pooled row only means anything if
    it pools the SAME population as its own rows — and thirteen separate calls
    hand that pooling to the reader, which is exactly the work the reader
    should not be doing.

    `layers` is handed over by the caller. This module does not know — and must
    not guess — which layers are the "plot typologies" in this drawing; that is
    classification, and it lives in the land-use config (G1, G4, G10). All that
    is promised here is: statistics over the population YOU named, with that
    population restated inside the answer.
    """
    raw = values_for(
        drawing_id,
        measure=measure,
        layout_name=layout_name,
        layers=layers,
        dxftype=dxftype,
        block_name=block_name,
    )
    scope_bits = [f"layout={layout_name}"]
    if dxftype:
        scope_bits.append(f"type={dxftype.upper()}")
    if block_name:
        scope_bits.append(f"block_name={block_name}")
    scope_tail = " AND ".join(scope_bits)

    rows: list[dict[str, Any]] = []
    for name in raw["layers_present"]:
        vals = raw["values"].get(name, [])
        rows.append(
            {
                "name": name,
                "matched_count": raw["matched"][name],
                "measured_count": len(vals),
                "mixed_population_note": None,
                "stats": describe(
                    vals,
                    measure=measure,
                    units=units,
                    population=f"layer={name} AND {scope_tail}",
                    matched_count=raw["matched"][name],
                    mode_decimals=mode_decimals,
                ),
            }
        )
    rows.sort(key=lambda r: r["stats"]["mean"]["value"] or -1.0, reverse=True)

    present = raw["layers_present"]
    pooled: list[float] = []
    for name in present:
        pooled.extend(raw["values"].get(name, []))

    overall_population = f"{len(present)} layers pooled on {scope_tail}"
    overall: dict[str, Any] = {
        "matched_count": raw["total_matched"],
        "measured_count": len(pooled),
        "populations": len(present),
        "mixed_population_note": None,
        "stats": describe(
            pooled,
            measure=measure,
            units=units,
            population=overall_population,
            matched_count=raw["total_matched"],
            mode_decimals=mode_decimals,
        ),
    }
    if len(present) > 1:
        # A portfolio figure. It describes no single object, and the only thing
        # keeping it from being read as "the typical area" is this sentence
        # standing right next to it.
        overall["mixed_population_note"] = (
            f"These figures mix {len(present)} different populations "
            f"({', '.join(present)}). A cross-population mean is a portfolio "
            "figure: it describes the mix, not an object. For a figure that "
            "describes something, read the per-layer rows."
        )

    unit, unit_reason = _unit_of(units, measure)
    result: dict[str, Any] = {
        "drawing_id": drawing_id,
        "measure": measure,
        "layout": layout_name,
        "scope_label": scope_tail,
        "scope_note": (
            f"{scope_tail}; block definitions are NOT included unless the "
            f"layout asked for is itself a block definition; units "
            f"{_unit_text(unit)}"
        ),
        "units": dict(units or {}),
        "unit_reason": unit_reason,
        "layers_requested": list(layers),
        "layers_present": present,
        "layers_absent": raw["layers_absent"],
        "mode_decimals": mode_decimals,
        "row_cap": MAX_ROWS,
        "by_layer": rows,
        "overall": overall,
    }
    if raw["layers_absent"]:
        # Never silently dropped: a population that was asked for and is not
        # there is a finding, and an answer that leaves it out looks complete.
        result["layers_absent_note"] = (
            f"{len(raw['layers_absent'])} requested layers have not a single "
            f"object on {scope_tail}: "
            f"{', '.join(raw['layers_absent'])}. They are part of no figure "
            "anywhere in this response."
        )
    result["statement"] = _by_layer_statement(result)
    return result


def _by_layer_statement(result: Mapping[str, Any]) -> str:
    rows = result["by_layer"]
    if not rows:
        return (
            f"Not one of the {len(result['layers_requested'])} requested "
            f"layers has any object on {result['scope_label']}."
        )
    overall = result["overall"]
    lead = (
        f"{len(rows)} populations summarised on {result['scope_label']}, "
        f"{overall['measured_count']} of {overall['matched_count']} objects "
        "measured."
    )
    if overall["mixed_population_note"]:
        lead += " " + overall["mixed_population_note"]
    top = rows[0]["stats"]
    lead += (
        f" Highest: {rows[0]['name']}, mean {top['mean']['value']} "
        f"{_unit_text(top['mean']['unit'])} over n={top['n']}."
    )
    if result.get("layers_absent_note"):
        lead += " " + result["layers_absent_note"]
    return lead


# --- Wiring into the `measure` that already exists ----------------------------


def attach(
    result: dict[str, Any],
    drawing_id: str,
    *,
    layout_name: str,
    layer: str | None = None,
    dxftype: str | None = None,
    block_name: str | None = None,
    group_by: str | None = None,
    mode_decimals: int = MODE_DECIMALS,
) -> dict[str, Any]:
    """Add a `stats` block to a finished `store.measure_entities` result.

    Kept apart from `stats_by_layer` so that `measure_entities` needs to change
    by no more than one line. The counts are not recomputed here — they are
    read from `result`, so `stats` and `measured_entities` cannot possibly tell
    two different stories about the same population.

    The change request for `store.py` is in `docs/PROGRESS.md`; that file is
    shared and is not touched from here.
    """
    measure = result["measure"]
    units = result.get("units")
    # Grouped on the same axis `measure` was asked for, so that
    # `group_by='type'` does not quietly lose its statistics block. Without
    # this the per-type rows are published without `stats` and nothing in the
    # response says why.
    group_field = group_by if group_by in ("layer", "type") else "layer"
    raw = values_for(
        drawing_id,
        measure=measure,
        layout_name=layout_name,
        layers=[layer] if layer else None,
        dxftype=dxftype,
        block_name=block_name,
        group_field=group_field,
    )
    pooled: list[float] = []
    for name in raw["layers_present"]:
        pooled.extend(raw["values"].get(name, []))

    population = result.get("scope_label") or f"layout={layout_name}"
    result["stats"] = describe(
        pooled,
        measure=measure,
        units=units,
        population=population,
        matched_count=int(result.get("total_matches") or 0),
        mode_decimals=mode_decimals,
    )
    result["stats_mode_decimals"] = mode_decimals
    if len(raw["layers_present"]) > 1:
        result["stats"]["mixed_population_note"] = (
            f"These figures mix {len(raw['layers_present'])} "
            f"{group_field}. A cross-population mean is a portfolio figure; "
            f"use group_by='{group_field}' for a figure that describes "
            "one population."
        )

    for row in result.get("by_" + group_field) or []:
        name = row.get("name")
        vals = raw["values"].get(str(name), [])
        row["stats"] = describe(
            vals,
            measure=measure,
            units=units,
            population=f"{group_field}={name} AND layout={layout_name}",
            matched_count=int(row.get("count") or 0),
            mode_decimals=mode_decimals,
        )
    return result
