"""land_use_summary + the land use config; then georeferencing

Owner: subagent A — UPLIFT-02, then UPLIFT-14.

This file is deliberately split out of `store.py` so that two specs worked on
at the same time never touch the same line. The rules, and their reasons:

- **Do not import from `.store`.** `store.py` imports this module at its
  bottom; an import back would form a cycle. Take `coll` and friends straight
  from `.mongo`, and geometry from `.region`.
- **Do not add lines to `store.py` or `main.py`.** Both are shared property,
  and every line added there is one merge conflict. Functions here are reached
  as `store.store_landuse.<function>`.
- **Every response that carries MEANING must carry `evidence`** as required by
  `docs/UPLIFT-08-EVIDENCE.md` — use the helpers in `app/evidence.py`, do not
  assemble the dict yourself.
- **Every number carries its unit** (G2), and a unit is allowed to be absent.
  A withheld number becomes `None` + a reason, never `0`.

The `docs/UPLIFT-GENERALITY-RULES.md` G1-G10 checklist is run before every
commit to this file.

One recorded exception to the first rule
-----------------------------------------
`_scope()` calls `store._unit_names()` through an import INSIDE the function.
The danger that rule forbids is a MODULE-level import cycle, and an import
deferred to call time closes no cycle at all: by the time `land_use_summary` is
called, `store` has already finished executing. The alternative that lost was
copying `_unit_names` here — and a unit that has two copies is a unit that will
differ once one of them is edited, in a repo where 11 of the 18 drawings are in
inches and 3 state no unit at all (G2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import evidence as ev
from . import landuse
from .extract import BLOCK_LAYOUT_PREFIX, RING_TYPES
from .landuse import crs as crs_math
from .mongo import COLL_DRAWINGS, COLL_ENTITIES, coll

# --- The Dossier seam (DOSSIER Phase 2, lane A) ------------------------------
#
# `dossier_read` is the ONLY module that reads `autocad_dossiers`, and unlike
# Phase 1's pure lanes it is Mongo-backed. It is therefore reached exactly the
# way `dossier.py` reaches its own sibling lanes: behind a guard, with the
# reason for its absence PUBLISHED rather than swallowed.
#
# `except Exception` and not `except ImportError` on purpose: a module that is
# absent and a module that is half-written are the same situation for this file
# -- the land use summary still has to answer -- and the difference between them
# is worth reporting rather than hiding. The reason string travels into the
# response, so "lane A has a SyntaxError" reads differently from "lane A has not
# been written yet", which is the point.
try:
    from . import dossier_read as _DOSSIER

    _DOSSIER_WHY: str | None = None
except Exception as exc:  # pragma: no cover - depends on lane A being present
    _DOSSIER = None  # type: ignore[assignment]
    _DOSSIER_WHY = f"{type(exc).__name__}: {exc}"

log = logging.getLogger(__name__)

#: How many layer rows are included in full in one land use block before being
#: truncated. A long tail of nearly empty layers is noise in a summary; the
#: limit is reported, not applied in silence (G7).
MAX_LAYER_ROWS = 25

#: How many layer names are sampled in the `unclassified_layers` block.
MAX_UNCLASSIFIED_SAMPLE = 10

#: How many unclassified layers are published WITH their geometric profile.
#:
#: A separate budget from the sample above, and deliberately not larger: a
#: profile is a nested block, not a name, and this drawing declares 251
#: unclassified layers. Publishing all of them would push the largest response
#: this API produces past what the agent can hold, and a tool that returns more
#: than its reader can hold silently loses the END of its own answer (G7). What
#: is dropped is counted and stated, never dropped in silence.
MAX_UNCLASSIFIED_PROFILES = 10

#: How many residual layers one zero-parcel row names (G7).
#:
#: Smaller than the profile budget because a residual is per `uses` row and a
#: response can carry several. Sixty `C-ROAD-*` sheet-layout layers match the
#: word "road" on the reference drawing; ranked by entity count the one real
#: centreline layer is first, and five rows are enough to show that the rest are
#: a different kind of thing. The full match count travels beside the list.
MAX_RESIDUAL_LAYERS = 5

#: How many anomaly kinds one profile names before truncation (G7).
MAX_ANOMALY_KINDS = 3

#: What is said in place of a unit when the drawing declares none (G2). Eleven
#: of the eighteen drawings here are in inches and three state nothing at all,
#: so a number that defaults to metres is a number that is wrong on most of
#: this corpus without ever looking wrong.
NO_UNIT_STATED = "this drawing states no unit; the number is in drawing units"

#: The sentence that accompanies every parcel figure. It names what IS counted
#: and what is not, because "how many plots" has four different answers
#: depending on whether hatches, labels, and block boundaries are included.
PARCEL_BASIS = (
    "parcel = an entity of type " + "/".join(sorted(RING_TYPES)) + " on a layer "
    "whose config states role=parcel. Hatches (role=overlay), labels "
    "(role=annotation), and block boundaries (role=structure) are NOT "
    "included, and their counts are reported separately in "
    "`roles_not_counted`. Networks (role=network) are not included either, and "
    "for a different reason: their native measure is LENGTH, not a count of "
    "plots, and their `roles_not_counted` row carries it. A zero in this frame "
    "is a statement about parcels and about nothing else."
)


class LandUseConfigInvalid(RuntimeError):
    """A land use config that cannot be used for this drawing.

    Wrapped from `landuse.ConfigError` so that callers at the HTTP layer have
    one type to catch without needing to know the contents of the config
    package. It carries `code`/`message`/`hint` like `store.MeasureRefused`, so
    the existing `@app.exception_handler` pattern can be reused and a violation
    comes out as an actionable 400 rather than a 500 that can only be stared
    at.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


# ---------------------------------------------------------------------------
# Scope and units
# ---------------------------------------------------------------------------


def _unit_names(drawing: Mapping[str, Any], layout: str | None) -> dict[str, Any]:
    """The units of this scope, as they come from `store`. See the note at the
    head of the file.

    `layout=None` deliberately does NOT use `_unit_names(drawing, None)`. That
    would return the drawing header's units, which describe model space only,
    for a total that also contains paper coordinates and coordinates inside
    block definitions. A total that adds sheet coordinates to model coordinates
    is not a quantity in any unit, and saying so is the only honest option —
    `_unit_names_for_layouts` holds that rule.
    """
    from . import store  # noqa: PLC0415 — deferred on purpose

    doc = dict(drawing)
    if layout is not None:
        return store._unit_names(doc, layout)
    names = [
        entry.get("name") if isinstance(entry, Mapping) else entry
        for entry in (doc.get("layouts") or [])
    ]
    return store._unit_names_for_layouts(doc, [str(n) for n in names if n])


def _includes_block_definitions(layout: str | None) -> bool:
    """Whether this scope contains entities inside block definitions.

    A block definition that was never placed is not in the drawing, and
    counting its parcels would report plots that are not drawn. That is why the
    answer travels with every response instead of being assumed.
    """
    if layout is None:
        return True
    return layout.startswith(BLOCK_LAYOUT_PREFIX)


def _scope(drawing: Mapping[str, Any], layout: str | None) -> ev.Scope:
    units = _unit_names(drawing, layout)
    in_blocks = _includes_block_definitions(layout)
    if layout is None:
        where = (
            "every layout in this drawing, including block definitions that "
            "may never have been placed"
        )
    elif in_blocks:
        where = f"only entities inside block definition {layout!r}"
    else:
        where = f"only entities on layout {layout!r}; block definitions are not included"
    unit_name = units.get("area_unit") or units.get("name") or "not stated"
    return ev.Scope(
        layout=layout,
        includes_block_definitions=in_blocks,
        units=units,
        note=f"{where}; areas in {unit_name}",
    )


# ---------------------------------------------------------------------------
# Reading the store
# ---------------------------------------------------------------------------


def _drawing(drawing_id: str) -> dict[str, Any] | None:
    return coll(COLL_DRAWINGS).find_one({"_id": drawing_id})


def _declared_layers(drawing: Mapping[str, Any]) -> dict[str, int]:
    """Layer name -> entity count across the WHOLE drawing, from the layer table.

    The layer table is the only place that knows about layers that are declared
    and unused. A query can only return layers that appear, and that
    seemingly complete list reads as "the others do not exist" — which is
    precisely the question being asked.
    """
    out: dict[str, int] = {}
    for entry in drawing.get("layers") or []:
        if isinstance(entry, Mapping):
            name = entry.get("name")
            if name:
                out[str(name)] = int(entry.get("entity_count") or 0)
        elif entry:
            out[str(entry)] = 0
    return out


def _groups_in_scope(drawing_id: str, layout: str | None) -> list[dict[str, Any]]:
    """Count and area per (layer, type) within the scope.

    One aggregation for the whole response. `drawing_id` is always the prefix
    of its filter, and that is not style: this cluster runs with `notablescan`,
    so a query without an indexed plan FAILS rather than slows down (and G10
    asks for the same thing for an entirely different reason).
    """
    match: dict[str, Any] = {"drawing_id": drawing_id}
    if layout is not None:
        match["layout"] = layout
    return list(
        coll(COLL_ENTITIES).aggregate(
            [
                {"$match": match},
                {
                    "$group": {
                        "_id": {"layer": "$layer", "type": "$type"},
                        "n": {"$sum": 1},
                        "n_measured": {
                            "$sum": {
                                "$cond": [
                                    {"$eq": [{"$ifNull": ["$area", None]}, None]},
                                    0,
                                    1,
                                ]
                            }
                        },
                        "area_sum": {"$sum": {"$ifNull": ["$area", 0.0]}},
                    }
                },
            ]
        )
    )


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------


def _area_quantity(
    rows: Sequence[Mapping[str, Any]],
    *,
    basis: str,
    unit: str | None,
    unit_reason: str | None,
) -> ev.Quantity:
    """A total area that cannot hide the part it could not measure.

    A HATCH carries no area, and a polygon with bulges has no ring to measure.
    Reporting `0` for either is the WRONG number, not an empty one — so the
    unmeasured part goes in as a `Quantity.withheld`, which has no `__float__`
    and blows up inside `sum()`.
    """
    method = "shoelace over the closed ring, computed in the ring's local frame"
    parts: list[ev.Quantity] = []
    for row in rows:
        layer = row["layer"]
        if row["n_measured"]:
            parts.append(
                ev.Quantity.measured(
                    basis=f"{row['n_measured']} parcels on layer {layer!r}",
                    method=method,
                    value=float(row["area_sum"]),
                    unit=unit,
                    unit_reason=unit_reason,
                )
            )
        withheld = int(row["n"]) - int(row["n_measured"])
        if withheld:
            parts.append(
                ev.Quantity.withheld(
                    basis=f"{withheld} parcels on layer {layer!r}",
                    method=method,
                    reason=(
                        "their polygons have no closed ring that can be "
                        "measured (bulge, open, non-planar, degenerate, or "
                        "oversize); their vertex chain is not the real boundary"
                    ),
                )
            )
    return ev.total(parts, basis=basis, method=method)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _layer_evidence(
    land_use: landuse.LandUse,
    *,
    claim: ev.Claim,
    drawing_id: str,
    scope: ev.Scope,
) -> ev.Evidence:
    return ev.Evidence.of(
        claim,
        drawing_id=drawing_id,
        scope=scope,
        provenance=land_use.provenance(),
        # The sources are taken WHOLE from the config and nothing is added to
        # them here. `Evidence.for_layer` would insert one extra layer-name
        # observation, and a source that appears in the response without
        # appearing in the YAML is exactly what makes a config stop being
        # reviewable. What guarantees there is no zero-source entry is the
        # loader's validator.
        observations=[s.observation(scope_layout=scope.layout) for s in land_use.sources],
        not_established=land_use.not_established,
        how_to_verify=land_use.how_to_verify,
    )


def _weakest_first(items: Sequence[ev.Evidence]) -> list[ev.Evidence]:
    """Order them so that `weakest()` inherits the weakest provenance.

    `evidence.weakest` takes its `provenance` from the FIRST member. A land use
    whose layers come partly from an override and partly from a global pattern
    could therefore report `verified: true` purely because of dict order — so
    the unverified ones go first, and the aggregate's `verified` is never
    stronger than the weakest config layer that makes it up.

    `sorted` is stable, so the order within each group stays the caller's order
    and the response does not shift between calls.
    """
    return sorted(items, key=lambda e: e.verified)


# ---------------------------------------------------------------------------
# The Dossier: what a layer IS, when no config says what it is FOR
# ---------------------------------------------------------------------------
#
# The campaign's origin is one sentence this file used to produce: **"0 road
# parcels"**, while 1,233 road entities and 83,962.565 m of centreline sat in
# `unclassified_layers` as a name in a list. The count was correct. Every word
# around it was missing.
#
# NOTHING IN THIS SECTION MAPS A LAND USE TO A GEOMETRIC ROLE, and nothing in it
# may. `road -> network` written in Python is a hardcoded ontology: it is what
# rule G1 forbids, it would need extending for every new use on every new
# client's drawing, and DOSSIER-02-DESIGN.md rejects it by name. What travels
# from here into lane A is the USE and the TOKENS that already speak for it in
# `landuse/_patterns.yaml` -- config, reviewable, and able to grow without a
# code change. Lane A does the matching with `evidence.speaks_for`, which is
# already implemented, already tested, and already refuses tokens that are too
# short or code-shaped.
#
# Two rules bind every function below:
#
# * **Additive only.** Not one existing key is renamed, removed or re-typed.
#   The viewer on 4310 and `web/app/api/agent/stream/route.ts` read this
#   response BY KEY, and a rename is invisible to pytest and fatal on screen.
# * **`None` and empty are different answers.** Lane A returns `None` when the
#   Dossier has not been built and a block with an empty list when the Dossier
#   exists and genuinely nothing matches. "Not computed" and "nothing there"
#   are the exact pair this whole campaign exists to keep apart, so they are
#   never allowed to collapse into the same empty list.


def _dossier_call(func_name: str, *args: Any, **kwargs: Any) -> tuple[Any, str | None]:
    """Call lane A, returning `(result, why_not_computed)`. Never raises.

    A module that is missing, a function it has not written yet, and a call that
    blew up on this particular drawing all produce a reason string that is
    published in the response. The summary is still built, the hole is named,
    and nothing anywhere becomes a zero because lane A was unavailable.
    """
    if _DOSSIER is None:
        detail = f" ({_DOSSIER_WHY})" if _DOSSIER_WHY else ""
        return None, f"cad_api/app/dossier_read.py is not available{detail}"
    func = getattr(_DOSSIER, func_name, None)
    if not callable(func):
        return None, (
            f"dossier_read.{func_name} is not available (the module exists but "
            "does not define it)"
        )
    try:
        return func(*args, **kwargs), None
    except Exception as exc:  # lane A must not be able to break this response
        return None, f"dossier_read.{func_name} raised {type(exc).__name__}: {exc}"


def _measure_block(raw: Any) -> dict[str, Any]:
    """One layer's native measure, read from the Dossier and never invented.

    Every field is copied or left `None`; not one of them is computed here. A
    measure this function cannot find is absent with a reason, which is the only
    honest shape for it (G8) -- a length that quietly became `0.0` because no
    Dossier entry existed would be indistinguishable from a layer that really
    holds nothing.
    """
    if not isinstance(raw, Mapping):
        return {
            "kind": None,
            "value": None,
            "unit": None,
            "unit_reason": "the Dossier entry for this layer carries no measure",
            "measured_entities": None,
            "unmeasured_entities": None,
            "basis": None,
        }
    unit = raw.get("unit")
    return {
        "kind": raw.get("kind"),
        "value": raw.get("value"),
        "unit": unit,
        # G2. A number without a unit says WHY it has none. It never falls back
        # to metres because the drawing in front of us happens to be in metres.
        "unit_reason": None if unit else (raw.get("unit_reason") or NO_UNIT_STATED),
        # G8. What could not be measured is counted and named beside the total,
        # never summed in as zero and never dropped from the denominator.
        "measured_entities": raw.get("measured_entities"),
        "unmeasured_entities": raw.get("unmeasured_entities"),
        "basis": raw.get("basis"),
    }


def _reads(role: Any, entities: Any, measure: Mapping[str, Any]) -> str:
    """One line a reader can compare two layers with, at a glance.

    This is the sentence that separates `00_Prop - Road - CL_` (network, 1,233
    entities, 83,962.565 m) from `C-ROAD-PROF-GRID-MINR` (a sheet-layout
    template with few entities and no measure). Both match the word "road"; only
    one of them is a road. The name proposes, the geometry disposes.
    """
    parts = [str(role) if role else "role not decided"]
    parts.append(
        f"{entities} entities"
        if isinstance(entities, int) and not isinstance(entities, bool)
        else "entity count not known"
    )
    kind = measure.get("kind")
    value = measure.get("value")
    if value is None:
        parts.append(f"{kind} not measured" if kind else "no length or area measured")
    else:
        unit = measure.get("unit")
        parts.append(
            f"{kind or 'measure'} {value} {unit}"
            if unit
            else f"{kind or 'measure'} {value} (no unit stated)"
        )
    return ", ".join(parts)


def _anomaly_kinds(raw: Any) -> list[str]:
    """The KINDS of anomaly on a layer, from any shape lane A may publish.

    A mapping with `flags` is one of those shapes, and missing it is not a
    hypothetical: the reference drawing's road layer arrived here with its
    triplication recorded and this function returned an empty list, so the
    residual published 83,962.565 m -- three copies of one estate -- with no
    hint that it counts the same road three times. The agent then quoted it
    faithfully. The tool misled the agent; the agent did not misread the tool.
    """
    if isinstance(raw, Mapping):
        for key in ("flags", "items", "anomalies", "findings"):
            if isinstance(raw.get(key), (list, tuple)):
                return _anomaly_kinds(raw[key])
        return []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        name = item.get("kind") if isinstance(item, Mapping) else item
        text = str(name).strip() if name is not None else ""
        if text and text not in out:
            out.append(text)
    return out


def _profile_row(layer: str, raw: Any) -> dict[str, Any]:
    """One layer as the Dossier measured it: role, count, native measure.

    Built key by key rather than by copying lane A's block whole. A Dossier
    layer bucket carries its full type map, its role-family counts, its census
    and lane 2's entire answer; pasting that per layer into a response that
    already runs to 60 KB on this drawing is precisely how a tool loses the end
    of its own answer (G7).
    """
    block = raw if isinstance(raw, Mapping) else {}
    measure = _measure_block(block.get("measure"))
    role = block.get("role")
    entities = block.get("entities")
    row: dict[str, Any] = {
        "layer": layer,
        "layout": block.get("layout"),
        "role": role,
        "role_basis": block.get("role_basis"),
        "entities": entities,
        "measure": measure,
        "reads": _reads(role, entities, measure),
    }
    kinds = _anomaly_kinds(block.get("anomalies")) or _anomaly_kinds(
        block.get("anomaly_flags")
    )
    if kinds:
        row["anomaly_kinds"] = kinds[:MAX_ANOMALY_KINDS]
        row["anomaly_kinds_truncated"] = len(kinds) > MAX_ANOMALY_KINDS
        row["anomaly_note"] = (
            "the Dossier flagged something about this layer's geometry. A "
            "measured total on a layer with `duplicate_clusters` can be a "
            "multiple of the truth -- ask `measure` before quoting it"
        )
        # The warning has to reach whoever quotes the FIGURE, and the figure
        # is right here. Sending the reader to another tool is not enough:
        # the measured case is a residual that published 83,962.565 m for a
        # layer drawn three times, and the number was quoted, in a sentence,
        # to a user, before anyone called `measure`. So the caveat travels
        # with the measure block it qualifies.
        if "duplicate_clusters" in kinds and isinstance(row.get("measure"), dict):
            row["measure"]["caveat"] = (
                "this layer is flagged `duplicate_clusters`: the total above "
                "sums every copy. Do not quote it as a real-world quantity "
                "without saying how many copies it counts -- `measure` on this "
                "layer carries the per-copy figure and the cluster count"
            )
    return row


def _rank(row: Mapping[str, Any]) -> tuple[int, int, str]:
    """Rank profiles by entity count, highest first; uncounted layers last.

    Ranking by SIZE is what makes the residual readable: the real feature layer
    arrives first and the template layers that share its name arrive after it,
    visibly smaller. A layer the Dossier could not count sorts last rather than
    sorting as though it held nothing.
    """
    n = row.get("entities")
    if isinstance(n, int) and not isinstance(n, bool):
        return (0, -n, str(row.get("layer") or ""))
    return (1, 0, str(row.get("layer") or ""))


def _profiles_by_layer(
    drawing_id: str, layer_names: Sequence[str], *, layout: str | None
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """`layer -> profile` from the Dossier, plus the reason it is empty if it is.

    The second return value is what keeps the two empties apart. `({}, None)`
    means the Dossier was read and simply has no entry for any of these layers;
    `({}, "<reason>")` means it was never read. A caller that publishes only the
    first value is publishing "nothing there" for both, which is the failure
    this campaign exists to end.

    A layer with no Dossier entry is ABSENT from the result rather than present
    with nulls -- lane A's contract, honoured here rather than papered over.
    """
    names = [str(n) for n in dict.fromkeys(str(n) for n in layer_names)]
    if not names:
        return {}, None
    result, why = _dossier_call("profiles_for", drawing_id, names, layout=layout)
    if why is not None:
        return {}, why
    if result is None:
        return {}, (
            "this drawing has no Dossier yet, so nothing has been measured "
            "about its layers. Build one with scripts/dossier_backfill.py. "
            "This is 'not computed'; it is NOT 'there is nothing there'"
        )
    if not isinstance(result, Mapping):
        return {}, (
            f"dossier_read.profiles_for returned {type(result).__name__}, not "
            "a mapping of layer name to profile"
        )
    return {str(k): _profile_row(str(k), v) for k, v in result.items()}, None


def _unclassified_profile_keys(
    drawing_id: str, layer_names: Sequence[str], *, layout: str | None
) -> dict[str, Any]:
    """The ADDITIVE keys of the `unclassified_layers` block.

    `count` and `sample` were always a dead end: the agent could see that 251
    layers are mapped by nobody and could read ten of their names, and had no
    way to tell a layer holding 1,233 road entities from an empty template
    layer. These three keys are what make the block's own note -- "a layer that
    is unclassified is not a layer that is empty" -- checkable instead of a
    disclaimer.
    """
    profiles, why = _profiles_by_layer(drawing_id, layer_names, layout=layout)
    rows = sorted(profiles.values(), key=_rank)
    shown = rows[:MAX_UNCLASSIFIED_PROFILES]
    asked = len({str(n) for n in layer_names})
    return {
        "profiles": shown,
        "profiles_status": {
            "status": "not_computed" if why else "computed",
            "why": why,
            "source": "dossier_read.profiles_for",
            "layers_asked_about": asked,
            "layers_the_dossier_knows": None if why else len(rows),
            "layers_shown": len(shown),
            "truncated": len(rows) > len(shown),
            "cap": MAX_UNCLASSIFIED_PROFILES,
            "ranked_by": (
                "entity count, highest first; layers the Dossier could not "
                "count come last"
            ),
        },
        "profiles_note": (
            "`count` and `sample` say WHICH layers no config maps; this list "
            "says what they ARE -- role, entity count, and the native measure "
            "for that role, with its unit. A layer the Dossier has no entry "
            "for is ABSENT from this list rather than present with nulls, and "
            "an empty list whose `profiles_status.status` is `not_computed` "
            "means the question was never asked, not that the answer is none"
        ),
    }


#: Key names lane A might publish its residual list under, in preference order.
#: The contract fixes the SIGNATURE of `residual_for`, not the field names
#: inside its block, and lane A is being written in parallel with this file. A
#: list looked up by several plausible names and reported by the name it was
#: actually found under is honest; a KeyError on the first guess is not.
_RESIDUAL_LIST_KEYS: tuple[str, ...] = (
    "layers",
    "candidates",
    "residual_layers",
    "profiles",
    "matches",
)


def _residual_list(block: Mapping[str, Any]) -> tuple[str | None, list[Any]]:
    """Lane A's residual list and the key it was found under."""
    for key in _RESIDUAL_LIST_KEYS:
        value = block.get(key)
        if isinstance(value, (list, tuple)):
            return key, list(value)
    for key, value in block.items():
        if (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, Mapping) for item in value)
        ):
            return str(key), list(value)
    return None, []


def _residual_row(item: Any) -> dict[str, Any]:
    """One residual entry, whether lane A sent a profile or a bare name."""
    if isinstance(item, Mapping):
        name = item.get("layer") or item.get("name") or item.get("layer_name") or ""
        return _profile_row(str(name), item)
    return _profile_row(str(item), {})


def _scalars(
    block: Mapping[str, Any], *, skip: Iterable[str] = ()
) -> tuple[dict[str, Any], list[str]]:
    """Lane A's own flat fields, plus the names of the ones not shown.

    Lane A answers with more than this file knows how to read, and dropping
    what it said in silence would be the same defect in miniature (G7). The
    scalars travel through whole; anything structured is named rather than
    pasted, so a reader can see that there is more and where it is.
    """
    skipped = set(skip)
    out: dict[str, Any] = {}
    not_shown: list[str] = []
    for key, value in block.items():
        name = str(key)
        if name in skipped:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            out[name] = value
        else:
            not_shown.append(name)
    return out, sorted(not_shown)


def _residual_keys(
    drawing_id: str,
    *,
    use: str,
    tokens: Sequence[str],
    classified_layers: Sequence[str],
    layout: str | None,
) -> dict[str, Any]:
    """The no-dead-end-zero rule, for one `uses` row that reports 0 parcels.

    A use that reports zero while unclassified layers hold entities must say so
    and must publish what those layers ARE.

    Two keys come back, and the difference between them is the whole point:

    * `residual_status` is ALWAYS present. It says whether the question was
      asked at all, and names the reason when it was not.
    * `residual` is present ONLY when the Dossier answered. Its absence is
      therefore readable as "this was never asked", and a `residual` that IS
      present and lists nothing is the other answer entirely -- it was asked,
      and no unclassified layer in this drawing speaks for this use.

    `tokens` come from `pattern_set.tokens_for(use)` and are passed straight
    through. This function does not know what a road is, has no table that says
    a road should be a network, and must never acquire one (G1).
    """
    status: dict[str, Any] = {
        "status": "not_computed",
        "why": None,
        "source": "dossier_read.residual_for",
        "use": use,
        "matched_on": list(tokens),
        "matched_on_source": (
            "the `vocabulary` block of landuse/_patterns.yaml, read with "
            "pattern_set.tokens_for(); the matching itself is "
            "evidence.speaks_for. No land use is mapped to a geometric role "
            "anywhere in this code (G1)"
        ),
        "cap": MAX_RESIDUAL_LAYERS,
    }
    out: dict[str, Any] = {"residual_status": status}

    if not tokens:
        status["why"] = (
            f"land use {use!r} has no tokens in the vocabulary, so there is no "
            "word to look for in a layer name. Add a `vocabulary:` entry for "
            "it in cad_api/app/landuse/_patterns.yaml"
        )
        return out

    result, why = _dossier_call(
        "residual_for",
        drawing_id,
        use,
        tuple(tokens),
        tuple(classified_layers),
        layout=layout,
    )
    if why is not None:
        status["why"] = why
        return out
    if result is None:
        status["why"] = (
            "this drawing has no Dossier yet, so the question 'is this land "
            "use present as something other than parcels' has not been asked. "
            "Build one with scripts/dossier_backfill.py. A zero with no "
            "residual is an unasked question, NOT an answer of none"
        )
        return out
    if not isinstance(result, Mapping):
        status["why"] = (
            f"dossier_read.residual_for returned {type(result).__name__}, not "
            "a mapping"
        )
        return out

    list_key, raw_rows = _residual_list(result)
    rows = sorted((_residual_row(item) for item in raw_rows), key=_rank)
    shown = rows[:MAX_RESIDUAL_LAYERS]
    extras, not_shown = _scalars(result, skip=[list_key] if list_key else [])

    status.update(
        {
            "status": "computed",
            "why": None,
            "layers_matched": len(rows),
            "layers_shown": len(shown),
            "truncated": len(rows) > len(shown),
            "dossier_list_key": list_key,
        }
    )
    out["residual"] = {
        "use": use,
        "matched_on": list(tokens),
        "layers": shown,
        "layers_matched": len(rows),
        "layers_shown": len(shown),
        "truncated": len(rows) > len(shown),
        "cap": MAX_RESIDUAL_LAYERS,
        "ranked_by": (
            "entity count, highest first; layers the Dossier could not count "
            "come last. Sixty sheet-layout layers can match the same word as "
            "one real feature layer, and only the count and the measure tell "
            "them apart -- the name proposes, the geometry disposes"
        ),
        "note": (
            f"{len(rows)} unclassified layer(s) in this drawing carry a name "
            f"that speaks for land use {use!r}. This row's 0 counts PARCELS: "
            "closed rings on layers a config maps to a use. No config maps "
            "these layers, so nothing here claims they ARE that land use -- "
            "what is published is their measured geometry, and a human still "
            "decides. An empty list is the Dossier answering 'nothing "
            "matches', which is not the same as the question not being asked"
        ),
        "from_dossier": extras,
        "from_dossier_keys_not_shown": not_shown,
    }

    # The name-blind half, forwarded whole rather than left in the
    # not-shown list. It is the answer to the question a name search cannot
    # ask: this drawing's `ROW` layer holds 91 right-of-way corridors over
    # 1,483,193.7 m2 and is the most defensible answer to "how many roads",
    # and the word "ROW" contains none of the nine road tokens, so the
    # matched list above will never contain it however good the vocabulary
    # gets. Published beside the matches, labelled as NOT name-matched, so
    # proximity in a list is never mistaken for a claim about meaning.
    largest = result.get("largest_unclassified")
    if isinstance(largest, list) and largest:
        # Published as the reader gave them, NOT re-cut to this file's own
        # smaller cap. `dossier_read` already ranks within each role and
        # interleaves the roles so a short list still shows one of each; a
        # second truncation here with a different number undoes that work
        # silently. Cutting these ten to five is what left the layer holding
        # 1.48 million m2 of right-of-way outside the answer while three
        # `mixed` layers were inside it.
        out["residual"]["largest_unclassified"] = [
            _profile_row(str(row.get("layer") or ""), row)
            for row in largest
            if isinstance(row, Mapping)
        ]
        out["residual"]["largest_unclassified_basis"] = result.get("largest_unclassified_basis")
        out["residual"]["largest_unclassified_note"] = (
            "the biggest unclassified layers in this drawing, ranked by what "
            "they HOLD rather than by what they are called, and excluding the "
            "name matches above. Nothing here claims to answer the question; "
            "a name search cannot see a layer whose name says nothing, so what "
            "it holds is offered and a human decides"
        )
    return out


def _network_keys(
    layer: str, *, profiles: Mapping[str, dict[str, Any]], why: str | None
) -> dict[str, Any]:
    """The LENGTH of a `role=network` layer, for its `roles_not_counted` row.

    This is why `network` was added to `landuse.ROLES`. Every other non-parcel
    role answers "it adds nothing to the plot count", which is true and tells
    its reader nothing at all. A network layer HAS a measured size; it is simply
    not a size in parcels, and reporting it as a parcel count of zero throws
    away the only number about it that exists.
    """
    profile = profiles.get(layer)
    out: dict[str, Any] = {
        "native_measure_note": (
            "role=network: this layer's native measure is LENGTH, not a count "
            "of plots. Its absence from the parcel count is a statement about "
            "parcels and about nothing else -- quote the length"
        ),
        "native_measure_status": {
            "status": "computed" if profile else "not_computed",
            "why": None
            if profile
            else (
                why
                or (
                    "this drawing's Dossier has no entry for this layer, so "
                    "its length has not been measured. Absent, not zero"
                )
            ),
            "source": "dossier_read.profiles_for",
        },
    }
    if not profile:
        return out

    out["native_measure"] = profile["measure"]
    out["dossier_role"] = profile["role"]
    out["dossier_entities"] = profile["entities"]
    out["reads"] = profile["reads"]
    dossier_role = profile.get("role")
    if dossier_role and dossier_role != landuse.NETWORK_ROLE:
        # Two different questions, and they are allowed to disagree. The config
        # says what a layer is FOR; the Dossier says what SHAPE its geometry
        # has. Resolving that silently in favour of either one would be a guess
        # wearing the clothes of a measurement.
        out["role_disagreement"] = (
            f"this drawing's config declares layer {layer!r} role=network, and "
            f"the Dossier measured its geometry as {dossier_role!r}. The config "
            "says what the layer is FOR and the Dossier says what shape it IS; "
            "they answer different questions and this one is not resolved here"
        )
    return out


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def land_use_summary(drawing_id: str, layout: str | None = None) -> dict[str, Any]:
    """What is in this drawing, read from the layer names via the land use config.

    **This is the FIRST tool for the question "what is in this drawing".** It
    answers "how many schools are there", "is there a mosque", "how many house
    plots are there and of what types" in a single call, with an evidence grade
    per land use.

    It is **not** the choice for finding one object — use `query_entities` —
    and **not** the choice for totalling one layer — use `measure`.

    Why it has to exist. A text search for "school" returns zero results on a
    drawing that contains nine education parcels, and that is correct: not one
    of the 43,109 TEXT/MTEXT contains that word. Zero results reads as "there
    are no schools", and a drawing containing 2,380 house plots was once
    answered with "there may be no houses here".

    What MUST be passed on when quoting its answer:

    * `evidence.grade` per land use. `stated`/`corroborated` may be answered
      plainly; `inferred` must be named as a conclusion together with its
      basis; `unknown` is answered with "I don't know" plus where to ask.
    * `empty_but_declared`. A layer that is in the layer table and contains
      nothing returns nil for any query, and that nil is not an absence.
    * that an area whose `withheld_reason` is filled in is NOT zero.
    * **`residual` beside a `parcels: 0`.** A zero counts PARCELS -- closed
      rings on layers a config maps to a use -- and says nothing about geometry
      no config claims. Where this drawing's Dossier could answer, a zero row
      names the unclassified layers whose NAME speaks for that use, with the
      role, entity count and native measure measured for each. A zero WITH a
      residual is not an absence, and "0 road parcels" quoted alone over 1,233
      road entities is the failure this whole tool exists to end.
    * `residual_status` when there is no `residual` at all: the question was
      not asked, because this drawing has no Dossier yet. Also not an absence.
    * `unclassified_layers.profiles`. The names in `sample` say which layers
      nobody has mapped; the profiles say what they ARE. A layer holding 1,233
      entities and 84 km of line and an empty template layer are the same word
      in `sample` and are plainly different here.

    What is ADDED by DOSSIER Phase 2, and it is added only -- not one existing
    key is renamed, removed or re-typed:

    * per `uses` row reporting 0 parcels: `residual_status`, and `residual`
      when the Dossier answered;
    * inside `unclassified_layers`: `profiles`, `profiles_status`,
      `profiles_note`;
    * on a `roles_not_counted` row whose config says `role=network`:
      `native_measure`, `native_measure_status`, `native_measure_note`,
      `dossier_role`, `dossier_entities`, `reads`, and `role_disagreement`
      when the config and the geometry do not agree;
    * at the top level: `residual_note`, and three more entries in `limits`.

    Every one of those blocks is size-budgeted and states its cap (G7), and
    every measure inside them carries its unit or the reason it has none (G2).

    Args:
        drawing_id: the drawing currently open.
        layout: one layout, e.g. model space. `None` means the whole drawing
            including block definitions, and the response says so.

    Returns:
        The land use summary, or `{}` if the drawing is not in the store.

    Raises:
        LandUseConfigInvalid: this drawing's config is structurally broken.
    """
    drawing = _drawing(drawing_id)
    if not drawing:
        return {}

    try:
        config = landuse.for_drawing(drawing_id)
        pattern_set = landuse.patterns()
    except landuse.ConfigError as exc:
        raise LandUseConfigInvalid(exc.code, exc.message, exc.hint) from exc

    declared = _declared_layers(drawing)
    scope = _scope(drawing, layout)
    # Resolved once. Two calls would be two reads of the drawing document for
    # one answer, and — worse — two chances for them to disagree.
    crs_choice = crs_for_drawing(drawing_id, drawing=drawing, config=config)
    units = dict(scope.units)
    area_unit = units.get("area_unit")
    unit_reason = (
        None
        if area_unit
        else (
            units.get("why_no_unit")
            or "this drawing states no unit; the numbers are in drawing units"
        )
    )

    # --- classification ----------------------------------------------------
    classified: dict[str, landuse.LandUse] = {}
    for layer in declared:
        hit = landuse.classify(
            drawing_id, layer, config=config, pattern_set=pattern_set
        )
        if hit is not None:
            classified[layer] = hit

    groups = _groups_in_scope(drawing_id, layout)
    in_scope: dict[str, dict[str, dict[str, Any]]] = {}
    for row in groups:
        layer = str(row["_id"].get("layer") or "")
        kind = str(row["_id"].get("type") or "")
        in_scope.setdefault(layer, {})[kind] = {
            "n": int(row["n"]),
            "n_measured": int(row["n_measured"]),
            "area_sum": float(row["area_sum"]),
        }

    # A layer that appears in the scope but not in the layer table must not
    # disappear in silence: a recovered file can contain entities on a layer
    # that its table does not name.
    for layer in in_scope:
        if layer not in declared and layer not in classified:
            hit = landuse.classify(
                drawing_id, layer, config=config, pattern_set=pattern_set
            )
            if hit is not None:
                classified[layer] = hit

    # Layers whose config declares them a NETWORK. Their native measure is
    # length, and it is fetched in ONE call for all of them rather than one call
    # per layer -- a summary over 292 layers must not turn into 292 reads.
    network_layers = sorted(n for n, lu in classified.items() if lu.is_network)
    network_profiles, network_why = _profiles_by_layer(
        drawing_id, network_layers, layout=layout
    )

    # --- rows per land use -------------------------------------------------
    by_use: dict[str, list[landuse.LandUse]] = {}
    for layer, land_use in classified.items():
        by_use.setdefault(land_use.use, []).append(land_use)

    uses: list[dict[str, Any]] = []
    not_counted: list[dict[str, Any]] = []
    use_evidence: dict[str, ev.Evidence] = {}
    all_open_questions: list[dict[str, Any]] = []

    for use in sorted(by_use):
        claim = ev.Claim(value=use, tokens=pattern_set.tokens_for(use))
        parcels_rows: list[dict[str, Any]] = []
        layer_rows: list[dict[str, Any]] = []
        absent: list[str] = []
        excluded_types: dict[str, int] = {}
        parcel_evidence: list[ev.Evidence] = []
        contributing_evidence: list[ev.Evidence] = []
        open_questions: list[dict[str, Any]] = []
        distinct_sources: dict[tuple[str, str], dict[str, Any]] = {}

        for land_use in sorted(by_use[use], key=lambda lu: lu.layer):
            evidence = _layer_evidence(
                land_use, claim=claim, drawing_id=drawing_id, scope=scope
            )
            types = in_scope.get(land_use.layer, {})

            if not land_use.is_parcel:
                not_counted_row: dict[str, Any] = {
                    **land_use.as_row(),
                    "entities": sum(t["n"] for t in types.values()),
                    "why_not_counted": (
                        f"role={land_use.role}: not a parcel, so it adds "
                        "nothing to the plot count and nothing to the area"
                    ),
                    "grade": evidence.grade.value,
                    "verified": evidence.verified,
                }
                if land_use.is_network:
                    # ADDITIVE. `why_not_counted` above is unchanged for every
                    # role, including this one -- it is read by name on screen.
                    # What a network row gains is the number that DOES exist
                    # for it.
                    not_counted_row.update(
                        _network_keys(
                            land_use.layer,
                            profiles=network_profiles,
                            why=network_why,
                        )
                    )
                not_counted.append(not_counted_row)
                continue

            ring_rows = [
                {"layer": land_use.layer, **stats}
                for kind, stats in types.items()
                if kind in RING_TYPES
            ]
            for kind, stats in types.items():
                if kind not in RING_TYPES:
                    excluded_types[kind] = excluded_types.get(kind, 0) + stats["n"]

            n_parcels = sum(int(r["n"]) for r in ring_rows)
            parcel_evidence.append(evidence)
            if n_parcels == 0:
                absent.append(land_use.layer)
            else:
                parcels_rows.extend(ring_rows)
                contributing_evidence.append(evidence)
                if land_use.not_established:
                    open_questions.append(
                        {
                            "layer": land_use.layer,
                            "not_established": land_use.not_established,
                            "how_to_verify": land_use.how_to_verify,
                        }
                    )
                for source in land_use.sources:
                    key = (source.origin.value, source.detail)
                    distinct_sources.setdefault(
                        key,
                        {
                            "origin": source.origin.value,
                            "detail": source.detail,
                            "observed": source.observed,
                            "layers": [],
                        },
                    )["layers"].append(land_use.layer)

            measured_here = sum(int(r["n_measured"]) for r in ring_rows)
            area_here = sum(float(r["area_sum"]) for r in ring_rows)
            layer_rows.append(
                {
                    **land_use.as_row(),
                    "parcels": n_parcels,
                    "parcels_measured": measured_here,
                    # Area per layer, and its mean, from the numbers that were
                    # computed just above and previously thrown away.
                    #
                    # This is not a convenience. "How many plots per type and
                    # what is their mean area" is a question that at the
                    # meeting of 23 August was answered outside the agent, and
                    # on 24 August the agent answered it with thirteen separate
                    # `stats` calls -- one per layer -- after a
                    # `land_use_summary` call that ALREADY contained the
                    # counts. On one of those runs the seventh call came out
                    # with a broken tool name and the whole answer was lost
                    # even though the other twelve calls had succeeded.
                    # Thirteen chances to slip, for a number that was already
                    # in hand.
                    #
                    # The divisor is `parcels_measured`, not `parcels`. A mean
                    # that divides by the full population while its numerator
                    # only sums the measured ones will be lower than the truth,
                    # and will look plausible.
                    "area_total": round(area_here, 6) if measured_here else None,
                    "area_mean": (
                        round(area_here / measured_here, 6) if measured_here else None
                    ),
                    "area_unit": area_unit,
                    "area_not_measured": (
                        None
                        if measured_here == n_parcels
                        else (
                            f"{n_parcels - measured_here} of {n_parcels} parcels "
                            "on this layer have no closed ring that can be "
                            "measured, so they enter neither the total nor the "
                            "mean -- and they do not enter as zero"
                        )
                    ),
                    "grade": evidence.grade.value,
                    "verified": evidence.verified,
                }
            )

        if not layer_rows:
            # Every layer of this land use has a non-parcel role: road labels,
            # hatches, block boundaries. It already appears in
            # `roles_not_counted`, and a `uses` row with 0 parcels and 0 layers
            # only invites its reader to add up something that is not a plot.
            continue

        # The evidence grade of this row describes THE PARCELS IT COUNTS, so it
        # is computed from the layers that actually contribute parcels. A land
        # use layer that is declared and EMPTY must not lower it:
        # `CS-Land use-00_Education` contributes zero parcels, and letting it
        # drag nine education parcels from `stated` down to `inferred` would
        # make the evidence grade sound weaker precisely because an empty layer
        # exists in the file. The empty layer still appears, named, in
        # `layers_with_no_entity_in_scope`.
        #
        # If nothing contributes at all, the grade is computed from every layer
        # with role parcel: the claim becomes "this land use is declared here",
        # and that still has a basis.
        merged = ev.weakest(
            _weakest_first(contributing_evidence or parcel_evidence), claim
        )
        use_evidence[use] = merged
        all_open_questions.extend({"use": use, **q} for q in open_questions)

        parcels = sum(int(r["n"]) for r in parcels_rows)
        measured = sum(int(r["n_measured"]) for r in parcels_rows)
        area = _area_quantity(
            parcels_rows,
            basis=f"Σ area of role=parcel parcels with land use {use!r}",
            unit=area_unit,
            unit_reason=unit_reason,
        )
        shown = layer_rows[:MAX_LAYER_ROWS]
        # Assembled into a name first, and appended below, so that the residual
        # can be added to it. Every key here is exactly the key it was before
        # (additive-only): the viewer and the web stream route read this row by
        # name, and a rename here is invisible to pytest and fatal on screen.
        row: dict[str, Any] = {
            "use": use,
            "parcels": parcels,
            "parcels_measured": measured,
            "parcels_area_withheld": parcels - measured,
            "total_area": area.as_dict(),
            # Two numbers, because "how many layers" has two answers and
            # naming only one once made an empty layer read as a land use
            # that is not there. `layers` counts the ones that actually
            # contribute parcels in this scope; the others are named one by
            # one below it.
            "layers": len(layer_rows) - len(absent),
            "layers_configured": len(layer_rows),
            "layers_with_no_entity_in_scope": absent,
            "types_on_these_layers_not_counted_as_parcels": [
                {"type": t, "entities": n} for t, n in sorted(excluded_types.items())
            ],
            "by_layer": shown,
            "by_layer_truncated": len(layer_rows) > len(shown),
            # An aggregate has no observations of its own: `weakest()`
            # deliberately publishes `sources: []`, because an aggregate
            # grade is computed from its members and not from new sources.
            # But "how do you know that is a house" demands that its bases
            # can be named, so the distinct bases are collected here —
            # as a list that is plainly derived, not as a fifth piece of
            # evidence that could raise the grade.
            "distinct_sources": sorted(
                distinct_sources.values(), key=lambda s: (s["origin"], s["detail"])
            )[:MAX_LAYER_ROWS],
            "open_questions": open_questions[:MAX_LAYER_ROWS],
            "evidence": merged.to_json(),
        }

        # THE NO-DEAD-END-ZERO RULE. A zero here counts parcels and nothing
        # else, and read on its own it says "there are none of these in this
        # drawing" -- which is how "0 road parcels" was once reported over 1,233
        # road entities and 84 km of centreline. A row that reports zero must
        # publish what this drawing holds under names that speak for its use,
        # and what those layers ARE.
        #
        # Only zero rows carry it. A row with parcels is already an answer, and
        # attaching a residual to it would spend the response's size budget
        # restating something the reader is not confused about (G7).
        if parcels == 0:
            row.update(
                _residual_keys(
                    drawing_id,
                    use=use,
                    # Straight through from the config vocabulary. This file has
                    # no table from land use to geometric role and must not
                    # acquire one (G1).
                    tokens=pattern_set.tokens_for(use),
                    classified_layers=sorted(classified),
                    layout=layout,
                )
            )
        uses.append(row)

    # Largest land use first: the first row is the one asked about most often,
    # and an agent that quotes only one row quotes that one. The layer name is
    # used as the tie-breaker so the order stays stable.
    uses.sort(key=lambda row: (-row["parcels"], row["use"]))

    # --- the unclassified --------------------------------------------------
    unclassified_declared = sorted(n for n in declared if n not in classified)
    unclassified_in_scope = sorted(
        n for n in in_scope if n not in classified and any(in_scope[n].values())
    )

    # --- the declared and empty --------------------------------------------
    #
    # Two sources, combined. The hand-written list in the config catches layers
    # that are not classified at all; the derived one catches every land use
    # layer whose layer table reports zero entities, so an empty layer that
    # first appears in the next revision does not have to wait for someone to
    # remember to add it. This is the most important block in the response: a
    # query to a layer like this returns nil, and that nil was once answered as
    # "there may be no houses in this drawing".
    listed = set(config.empty_but_declared) if config else set()
    derived = {name for name in classified if declared.get(name) == 0}
    empty_declared = sorted(listed | derived)
    empty_facts = [
        {
            "layer": name,
            "entities_in_file": declared.get(name),
            "in_layer_table": name in declared,
            "land_use": classified[name].use if name in classified else None,
            "from": (
                "config + derived"
                if name in listed and name in derived
                else ("config" if name in listed else "derived from the layer table")
            ),
        }
        for name in empty_declared
    ]

    body: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "scope_note": scope.note,
        "land_use_available": bool(classified),
        "config_version": config.version if config else pattern_set.version,
        "config_file": config.path if config else None,
        "config_file_expected_at": landuse.config_path(drawing_id).name,
        "config_layers_used": sorted(
            {lu.config_layer.value for lu in classified.values()}
        ),
        "config_problems": landuse.problems_against_drawing(config, declared),
        "vocabulary_problems": list(pattern_set.vocabulary_problems),
        "uses": uses,
        # `evidence.not_established` carries only ONE representative —
        # `weakest()` takes the first non-empty one and has nowhere to put the
        # rest. The full list is here, because "what does this typology code
        # mean" is answered from the individual rows and not from one
        # representative.
        "open_questions": all_open_questions,
        "roles_not_counted": not_counted,
        "parcel_basis": PARCEL_BASIS,
        "unclassified_layers": {
            "count": len(unclassified_declared),
            "count_basis": "layers this drawing declares that no config maps",
            "in_scope_count": len(unclassified_in_scope),
            "in_scope_basis": "of that number, the ones that actually contain "
            "entities within the scope of this response",
            "sample": unclassified_in_scope[:MAX_UNCLASSIFIED_SAMPLE]
            or unclassified_declared[:MAX_UNCLASSIFIED_SAMPLE],
            "sample_truncated": max(
                len(unclassified_in_scope), len(unclassified_declared)
            )
            > MAX_UNCLASSIFIED_SAMPLE,
            "note": "most of these are technical layers with no land use. "
            "A layer that is unclassified is not a layer that is empty",
            # ADDITIVE. `count`, `count_basis`, `in_scope_count`,
            # `in_scope_basis`, `sample`, `sample_truncated` and `note` above
            # are untouched. The population profiled is the same one `sample`
            # names, so the two describe the same layers and can be read
            # together.
            **_unclassified_profile_keys(
                drawing_id,
                unclassified_in_scope or unclassified_declared,
                layout=layout,
            ),
        },
        "empty_but_declared": empty_declared,
        "empty_but_declared_facts": empty_facts,
        "empty_but_declared_note": (
            "these layers are in the layer table and contain nothing. A query "
            "here returns nil; that does NOT mean their land use is absent "
            "from the drawing — check layers under other names before "
            "answering that there is none"
        ),
        "unverified_note": (
            "`verified` answers whether the layer→land use mapping has been "
            "confirmed by hand for THIS drawing: true for a per-drawing "
            "override, always false for a global pattern hit. It is a "
            "DIFFERENT axis from `grade`, which answers what the file says. A "
            "row can be verified=true and grade=inferred at once, and that is "
            "not a contradiction"
        ),
        "inferred_note": (
            "a row with grade `inferred` is not stated by the file; it is "
            "concluded from geometry, topology, or elimination. Say so when "
            "answering, and name the basis"
        ),
        # The coordinate system, as it is, including when there is none. This
        # block is here so that an agent asking "what is in this drawing" also
        # knows whether the answer can be put on a map — and so that it knows
        # when it cannot. Its evidence is nested inside it because the
        # top-level `evidence` belongs to the land use composition, not to the
        # CRS.
        # Resolved through the same order as every other answer. A drawing
        # whose file declares its zone must not read as "cannot be mapped"
        # here while `get_entity` places it — one of the two would be wrong,
        # and there is no way for a reader to tell which.
        "crs": _crs_dict(
            config,
            _crs_evidence(
                config, drawing_id=drawing_id, scope=scope, crs=crs_choice.crs
            ),
            crs_choice.crs,
        ),
        "area_unit": area_unit,
        "area_unit_reason": unit_reason,
        "area_scope": (
            "Σ area of role=parcel parcels that have a closed ring. A HATCH "
            "carries no area and is not included; a polygon with bulges has no "
            "ring that can be measured and its area is WITHHELD, not zero"
        ),
        "limits": {
            "layer_rows_per_use": MAX_LAYER_ROWS,
            "unclassified_sample": MAX_UNCLASSIFIED_SAMPLE,
            "unclassified_profiles": MAX_UNCLASSIFIED_PROFILES,
            "residual_layers_per_use": MAX_RESIDUAL_LAYERS,
            "anomaly_kinds_per_profile": MAX_ANOMALY_KINDS,
        },
        # The rule that made this whole campaign necessary, written where the
        # agent reads it rather than only in a design document.
        "residual_note": (
            "a `uses` row reporting `parcels: 0` carries `residual_status`, and "
            "carries a `residual` block whenever this drawing's Dossier was "
            "able to answer. The residual names unclassified layers whose NAME "
            "speaks for that land use, with the role, entity count and native "
            "measure the Dossier measured for each. A zero WITH a residual is "
            "NOT an absence: it means no config claims those layers, not that "
            "the thing is undrawn. Relay the residual in the same breath as "
            "the zero. A zero with `residual_status.status: not_computed` is a "
            "question that was never asked -- also not an absence"
        ),
    }

    if use_evidence:
        summary_claim = ev.Claim(
            value="the land use composition of this drawing", tokens=()
        )
        # The order follows the display order, not alphabetical order.
        # `weakest()` inherits `provenance` and `not_established` from the
        # FIRST member, so an unconsidered order would make a response about
        # 2,380 house plots carry the note of a commercial layer. The full list
        # is still in `open_questions`, because one representative is not all
        # of them.
        ordered = [use_evidence[row["use"]] for row in uses]
        overall = ev.weakest(_weakest_first(ordered), summary_claim)
    else:
        overall = _nothing_classified(drawing_id, scope, config, pattern_set)

    return ev.attach(body, overall)


def _nothing_classified(
    drawing_id: str,
    scope: ev.Scope,
    config: landuse.DrawingConfig | None,
    pattern_set: landuse.PatternSet,
) -> ev.Evidence:
    """The answer for a drawing with not one classified layer (G3).

    Not a `500`, not `{}`, and not a guessed classification. A new drawing is
    never refused because its config does not exist yet — it comes in, is read
    in part, and this response says what is missing and which file needs to be
    created.
    """
    path = landuse.config_path(drawing_id).name
    if config is None:
        gap = (
            f"this drawing has no land use config yet, and not one of its "
            f"layer names matches any of the {len(pattern_set.rules)} global "
            f"patterns"
        )
        how = (
            f"create `cad_api/app/landuse/{path}` and map its layer names to "
            "the land use vocabulary. Until that exists, the only honest "
            "answer about this drawing's land use is that it has not been read"
        )
    else:
        gap = (
            f"config {config.path} exists but not one of its layers matches "
            "this drawing's layer table"
        )
        how = (
            "check `config_problems` in this response: it names the layers "
            "that are in the config and not in the drawing"
        )
    return ev.Evidence.unknown(
        ev.Claim(value="the land use composition of this drawing", tokens=()),
        drawing_id=drawing_id,
        scope=scope,
        provenance=ev.Provenance(
            config_layer=(
                ev.ConfigLayer.NO_CONFIG if config is None else ev.ConfigLayer.DRAWING_OVERRIDE
            ),
            config_version=config.version if config else None,
            note="no layer in this drawing is classified",
        ),
        not_established=gap,
        how_to_verify=how,
    )


# ---------------------------------------------------------------------------
# Georeferencing (UPLIFT-14)
# ---------------------------------------------------------------------------

#: The order of points used to place an entity on the earth, and the sentence
#: that accompanies each one. `bbox_centre` is on this list and ALWAYS carries
#: its caveat: it is the centre of the bounding box, not the centre of the
#: shape, and for an L-shaped polygon the two can be far apart. It is used
#: because a LINE or a CIRCLE genuinely has nothing else — not as a silent
#: fallback for a polygon whose centroid failed to compute.
#: Public alias. `geo_h3` reads the same priority so that an entity's cell and
#: its reported position are computed from the SAME point — on a non-convex
#: polygon the centroid and the bounding-box centre are 72.7 m apart on this
#: drawing, which is a different parcel.
_POINT_FIELDS: tuple[tuple[str, str], ...] = (
    ("polygon_centroid", "the polygon's area centroid, computed in the ring's local frame"),
    ("anchor_point", "the entity's insertion point, as AutoCAD places it"),
    (
        "bbox_centre",
        "THE CENTRE OF THE BOUNDING BOX, not the centre of the shape; for "
        "shapes that are not convex the two can be far apart",
    ),
)


#: "the caller did not resolve a CRS for me" — distinct from `None`, which
#: means "there is no CRS for this drawing". Without the distinction the two
#: helpers below could not keep their old single-argument behaviour, and every
#: existing caller would have had to be found and changed correctly on the
#: first try.
_UNSET_CRS: Any = object()


@dataclass(frozen=True)
class CrsChoice:
    """Which coordinate system this drawing gets, and why that one.

    Three fields because three different readers need three different things:
    the converter needs `crs`, an auditor needs `layer` to know whether the
    file was believed or a human was, and anybody debugging a drawing that
    produces no lat/long needs `note` — which is the only one of the three
    that is populated precisely when the other two are least useful.
    """

    crs: crs_math.Crs | None
    #: "declared_in_file", "drawing_config", or None when there is no CRS.
    layer: str | None
    #: Why the file's own declaration was not used, when there was one and it
    #: was not. `None` when the file declared nothing, or when what it
    #: declared is what is being used.
    note: str | None = None


#: How a CRS the file states is recorded as evidence. `XDATA` is the origin
#: for structured data carried by the file itself; through `CORPUS_OF` it maps
#: to `STRUCTURED`, which — unlike geometry and absence — is allowed to STATE.
#: That is the whole difference between a declared CRS and an inferred one:
#: the inferred one is capped at `inferred` no matter how convincing it is,
#: and this one is the file speaking.
_DECLARED_ORIGIN = ev.Origin.XDATA


def _crs_from_declaration(declared: Mapping[str, Any]) -> tuple[crs_math.Crs | None, str | None]:
    """A `Crs` built from what the file declared, or the reason there is none.

    Never raises. A declaration this service cannot act on is a sentence, not
    an exception: the drawing still has to be readable, and every other answer
    about it is unaffected by the fact that nobody can put it on a map.
    """
    unreadable = declared.get("unreadable")
    if unreadable:
        return None, (
            "this file carries a GEODATA object, but its coordinate system "
            f"could not be read ({unreadable})"
        )

    epsg = declared.get("epsg")
    if epsg is None:
        return None, (
            "this file carries a GEODATA object that names no EPSG code"
        )

    if declared.get("xy_ordering") is False:
        # Northing-first. The converter takes (easting, northing); handing it
        # a northing-first pair produces a position that looks perfectly
        # ordinary and is thousands of kilometres away. Refused rather than
        # silently swapped, because "swap them" is a guess about what the
        # author meant.
        return None, (
            f"this file declares EPSG:{epsg} with a northing-first axis "
            "order, which this service does not convert. Swapping the axes "
            "would be a guess, and a wrong guess here still produces numbers "
            "shaped like a position"
        )

    try:
        crs_math.zone_of(int(epsg))
    except crs_math.CrsUnsupported as exc:
        return None, f"this file declares EPSG:{epsg}, and {exc.message[0].lower()}{exc.message[1:]} {exc.hint}"

    zone, north = crs_math.zone_of(int(epsg))
    return (
        crs_math.Crs(
            epsg=int(epsg),
            name=f"WGS 84 / UTM zone {zone}{'N' if north else 'S'}",
            declared_in_file=True,
            note="read from the GEODATA object in the file, not configured",
            sources=(
                landuse.Source(
                    origin=_DECLARED_ORIGIN,
                    detail=str(declared.get("source") or "GEODATA object"),
                    observed=f"EPSG:{epsg}",
                ),
            ),
        ),
        None,
    )


def crs_for_drawing(
    drawing_id: str,
    *,
    drawing: Mapping[str, Any] | None = None,
    config: "landuse.DrawingConfig | None" = None,
) -> CrsChoice:
    """The coordinate system to use for one drawing, and which layer won.

    The order is **declared in the file, then per-drawing config, then
    nothing**, and the reason is not preference — it is authority. A GEODATA
    object is the drawing's author saying which system these coordinates are
    in. Our config is somebody reading where the extents land and concluding a
    zone; on the reference drawing that inference is convincing, and it is
    still an inference, which is why every lat/long it produces carries a
    warning that the declared path does not need.

    A declaration this service cannot act on does NOT silently promote the
    config to first place. It falls back — that is what a fallback is — but
    the refusal travels with the answer in `note`, so a drawing being placed
    by an inferred zone *while its own file says something else* is visible
    rather than discovered later. That case is the one worth being loud
    about: it is the shape of a wrong map that nobody questions.

    Passing `drawing` and `config` avoids re-reading them when the caller
    already has them; both are fetched when they are not supplied.
    """
    if drawing is None:
        drawing = _drawing(drawing_id) or {}
    if config is None:
        config = landuse.for_drawing(drawing_id)

    declared = drawing.get("declared_crs") if drawing else None
    note: str | None = None
    if isinstance(declared, Mapping):
        crs, note = _crs_from_declaration(declared)
        if crs is not None:
            return CrsChoice(crs=crs, layer="declared_in_file", note=None)

    configured = config.crs if config else None
    if configured is not None:
        if note:
            note = (
                f"{note}. The zone in this drawing's config is being used "
                "instead, and it is an inference — check the two against each "
                "other before quoting any position from this drawing"
            )
        return CrsChoice(crs=configured, layer="drawing_config", note=note)

    return CrsChoice(crs=None, layer=None, note=note)


#: See the comment on `_POINT_FIELDS`.
POINT_FIELDS = _POINT_FIELDS


def _crs_evidence(
    config: landuse.DrawingConfig | None,
    *,
    drawing_id: str,
    scope: ev.Scope,
    crs: crs_math.Crs | None = _UNSET_CRS,
) -> ev.Evidence:
    """Evidence for the claim "this drawing sits in coordinate system X".

    Its basis is geometry (where the extents land after inversion) and absence
    (there is no GEODATA object). Both are corpora that NEVER state, so this
    claim has no route above `inferred` — and that is exactly its class: zone
    38N puts the reference drawing in Janadriyah and zone 37N in the Red Sea,
    which is convincing without ever becoming a statement by the file.
    """
    if crs is _UNSET_CRS:
        crs = config.crs if config else None
    if crs is None:
        return ev.Evidence.unknown(
            ev.Claim(value="the coordinate system of this drawing", tokens=()),
            drawing_id=drawing_id,
            scope=scope,
            provenance=ev.Provenance(
                config_layer=(
                    ev.ConfigLayer.NO_CONFIG
                    if config is None
                    else ev.ConfigLayer.DRAWING_OVERRIDE
                ),
                config_version=config.version if config else None,
                note="there is no `crs:` block for this drawing",
            ),
            not_established=(
                "this drawing has no CRS config, and no zone has been guessed "
                "for it"
            ),
            how_to_verify=(
                "check whether the file has a GEODATA object; if not, ask the "
                "drawing's author for its coordinate system and write a "
                f"`crs:` block in cad_api/app/landuse/{landuse.config_path(drawing_id).name}"
            ),
        )
    return ev.Evidence.of(
        ev.Claim(value=f"{crs.name} (EPSG:{crs.epsg})", tokens=()),
        drawing_id=drawing_id,
        scope=scope,
        provenance=ev.Provenance(
            config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
            config_version=config.version if config else None,
            note=crs.note,
        ),
        observations=[s.observation(scope_layout=scope.layout) for s in crs.sources],
        not_established=crs.not_established,
        how_to_verify=crs.how_to_verify,
    )


def _crs_dict(
    config: landuse.DrawingConfig | None,
    evidence: ev.Evidence,
    crs: crs_math.Crs | None = _UNSET_CRS,
) -> dict[str, Any]:
    if crs is _UNSET_CRS:
        crs = config.crs if config else None
    if crs is None:
        block = crs_math.unknown_crs(
            evidence.not_established or "there is no CRS config for this drawing",
            evidence.how_to_verify
            or "ask the drawing's author for its coordinate system",
        )
    else:
        block = crs.as_dict()
        block["caveat"] = crs_math.caveat(crs)
        block["grade"] = evidence.grade.value
        # Two keys that answer two different questions, and naming only one is
        # the easiest way to make an inferred lat/long read as official.
        # `verified` = someone typed an override for this drawing (always true
        # for a hand-written config, D-082). `confirmed_by` = someone OUTSIDE
        # the file confirmed the zone, and the evidence contract demands that a
        # source of that kind names the document or the person.
        block["verified"] = evidence.verified
        block["confirmed_by"] = crs_math.confirmed_by(crs)
    block["evidence"] = evidence.to_json()
    return block


def describe_crs(drawing_id: str) -> dict[str, Any]:
    """The `crs` block for `describe_drawing`, including when there is none.

    It returns THREE keys — `crs`, `scope_note`, `evidence` — and all three
    have to be merged into the caller's response. `crs` is a meaning-bearing
    key according to `evidence.MEANING_BEARING_KEYS`, so a response that
    carries it without a scope and without evidence will fail
    `audit_response()`.

    For a drawing with no CRS config it returns `{"known": false, ...}`
    together with the reason — not silence, not `{}`, and not a guessed zone.
    """
    drawing = _drawing(drawing_id)
    if not drawing:
        return {}
    try:
        config = landuse.for_drawing(drawing_id)
    except landuse.ConfigError as exc:
        raise LandUseConfigInvalid(exc.code, exc.message, exc.hint) from exc

    scope = _scope(drawing, None)
    choice = crs_for_drawing(drawing_id, drawing=drawing, config=config)
    evidence = _crs_evidence(
        config, drawing_id=drawing_id, scope=scope, crs=choice.crs
    )
    block = _crs_dict(config, evidence, choice.crs)
    if choice.crs is not None:
        block["bounds"] = crs_math.bounds_lat_lon(choice.crs, drawing.get("extents"))
    # Which layer answered, and — when the file's own declaration was passed
    # over — why. Both are additive keys; `known` still answers the question
    # everybody asks first.
    block["source_layer"] = choice.layer
    block["declaration_note"] = choice.note
    body = {"drawing_id": drawing_id, "crs": block, "scope_note": scope.note}
    return ev.attach(body, evidence)


def lat_lon_for_entity(drawing_id: str, handle: str) -> dict[str, Any] | None:
    """`lat_lon` for one entity, or `None` if it cannot be placed.

    To be attached to a `get_entity` row under the key **`lat_lon`** — not as a
    bare `lat` and `lon` at the top level, which would require the whole
    `get_entity` response to carry the evidence contract. The evidence travels
    inside this block.

    `None` for three different reasons, and the caller does not need to tell
    them apart: a drawing with no CRS config, an entity that does not exist,
    and an entity with no point at all. All three mean the same thing on
    screen — no lat/long may be written — and what tells them apart lives in
    `describe_crs()`.
    """
    try:
        config = landuse.for_drawing(drawing_id)
    except landuse.ConfigError as exc:
        raise LandUseConfigInvalid(exc.code, exc.message, exc.hint) from exc
    drawing = _drawing(drawing_id)
    choice = crs_for_drawing(drawing_id, drawing=drawing or {}, config=config)
    crs = choice.crs
    if crs is None:
        return None

    row = coll(COLL_ENTITIES).find_one(
        {"drawing_id": drawing_id, "handle": handle},
        {f: 1 for f, _ in _POINT_FIELDS} | {"layout": 1},
    )
    if not row:
        return None

    for field, basis in _POINT_FIELDS:
        point = crs.point(row.get(field))
        if point is None:
            continue
        scope = _scope(drawing or {}, row.get("layout"))
        evidence = _crs_evidence(
            config, drawing_id=drawing_id, scope=scope, crs=crs
        )
        return {
            **point,
            "from_field": field,
            "basis": basis,
            "epsg": crs.epsg,
            "crs_name": crs.name,
            "crs_declared_in_file": crs.declared_in_file,
            "crs_verified": evidence.verified,
            "crs_confirmed_by": crs_math.confirmed_by(crs),
            "crs_grade": evidence.grade.value,
            "caveat": crs_math.caveat(crs),
            # Which layer this zone came from, and the reason the file's own
            # declaration was not used when there was one. A position placed
            # by an inferred zone while the file says something else is the
            # shape of a wrong map nobody questions, so it says so here.
            "crs_source_layer": choice.layer,
            "crs_declaration_note": choice.note,
            "evidence": evidence.to_json(),
        }
    return None


#: Decimal places kept on every projected coordinate. The seventh decimal of a
#: degree is about 11 mm, which is already far finer than a CRS nobody has
#: confirmed can justify -- and the digits past it are the Krüger series
#: talking to itself. Stated here rather than left to `repr(float)` because a
#: ring of 4,096 vertices is a payload, and seventeen digits of it would be
#: sixty percent noise.
GEOJSON_DECIMALS = 7


def _ring_as_geojson(
    crs: crs_math.Crs, ring: Sequence[Sequence[float]]
) -> list[list[float]] | None:
    """One stored ring as a GeoJSON linear ring, or `None` if it cannot be one.

    Three things happen here and each is a rule of the format rather than a
    choice:

    - **[lon, lat], not [lat, lon].** GeoJSON puts longitude first. Our own
      converter returns latitude first, which is the older convention and the
      reason this is the single easiest way to put Janadriyah in Somalia.
    - **The ring is closed.** Stored rings are deduplicated, so the first
      vertex is never repeated at the end; RFC 7946 requires that it is.
    - **The exterior ring is counterclockwise**, by RFC 7946. Decided from the
      signed area of the PROJECTED ring rather than from the stored
      `ring_orientation`, which describes the drawing's own frame -- one
      assumption fewer about what the projection does to winding.
    """
    out: list[list[float]] = []
    for xy in ring:
        if xy is None or len(xy) < 2:
            return None
        lat, lon = crs.to_lat_lon(float(xy[0]), float(xy[1]))
        out.append([round(lon, GEOJSON_DECIMALS), round(lat, GEOJSON_DECIMALS)])
    if len(out) < 3:
        return None

    # Shoelace over the projected ring. Degrees, and only its SIGN is read --
    # this is a winding test, never an area. Area stays a drawing-coordinate
    # truth and is not written into any GeoJSON this module produces.
    twice_area = 0.0
    for i in range(len(out)):
        x1, y1 = out[i]
        x2, y2 = out[(i + 1) % len(out)]
        twice_area += x1 * y2 - x2 * y1
    if twice_area < 0:
        out.reverse()

    out.append(list(out[0]))
    return out


def geometry_geojson_for_entity(drawing_id: str, handle: str) -> dict[str, Any] | None:
    """One entity's boundary as a GeoJSON Feature, or `None`.

    **Display-grade only.** Every coordinate here has been through a projection
    inferred from where the drawing's extents sit, and the whole point of the
    inference is that nobody outside the file has confirmed it. Areas and
    lengths continue to be measured in drawing coordinates and are deliberately
    NOT written into these properties: a reader who could take an area off this
    feature would be taking it off the approximation instead of off the
    drawing.

    A Feature rather than a bare geometry, and that is the one debatable
    decision in this function. `geometry_geojson` names a geometry, and a
    geometry is what a mapping tool wants. But a bare geometry has nowhere to
    carry the sentence saying the coordinate system was inferred, and this repo
    has exactly one inviolable rule about lat/long: the warning travels with
    the number, never with the documentation (`UPLIFT-14`, DoD 4). A Feature
    can carry it in `properties`, is still valid GeoJSON to every tool that
    reads one, and drops into a FeatureCollection unchanged.

    `None` for a drawing with no CRS config, for an entity that does not
    exist, and for any ring this store will not vouch for. Nothing is invented
    to fill the gap: the entity's own `geometry_note` already says why a ring
    is unusable -- bulged, open, non-planar, degenerate or oversize -- and
    repeating it here would be a second copy free to disagree with the first.
    """
    try:
        config = landuse.for_drawing(drawing_id)
    except landuse.ConfigError as exc:
        raise LandUseConfigInvalid(exc.code, exc.message, exc.hint) from exc
    crs = crs_for_drawing(drawing_id, config=config).crs
    if crs is None:
        return None

    row = coll(COLL_ENTITIES).find_one(
        {"_id": f"{drawing_id}:{handle}"},
        {
            "ring": 1,
            "ring_status": 1,
            "ring_vertex_count": 1,
            "layer": 1,
            "layout": 1,
            "type": 1,
        },
    )
    if not row or row.get("ring_status") != "complete":
        return None
    coordinates = _ring_as_geojson(crs, row.get("ring") or [])
    if coordinates is None:
        return None

    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
        "properties": {
            "handle": handle,
            "drawing_id": drawing_id,
            "layer": row.get("layer"),
            "layout": row.get("layout"),
            "type": row.get("type"),
            "epsg": crs.epsg,
            "crs_name": crs.name,
            "crs_declared_in_file": crs.declared_in_file,
            "caveat": crs_math.caveat(crs),
            "ring_vertex_count": row.get("ring_vertex_count"),
            "note": (
                "display only: reprojected vertex by vertex from drawing "
                "coordinates. Areas and lengths must be read from the "
                "drawing, not measured off this shape."
            ),
        },
    }


#: Cells returned inside one entity's `h3` block before the list is withheld.
#:
#: `get_entity` goes straight into an agent tool call. The site-boundary
#: polygon covers 20,000 cells and took that response from 15 KB to 392 KB
#: when the raw field was included, so the covering is only handed over when
#: it is small enough to read. Above the cap the count is still there and the
#: block says the list was withheld — never a silent truncation, and never a
#: shorter list pretending to be the whole one (G7).
MAX_ENTITY_CELLS = 500


def h3_for_entity(drawing_id: str, handle: str) -> dict[str, Any] | None:
    """One entity's cells and the resolution they are at, or `None`.

    The shaped view of the `h3_*` fields the store writes, and the reason it
    exists is that the flat fields cannot carry the covering: `get_entity`
    withholds `h3_cells` for the same reason it withholds `ring`, which leaves
    a reader of one object with no way to see which cells it occupies.

    `None` for an entity with no cells at all. The flat `h3_note` already says
    why in that case — paper space, inside a block definition, or no point
    inside the drawing — and a second copy of that sentence is a second copy
    free to disagree with the first.
    """
    row = coll(COLL_ENTITIES).find_one(
        {"_id": f"{drawing_id}:{handle}"},
        {
            "h3_cell": 1,
            "h3_cells": 1,
            "h3_res": 1,
            "h3_cell_count": 1,
            "h3_strategy": 1,
            "h3_cells_truncated": 1,
        },
    )
    if not row or not row.get("h3_cell"):
        return None

    cells = list(row.get("h3_cells") or [])
    withheld = len(cells) > MAX_ENTITY_CELLS
    return {
        "resolution": row.get("h3_res"),
        "cell": row.get("h3_cell"),
        "cells": None if withheld else cells,
        "cell_count": row.get("h3_cell_count"),
        "strategy": row.get("h3_strategy"),
        "cells_truncated": row.get("h3_cells_truncated"),
        "cells_withheld": (
            None
            if not withheld
            else (
                f"{len(cells):,} cells is more than one entity response may "
                f"carry; the limit is {MAX_ENTITY_CELLS:,}. The covering for "
                "the whole drawing, at any resolution, is on "
                "GET /drawings/{id}/geo"
            )
        ),
    }


def geo_for_entity(drawing_id: str, handle: str) -> dict[str, Any]:
    """The georeferencing keys `get_entity` gains, including when there are none.

    Always the same three keys, so its reader needs no branch: `lat_lon`,
    `geometry_geojson`, and `geo_note` -- the last saying why the first two are
    null when they are. Silence was the alternative, and an endpoint that
    simply omits a lat/long is indistinguishable from one that has not been
    deployed yet (G3).

    `geo_note` answers only the drawing-level question, "may this drawing have
    a lat/long at all". The entity-level reasons are already in the response:
    an entity with no point has no point fields, and a ring this store will not
    vouch for carries `geometry_note` saying which of the five reasons applies.
    """
    try:
        config = landuse.for_drawing(drawing_id)
    except landuse.ConfigError as exc:
        # Deliberately NOT raised. The land use config is an extra on this
        # endpoint, and a drawing whose YAML has a typo must still be readable
        # -- get_entity answered before this block existed and has to go on
        # answering. The reason travels in the note instead of in a 4xx.
        return {
            "lat_lon": None,
            "geometry_geojson": None,
            "h3": None,
            "geo_note": (
                "this drawing's config cannot be read, so no lat/long is "
                f"written: {exc.message}"
            ),
        }

    choice = crs_for_drawing(drawing_id, config=config)
    if choice.crs is None:
        # The file's own refusal, when it made one, outranks the generic
        # sentence: "this GEODATA declares a projection we cannot invert" is a
        # lead, and "there is no CRS config" is not.
        return {
            "lat_lon": None,
            "geometry_geojson": None,
            "h3": None,
            "geo_note": choice.note
            or (
                "this drawing has no CRS config, so no lat/long is written for "
                "it. Drawing coordinates are the only thing the file states; a "
                "zone guessed for it would produce a plausible-looking "
                "position somewhere else in the world."
            ),
        }

    lat_lon = lat_lon_for_entity(drawing_id, handle)
    geometry = geometry_geojson_for_entity(drawing_id, handle)
    note = choice.note
    if lat_lon is None and note is None:
        note = (
            "this drawing is georeferenced, but this entity carries no point "
            "that could be placed"
        )
    return {
        "lat_lon": lat_lon,
        "geometry_geojson": geometry,
        "geo_note": note,
        "h3": h3_for_entity(drawing_id, handle),
    }


# ---------------------------------------------------------------------------
# Enrichment of existing tools
# ---------------------------------------------------------------------------


def classify_layers(
    drawing_id: str, layer_names: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Land use per layer name, to be attached to `query_entities` rows.

    A layer that is not classified does NOT appear in the result. The caller
    writes `null` for all three — not `"unknown"`, which reads as a statement
    that this drawing says its land use is unknown.
    """
    config = landuse.for_drawing(drawing_id)
    pattern_set = landuse.patterns()
    out: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(layer_names):
        hit = landuse.classify(
            drawing_id, str(name), config=config, pattern_set=pattern_set
        )
        if hit is None:
            continue
        out[str(name)] = {
            "land_use": hit.use,
            "land_use_subtype": hit.subtype,
            "land_use_role": hit.role,
            "land_use_verified": hit.provenance().verified,
            "land_use_config_layer": hit.config_layer.value,
        }
    return out


def availability(drawing_id: str, drawing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Whether `land_use_summary` is worth calling for this drawing.

    For `describe_drawing`: an agent that does not know this tool exists will
    fall back to a text search, and a text search is the wrong instrument for
    this question.
    """
    try:
        config = landuse.for_drawing(drawing_id)
        pattern_set = landuse.patterns()
    except landuse.ConfigError as exc:
        return {
            "land_use_available": False,
            "land_use_config_version": None,
            "land_use_note": f"this drawing's land use config cannot be used: {exc.message}",
        }

    doc = drawing if drawing is not None else _drawing(drawing_id)
    declared = _declared_layers(doc) if doc else {}
    hits = sum(
        1
        for layer in declared
        if landuse.classify(drawing_id, layer, config=config, pattern_set=pattern_set)
        is not None
    )
    return {
        "land_use_available": bool(hits),
        "land_use_config_version": config.version if config else pattern_set.version,
        "land_use_layers_classified": hits,
        "land_use_layers_declared": len(declared),
        "land_use_note": (
            "call land_use_summary for the question 'what is in this drawing'"
            if hits
            else "this drawing has no land use config yet; "
            f"the file being waited for is cad_api/app/landuse/{landuse.config_path(drawing_id).name}"
        ),
    }


def uses_matching(drawing_id: str, query: str) -> list[dict[str, Any]]:
    """The land uses the search word names, for `search_text` near-misses.

    A text search for "SCHOOL" returns zero rows on a drawing that contains
    nine education parcels. Zero rows that are correct and misleading is the
    most expensive failure in this project, and this list is what makes the
    answer name the land use rather than only the layer name.
    """
    needle = (query or "").strip()
    if not needle:
        return []
    try:
        config = landuse.for_drawing(drawing_id)
        pattern_set = landuse.patterns()
    except landuse.ConfigError:
        return []

    by_use: dict[str, list[str]] = {}
    for layer in _declared_layers(_drawing(drawing_id) or {}):
        hit = landuse.classify(
            drawing_id, layer, config=config, pattern_set=pattern_set
        )
        if hit is not None:
            by_use.setdefault(hit.use, []).append(layer)

    out: list[dict[str, Any]] = []
    for use in sorted(by_use):
        token = ev.speaks_for(needle, pattern_set.tokens_for(use))
        if token is None:
            continue
        out.append(
            {"use": use, "matched_token": token, "layers": sorted(by_use[use])}
        )
    return out
