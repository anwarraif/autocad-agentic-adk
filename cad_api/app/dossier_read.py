"""The only module that reads `autocad_dossiers`.

Owner: lane A of DOSSIER Phase 2 (`docs/DOSSIER-02-DESIGN.md`).

Phase 1 made the drawing knowable; this module is how the tools *say* it.
Nothing else in the codebase queries the Dossier collection, so the storage
shape stays changeable: a key that moves inside the document is a change to
this file and to nothing else.

Four properties are structural here, and every one of them exists because it
was got wrong somewhere before.

1.  **`None` and "empty" are DIFFERENT ANSWERS and never collapse.**
    This campaign started because `land_use_summary` answered *"0 road
    parcels"* while 1,233 road entities sat in `unclassified_layers` -- a
    number returned for something that had simply never been looked at. The
    rule, applied at every function below:

        Dossier missing  -> None                 "not computed"
        Dossier present, nothing matched -> a well-formed block with an
                                            empty list, and a verdict that
                                            says what WAS checked

    `duplicate_warning` is the one place where a naive reading of the
    signature would re-create the defect ("no warning" and "never looked"
    would both be `None`), so it returns a block carrying `warn` and
    `checked` for every case except a missing Dossier. See its docstring.

2.  **No map from a land use to a geometric role** (rule G1). `residual_for`
    is handed its `tokens` by the caller -- they come from
    `landuse.patterns().tokens_for(use)`, which is reviewable config that
    grows without code -- and it decides whether a NAME speaks for a use with
    `evidence.speaks_for`, which is already implemented, already tested, and
    already refuses short and code-shaped tokens. There is no `road ->
    network` line anywhere in this file, and there must never be one: it
    would need extending for every new use on every new client's drawing.

    The name proposes; the geometry disposes. Each match arrives carrying its
    Dossier profile, which is what lets a reader tell a real road network
    (`network`, 1,233 entities, 83,962.565 m) from a Civil3D sheet-layout
    layer that merely has "ROAD" in its name (`annotation`/`mixed`, a handful
    of entities).

3.  **Every measure republished here carries its unit, and the unit may be
    absent** (rule G2). Units are copied verbatim from the Dossier bucket,
    never derived and never defaulted: 11 of 18 drawings are in inches and 3
    declare nothing. Measures are also **never summed across layouts** -- the
    reference road layer is 83,962.565 **m** in `Model` and 27,987.522 in a
    block definition that declares no unit at all, and adding those two would
    publish a number in no unit whatsoever.

4.  **Every list states its cap and what it dropped** (rule G7). `summary`
    goes into every `describe_drawing` call and Janadriyah holds 176 buckets;
    pasting all of them would flood the agent's context and make the tool
    worse, not better. The caps are the module constants below, they are
    named in the responses, and what falls outside them is counted.

Reading the store: every query goes through `coll(COLL_DOSSIERS)` from
`app.mongo` -- the single gate -- and every query filters on `_id`, which is
the primary key. The cluster runs `notablescan`, so an unindexed query FAILS
rather than merely being slow; there is no query in this file that could.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

from .evidence import speaks_for
from .mongo import COLL_DOSSIERS, coll

log = logging.getLogger(__name__)

# --- Stated caps (G7) ---------------------------------------------------------

#: Layers named in a `summary` before truncation begins. This is the budget
#: that matters most: `summary` rides on every `describe_drawing`, and the
#: reference drawing has 176 buckets over 82 layers.
SUMMARY_TOP_LAYERS: int = 15

#: The most a caller may raise `top` to. A caller that asks for more gets this,
#: and the response says both numbers, so a truncation is never a surprise.
MAX_SUMMARY_TOP: int = 60

#: Layouts named in a `summary`. Block definitions each count as a layout, so
#: this list is long on a drawing full of blocks.
MAX_SUMMARY_LAYOUTS: int = 12

#: Layer names `profiles_for` will look up in one call. Names beyond it are
#: NOT dropped silently -- they come back with `profiled: false` and a reason,
#: because "we did not look" must never wear the same clothes as "there is no
#: entry for it".
MAX_PROFILE_LAYERS: int = 40

#: Rows published per geometric role in the name-blind list, so one
#: crowded role cannot fill the whole budget and hide the others.
MAX_ROWS_PER_ROLE: int = 3

#: Residual matches published per use. Ranked by entity count descending, so
#: the layer that actually holds the geometry is first.
MAX_RESIDUAL_MATCHES: int = 10

#: Declared-but-empty layer names published beside the residual matches.
MAX_EMPTY_LAYER_MATCHES: int = 10

#: Anomaly flags published per layer profile.
MAX_ANOMALIES_PER_LAYER: int = 5

#: Duplicate groups detailed by `duplicate_warning`.
MAX_DUPLICATE_GROUPS: int = 3

#: Other layouts named beside the bucket a profile chose.
MAX_OTHER_LAYOUTS: int = 5

#: Search tokens echoed back in a residual block.
MAX_TOKENS_ECHOED: int = 20

#: Characters of republished prose (a `caveat`, a `role_basis`, a `verdict`).
#: The reference road layer's caveat alone is ~2,400 characters; fifteen of
#: those in one `describe_drawing` is 36 KB of context spent on footnotes.
#: What is cut is marked, and the full text is in the Dossier.
MAX_TEXT: int = 320

#: The same budget inside a `summary` row, which is tighter because there are
#: `SUMMARY_TOP_LAYERS` of them in a response that rides on every
#: `describe_drawing` call. The standing prose that would otherwise repeat on
#: every row -- what a role means, how a bucket was chosen, where the anomaly
#: evidence lives -- is stated ONCE at the top of the summary instead.
SUMMARY_TEXT: int = 160

#: Fields this module never republishes, dropped at the database rather than
#: after: lane 2's full `detail` block, lane 3's cluster evidence, lane 1's
#: type-truncation bookkeeping and lane 4's census. Halves the reference
#: document (789 KB -> 353 KB) and the largest one in the store (1.38 MB ->
#: 627 KB). `duplicate_warning` does NOT use it -- it needs the evidence.
_LIGHT_PROJECTION: Mapping[str, int] = {
    "layers.measure.detail": 0,
    "layers.anomalies.evidence": 0,
    "layers.types_truncation": 0,
    "layers.census": 0,
}

_ELLIPSIS = "…"

_NO_UNIT_PHRASE = "drawing units (no unit is declared for this bucket)"

#: Published in responses, so it says nothing about any particular drawing
#: (G10). The measured example that made the rule -- one layer reading
#: 83,962.565 m in model space and 27,987.522 with no unit at all inside a
#: block definition -- is in this module's docstring, where it stays with the
#: reasoning instead of travelling into an answer about a different file.
_NEVER_SUM_ACROSS_LAYOUTS = (
    "Measures are never summed across layouts: a layout can carry a different "
    "unit, or none at all, and one layer's geometry inside a block definition "
    "is scaled again at every insertion. A sum across them would be a number "
    "in no unit whatsoever, wearing the same clothes as one that had a unit "
    "(G2)."
)

_NAME_IS_NOT_A_ROLE = (
    "`role` and every measure beside it are decided from geometry -- DXF type "
    "and ring status -- never from the layer name (G1). A layer named after a "
    "road that holds closed rings profiles as `region`, and that disagreement "
    "is a finding rather than a bug."
)


class DossierReadError(TypeError):
    """A caller handed this module something it cannot read."""


# --- The single seam to the store ---------------------------------------------


def _fetch(
    drawing_id: Any, *, projection: Mapping[str, int] | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """`(document, why_it_is_absent)` -- the ONLY place this file reads Mongo.

    Every other function reaches the store through here, which is also what
    makes the whole module testable with fixtures: a test replaces this one
    function and no database is involved.

    The filter is `{"_id": drawing_id}`. `_id` is the primary key, so the
    query is always indexed -- necessary and not merely nice, because the
    cluster runs `notablescan` and an unindexed query fails outright.

    A store that cannot be reached and a Dossier that was never built are
    DIFFERENT reasons and both are returned in words. The callers below turn
    both into `None`, as the contract requires, but `not_computed_block()`
    can still say which one happened, so an agent is never told "not computed"
    when the truth is "the database is down".
    """
    if not isinstance(drawing_id, str) or not drawing_id.strip():
        return None, (
            "no drawing id was given, so no Dossier could be looked up. A "
            "Dossier is keyed by drawing_id (a content hash)."
        )
    try:
        document = coll(COLL_DOSSIERS).find_one({"_id": drawing_id}, projection)
    except Exception as exc:  # a store failure must not take a tool down
        log.warning(
            "dossier lookup failed for %s: %s", drawing_id, type(exc).__name__
        )
        return None, (
            f"the Dossier store could not be read ({type(exc).__name__}). "
            "This is not the same as 'no Dossier has been built': nothing was "
            "established either way."
        )
    if document is None:
        return None, (
            "no Dossier has been built for this drawing yet, so what the file "
            "contains has not been computed. This is NOT a statement that the "
            "drawing is empty."
        )
    return dict(document), None


def _resolve(
    drawing_id: Any,
    dossier: Mapping[str, Any] | None,
    *,
    light: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """The Dossier for this call: the one handed in, or a fresh read.

    `dossier` exists so a caller that publishes several blocks from one
    document -- `land_use_summary` asks for a residual per use that reports
    zero -- reads the store once instead of once per block.
    """
    if dossier is None:
        return _fetch(
            drawing_id, projection=dict(_LIGHT_PROJECTION) if light else None
        )
    if not isinstance(dossier, Mapping):
        raise DossierReadError(
            f"`dossier` must be the stored document (a mapping), not "
            f"{type(dossier).__name__}. Pass the result of dossier_for(), or "
            "pass nothing and let this module read the store."
        )
    return dict(dossier), None


# --- Small helpers ------------------------------------------------------------


def _clip(text: Any, *, limit: int = MAX_TEXT) -> tuple[str | None, bool]:
    """`(text, was_truncated)`. `None` stays `None` -- absence is not an
    empty string, and an empty string is not a sentence."""
    if text is None:
        return None, False
    value = str(text)
    if not value.strip():
        return None, False
    if len(value) <= limit:
        return value, False
    return value[: limit - 1].rstrip() + _ELLIPSIS, True


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _entities(block: Mapping[str, Any]) -> int:
    """A bucket's entity count, defaulting to 0 only for a row that carries
    none. A bucket with no count cannot be ranked, and ranking it last is
    honest; it is still listed."""
    count = block.get("entities")
    if isinstance(count, bool) or not isinstance(count, (int, float)):
        return 0
    return int(count)


def _layer_blocks(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    blocks = document.get("layers")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        return []
    return [b for b in blocks if isinstance(b, Mapping)]


def _in_scope(
    block: Mapping[str, Any], layout: str | None
) -> bool:
    """`layout=None` means every layout, including block definitions.

    A layout is matched by its exact stored name. Block definitions are stored
    as `[block] <name>`, and they are in scope on purpose when no layout was
    asked for: the reference drawing's road geometry lives in one, and hiding
    it would be the coverage hole this campaign exists to close.
    """
    if layout is None:
        return True
    return str(block.get("layout")) == str(layout)


def _scoped(
    document: Mapping[str, Any], layout: str | None
) -> list[Mapping[str, Any]]:
    return [b for b in _layer_blocks(document) if _in_scope(b, layout)]


def _by_layer(
    blocks: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Buckets grouped by layer name, each group largest bucket first."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for block in blocks:
        grouped.setdefault(str(block.get("layer")), []).append(block)
    for group in grouped.values():
        group.sort(key=lambda b: (-_entities(b), str(b.get("layout"))))
    return grouped


def _scope_block(layout: str | None) -> dict[str, Any]:
    return {
        "layout": layout,
        "basis": (
            f"only layout {layout!r} is included"
            if layout is not None
            else (
                "every layout in the file is included, block definitions among "
                "them -- geometry inside a block is still geometry in the file"
            )
        ),
    }


# --- Views over one bucket ----------------------------------------------------


def _measure_view(
    block: Mapping[str, Any], *, verbose: bool
) -> dict[str, Any]:
    """The native measure, copied verbatim, with its prose budgeted.

    `unit` and `unit_reason` always travel together and are never invented
    here: whatever the Dossier established for this bucket is what is
    published, and a bucket with no unit publishes `null` plus the reason
    (G2). `value` may be `null`, and `null` is not `0` -- a bucket where
    nothing was measurable says so rather than reporting a zero that was never
    measured (G8).
    """
    measure = block.get("measure")
    measure = measure if isinstance(measure, Mapping) else {}
    limit = MAX_TEXT if verbose else SUMMARY_TEXT
    caveat, caveat_truncated = _clip(measure.get("caveat"), limit=limit)

    view: dict[str, Any] = {
        "kind": measure.get("kind"),
        "value": measure.get("value"),
        "unit": measure.get("unit"),
        "unit_reason": measure.get("unit_reason"),
        "measured_entities": measure.get("measured_entities"),
        "unmeasured_entities": measure.get("unmeasured_entities"),
        "bucket_entities": measure.get("bucket_entities"),
        "value_is_floor": measure.get("value_is_floor"),
        "caveat": caveat,
        "caveat_truncated": caveat_truncated,
    }
    if "chains" in measure:
        view["chains"] = measure.get("chains")
        view["snap_tolerance"] = measure.get("snap_tolerance")
    if measure.get("value") is None:
        view["value_is_absent_not_zero"] = (
            "absence, not zero (G8): the entity count beside it is exact"
        )
    if verbose:
        basis, basis_truncated = _clip(measure.get("basis"))
        view["basis"] = basis
        view["basis_truncated"] = basis_truncated
    return view


def _anomaly_view(block: Mapping[str, Any], *, verbose: bool) -> dict[str, Any]:
    """Anomaly flags for one bucket, capped, with the uncomputed case visible.

    `flags: []` on its own would read as "this layer is unremarkable", which
    is a claim nobody made when lane 3 never ran. `status.computed` carries
    that difference through, exactly as the Dossier stores it.
    """
    raw = block.get("anomalies")
    items = [a for a in raw if isinstance(a, Mapping)] if isinstance(raw, list) else []
    kept = items[:MAX_ANOMALIES_PER_LAYER]

    flags: list[dict[str, Any]] = []
    for item in kept:
        detail, truncated = _clip(
            item.get("detail"), limit=MAX_TEXT if verbose else SUMMARY_TEXT
        )
        flags.append(
            {
                "kind": item.get("kind"),
                "detail": detail,
                "detail_truncated": truncated,
                "counts": item.get("counts"),
            }
        )

    status = block.get("anomalies_status")
    if isinstance(status, Mapping):
        status_view = {
            "computed": status.get("computed"),
            "why": status.get("why"),
        }
    else:
        status_view = {
            "computed": None,
            "why": (
                "this Dossier records no anomalies_status, so whether the "
                "layer was examined for anomalies is unknown"
            ),
        }

    view: dict[str, Any] = {
        "flags": flags,
        "kinds": sorted({str(f["kind"]) for f in flags if f.get("kind")}),
        "total": len(items),
        "reported": len(flags),
        "dropped": max(0, len(items) - len(flags)),
        "cap": MAX_ANOMALIES_PER_LAYER,
        "status": status_view,
    }
    if verbose:
        view["evidence_note"] = (
            "The evidence behind each flag -- thresholds, cluster boxes, "
            "example handles -- stays in the Dossier and is not republished "
            "here (G7). `duplicate_warning` publishes it for the one flag a "
            "measured total must not be quoted without."
        )
    return view


def _role_leader(block: Mapping[str, Any]) -> dict[str, Any] | None:
    """The winning type family and its share, computed from `role_shares`.

    Cheaper than republishing `role_basis` in a summary row and it carries the
    same signal: a 61% call and a 99% call produce the same word and are not
    the same call.
    """
    shares = block.get("role_shares")
    if not isinstance(shares, Mapping) or not shares:
        return None
    ordered = sorted(
        ((str(k), v) for k, v in shares.items() if isinstance(v, (int, float))),
        key=lambda kv: (-float(kv[1]), kv[0]),
    )
    if not ordered:
        return None
    family, share = ordered[0]
    runner = ordered[1] if len(ordered) > 1 else None
    return {
        "family": family,
        "share": round(float(share), 6),
        "runner_up": runner[0] if runner else None,
        "runner_up_share": round(float(runner[1]), 6) if runner else None,
        "threshold": block.get("role_threshold"),
    }


def _profile(
    layer: str,
    blocks: Sequence[Mapping[str, Any]],
    *,
    layout: str | None,
    verbose: bool,
) -> dict[str, Any]:
    """One layer's profile: role, entity count, native measure, anomaly flags.

    When a layer holds buckets in several layouts, the profile describes the
    LARGEST bucket and names the others beside it. It does not merge them, and
    the reason is in `_NEVER_SUM_ACROSS_LAYOUTS`.

    `verbose=False` is the `summary` row: the same FACTS, without the standing
    prose that would otherwise be repeated on all fifteen of them. That prose
    is stated once at the top of the summary instead, which is a size budget
    honoured rather than an explanation dropped (G7).
    """
    chosen = blocks[0]
    others = blocks[1:]
    listed = others[:MAX_OTHER_LAYOUTS]

    scope: dict[str, Any] = {
        "layout_requested": layout,
        "buckets_for_this_layer": len(blocks),
        "entities_all_buckets": sum(_entities(b) for b in blocks),
        "other_layouts": [
            {
                "layout": b.get("layout"),
                "entities": _entities(b),
                "role": b.get("role"),
                "measure_kind": (b.get("measure") or {}).get("kind"),
                "measure_value": (b.get("measure") or {}).get("value"),
                "unit": (b.get("measure") or {}).get("unit"),
            }
            for b in listed
            if isinstance(b, Mapping)
        ],
        "other_layouts_total": len(others),
        "other_layouts_dropped": max(0, len(others) - len(listed)),
        "other_layouts_cap": MAX_OTHER_LAYOUTS,
    }
    if verbose:
        scope["chosen_basis"] = (
            "the bucket with the most entities. " + _NEVER_SUM_ACROSS_LAYOUTS
        )

    profile: dict[str, Any] = {
        "layer": layer,
        "layout": chosen.get("layout"),
        "entities": _entities(chosen),
        "role": chosen.get("role"),
        "role_leader": _role_leader(chosen),
        "role_threshold": chosen.get("role_threshold"),
        "measure": _measure_view(chosen, verbose=verbose),
        "anomalies": _anomaly_view(chosen, verbose=verbose),
        "scope": scope,
    }
    if verbose:
        profile["role_shares"] = chosen.get("role_shares")
        basis, truncated = _clip(chosen.get("role_basis"))
        profile["role_basis"] = basis
        profile["role_basis_truncated"] = truncated
        profile["role_note"] = _NAME_IS_NOT_A_ROLE
    return profile


# --- 1. The document ----------------------------------------------------------


def dossier_for(drawing_id: str) -> dict | None:
    """The stored Dossier, or `None` when it has not been built.

    `None` means **not computed**, and every caller must treat it that way. It
    never means "the drawing holds nothing": a drawing whose Dossier was never
    built and a drawing that genuinely has no layers are different answers,
    and this module keeps them apart at every level.
    """
    document, _why = _fetch(drawing_id)
    return document


def not_computed_block(
    drawing_id: str, *, what: str = "dossier"
) -> dict[str, Any]:
    """The block a caller publishes INSTEAD of silence when there is no Dossier.

    Not one of the five contract functions -- a convenience so that every tool
    says the same thing, in the same words, when the answer is "not computed".
    `describe_drawing` in particular must never fall silent here: an agent that
    receives nothing will fill the gap itself.
    """
    _document, why = _fetch(drawing_id, projection={"_id": 1})
    return {
        "available": False,
        "drawing_id": drawing_id,
        "what": what,
        "why": why
        or (
            "a Dossier is stored for this drawing but was not read for this "
            "response"
        ),
        "meaning": (
            "NOT COMPUTED -- not 'nothing found'. Nothing here may be read as "
            "a statement about what the drawing does or does not contain."
        ),
        "how_to_build": (
            "python scripts/dossier_backfill.py --drawing "
            f"{drawing_id} -- it reads the entities already in Mongo and "
            "writes one document to `autocad_dossiers`; no DXF is re-read."
        ),
    }


# --- 2. The size-budgeted overview -------------------------------------------


def summary(
    drawing_id: str,
    *,
    layout: str | None = None,
    top: int = SUMMARY_TOP_LAYERS,
    dossier: Mapping[str, Any] | None = None,
) -> dict | None:
    """Size-budgeted overview for `describe_drawing`.

    Roles with their bucket counts and totals, plus the top-N layers by entity
    count with role and native measure. It states what it truncated.

    Returns `None` when no Dossier has been built -- **not computed**, and the
    caller publishes `not_computed_block()` rather than nothing. A drawing
    whose Dossier exists but holds no layers comes back as a well-formed block
    with empty lists and a coverage verdict, which is a different answer.

    Role totals are grouped by `(measure kind, unit)` and summed only inside a
    group. A drawing with model-space metres and a unit-less block definition
    therefore reports two totals rather than one meaningless one (G2).
    """
    document, _why = _resolve(drawing_id, dossier)
    if document is None:
        return None

    cap = _int(top)
    cap = SUMMARY_TOP_LAYERS if cap is None else cap
    applied = max(0, min(cap, MAX_SUMMARY_TOP))

    scoped = _scoped(document, layout)
    grouped = _by_layer(scoped)
    ranked = sorted(
        grouped.items(),
        key=lambda kv: (-sum(_entities(b) for b in kv[1]), kv[0]),
    )
    kept = ranked[:applied]
    dropped = ranked[applied:]

    return {
        "available": True,
        "drawing_id": document.get("drawing_id") or document.get("_id"),
        "dossier_version": document.get("dossier_version"),
        "computed_at": document.get("computed_at"),
        "scope": _scope_block(layout),
        "entity_total": document.get("entity_total"),
        "entities_in_scope": sum(_entities(b) for b in scoped),
        "buckets_in_scope": len(scoped),
        "layers_in_scope": len(grouped),
        "coverage": _coverage_view(document),
        "units": _units_view(document),
        "roles": _roles_view(scoped),
        "layers": [
            _profile(name, blocks, layout=layout, verbose=False)
            for name, blocks in kept
        ],
        "truncation": {
            "cap_applied": applied,
            "cap_requested": cap,
            "cap_ceiling": MAX_SUMMARY_TOP,
            "layers_present": len(ranked),
            "layers_reported": len(kept),
            "layers_dropped": len(dropped),
            "entities_dropped": sum(
                sum(_entities(b) for b in blocks) for _name, blocks in dropped
            ),
            "basis": (
                "Layers are ranked by entity count over every bucket in scope "
                f"and capped at {applied} (G7). Nothing is dropped silently: "
                "the counts above say how much is not named here, and the "
                "role totals beside them are computed over ALL buckets in "
                "scope, not only the ones listed."
            ),
        },
        "anomalies": _anomaly_rollup(scoped),
        "layouts": _layouts_view(document),
        "how_to_read": (
            "This is what the file CONTAINS, computed once from the stored "
            "entities -- no scan is needed to answer 'what is in this "
            "drawing'. Each role answers in its own native measure: a "
            "`network` layer answers in LENGTH, not in a parcel count of "
            "zero, and a `region` layer answers in area. "
            + _NAME_IS_NOT_A_ROLE
        ),
    }


def _coverage_view(document: Mapping[str, Any]) -> dict[str, Any]:
    """The campaign's whole claim, compacted but never softened."""
    coverage = document.get("coverage")
    if not isinstance(coverage, Mapping):
        return {
            "complete": None,
            "why": (
                "this Dossier carries no coverage block, so whether every "
                "entity in the file is accounted for was never computed"
            ),
        }
    verdict, truncated = _clip(coverage.get("verdict"))
    return {
        "accounted": coverage.get("accounted"),
        "total": coverage.get("total"),
        "difference": coverage.get("difference"),
        "complete": coverage.get("complete"),
        "buckets": coverage.get("buckets"),
        "verdict": verdict,
        "verdict_truncated": truncated,
    }


def _units_view(document: Mapping[str, Any]) -> dict[str, Any]:
    """What the FILE declares, which is not the same as what a bucket carries.

    `$INSUNITS` describes model space only. It is published here as a file
    fact so a reader can see that a bucket's absent unit is the file's silence
    and not this module's, and it is never copied onto a bucket.
    """
    facts = document.get("file_facts")
    units = (facts or {}).get("units") if isinstance(facts, Mapping) else None
    if not isinstance(units, Mapping):
        return {
            "declared_in_file": None,
            "why": (
                "this Dossier records no unit facts, so whether the file "
                "declares a unit is unknown. No unit may be assumed from it."
            ),
        }
    return {
        "units_name": units.get("units_name"),
        "units_code": units.get("units_code"),
        "declared_in_file": units.get("declared_in_file"),
        "basis": (
            "A drawing-header field describing MODEL space. Each bucket's "
            "measure carries its own unit, which may be absent; this is never "
            "substituted for it (G2)."
        ),
    }


def _roles_view(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per role: buckets, entities, and totals grouped by `(kind, unit)`.

    Two buckets are added together only when they measure the same KIND in the
    same UNIT. A bucket with no unit is its own group and is never folded into
    a unit that was declared somewhere else, so a metre total never absorbs an
    inch one and never absorbs a number in no unit at all (G2). A bucket whose
    value is absent is counted, never summed as zero (G8).
    """
    per_role: dict[Any, dict[str, Any]] = {}
    for block in blocks:
        role = block.get("role")
        row = per_role.setdefault(
            role,
            {
                "role": role,
                "buckets": 0,
                "entities": 0,
                "_groups": {},
                "buckets_without_value": 0,
            },
        )
        row["buckets"] += 1
        row["entities"] += _entities(block)

        measure = block.get("measure")
        measure = measure if isinstance(measure, Mapping) else {}
        value = measure.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            row["buckets_without_value"] += 1
            continue
        key = (measure.get("kind"), measure.get("unit"))
        group = row["_groups"].setdefault(
            key,
            {
                "kind": measure.get("kind"),
                "unit": measure.get("unit"),
                "unit_reason": measure.get("unit_reason"),
                "value": 0.0,
                "buckets": 0,
            },
        )
        group["value"] += float(value)
        group["buckets"] += 1

    rows: list[dict[str, Any]] = []
    for row in per_role.values():
        groups = list(row.pop("_groups").values())
        for group in groups:
            group["value"] = round(group["value"], 6)
        groups.sort(key=lambda g: (str(g["kind"]), str(g["unit"])))
        row["totals"] = groups
        row["totals_basis"] = (
            "Summed only within one (measure kind, unit) group. A bucket "
            "whose unit is absent forms its own group and is never folded "
            "into a declared unit (G2); a bucket whose value is absent is "
            "counted in `buckets_without_value`, never summed as zero (G8)."
        )
        rows.append(row)

    rows.sort(key=lambda r: (-int(r["entities"]), str(r["role"])))
    return rows


def _anomaly_rollup(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How many buckets carry a flag, how many were never examined."""
    kinds: dict[str, int] = {}
    flagged = 0
    unchecked = 0
    for block in blocks:
        raw = block.get("anomalies")
        items = [a for a in raw if isinstance(a, Mapping)] if isinstance(raw, list) else []
        if items:
            flagged += 1
        for item in items:
            kind = str(item.get("kind"))
            kinds[kind] = kinds.get(kind, 0) + 1
        status = block.get("anomalies_status")
        if not isinstance(status, Mapping) or status.get("computed") is not True:
            unchecked += 1
    return {
        "buckets_flagged": flagged,
        "kinds": dict(sorted(kinds.items())),
        "buckets_not_examined": unchecked,
        "basis": (
            "`buckets_not_examined` counts buckets whose anomaly check did "
            "not run. Those are not clean bills of health, and a zero here is "
            "the only thing that makes the rest of this block a measurement."
        ),
    }


def _layouts_view(document: Mapping[str, Any]) -> dict[str, Any]:
    layouts = document.get("layouts")
    rows = [x for x in layouts if isinstance(x, Mapping)] if isinstance(layouts, list) else []
    ordered = sorted(
        rows,
        key=lambda r: (-(_int(r.get("profiled_entities")) or 0), str(r.get("name"))),
    )
    kept = ordered[:MAX_SUMMARY_LAYOUTS]
    return {
        "items": [
            {
                "name": r.get("name"),
                "profiled_entities": r.get("profiled_entities"),
                "declared_entities": r.get("declared_entities"),
                "agrees": r.get("agrees"),
                "is_modelspace": r.get("is_modelspace"),
                "is_block": r.get("is_block"),
            }
            for r in kept
        ],
        "total": len(rows),
        "reported": len(kept),
        "dropped": max(0, len(rows) - len(kept)),
        "cap": MAX_SUMMARY_LAYOUTS,
    }


# --- 3. Profiles for named layers ---------------------------------------------


def profiles_for(
    drawing_id: str,
    layer_names,
    *,
    layout: str | None = None,
    dossier: Mapping[str, Any] | None = None,
) -> dict[str, dict]:
    """Role + native measure + anomaly flags, per named layer.

    For the `unclassified_layers` block. **A layer with no Dossier entry is
    ABSENT from the result** rather than present with nulls -- a row of nulls
    would read as "we looked and there is nothing", and there is a difference.

    An absent Dossier gives `{}` for the same reason it gives `None` elsewhere:
    there is nothing to report. The caller has already been told by
    `summary()`/`dossier_for()` that nothing is computed, and it publishes
    `not_computed_block()`; a `{}` here must never be relayed as "none of
    these layers exist".

    Names beyond `MAX_PROFILE_LAYERS` come back with `profiled: false` and a
    reason. They are NOT omitted: omission means "no entry for it", and "we
    stopped looking" is a different sentence (G7).
    """
    document, _why = _resolve(drawing_id, dossier)
    if document is None:
        return {}

    if isinstance(layer_names, str):
        requested: list[str] = [layer_names]
    elif isinstance(layer_names, Iterable):
        requested = [str(n) for n in layer_names]
    else:
        raise DossierReadError(
            f"`layer_names` must be a sequence of layer names, not "
            f"{type(layer_names).__name__}."
        )

    seen: list[str] = []
    for name in requested:
        if name not in seen:
            seen.append(name)

    grouped = _by_layer(_scoped(document, layout))
    out: dict[str, dict] = {}
    for index, name in enumerate(seen):
        if index >= MAX_PROFILE_LAYERS:
            out[name] = {
                "layer": name,
                "profiled": False,
                "why": (
                    f"this call named {len(seen)} layers and stops at "
                    f"{MAX_PROFILE_LAYERS} (G7). This layer was NOT examined; "
                    "that is not a statement about what it holds. Ask for it "
                    "in a smaller call."
                ),
            }
            continue
        blocks = grouped.get(name)
        if not blocks:
            # Absent, by contract. No entry, no nulls, no invented row.
            continue
        profile = _profile(name, blocks, layout=layout, verbose=True)
        profile["profiled"] = True
        out[name] = profile
    return out


# --- 4. The residual rule -----------------------------------------------------


def residual_for(
    drawing_id: str,
    use,
    tokens,
    classified_layers,
    *,
    layout: str | None = None,
    dossier: Mapping[str, Any] | None = None,
) -> dict | None:
    """Unclassified layers whose name speaks for `use`, with their profiles.

    The rule this implements: *a use that reports 0 while unclassified layers
    hold entities must say so, and must publish what those layers ARE.*

    Returns `None` when the Dossier is missing (**unknown**), and a block with
    an empty list when the Dossier exists and genuinely nothing matches
    (**known**). Those are different answers and they must not collapse -- the
    zero this campaign exists to fix was a "known" printed where an "unknown"
    belonged.

    There is deliberately **no map from a use to a geometric role**. `tokens`
    are handed in by the caller from `landuse.patterns().tokens_for(use)` --
    reviewable config that grows without code -- and a name speaks for the use
    only through `evidence.speaks_for`, which tests Latin tokens on word
    boundaries and already refuses tokens that are short or code-shaped. A
    `road -> network` table here would be a hardcoded ontology and would need
    extending for every new use on every new client's drawing (G1).

    What comes back is not merely a list of names: each match carries its
    Dossier profile, so `00_Prop - Road - CL_` (network, 1,233 entities,
    83,962.565 m) and a Civil3D sheet-layout layer with "ROAD" in its name are
    visibly different rows in the same list. The name proposes; the geometry
    disposes.

    `tokens` empty is a THIRD state and is reported as one: nothing could be
    searched for, so `searchable` and `checked` are both false and the empty
    list is not a checked zero.
    """
    document, _why = _resolve(drawing_id, dossier)
    if document is None:
        return None

    token_list: list[str] = []
    if isinstance(tokens, str):
        token_list = [tokens]
    elif isinstance(tokens, Iterable):
        for token in tokens:
            text = str(token).strip()
            if text and text not in token_list:
                token_list.append(text)

    classified: set[str] = set()
    if isinstance(classified_layers, str):
        classified = {classified_layers.casefold()}
    elif isinstance(classified_layers, Iterable):
        classified = {str(n).casefold() for n in classified_layers}

    scoped = _scoped(document, layout)
    unclassified = [
        b for b in scoped if str(b.get("layer")).casefold() not in classified
    ]
    grouped = _by_layer(unclassified)

    matches: list[tuple[int, str, str, list[Mapping[str, Any]]]] = []
    if token_list:
        for name, blocks in grouped.items():
            token = speaks_for(name, token_list)
            if token is None:
                continue
            matches.append(
                (sum(_entities(b) for b in blocks), name, token, blocks)
            )
    matches.sort(key=lambda m: (-m[0], m[1]))
    matched_names = {m[1] for m in matches}
    kept = matches[:MAX_RESIDUAL_MATCHES]
    dropped = matches[MAX_RESIDUAL_MATCHES:]

    rows: list[dict[str, Any]] = []
    for entities, name, token, blocks in kept:
        profile = _profile(name, blocks, layout=layout, verbose=True)
        profile["matched_token"] = token
        profile["match_basis"] = (
            f"the layer NAME contains the word {token!r}, one of the "
            f"{len(token_list)} tokens config lists for {use!r} "
            "(evidence.speaks_for: Latin tokens on word boundaries, non-Latin "
            "by containment). The name is why this row is here; the role and "
            "measure beside it are what it actually IS."
        )
        profile["entities_in_scope"] = entities
        rows.append(profile)

    empty_matches, empty_complete, empty_total = _empty_layer_matches(
        document, token_list, classified
    )

    block: dict[str, Any] = {
        "use": use,
        "checked": bool(token_list),
        "searchable": bool(token_list),
        "scope": _scope_block(layout),
        "tokens": token_list[:MAX_TOKENS_ECHOED],
        "tokens_total": len(token_list),
        "tokens_truncated": len(token_list) > MAX_TOKENS_ECHOED,
        "tokens_basis": (
            "Supplied by the caller from the `vocabulary` block of "
            "landuse/_patterns.yaml (pattern_set.tokens_for). This module "
            "holds no map from a land use to a geometric role: such a map "
            "would be a hardcoded ontology needing an entry for every new use "
            "on every new drawing (G1)."
        ),
        "unclassified_layers_examined": len(grouped),
        "unclassified_buckets_examined": len(unclassified),
        "classified_layers_excluded": len(classified),
        "matches": rows,
        "matches_total": len(matches),
        "entities_in_matches": sum(m[0] for m in matches),
        # The name-blind half, and it exists because the name-matched half is
        # not enough. Asked "how many roads", this drawing's residual returns
        # three layers whose names carry the word -- and NOT `ROW`, which holds
        # 91 right-of-way corridors covering 1,483,193.7 m2 and is the single
        # most defensible answer to the question. "ROW" contains none of the
        # nine road tokens, so a name-gated search cannot see it, however good
        # the vocabulary gets.
        #
        # This campaign's whole thesis is that the name proposes and the
        # geometry disposes. Until now the name was still the gatekeeper for
        # what got SHOWN. So the biggest unclassified buckets are published
        # too, ranked by what they hold rather than by what they are called,
        # and labelled as not-name-matched so no one mistakes proximity in a
        # list for a claim about meaning.
        "largest_unclassified": _largest_unclassified(grouped, matched_names),
        "largest_unclassified_basis": (
            "The largest unclassified layers in scope by entity count, "
            "WHATEVER they are called, excluding the name matches above. No "
            "claim is made that any of them answers the question -- a name "
            "search cannot see a layer whose name says nothing, so what it "
            "holds is offered instead and the reader decides. Ranked by "
            f"entities, capped at {MAX_RESIDUAL_MATCHES} (G7)."
        ),
        "truncation": {
            "cap": MAX_RESIDUAL_MATCHES,
            "reported": len(kept),
            "dropped": len(dropped),
            "entities_dropped": sum(m[0] for m in dropped),
            "names_dropped": [m[1] for m in dropped[:MAX_RESIDUAL_MATCHES]],
            "basis": (
                "Matches are ranked by entity count descending and capped at "
                f"{MAX_RESIDUAL_MATCHES} (G7), so the layer that actually "
                "holds the geometry is first and what was cut is counted."
            ),
        },
        "declared_but_empty_matches": empty_matches,
        "declared_but_empty_search_complete": empty_complete,
        "declared_but_empty_basis": (
            "Layers the file declares that hold no entity in any layout. They "
            "have no geometry to disclose, so they are named only. The "
            f"Dossier caps that list upstream ({empty_total} such layers in "
            "this drawing), so when `declared_but_empty_search_complete` is "
            "false, finding none here does NOT mean there are none."
        ),
        "caveat": (
            "This search reads layer NAMES. A layer whose name says nothing "
            "about this use can still hold it, and no absence here proves the "
            "use is missing from the drawing (G10: meaning is read per "
            "drawing, never carried in from another)."
        ),
        "how_to_read": _NAME_IS_NOT_A_ROLE,
    }
    block["verdict"] = _residual_verdict(block, use, rows)
    return block


def _empty_layer_matches(
    document: Mapping[str, Any],
    tokens: Sequence[str],
    classified: set[str],
) -> tuple[list[str], bool, int]:
    """Declared-but-empty layer names that speak for the use.

    The Dossier stores this list already truncated (25 of 210 on the reference
    drawing), so the search over it can be incomplete -- and it says so. A
    partial search that reported "none found" as a finding would be the same
    defect this whole campaign is about, one level down.
    """
    facts = document.get("file_facts")
    layers = (facts or {}).get("layers") if isinstance(facts, Mapping) else None
    if not isinstance(layers, Mapping):
        return [], False, 0

    names = layers.get("declared_but_unprofiled_items")
    names = [str(n) for n in names] if isinstance(names, list) else []
    truncation = layers.get("declared_but_unprofiled_truncation")
    dropped = (
        _int((truncation or {}).get("dropped"))
        if isinstance(truncation, Mapping)
        else None
    )
    total = _int(layers.get("declared_but_unprofiled")) or len(names)
    complete = dropped == 0

    if not tokens:
        return [], complete, total

    hits = [
        name
        for name in names
        if name.casefold() not in classified and speaks_for(name, tokens)
    ]
    return hits[:MAX_EMPTY_LAYER_MATCHES], complete, total


def _residual_verdict(
    block: Mapping[str, Any], use: Any, rows: Sequence[Mapping[str, Any]]
) -> str:
    """The sentence an agent can relay. Generated, never handed in."""
    if not block["searchable"]:
        return (
            f"NOT CHECKED for {use!r}: config lists no vocabulary token for "
            "this use, so no layer name could speak for it and no search was "
            "made. This is not a zero -- it is an unasked question. Add "
            f"tokens for {use!r} to the `vocabulary` block of "
            "landuse/_patterns.yaml to make it answerable."
        )
    if rows:
        first = rows[0]
        measure = first.get("measure") or {}
        value = measure.get("value")
        unit = measure.get("unit")
        if value is None:
            figure = "no measure was established for it"
        else:
            figure = f"{value} {unit}" if unit else f"{value} {_NO_UNIT_PHRASE}"
        return (
            f"A zero for {use!r} is NOT an absence here. "
            f"{block['matches_total']} unclassified layer(s) hold "
            f"{block['entities_in_matches']} entities and carry that word in "
            f"their name. The largest is {first.get('layer')!r} in layout "
            f"{first.get('layout')!r}: role {first.get('role')!r}, "
            f"{first.get('entities')} entities, {figure}. Relay this together "
            "with the zero -- a count of parcels is the wrong measure for a "
            "network, and dropping the residual reproduces the answer this "
            "rule exists to prevent."
        )
    return (
        f"CHECKED and none found: {block['unclassified_layers_examined']} "
        f"unclassified layer(s) in scope, and not one of their names carries "
        f"any of the {block['tokens_total']} token(s) config lists for "
        f"{use!r}. This zero was established, not assumed -- but it is a "
        "statement about NAMES: a layer whose name says nothing about this "
        "use can still hold it."
    )


# --- 5. The duplicate warning -------------------------------------------------


def _largest_unclassified(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    matched_names: set[str],
) -> list[dict[str, Any]]:
    """The biggest unclassified layers in scope, ranked by what they hold.

    Deliberately name-blind. A layer called `ROW` says nothing to a search for
    the word "road" and holds the right-of-way corridors anyway; a search that
    can only follow names will never reach it, and no amount of vocabulary
    fixes that, because the next drawing will call it something else again.

    Nothing here is a claim that any row answers the question. Each carries
    its role and its native measure, and the reader decides -- which is the
    same division of labour the whole Dossier rests on.
    """
    # Ranked WITHIN each role by that role's own measure, and the reason is a
    # mistake this list made on its first run: ranked by raw entity count it
    # returned `DIM` -- 9,738 dimension marks -- ahead of `ROW`, which holds 91
    # right-of-way corridors covering 1,483,193.7 m2. Counting entities makes a
    # layer of tick marks outrank the road corridors, because a count is not a
    # size.
    #
    # Areas compare with areas and lengths with lengths; comparing an area to a
    # length is the cross-kind arithmetic rule G2 exists to forbid. So each role
    # is ranked in its own terms and the top of each is published, which puts
    # the largest region and the longest network both in front of the reader
    # without either being ranked against the other.
    per_role: dict[str, list[tuple[float, int, str, Sequence[Mapping[str, Any]]]]] = {}
    for name, blocks in grouped.items():
        if name in matched_names:
            continue
        entities = sum(_entities(b) for b in blocks)
        if not entities:
            continue
        profile = _profile(name, blocks, layout=None, verbose=False)
        role = str(profile.get("role") or "unknown")
        measure = profile.get("measure")
        value = 0.0
        if isinstance(measure, Mapping):
            try:
                value = float(measure.get("value") or 0.0)
            except (TypeError, ValueError):
                value = 0.0
        per_role.setdefault(role, []).append((value, entities, name, blocks))

    # Interleaved across roles, one from each before any role gets a second.
    # Grouping them in blocks looked tidier and failed for a mundane reason:
    # a caller further down publishes only the first five rows, and five rows
    # of role-blocks were three `mixed` and two `network` -- the region layer
    # holding 1.48 million m2 of right-of-way sat at position eight and never
    # reached the reader. Interleaving makes any downstream truncation keep
    # the roles diverse instead of keeping one role complete.
    #
    # Annotation goes last within each round: labels describe a drawing, they
    # are not the thing being asked about, and 9,738 dimension marks at the
    # top of a list bury whatever the reader came for.
    order = sorted(per_role, key=lambda r: (r == "annotation", r))
    for role in order:
        per_role[role].sort(key=lambda row: (-row[0], -row[1], row[2]))

    out: list[dict[str, Any]] = []
    for rank in range(MAX_ROWS_PER_ROLE):
        for role in order:
            rows_for_role = per_role[role]
            if rank >= len(rows_for_role):
                continue
            value, entities, name, blocks = rows_for_role[rank]
            profile = _profile(name, blocks, layout=None, verbose=False)
            profile["entities_in_scope"] = entities
            profile["name_matched"] = False
            profile["ranked_within_role_by"] = (
                "the native measure of this role, compared only against other "
                "layers of the same role; roles are interleaved so a truncated "
                "list still shows one of each"
            )
            out.append(profile)
            if len(out) >= MAX_RESIDUAL_MATCHES:
                return out
    return out


def duplicate_warning(
    drawing_id: str,
    *,
    layer,
    layout,
    dossier: Mapping[str, Any] | None = None,
) -> dict | None:
    """The anomaly a measured total must not be quoted without.

    Janadriyah's road layer is the case: 83,962.565 m is three copies of
    27,987.522 m, and a total quoted without that is three times the truth.

    **`None` means "no Dossier", and nothing else.** Returning `None` for "no
    duplicates" as well would collapse *not computed* into *nothing found* --
    the exact defect this campaign exists to end -- so every other outcome is
    a block carrying two booleans:

        warn=True,  checked=True   identical clusters were found; do not quote
                                   the total without this
        warn=False, checked=True   the layer was examined and holds none
        warn=False, checked=False  it was NOT examined, and `why` says so
                                   (no such bucket, or lane 3 never ran)

    A caller decides on `warn`; a caller that treats a falsy return as "safe"
    is wrong twice over, and the block is shaped so that mistake is visible.

    `layout` may be `None`, meaning every layout: a `measure` call that was
    not scoped to one gets warned about all of them.
    """
    document, _why = _resolve(drawing_id, dossier, light=False)
    if document is None:
        return None

    scoped = [
        b
        for b in _scoped(document, layout)
        if str(b.get("layer")) == str(layer)
    ]
    head: dict[str, Any] = {
        "drawing_id": document.get("drawing_id") or document.get("_id"),
        "layer": layer,
        "layout": layout,
    }

    if not scoped:
        return {
            **head,
            "warn": False,
            "checked": False,
            "why": (
                f"the Dossier holds no bucket for layer {layer!r} in layout "
                f"{layout!r}, so it was not examined for duplicates. That is "
                "not a clean bill of health."
            ),
            "meaning": (
                "NOT EXAMINED. `warn: false` here says nothing about the "
                "layer; it says the question was not asked."
            ),
        }

    groups: list[dict[str, Any]] = []
    examined = 0
    not_examined: list[str] = []
    for block in scoped:
        status = block.get("anomalies_status")
        computed = (
            status.get("computed") if isinstance(status, Mapping) else None
        )
        if computed is True:
            examined += 1
        else:
            why = (
                status.get("why") if isinstance(status, Mapping) else None
            ) or "no anomalies_status is recorded for this bucket"
            not_examined.append(f"{block.get('layout')!r}: {why}")
        groups.extend(_duplicate_groups(block))

    groups.sort(key=lambda g: -(g.get("entities_involved") or 0))
    kept = groups[:MAX_DUPLICATE_GROUPS]

    if not groups:
        return {
            **head,
            "warn": False,
            "checked": examined > 0,
            "buckets_examined": examined,
            "buckets_not_examined": not_examined,
            "why": (
                "no `duplicate_clusters` anomaly is recorded for this layer"
                if examined
                else (
                    "the anomaly check did not run on this layer, so nothing "
                    "was established: " + "; ".join(not_examined)
                )
            ),
            "meaning": (
                "Examined and clean." if examined else "NOT examined."
            )
            + (
                " The detector fails one-sidedly: if a layer's widest single "
                "feature exceeds the void between its copies, the copies merge "
                "and nothing is reported. A missed finding, never an invented "
                "one."
            ),
        }

    return {
        **head,
        "warn": True,
        "checked": True,
        "buckets_examined": examined,
        "buckets_not_examined": not_examined,
        "groups": kept,
        "groups_total": len(groups),
        "groups_dropped": max(0, len(groups) - len(kept)),
        "groups_cap": MAX_DUPLICATE_GROUPS,
        "statement": _duplicate_statement(kept[0]),
        "basis": (
            "Detected by partitioning the layer at voids wider than its own "
            "widest feature and comparing the resulting clusters by (entity "
            "count, bounding-box width, bounding-box height). The threshold is "
            "derived from this layer's own geometry, so it behaves identically "
            "on a drawing in inches. The full evidence -- threshold, the entity "
            "that set it, each cluster's box and example handles -- is in the "
            "Dossier."
        ),
        "meaning": (
            "Quote the per-copy figure, or quote the total together with this "
            "warning. A total published on its own overstates the layer by the "
            "number of copies."
        ),
    }


def _duplicate_groups(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Duplicate-cluster groups for one bucket, with the layer's own total."""
    raw = block.get("anomalies")
    items = [a for a in raw if isinstance(a, Mapping)] if isinstance(raw, list) else []
    measure = block.get("measure")
    measure = measure if isinstance(measure, Mapping) else {}

    out: list[dict[str, Any]] = []
    for item in items:
        if item.get("kind") != "duplicate_clusters":
            continue
        evidence = item.get("evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        unit = evidence.get("unit") or measure.get("unit")
        detail, detail_truncated = _clip(item.get("detail"))
        raw_groups = evidence.get("groups")
        raw_groups = raw_groups if isinstance(raw_groups, list) else []
        for group in raw_groups:
            if not isinstance(group, Mapping):
                continue
            per_copy = group.get("length_total_each")
            total = measure.get("value")
            ratio = None
            if (
                isinstance(per_copy, (int, float))
                and not isinstance(per_copy, bool)
                and per_copy
                and isinstance(total, (int, float))
                and not isinstance(total, bool)
            ):
                ratio = round(float(total) / float(per_copy), 3)
            out.append(
                {
                    "layout": block.get("layout"),
                    "copies": group.get("members"),
                    "entities_each": group.get("entities_each"),
                    "entities_involved": group.get("entities_involved"),
                    "bbox_width": group.get("bbox_width"),
                    "bbox_height": group.get("bbox_height"),
                    "per_copy": {
                        "kind": "length",
                        "value": per_copy,
                        "unit": unit,
                        "unit_reason": None
                        if unit
                        else measure.get("unit_reason"),
                        "agrees_across_copies": group.get("length_agrees"),
                    },
                    "layer_total": {
                        "kind": measure.get("kind"),
                        "value": total,
                        "unit": measure.get("unit"),
                        "unit_reason": measure.get("unit_reason"),
                    },
                    "total_over_per_copy": ratio,
                    "detail": detail,
                    "detail_truncated": detail_truncated,
                    "counts": item.get("counts"),
                }
            )
    return out


def _duplicate_statement(group: Mapping[str, Any]) -> str:
    """One sentence naming the cluster count and the per-copy figure."""
    copies = group.get("copies")
    per_copy = (group.get("per_copy") or {}).get("value")
    unit = (group.get("per_copy") or {}).get("unit")
    total = (group.get("layer_total") or {}).get("value")
    total_unit = (group.get("layer_total") or {}).get("unit")
    ratio = group.get("total_over_per_copy")

    unit_text = unit if unit else _NO_UNIT_PHRASE
    total_unit_text = total_unit if total_unit else _NO_UNIT_PHRASE

    if per_copy is None:
        return (
            f"This layer holds {copies} identical copies of the same geometry "
            f"({group.get('entities_each')} entities each). No per-copy "
            "measure was established, so the layer total cannot be divided "
            "here -- but it must not be quoted as if it were one copy."
        )
    head = (
        f"{copies} identical copies of the same geometry, "
        f"{group.get('entities_each')} entities each, {per_copy} {unit_text} "
        "per copy."
    )
    if total is None:
        return head + " The layer has no measured total to compare against."
    tail = f" The layer's measured total is {total} {total_unit_text}"
    if ratio is not None:
        tail += f", which is {ratio}x the per-copy figure"
    return head + tail + ". Quoting the total alone overstates this layer."


def duplicate_scan(
    drawing_id: str,
    *,
    layout: str | None = None,
    dossier: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Every layer the Dossier records as holding detached identical copies.

    `duplicate_warning` answers "is THIS layer duplicated". This answers "what
    in this drawing is", which is the question `find_duplicates` is asked and
    cannot fully answer: that tool matches by `shape_key`, and `shape_key` is
    null on open geometry, so a road network drawn three times is invisible to
    it. It reports a true answer to a narrower question, and without this the
    narrowing is never mentioned.

    `None` means no Dossier -- not computed, and NOT "nothing is duplicated".
    A Dossier that was read and holds no such layer returns a block with an
    empty list, which is the checked answer. Those are different, as
    everywhere else in this module.
    """
    document = dossier if dossier is not None else dossier_for(drawing_id)
    if not document:
        return None

    # Split by the INSTRUMENT that found each one, and it matters more than it
    # looks. Both instruments file their findings under the same `kind`, so a
    # reader that only matches the kind reports all of them as one thing. On
    # the reference drawing that is 27 findings from `shape_key` and 5 from
    # spatial clustering -- and `find_duplicates` can already see all 27 of
    # the first sort, because they ARE keyed rings. Telling it that those are
    # its blind spot would be a fresh false statement written in the course of
    # fixing an old one.
    #
    # The blind spot is only what `shape_key` cannot key: the spatially
    # clustered findings on open geometry.
    unseen: list[dict[str, Any]] = []
    keyed: list[dict[str, Any]] = []
    for block in document.get("layers") or []:
        if not isinstance(block, Mapping):
            continue
        if layout is not None and block.get("layout") != layout:
            continue
        for flag in _anomaly_flags(block):
            if flag.get("kind") != "duplicate_clusters":
                continue
            evidence = flag.get("evidence")
            instrument = (
                evidence.get("instrument") if isinstance(evidence, Mapping) else None
            )
            counts = flag.get("counts")
            counts = counts if isinstance(counts, Mapping) else {}
            row = {
                "layer": block.get("layer"),
                "layout": block.get("layout"),
                "role": block.get("role"),
                "entities": block.get("entities"),
                "instrument": instrument,
                # Published as the instrument recorded them rather than
                # flattened into fields chosen here: the two instruments count
                # different things (`clusters`/`entities_each` against
                # `repeated_keys`), and forcing them into one shape would
                # invent an equivalence neither of them claims.
                "counts": dict(counts),
                "detail": flag.get("detail"),
            }
            (unseen if instrument == "spatial_cluster" else keyed).append(row)
            break

    unseen.sort(key=lambda row: -(row.get("entities") or 0))
    keyed.sort(key=lambda row: -(row.get("entities") or 0))
    return {
        "drawing_id": drawing_id,
        "layout": layout,
        "layers": unseen,
        "layers_shape_key_can_see": keyed,
        "checked": True,
        "basis": (
            "every (layer x layout) bucket the Dossier holds for this drawing "
            "was read. `layers` lists only the findings made by SPATIAL "
            "CLUSTERING -- detached identical copies on geometry `shape_key` "
            "cannot key, which is what a shape-matching duplicate check cannot "
            "see. Findings the shape-key instrument made itself are listed "
            "separately, because that check already reports them."
        ),
    }


def _anomaly_flags(block: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The anomaly entries of one bucket, from any shape the profiler wrote.

    Written tolerantly on purpose. The one time this repo assumed a single
    shape for anomalies, the producing lane wrote `anomalies.flags[]` and the
    consuming lane read `anomaly_kinds`, so a recorded triplication reached a
    user as a bare length with no caveat. A reader that accepts several shapes
    costs a few lines; a reader that accepts one costs a wrong answer.
    """
    raw = block.get("anomalies")
    if isinstance(raw, Mapping):
        for key in ("flags", "items", "anomalies", "findings"):
            if isinstance(raw.get(key), (list, tuple)):
                raw = raw[key]
                break
        else:
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]
