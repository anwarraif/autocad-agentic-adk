"""DOSSIER Phase 8b — one recipe that composes two others, server-side.

Owner: the COMBINE lane.

## The measurement this file was written from

A combination question — *which parcels are BOTH size outliers AND without road
frontage* — was asked five ways over two days. The correct answer, computed
independently, is one parcel. What came back:

* "there are none" — the agent had intersected an empty list that a previous
  step produced because that step had established NOTHING, not because nothing
  matched;
* a set of parameters the agent invented for itself when told to act on a hint;
* one empty reply.

None of those is a model that needs a better prompt. Of the recipes registered
before this one, exactly one accepts `handles`, while the frontage check
publishes two hundred handle rows that nothing at all can consume. So the
composition had nowhere to happen except inside the model's head, over a
population of 2,380 parcels — and set arithmetic over 2,380 items is not
something a language model does, it is something a language model approximates.

This recipe moves the arithmetic to where arithmetic belongs. Both sides run
through `registry.run` — the same entry point every other caller uses, with the
same validation, the same limits, the same envelope — and the intersection,
union, or difference of their answer sets is computed in Python.

## Four decisions, each of which is the point rather than a detail

1.  **An unusable side is refused, never composed.** If either side comes back
    with `conclusion_safety.usable_as_evidence: false`, this recipe does not
    return an empty intersection. An empty set inherited from a step that
    settled nothing reads downstream exactly like a clean negative finding, and
    that confusion is the whole reason this file exists. The refusal names the
    side, repeats its reason, and carries its own `how_to_get_an_answer` and
    `retry_with` forward — plus a ready-to-run version of THIS call with that
    side corrected.

2.  **Which part of a response is its answer set is DECIDED, per recipe, and
    published.** A recipe reports many handles: the ones it tested, the ones it
    skipped, the ones it could not measure, and the ones it FOUND. Only the
    last of those is an answer set. That choice is encoded here one recipe at a
    time, together with what it deliberately leaves out, and travels in the
    response as `left.answer_set_from` and `left.answer_set_rule` so a reader
    can disagree with it in the open. A recipe with no encoded rule is REFUSED
    with the reason its shape is ambiguous — guessing would return the wrong
    set in the right shape, and nothing in the response would show it.

3.  **Every handle says where it came from.** A row carries `sides`, and the
    rows the sides themselves published about it. A reader retraces the answer
    rather than trusting it.

4.  **One level, two runs, no loops.** A composition may not compose another
    composition, and neither side may name this recipe. Each side runs exactly
    once. There is no branch here that runs a third thing.

## What this recipe does NOT do

It does not measure. Every number in the result was measured by a side, and
each side's own `evidence`, `scope_note`, and caveats are carried forward whole
rather than summarised. What this file adds is a set operation and the
bookkeeping that makes it checkable — which is why its own evidence can never
rise above `inferred`, and why the weaknesses of a side are repeated here
instead of being averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping, Sequence

from .. import evidence as ev
from . import selfheal
from .registry import (
    COMPUTED_PROVENANCE,
    MAX_ROWS_RETURNED,
    Param,
    Recipe,
    RecipeRefused,
    get,
    register,
    run as _run_recipe,
    scope_for,
)

#: The name, in one place, because it is also refused as a side (rule 4).
RECIPE_NAME: Final[str] = "combine_findings"

#: The two sides. Two, not N: three sides need a precedence rule between the
#: operations, and a precedence rule invented inside a tool is a rule nobody
#: reading the answer knows about. Two sides compose by being called twice.
SIDES: Final[tuple[str, str]] = ("left", "right")

#: The set operations. `difference` is LEFT minus RIGHT and says so in its own
#: name everywhere it appears, because "difference" alone is the one operation
#: a reader can get backwards without noticing.
OPERATIONS: Final[dict[str, str]] = {
    "intersect": (
        "handles in BOTH answer sets — the plain reading of 'which parcels are "
        "both X and Y'"
    ),
    "union": (
        "handles in EITHER answer set — 'which parcels have at least one of "
        "these problems'. A handle found by both sides appears once, and its "
        "row says so"
    ),
    "difference": (
        "handles in the LEFT answer set and not in the right one — 'which "
        "parcels are X but not Y'. Not symmetric: swapping the sides asks a "
        "different question"
    ),
}

#: How many of a side's own rows are kept per handle. A handle usually appears
#: in one row; the overlap scan is the exception, where one parcel can be a
#: member of several overlapping pairs. Capped so that one badly drawn parcel
#: cannot fill the response, and the number left out is counted (G7, G8).
MAX_SIDE_ROWS_PER_HANDLE: Final[int] = 3

#: The cap on handles returned, taken from the registry rather than copied, so
#: that it cannot drift away from the limit every other recipe applies.
MAX_HANDLES_RETURNED: Final[int] = MAX_ROWS_RETURNED


# =============================================================================
# The answer set: what part of a response IS the finding
# =============================================================================


@dataclass(frozen=True, slots=True)
class _Extraction:
    """One side's answer set, plus everything that bounds how it may be read."""

    handles: tuple[str, ...]
    #: handle -> {"layer": str | None, "rows": [...], "rows_omitted": int}
    context: Mapping[str, dict[str, Any]]
    #: False when the list published is not the whole list the side found —
    #: truncated by a cap, or with members withheld. A caveat does not make a
    #: set incomplete; a missing member does.
    complete: bool
    incomplete_because: str | None
    #: Things the side said that bound the reading without emptying it. Carried
    #: forward verbatim and never averaged away.
    caveats: tuple[str, ...]
    #: Set when the side's own response says it established nothing. Recipes
    #: that publish `conclusion_safety` are handled generically; this covers the
    #: ones that say it another way.
    unusable_because: str | None = None


def _add(context: dict[str, dict[str, Any]], handle: Any, layer: Any, row: Any) -> None:
    """Record one handle and the row the side published about it."""
    key = str(handle or "").strip()
    if not key:
        return
    slot = context.setdefault(
        key, {"layer": None, "rows": [], "rows_omitted": 0}
    )
    if slot["layer"] is None and layer is not None:
        slot["layer"] = layer
    if len(slot["rows"]) < MAX_SIDE_ROWS_PER_HANDLE:
        slot["rows"].append(row)
    else:
        slot["rows_omitted"] += 1


def _rows_of(body: Mapping[str, Any], *keys: str) -> Any:
    """Walk a key path, returning the sentinel `None` for anything missing.

    Deliberately not tolerant: a response whose shape has moved must produce a
    REFUSAL, not an empty set. An extractor that silently returned `[]` for a
    renamed key would reintroduce the exact bug this recipe was written to end.
    """
    node: Any = body
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def _from_outliers(body: Mapping[str, Any]) -> _Extraction | None:
    rows = _rows_of(body, "outliers")
    if not isinstance(rows, list):
        return None

    context: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            _add(context, row.get("handle"), row.get("layer"), dict(row))

    caveats: list[str] = []
    note = body.get("not_measured")
    if isinstance(note, str) and note.strip():
        caveats.append(note.strip())

    # `why_empty` is this recipe's own way of saying it settled nothing: the
    # population held fewer than two values, or every value was identical, so
    # no value COULD deviate. The empty list that comes with it is an unasked
    # question wearing the clothes of a negative finding.
    unusable = None
    why_empty = body.get("why_empty")
    if isinstance(why_empty, str) and why_empty.strip():
        unusable = why_empty.strip()

    truncated = bool(body.get("outliers_truncated"))
    return _Extraction(
        handles=tuple(sorted(context)),
        context=context,
        complete=not truncated,
        incomplete_because=(
            f"`outliers` returned {len(context)} rows and set "
            f"`outliers_truncated`, so more parcels deviate than are named here"
            if truncated
            else None
        ),
        caveats=tuple(caveats),
        unusable_because=unusable,
    )


def _from_frontage_check(body: Mapping[str, Any]) -> _Extraction | None:
    rows = _rows_of(body, "no_frontage", "rows")
    if not isinstance(rows, list):
        return None

    context: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            _add(context, row.get("handle"), row.get("layer"), dict(row))

    block = body.get("no_frontage")
    block = block if isinstance(block, Mapping) else {}
    truncated = bool(block.get("truncated"))

    caveats: list[str] = []
    unproven = _rows_of(body, "frontage_unproven", "count")
    if isinstance(unproven, int) and unproven:
        caveats.append(
            f"{unproven} parcels came back UNPROVEN — inside the tolerance only "
            "against a road entity's bounding box. They are on neither side of "
            "this composition, and a parcel missing from the result may be one "
            "of them rather than one that passed"
        )
    beyond = block.get("beyond_the_search_window")
    if isinstance(beyond, int) and beyond:
        caveats.append(
            f"{beyond} of the parcels without frontage had no road inside the "
            "search window at all, so their distance is withheld rather than "
            "measured"
        )
    warning = body.get("road_geometry_warning")
    if isinstance(warning, str) and warning.strip():
        caveats.append(warning.strip())

    return _Extraction(
        handles=tuple(sorted(context)),
        context=context,
        complete=not truncated,
        incomplete_because=(
            f"`frontage_check` published {len(context)} of "
            f"{block.get('count')} parcels without frontage and marked the list "
            "truncated"
            if truncated
            else None
        ),
        caveats=tuple(caveats),
    )


def _from_overlap_scan(body: Mapping[str, Any]) -> _Extraction | None:
    rows = _rows_of(body, "overlaps", "rows")
    if not isinstance(rows, list):
        return None

    context: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        handles = row.get("handles")
        layers = row.get("layers")
        if not isinstance(handles, Sequence) or isinstance(handles, str):
            continue
        # Both members of the pair, because the finding is about the pair and a
        # parcel that overlaps its neighbour is involved whichever side of the
        # pair it was written on.
        for position, handle in enumerate(handles):
            layer = None
            if isinstance(layers, Sequence) and not isinstance(layers, str):
                if position < len(layers):
                    layer = layers[position]
            _add(context, handle, layer, dict(row))

    block = body.get("overlaps")
    block = block if isinstance(block, Mapping) else {}
    truncated = bool(block.get("truncated"))
    withheld = body.get("pairs_withheld_total")
    withheld = withheld if isinstance(withheld, int) else 0

    reasons: list[str] = []
    if truncated:
        reasons.append(
            f"the overlap list was truncated at {block.get('cap')} of "
            f"{block.get('count')} pairs"
        )
    if withheld:
        reasons.append(
            f"{withheld} pairs were above the per-pair vertex limit and their "
            "shared area is withheld, so whether those parcels overlap is not "
            "known either way"
        )

    caveats: list[str] = []
    slivers = _rows_of(body, "slivers_below_tolerance", "count")
    if isinstance(slivers, int) and slivers:
        caveats.append(
            f"{slivers} further pairs share ground BELOW the tolerance and are "
            "reported as slivers, not overlaps. They are deliberately not in "
            "this answer set — the tolerance is what decides, and it is "
            "published in the side's own response"
        )
    skipped = _rows_of(body, "ring_census", "skipped_total")
    if isinstance(skipped, int) and skipped:
        caveats.append(
            f"{skipped} polygons on those layers have no ring that could be "
            "tested and could not enter the scan at all"
        )

    return _Extraction(
        handles=tuple(sorted(context)),
        context=context,
        complete=not reasons,
        incomplete_because="; ".join(reasons) or None,
        caveats=tuple(caveats),
    )


@dataclass(frozen=True, slots=True)
class AnswerSetRule:
    """The decision about one recipe: WHICH of its handles are its finding."""

    recipe: str
    #: Published as `answer_set_from`. A key path, so it can be looked up in
    #: the side's own response, which travels beside it.
    path: str
    #: Why that path and not another one.
    rule: str
    #: What in the SAME response is deliberately not in the answer set. Written
    #: down because the difference between "not found" and "not looked at" is
    #: the whole subject of this file.
    excludes: str
    extract: Callable[[Mapping[str, Any]], _Extraction | None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "answer_set_from": self.path,
            "answer_set_rule": self.rule,
            "deliberately_excluded": self.excludes,
        }


ANSWER_SETS: Final[tuple[AnswerSetRule, ...]] = (
    AnswerSetRule(
        recipe="outliers",
        path="outliers[].handle",
        rule=(
            "the entities whose measured value falls outside the published "
            "band of mean +/- z x standard deviation. That list IS the finding: "
            "the recipe names no other handles, and the entities inside the "
            "band are counted but never listed"
        ),
        excludes=(
            "the entities that carry no measure at all. They are counted in "
            "`not_measured`, they cannot deviate, and they are not on either "
            "side of this composition"
        ),
        extract=_from_outliers,
    ),
    AnswerSetRule(
        recipe="frontage_check",
        path="no_frontage.rows[].handle",
        rule=(
            "the parcels CONFIRMED to be further from road geometry than the "
            "tolerance. The recipe's own answer is 'which parcels do not touch "
            "a road, named by handle', and this is the only one of its three "
            "outcomes that is both a finding and named — `frontage_established` "
            "is a count with no handles behind it"
        ),
        excludes=(
            "`frontage_unproven`, the parcels measured only against a road "
            "entity's bounding box. That is neither a pass nor a failure, and "
            "adding it to either side of a set operation would publish a "
            "verdict nobody measured"
        ),
        extract=_from_frontage_check,
    ),
    AnswerSetRule(
        recipe="overlap_scan",
        path="overlaps.rows[].handles[]",
        rule=(
            "both members of every pair that shares more ground than that "
            "pair's tolerance. The finding is about a pair, so a parcel is "
            "involved whichever side of the pair it was written on, and one "
            "parcel appearing in several pairs appears once here with all of "
            "its rows"
        ),
        excludes=(
            "`slivers_below_tolerance` — pairs that do share ground but less "
            "than their tolerance — and `pairs_meeting_with_zero_shared_area`, "
            "which is how every neighbouring plot in a master plan meets. "
            "Pairs whose area was WITHHELD are excluded too, and their presence "
            "marks the answer set incomplete rather than being passed over"
        ),
        extract=_from_overlap_scan,
    ),
)

_BY_NAME: Final[dict[str, AnswerSetRule]] = {r.recipe: r for r in ANSWER_SETS}


#: Why a registered recipe is NOT composable. Written per recipe, because
#: "unsupported" tells a reader nothing and invites them to try the next one.
#: A recipe that is not in this map either is composable or gets the general
#: sentence below — which is honest for a recipe this file has never read.
NOT_COMPOSABLE: Final[dict[str, str]] = {
    "parcel_inventory": (
        "it reports counts and areas per land use and never names an entity by "
        "handle, so there is no set to compose. Ask it directly instead"
    ),
    "label_coverage": (
        "its `rows` list every target it tested — labelled and unlabelled alike "
        "— and the list is capped far below the population. The parcels WITHOUT "
        "a label are counted, never named, so a set taken from `rows` would be "
        "neither the finding nor complete"
    ),
    "size_module": (
        "it groups shapes into modules. A group describes a size that repeats; "
        "it is not a set of entities the recipe found something wrong with"
    ),
    "adjacency": (
        "its input IS a set of handles and its output is a distance matrix over "
        "them. Intersecting with it would return the caller's own input dressed "
        "as a finding"
    ),
    "containment": (
        "every row carries an outer handle and a list of inner handles, and the "
        "answer set differs for 'which blocks are empty' and 'which parcels sit "
        "in no block'. Two answers under one key is exactly the case this "
        "recipe will not guess at"
    ),
    "coverage_gap": (
        "it publishes two counts and their difference. The numbers without a "
        "parcel are counted, not named, so nothing here can be composed"
    ),
    "cross_check_classification": (
        "it reports how far two readings agree, and says itself that a "
        "disagreement is not an error. There is no finding set to intersect"
    ),
    "dimension_truth": (
        "every row carries TWO handles — the text that prints a number and the "
        "geometry it was measured against — and which of them is 'the answer' "
        "depends on whether the question is about the label or about the thing "
        "labelled. Its `show_all` parameter changes what `rows` contains as "
        "well, so one rule could not mean the same thing on two calls"
    ),
    "drafting_hygiene": (
        "it publishes several independent findings; some name handles, some "
        "only count them, and each handle list is capped per finding with the "
        "remainder in `handles_omitted`. A union of what it prints is not the "
        "set it found"
    ),
    "revision_diff": (
        "it compares two drawings. A handle from the other drawing is not a "
        "handle in this one, and one response belongs to one drawing (G10)"
    ),
}

_NO_RULE: Final[str] = (
    "no answer-set rule is encoded for this recipe: nothing here has decided "
    "which part of its response is the set it FOUND, as opposed to the part it "
    "tested, skipped, or could not measure. Guessing would compose the wrong "
    "set in the right shape, and nothing in the answer would show it"
)


def why_not_composable(name: str) -> str:
    return NOT_COMPOSABLE.get(name, _NO_RULE)


# =============================================================================
# Parameters: flat, typed, and routed by an explicit prefix
# =============================================================================
#
# A side is (a recipe name, its parameters), which is a nested object — and the
# registry's parameter shapes are deliberately flat: `layers`, `layer`,
# `number`, `text`, `range`, `flag`, and nothing that could carry structure.
# That list is short on purpose (see the head of `registry.py`), and widening it
# from here to make one recipe more convenient would be the wrong trade.
#
# So the surface is flattened instead: `left` and `right` name the recipes, and
# every parameter of a composable recipe is declared TWICE, once per side, with
# the prefix saying where it goes. Nothing is inferred, nothing is routed by a
# lookup table the caller cannot see, and a parameter aimed at a side that does
# not accept it is REFUSED rather than dropped — a dropped parameter answers a
# different question from the one that was asked, and the response would not
# show it.

#: (name, kind, what it chooses, the composable recipes that accept it). The
#: kinds mirror the target recipes' own declarations, and a test asserts that
#: they still do rather than trusting this list to have kept up.
FORWARDABLE: Final[tuple[tuple[str, str, str, tuple[str, ...]], ...]] = (
    (
        "layers",
        "layers",
        "the polygon layers that side works over. Left empty, that side derives "
        "them from this drawing's land use config and names them in its own "
        "response",
        ("outliers", "overlap_scan"),
    ),
    (
        "field",
        "text",
        "`area` or `length` — which measure the deviation is judged on",
        ("outliers",),
    ),
    (
        "z",
        "number",
        "how many standard deviations from the mean before a value counts as "
        "deviating",
        ("outliers",),
    ),
    (
        "parcel_layers",
        "layers",
        "the polygon layers whose rings are checked for frontage",
        ("frontage_check",),
    ),
    (
        "road_layers",
        "layers",
        "the layers carrying the road geometry. This is the parameter that "
        "decides whether a frontage run establishes anything at all: layers of "
        "CLOSED rings can be measured exactly, open paths only through their "
        "bounding boxes",
        ("frontage_check",),
    ),
    (
        "touch_tolerance",
        "number",
        "the distance within which a parcel counts as touching a road, in "
        "DRAWING units",
        ("frontage_check",),
    ),
    (
        "boundary_layer",
        "layer",
        "the single ring layer carrying the site boundary, for the gap half of "
        "the overlap scan",
        ("overlap_scan",),
    ),
    (
        "min_overlap_fraction",
        "number",
        "the fraction of the smaller parcel's area above which a shared area "
        "counts as an overlap",
        ("overlap_scan",),
    ),
)

_FORWARDABLE_NAMES: Final[frozenset[str]] = frozenset(row[0] for row in FORWARDABLE)


def _side_params() -> tuple[Param, ...]:
    out: list[Param] = []
    for side in SIDES:
        for name, kind, about, accepted in FORWARDABLE:
            out.append(
                Param(
                    f"{side}_{name}",
                    kind,
                    f"handed to the {side.upper()} recipe as its `{name}`: "
                    f"{about}. Accepted by: {', '.join(accepted)}. Naming it "
                    f"for a {side} recipe that does not declare `{name}` is "
                    "refused, not ignored.",
                )
            )
    return tuple(out)


# =============================================================================
# Running one side
# =============================================================================


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _recipe_for_side(side: str, value: Any) -> AnswerSetRule:
    """The rule for one side's recipe, or a refusal that says which and why."""
    name = str(value or "").strip()
    if not name:
        raise RecipeRefused(
            "RECIPE_PARAM_REQUIRED",
            f"{side!r} is empty; it must name the recipe to run on that side.",
            "It is the NAME of a recipe, as a string — for example "
            f"{side}='outliers'. Not an object: the parameters for that side "
            f"are given separately as `{side}_<parameter>`. Composable today: "
            + ", ".join(sorted(_BY_NAME))
            + ".",
        )
    if name == RECIPE_NAME:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"{side!r} names {RECIPE_NAME!r}, and a composition cannot compose "
            "another composition.",
            "One level, deliberately. Nesting these would let one call fan out "
            "into an unbounded number of runs, and the answer would name a "
            "chain nobody could retrace. Compose two ordinary recipes; if the "
            "question really needs three, run this twice and say so.",
        )
    if get(name) is None:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"{side!r} names {name!r}, which is not a registered recipe.",
            "Composable today: " + ", ".join(sorted(_BY_NAME)) + ". Run "
            "`list_analyses` for the whole catalogue — an answer built on a "
            "recipe that does not exist is not a smaller answer, it is a "
            "different one.",
        )
    rule = _BY_NAME.get(name)
    if rule is None:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"{side!r} names {name!r}, which exists but cannot be composed: "
            + why_not_composable(name)
            + ".",
            "Composable today: " + ", ".join(sorted(_BY_NAME)) + ". This is a "
            "refusal rather than a guess on purpose: a recipe reports the "
            "handles it tested, the ones it skipped and the ones it FOUND, and "
            "composing the wrong one of those returns a wrong answer that looks "
            "exactly like a right one.",
        )
    return rule


def _params_for_side(
    side: str, rule: AnswerSetRule, forwarded: Mapping[str, Any]
) -> dict[str, Any]:
    """The parameters really handed to one side, refusing the ones it cannot take."""
    prefix = f"{side}_"
    params: dict[str, Any] = {}
    recipe = get(rule.recipe)
    for key, value in sorted(forwarded.items()):
        if not key.startswith(prefix) or value is None:
            continue
        name = key[len(prefix) :]
        if recipe is None or recipe.param(name) is None:
            accepted = (
                ", ".join(f"{prefix}{p.name}" for p in recipe.params)
                if recipe is not None and recipe.params
                else "none"
            )
            raise RecipeRefused(
                "RECIPE_PARAM_UNKNOWN",
                f"{key!r} was given, but the {side} recipe {rule.recipe!r} "
                f"declares no parameter named {name!r}.",
                f"What {rule.recipe!r} accepts on this side: {accepted}. A "
                "parameter that is quietly dropped produces an answer over a "
                "scope different from the one that was asked for, and there is "
                "nothing in the response that shows it happened.",
            )
        params[name] = value
    return params


def _run_side(
    side: str,
    rule: AnswerSetRule,
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """One side, through the same entry point every other caller uses.

    Not a private copy of the recipe's logic and not a second route into it:
    the same `run`, so the side gets the same parameter validation, the same
    limits, the same envelope, and the same refusals it would get if a person
    had called it directly. Anything else and two callers would be able to
    receive two different answers to the same question.
    """
    try:
        body = _run_recipe(
            drawing_id, rule.recipe, dict(params), layout=layout, units=dict(units)
        )
        # The same self-heal the HTTP route applies, for the same reason: a
        # side that settles nothing but publishes the call that would settle it
        # is not an answer, it is a question left on the floor. Bounded at one
        # retry per side by `selfheal` itself, so a composition runs at most
        # two and cannot loop. What comes back is labelled RETRIED, NOT
        # VERIFIED and travels that way into the answer.
        return selfheal.heal(
            body,
            drawing_id=drawing_id,
            layout=layout,
            units=units,
            run=_run_recipe,
        )
    except RecipeRefused as exc:
        # Re-raised rather than swallowed, with the side named. The original
        # code is kept so that the refusal can still be looked up; only the
        # message gains the one fact the caller is missing — WHICH half of the
        # composition refused.
        raise RecipeRefused(
            exc.code,
            f"the {side} side ({rule.recipe}) refused: {exc.message}",
            exc.hint,
        ) from exc


# =============================================================================
# Composition
# =============================================================================


def _side_block(
    side: str,
    rule: AnswerSetRule,
    params: Mapping[str, Any],
    response: Mapping[str, Any],
    extraction: _Extraction | None,
) -> dict[str, Any]:
    safety = response.get("conclusion_safety")
    # Whether the side SETTLED its half of the question, kept apart from
    # whether the list it published is complete. They are two different
    # failures and were measured as one: a frontage run that established
    # nothing published a list of parcels without frontage that was empty and
    # untruncated — complete, and worth nothing.
    #
    # A side that healed itself counts as settled here. `usable_as_evidence`
    # stays False on it -- nothing verified the parameters the retry chose --
    # and `usable_for_composition` is the narrower permission the retry grants:
    # compute with the figure, and carry its caveat into everything built on
    # it. Refusing to compute would publish "no parcel is both" over a side
    # that was never asked, which is the failure this whole path exists to
    # stop.
    settled = selfheal.composable(response) and not (
        extraction is not None and extraction.unusable_because
    )
    block: dict[str, Any] = {
        "side": side,
        **rule.as_dict(),
        "answer_set_settled": settled,
        "answer_set_settled_note": (
            None
            if settled
            else (
                "this side did not settle its half of the question. Its answer "
                "set is not empty in the sense of 'nothing matched'; it is "
                "empty in the sense of 'not asked'"
            )
        ),
        "params_used": _jsonable(response.get("params_used") or dict(params)),
        "retried": selfheal.was_retried(response),
        "retried_with": _jsonable(
            ((response.get("auto_retry") or {}).get("plan") or {}).get("params")
        )
        if selfheal.was_retried(response)
        else None,
        "retried_because": ((response.get("auto_retry") or {}).get("because"))
        if selfheal.was_retried(response)
        else None,
        "scope_note": response.get("scope_note"),
        "evidence": response.get("evidence"),
        "conclusion_safety": safety if isinstance(safety, Mapping) else None,
    }
    if extraction is None:
        block.update(
            {
                "answer_set_size": None,
                "answer_set_complete": False,
                "answer_set_incomplete_because": (
                    f"the key path {rule.path!r} is absent from the response"
                ),
                "answer_set_missing": (
                    f"the key path {rule.path!r} is not present in what "
                    f"{rule.recipe!r} returned, so its answer set could not be "
                    "read at all. Withheld is not empty (G8)"
                ),
                "caveats": [],
            }
        )
        return block
    block.update(
        {
            "answer_set_size": len(extraction.handles),
            "answer_set_complete": extraction.complete,
            "answer_set_incomplete_because": extraction.incomplete_because,
            "caveats": list(extraction.caveats),
        }
    )
    return block


def _apply(operation: str, left: frozenset[str], right: frozenset[str]) -> frozenset[str]:
    if operation == "intersect":
        return left & right
    if operation == "union":
        return left | right
    return left - right


def _result_rows(
    handles: Sequence[str],
    left_rule: AnswerSetRule,
    right_rule: AnswerSetRule,
    left: _Extraction,
    right: _Extraction,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for handle in handles:
        in_left = handle in left.context
        in_right = handle in right.context
        left_slot = left.context.get(handle) or {}
        right_slot = right.context.get(handle) or {}
        sides = [s for s, flag in zip(SIDES, (in_left, in_right)) if flag]
        rows.append(
            {
                "handle": handle,
                "layer": left_slot.get("layer") or right_slot.get("layer"),
                "sides": sides,
                "found_by": [
                    (left_rule.recipe if in_left else None),
                    (right_rule.recipe if in_right else None),
                ],
                "left": (
                    {
                        "recipe": left_rule.recipe,
                        "answer_set_from": left_rule.path,
                        "rows": left_slot.get("rows") or [],
                        "rows_omitted": left_slot.get("rows_omitted") or 0,
                    }
                    if in_left
                    else None
                ),
                "right": (
                    {
                        "recipe": right_rule.recipe,
                        "answer_set_from": right_rule.path,
                        "rows": right_slot.get("rows") or [],
                        "rows_omitted": right_slot.get("rows_omitted") or 0,
                    }
                    if in_right
                    else None
                ),
            }
        )
    # `found_by` keeps its Nones rather than being compacted: the position says
    # which side, and a compacted list of one name does not say which side it
    # came from.
    return rows


def _retry_plans(
    *,
    side: str,
    safety: Mapping[str, Any],
    given: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """The side's own retry plans, rewritten as calls to THIS recipe.

    Naming a candidate was measured to be not enough: told to act on a hint
    without an exact call, the agent composed parameters of its own and was
    wrong by a new route. So the plan handed back here is the whole composition,
    ready to run, with the one parameter changed — nothing left to compose.
    """
    plans: list[dict[str, Any]] = []
    skipped: list[str] = []
    raw = safety.get("retry_with")
    if not isinstance(raw, list):
        return plans, skipped

    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        recipe = str(entry.get("recipe") or "").strip()
        if not recipe or recipe not in _BY_NAME:
            skipped.append(
                f"a retry naming {recipe or 'no recipe'} was left out: "
                + why_not_composable(recipe)
            )
            continue
        params = entry.get("params")
        params = params if isinstance(params, Mapping) else {}
        unusable = sorted(k for k in params if k not in _FORWARDABLE_NAMES)
        if unusable:
            skipped.append(
                f"a retry setting {', '.join(unusable)} was left out: this "
                "composition declares no parameter of that name for either side"
            )
            continue
        call = {k: _jsonable(v) for k, v in given.items() if v is not None}
        call[side] = recipe
        for key, value in params.items():
            call[f"{side}_{key}"] = _jsonable(value)
        plans.append(
            {
                "recipe": RECIPE_NAME,
                "params": call,
                "because": entry.get("because"),
            }
        )
    return plans, skipped


# =============================================================================
# The recipe
# =============================================================================


def _observation(detail: str, *, locator: str | None, layout: str | None) -> ev.Observation:
    """A set relation between two computed answers.

    `Origin.TOPOLOGY` sits in the `coordinates` corpus, which never STATES. That
    is the right ceiling: no word anywhere in a drawing makes that file say a
    parcel is both a size outlier and without frontage. It is inferred from two
    measurements, and it can be no stronger than the weaker of them.
    """
    return ev.Observation(
        origin=ev.Origin.TOPOLOGY, detail=detail, locator=locator, layout=layout
    )


def _absence(detail: str, *, locator: str | None, layout: str | None) -> ev.Observation:
    return ev.Observation(
        origin=ev.Origin.ABSENCE, detail=detail, locator=locator, layout=layout
    )


def _run_combine(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    left: str,
    right: str,
    operation: str,
    **forwarded: Any,
) -> dict[str, Any]:
    op = str(operation or "").strip().lower()
    if op not in OPERATIONS:
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"operation {operation!r} is not one of "
            + ", ".join(sorted(OPERATIONS))
            + ".",
            "`intersect` is 'both', `union` is 'either', and `difference` is "
            "LEFT minus RIGHT — which is not symmetric, so swapping the sides "
            "asks a different question. No operation is assumed from the "
            "wording of a question.",
        )

    rules = {
        "left": _recipe_for_side("left", left),
        "right": _recipe_for_side("right", right),
    }
    params = {
        side: _params_for_side(side, rules[side], forwarded) for side in SIDES
    }

    # Two runs. Exactly two, always, whatever the answer is: nothing below
    # loops, retries, or widens a scope on its own.
    responses = {
        side: _run_side(
            side,
            rules[side],
            drawing_id=drawing_id,
            layout=layout,
            units=units,
            params=params[side],
        )
        for side in SIDES
    }
    extractions = {
        side: rules[side].extract(responses[side]) for side in SIDES
    }

    blocks = {
        side: _side_block(
            side, rules[side], params[side], responses[side], extractions[side]
        )
        for side in SIDES
    }

    #: The parameters as this call received them, so a retry can be handed back
    #: whole rather than as an instruction to reassemble one.
    given: dict[str, Any] = {
        "left": rules["left"].recipe,
        "right": rules["right"].recipe,
        "operation": op,
        **{k: v for k, v in forwarded.items() if v is not None},
    }

    scope_note = (
        f"layout {layout!r}; {op} of the answer set of {rules['left'].recipe!r} "
        f"({rules['left'].path}) and the answer set of {rules['right'].recipe!r} "
        f"({rules['right'].path}), both run once through the analysis registry "
        f"on drawing {drawing_id}; handles are compared as exact strings within "
        f"this one drawing and at most {MAX_HANDLES_RETURNED} are returned"
    )
    scope = scope_for(layout=layout, units=units, note=scope_note)

    common: dict[str, Any] = {
        "scope_note": scope_note,
        "operation": op,
        "operation_rule": OPERATIONS[op],
        "operations_available": {k: v for k, v in sorted(OPERATIONS.items())},
        "left": blocks["left"],
        "right": blocks["right"],
        "where_the_arithmetic_happened": (
            "here, in Python, over two answer sets that were fetched once each. "
            "The set operation is not performed by a language model and is not "
            "described to one: at 2,380 parcels an intersection held in a "
            "model's head is a guess wearing the shape of an answer"
        ),
        "one_level_only": (
            "a composition cannot compose another composition, and neither side "
            "may name this recipe. Two runs per call, always"
        ),
        "composable_recipes": [rule.as_dict() for rule in ANSWER_SETS],
        "not_composable": _not_composable_rows(),
        "limits_applied": {
            "sides": len(SIDES),
            "runs_per_call": len(SIDES),
            "handles_returned": MAX_HANDLES_RETURNED,
            "side_rows_per_handle": MAX_SIDE_ROWS_PER_HANDLE,
            "recursion_depth": 1,
        },
    }

    unusable = _first_unusable(rules, responses, extractions)
    if unusable is not None:
        return _refused(
            unusable,
            common=common,
            rules=rules,
            responses=responses,
            given=given,
            drawing_id=drawing_id,
            scope=scope,
            layout=layout,
            op=op,
        )

    # `_first_unusable` returns for a side whose answer set could not be read at
    # all, so both are present here. Checked rather than assumed: an assertion
    # can be switched off at the interpreter, and what would follow is a
    # traceback in place of an answer.
    left_x = extractions["left"]
    right_x = extractions["right"]
    if left_x is None or right_x is None:  # pragma: no cover - unreachable
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            "a side's answer set could not be read and was not refused earlier.",
            "This is a defect in this recipe, not in the call. Report it with "
            "the two recipe names above.",
        )

    resolved = sorted(
        _apply(op, frozenset(left_x.handles), frozenset(right_x.handles))
    )
    kept = resolved[:MAX_HANDLES_RETURNED]
    rows = _result_rows(kept, rules["left"], rules["right"], left_x, right_x)

    complete = left_x.complete and right_x.complete
    incomplete_reasons = [
        f"{side}: {extractions[side].incomplete_because}"  # type: ignore[union-attr]
        for side in SIDES
        if extractions[side] is not None and extractions[side].incomplete_because  # type: ignore[union-attr]
    ]
    caveats = [
        f"{rules[side].recipe}: {text}"
        for side in SIDES
        for text in (extractions[side].caveats if extractions[side] else ())
    ]
    # A retried side is a caveat on everything computed from it, so it travels
    # with the other caveats rather than only in that side's own block -- the
    # sides' blocks are where a reader looks second.
    retried_notes = [
        f"the {side} side ({rules[side].recipe}) settled nothing on its first "
        f"run and was re-run once with "
        f"{_jsonable(blocks[side].get('retried_with'))}, chosen automatically "
        f"and not confirmed. Reason the plan gave: "
        f"{blocks[side].get('retried_because') or 'none published'}"
        for side in SIDES
        if blocks[side].get("retried")
    ]
    caveats.extend(retried_notes)

    observations = [
        _observation(
            f"{rules[side].recipe} identified "
            f"{len(extractions[side].handles)} handles through "  # type: ignore[union-attr]
            f"{rules[side].path}",
            locator=rules[side].recipe,
            layout=layout,
        )
        for side in SIDES
    ]
    observations.append(
        _observation(
            f"{len(resolved)} handles satisfy {op} over those two sets",
            locator=f"{rules['left'].recipe}+{rules['right'].recipe}",
            layout=layout,
        )
    )
    for reason in incomplete_reasons:
        observations.append(
            _absence(reason, locator=rules["left"].recipe, layout=layout)
        )

    evidence = ev.Evidence.of(
        ev.Claim(
            value=(
                f"{len(resolved)} entities satisfy "
                f"{rules['left'].recipe} {op} {rules['right'].recipe} in layout "
                f"{layout!r}"
            ),
            tokens=(),
        ),
        drawing_id=drawing_id,
        scope=scope,
        provenance=COMPUTED_PROVENANCE,
        observations=tuple(observations),
        not_established=(
            "anything beyond set membership. This recipe measured nothing: it "
            "read two answers and applied one set operation to them, so it "
            "cannot be more right than the weaker side. Whether being in both "
            "sets MEANS anything is the sides' question, and both sides' "
            "`evidence` and `caveats` travel above under `left` and `right` "
            "rather than being averaged into this grade"
        ),
        how_to_verify=(
            "run each side on its own with the parameters published under "
            "`left.params_used` and `right.params_used` — they are the same "
            "calls this made — and check that each returned handle appears in "
            "the side's own list. Then open the handles and look"
        ),
        ceiling=ev.Grade.INFERRED,
    )

    body = dict(common)
    body.update(
        {
            "composed": True,
            "refusal": None,
            "handles": [row["handle"] for row in rows],
            "handles_total": len(resolved),
            "handles_returned": len(kept),
            "handles_dropped": len(resolved) - len(kept),
            "handles_truncated": len(resolved) > MAX_HANDLES_RETURNED,
            "rows": rows,
            "rows_note": (
                "every row names the side or sides that produced it and carries "
                "those sides' own rows, so the answer can be retraced instead "
                "of trusted"
            ),
            "caveats_from_the_sides": caveats,
            "caveats_note": (
                "carried forward verbatim. A caveat bounds how the result is "
                "read; it does not empty it, and it is not summarised away"
            ),
            "conclusion_safety": _safety(
                complete=complete,
                reasons=incomplete_reasons,
                op=op,
                found=len(resolved),
                retried=retried_notes,
            ),
            "not_measured": (
                "this recipe measures nothing of its own. It ran "
                f"{rules['left'].recipe} and {rules['right'].recipe} once each "
                "and applied one set operation to the handles they published. "
                "Parcels that neither side could measure are in neither set and "
                "are therefore absent from this result without being ruled out "
                "(G8) — the sides' `caveats` above say how many"
            ),
        }
    )
    return ev.attach(body, evidence)


def _not_composable_rows() -> list[dict[str, str]]:
    """Every registered recipe that cannot be a side, with the reason.

    Read from the live registry rather than from a list written here, so that a
    recipe mounted by another lane appears the day it is mounted — with the
    general sentence, which is the honest thing to say about a response shape
    this file has never read.
    """
    from .registry import known

    return [
        {"recipe": name, "why": why_not_composable(name)}
        for name in known()
        if name not in _BY_NAME and name != RECIPE_NAME
    ]


def _first_unusable(
    rules: Mapping[str, AnswerSetRule],
    responses: Mapping[str, Mapping[str, Any]],
    extractions: Mapping[str, _Extraction | None],
) -> dict[str, Any] | None:
    """The first side that settled nothing, in a fixed order. Never both at once.

    Left before right, always, so that two runs of the same call name the same
    side. A refusal that moved between sides would look like two different
    problems.
    """
    for side in SIDES:
        response = responses[side]
        extraction = extractions[side]
        if extraction is None:
            return {
                "side": side,
                "recipe": rules[side].recipe,
                "code": "COMBINE_ANSWER_SET_MISSING",
                "why": (
                    f"{rules[side].recipe!r} returned a response with no "
                    f"{rules[side].path!r} in it, so the part of it that is its "
                    "ANSWER could not be read. That is a contract change, not "
                    "an empty result"
                ),
                "how_to_get_an_answer": (
                    f"run {rules[side].recipe!r} on its own and compare its "
                    "response with the key path above. If the recipe's shape "
                    "has moved, the rule in this file has to move with it — "
                    "through a merge request, like every other recipe change"
                ),
                "retry_with": [],
            }
        # A side that healed itself is past this gate. `usable_as_evidence`
        # stays False on a retried result -- nothing verified the parameters
        # the plan chose -- so reading that flag alone would refuse an answer
        # the recipe did in fact produce. `composable` is the narrower
        # permission the retry grants, and the composed verdict is demoted to
        # ESTABLISHED, ONE SIDE RETRIED further down, with the retry named.
        safety = response.get("conclusion_safety")
        if (
            not selfheal.composable(response)
            and isinstance(safety, Mapping)
            and safety.get("usable_as_evidence") is False
        ):
            return {
                "side": side,
                "recipe": rules[side].recipe,
                "code": "COMBINE_SIDE_ESTABLISHED_NOTHING",
                "verdict": safety.get("verdict"),
                "why": safety.get("why"),
                "side_do_not": safety.get("do_not"),
                "how_to_get_an_answer": safety.get("how_to_get_an_answer"),
                "candidates": safety.get("candidate_road_layers"),
                "retry_with": safety.get("retry_with") or [],
            }
        if extraction.unusable_because:
            return {
                "side": side,
                "recipe": rules[side].recipe,
                "code": "COMBINE_SIDE_ESTABLISHED_NOTHING",
                "verdict": "NOTHING ESTABLISHED",
                "why": extraction.unusable_because,
                "side_do_not": (
                    "do not read this side's empty list as 'nothing matched': "
                    "the question was not settled"
                ),
                "how_to_get_an_answer": (
                    f"run {rules[side].recipe!r} on its own, read the reason "
                    "above, and narrow or widen the population it works over"
                ),
                "retry_with": [],
            }
    return None


def _refused(
    unusable: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
    rules: Mapping[str, AnswerSetRule],
    responses: Mapping[str, Mapping[str, Any]],
    given: Mapping[str, Any],
    drawing_id: str,
    scope: ev.Scope,
    layout: str | None,
    op: str,
) -> dict[str, Any]:
    """No composition, and no empty set standing in for one.

    This is the whole reason the file exists. An empty intersection and an
    unasked question are indistinguishable downstream unless the response says
    which it was — and the measured failure was exactly that: a frontage run
    that settled nothing produced an empty list of landlocked parcels, the
    empty list was intersected with thirty outliers, and the answer came back
    "there are none". One parcel is in fact both.
    """
    side = str(unusable["side"])
    other = SIDES[0] if side == SIDES[1] else SIDES[1]
    safety = responses[side].get("conclusion_safety")
    plans, plans_skipped = _retry_plans(
        side=side,
        safety=safety if isinstance(safety, Mapping) else {},
        given=given,
    )

    evidence = ev.Evidence.unknown(
        ev.Claim(
            value=(
                f"whether any entity satisfies {rules['left'].recipe} {op} "
                f"{rules['right'].recipe} in layout {layout!r}"
            ),
            tokens=(),
        ),
        drawing_id=drawing_id,
        scope=scope,
        provenance=COMPUTED_PROVENANCE,
        observations=(
            _absence(
                f"the {side} side ({rules[side].recipe}) established nothing, so "
                "no set operation was performed",
                locator=rules[side].recipe,
                layout=layout,
            ),
        ),
        not_established=(
            "nothing at all about this combination. The composition was not "
            f"performed: the {side} side settled neither that its condition "
            "holds nor that it fails, and an empty result taken from it would "
            "read as a finding while resting on a question that was never asked"
        ),
        how_to_verify=(
            str(unusable.get("how_to_get_an_answer") or "")
            or f"run {rules[side].recipe!r} on its own and read why it settled nothing"
        ),
    )

    body = dict(common)
    body.update(
        {
            "composed": False,
            "refusal": {
                **{k: v for k, v in unusable.items() if k != "retry_with"},
                "other_side": {
                    "side": other,
                    "recipe": rules[other].recipe,
                    "ran": True,
                    "note": (
                        "this side ran and its own response is above under "
                        f"`{other}`. It is not the reason for the refusal and "
                        "its answer set stands on its own"
                    ),
                },
                "side_retry_with": list(unusable.get("retry_with") or []),
                "side_retry_note": (
                    "these are the retry plans the unusable side published for "
                    "ITSELF. They are repeated here unchanged; the ready-to-run "
                    "versions of THIS call are in `retry_with` below"
                ),
            },
            # G8, and the sentence this whole recipe is built around: no empty
            # list. A caller reading `handles` gets nothing to mistake for a
            # finding, and a reason instead.
            "handles": None,
            "handles_total": None,
            "handles_returned": 0,
            "handles_dropped": 0,
            "handles_truncated": False,
            "rows": [],
            "handles_withheld_reason": (
                f"the {side} side ({rules[side].recipe}) established nothing, so "
                "there is no set to intersect, unite, or subtract. An empty list "
                "here would be read as 'there are none', which is the wrong "
                "answer measured five times before this recipe existed"
            ),
            "caveats_from_the_sides": [],
            # Deliberately NOT inside `conclusion_safety`: the API layer re-runs
            # a call whose `conclusion_safety.retry_with` is populated, and a
            # re-run there would replace this composition with one side's answer.
            # The plans below are complete calls to THIS recipe, and the caller
            # runs one of them.
            "retry_with": plans,
            "retry_note": (
                "each entry is this same composition with the unusable side "
                "corrected, ready to run as it stands. Run one VERBATIM — do "
                "not compose parameters of your own. Measured: told to act on a "
                "hint without an exact call, the agent invented its own "
                "parameters and was wrong by a new route"
            ),
            "retry_plans_left_out": plans_skipped,
            "conclusion_safety": {
                "usable_as_evidence": False,
                "verdict": "NOTHING TO COMPOSE",
                "why": (
                    f"the {side} side ({rules[side].recipe}) came back with "
                    "`usable_as_evidence: false` — "
                    + str(unusable.get("why") or "it established nothing")
                ),
                "do_not": (
                    "do NOT report an empty result, and do not say 'there are "
                    "none'. Nothing was ruled out here. Say which side settled "
                    "nothing, why, and what would settle it — or run one of "
                    "`retry_with` and answer the question"
                ),
                "what_to_do": (
                    "run one entry from `retry_with` above, verbatim, then "
                    "report the composed answer and name the parameter that "
                    "changed"
                ),
            },
            "not_measured": (
                "the composition itself. Neither the intersection nor its "
                "absence was established, because one side never settled its "
                "half of the question"
            ),
        }
    )
    return ev.attach(body, evidence)


def _safety(
    *,
    complete: bool,
    reasons: Sequence[str],
    op: str,
    found: int,
    retried: Sequence[str] = (),
) -> dict[str, Any]:
    """Whether this composition may be quoted as evidence.

    Deliberately carries no `retry_with` key. The API layer re-runs any response
    whose `conclusion_safety.retry_with` is populated, and what it would re-run
    is a single recipe — replacing a composition with half of itself. The
    ready-to-run calls live at the top level of the body instead.
    """
    if complete and retried:
        # The arithmetic is sound and the input is not confirmed, which is one
        # state and not two. Refusing here would publish "no parcel is both"
        # over a side that was never asked -- the exact failure this path
        # exists to stop -- and stamping ESTABLISHED would hide that a
        # parameter was chosen by machine. So: quotable, with the sentence that
        # makes it true travelling in the same object.
        return {
            "usable_as_evidence": True,
            "verdict": "ESTABLISHED, ONE SIDE RETRIED",
            "why": (
                f"both sides published a complete answer set and the {op} of "
                f"them holds {found} handles. One side settled nothing on its "
                "first run and was re-run once with the parameters its own "
                "response named: " + "; ".join(retried)
            ),
            "must_say": (
                "every answer built on this must name the retried side and the "
                "parameters the retry chose, and must say they were chosen "
                "automatically and not confirmed. The figure is real; the "
                "scope it was measured over is a candidate"
            ),
            "do_not": (
                "do not present this as verified, and do not read it as a "
                "statement about parcels neither side could measure — they are "
                "in neither set and are listed in `caveats_from_the_sides`, "
                "not ruled out"
            ),
            "to_verify": (
                "open the drawing and check that the parameters named above "
                "are the right ones for the question, then re-run this call "
                "with them stated explicitly"
            ),
        }
    if complete:
        return {
            "usable_as_evidence": True,
            "verdict": "ESTABLISHED",
            "why": (
                f"both sides published a complete answer set and the {op} of "
                f"them holds {found} handles"
            ),
            "do_not": (
                "do not read this as a statement about parcels neither side "
                "could measure — they are in neither set and are listed in "
                "`caveats_from_the_sides`, not ruled out"
            ),
        }
    return {
        "usable_as_evidence": False,
        "verdict": "INCOMPLETE INPUT SET",
        "why": (
            "a side published only part of the set it found, so this "
            "composition was computed over part of the population: "
            + "; ".join(reasons)
        ),
        "do_not": (
            "do not read a handle's ABSENCE from this result as evidence that "
            "it fails the test. Every handle returned really is in the sets it "
            "names — positives are sound — but the list may be short, so it "
            "cannot support 'these are the only ones'"
        ),
        "how_to_get_an_answer": (
            "narrow the side that truncated — name fewer layers, or a tighter "
            "population — until its own response reports its list complete, "
            "then compose again"
        ),
    }


register(
    Recipe(
        name=RECIPE_NAME,
        answers=(
            "which entities are in the findings of TWO recipes at once — the "
            "intersection, union, or difference of their answer sets, computed "
            "server-side"
        ),
        when_to_use=(
            "any question with an AND, an OR, or a BUT NOT across two "
            "computations: 'which parcels are both size outliers and without "
            "road frontage', 'which parcels overlap a neighbour but are not "
            "outliers'. Use it INSTEAD of running two recipes and comparing "
            "their handle lists yourself — a set operation over thousands of "
            "handles is not something to do by reading. Example params: "
            "{\"left\": \"outliers\", \"right\": \"frontage_check\", "
            "\"operation\": \"intersect\", \"right_road_layers\": [\"<a layer "
            "of closed rings>\"]}. `left` and `right` are recipe NAMES, and "
            "each side's own parameters are prefixed with that side."
        ),
        params=(
            Param(
                "left",
                "text",
                "the NAME of the recipe on the left, as a string — not an "
                "object. Composable: " + ", ".join(sorted(_BY_NAME)) + ". Any "
                "other registered recipe is refused with the reason its answer "
                "set is ambiguous, and `not_composable` in the response lists "
                "them all.",
                required=True,
            ),
            Param(
                "right",
                "text",
                "the NAME of the recipe on the right. Same list as `left`.",
                required=True,
            ),
            Param(
                "operation",
                "text",
                "`intersect` (in both), `union` (in either), or `difference` "
                "(in LEFT and not in right — not symmetric, so the order of the "
                "sides is part of the question).",
                default="intersect",
            ),
        )
        + _side_params(),
        returns=(
            "composed",
            "handles",
            "handles_total",
            "rows[].sides",
            "rows[].left",
            "rows[].right",
            "left.answer_set_from",
            "right.answer_set_from",
            "left.answer_set_complete",
            "caveats_from_the_sides",
            "conclusion_safety",
            "refusal.why",
            "retry_with",
            "not_composable",
        ),
        built_on=("recipes.registry.run", "recipes.registry.get"),
        limits={
            "sides": len(SIDES),
            "runs_per_call": len(SIDES),
            "recursion_depth": 1,
            "handles_returned": MAX_HANDLES_RETURNED,
            "side_rows_per_handle": MAX_SIDE_ROWS_PER_HANDLE,
            "composable_recipes": len(ANSWER_SETS),
        },
        run=_run_combine,
    )
)
