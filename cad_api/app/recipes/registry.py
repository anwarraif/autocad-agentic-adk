"""Analysis recipe registry — the engine, not the recipes.

Owner: subagent R — UPLIFT-09.

One decision determines the whole shape of this file, and it is a decision
about what was NOT built:

    There is no `run_python(code)`, no `eval`, no `exec`, and no way for a
    caller to hand over logic. What can be handed over is only the NAME of a
    recipe and the VALUES of its parameters.

There are three reasons, and not one of them is generic caution:

1.  **A drawing's contents are untrusted input.** Drawings come from
    contractors and consultants; 43,109 text strings from a single file enter
    the agent's context. Text that reads like an instruction is prompt
    injection, and a tool that runs code hands it an execution path.
2.  **The credentials live in the same process.** `cad-api` holds
    `MONGODB_URI` to a database other teams use. The blast radius is not just
    this project.
3.  **Answers cannot be reproduced.** One-off code that differs on every call
    produces numbers that cannot be compared across sessions — exactly the
    disease `docs/GROUND-TRUTH-2.md` exists to cure.

That is why a new recipe **always** goes through a merge request, never
through the runtime. `register()` can only be called at module import, and
`run()` has no branch that accepts code. That is tested, not assumed.

## The rules that bind every recipe

1.  **Deterministic.** Same input → same output, forever. No clock, no
    randomness, no network calls — and that is tested as a property of the
    file, not relied on as the author's intent. If an answer changes without
    the drawing changing, it is not a measurement.
2.  **Bounded** (G7). Every recipe states the size limit of its input through
    `limits` and fails with a **suggestion** when it is exceeded — not running
    for ten minutes and not truncating silently.
3.  **Labelled** (UPLIFT-08). A recipe that says what something IS must carry
    `evidence`; a recipe that only measures must carry `not_measured`.
    Two different axes, and `carries_meaning` is what picks between them.
4.  **Tested.** A recipe without a test is not a recipe, and every recipe
    carries at least one Janadriyah ground-truth number in its test.
5.  **Riding on, not rewriting.** Recipes recompose functions that already
    exist in `store_spatial`, `store_geometry`, `store_stats`, and
    `store_landuse`. `built_on` names them one by one inside the response,
    so that an answer can be traced to the code that computed it without
    reading this file.

## What happens when no recipe fits

`run()` with an unknown name returns the **catalogue**, not a bare error. An
agent that receives "unknown recipe" will force the wrong tool and produce
the wrong number in the right tone; an agent that receives a catalogue can
say what does not exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping, Sequence

from .. import evidence as ev

#: Raised only when the shape of the response envelope changes so that old and
#: new answers cannot be told apart without looking at them.
RECIPE_CONTRACT_VERSION: Final[int] = 1

#: The sentence the agent MUST say, verbatim, when no recipe fits. It lives
#: here, not only in the agent instructions, so that the catalogue itself
#: carries it — a sentence that lives only in a prompt is lost at the first
#: paraphrase.
NO_RECIPE_SENTENCE: Final[str] = (
    "This question needs a computation that is not available yet: {what}. "
    "What I can give you right now: {nearest}. "
    "That computation can be added as an analysis recipe."
)

#: Limits that apply to EVERY recipe, on top of the recipe's own limits.
MAX_LAYERS_PER_PARAM: Final[int] = 200
MAX_ROWS_RETURNED: Final[int] = 200
MAX_TEXT_PARAM: Final[int] = 200


# --- refusals ----------------------------------------------------------------


#: A closed list, for the same reason as `evidence.ERROR_CODES`: a new code
#: that simply appears cannot be looked up in the documentation, and two
#: different causes sharing one code can no longer be told apart when read
#: from the log.
RECIPE_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "RECIPE_LAYOUT_REQUIRED",
        "RECIPE_PARAM_REQUIRED",
        "RECIPE_PARAM_UNKNOWN",
        "RECIPE_PARAM_INVALID",
        "RECIPE_PARAM_TOO_MANY",
        "RECIPE_INPUT_TOO_LARGE",
        "RECIPE_NO_CONFIG",
        "RECIPE_WITHOUT_EVIDENCE",
        "RECIPE_WITHOUT_SCOPE",
        "RECIPE_DUPLICATE_NAME",
    }
)


class RecipeRefused(Exception):
    """An actionable refusal, not an infrastructure failure.

    Its shape mirrors `store.MeasureRefused` (`code`/`message`/`hint`) so that
    the `@app.exception_handler` pattern already present in `main.py` can be
    reused as is and the refusal comes out as a 400. It does not INHERIT that
    class because that class lives in `store.py`, and importing it would close
    the import cycle that this file's separation exists to break.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        if code not in RECIPE_ERROR_CODES:
            raise AssertionError(f"refusal code {code!r} is not registered")
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


# --- parameters --------------------------------------------------------------


#: The shapes a parameter can take. The list is deliberately short and
#: deliberately holds no "callable", "expression", or "code": every new shape
#: here is one step towards free code execution, and that step is refused at
#: the head of this file.
PARAM_KINDS: Final[tuple[str, ...]] = ("layers", "layer", "number", "text", "range", "flag")


@dataclass(frozen=True, slots=True)
class Param:
    """One recipe parameter: what it CHOOSES, not what its type is.

    `about` is written for the model that reads the catalogue. A docstring is
    the real interface of a tool, and the sentence "layer name" tells nobody
    WHICH layers make sense.
    """

    name: str
    kind: str
    about: str
    required: bool = False
    default: Any = None

    def __post_init__(self) -> None:
        if self.kind not in PARAM_KINDS:
            raise AssertionError(
                f"parameter {self.name!r} uses shape {self.kind!r}, which is "
                f"not registered; the valid ones: {', '.join(PARAM_KINDS)}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "about": self.about,
            "required": self.required,
            "default": self.default,
        }


def _refuse_invalid(name: str, value: Any, expected: str) -> RecipeRefused:
    return RecipeRefused(
        "RECIPE_PARAM_INVALID",
        f"parameter {name!r} has the value {value!r}, which is not {expected}.",
        "Recipe parameters are not silently forced into the right shape. "
        "A forced value produces an answer to a question different from the "
        "one the person typed, and nothing in the response shows that this "
        "happened.",
    )


def coerce(param: Param, value: Any) -> Any:
    """A cleaned value, or `RecipeRefused`. Never guesses."""
    if param.kind == "layers":
        if isinstance(value, str):
            items = [v.strip() for v in value.split(",")]
        elif isinstance(value, Sequence):
            items = [str(v).strip() for v in value]
        else:
            raise _refuse_invalid(param.name, value, "a list of layer names")
        items = [v for v in items if v]
        if not items:
            raise _refuse_invalid(param.name, value, "a non-empty list of layer names")
        if len(items) > MAX_LAYERS_PER_PARAM:
            raise RecipeRefused(
                "RECIPE_PARAM_TOO_MANY",
                f"{len(items)} layers in {param.name!r}, the limit is {MAX_LAYERS_PER_PARAM}.",
                "A list this long usually means 'every layer', and 'every "
                "layer' is a different question from the one this recipe "
                "answers. Narrow it, or leave it empty so it is derived from "
                "this drawing's land use config.",
            )
        # Order-preserving deduplication: a layer named twice would count its
        # parcels twice in every $in.
        return tuple(dict.fromkeys(items))

    if param.kind == "layer":
        if not isinstance(value, str) or not value.strip():
            raise _refuse_invalid(param.name, value, "a single layer name")
        return value.strip()

    if param.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise _refuse_invalid(param.name, value, "a number")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise _refuse_invalid(param.name, value, "a number") from None

    if param.kind == "text":
        if not isinstance(value, str):
            raise _refuse_invalid(param.name, value, "text")
        text = value.strip()
        if len(text) > MAX_TEXT_PARAM:
            raise RecipeRefused(
                "RECIPE_PARAM_TOO_MANY",
                f"{param.name!r} is {len(text)} characters long, the limit is "
                f"{MAX_TEXT_PARAM}.",
                "A recipe parameter is a choice, not prose. A value as long "
                "as a paragraph is a question that has not yet been turned "
                "into a recipe.",
            )
        return text

    if param.kind == "flag":
        if not isinstance(value, bool):
            raise _refuse_invalid(param.name, value, "true or false")
        return value

    # range
    if isinstance(value, Mapping):
        pair = (value.get("min"), value.get("max"))
    elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        pair = (value[0], value[1])
    else:
        raise _refuse_invalid(param.name, value, "a range [min, max]")
    try:
        lo, hi = float(pair[0]), float(pair[1])
    except (TypeError, ValueError):
        raise _refuse_invalid(param.name, value, "a range [min, max] of numbers") from None
    if lo > hi:
        raise _refuse_invalid(param.name, value, "a range with min <= max")
    return (lo, hi)


# --- recipes -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Recipe:
    """One named, parameterised, already-reviewed computation.

    `built_on` is not decoration. It is the list of store functions that are
    really called, published in every response, so that "where does this
    number come from" can be answered without reading a single line of this
    file — and so that a recipe that quietly rewrites geometry is visible from
    its own catalogue entry.
    """

    name: str
    answers: str
    when_to_use: str
    params: tuple[Param, ...]
    returns: tuple[str, ...]
    built_on: tuple[str, ...]
    limits: Mapping[str, Any]
    run: Callable[..., dict[str, Any]]
    needs_layout: bool = True
    #: True when the response says what something IS — then it must carry
    #: `evidence` (UPLIFT-08). False when it only measures — then it must
    #: carry `not_measured`, because a straight-line distance that does not
    #: say it is a straight line will be read as a walking distance.
    carries_meaning: bool = True

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.name,
            "answers": self.answers,
            "when_to_use": self.when_to_use,
            "needs_layout": self.needs_layout,
            "params": [p.as_dict() for p in self.params],
            "returns": list(self.returns),
            "built_on": list(self.built_on),
            "limits": dict(self.limits),
            "carries_meaning": self.carries_meaning,
        }


_REGISTRY: dict[str, Recipe] = {}


def register(recipe: Recipe) -> Recipe:
    """Register one recipe. Only called at module import, never from a request.

    There is no `unregister`, and that is deliberate: a registry that can be
    changed while running is free code execution with an extra step, and two
    concurrent requests would see different catalogues.
    """
    if recipe.name in _REGISTRY:
        raise RecipeRefused(
            "RECIPE_DUPLICATE_NAME",
            f"recipe {recipe.name!r} is already registered.",
            "Two recipes under one name means one of them will never be "
            "called, and there is nothing in the response that shows which "
            "one.",
        )
    _REGISTRY[recipe.name] = recipe
    return recipe


def known() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get(name: str) -> Recipe | None:
    return _REGISTRY.get(str(name or "").strip())


# --- catalogue ---------------------------------------------------------------


#: Honest limits, to be reported upwards. They travel in every catalogue, not
#: only in the document, so that an agent asked "what can you do" answers with
#: the SAME list a human reads.
HONEST_LIMITS: Final[tuple[dict[str, str], ...]] = (
    {
        "cannot": "write a new computation on the spot",
        "because": "deliberate — a drawing's contents are untrusted input and "
        "the shared database credentials live in the same process",
        "fixable": "no, and it should not be",
    },
    {
        "cannot": "read raw files beyond the ones already ingested",
        "because": "cad-api is the one that holds the files, deliberately",
        "fixable": "yes, through a new recipe",
    },
    {
        "cannot": "know what a typology code means",
        "because": "it is not in any file",
        "fixable": "only with a key from the drawing's owner",
    },
    {
        "cannot": "read Arabic text with certainty",
        "because": "the SHX font files are not included",
        "fixable": "only with the original font files",
    },
    {
        "cannot": "walking distance",
        "because": "the street network is not modelled as a graph yet",
        "fixable": "yes, a piece of work of its own",
    },
    {
        "cannot": "judge whether the design is good",
        "because": "not a data question",
        "fixable": "no",
    },
)


def catalog() -> dict[str, Any]:
    """Name, one sentence on when to use it, parameters, output shape.

    This catalogue is what the model reads — just as a docstring is the real
    interface of a tool. It is also what `run()` returns for an unknown recipe
    name, so that an agent that guessed the name wrong sees the right one
    straight away instead of receiving a failure that invites it to force
    another tool.
    """
    return {
        "contract_version": RECIPE_CONTRACT_VERSION,
        "recipes": [_REGISTRY[name].as_dict() for name in sorted(_REGISTRY)],
        "count": len(_REGISTRY),
        "how_to_add": (
            "a new recipe = a merge request. There is no way to add a "
            "computation while running, and that is what separates this "
            "registry from free code execution: a drawing's contents come "
            "from outside and the shared database credentials live in the "
            "same process."
        ),
        "no_recipe_sentence": NO_RECIPE_SENTENCE,
        "when_nothing_fits": (
            "do not invent a workaround and do not force the recipe that "
            "looks closest. Say `no_recipe_sentence` verbatim: name the "
            "computation that is missing, name the nearest one that DOES "
            "exist, then say that the computation can be added as a recipe."
        ),
        "honest_limits": [dict(row) for row in HONEST_LIMITS],
        "determinism": (
            "the same input produces the same output, forever: no clock, no "
            "randomness, no network calls. Two sessions asking the same thing "
            "about the same drawing must be able to compare their numbers."
        ),
    }


def unknown_recipe(name: str) -> dict[str, Any]:
    """The answer for an unknown recipe name: the catalogue, not a bare error."""
    return {
        "recipe_requested": name,
        "error": "RECIPE_UNKNOWN",
        "message": (
            f"there is no recipe named {name!r}. What exists: "
            + ", ".join(known())
            + "."
        ),
        "hint": (
            "Pick one from the catalogue below. If none of them answers the "
            "question, say so using `no_recipe_sentence` and do not force the "
            "closest match — a wrong recipe produces the wrong number in the "
            "right tone."
        ),
        **catalog(),
    }


# --- execution ---------------------------------------------------------------


def _bind(recipe: Recipe, params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validated parameters, plus the declared defaults.

    A foreign key is REFUSED, not ignored. A misspelled parameter that is
    silently dropped produces an answer over the default scope, and there is
    nothing in the response that says this was not what was asked for.
    """
    given = dict(params or {})
    accepted = {p.name for p in recipe.params}
    unknown = sorted(k for k in given if k not in accepted)
    if unknown:
        raise RecipeRefused(
            "RECIPE_PARAM_UNKNOWN",
            f"unknown parameter for recipe {recipe.name!r}: "
            f"{', '.join(unknown)}.",
            "Accepted: " + ", ".join(sorted(accepted)) + ". A misspelled "
            "parameter that is silently ignored produces an answer over a "
            "scope different from the one that was asked for.",
        )

    bound: dict[str, Any] = {}
    for p in recipe.params:
        if p.name not in given or given[p.name] is None:
            if p.required:
                raise RecipeRefused(
                    "RECIPE_PARAM_REQUIRED",
                    f"recipe {recipe.name!r} needs parameter {p.name!r}.",
                    p.about
                    + " It has no default because a default inside a `.py` is "
                    "one drawing's trait disguised as a property of every "
                    "drawing (G1).",
                )
            bound[p.name] = p.default
            continue
        bound[p.name] = coerce(p, given[p.name])
    return bound


def refuse_oversize(*, what: str, size: int, limit: int, hint: str) -> None:
    """A limit that is stated and that really binds (G7).

    Checked BEFORE the work is run, so that the limit is a working limit and
    not a report after the work has already been done.
    """
    if size > limit:
        raise RecipeRefused(
            "RECIPE_INPUT_TOO_LARGE",
            f"{what}: {size}, above the limit of {limit}.",
            hint,
        )


def envelope(
    recipe: Recipe,
    *,
    drawing_id: str,
    layout: str | None,
    params_used: Mapping[str, Any],
    body: Mapping[str, Any],
) -> dict[str, Any]:
    """The same envelope for every recipe, with the recipe's body inside it.

    Merged FLAT, not nested: `evidence.audit_response` and `evidence.forward`
    both read `evidence`, `scope_note`, and the quantities at the top level,
    and a response that hides them one level deeper passes the audit in a way
    nobody can see.
    """
    out: dict[str, Any] = {
        "recipe": recipe.name,
        "recipe_contract_version": RECIPE_CONTRACT_VERSION,
        "answers": recipe.answers,
        "drawing_id": drawing_id,
        "layout": layout,
        # The parameters that were REALLY used, after cleaning. Determinism
        # that cannot be checked is not determinism: two sessions can only
        # compare numbers if both know which input produced them.
        "params_used": {k: _plain(v) for k, v in sorted(params_used.items())},
        "deterministic": True,
        "built_on": list(recipe.built_on),
        "limits": dict(recipe.limits),
        "computed_by": (
            "a reviewed server-side recipe; the agent picks its name and "
            "fills in its parameters, and never writes its logic"
        ),
    }
    out.update(body)

    if not (out.get("scope_note") or "").strip():
        raise RecipeRefused(
            "RECIPE_WITHOUT_SCOPE",
            f"recipe {recipe.name!r} published without a `scope_note`.",
            "Name which layout, what falls inside the scope, and which units. "
            "A number without a scope cannot be checked by anyone.",
        )
    if recipe.carries_meaning:
        block = out.get("evidence")
        if not isinstance(block, Mapping) or "grade" not in block:
            raise RecipeRefused(
                "RECIPE_WITHOUT_EVIDENCE",
                f"recipe {recipe.name!r} says what something IS without `evidence`.",
                "Use the helpers in `app/evidence.py`. If this recipe really "
                "only measures, mark it `carries_meaning=False` and carry "
                "`not_measured` instead.",
            )
    elif not out.get("not_measured"):
        raise RecipeRefused(
            "RECIPE_WITHOUT_EVIDENCE",
            f"recipe {recipe.name!r} measures without naming what it did NOT measure.",
            "State the bounds of the measurement. A straight-line distance "
            "that does not say it is a straight line will be read as a "
            "walking distance, and that is a right answer used for the wrong "
            "thing.",
        )
    return out


def _plain(value: Any) -> Any:
    """A JSON-able shape, without changing its meaning."""
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    return value


def run(
    drawing_id: str,
    recipe: str,
    params: Mapping[str, Any] | None = None,
    *,
    layout: str | None = None,
    units: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one recipe. An unknown name returns the catalogue.

    `units` is handed over by the caller as it is, from `store._unit_names(
    drawing, layout)` — the same as `store_geometry.find_duplicates`. This
    module does not fetch it itself: `store.py` is shared property, and 11 of
    16 drawings are in inches while 3 declare no units at all, so units
    guessed here would be wrong for most drawings (G2).
    """
    found = get(recipe)
    if found is None:
        return unknown_recipe(str(recipe))

    if found.needs_layout and not (layout or "").strip():
        raise RecipeRefused(
            "RECIPE_LAYOUT_REQUIRED",
            f"recipe {found.name!r} needs `layout`, and has no default.",
            "Modelspace and paper sheets do not share a coordinate frame, so "
            "there is no right answer for 'all of it'. Name one layout; "
            "`Model` is the one usually meant.",
        )

    bound = _bind(found, params)
    body = found.run(
        drawing_id=drawing_id,
        layout=(layout or "").strip() or None,
        units=dict(units or {}),
        **bound,
    )
    return envelope(
        found,
        drawing_id=drawing_id,
        layout=(layout or "").strip() or None,
        params_used=bound,
        body=body,
    )


# --- shared evidence tooling -------------------------------------------------


def scope_for(
    *, layout: str | None, units: Mapping[str, Any], note: str
) -> ev.Scope:
    """`Scope` for a recipe, with the units the caller handed over.

    `includes_block_definitions` is derived from the layout and never guessed:
    a pseudo-layout `[block] …` does contain block definitions, and the
    geometry inside it is rescaled by every insertion of it.
    """
    from ..extract import BLOCK_LAYOUT_PREFIX

    return ev.Scope(
        layout=layout,
        includes_block_definitions=(
            True if layout is None else layout.startswith(BLOCK_LAYOUT_PREFIX)
        ),
        units=units,
        note=note,
    )


#: Every recipe computes; not one of them reads vocabulary. Its provenance is
#: therefore `NO_CONFIG` unless the recipe really does read the land use
#: config, and one that does hands over its own `Provenance`.
COMPUTED_PROVENANCE: Final[ev.Provenance] = ev.Provenance(
    config_layer=ev.ConfigLayer.NO_CONFIG,
    note="computed from the stored geometry and topology, not read from config",
)
