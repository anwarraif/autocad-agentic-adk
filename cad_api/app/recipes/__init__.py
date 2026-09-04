"""The analysis lane — so that unforeseen questions still get answered.

Owner: subagent R — UPLIFT-09.

The original request: *make this agent able to do anything a human analysis
session can do.* Before designing, one thing was said honestly first — what is
it that actually makes such a session able to answer? Three things, and not
one of them is the model's intelligence:

1.  **Full access to the file, not to a summary.** Solved by `UPLIFT-01`:
    geometry is now stored, not discarded at ingest.
2.  **One general computation step.** Mostly solved by `UPLIFT-02…08` for the
    questions that are ALREADY known. The rest — the questions nobody has
    thought of yet — is handled by this package.
3.  **Verification against ground truth computed separately.** That is what
    `cad_api/tests/test_recipes.py` does: every recipe carries at least one
    Janadriyah number computed outside the path it tests.

What is **not** done, and the reason is at the head of `registry.py`: there is
no `run_python(code)` and no new write tool. A drawing's contents come from
contractors, the shared database credentials live in the same process, and
answers from one-off code cannot be compared across sessions. New recipes are
added through **review**, not through the runtime — and that is the only thing
that separates this package from free code execution.

Two functions are used from outside:

```python
from .recipes import catalog, run

catalog()                                    # what exists, and its limits
run(drawing_id, "coverage_gap", {...},        # one named computation
    layout="Model", units=store._unit_names(drawing, "Model"))
```

`units` is handed over by the caller as it is, the same as
`store_geometry.find_duplicates`: 11 of 16 drawings are in inches and 3
declare no units at all, so units guessed in here would be wrong for most
drawings (G2).
"""

from __future__ import annotations

from .registry import (  # noqa: F401
    HONEST_LIMITS,
    MAX_LAYERS_PER_PARAM,
    MAX_ROWS_RETURNED,
    NO_RECIPE_SENTENCE,
    RECIPE_CONTRACT_VERSION,
    RECIPE_ERROR_CODES,
    Param,
    Recipe,
    RecipeRefused,
    catalog,
    get,
    known,
    register,
    run,
    unknown_recipe,
)

# Imported for their side effect: each module registers its recipes when it is
# loaded. Placed AFTER the re-export so that they can import from `.registry`
# without waiting for this file to finish, and written here — not left to the
# caller — so that `catalog()` never returns an empty list just because someone
# forgot to import a module.
#
# One line per module, and the split is per SUBJECT rather than per size:
# `library` holds the eight recipes that answer "what is in this drawing",
# `topology` the two that ask whether its parcels fit together, and `truth`
# the two that ask whether the drawing agrees with itself. A recipe added to
# the wrong file still works and still becomes harder to find, so the naming
# is the whole discipline here.
from . import library  # noqa: E402,F401
from . import topology  # noqa: E402,F401
from . import truth  # noqa: E402,F401
from . import combine  # noqa: E402,F401
from . import junctions  # noqa: E402,F401
from . import revision  # noqa: E402,F401
from . import vicinity  # noqa: E402,F401
from . import hierarchy  # noqa: E402,F401

__all__ = [
    "HONEST_LIMITS",
    "NO_RECIPE_SENTENCE",
    "RECIPE_CONTRACT_VERSION",
    "RECIPE_ERROR_CODES",
    "Param",
    "Recipe",
    "RecipeRefused",
    "catalog",
    "get",
    "known",
    "run",
    "unknown_recipe",
]
