"""One automatic retry, applied wherever a recipe is run.

A recipe that settles nothing can publish the exact call that would settle it.
Executing that plan used to happen in the HTTP handler alone, which meant one
recipe gave two different answers depending on who asked. Measured: asked for
the parcels that are both size outliers and without road frontage,
`combine_findings` ran the frontage side in-process, received the un-healed
empty answer, intersected it with the outliers, and reported that no parcel is
both. The same frontage call over HTTP healed itself and found two. Two
callers, two answers, and the wrong one carrying no warning at all.

So the retry lives here, beside the runner, and every caller goes through it.

What it may and may not do:

  * ONE retry per run. Not one per request and not one per question: a result
    that already carries `auto_retry` is never healed again, so a composition
    of two sides runs at most two retries and cannot loop.
  * The recipe's OWN plan, taken whole. Nothing here composes a parameter, and
    a plan naming a different recipe is left alone -- swapping the recipe
    changes the question rather than answering it.
  * The verdict is demoted, never promoted. `usable_as_evidence` stays False,
    because nothing in an automatic re-run verified the choice it made. What
    the retry adds is `usable_for_composition`: the figures may be computed
    WITH, provided every answer built on them says so and names the thing to
    confirm. A number nobody may quote and nobody may compute with is the same
    as no number; a number quoted without its caveat is worse than none.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Final

#: How many times one run may retry itself. One. A second retry is a loop
#: waiting for a drawing that never satisfies it, and this runs inside a
#: request someone is waiting on.
MAX_AUTO_RETRIES: Final[int] = 1

#: The verdict a retried result carries. Named in one place because three
#: layers test for it and a typo would silently switch the check off.
RETRIED_VERDICT: Final[str] = "RETRIED, NOT VERIFIED"

#: A recipe runner: (drawing_id, recipe, params, layout=, units=) -> body.
Runner = Callable[..., dict[str, Any]]


def first_plan(result: Any) -> Mapping[str, Any] | None:
    """The result's own first retry plan, or None if it must not be re-run.

    None covers four different situations on purpose -- already retried, no
    verdict, a settled verdict, no plan -- because the caller does the same
    thing in all four: leave the answer exactly as the recipe wrote it.
    """
    if not isinstance(result, Mapping):
        return None
    if result.get("auto_retry") is not None:
        return None
    safety = result.get("conclusion_safety")
    if not isinstance(safety, Mapping) or safety.get("usable_as_evidence") is not False:
        return None
    plans = safety.get("retry_with")
    if not isinstance(plans, list) or not plans:
        return None
    plan = plans[0]
    if not isinstance(plan, Mapping) or not plan.get("recipe"):
        return None
    return plan


def heal(
    result: dict[str, Any],
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    run: Runner,
) -> dict[str, Any]:
    """Re-run once, in place, when a recipe settled nothing and said how.

    Why this is executed rather than described, measured over four attempts on
    one question -- which parcels are both size outliers and without frontage:

    1. With no marker at all, the agent intersected an empty result and
       reported "there are none". Confidently wrong.
    2. Marking the result unusable made it honest: "I cannot determine this".
    3. Told to act on the hint, it composed parameters of its own, narrowed an
       unrelated recipe to two layers it picked, and was wrong by a new route.
    4. Handed the exact call and told to run it verbatim, it stopped calling
       the recipes at all -- by then the instruction was 35,000 characters and
       each addition made behaviour less predictable rather than more.

    An instruction is not a control surface. This is: a stated rule, executed
    the same way every time, bounded at one retry, with BOTH results returned
    so nothing is hidden behind the automation.
    """
    plan = first_plan(result)
    if plan is None:
        return result
    first_safety = result.get("conclusion_safety")
    first_safety = first_safety if isinstance(first_safety, Mapping) else {}

    try:
        again = run(
            drawing_id,
            str(plan["recipe"]),
            dict(plan.get("params") or {}),
            layout=layout,
            units=dict(units),
        )
    except Exception as exc:  # noqa: BLE001 -- the first answer still stands
        result["auto_retry"] = {
            "attempted": True,
            "succeeded": False,
            "why": f"{type(exc).__name__}: {exc}",
            "plan": dict(plan),
        }
        return result

    again_safety = again.get("conclusion_safety")
    settled = (
        isinstance(again_safety, Mapping)
        and again_safety.get("usable_as_evidence") is True
    )
    # The retry is returned BESIDE the first answer, never in place of it, and
    # its verdict is demoted on the way through.
    #
    # Measured the first time this ran unrestrained: the candidate ranked
    # highest was the layer holding the SITE BOUNDARY, because ranking was by
    # area and a site boundary is the largest closed ring there is. Every
    # parcel lies inside it, so every parcel "had frontage", and the answer
    # came back stamped ESTABLISHED. A wrong answer with a confident label is
    # worse than the honest refusal it replaced.
    #
    # Ranking is no longer by size -- a plan argues for its parameters from
    # geometry and publishes the argument -- so the demotion no longer claims
    # to know HOW the plan chose. It quotes the plan's own reason instead, and
    # the reason is the thing to confirm.
    if settled and isinstance(again_safety, dict):
        again_safety["usable_as_evidence"] = False
        again_safety["usable_for_composition"] = True
        again_safety["verdict"] = RETRIED_VERDICT
        again_safety["why"] = (
            "this result came from an automatic re-run with the parameters the "
            "first attempt's own response named. Nothing has confirmed that "
            "choice. Read it as one candidate reading: name the parameters "
            "whenever you quote a figure from it, and say it was retried. The "
            "plan's own reason for choosing them: "
            + str(plan.get("because") or "none was published")
            + " | original verdict: "
            + str(first_safety.get("verdict"))
        )
    again["auto_retry"] = {
        "attempted": True,
        "succeeded": settled,
        "retries_allowed": MAX_AUTO_RETRIES,
        "plan": dict(plan),
        "because": plan.get("because"),
        "note": (
            "the call as first made established nothing, so it was re-run ONCE "
            "with the parameters its own response named. Nothing has verified "
            "those parameters: this is a candidate reading, not a settled "
            "answer. Name them whenever you quote a figure from it, and say it "
            "was retried"
        ),
        "chosen_by": plan.get("because"),
        "not_verified": (
            "that the parameters the plan chose are the right ones for this "
            "question. The plan argues for them from the drawing's geometry and "
            "publishes the argument above; no one has looked at the drawing to "
            "check it"
        ),
        "first_attempt": {
            "params": dict(result.get("params_used") or {}),
            "verdict": first_safety.get("verdict"),
            "why": first_safety.get("why"),
        },
    }
    return again


def was_retried(result: Any) -> bool:
    """Did this body come back from a retry that settled its question?"""
    if not isinstance(result, Mapping):
        return False
    auto = result.get("auto_retry")
    return isinstance(auto, Mapping) and auto.get("succeeded") is True


def composable(result: Any) -> bool:
    """May this body's answer set be computed with, retried or not?

    Only an EXPLICIT refusal refuses. A recipe that publishes no
    `conclusion_safety` at all has not said its answer is unusable -- most do
    not publish one -- and reading silence as a refusal would quietly empty
    every composition built on them. Measured: written the other way round,
    this marked `outliers` unsettled on a body that had simply never carried a
    verdict.

    So the one thing this adds to the old rule is the second line: a side that
    settled nothing and then healed itself is composable again, because the
    retry gave it a real answer set over a scope the caller must name.
    """
    if not isinstance(result, Mapping):
        return False
    safety = result.get("conclusion_safety")
    if not isinstance(safety, Mapping):
        return True
    if safety.get("usable_as_evidence") is False:
        return safety.get("usable_for_composition") is True
    return True
