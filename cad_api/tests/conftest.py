"""Shared facts about the corpus these tests run against.

Several invariants here are of the form "a drawing with no coordinate system
must refuse rather than answer". For a long time exactly one drawing had a
CRS config, so "every drawing except Janadriyah" and "every drawing without a
config" were the same set, and the tests named the drawing.

They are not the same set any more: Sedra ships a CRS config too, deliberately.
A test that still says "everything except Janadriyah" now asserts something
false about a drawing that was given a config on purpose, and it would break
again for the drawing after that.

So the set is DERIVED from the config tree. Adding a config for a new drawing
is then a deliberate act these tests follow, rather than a change that breaks
them for a reason that looks unrelated.
"""

from __future__ import annotations


import pytest

from app import landuse, store


def configured_drawing_ids() -> set[str]:
    """Drawing ids that ship a per-drawing config, asked of the code that owns it.

    A previous version globbed the image tree here. It was correct on the day
    it was written and wrong three commits later, when the loader learned to
    read a writable overlay first: the accepted config for a freshly uploaded
    drawing lived in a directory this helper had never heard of, and eight
    guards fired at behaviour that was right. The lesson is the NO SECOND
    OPINIONS rule -- `landuse` decides where configs live, so `landuse` is
    asked, and a third location someday will move both answers at once.
    """
    return landuse.configured_drawing_ids()


def all_drawing_ids() -> list[str]:
    """Every ingested drawing, or nothing when there is no database.

    Collected without raising: a parametrize that throws takes the whole
    module down and the other tests stop running for a reason nobody would
    guess.
    """
    try:
        return [d["_id"] for d in store.list_drawings()]
    except Exception:  # pragma: no cover - depends on the environment
        return []


def unconfigured_drawing_ids() -> list[str]:
    """Ingested drawings that have NO per-drawing config.

    The set these "must refuse" invariants are actually about.
    """
    configured = configured_drawing_ids()
    return [d for d in all_drawing_ids() if d not in configured]


@pytest.fixture
def unconfigured_drawing() -> str:
    """One drawing with no config, for a test that needs a single example."""
    ids = unconfigured_drawing_ids()
    if not ids:
        pytest.skip("no unconfigured drawing in the store")
    return ids[0]


def pytest_configure(config: pytest.Config) -> None:
    """Refuse to run the suite against anything but a local database.

    Section 5 of docs/INTAKE-TO-AGENT-PLAN.md calls this the likeliest way to
    damage the shared cluster, precisely because nobody would intend it: the
    moment the environment carries the Atlas URI, anything that runs tests
    against a real database writes into a cluster that other teams' live
    services share, and fixtures here create and delete freely by design.

    It is not hypothetical. After the Phase B cutover `.env` holds the Atlas
    URI, so `docker compose run cad-api python -m pytest` -- the obvious
    command, and the one this project's own notes used before -- hands the
    suite the company cluster. Proving the URI by hand before each run is a
    habit; this is a lock.

    `mongo.target_is_local` is asked rather than the string being re-parsed
    here, so a third opinion about what "local" means cannot appear.
    """
    from app import mongo  # noqa: PLC0415 -- deferred, see the module header

    if mongo.target_is_local():
        return
    raise pytest.UsageError(
        "REFUSING TO RUN: MONGODB_URI does not point at a local database, and "
        "this suite creates and deletes collections. Point it at local Mongo "
        "-- `-e MONGODB_URI=mongodb://host.docker.internal:27017` on a "
        "`docker run`, which is not the same as `docker compose run`, because "
        "compose loads .env and .env now carries Atlas."
    )


@pytest.fixture(autouse=True)
def _clear_request_caches():
    """Empty the per-version caches between tests.

    `geometry_map` remembers coverage, totals and layer lists keyed by a token
    built from the drawing's version facts. That is right in production, where
    those facts change whenever the data does. It is wrong ACROSS TESTS, which
    reuse a drawing id and layout over completely different fixture
    collections and would otherwise be served the previous test's answer --
    which is exactly what happened: a coverage assertion of 0 came back as 2.

    Autouse, because a cache that has to be remembered about is a cache that
    will be forgotten about.
    """
    from app import geometry_map

    for cache in (geometry_map._COVERAGE, geometry_map._TOTALS, geometry_map._LAYERS):
        cache.clear()
    yield
    for cache in (geometry_map._COVERAGE, geometry_map._TOTALS, geometry_map._LAYERS):
        cache.clear()
