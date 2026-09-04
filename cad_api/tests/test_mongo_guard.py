"""The MongoDB prefix guard.

`roshn-gpt-dev-main` is shared with other teams. `coll()` is the only door to
it, and these tests are what stop that door being widened by accident. They
need no database: the guard must reject a forbidden name before any
connection is attempted.
"""

from __future__ import annotations

import pytest

from app import mongo


def test_database_name_is_pinned_in_code():
    """The database name must not be configurable.

    If this ever becomes an environment lookup, a stray MONGODB_DB_NAME could
    point writes at another team's database.
    """
    assert mongo.DB_NAME == "roshn-gpt-dev-main"
    assert mongo.PREFIX == "autocad_"


@pytest.mark.parametrize(
    "name",
    [
        "boq_runs",
        "conversation-adk",
        "document-dim",
        "model_registry",
        "appsettings",
        "users",
        "",
        "autocad",          # prefix without the underscore
        "AUTOCAD_drawings",  # wrong case
        "x_autocad_drawings",  # prefix not at the start
        " autocad_drawings",   # leading space
    ],
)
def test_coll_refuses_foreign_collections(name):
    """Any name outside the prefix raises, before touching the network."""
    with pytest.raises(mongo.ForbiddenCollectionError) as excinfo:
        mongo.coll(name)
    # The message must name the offending collection, or a 3am debugging
    # session starts from nothing.
    assert repr(name) in str(excinfo.value)


def test_guard_raises_rather_than_asserts():
    """`assert` is stripped under `python -O`; this guard must not be.

    Verified by checking the raised type is a real exception class rather than
    AssertionError.
    """
    with pytest.raises(mongo.ForbiddenCollectionError):
        mongo.coll("boq_core")
    assert not issubclass(mongo.ForbiddenCollectionError, AssertionError)


def test_owned_collections_all_carry_the_prefix():
    """Every collection this project declares must be inside the fence."""
    assert mongo.OWNED_COLLECTIONS, "the owned list must not be empty"
    for name in mongo.OWNED_COLLECTIONS:
        assert name.startswith(mongo.PREFIX), name


def test_no_module_bypasses_the_door():
    """No module may index a database directly.

    The guard is worthless if some other file reaches around it, so this
    greps the source for the bypass pattern rather than trusting review.
    """
    import pathlib
    import re

    app_dir = pathlib.Path(mongo.__file__).parent
    # `client[db][name]` / `get_database()[name]` style indexing.
    bypass = re.compile(r"(get_database\(\)|_client|client)\s*\[[^\]]+\]\s*\[")

    offenders: list[str] = []
    for path in app_dir.glob("*.py"):
        if path.name == "mongo.py":
            continue  # the door itself is allowed to open
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if bypass.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")

    assert not offenders, "these lines bypass mongo.coll():\n" + "\n".join(offenders)


def test_selections_collection_is_inside_the_fence():
    """The selection store is new; it must obey the same prefix rule.

    A selection is scratch state written on every region draw, so it is
    exactly the kind of collection that could have been named `selections`
    in a hurry -- on a database shared with other teams.
    """
    from app import mongo as m

    assert m.COLL_SELECTIONS == "autocad_selections"
    assert m.COLL_SELECTIONS in m.OWNED_COLLECTIONS
    assert m.COLL_SELECTIONS.startswith(m.PREFIX)


def test_selections_expire_rather_than_accumulate():
    """Scratch state must clean itself up.

    Without a TTL, every box a user ever drags leaves a document behind for
    ever. The number is checked too: a selection has to outlive the
    conversation about it (a working day) without becoming a record.
    """
    from app import mongo as m

    assert 60 * 60 <= m.SELECTION_TTL_SECONDS <= 7 * 24 * 60 * 60
