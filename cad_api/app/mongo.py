"""The only door to MongoDB.

`roshn-gpt-dev-main` is shared with other teams. It holds `boq_*`,
`conversation-adk`, `document-dim`, `model_registry` and more; damaging any of
them damages someone else's work.

The guardrail is a name prefix, enforced here rather than by discipline:
**every collection this project touches must be named `autocad_*`**. Nothing
outside this module is allowed to call `client[db][name]`, so there is exactly
one place where the rule can be broken, and it raises.

The database name is hard-coded on purpose. `MONGODB_DB_NAME` from the shared
`.env` belongs to another project and may point somewhere else entirely; it is
read only to be compared, never to be obeyed.
"""

from __future__ import annotations

import logging
import os
from typing import Final

from pymongo import ASCENDING, DESCENDING, MongoClient, TEXT
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

log = logging.getLogger(__name__)

DB_NAME: Final[str] = "roshn-gpt-dev-main"
PREFIX: Final[str] = "autocad_"

COLL_DRAWINGS: Final[str] = "autocad_drawings"
COLL_ENTITIES: Final[str] = "autocad_entities"
COLL_COMMENTS: Final[str] = "autocad_comments"
COLL_RENDERS: Final[str] = "autocad_renders"
#: Saved region selections, referenced by id instead of being passed around.
#: Short-lived by design -- see the TTL index in `ensure_indexes`.
COLL_SELECTIONS: Final[str] = "autocad_selections"

#: How long a saved selection survives. Long enough to outlive any
#: conversation about it, short enough that nothing accumulates: a selection
#: is scratch state, and the agent only ever reads it during the session that
#: created it.
SELECTION_TTL_SECONDS: Final[int] = 24 * 60 * 60

#: Every collection this project is allowed to create. Anything else is a bug.
#: The business names people give a layer. Registered here rather than let
#: through on a prefix check alone: `baseline diff` treats an unregistered
#: collection as a surprise, and a collection that appears without ever having
#: been named is exactly the kind of surprise this fence prevents.
COLL_TAGS: Final[str] = "autocad_tags"

#: The complete account of what one drawing contains -- one document per
#: drawing, keyed by `drawing_id`. Because that id is a content hash, a new
#: version of a file gets its own Dossier and a re-run of the backfill
#: overwrites in place, so versioning needs no machinery of its own.
COLL_DOSSIERS: Final[str] = "autocad_dossiers"

#: One document per ingestion ATTEMPT, keyed by a run id rather than by
#: `drawing_id`: the same drawing can be ingested more than once, and a
#: refused upload has no drawing at all yet still has to leave a record.
#: Written stage by stage while the pipeline runs, so a browser refresh
#: re-reads the same truth instead of restarting a guess, and a refusal is
#: an entry here rather than a message that scrolls away.
COLL_INGEST_RUNS: Final[str] = "autocad_ingest_runs"

OWNED_COLLECTIONS: Final[tuple[str, ...]] = (
    COLL_DRAWINGS,
    COLL_ENTITIES,
    COLL_COMMENTS,
    COLL_RENDERS,
    COLL_SELECTIONS,
    COLL_TAGS,
    COLL_DOSSIERS,
    COLL_INGEST_RUNS,
)

_client: MongoClient | None = None


class ForbiddenCollectionError(RuntimeError):
    """Raised when code asks for a collection outside the `autocad_` prefix."""


def get_client() -> MongoClient:
    """Process-wide MongoClient, built from `MONGODB_URI`.

    The URI is read from the environment and never logged, echoed, or copied.

    Raises:
        RuntimeError: if `MONGODB_URI` is not set.
    """
    global _client
    if _client is not None:
        return _client

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not set. cad-api cannot start without a database."
        )

    configured_db = os.environ.get("MONGODB_DB_NAME")
    if configured_db and configured_db != DB_NAME:
        # Not fatal: we simply refuse to follow it. Said out loud so nobody
        # later wonders why the variable appears to be ignored.
        log.warning(
            "MONGODB_DB_NAME is %r but this service is pinned to %r; "
            "the environment value is ignored by design",
            configured_db,
            DB_NAME,
        )

    _client = MongoClient(
        uri,
        appname="cad-api",
        serverSelectionTimeoutMS=int(os.environ.get("MONGO_TIMEOUT_MS", "8000")),
        # A remote cluster makes latency real, and a socket with no deadline
        # turns one lost packet into a hang with no error to read. Generous
        # rather than tight: a bulk write of 1,000 documents across the
        # network is normal work, and a timeout that fires on normal work is
        # worse than none. Both are env-tunable so a slower link is a setting
        # rather than an edit.
        connectTimeoutMS=int(os.environ.get("MONGO_CONNECT_TIMEOUT_MS", "20000")),
        socketTimeoutMS=int(os.environ.get("MONGO_SOCKET_TIMEOUT_MS", "120000")),
        # Left ON, and stated here because it is easy to lose in a URI: a
        # retryable write is what makes a transient network blip during a
        # 990,790-row ingest a pause rather than a failed run.
        retryWrites=True,
        tz_aware=True,
    )
    return _client


#: Hosts that can only be this machine. Anything else is treated as the
#: shared company cluster, whatever it is called.
LOCAL_HOSTS: Final[frozenset[str]] = frozenset(
    {"localhost", "127.0.0.1", "mongo", "mongo-local", "host.docker.internal"}
)


def target_hosts() -> str:
    """The host(s) `MONGODB_URI` points at, with credentials stripped.

    The local database and the company cluster carry the SAME database name
    (`roshn-gpt-dev-main`) by design, so the name cannot tell them apart. The
    host can. Parsed by hand rather than with `pymongo.uri_parser` because
    that resolves `mongodb+srv://` over DNS, which is exactly what fails on
    the connection this function exists to diagnose.

    Never returns the user or password: everything before `@` is discarded
    before anything is returned, so this is safe to log and safe to serve.
    """
    uri = os.environ.get("MONGODB_URI", "")
    if not uri:
        return "(unset)"
    body = uri.split("://", 1)[-1]
    body = body.rsplit("@", 1)[-1]
    return body.split("/", 1)[0].split("?", 1)[0] or "(unset)"


def target_is_local() -> bool:
    """True when every host in `MONGODB_URI` is this machine.

    False means writes land in the shared cluster. Callers that destroy data
    must gate on this; `/ready` reports it so the answer is one curl away.
    """
    hosts = [h for h in target_hosts().split(",") if h]
    if not hosts:
        return False
    return all(h.split(":", 1)[0] in LOCAL_HOSTS for h in hosts)


def get_database() -> Database:
    """The pinned database handle. Prefer `coll()` over using this directly."""
    return get_client()[DB_NAME]


def coll(name: str) -> Collection:
    """Return a collection, refusing any name outside the `autocad_` prefix.

    This is the single choke point for database access. An assert would be
    stripped under `python -O`; a raised exception is not.

    Args:
        name: full collection name, e.g. ``"autocad_entities"``.

    Raises:
        ForbiddenCollectionError: if `name` does not start with `autocad_`.
    """
    if not name.startswith(PREFIX):
        raise ForbiddenCollectionError(
            f"refusing to touch collection {name!r}: this project may only "
            f"access collections prefixed {PREFIX!r} in the shared database "
            f"{DB_NAME!r}"
        )
    return get_database()[name]


def list_our_collections() -> list[str]:
    """Names of `autocad_*` collections that currently exist.

    Deliberately filtered: the baseline snapshot needs the full list once (see
    `list_all_collection_names`), but routine code has no business enumerating
    other teams' collections.
    """
    return sorted(
        name
        for name in get_database().list_collection_names()
        if name.startswith(PREFIX)
    )


def list_all_collection_names() -> list[str]:
    """Every collection name in the shared database.

    Used only to write `docs/MONGO-BASELINE.md`, so that "we changed nothing
    outside our prefix" is a checkable claim rather than an assurance. Names
    only -- no document from a foreign collection is ever read.
    """
    return sorted(get_database().list_collection_names())


def ensure_indexes() -> list[str]:
    """Create the indexes this project needs. Idempotent.

    Returns the names of the indexes that exist afterwards.

    Note on spatial indexing: the schema sketch in `cad-graph-model` suggests a
    `2dsphere` index on a GeoJSON `geom` field, but `2dsphere` interprets
    coordinates as longitude/latitude on a globe. These drawings are in local
    CAD coordinates (Janadriyah spans hundreds of metres, the samples are in
    inches), so a `2dsphere` index would be silently wrong -- it rejects any
    |x| > 180. We index the plain bbox numbers instead and do rectangle
    filtering in the query. See docs/DECISIONS-LOG.md D-004.
    """
    created: list[str] = []

    entities = coll(COLL_ENTITIES)
    created.append(entities.create_index([("drawing_id", ASCENDING), ("layer", ASCENDING)]))
    created.append(entities.create_index([("drawing_id", ASCENDING), ("type", ASCENDING)]))
    created.append(
        entities.create_index([("drawing_id", ASCENDING), ("block_name", ASCENDING)])
    )
    created.append(
        entities.create_index([("drawing_id", ASCENDING), ("layout", ASCENDING)])
    )
    # Rectangle pre-filter for spatial queries.
    created.append(
        entities.create_index(
            [
                ("drawing_id", ASCENDING),
                ("bbox.min.0", ASCENDING),
                ("bbox.min.1", ASCENDING),
            ],
            name="drawing_bbox_min",
        )
    )
    # A compound {drawing_id, layout, layer, type} index was added here and
    # taken out again the same night. Both judges of the design panel called
    # it the thing all three proposals had missed, and on the numbers they
    # were right -- it is the index the measurement filter actually wants.
    # On this cluster the build never completed: it left the index listed but
    # unusable, and every later create_index on the collection blocked behind
    # it, which hung cad-api's startup because ensure_indexes runs there. A
    # shared cluster is the wrong place to discover that, and measurement is
    # fast enough without it (the reference aggregation returns in ~270 ms on
    # the two-field indexes). Recorded in TECH-DEBT.md rather than retried.
    # What the map feed actually asks for: entities in model scope, of a
    # geometry-bearing type. Scope is an `$or` of `layout` and
    # `h3_world_placed` (see `geo_h3.scope_match`), and the second half had no
    # index, so neither branch could be served from one and the query fell
    # back to fetching documents to test them. Measured on a 990,790 row
    # drawing: finding 500 rings took 4.93 s without this index and 0.80 s
    # with it.
    #
    # Built in 2.8 s locally. The caution above about the withdrawn compound
    # index applies to the shared cluster, not to this one: check the build
    # completes before relying on it there.
    created.append(
        entities.create_index(
            [
                ("drawing_id", ASCENDING),
                ("h3_world_placed", ASCENDING),
                ("type", ASCENDING),
            ],
            name="drawing_scope_type",
        )
    )

    # The viewport query: which entities fall in the coarse cells covering
    # what is on screen. Without this the map fetches an arbitrary page in
    # storage order, which on a large drawing is a spatial corner rather than
    # the view, and every layer toggle but the biggest one appears to do
    # nothing because none of its objects were fetched.
    created.append(
        entities.create_index(
            [("drawing_id", ASCENDING), ("h3_coarse", ASCENDING)],
            name="drawing_viewport_cell",
        )
    )

    created.append(entities.create_index([("text", TEXT)], name="entity_text"))

    comments = coll(COLL_COMMENTS)
    created.append(
        comments.create_index(
            [("drawing_id", ASCENDING), ("entity_handle", ASCENDING)]
        )
    )
    # Idempotency key for agent retries: a repeated add_comment with the same
    # client_request_id must not create a second comment.
    created.append(
        comments.create_index(
            [("client_request_id", ASCENDING)],
            unique=True,
            partialFilterExpression={"client_request_id": {"$type": "string"}},
            name="comment_idempotency",
        )
    )

    # Selections expire on their own. They are a working artefact of one
    # viewing session -- a handle list a user drew a box around -- not a
    # record anyone should keep, and an unbounded collection of them on a
    # database shared with other teams is exactly the kind of litter nobody
    # notices until it is large. MongoDB removes them; no cron, no cleanup
    # script that can be forgotten.
    selections = coll(COLL_SELECTIONS)
    created.append(
        selections.create_index(
            [("created_at", ASCENDING)],
            expireAfterSeconds=SELECTION_TTL_SECONDS,
            name="selection_ttl",
        )
    )
    created.append(
        selections.create_index([("drawing_id", ASCENDING)], name="selection_drawing")
    )

    renders = coll(COLL_RENDERS)
    created.append(
        renders.create_index(
            [("drawing_id", ASCENDING), ("layout", ASCENDING)],
            unique=True,
            name="render_key",
        )
    )

    # Ingest runs are read two ways and only two ways: newest-first overall,
    # and newest-first for one drawing. Both are served by this one compound
    # index, because a query with no `drawing_id` can still use it to sort.
    # Safe to build even on the shared cluster (unlike the entity index
    # withdrawn above): the collection is new, so there is nothing to scan.
    ingest_runs = coll(COLL_INGEST_RUNS)
    created.append(
        ingest_runs.create_index(
            [("drawing_id", ASCENDING), ("started_at", DESCENDING)],
            name="ingest_run_recency",
        )
    )

    # The REAPER's query: runs still marked running whose last update is
    # older than the cutoff. It had no index, and on Atlas that is not a slow
    # sweep, it is a refusal -- so the one mechanism whose whole job is to
    # declare a dead ingest dead could not run on the production cluster at
    # all. Found by running the suite against a local Mongo with
    # `notablescan` switched on, which is the only way this class of bug is
    # visible before it reaches Atlas.
    created.append(
        ingest_runs.create_index(
            [("overall", ASCENDING), ("updated_at", ASCENDING)],
            name="ingest_run_reaper",
        )
    )

    # The drawing PICKER's own query: every drawing that finished ingesting,
    # newest first. It had no index, which on local Mongo means a scan of 22
    # documents and on Atlas means a REFUSAL -- the cluster runs with
    # `notablescan`, so an unindexed query does not return slowly, it returns
    # `NoQueryExecutionPlans`. Measured after the Phase B cutover: `/drawings`
    # answered 503 and the picker came up empty, on a database that held every
    # drawing correctly. Local could never have shown this.
    #
    # `ingest_incomplete` leads because the filter is on it and the sort
    # follows; a sort-only index cannot serve the filter, and the reverse
    # cannot serve the sort.
    drawings = coll(COLL_DRAWINGS)
    created.append(
        drawings.create_index(
            [("ingest_incomplete", ASCENDING), ("ingested_at", DESCENDING)],
            name="drawing_picker",
        )
    )

    log.info("ensured indexes", extra={"count": len(created)})
    return created


def ping() -> bool:
    """True if the cluster answers. Used by `/ready`."""
    try:
        get_client().admin.command("ping")
        return True
    except PyMongoError as exc:
        log.warning("mongo ping failed: %s", type(exc).__name__)
        return False


def close() -> None:
    """Drop the client. Called on application shutdown."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
