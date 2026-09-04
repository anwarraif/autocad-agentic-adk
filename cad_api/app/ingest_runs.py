"""Stage-by-stage history of one ingestion attempt.

Ingesting a large drawing takes minutes. Until now the only way to watch one
was the CLI's stdout, so a browser had nothing to show but a spinner and a
refresh lost even that. The owner's rule for this feature is that the user
must never stare at a spinner: every stage is written to MongoDB AS IT
HAPPENS, so a refresh re-reads the same truth rather than restarting a guess.

Two things this module deliberately is not.

**It is not an orchestrator.** `ingest_file` and `profile_on_arrival` keep
their control flow exactly as they had it. A `Recorder` is handed to them and
they report into it; nothing here decides what runs or in what order. That is
why the stage list below is a transcript of the existing pipeline rather than
a plan for one, and why every method is safe to call out of order.

**It is not allowed to break an ingest.** A recorder that raised would turn a
history feature into a new way to fail an ingestion that would otherwise have
succeeded, which is a strictly worse trade than losing a progress line. Every
write is therefore best-effort: failures are logged and swallowed. The one
exception is `create`, which the HTTP route awaits before answering, because
a run the caller cannot poll is worse than an honest error.

The document shape is fixed by `docs/INGEST-RUN-CONTRACT.md`.
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Final, Iterable, Mapping
from uuid import uuid4

from pymongo.errors import PyMongoError

from .mongo import COLL_INGEST_RUNS, coll

#: How much of a traceback rides on the run document. Enough to name the
#: frames that matter, far short of anything that could bloat the document.
_TRACEBACK_CHARS: Final[int] = 4000

log = logging.getLogger(__name__)

#: The pipeline, in the order it actually runs. `upload` is first because the
#: bytes have to arrive and be hashed before anything else can be decided;
#: `done` is a terminal marker rather than work, so the UI has something to
#: show as complete instead of inferring completeness from absence.
STAGES: Final[tuple[str, ...]] = (
    "upload",
    "convert",
    "extract",
    "store",
    "render",
    "dossier",
    "landuse_draft",
    "h3",
    "done",
)

#: Stage states. `skipped` is not a failure and not a success: a `.dxf` skips
#: conversion, and a drawing already stored and current skips everything after
#: extract. Collapsing either into `done` would make the history lie.
PENDING: Final[str] = "pending"
RUNNING: Final[str] = "running"
DONE: Final[str] = "done"
FAILED: Final[str] = "failed"
SKIPPED: Final[str] = "skipped"

#: Overall states.
OVERALL_RUNNING: Final[str] = "running"
OVERALL_DONE: Final[str] = "done"
OVERALL_FAILED: Final[str] = "failed"
OVERALL_REFUSED: Final[str] = "refused"
OVERALL_SKIPPED: Final[str] = "skipped"

TERMINAL_OVERALL: Final[frozenset[str]] = frozenset(
    {OVERALL_DONE, OVERALL_FAILED, OVERALL_REFUSED, OVERALL_SKIPPED}
)

#: How long a `running` run may go without writing before the UI is entitled
#: to say it has stopped.
#:
#: This is a measure of SILENCE, not of health, and the threshold has to clear
#: the longest single stage a real drawing produces or it accuses healthy work
#: of hanging. It was 180 s, chosen from Janadriyah's ~178 s render, and Sedra
#: immediately proved that too small: its extract stage is one atomic
#: `ezdxf.recover` call over 500 MB that ran for 24 minutes and wrote nothing
#: while it worked, so the run was flagged as stalled for twenty of them.
#:
#: 1800 s is that measured worst case with margin. The honest limit of the
#: whole idea is worth stating: a stage that cannot report progress from
#: inside can only ever be timed, so this distinguishes "silent for a long
#: time" from "silent for longer than any drawing we have measured", and
#: nothing more. Making it sharper needs a heartbeat inside extract, which is
#: a change to the extractor rather than to this file.
STALL_AFTER_S: Final[int] = 1800


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    """A run id. Not the drawing id: one drawing has many attempts."""
    return uuid4().hex[:16]


def _blank_stages() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "status": PENDING,
            "started_at": None,
            "finished_at": None,
            "detail": {},
        }
        for name in STAGES
    ]


def create(
    *,
    filename: str,
    source: str,
    drawing_id: str | None = None,
    bytes_received: int | None = None,
    run_id: str | None = None,
) -> str:
    """Open a run document and return its id.

    Unlike every other write here this one is allowed to raise: the HTTP
    route hands the id straight back to the caller, and a run that cannot be
    polled is worse than an error at the moment of upload.
    """
    run_id = run_id or new_run_id()
    now = _utcnow()
    doc = {
        "_id": run_id,
        "drawing_id": drawing_id,
        "filename": filename,
        "source": source,
        "bytes": bytes_received,
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
        "overall": OVERALL_RUNNING,
        "refusal": None,
        "revision_of": None,
        "stages": _blank_stages(),
    }
    coll(COLL_INGEST_RUNS).insert_one(doc)
    return run_id


class Recorder:
    """Writes one run document as the pipeline moves through it.

    Handed into `ingest_file`. A `None` recorder is a legitimate caller (the
    old CLI behaviour, and every existing test), which is why `NullRecorder`
    exists rather than a pile of `if recorder is not None` at each call site.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._started = time.perf_counter()

    # -- internals ---------------------------------------------------------

    def _update(self, update: Mapping[str, Any]) -> None:
        """Apply one update, never letting a history failure fail an ingest."""
        try:
            coll(COLL_INGEST_RUNS).update_one({"_id": self.run_id}, update)
        except PyMongoError as exc:
            # Deliberately swallowed. See the module docstring: losing a
            # progress line is a far better outcome than losing the ingest.
            log.warning(
                "ingest run history write failed: %s", type(exc).__name__,
                extra={"run_id": self.run_id},
            )

    def _set_stage(self, name: str, fields: Mapping[str, Any]) -> None:
        if name not in STAGES:
            # A typo would otherwise write a stage nothing renders.
            log.warning("unknown ingest stage %r", name, extra={"run_id": self.run_id})
            return
        index = STAGES.index(name)
        payload: dict[str, Any] = {f"stages.{index}.{k}": v for k, v in fields.items()}
        payload["updated_at"] = _utcnow()
        self._update({"$set": payload})

    # -- stage transitions -------------------------------------------------

    def start(self, name: str, **detail: Any) -> None:
        fields: dict[str, Any] = {"status": RUNNING, "started_at": _utcnow()}
        if detail:
            fields["detail"] = detail
        self._set_stage(name, fields)

    def finish(self, name: str, **detail: Any) -> None:
        self._set_stage(
            name,
            {"status": DONE, "finished_at": _utcnow(), "detail": detail},
        )

    def skip(self, name: str, why: str, **detail: Any) -> None:
        self._set_stage(
            name,
            {
                "status": SKIPPED,
                "finished_at": _utcnow(),
                "detail": {"why": why, **detail},
            },
        )

    def fail(self, name: str, why: str, **detail: Any) -> None:
        self._set_stage(
            name,
            {
                "status": FAILED,
                "finished_at": _utcnow(),
                "detail": {"why": why, **detail},
            },
        )

    # -- run-level facts ---------------------------------------------------

    def set_drawing(self, drawing_id: str) -> None:
        """The drawing id is only known once the bytes have been hashed."""
        self._update({"$set": {"drawing_id": drawing_id, "updated_at": _utcnow()}})

    def set_revision_of(self, predecessor: Mapping[str, Any] | str | None) -> None:
        """Flag this run as a candidate revision of an earlier drawing.

        Accepts either the contract's block or a bare drawing id, because
        `recipes.revision.find_predecessors` answers with an ID: the RULE is
        about identity, and a name is decoration on top of it. Normalising
        here rather than at each call site is what stops a bare id reaching
        `dict()` and raising, which is exactly how this went wrong once.
        """
        if not predecessor:
            return
        block = (
            dict(predecessor)
            if isinstance(predecessor, Mapping)
            else {"drawing_id": str(predecessor)}
        )
        self._update({"$set": {"revision_of": block, "updated_at": _utcnow()}})

    def _close(self, overall: str, refusal: Mapping[str, Any] | None = None) -> None:
        now = _utcnow()
        fields: dict[str, Any] = {
            "overall": overall,
            "finished_at": now,
            "updated_at": now,
            "elapsed_ms": int((time.perf_counter() - self._started) * 1000),
        }
        if refusal is not None:
            fields["refusal"] = dict(refusal)
        self._update({"$set": fields})

    def succeeded(self, **detail: Any) -> None:
        self.finish("done", **detail)
        self._close(OVERALL_DONE)

    def already_current(self, **detail: Any) -> None:
        """The content is already stored and current. Not a failure."""
        for name in ("store", "render", "dossier", "landuse_draft", "h3"):
            self.skip(name, "this content is already stored and current")
        self.finish("done", **detail)
        self._close(OVERALL_SKIPPED)

    def failed(self, stage: str, why: str, hint: str | None = None) -> None:
        self.fail(stage, why)
        self._close(
            OVERALL_FAILED,
            {"code": "INGEST_FAILED", "message": why, "hint": hint},
        )

    def running_stage(self) -> str | None:
        """The stage currently marked RUNNING, or None."""
        try:
            doc = coll(COLL_INGEST_RUNS).find_one({"_id": self.run_id}, {"stages": 1})
        except PyMongoError:
            return None
        for stage in (doc or {}).get("stages") or []:
            if stage.get("status") == RUNNING:
                return str(stage.get("name"))
        return None

    def crashed(self, exc: BaseException) -> str:
        """Record an exception that escaped the ingest, and name its cause.

        Against the stage that was actually RUNNING rather than always against
        `done`. Blaming `done` leaves the stage that really failed sitting at
        `pending`, so the run points at the wrong place -- and on this project
        a stage stuck at `pending` is already the shape of a silent death.

        The traceback travels too. "unexpected OperationFailure: ..." names a
        type and a message but not a line, and on a cluster that refuses
        un-indexed queries the message alone does not say WHICH query. Tail
        rather than head: the frames nearest the raise are the ones that say
        where.

        Returns the stage it recorded against, so the caller can log the same
        name it wrote.
        """
        name = self.running_stage() or "done"
        tail = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-_TRACEBACK_CHARS:]
        why = f"unexpected {type(exc).__name__}: {exc}"
        self.fail(name, why, traceback_tail=tail)
        self._close(
            OVERALL_FAILED,
            {
                "code": "INGEST_CRASHED",
                "message": why,
                "hint": (
                    f"the {name!r} stage raised; `traceback_tail` on that "
                    "stage carries the frames nearest the raise"
                ),
            },
        )
        return name

    def refused(self, code: str, message: str, hint: str | None = None) -> None:
        """A refusal is a recorded event, never a silent drop."""
        self.fail("upload", message)
        self._close(OVERALL_REFUSED, {"code": code, "message": message, "hint": hint})


class NullRecorder(Recorder):
    """Records nothing. The default, so existing callers are unchanged."""

    def __init__(self) -> None:  # noqa: D107 - deliberately no run id
        self.run_id = ""
        self._started = time.perf_counter()

    def _update(self, update: Mapping[str, Any]) -> None:
        return None


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _decorate(doc: dict[str, Any]) -> dict[str, Any]:
    """Add the derived watchdog fields.

    Derived on read and never stored, because a stalled run is by definition
    one that stopped writing: a flag it would have had to set itself is
    exactly the flag that would be missing.
    """
    doc = dict(doc)
    doc["run_id"] = doc.pop("_id")
    stalled = False
    updated = doc.get("updated_at")
    if doc.get("overall") == OVERALL_RUNNING and isinstance(updated, datetime):
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        stalled = _utcnow() - updated > timedelta(seconds=STALL_AFTER_S)
    doc["stalled"] = stalled
    doc["stalled_since"] = updated if stalled else None
    return doc


#: How long past the stall threshold a `running` run may stay open before it
#: is closed as abandoned. Generous on purpose: the stall verdict already
#: tells a reader the run has gone quiet, and this is the harder claim that
#: it is never coming back.
ABANDON_AFTER_S: Final[int] = STALL_AFTER_S * 4


def reap_abandoned(*, older_than_s: int | None = None) -> int:
    """Close runs whose process died without being able to say so.

    `_run_ingest` catches exceptions, but a SIGKILL is not an exception: when
    the OOM killer took an ingest, nothing wrote to the run document and it
    read `running` for ever, which is exactly the spinner this feature exists
    to abolish. The stall verdict tells a reader the run has gone quiet; this
    is what eventually turns that into a verdict.

    Time-based rather than process-based on purpose. A run is a document, not
    a handle: the worker may be in another container, or in a process that no
    longer exists, and asking "is that pid alive" is a question this side
    cannot answer honestly. Silence past a generous threshold can be.
    """
    cutoff = _utcnow() - timedelta(seconds=older_than_s or ABANDON_AFTER_S)
    result = coll(COLL_INGEST_RUNS).update_many(
        {"overall": OVERALL_RUNNING, "updated_at": {"$lt": cutoff}},
        {
            "$set": {
                "overall": OVERALL_FAILED,
                "finished_at": _utcnow(),
                "updated_at": _utcnow(),
                "refusal": {
                    "code": "INGEST_ABANDONED",
                    "message": (
                        "this run stopped writing and never reached a terminal "
                        "state, so its process died without being able to "
                        "report why. The usual cause is the process being "
                        "killed rather than failing, which an exception "
                        "handler cannot catch"
                    ),
                    "hint": (
                        "check the container logs around the run's last "
                        "update, then ingest the file again"
                    ),
                },
            }
        },
    )
    if result.modified_count:
        log.warning("closed %d abandoned ingest run(s)", result.modified_count)
    return result.modified_count


def get(run_id: str) -> dict[str, Any] | None:
    doc = coll(COLL_INGEST_RUNS).find_one({"_id": run_id})
    return _decorate(doc) if doc else None


def latest(
    *, drawing_id: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    """Runs, newest first, optionally for one drawing.

    The "latest per drawing" read the contract promises is this call with a
    `drawing_id` and the first row taken; a separate route for it would be a
    second way to ask one question.
    """
    # Close anything abandoned before reporting. Done here rather than on the
    # single-run read, which the progress panel polls every 1.5 s: a listing
    # is occasional, and one update_many on it is cheap.
    try:
        reap_abandoned()
    except PyMongoError as exc:  # pragma: no cover - reporting must not fail
        log.warning("could not reap abandoned runs: %s", type(exc).__name__)

    query: dict[str, Any] = {}
    if drawing_id:
        query["drawing_id"] = drawing_id
    cursor = (
        coll(COLL_INGEST_RUNS)
        .find(query)
        .sort("started_at", -1)
        .limit(max(1, min(limit, 100)))
    )
    return [_decorate(doc) for doc in cursor]


def stage_names() -> Iterable[str]:
    return STAGES
