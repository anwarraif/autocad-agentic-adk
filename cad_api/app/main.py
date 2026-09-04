"""cad-api — HTTP surface over the extracted drawings.

Endpoints
---------
    GET  /health                          liveness, no dependencies
    GET  /ready                           readiness, checks MongoDB
    GET  /drawings                        the drawing picker's data source
    GET  /drawings/{id}                   full metadata: layers, blocks, layouts
    GET  /drawings/{id}/svg               rendered SVG, gzip, with data-handle
    GET  /drawings/{id}/entities          structural query, paginated
    GET  /drawings/{id}/entities/{handle} one entity plus its comments
    GET  /drawings/{id}/search            text search over annotations
    GET  /drawings/{id}/spatial           bbox overlap query
    GET  /drawings/{id}/distinct          distinct-value counts for a field
    POST /drawings/{id}/selection         region selection, summarised
    POST /drawings/{id}/selection/rows    one (layer, type) group of a region
    POST /drawings/{id}/selections        store a selection, get an id back
    POST /drawings/{id}/selection/describe aggregate over a stored selection
    GET  /comments                        comments for a drawing or entity
    POST /comments                        anchor a comment to an entity

Errors are returned as `{"error": CODE, "message": ..., "hint": ...}` with the
same taxonomy the MCP tools use, so a failure reads the same whether it
reaches a human or the agent.
"""

from __future__ import annotations

import logging
import math

import yaml
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final, Mapping, AsyncIterator

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pymongo.errors import PyMongoError

from . import export as export_mod
from . import dossier_read, geo_view, ingest_runs, mongo, recipes, store
from .recipes import selfheal
from .extract import is_block_layout
from .config import Settings, get_settings
from .logging_conf import configure_logging
from .models import (
    CommentIn,
    CommentOut,
    DescribeSelectionIn,
    RegionIn,
    RegionRowsIn,
    SaveSelectionIn,
)

log = logging.getLogger(__name__)


class ApiError(Exception):
    """An error with a machine-readable code and a way forward.

    Every failure carries a `hint`. A caller — human or model — that is told
    only "not found" retries the same wrong thing; one that is told where to
    get a valid id does not.
    """

    def __init__(self, code: str, message: str, hint: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.status = status


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info(
        "cad-api starting",
        extra={
            "dxf_dir": str(settings.dxf_dir),
            "svg_dir": str(settings.svg_dir),
            "database": mongo.DB_NAME,
            "collection_prefix": mongo.PREFIX,
            "mongo_target": mongo.target_hosts(),
            "mongo_is_local": mongo.target_is_local(),
        },
    )
    try:
        mongo.ensure_indexes()
    except PyMongoError as exc:
        # Not fatal at boot: /ready will report not-ready and the container
        # stays up so the failure is visible in logs rather than a crash loop.
        log.error("could not ensure indexes at startup: %s", type(exc).__name__)
    try:
        # A container restart is the commonest way for an ingest process to
        # die without writing: nothing in the old process gets to run, so the
        # run it was working on would read `running` for ever. This is the
        # first moment the new process can say otherwise.
        closed = ingest_runs.reap_abandoned()
        if closed:
            log.warning("closed %d ingest run(s) abandoned before restart", closed)
    except PyMongoError as exc:
        log.error("could not reap abandoned runs at startup: %s", type(exc).__name__)
    yield
    mongo.close()


app = FastAPI(
    title="cad-api",
    version="0.1.0",
    summary="Read AutoCAD drawings as structured data, and render them with clickable handles.",
    lifespan=lifespan,
)

_settings = get_settings()

#: Responses smaller than this are sent as they are. Compressing a 400-byte
#: answer costs a round of CPU on both ends to save a few dozen bytes, and the
#: gzip header itself is 18 of them. The figure is Starlette's own default,
#: stated here rather than left implicit because it is the line between "this
#: response is compressed" and "this one is not", and somebody will one day
#: measure a small response and wonder why.
GZIP_MIN_BYTES = 500

# Compression before CORS in source order, which puts it OUTSIDE CORS at
# runtime: Starlette wraps each middleware around the ones added after it, so
# the last one added is the innermost. Either order works here — the headers
# CORS writes compress harmlessly — and this one is chosen so that the bytes
# on the wire are the last thing that happens, which is where a reader looks
# for it.
#
# It changes nothing for a client that does not ask: gzip is applied only when
# the request carries `Accept-Encoding: gzip`, so every existing caller keeps
# receiving exactly the bytes it received before. Measured on the payload that
# needs it — `/geo` with parcels, 997,594 bytes — see the tests.
app.add_middleware(GZipMiddleware, minimum_size=GZIP_MIN_BYTES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    # `ETag` is not a CORS-safelisted response header, so without this a
    # browser on another origin cannot read it and neither can any client
    # code. The viewer runs on :4310 against this API on :4311, which is
    # exactly that case: the tag was being computed, served, and then dropped
    # before anything could use it. Exposing it costs nothing — it is a hash
    # of inputs the caller already has.
    expose_headers=["ETag"],
)


@app.exception_handler(ApiError)
async def _api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={"error": exc.code, "message": exc.message, "hint": exc.hint},
    )


@app.exception_handler(store.MeasureRefused)
async def _refusal_handler(
    _request: Request, exc: store.MeasureRefused
) -> JSONResponse:
    """A refusal is an answer, and it belongs to every endpoint that can raise it.

    `MeasureRefused` began life inside the measurement endpoint and was caught
    there by hand. The empty-filter guard raises it from `query_entities`,
    `distinct_values` and `search_text` as well, and an uncaught one there would
    surface as a 500 -- an infrastructure failure where the truth is "you asked
    for something that cannot be answered as asked". The distinction matters:
    a caller retries a 500 and reads a 400.
    """
    return JSONResponse(
        status_code=400,
        content={"error": exc.code, "message": exc.message, "hint": exc.hint},
    )


@app.exception_handler(PyMongoError)
async def _mongo_error_handler(_request: Request, exc: PyMongoError) -> JSONResponse:
    """Keep database failures inside the error taxonomy.

    Without this, a dropped connection on any read endpoint produced a bare
    FastAPI 500 with a raw traceback body. The MCP layer parses
    `{error, message, hint}`, so an untyped 500 reaches the agent as
    `HTTP_500` plus 200 characters of stack trace — noise it cannot act on.
    """
    log.exception("database error")
    return JSONResponse(
        status_code=503,
        content={
            "error": "STORAGE_ERROR",
            "message": f"The database is not answering: {type(exc).__name__}.",
            "hint": "Check GET /ready. This is an infrastructure failure, not a bad request.",
        },
    )


@app.exception_handler(store.CommentIdConflict)
async def _comment_conflict_handler(
    _request: Request, exc: store.CommentIdConflict
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": "REQUEST_ID_REUSED",
            "message": str(exc),
            "hint": "Generate a fresh client_request_id per distinct comment; "
            "reuse it only when retrying the exact same write.",
        },
    )


def _require_drawing(drawing_id: str) -> dict[str, Any]:
    doc = store.get_drawing(drawing_id)
    if doc is None:
        raise ApiError(
            "DRAWING_NOT_FOUND",
            f"No drawing with id {drawing_id!r}.",
            "Call GET /drawings to list the ingested drawings and their ids.",
            status=404,
        )
    return doc


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness: the process is up. Deliberately touches no dependency."""
    return {"status": "ok", "service": "cad-api"}


@app.get("/ready", tags=["ops"])
def ready(response: Response) -> dict[str, Any]:
    """Readiness: MongoDB answers and the SVG cache is reachable."""
    settings = get_settings()
    mongo_ok = mongo.ping()
    svg_ok = settings.svg_dir.exists() or settings.svg_dir.parent.exists()
    ok = mongo_ok and svg_ok
    if not ok:
        response.status_code = 503
    return {
        "status": "ready" if ok else "not-ready",
        "mongo": mongo_ok,
        "svg_dir": svg_ok,
        "database": mongo.DB_NAME,
        # Same database name on the local server and on the company
        # cluster: the host is the only thing that says which one.
        "target": mongo.target_hosts(),
        "is_local": mongo.target_is_local(),
    }


# ---------------------------------------------------------------------------
# Drawings
# ---------------------------------------------------------------------------


#: Read size for the upload stream. The same 1 MiB the content hash already
#: reads in (`extract.compute_drawing_id`), so a 500 MB drawing costs one
#: buffer rather than 500 MB of resident memory.
UPLOAD_CHUNK_BYTES: Final[int] = 1024 * 1024

#: Practical ceiling for an upload. Sedra's DXF is 500 MB and must fit; past
#: this the CLI is the documented door, because a browser upload that large
#: is more likely to be a mistake than a drawing, and refusing early beats
#: filling the volume and failing at the end.
UPLOAD_MAX_BYTES: Final[int] = 600 * 1024 * 1024

#: What intake accepts. Everything else is refused by name rather than being
#: handed to a converter that will fail less clearly.
UPLOAD_SUFFIXES: Final[frozenset[str]] = frozenset({".dxf", ".dwg"})

#: One ingest at a time, off the event loop. Ingestion is minutes of blocking
#: CPU work: running it in the request handler, or in a FastAPI background
#: task (same loop), would freeze every other request including the polling
#: the progress panel depends on. Single worker because two concurrent
#: ingests of this size compete for the same disk and neither finishes sooner.
_INGEST_POOL: Final[ThreadPoolExecutor] = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="ingest"
)


def _upload_dir(settings: Settings) -> Path:
    """Where uploaded drawings are kept.

    Not `CAD_DXF_DIR`: that mount is read-only by design (docker-compose.yml
    mounts both source directories `:ro`, because the source drawings are not
    ours to modify). The SVG volume is the writable, persisted one, and it
    already holds `converted/` and `exports/` for the same reason.
    """
    return settings.svg_dir / "uploads"


def _stream_to_disk(upload: UploadFile, target: Path) -> int:
    """Copy the upload to `target` in chunks, refusing over the ceiling.

    Returns the byte count. Raises `ApiError` (413) without ever holding the
    whole file: the check is inside the loop, so an oversized upload is
    stopped on the way in rather than measured after it has landed.
    """
    written = 0
    with target.open("wb") as out:
        while True:
            chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > UPLOAD_MAX_BYTES:
                out.close()
                target.unlink(missing_ok=True)
                raise ApiError(
                    "FILE_TOO_LARGE",
                    f"{upload.filename!r} is larger than the "
                    f"{UPLOAD_MAX_BYTES // (1024 * 1024)} MB upload limit.",
                    "Ingest it from the command line instead: "
                    "docker compose exec cad-api python -m app.ingest "
                    "--file <name>. The CLI writes the same run history.",
                    status=413,
                )
            out.write(chunk)
    return written


def _refuse_upload(
    run_id: str | None, code: str, message: str, hint: str, status: int
) -> ApiError:
    """Record a refusal, then describe it.

    Every refused upload is an entry in the run history, because "never
    silently drop" applies to the files we turn away as much as to the ones
    we keep. The run id travels back in the error body so the UI can show the
    refusal in the same panel it would have shown progress in.
    """
    if run_id:
        ingest_runs.Recorder(run_id).refused(code, message, hint)
    return ApiError(code, message, f"{hint} (run {run_id})" if run_id else hint, status)


@app.post("/drawings/upload", tags=["drawings"], status_code=202)
def upload_drawing(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a drawing, hash it, and start ingesting it in the background.

    The order here is the whole design. The bytes are streamed to disk and
    hashed BEFORE anything is stored, because the content hash is the
    drawing's identity (`extract.compute_drawing_id`) and identity is what
    decides whether this upload is a duplicate, a revision, or new. There is
    no second hash anywhere in this path.

    Returns 202 with a `run_id`: ingestion happens off the request, and the
    caller polls `/drawings/ingest-runs/{run_id}` to watch it. The response is
    immediate even for a 500 MB drawing, which is the point.
    """
    from .extract import compute_drawing_id
    from .ingest import ingest_file, predecessor_block

    settings = get_settings()
    filename = Path(file.filename or "").name
    if not filename:
        raise ApiError(
            "UNSUPPORTED_FILE", "The upload carried no filename.",
            "Send the file as multipart form field 'file'.", status=415,
        )

    run_id = ingest_runs.create(filename=filename, source="upload")
    recorder = ingest_runs.Recorder(run_id)
    recorder.start("upload")

    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise _refuse_upload(
            run_id, "UNSUPPORTED_FILE",
            f"{filename!r} is not a drawing: intake accepts "
            f"{', '.join(sorted(UPLOAD_SUFFIXES))}, not {suffix or 'a file with no extension'}.",
            "Export the drawing as DXF or DWG and upload that.",
            415,
        )

    upload_dir = _upload_dir(settings)
    upload_dir.mkdir(parents=True, exist_ok=True)
    staged = upload_dir / f".incoming-{run_id}{suffix}"
    try:
        written = _stream_to_disk(file, staged)
    except ApiError as exc:
        raise _refuse_upload(run_id, exc.code, exc.message, exc.hint or "", exc.status)
    if written == 0:
        staged.unlink(missing_ok=True)
        raise _refuse_upload(
            run_id, "UNSUPPORTED_FILE", f"{filename!r} is empty.",
            "Check the file on your machine and upload it again.", 415,
        )

    drawing_id = compute_drawing_id(staged)
    recorder.set_drawing(drawing_id)

    existing = store.get_drawing(drawing_id)
    # A drawing whose entities never landed is not a duplicate of anything
    # usable: refusing the re-upload would leave the user with a broken entry
    # and no way to replace it through the UI. The same applies to a drawing
    # deleted from the store, which simply is not found here.
    if existing is not None and existing.get("ingest_incomplete"):
        log.info(
            "re-accepting an upload of a drawing whose previous ingest did not "
            "complete", extra={"drawing_id": drawing_id},
        )
        existing = None
    if existing is not None:
        staged.unlink(missing_ok=True)
        ingested_at = existing.get("ingested_at")
        when = ingested_at.isoformat() if hasattr(ingested_at, "isoformat") else str(ingested_at)
        raise _refuse_upload(
            run_id, "ALREADY_INGESTED",
            f"This file is already ingested as {drawing_id} "
            f"({existing.get('original_filename')}, {when}). The content is "
            "identical, byte for byte.",
            "Open it from the drawing picker. To re-ingest it deliberately, "
            "use the CLI with --force.",
            409,
        )

    # Same name, different content: the established predecessor rule. Asked
    # here only so the answer can travel with the 202; the authoritative
    # value is written by the arrival path after ingest, using this same
    # function, so there is one rule and not two.
    revision_of = None
    try:
        found = recipes.revision.find_predecessors(
            {"_id": drawing_id, "original_filename": filename},
            store.list_drawings(include_fixtures=True),
        )
        revision_of = predecessor_block(found.get("predecessor"))
        recorder.set_revision_of(revision_of)
    except Exception as exc:  # noqa: BLE001 - a flag, never a gate
        log.warning("predecessor pre-check failed: %s", exc)

    final = upload_dir / filename
    if final.exists() and final != staged:
        final.unlink()
    staged.rename(final)
    recorder.finish(
        "upload", bytes=written, sha16=drawing_id, stored_at=str(final)
    )

    _INGEST_POOL.submit(_run_ingest, ingest_file, final, run_id)

    return {
        "run_id": run_id,
        "drawing_id": drawing_id,
        "filename": filename,
        "bytes": written,
        "overall": ingest_runs.OVERALL_RUNNING,
        "revision_of": revision_of,
    }


def _run_ingest(ingest_file: Any, path: Path, run_id: str) -> None:
    """Run one ingestion on the pool thread, recording into `run_id`.

    Wrapped because a thread that dies with an exception leaves the run
    document saying `running` for ever, which is precisely the spinner this
    feature exists to abolish. Anything that escapes ingest is written into
    the run before it is re-logged.
    """
    recorder = ingest_runs.Recorder(run_id)
    try:
        ingest_file(path, recorder=recorder)
    except Exception as exc:  # noqa: BLE001 - the run must never be left open
        # The run is told WHICH stage raised and what the traceback said, not
        # merely that something did. A background task whose exception reaches
        # only the container log is worse than a crash: the reaper eventually
        # closes the run, and nobody can say why it died.
        stage = recorder.crashed(exc)
        log.exception(
            "background ingest crashed",
            extra={"run_id": run_id, "stage": stage},
        )


@app.get("/drawings/ingest-runs", tags=["drawings"])
def list_ingest_runs(
    drawing_id: str | None = Query(
        None, description="Only runs for this drawing. Omit for every run."
    ),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """Ingest runs, newest first.

    The "latest run for a drawing" the UI needs is this call with a
    `drawing_id` and the first row taken, rather than a second route asking
    one question a second way.
    """
    runs = ingest_runs.latest(drawing_id=drawing_id, limit=limit)
    return {"total_returned": len(runs), "runs": runs}


@app.get("/drawings/ingest-runs/{run_id}", tags=["drawings"])
def get_ingest_run(run_id: str) -> dict[str, Any]:
    """One ingest run, with the derived stall verdict.

    This is the only thing the progress panel reads. A browser refresh in the
    middle of an ingest re-reads it and picks the run back up, because the run
    document is the state and the UI holds none of its own.
    """
    run = ingest_runs.get(run_id)
    if run is None:
        raise ApiError(
            "RUN_NOT_FOUND", f"No ingest run with id {run_id!r}.",
            "Runs are listed at GET /drawings/ingest-runs.", status=404,
        )
    return run



@app.get("/drawings/{drawing_id}/crs/candidates", tags=["drawings"])
def crs_candidates(drawing_id: str) -> dict[str, Any]:
    """Which UTM zones this drawing's own numbers could be, and where each lands.

    A drawing that does not state its coordinate system cannot be mapped, and
    nothing in the file says which zone it is in. What the file DOES give is
    its extents, and inverting those through a candidate zone says where the
    drawing would be if that zone were right. Somewhere in open sea, or on
    another continent, answers itself; the plausible ones are for a human.

    This is arithmetic, not knowledge. It proposes and refuses to decide,
    which is the division `landuse_draft` already uses for land use.
    """
    from .landuse import crs as crs_math  # noqa: PLC0415

    drawing = _require_drawing(drawing_id)
    extents = drawing.get("extents") or {}
    lo, hi = extents.get("min") or [], extents.get("max") or []
    if len(lo) < 2 or len(hi) < 2:
        raise ApiError(
            "NO_EXTENTS",
            f"{drawing.get('original_filename')!r} has no usable extents, so "
            "there is nothing to invert and no candidate to offer.",
            "Ask the drawing's author for the coordinate system directly.",
            status=422,
        )

    cx = (float(lo[0]) + float(hi[0])) / 2
    cy = (float(lo[1]) + float(hi[1])) / 2
    candidates: list[dict[str, Any]] = []
    for epsg in list(range(32601, 32661)) + list(range(32701, 32761)):
        try:
            lat, lon = crs_math.to_lat_lon(cx, cy, epsg=epsg)
            zone, north = crs_math.zone_of(epsg)
        except Exception:  # noqa: BLE001 - a zone this module will not invert
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        candidates.append(
            {
                "epsg": epsg,
                "name": "WGS 84 / UTM zone %d%s" % (zone, "N" if north else "S"),
                "centre_lat": round(lat, 5),
                "centre_lon": round(lon, 5),
                "central_meridian": crs_math.central_meridian(zone),
            }
        )
    # Ordered by zone, and NOT ranked. There was a ranking here, by distance
    # from each zone's central meridian, and it was worthless: that distance
    # is a function of the easting, the easting is the same for every
    # candidate, so the number came out identical for all sixty zones and the
    # "best" one was simply zone 1. A number that cannot separate the answers
    # must not be presented as though it had.
    candidates.sort(key=lambda c: c["epsg"])

    return {
        "drawing_id": drawing_id,
        "filename": drawing.get("original_filename"),
        "declared_in_file": bool(drawing.get("declared_crs")),
        "extents_centre": [cx, cy],
        "units": drawing.get("units_name"),
        "candidates": candidates,
        "basis": (
            "each candidate is this drawing's own extents centre inverted "
            "through that UTM zone. It says where the drawing WOULD be if the "
            "zone were right, and nothing about whether it is"
        ),
        "what_varies": (
            "the latitude is the same in every candidate, because it comes "
            "from the northing, which does not change. Only the longitude "
            "moves, by six degrees per zone. So choosing a zone is choosing a "
            "longitude, and a person who knows where the project is answers "
            "it immediately"
        ),
        "how_to_choose": (
            "pick the candidate whose longitude is where the drawing was "
            "actually surveyed. The rest land in open sea or on another "
            "continent, which is what makes this checkable rather than a guess."
        ),
        "how_to_accept": "POST /drawings/%s/crs with an epsg" % drawing_id,
    }


@app.post("/drawings/{drawing_id}/crs", tags=["drawings"], status_code=201)
def accept_crs(drawing_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Record a human's decision about which coordinate system a drawing uses.

    Written to the accepted-config overlay rather than the image tree, so a
    drawing can be georeferenced by the person looking at it instead of by a
    rebuild. Until this existed, every newly uploaded drawing stopped in the
    same place: ingested, stored, dossiered and unmappable, with no way
    forward that did not involve a developer.

    `declared_in_file` stays FALSE. A person accepting a candidate is not the
    file stating one, and every response that quotes a latitude carries that
    distinction; only the drawing's author can change it.

    Cells are assigned immediately afterwards, because a coordinate system
    that has been accepted and not applied leaves the drawing answering "not
    georeferenced" to every question, which is the confusing half of having
    accepted it at all.
    """
    from . import geo_backfill  # noqa: PLC0415
    from .landuse import ACCEPTED_DIR  # noqa: PLC0415
    from .landuse import crs as crs_math  # noqa: PLC0415

    drawing = _require_drawing(drawing_id)
    epsg = body.get("epsg")
    if not isinstance(epsg, int):
        raise ApiError(
            "CRS_EPSG_REQUIRED",
            "This route needs an integer EPSG code in the body.",
            "Candidates: GET /drawings/%s/crs/candidates" % drawing_id,
            status=400,
        )
    try:
        zone, north = crs_math.zone_of(epsg)
    except Exception as exc:  # noqa: BLE001 - the module states its own range
        raise ApiError(
            "CRS_UNSUPPORTED",
            str(exc),
            "Only WGS 84 UTM zones can be inverted by this project.",
            status=422,
        ) from exc

    confirmed_by = str(body.get("confirmed_by") or "").strip()
    note = str(body.get("note") or "").strip()
    name = "WGS 84 / UTM zone %d%s" % (zone, "N" if north else "S")
    detail = (
        "accepted through the viewer against the candidates produced by "
        "inverting this drawing's own extents"
    )
    if confirmed_by:
        detail += "; confirmed by " + confirmed_by

    document = {
        "version": 1,
        "drawing_id": drawing_id,
        "drawing_name": str(drawing.get("original_filename") or drawing_id),
        "default_use": "unknown",
        "crs": {
            "epsg": epsg,
            "name": name,
            "declared_in_file": False,
            "sources": [
                {
                    "origin": "geometry",
                    "locator": "modelspace extents",
                    "detail": detail,
                }
            ],
            "unknown": "the file itself states no coordinate system",
            "how_to_verify": (
                "confirm with the drawing's author, or against any single "
                "surveyed point in the drawing whose latitude is known"
            ),
            "note": note
            or (
                "accepted by a human from measured candidates; inferred, and "
                "not to be read as stated by the file"
            ),
        },
        "layers": {},
    }

    ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
    target = ACCEPTED_DIR / ("%s.yaml" % drawing_id)
    target.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    report = geo_backfill.assign_cells(drawing_id)
    return {
        "drawing_id": drawing_id,
        "epsg": epsg,
        "name": name,
        "declared_in_file": False,
        "written_to": str(target),
        "cells": {
            "ok": report.get("ok"),
            "assigned": report.get("assigned"),
            "world_placed": report.get("world_placed"),
            "excluded": report.get("excluded"),
            "resolution": report.get("resolution"),
            "reason": report.get("reason"),
        },
    }


@app.get("/drawings", tags=["drawings"])
def get_drawings(include_fixtures: bool = False) -> dict[str, Any]:
    """Every ingested drawing. Feeds the viewer's drawing picker.

    Test fixtures -- the synthetic before/after pair `revision_diff` is proven
    against -- are left out unless asked for. They stay ingested and reachable
    by id; what changes is that the picker no longer offers a file nobody drew
    as though it were one of this project's drawings.

    Withheld is counted, never silently dropped: `fixtures_withheld` says how
    many and `include_fixtures=true` returns them.
    """
    rows = store.list_drawings(include_fixtures=include_fixtures)
    withheld = 0 if include_fixtures else store.count_fixtures()
    return {
        "total": len(rows),
        "drawings": rows,
        "fixtures_withheld": withheld,
        "fixtures_note": (
            "drawings ingested from the fixture directory are excluded from "
            "this list. They are still stored and still reachable at "
            "/drawings/{id}; pass include_fixtures=true to list them"
        )
        if withheld
        else None,
    }


@app.get("/drawings/{drawing_id}", tags=["drawings"])
def get_drawing_detail(drawing_id: str) -> dict[str, Any]:
    """Full metadata for one drawing: layers, blocks, layouts, extents, units."""
    doc = _require_drawing(drawing_id)
    doc["renders"] = store.list_render_meta(drawing_id)
    doc["comment_counts"] = store.comment_counts(drawing_id)
    # The Dossier summary: what this drawing HOLDS, by geometric role, with the
    # measure each role earns. Published under exactly this key because
    # `cad_mcp/server.py` reads `data["dossier"]` by name -- mount it anywhere
    # else and the agent is told "not computed" while the answer sits in the
    # store. When it has not been built, `not_computed_block` says so and how
    # to build it: silence here would read as an empty drawing, which is the
    # confusion this whole campaign exists to end.
    doc["dossier"] = dossier_read.summary(drawing_id) or dossier_read.not_computed_block(
        drawing_id
    )
    # What can be COMPUTED about this drawing, published where the agent
    # already looks. `describe_drawing` is the first call on any drawing;
    # `list_analyses` is a separate call nobody makes before deciding a
    # question is impossible.
    #
    # Measured today: asked whether every plot has road frontage, the agent
    # called this endpoint, called land_use_summary, and answered that the
    # available tools "do not support advanced geometric analysis ... beyond
    # the current capabilities". `frontage_check` was in the catalogue the
    # whole time and names the two landlocked parcels by handle. A false claim
    # about its own reach is the same defect as a false claim about the
    # drawing, and it is harder to catch because nothing in the answer looks
    # like a number that could be wrong.
    #
    # Names and one line each, not the full catalogue: enough to know the
    # question is answerable and which name to pass to `run_analysis`.
    # Whether this drawing can be put on a map, as one boolean plus its
    # reason. Published here because it is the question a client has to answer
    # BEFORE offering an export, and the only alternative was to run the whole
    # aggregation and look at whether anything came back — a second of work to
    # decide whether to draw a button.
    #
    # Cheap: a config read and a drawing document, the same two the ETag is
    # built from. It never raises: a drawing that cannot be mapped is the
    # ordinary case here, seventeen times out of eighteen.
    try:
        choice = store.store_landuse.crs_for_drawing(drawing_id)
        # Two questions, not one: can this drawing be mapped, and has it been
        # indexed yet. A client that offers an export off the first alone
        # hands somebody an empty file.
        indexed = bool(
            choice.crs is not None
            and mongo.coll(mongo.COLL_ENTITIES).find_one(
                {"drawing_id": drawing_id, "h3_cell": {"$ne": None}}, {"_id": 1}
            )
        )
        doc["geo"] = {
            "georeferenced": choice.crs is not None,
            "indexed": indexed,
            "epsg": choice.crs.epsg if choice.crs else None,
            "crs_source_layer": choice.layer,
            "reason": (
                None
                if indexed
                else (
                    (
                        choice.note
                        or "this drawing declares no coordinate system and has "
                        "no CRS config, so it has no position on the earth"
                    )
                    if choice.crs is None
                    else (
                        "this drawing is georeferenced but has no H3 cells "
                        "yet, so there is nothing to export. Re-ingesting a "
                        "drawing removes them"
                    )
                )
            ),
            "how_to_fix": (
                None
                if indexed or choice.crs is None
                else f"python scripts/backfill_h3.py --drawing {drawing_id}"
            ),
            "export_url": (
                f"/drawings/{drawing_id}/geo/export.geojson" if indexed else None
            ),
        }
    except Exception as exc:  # pragma: no cover - depends on the environment
        doc["geo"] = {
            "georeferenced": False,
            "reason": f"this drawing's CRS could not be resolved: {exc}",
            "export_url": None,
        }

    try:
        catalogue = recipes.catalog()
        doc["analyses_available"] = {
            "count": catalogue.get("count"),
            "recipes": [
                {"recipe": row.get("recipe"), "answers": row.get("answers")}
                for row in (catalogue.get("recipes") or [])
            ],
            "how_to_run": "run_analysis(drawing_id, recipe, params, layout=...)",
            "note": (
                "these are computations that already exist. Before saying a "
                "question cannot be answered, look here: a name that matches "
                "the question means the answer is one call away, and saying "
                "'not possible' when it is listed is a false statement about "
                "this service rather than about the drawing"
            ),
        }
    except Exception as exc:  # noqa: BLE001 -- never fail the drawing on this
        doc["analyses_available"] = {
            "count": None,
            "recipes": [],
            "note": f"the analysis catalogue could not be read: {exc}",
        }
    return doc


@app.get("/drawings/{drawing_id}/svg", tags=["drawings"])
def get_drawing_svg(
    drawing_id: str,
    layout: str | None = Query(None, description="Layout name; defaults to the largest."),
) -> Response:
    """The rendered SVG for one layout, served gzip-compressed.

    Every entity is wrapped in `<g data-handle=... data-layer=... data-type=...>`,
    which is what lets the viewer map a click back to a DWG entity.

    The bytes are stored gzipped and passed through untouched — a 36 MB SVG
    compresses to ~385 KB, and decompressing it here only to have the browser
    recompress it would be pure waste.
    """
    drawing = _require_drawing(drawing_id)
    layout_name = layout or _default_layout(drawing, store.list_render_meta(drawing_id))

    settings = get_settings()
    payload = store.load_svg_gzip(settings.svg_dir, drawing_id, layout_name)
    if payload is None:
        available = [r.get("layout") for r in store.list_render_meta(drawing_id)]
        raise ApiError(
            "RENDER_NOT_FOUND",
            f"No cached render for layout {layout_name!r} of drawing {drawing_id!r}.",
            f"Available layouts for this drawing: {available or 'none'}. "
            "Re-run `python -m app.ingest --file <name> --force` to render it.",
            status=404,
        )
    return Response(
        content=payload,
        media_type="image/svg+xml",
        headers={
            "Content-Encoding": "gzip",
            "Cache-Control": "public, max-age=3600",
            "X-Cad-Layout": layout_name,
        },
    )


def _default_layout(
    drawing: dict[str, Any], renders: list[dict[str, Any]]
) -> str:
    """Pick the layout to show when the caller did not choose one.

    **A paper sheet if the drawing has one**, otherwise modelspace, otherwise a
    block definition. This mirrors what AutoCAD and the Autodesk viewer do, and
    it is what an engineer means by "the drawing": a sheet carries the border,
    the title block, the revision table and the details arranged at their
    intended scale. Modelspace is the raw geometry those sheets are cut from.

    Choosing by entity count -- the obvious heuristic -- gets this exactly
    backwards. A paperspace sheet contains almost nothing of its own: for
    `architectural_-_annotation_scaling_and_multileaders` the sheet holds **9**
    entities (a border, a title block reference, a few viewports) against
    modelspace's 1,367. Yet those 9 expand to 959 drawn objects, because a
    VIEWPORT projects modelspace through itself. Ranking by raw entity count
    therefore always prefers modelspace and never shows the sheet.

    So sheets are preferred by *class* first, and only ranked among themselves
    by `rendered_entities` -- what was actually drawn, viewport expansion
    included -- rather than by how many entities the layout nominally owns.
    """
    by_name = {r["layout"]: r for r in renders}
    if not by_name:
        return "Model"

    def drawn(name: str) -> int:
        return by_name[name].get("rendered_entities", 0)

    names = [n for n in by_name if drawn(n) > 0] or list(by_name)

    sheets = [
        n for n in names if not is_block_layout(n) and n.lower() != "model"
    ]
    if sheets:
        return max(sheets, key=drawn)

    model = [n for n in names if n.lower() == "model"]
    if model:
        return model[0]

    return max(names, key=drawn)


@app.get("/drawings/{drawing_id}/export", tags=["drawings"])
def export_drawing(drawing_id: str, format: str = "dxf") -> FileResponse:
    """A derived DXF or DWG carrying every comment *inside the file*.

    Each commented entity gets three things: XDATA (machine-readable, travels
    with the entity), a hyperlink whose description AutoCAD shows as a hover
    tooltip, and a visible note on the `AI_MARKUP` layer. The original file is
    never touched — this is always a new, derived document.

    `format=dwg` converts the built DXF to DWG with ODA, first stripping the
    render-preset objects ODA cannot restore (they made TrueView demand
    recovery on Janadriyah); the count is returned in
    `X-Unconvertible-Objects-Removed`. DWG is what Autodesk's web viewer
    accepts most reliably, and ~10x smaller than the DXF.

    Slow for large drawings (Janadriyah: ~2 minutes on first build); the
    result is cached and only rebuilt when the comment set or drawing changes.
    """
    if format not in ("dxf", "dwg"):
        raise ApiError(
            "BAD_FORMAT",
            f"Unknown export format {format!r}.",
            "Use format=dxf (default) or format=dwg.",
            status=400,
        )
    drawing = _require_drawing(drawing_id)
    settings = get_settings()

    source = _locate_source_dxf(drawing, settings)
    if source is None:
        raise ApiError(
            "SOURCE_NOT_FOUND",
            f"The source file for {drawing.get('original_filename')!r} is not "
            "mounted in this container.",
            "The export needs the original DXF/DWG under CAD_DXF_DIR or "
            "/data/dwg. Restore the file and retry.",
            status=404,
        )

    export_dir = settings.svg_dir / "exports"
    try:
        path, result = export_mod.build_export(export_dir=export_dir, source_dxf=source, drawing=drawing)
    except Exception as exc:  # noqa: BLE001 - surfaced with the taxonomy
        log.exception("export failed", extra={"drawing_id": drawing_id})
        raise ApiError(
            "EXPORT_FAILED",
            f"Could not build the export: {type(exc).__name__}: {exc}",
            "Check the cad-api logs; the source file may be unreadable.",
            status=500,
        ) from exc

    headers = {
        "X-Comments-Embedded": str(result.comments_embedded),
        "X-Entities-Annotated": str(result.entities_annotated),
        # Non-zero means the save dropped entities (e.g. ACIS solids whose
        # payload does not survive re-serialisation). Surfaced so a caller
        # can warn instead of silently distributing a lossy file.
        "X-Entities-Lost": str(getattr(result, "entities_lost_on_save", 0)),
        "X-Corrupt-Objects-Removed": str(
            getattr(result, "corrupt_objects_removed", 0)
        ),
    }
    stem = Path(drawing.get("original_filename", "drawing")).stem

    if format == "dwg":
        try:
            path, removed = export_mod.build_dwg_export(
                dxf_export=path, export_dir=export_dir
            )
        except Exception as exc:  # noqa: BLE001 - surfaced with the taxonomy
            log.exception("dwg export failed", extra={"drawing_id": drawing_id})
            raise ApiError(
                "EXPORT_FAILED",
                f"Could not build the DWG export: {type(exc).__name__}: {exc}",
                "The DXF export succeeded; retry with format=dxf or check "
                "the cad-api logs.",
                status=500,
            ) from exc
        headers["X-Unconvertible-Objects-Removed"] = str(removed)
        return FileResponse(
            path,
            media_type="application/acad",
            filename=f"{stem}__comments.dwg",
            headers=headers,
        )

    return FileResponse(
        path,
        media_type="application/dxf",
        filename=f"{stem}__comments.dxf",
        headers=headers,
    )


def _locate_source_dxf(drawing: dict[str, Any], settings: Settings) -> Path | None:
    """Find the file this drawing was ingested from.

    Preference order: the DXF in the mounted DXF directory, then the original
    DWG (converted on the fly — the converter lives in this image now).
    """
    candidate = settings.dxf_dir / drawing.get("original_filename", "")
    if candidate.is_file():
        return candidate

    source = Path(str(drawing.get("source_path", "")))
    if source.suffix.lower() == ".dwg" and source.is_file():
        from .convert import ConversionError, convert

        try:
            return convert(source, settings.svg_dir / "converted").dxf_path
        except ConversionError as exc:
            # Returning None here used to send the caller down the
            # SOURCE_NOT_FOUND path, which tells the user the file is not
            # mounted. The file IS mounted; the converters refused it. Saying
            # the wrong one sends them to restore a file that is already
            # there, so the converter's own account is raised instead.
            log.error("export conversion failed: %s", exc)
            raise ApiError(
                "CONVERSION_FAILED",
                str(exc),
                "The original DWG is mounted but could not be converted. The "
                "message above names each converter that was tried.",
                status=422,
            ) from exc
    if source.is_file():
        return source
    return None


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@app.get("/drawings/{drawing_id}/entities", tags=["entities"])
def get_entities(
    drawing_id: str,
    layer: str | None = None,
    type: str | None = Query(None, description="DXF type, e.g. INSERT, LINE, MTEXT."),
    block_name: str | None = None,
    layout: str | None = None,
    text_contains: str | None = None,
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Structural query over entities. Always paginated, always counted.

    The response carries `total_matches` and `truncated`; a result set larger
    than the configured threshold returns a per-layer/per-type breakdown
    instead of rows, so a too-broad query gets a way forward rather than a
    wall of data.
    """
    _require_drawing(drawing_id)
    settings = get_settings()
    limit = min(limit, settings.max_rows)

    result = store.query_entities(
        drawing_id,
        layer=layer,
        dxftype=type,
        block_name=block_name,
        layout_name=layout,
        text_contains=text_contains,
        limit=limit,
        offset=offset,
    )
    # Aggregate over the query that actually produced total_matches, taken
    # from the result itself. Rebuilding it here previously dropped
    # block_name/layout/text_contains and skipped the upper-casing of `type`,
    # so the breakdown silently described a different set of entities than
    # the number it sat beside.
    executed_query = result.pop("_query", {"drawing_id": drawing_id})
    if result["total_matches"] > settings.too_many_threshold and offset == 0:
        result["too_many"] = True
        result["breakdown"] = store.entity_breakdown(executed_query)
        result["hint"] = (
            "Narrow the query with layer, type or block_name, or page through "
            "with offset."
        )
    elif result["total_matches"] == 0 and any(
        (layer, type, block_name, layout, text_contains)
    ):
        # Zero matches is the most misleading answer this API can give: a
        # caller reads it as "there is nothing there" and reports absence as
        # fact. It has at least four causes and they look identical in the
        # response, so the empty result is explained rather than left bare.
        # Costs one document read plus one aggregation, and only on the path
        # that already found nothing.
        result.update(
            store.diagnose_empty_query(
                drawing_id,
                layer=layer,
                dxftype=type,
                layout_name=layout,
                block_name=block_name,
            )
        )
    return result


@app.get("/drawings/{drawing_id}/schedule", tags=["drawings"])
def get_schedule(drawing_id: str) -> dict[str, Any]:
    """The schedule the drawing draws for itself.

    An `ACAD_TABLE` renders as lines and letters, so a viewer cannot tell it
    from any other geometry and a text search sees only loose strings. Read as
    a table, the reference drawing turns out to carry sixty of them holding
    2,449 rows of plot number and plot area -- the author's own statement of a
    quantity this system had only ever measured.

    Which column is the key and which is the value is worked out from the
    values themselves. That schedule has no header row at all, so there is
    nothing else to go on, and a drawing whose tables are a door schedule
    reads exactly the same way.
    """
    drawing = _require_drawing(drawing_id)
    return store.store_schedule.read_schedule(drawing_id, drawing)


@app.get("/drawings/{drawing_id}/schedule/check", tags=["drawings"])
def check_schedule(
    drawing_id: str,
    layout: str = Query(..., description="layout whose geometry is compared"),
    label_layer: str | None = Query(
        None, description="layer of the number text; found by itself when empty"
    ),
    tolerance: float = Query(
        store.store_schedule.DEFAULT_TOLERANCE, gt=0, le=1, description="difference tolerance"
    ),
    limit: int = Query(store.store_schedule.DEFAULT_LIMIT, ge=1, le=5000),
    only_mismatches: bool = Query(True),
) -> dict[str, Any]:
    """Stated against measured, everywhere, and where they disagree.

    This is the question the two sources exist to answer. On the reference
    drawing 2,443 of 2,449 scheduled plots have an outline of their own, 2,383
    of those agree within two percent, and sixty do not -- and a plot whose
    drawn outline does not match its scheduled area is a defect in the
    drawing, which is the kind of thing worth finding.
    """
    _require_drawing(drawing_id)
    return store.store_schedule.check_areas(
        drawing_id,
        layout=layout,
        label_layer=label_layer,
        tolerance=tolerance,
        limit=limit,
        only_mismatches=only_mismatches,
    )


@app.get("/drawings/{drawing_id}/schedule/{key}", tags=["drawings"])
def get_scheduled_item(
    drawing_id: str,
    key: int,
    layout: str = Query(..., description="layout whose geometry is compared"),
    label_layer: str | None = Query(None),
    tolerance: float = Query(store.store_schedule.DEFAULT_TOLERANCE, gt=0, le=1),
) -> dict[str, Any]:
    """One numbered thing, from both sources.

    "What is the area of plot 2043" answered the way a drawing can actually
    support it: 300 because the table says so, 299.9999992 because the
    outline measures so, and the fact that those are the same answer.
    """
    _require_drawing(drawing_id)
    return store.store_schedule.area_of(
        drawing_id, key, layout=layout, label_layer=label_layer, tolerance=tolerance
    )


@app.get("/drawings/{drawing_id}/embedded", tags=["drawings"])
def list_embedded(drawing_id: str) -> dict[str, Any]:
    """Documents embedded inside this drawing that no renderer draws.

    The viewer has always reported these as a count and a sentence -- *"the
    payload lives outside the drawing geometry"* -- and for the reference
    drawing that sentence stood over two 11 MB bitmaps holding the approvals
    sheet and the land-use key: the page that states, in Arabic, which plot
    numbers are commercial, which are educational, and that the rest are
    residential villas. Weeks were spent inferring that by elimination and
    grading it `inferred` while the drawing said it in a picture.

    Nothing here knows anything about a client or a project. Any DXF can carry
    an embedded document, and a drawing whose meaning is written on a pasted
    sheet is a property of how drawings are made, not of who made this one.
    """
    drawing = _require_drawing(drawing_id)
    return store.embedded_index(drawing_id, drawing)


@app.get("/drawings/{drawing_id}/embedded/{handle}", tags=["drawings"])
def get_embedded(drawing_id: str, handle: str) -> Response:
    """One embedded document, as its own bytes and its own media type."""
    drawing = _require_drawing(drawing_id)
    found = store.embedded_file(drawing_id, handle, drawing)
    if found is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no embedded payload with handle {handle!r} in drawing "
                f"{drawing_id!r}."
            ),
        )
    path, media_type = found
    return Response(
        content=path.read_bytes(),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@app.get("/drawings/{drawing_id}/entities/handles", tags=["entities"])
def entity_handles(
    drawing_id: str,
    layout: str | None = None,
    layer: str | None = None,
    type: str | None = None,
    block_name: str | None = None,
    limit: int = Query(25_000, ge=1, le=100_000),
) -> dict[str, Any]:
    """Just the handles for a filter, in one request instead of a hundred.

    This exists for the viewer, not for the agent, and the difference is the
    whole point. `/entities` caps a listing at 100 rows because those rows go
    into a model's context and a thousand of them would crowd out the answer.
    Marking objects on screen needs no rows at all -- a handle is seven
    characters -- so the same cap turned "show me every dimension" into
    ninety-nine round trips and 133 seconds of waiting for a picture.

    Nothing here is expensive: one indexed query, one projected field, no
    geometry. The ceiling is high enough that no drawing in this stack reaches
    it, and `truncated` says so when one does rather than quietly returning
    less.
    """
    _require_drawing(drawing_id)
    return store.entity_handles(
        drawing_id,
        layout_name=layout,
        layer=layer,
        dxftype=type,
        block_name=block_name,
        limit=limit,
    )


@app.get("/drawings/{drawing_id}/entities/{handle}", tags=["entities"])
def get_entity_detail(drawing_id: str, handle: str) -> dict[str, Any]:
    """One entity in full, with any comments already anchored to it."""
    _require_drawing(drawing_id)
    entity = store.get_entity(drawing_id, handle)
    if entity is None:
        raise ApiError(
            "ENTITY_NOT_FOUND",
            f"Handle {handle!r} does not exist in drawing {drawing_id!r}.",
            "Handles are unique per drawing. Take one from "
            "GET /drawings/{id}/entities rather than constructing it.",
            status=404,
        )
    entity["comments"] = store.list_comments(drawing_id, handle)
    # A bare `length` here plus a drawing-level `units` from describe_drawing is
    # the paper-space unit bug (D-076) reachable in two tool calls: this entity
    # may be on a sheet, where the drawing's metres do not apply. The sweep
    # missed this endpoint entirely.
    drawing = store.get_drawing(drawing_id) or {}
    entity["units"] = store._unit_names(drawing, entity.get("layout"))
    # Where this object is on the earth, for the drawings that can say. Three
    # keys, always present, null with a reason on the 17 drawings that have no
    # CRS config -- see store_landuse.geo_for_entity. Nothing above is touched:
    # the measurements stay in drawing coordinates, which is the only frame
    # this file states.
    entity.update(store.store_landuse.geo_for_entity(drawing_id, handle))
    return entity


@app.get("/drawings/{drawing_id}/search", tags=["entities"])
def search(
    drawing_id: str,
    q: str = Query(..., min_length=1, description="Substring to find in entity text."),
    limit: int = Query(20, ge=1),
) -> dict[str, Any]:
    """Text search over annotations: MTEXT, TEXT, dimensions, block attributes."""
    _require_drawing(drawing_id)
    settings = get_settings()
    return store.search_text(drawing_id, q, limit=min(limit, settings.max_rows))


@app.get("/drawings/{drawing_id}/spatial", tags=["entities"])
def spatial(
    drawing_id: str,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    layout: str | None = Query(
        None,
        description=(
            "Restrict to one layout. Without it the query spans model space, "
            "every paper sheet and every block definition at once — which for "
            "a location question is almost never what was meant."
        ),
    ),
    layer: str | None = None,
    type: str | None = None,
    limit: int = Query(100, ge=1),
) -> dict[str, Any]:
    """Entities whose bounding box overlaps the given rectangle."""
    _require_drawing(drawing_id)
    if maxx < minx or maxy < miny:
        raise ApiError(
            "INVALID_BBOX",
            f"Degenerate rectangle: [{minx}, {miny}, {maxx}, {maxy}].",
            "Pass minx <= maxx and miny <= maxy, in drawing units.",
        )
    settings = get_settings()
    return store.spatial_query(
        drawing_id,
        bbox=[minx, miny, maxx, maxy],
        layout_name=layout,
        layer=layer,
        dxftype=type,
        limit=min(limit, settings.max_rows),
    )


@app.get("/drawings/{drawing_id}/distinct", tags=["entities"])
def distinct(
    drawing_id: str,
    field: str = Query(
        ...,
        description="text, layer, type, layout or block_name.",
    ),
    layer: str | None = None,
    layout: str | None = None,
    type: str | None = None,
    top: int = Query(25, ge=1, le=200),
) -> dict[str, Any]:
    """How many distinct values a field takes, plus the commonest ones.

    Answers "how many different text strings are in model space" in one call.
    Without it the only route was to page through every row and count in the
    caller — 56 requests and about a million characters to produce one
    integer.
    """
    _require_drawing(drawing_id)
    try:
        return store.distinct_values(
            drawing_id,
            field,
            layer=layer,
            layout_name=layout,
            dxftype=type,
            top=top,
        )
    except ValueError as exc:
        raise ApiError(
            "INVALID_FIELD",
            str(exc),
            "Countable fields are: text, layer, type, layout, block_name.",
        ) from exc


def _region_points(payload: RegionIn) -> list[tuple[float, float]]:
    """Validate a region and return its vertices as (x, y) pairs.

    Shared by every region endpoint on purpose: a rectangle accepted by the
    summary and rejected by the row listing (or vice versa) would show a
    person a count they could not then open.
    """
    points = [(p[0], p[1]) for p in payload.points if len(p) >= 2]
    if len(points) != len(payload.points):
        raise ApiError(
            "INVALID_REGION",
            "Every vertex must have at least two numbers, [x, y].",
            "Send drawing coordinates, not screen pixels.",
        )
    # NaN and infinity are accepted by JSON parsers and destroy every
    # comparison downstream in silence: NaN makes each test False, so the
    # region matches nothing and looks empty, while infinity matches the whole
    # drawing. Both come back 200 OK. They are rejected here, loudly.
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ApiError(
                "INVALID_REGION",
                f"Vertex ({x}, {y}) is not a finite number.",
                "Send real drawing coordinates; NaN and Infinity are not points.",
            )
        if abs(x) > MAX_COORDINATE or abs(y) > MAX_COORDINATE:
            raise ApiError(
                "INVALID_REGION",
                f"Vertex ({x}, {y}) is beyond the {MAX_COORDINATE:g} limit.",
                "CAD coordinates far outside this range overflow the polygon "
                "arithmetic to infinity, which silently matches nothing.",
            )
    if payload.kind == "rect":
        if len(points) != 2:
            raise ApiError(
                "INVALID_REGION",
                f"kind='rect' takes exactly two opposite corners, got {len(points)}.",
                "Send [[minx, miny], [maxx, maxy]], or use kind='polygon'.",
            )
        # Two corners in any order; the store normalises via the bbox.
        (x0, y0), (x1, y1) = points
        if x0 == x1 or y0 == y1:
            raise ApiError(
                "INVALID_REGION",
                "The rectangle has zero width or height.",
                "Drag a region with some area before releasing.",
            )
    elif payload.kind == "circle":
        if len(points) != 2:
            raise ApiError(
                "INVALID_REGION",
                f"kind='circle' takes exactly two points, got {len(points)}.",
                "Send [[centre_x, centre_y], [rim_x, rim_y]]; the radius is "
                "the distance between them.",
            )
        (cx, cy), (rx, ry) = points
        if (cx, cy) == (rx, ry):
            # A click with no drag. Left through, a zero-radius circle in
            # crossing mode still matches whatever bounding box contains that
            # one point, which looks like a working selection of unrelated
            # objects rather than the mis-gesture it is.
            raise ApiError(
                "INVALID_REGION",
                "The circle has zero radius.",
                "Press at the centre and drag outwards before releasing.",
            )
    elif len(points) < 3:
        raise ApiError(
            "INVALID_REGION",
            f"kind='polygon' needs at least three vertices, got {len(points)}.",
            "A region with fewer than three vertices encloses no area.",
        )
    elif _shoelace_area(points) == 0.0:
        # A zero-area rectangle was already rejected; a zero-area polygon was
        # not, so three clicks on one spot -- an easy mis-gesture with the
        # polygon tool -- returned a credible-looking selection of whatever
        # bounding boxes happened to touch that point.
        raise ApiError(
            "INVALID_REGION",
            "The polygon encloses no area: its vertices are identical or collinear.",
            "Place at least three vertices that are not on one line.",
        )
    return points


#: Largest coordinate magnitude a region vertex may carry.
#:
#: Not a taste limit. The polygon tests multiply differences of coordinates,
#: so near the float ceiling they overflow to infinity and every comparison
#: becomes False -- a polygon enclosing the whole drawing then reports zero
#: matches with a 200 OK. 1e12 is ten million kilometres in millimetres, far
#: beyond any real drawing and far below where the arithmetic breaks.
MAX_COORDINATE = 1e12


def _shoelace_area(points: list[tuple[float, float]]) -> float:
    """Twice the signed area of a polygon; zero means it encloses nothing."""
    total = 0.0
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return abs(total)


@app.post("/drawings/{drawing_id}/selection", tags=["entities"])
def select_region(drawing_id: str, payload: RegionIn = Body(...)) -> dict[str, Any]:
    """Entities inside — or touching — a region, summarised before listed.

    POST rather than GET for a read, deliberately: a polygon is up to 200
    vertices, and packing those into a query string means a length limit, an
    encoding to invent, and a URL nobody can read in a log. The body is also
    exactly what the viewer re-sends when the user flips window/crossing.

    The response leads with counts per layer and per type. A region drawn over
    a site plan routinely matches thousands of entities, and a raw list of
    those is unusable by a person and unaffordable for a model — so the list
    of handles is capped while the counts stay exact.
    """
    _require_drawing(drawing_id)
    points = _region_points(payload)
    if payload.layout:
        _require_layout(drawing_id, payload.layout)

    try:
        result = store.select_region(
            drawing_id,
            points=points,
            kind=payload.kind,
            mode=payload.mode,
            layout=payload.layout,
            max_handles=payload.max_handles,
        )
    except store.TooManyCandidates as exc:
        raise ApiError(
            "REGION_TOO_LARGE",
            str(exc),
            "Draw a smaller polygon, or use a rectangle — rectangles are "
            "answered entirely in the database and have no such limit.",
            status=413,
        ) from exc
    return result


@app.post("/drawings/{drawing_id}/selection/rows", tags=["entities"])
def select_region_rows(
    drawing_id: str, payload: RegionRowsIn = Body(...)
) -> dict[str, Any]:
    """One (layer, type) group of a region, as rows.

    Called when a person opens a group in the selection summary. Everything
    above that point is counts; this is the only call that returns entities,
    and it returns one group of them at a time.
    """
    _require_drawing(drawing_id)
    points = _region_points(payload)
    if payload.layout:
        _require_layout(drawing_id, payload.layout)
    try:
        return store.select_region_rows(
            drawing_id,
            points=points,
            kind=payload.kind,
            mode=payload.mode,
            layer=payload.layer,
            dxftype=payload.type,
            layout=payload.layout,
            offset=payload.offset,
            limit=payload.limit,
        )
    except store.TooManyCandidates as exc:
        raise ApiError(
            "REGION_TOO_LARGE",
            str(exc),
            "Draw a smaller polygon, or use a rectangle.",
            status=413,
        ) from exc


@app.post("/drawings/{drawing_id}/selections", tags=["entities"])
def save_selection(
    drawing_id: str, payload: SaveSelectionIn = Body(...)
) -> dict[str, Any]:
    """Store a selection once and get a reference to it.

    The reference is what travels to the agent from here on. Passing the
    handle list itself meant re-sending ~16 KB with every message of a
    conversation and capping it at 2,000, so larger selections were answered
    partially. Stored, the tools read the whole thing server-side.

    Selections expire by themselves (TTL index); nothing accumulates.
    """
    _require_drawing(drawing_id)
    if payload.layout:
        _require_layout(drawing_id, payload.layout)
    counts = None
    if payload.total is not None or payload.by_layer or payload.by_type:
        counts = {
            "total": payload.total,
            "by_layer": [r.model_dump() for r in (payload.by_layer or [])],
            "by_type": [r.model_dump() for r in (payload.by_type or [])],
        }
    return store.save_selection(
        drawing_id,
        payload.handles,
        layout=payload.layout,
        mode=payload.mode,
        kind=payload.kind,
        summary=payload.summary,
        counts=counts,
    )


@app.post("/drawings/{drawing_id}/selection/describe", tags=["entities"])
def describe_selection(
    drawing_id: str, payload: DescribeSelectionIn = Body(...)
) -> dict[str, Any]:
    """Aggregate description of an explicit set of handles.

    The endpoint behind the `describe_selection` MCP tool. It exists so that a
    model handed a selection receives its *shape* — how many, on which layers,
    of which types, how long, how large, and where — instead of receiving the
    entities. Anything it then wants in detail it fetches one handle at a time
    through `get_entity`, and every claim it makes stays traceable to a handle.
    """
    _require_drawing(drawing_id)

    handles = payload.handles
    stored_summary: str | None = None
    # Initialised beside its sibling, not inside the branch that fills it.
    # Left unbound it raised UnboundLocalError on the OTHER path -- describe by
    # explicit handles -- turning a working call into a 500. That path is not
    # rare: the viewer probes a sheet layout with it on every region attempt,
    # and cad-mcp falls back to it when the agent has no selection id. The
    # failure was silent from the browser, which is how it survived a test
    # suite that only exercised the stored-selection route.
    stored_counts: dict[str, Any] | None = None
    if payload.selection_id:
        stored = store.get_selection(drawing_id, payload.selection_id)
        if stored is None:
            raise ApiError(
                "SELECTION_NOT_FOUND",
                f"No selection {payload.selection_id!r} for drawing {drawing_id!r}.",
                "Selections expire after a day, and they are scoped to one "
                "drawing. Ask the user to make the selection again.",
                status=404,
            )
        handles = stored.get("handles", [])
        stored_summary = stored.get("summary")
        stored_counts = stored.get("counts")

    if not handles:
        raise ApiError(
            "EMPTY_SELECTION",
            "Neither a stored selection nor a handle list was given.",
            "Pass selection_id (preferred) or handles.",
        )

    result = store.describe_handles(
        drawing_id, handles, sample=payload.sample, exact=stored_counts
    )
    if payload.selection_id:
        result["selection_id"] = payload.selection_id
        # `covers` used to be the constant string "the entire stored
        # selection". That was true when it was written and became a lie the
        # moment a selection could exceed the enumeration ceiling: it said
        # "entire" over a list holding 5,000 of 8,122. It is computed now, and
        # it distinguishes the two things that were being conflated -- how
        # many objects the selection HAS, and how many of them can be NAMED.
        total = result.get("selection_total")
        named = result.get("found", 0)
        if isinstance(total, int) and total > named:
            result["covers"] = (
                f"{named} of {total} objects can be named individually; the "
                f"counts and breakdowns above are exact for all {total}"
            )
        elif isinstance(total, int):
            result["covers"] = "the entire stored selection"
        else:
            # No exact total came with the selection, which happens only when
            # the viewer could not compute one it was willing to stand behind
            # (overlapping parts, one of them capped). Saying "entire" here
            # would be a guess dressed as a fact.
            result["covers"] = (
                f"the {named} objects saved with this selection; whether that "
                f"is all of them was not established when it was saved"
            )
        if stored_summary:
            result["selection_summary"] = stored_summary
    return result


def _require_layout(drawing_id: str, layout: str) -> None:
    """Fail loudly on a layout name that does not exist in this drawing.

    Without this a typo silently returns zero entities, which reads as "the
    region is empty" — the most misleading answer this API can give.
    """
    drawing = store.get_drawing(drawing_id) or {}
    names = [l.get("name") for l in drawing.get("layouts", [])]
    if layout not in names:
        raise ApiError(
            "LAYOUT_NOT_FOUND",
            f"Drawing {drawing_id!r} has no layout named {layout!r}.",
            f"Layouts in this drawing: {names}.",
            status=404,
        )


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


@app.get("/drawings/{drawing_id}/measure", tags=["entities"])
def measure(
    drawing_id: str,
    measure: str = Query(..., description="'length' or 'area'."),
    layout: str = Query(..., description="Required. A total means nothing without a scope."),
    layer: str | None = None,
    type: str | None = None,
    block_name: str | None = None,
    group_by: str | None = Query(None, description="'layer' or 'type'."),
    top: int = Query(25, ge=1, le=200),
) -> dict[str, Any]:
    """Sum length or area over a filter, with the denominator that makes it readable.

    The quantity take-off endpoint. Before it existed the only way to total a
    layer was to page every entity and add them up, which one agent attempted
    in 136 calls and abandoned; the same figure is a single aggregation.

    `layout` is required rather than optional, and that is the one deliberate
    friction here. Unscoped, a sum spans model space, every paper sheet and
    every block definition at once, so the same geometry is counted more than
    once and the answer looks fine. "How long is layer X" genuinely has a
    different answer per layout, and a caller who has not chosen one has not
    finished asking the question.
    """
    _require_drawing(drawing_id)
    _require_layout(drawing_id, layout)
    try:
        return store.measure_entities(
            drawing_id,
            measure=measure,
            layout_name=layout,
            layer=layer,
            dxftype=type,
            block_name=block_name,
            group_by=group_by,
            top=top,
        )
    except store.MeasureRefused as exc:
        # Refusals carry the corrected call in `hint`. A refusal a caller
        # cannot act on is worse than no tool at all: it sends an agent back
        # to fetching entities one at a time.
        raise ApiError(exc.code, exc.message, exc.hint) from exc


@app.get("/comments", tags=["comments"], response_model=None)
def get_comments(
    drawing_id: str = Query(..., description="Drawing whose comments to return."),
    handle: str | None = Query(None, description="Narrow to one entity."),
) -> dict[str, Any]:
    """Comments for a drawing, optionally for a single entity."""
    _require_drawing(drawing_id)
    rows = store.list_comments(drawing_id, handle)
    return {"drawing_id": drawing_id, "handle": handle, "total": len(rows), "comments": rows}


@app.post("/comments", tags=["comments"], status_code=201, response_model=CommentOut)
def post_comment(payload: CommentIn = Body(...)) -> dict[str, Any]:
    """Anchor a comment to one entity.

    The original DWG is never touched — comments live in MongoDB, keyed by
    `(drawing_id, entity_handle)`. Sending the same `client_request_id` twice
    returns the first comment instead of creating a duplicate.
    """
    _require_drawing(payload.drawing_id)
    entity = store.get_entity(payload.drawing_id, payload.handle)
    if entity is None:
        raise ApiError(
            "ENTITY_NOT_FOUND",
            f"Handle {payload.handle!r} does not exist in drawing "
            f"{payload.drawing_id!r}; refusing to anchor a comment to nothing.",
            "Take the handle from a query result or from a click in the viewer.",
            status=404,
        )
    try:
        return store.add_comment(
            payload.drawing_id,
            payload.handle,
            payload.body,
            author=payload.author,
            anchor=payload.anchor,
            client_request_id=payload.client_request_id,
            provenance=payload.provenance or {"tool": "http", "source": "cad-api"},
        )
    except PyMongoError as exc:
        log.exception("comment write failed")
        raise ApiError(
            "STORAGE_ERROR",
            f"Could not store the comment: {type(exc).__name__}.",
            "The database rejected the write. Check /ready and retry.",
            status=503,
        ) from exc


# ---------------------------------------------------------------------------
# UPLIFT — Wave 3 wiring
#
# The eight Wave 2 subagents wrote 4,556 lines of machinery across five
# modules, and not one of them was allowed to touch this file. That ban was
# right: eight people editing one file is eight merge conflicts. The
# consequence is that none of that machinery could be called by anyone until
# this section was written.
#
# Units are handled two ways, and the difference is deliberately recorded here
# so that it does not read as an inconsistency. `store_stats` and
# `store_geometry` RECEIVE `units` from these routes, because neither of them
# may import `.store` -- `store.py` imports them at its own bottom precisely
# to break the import cycle. `store_spatial` and `store_landuse` fetch them
# themselves through a deferred import inside the function, which resolves the
# same cycle another way.
#
# What does not differ: not one response may go out without its units. Eleven
# of the eighteen drawings in this store are in inches and three state nothing
# at all, so a number without a unit here is not a number that is less
# complete -- it is a number that is wrong in most of the drawings.
# ---------------------------------------------------------------------------


@app.exception_handler(store.store_schedule.ScheduleError)
async def _schedule_refusal(
    _request: Request, exc: store.store_schedule.ScheduleError
) -> JSONResponse:
    """A drawing with no readable schedule is an answer, not a fault.

    Most drawings carry no drawn table at all, and the ones that do may carry
    a table whose columns say nothing a comparison can use. Both are ordinary,
    and both deserve a sentence rather than a stack trace.
    """
    return JSONResponse(
        status_code=400,
        content={"error": exc.code, "message": exc.message, "hint": exc.hint},
    )


@app.exception_handler(store.store_spatial.SpatialRefused)
async def _spatial_refusal(
    _request: Request, exc: store.store_spatial.SpatialRefused
) -> JSONResponse:
    """A spatial refusal is an answer, not an infrastructure failure.

    A separate class from `MeasureRefused` because `store_spatial` may not
    import `.store`; its shape is deliberately identical so that a caller
    never has to know the difference.
    """
    return JSONResponse(
        status_code=400,
        content={"error": exc.code, "message": exc.message, "hint": exc.hint},
    )


@app.exception_handler(store.store_stats.StatsRefused)
async def _stats_refusal(
    _request: Request, exc: store.store_stats.StatsRefused
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": exc.code, "message": exc.message, "hint": exc.hint},
    )


@app.exception_handler(store.store_geometry.DuplicateRefused)
async def _duplicate_refusal(
    _request: Request, exc: store.store_geometry.DuplicateRefused
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": exc.code, "message": exc.message, "hint": exc.hint},
    )


@app.exception_handler(store.store_landuse.LandUseConfigInvalid)
async def _landuse_refusal(
    _request: Request, exc: store.store_landuse.LandUseConfigInvalid
) -> JSONResponse:
    """A broken land-use config is a 400, not a 500.

    The difference is not cosmetic: a caller retries a 500 and reads a 400. A
    mistyped config file has a way forward that can be acted on, and serving
    it as an infrastructure failure hides that.
    """
    return JSONResponse(
        status_code=400,
        content={"error": exc.code, "message": exc.message, "hint": exc.hint},
    )


@app.exception_handler(recipes.RecipeRefused)
async def _recipe_refusal(
    _request: Request, exc: recipes.RecipeRefused
) -> JSONResponse:
    """A recipe refusal is an answer, not an infrastructure failure.

    A missing parameter, a population that is too large, and a drawing that
    does not yet have a land-use config all have a way forward that can be
    acted on; serving them as a 500 hides that, and a caller retries a 500
    while it reads a 400.
    """
    return JSONResponse(
        status_code=400,
        content={"error": exc.code, "message": exc.message, "hint": exc.hint},
    )


@app.exception_handler(store.store_recipes.TagRefused)
async def _tag_refusal(
    _request: Request, exc: store.store_recipes.TagRefused
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": exc.code, "message": exc.message, "hint": exc.hint},
    )


def _units_for(drawing_id: str, layout: str) -> dict[str, Any]:
    """Units for one layout, handed verbatim to the spec modules."""
    drawing = store.get_drawing(drawing_id) or {}
    return store._unit_names(drawing, layout)


@app.get("/drawings/{drawing_id}/join-labels", tags=["uplift"])
def join_labels(
    drawing_id: str,
    layout: str = Query(..., description="Required. A pairing across layouts means nothing."),
    target_layer: str | None = None,
    label_layer: str | None = None,
    label_type: str = "TEXT",
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Which text sits INSIDE which polygon -- its shape, not its box.

    This is what closes the 23 August failure: asked for a plot's area after
    clicking its number, the agent answered about its text entity. A bbox test
    on one parcel pulled in 42 neighbouring texts; point-in-polygon leaves the
    one that is genuinely inside it.

    Only `ring_status == "complete"` polygons become targets. The rest are
    skipped AND counted, broken down by cause -- a target that was skipped is
    not a target that did not match.
    """
    _require_drawing(drawing_id)
    _require_layout(drawing_id, layout)
    return store.store_spatial.join_labels(
        drawing_id,
        layout=layout,
        target_layer=target_layer,
        label_layer=label_layer,
        label_type=label_type,
        limit=limit,
    )


@app.get("/drawings/{drawing_id}/distance-matrix", tags=["uplift"])
def distance_matrix(
    drawing_id: str,
    layout: str = Query(...),
    layers: str | None = Query(None, description="Comma-separated."),
    handles: str | None = Query(None, description="Comma-separated."),
    mode: str = Query("centroid", description="'centroid' or 'edge'."),
    cluster_threshold: float | None = Query(
        None,
        description=(
            "The distance that separates two clusters. Has no default: that is "
            "a project's planning figure, not a property of a CAD drawing."
        ),
    ),
) -> dict[str, Any]:
    """Distances between objects, every combination, in one call.

    At the 23 August meeting this question was very slow and then stalled. The
    distance is measured from `polygon_centroid`, not the centre of the box:
    for large polygons the two can be tens of metres apart.

    The response states that this is a straight line. A distance that does not
    say so is read as a walking distance.
    """
    _require_drawing(drawing_id)
    _require_layout(drawing_id, layout)
    hs = [h.strip() for h in handles.split(",")] if handles else None
    ls = [l.strip() for l in layers.split(",")] if layers else None
    return store.store_spatial.distance_matrix(
        drawing_id,
        layout=layout,
        handles=hs,
        layers=ls,
        mode=mode,
        cluster_threshold=cluster_threshold,
    )


@app.get("/drawings/{drawing_id}/proximity-count", tags=["uplift"])
def proximity_count(
    drawing_id: str,
    layout: str = Query(...),
    around_layers: str = Query(..., description="Comma-separated. The objects surrounded."),
    count_layers: str = Query(..., description="Comma-separated. The objects counted."),
    radius: float = Query(..., description="In drawing units. The units are echoed in the response."),
    measure_from: str = Query("centroid", description="'centroid' or 'edge'."),
) -> dict[str, Any]:
    """How many objects surround each other object, within a given radius.

    This is the question that went missing at the 23 August meeting --
    "vicinity of houses around each school" -- because the session stalled
    before it could be answered.

    `radius` does not bury a unit inside its name. Eleven of the eighteen
    drawings are in inches; a parameter named `radius_m` would lie in most of
    them.
    """
    _require_drawing(drawing_id)
    _require_layout(drawing_id, layout)
    return store.store_spatial.proximity_count(
        drawing_id,
        layout=layout,
        around_layers=[l.strip() for l in around_layers.split(",")],
        count_layers=[l.strip() for l in count_layers.split(",")],
        radius=radius,
        measure_from=measure_from,
    )


@app.get("/drawings/{drawing_id}/land-use", tags=["uplift"])
def land_use(
    drawing_id: str,
    layout: str = Query(...),
) -> dict[str, Any]:
    """This drawing's land use, per type, with the grade of its evidence.

    This is what closes the hardest failure of the meeting: asked how many
    houses, the agent answered "maybe there are no houses" on a drawing
    holding 2,380 plots.

    Every row carries `evidence`. Houses come out as `inferred` and stay that
    way however much ground there is for them, because the file genuinely
    never states it -- and that is what a person must read, not have hidden.
    """
    _require_drawing(drawing_id)
    _require_layout(drawing_id, layout)
    return store.store_landuse.land_use_summary(drawing_id, layout=layout)


@app.get("/drawings/{drawing_id}/stats", tags=["uplift"])
def stats(
    drawing_id: str,
    layout: str = Query(...),
    measure: str = Query("area", description="'area' or 'length'."),
    layers: str | None = Query(None, description="Comma-separated. Empty = every layer."),
    type: str | None = None,
) -> dict[str, Any]:
    """Mean, median, mode and standard deviation -- per layer, not mixed.

    At the 23 August meeting, "average area per type" was answered outside the
    agent by opening the DXF directly. This is what lets the agent answer it
    itself.

    Objects that cannot be measured never enter as zero; `basis` names the
    denominator, and the remainder is counted and stated.
    """
    _require_drawing(drawing_id)
    _require_layout(drawing_id, layout)
    ls = [l.strip() for l in layers.split(",")] if layers else None
    if ls is None:
        drawing = store.get_drawing(drawing_id) or {}
        ls = [l.get("name") for l in drawing.get("layers", []) if l.get("name")]
    return store.store_stats.stats_by_layer(
        drawing_id,
        measure=measure,
        layout_name=layout,
        layers=ls,
        dxftype=type,
        units=_units_for(drawing_id, layout),
    )


@app.get("/drawings/{drawing_id}/duplicates", tags=["uplift"])
def duplicates(
    drawing_id: str,
    layout: str = Query(...),
    kind: str = Query("geometry", description="'geometry' or 'shape'."),
    tolerance: float = Query(0.01, description="Drawing units; echoed in the response."),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Polygons drawn twice, and shapes that repeat.

    These are two different questions and `kind` deliberately separates them.
    A repeating shape is a hallmark of a masterplan, not a defect; a
    coincident copy is a defect, and it makes every plot number fall in two
    polygons at once.
    """
    _require_drawing(drawing_id)
    _require_layout(drawing_id, layout)
    return store.store_geometry.find_duplicates(
        drawing_id,
        layout=layout,
        units=_units_for(drawing_id, layout),
        kind=kind,
        tolerance=tolerance,
        limit=limit,
    )


@app.get("/drawings/{drawing_id}/shapes", tags=["uplift"])
def shapes(
    drawing_id: str,
    layout: str = Query(...),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Plot modules read from the geometry itself, with no layer names.

    This is what makes the reading survive the 17th drawing: another
    contractor will use layer names that are entirely different, but a 12x25
    plot is still a 12x25 plot.
    """
    _require_drawing(drawing_id)
    _require_layout(drawing_id, layout)
    return store.store_geometry.shape_fingerprint(
        drawing_id, layout=layout, units=_units_for(drawing_id, layout), limit=limit
    )


@app.get("/drawings/{drawing_id}/tables", tags=["uplift"])
def tables(drawing_id: str) -> dict[str, Any]:
    """The drawing's dictionary: layer, text style, linetype, dim style,
    header, page setup.
    """
    _require_drawing(drawing_id)
    return store.store_tables.drawing_tables(drawing_id)


@app.get("/drawings/{drawing_id}/find-by-name", tags=["uplift"])
def find_by_name(
    drawing_id: str,
    q: str = Query(..., description="The name to look for; a partial one will do."),
    kinds: str | None = Query(None, description="Comma-separated."),
) -> dict[str, Any]:
    """Search names in the drawing's tables, not in its text.

    `search_text` searches the annotations; this searches the dictionary. A
    layer that is in the table and is never used will never be found by the
    first.
    """
    _require_drawing(drawing_id)
    ks = [k.strip() for k in kinds.split(",")] if kinds else None
    return store.store_tables.find_by_name(drawing_id, q, kinds=ks)


@app.get("/drawings/{drawing_id}/tags", tags=["uplift"])
def list_tags(drawing_id: str) -> dict[str, Any]:
    """Business names people have already given layers in this drawing."""
    _require_drawing(drawing_id)
    return {"drawing_id": drawing_id, "tags": store.store_recipes.list_tags(drawing_id)}


@app.get("/drawings/{drawing_id}/tags/export.yaml", tags=["uplift"])
def export_tags_yaml(drawing_id: str) -> Response:
    """This drawing's tags as config-shaped YAML, ready to review and commit.

    This is the last step of a flow that was only half built: tag from the
    screen, save to Mongo, then -- here -- emit it as a file a person can
    read, diff, and put into the repo. Without this step every business
    decision lives in one shared database that has no review history, and the
    next drawing starts from zero again.

    The file is deterministic and carries no timestamp, so exporting twice
    with no change produces an identical file. A spurious diff is a diff that
    stops being read.

    It holds for any drawing: what is exported are that drawing's own layer
    names, and not one layer name is written into the code.
    """
    _require_drawing(drawing_id)
    rows = store.store_recipes.list_tags(drawing_id)
    body = store.store_recipes.export_yaml(drawing_id, rows)
    return Response(
        content=body,
        media_type="application/yaml",
        headers={
            # `inline` and not `attachment`: the file is short and the first
            # thing a person wants to do is READ it before deciding to save
            # it. The filename is still stated, so "save as" still lands with
            # the right name.
            "Content-Disposition": f'inline; filename="{drawing_id}.yaml"',
        },
    )


@app.get("/drawings/{drawing_id}/tags/{layer}/impact", tags=["uplift"])
def tag_impact(drawing_id: str, layer: str, layout: str | None = None) -> dict[str, Any]:
    """How many objects would be touched if this layer were named.

    Shown BEFORE the save, because "name it once, it applies to all its plots"
    is only safe if the person sees how many "all" is.
    """
    _require_drawing(drawing_id)
    return store.store_recipes.layer_impact(drawing_id, layer, layout)


@app.post("/drawings/{drawing_id}/tags/{layer}", tags=["uplift"], status_code=201)
def put_tag(drawing_id: str, layer: str, body: dict[str, Any]) -> dict[str, Any]:
    """Give one layer a business name, and state where the name came from.

    `source` is required and has no default value. A tag without a provenance
    is a claim without evidence, and the UPLIFT-08 contract exists precisely
    to refuse that -- including when the one claiming is a human.
    """
    _require_drawing(drawing_id)
    return store.store_recipes.put_layer_tag(
        drawing_id=drawing_id,
        layer=layer,
        business_name=body.get("business_name", ""),
        source=body.get("source", ""),
        author=body.get("author", "unknown"),
        use=body.get("use"),
    )


@app.post("/drawings/{drawing_id}/geometry/points", tags=["uplift"])
def geometry_points(drawing_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Points for a batch of handles at once, so the viewer can mark them.

    One call, not two hundred. A page whose measured defect is precisely that
    it stalls must not pay one round trip per object.

    `polygon_centroid` is what comes back, not `bbox_centre`: for large
    polygons the two are tens of metres apart, and a marker that far off lands
    in the neighbouring plot.
    """
    _require_drawing(drawing_id)
    handles = [str(h) for h in (body.get("handles") or [])][:2000]
    if not handles:
        raise ApiError(
            "NO_HANDLES",
            "The handle list is empty.",
            "Send {\"handles\": [...]} with at least one handle.",
        )
    ids = [f"{drawing_id}:{h}" for h in handles]
    rows = list(
        mongo.coll(mongo.COLL_ENTITIES).find(
            {"_id": {"$in": ids}},
            {
                "handle": 1,
                "polygon_centroid": 1,
                "bbox_centre": 1,
                "ring_status": 1,
                "layer": 1,
            },
        )
    )
    found = {r.get("handle") for r in rows}
    return {
        "drawing_id": drawing_id,
        "points": [
            {
                "handle": r.get("handle"),
                "layer": r.get("layer"),
                "polygon_centroid": r.get("polygon_centroid"),
                "bbox_centre": r.get("bbox_centre"),
                "ring_status": r.get("ring_status"),
            }
            for r in rows
        ],
        "missing": sorted(h for h in handles if h not in found),
        "point_basis": (
            "polygon_centroid where there is one, otherwise bbox_centre. Both "
            "are included so the caller knows which one it is using."
        ),
    }


@app.get(
    "/drawings/{drawing_id}/geo",
    tags=["uplift"],
    summary="Cells and parcels for a map",
    response_description=(
        "`cells` for a hexagon layer, `parcels` as a GeoJSON FeatureCollection, "
        "and `totals` for the WHOLE drawing even when a viewport is applied. "
        "`georeferenced: false` with a reason for a drawing that has no "
        "coordinate system — the same shape, so no second branch is needed."
    ),
)
def drawing_geo(
    drawing_id: str,
    res: int | None = Query(
        None,
        description=(
            "H3 resolution to aggregate at. Defaults to the resolution the "
            "drawing's cells are stored at. Coarser is derived; finer is "
            "refused, because a finer cell is not a better answer than a "
            "coarser one, it is an invented one."
        ),
    ),
    layout: str = Query(
        "Model",
        description=(
            "The layout to read. Only model space has real-world coordinates: "
            "a sheet's numbers are page geometry and a block definition's are "
            "the block's own frame, so neither has cells at all."
        ),
    ),
    parcels: bool = Query(
        True, description="Include the GeoJSON FeatureCollection of parcels."
    ),
    compact: bool = Query(
        False,
        description=(
            "Also return the compacted COVERAGE — the set of cells the parcels "
            "occupy, merged to the coarsest cells that still describe it. The "
            "counted cells are never compacted: a count cannot survive a merge."
        ),
    ),
    dimension: str | None = Query(
        None,
        description=(
            "What the cells are coloured and counted BY. Omit to let the "
            "drawing decide: land use where its config classifies parcels, "
            "layer where nothing does. Pass 'layer' to see a drawing whose "
            "config classifies only a handful of its shapes. The response's "
            "`dimension` block always says which was used, what it covers and "
            "what else is available."
        ),
    ),
    limit: int = Query(
        geo_view.MAX_PARCEL_FEATURES,
        ge=1,
        le=geo_view.MAX_PARCEL_FEATURES,
        description="Parcel features returned before the collection is truncated.",
    ),
    outline: bool = Query(
        False,
        description=(
            "Also return `coverage_outline`: the ground the parcels' cells "
            "cover, dissolved into ONE MultiPolygon boundary. For drawing the "
            "site's extent without drawing 25,627 hexagons. Its edges are "
            "hexagon edges — it is not a surveyed boundary."
        ),
    ),
    bbox: str | None = Query(
        None,
        description=(
            "Restrict `cells` and `parcels` to a viewport, as "
            "`west,south,east,north` in degrees — longitude first, the order a "
            "GeoJSON `bbox` uses. `totals` is NOT restricted: it stays the "
            "whole drawing, so a filtered view can never be read as the site. "
            "Example: bbox=46.89,24.85,46.90,24.86"
        ),
    ),
    highlight: str | None = Query(
        None,
        description=(
            "Comma-separated handles. Every cell in the response gains "
            "`highlighted`, true for the cells those objects sit in — the way "
            "an agent's answer becomes a set of hexagons to light. The grid "
            "is NOT filtered: the surrounding cells stay in the response, "
            "because the point of the flag is to stand out against them. "
            "Tokens that are not objects in this drawing are counted in "
            "`highlight.handles_not_objects` rather than reported as missing."
        ),
    ),
    request: Request = None,  # type: ignore[assignment]
    response: Response = None,  # type: ignore[assignment]
) -> Any:
    """Cells to colour and parcels to draw, for a map.

    Additive in every direction: it reads the cells written at ingest, adds no
    field to any other response, and touches no existing route.

    Cacheable, and the ETag is computed BEFORE the aggregation rather than
    from its result. That ordering is the whole value of it: a client that
    already holds the answer gets a 304 having cost two small reads, instead
    of paying for 2,576 parcels to be grouped and reprojected so the bytes can
    be thrown away.
    """
    _require_drawing(drawing_id)

    if res is not None and not 0 <= res <= 15:
        raise ApiError(
            "H3_RESOLUTION_OUT_OF_RANGE",
            f"res={res} is not an H3 resolution.",
            "H3 defines 0 to 15. Omit `res` to get the resolution this "
            "drawing's cells are stored at.",
        )

    # The cache key. Everything the answer depends on and nothing else: the
    # drawing's content hash, the config version behind its classification,
    # the resolution its cells are stored at, and the normalised parameters.
    # See `geo_view.ETAG_INPUTS` for why the last three are not redundant.
    # Parsed before the ETag, so a malformed viewport is a 400 rather than a
    # cache entry, and so the box that is hashed is the normalised one.
    try:
        box = geo_view.parse_bbox(bbox)
    except geo_view.GeoBadRequest as exc:
        raise ApiError(exc.code, exc.message, exc.hint) from exc

    # Normalised before it reaches the ETag so that `?highlight=b,a` and
    # `?highlight=A,B` are one cache entry rather than two, and so the value
    # hashed is the value the aggregation will actually use.
    marks = sorted(
        {token.strip().upper() for token in (highlight or "").split(",") if token.strip()}
    )

    # This route is also called as a plain Python function -- several suites
    # do exactly that rather than going through the ASGI stack -- and then the
    # defaults arrive as FastAPI's own `Query(...)` objects instead of the
    # values they wrap. A string is the only thing the dimension check can
    # honestly judge, so anything else is "not asked for".
    chosen_dimension = dimension if isinstance(dimension, str) else None

    params = {
        "res": res,
        "layout": layout,
        "parcels": parcels,
        "compact": compact,
        # In the ETag because it CHANGES THE BODY. Left out at first, and the
        # browser held a 1-hexagon answer against a server that had 68: the
        # revalidation returned 304 for a different question. Any parameter
        # `geo()` reads belongs here.
        "dimension": chosen_dimension,
        "limit": limit,
        "outline": outline,
        "bbox": (
            None
            if box is None
            else [box.west, box.south, box.east, box.north]
        ),
        "highlight": marks or None,
    }
    facts = geo_view.preconditions(drawing_id)
    tag = geo_view.etag_for(drawing_id, params=params, **facts)
    if request is not None and geo_view.etag_matches(
        request.headers.get("if-none-match"), tag
    ):
        # 304 carries no body by definition, and it carries the tag so the
        # client's copy stays validated for the next request too.
        return Response(
            status_code=304, headers={"ETag": tag, "Cache-Control": "no-cache"}
        )
    if response is not None:
        response.headers["ETag"] = tag
        # `no-cache` means "revalidate before reusing", not "do not store".
        # Without it a browser keeps nothing and never sends `If-None-Match`,
        # so the ETag is computed, served and ignored — which is what it was
        # doing: a return to an already-visited drawing refetched the whole
        # payload and got a 200 every time.
        response.headers["Cache-Control"] = "no-cache"

    try:
        body = geo_view.geo(
            drawing_id,
            res=res,
            layout=layout,
            parcels=parcels,
            compact=compact,
            dimension=chosen_dimension,
            limit=limit,
            bbox=box,
            outline=outline,
            highlight=marks,
        )
    except geo_view.GeoNotIndexed as exc:
        # Georeferenced and not yet indexed. Deliberately NOT folded into the
        # branch below: "this drawing has no coordinate system" and "this
        # drawing has no cells yet" call for different actions by different
        # people, and one shared `georeferenced: false` would send both to the
        # wrong one.
        return {
            "drawing_id": drawing_id,
            "georeferenced": True,
            "indexed": False,
            "reason": exc.reason,
            "how_to_fix": exc.how_to_fix,
            "cells": [],
            "parcels": {"type": "FeatureCollection", "features": []},
            "totals": {"parcels": 0, "counts": {}, "area_m2": None},
        }
    except geo_view.GeoUnavailable as exc:
        # An explicit answer, not an empty success. The shape is the same one
        # the caller gets for a drawing that CAN be mapped, so a client needs
        # no second branch to read it -- `georeferenced: false` is what tells
        # them apart, and the reason says what would have to change.
        #
        # 200 rather than a 4xx on purpose, and for the same reason
        # `land_use_available: false` is a 200: the request was well formed and
        # the drawing was found. What is missing is a coordinate system, which
        # is a fact about the drawing rather than a fault in the call.
        return {
            "drawing_id": drawing_id,
            "georeferenced": False,
            "indexed": False,
            "reason": exc.reason,
            "how_to_fix": (
                "give this drawing a CRS: either it carries a GEODATA object "
                "the extractor can read, or somebody writes a `crs:` block in "
                "its config in cad_api/app/landuse/. No zone is ever guessed."
            ),
            "cells": [],
            "parcels": {"type": "FeatureCollection", "features": []},
            "totals": {"parcels": 0, "counts": {}, "area_m2": None},
        }

    except geo_view.GeoBadRequest as exc:
        # The store's own refusal, handed over with its code and its hint
        # rather than reworded here. Two wordings of one rule drift.
        raise ApiError(exc.code, exc.message, exc.hint) from exc
    return body


@app.get(
    "/drawings/{drawing_id}/geo/export.geojson",
    tags=["uplift"],
    summary="The map as one GeoJSON file",
    response_description=(
        "A styled FeatureCollection: parcels coloured by land use, hexagon "
        "cells as an unfilled overlay, and the coordinate-system caveat on "
        "every feature. Drag it into geojson.io. `georeferenced: false` with "
        "a reason for a drawing that has no coordinate system."
    ),
)
def drawing_geo_export(
    drawing_id: str,
    res: int | None = Query(
        None,
        description=(
            "H3 resolution for the cell overlay. Defaults to the resolution "
            "the drawing's cells are stored at; coarser is derived, finer is "
            "refused."
        ),
    ),
    layout: str = Query("Model", description="The layout to read."),
    cells: bool = Query(True, description="Include the hexagon overlay."),
    parcels: bool = Query(True, description="Include the parcel polygons."),
    bbox: str | None = Query(
        None,
        description=(
            "Restrict the features to a viewport, `west,south,east,north` in "
            "degrees. The file still reports the whole drawing's totals."
        ),
    ),
    limit: int = Query(
        geo_view.MAX_PARCEL_FEATURES,
        ge=1,
        le=geo_view.MAX_PARCEL_FEATURES,
        description="Parcel features before the collection is truncated.",
    ),
    handles: str | None = Query(
        None,
        description=(
            "Comma-separated handles: export only these objects. This is what "
            "turns an agent answer into a file — the handles it named are the "
            "handles it marked. The cell overlay follows them, and "
            "`totals_whole_drawing` still describes the whole drawing."
        ),
    ),
    rings: int = Query(
        0,
        ge=0,
        le=geo_view.MAX_EXPORT_RINGS,
        description=(
            "Also draw a catchment: this many H3 rings around the exported "
            "objects' own cells, dissolved into one outline. Only meaningful "
            "with `handles`. 2 rings is about 100 m at resolution 11 — the "
            "same disk `h3_vicinity` counts inside."
        ),
    ),
    context: bool = Query(
        False,
        description=(
            "Also draw the REST of the site's grid, faint, as "
            "`kind: \"cell_context\"`. The answer's own cells keep "
            "`kind: \"cell\"` and a stronger style, so the file renders as "
            "bright clusters on a faint grid instead of a handful of hexagons "
            "floating on an empty map. Only meaningful with `handles`: "
            "without them the export is already the whole site."
        ),
    ),
    request: Request = None,  # type: ignore[assignment]
    response: Response = None,  # type: ignore[assignment]
) -> Any:
    """One file, ready to drag into a map.

    It is `/geo` with the colours a viewer reads and the caveat written into
    every feature. The second half is the reason it is a separate route
    rather than a flag: a file gets forwarded, and it arrives with no route,
    no documentation and nobody to ask. Whatever it needs to be read
    correctly has to be inside it.

    The download name is set so a browser saves something recognisable rather
    than `export.geojson` from four different drawings.
    """
    _require_drawing(drawing_id)

    if res is not None and not 0 <= res <= 15:
        raise ApiError(
            "H3_RESOLUTION_OUT_OF_RANGE",
            f"res={res} is not an H3 resolution.",
            "H3 defines 0 to 15. Omit `res` for the resolution this drawing's "
            "cells are stored at.",
        )
    try:
        box = geo_view.parse_bbox(bbox)
    except geo_view.GeoBadRequest as exc:
        raise ApiError(exc.code, exc.message, exc.hint) from exc

    wanted = [h.strip() for h in (handles or "").split(",") if h.strip()]
    if len(wanted) > geo_view.MAX_EXPORT_HANDLES:
        raise ApiError(
            "TOO_MANY_HANDLES",
            f"{len(wanted)} handles were asked for; the limit is "
            f"{geo_view.MAX_EXPORT_HANDLES}.",
            "Export the whole drawing instead of naming every object in it: "
            "omit `handles`. A list that long is the drawing.",
        )

    params = {
        "export": True,
        "res": res,
        "layout": layout,
        "cells": cells,
        "parcels": parcels,
        "limit": limit,
        "bbox": (None if box is None else [box.west, box.south, box.east, box.north]),
        # Sorted, so that the same answer exported twice is one cache entry
        # however the client happened to order its handles.
        "handles": sorted(h.upper() for h in wanted) or None,
        "rings": rings,
        "context": context,
    }
    facts = geo_view.preconditions(drawing_id)
    tag = geo_view.etag_for(drawing_id, params=params, **facts)
    if request is not None and geo_view.etag_matches(
        request.headers.get("if-none-match"), tag
    ):
        return Response(status_code=304, headers={"ETag": tag})

    try:
        body = geo_view.export_geojson(
            drawing_id,
            res=res,
            layout=layout,
            bbox=box,
            cells=cells,
            parcels=parcels,
            limit=limit,
            handles=wanted or None,
            rings=rings,
            context=context,
        )
    except geo_view.GeoBadRequest as exc:
        raise ApiError(exc.code, exc.message, exc.hint) from exc
    except geo_view.GeoNotIndexed as exc:
        # A file with no features and no explanation is a file somebody
        # forwards, opens on an empty map, and concludes the site is empty
        # from. So the explanation is IN it.
        return JSONResponse(
            status_code=200,
            headers={"ETag": tag},
            content={
                "type": "FeatureCollection",
                "features": [],
                "georeferenced": True,
                "indexed": False,
                "reason": exc.reason,
                "how_to_fix": exc.how_to_fix,
            },
        )
    except geo_view.GeoUnavailable as exc:
        # The same shape a refusal takes on /geo, so a caller needs no second
        # branch — and an empty FeatureCollection, which is what a map would
        # otherwise draw as "this site has nothing in it".
        return JSONResponse(
            status_code=200,
            headers={"ETag": tag},
            content={
                "type": "FeatureCollection",
                "features": [],
                "georeferenced": False,
                "indexed": False,
                "reason": exc.reason,
                "how_to_fix": (
                    "give this drawing a CRS: either it carries a GEODATA "
                    "object the extractor can read, or somebody writes a "
                    "`crs:` block in its config in cad_api/app/landuse/. No "
                    "zone is ever guessed."
                ),
            },
        )

    return JSONResponse(
        content=body,
        headers={
            "ETag": tag,
            # A recognisable name on disk. Four drawings exported in one
            # afternoon are otherwise four files called export.geojson.
            "Content-Disposition": (
                f'attachment; filename="{drawing_id}-'
                + ("answer" if wanted else "geo")
                + f'-res{body["properties"]["resolution"]}.geojson"'
            ),
        },
    )


@app.get("/drawings/{drawing_id}/entities/{handle}/containing", tags=["uplift"])
def containing(
    drawing_id: str,
    handle: str,
    layout: str | None = None,
    label_type: str = "TEXT",
    limit: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """How this entity relates to the polygons around it -- both directions.

    `contained_by` answers the question that failed at the 23 August meeting:
    someone clicked a PLOT NUMBER, asked for its area, and was answered about
    its TEXT entity. From the label to the parcel that holds it, then to that
    parcel's area.

    `labels` answers the other direction -- what text is inside this polygon
    -- for when it is the parcel that was clicked.

    **Both keys are ALWAYS present, even when one of them is empty.** That is
    not a leniency; it is the fix for the defect that had this route rewritten.
    For a whole day the docstring here promised the first direction while the
    code called `labels_inside`, and its reader in the viewer read
    `contained_by` out of a response that never carried it. A response shape
    that changes according to what was clicked is a shape every caller of it
    has to guess at; this one does not.
    """
    _require_drawing(drawing_id)
    spatial = store.store_spatial
    containers = spatial.parcels_containing(
        drawing_id, handle, layout=layout, limit=limit
    )
    inside = spatial.labels_inside(
        drawing_id, handle, label_type=label_type, limit=limit
    )
    note = _containing_note(containers, inside)
    return {
        **containers,
        "labels": inside.get("labels", []),
        "labels_total": inside.get("total"),
        "labels_not_measured": inside.get("not_measured"),
        "note": note,
    }


def _containing_note(containers: dict[str, Any], inside: dict[str, Any]) -> str | None:
    """One sentence a person reads and the agent is sent, from one place.

    If the panel and the model each composed their own sentence, the two would
    differ on the day one of them was changed, and what the user reads would no
    longer be what the model trusts.
    """
    rows = containers.get("contained_by") or []
    selected = containers.get("selected") or {}
    if rows:
        first = rows[0]
        text = selected.get("text")
        what = f'this {selected.get("type") or "entity"}'
        if text:
            what += f' (reading "{text}")'
        # The count is pluralised because English marks plurals and the
        # Indonesian this sentence was translated from does not. Left alone,
        # an entity sitting inside exactly two polygons reads "1 other, larger
        # polygons" -- a sentence that makes a careful reader distrust the
        # number rather than the grammar.
        others = len(rows) - 1
        return (
            f"{what} sits inside {first.get('handle')} on layer "
            f"{first.get('layer')!r}"
            + (
                f", and {others} other, larger polygon{'' if others == 1 else 's'}"
                if others
                else ""
            )
            + ". A question about area, size, or type is almost certainly "
            "about ITS PARCEL, not about the entity that was clicked."
        )
    if inside.get("labels"):
        total = inside.get("total")
        return (
            f"this entity is a polygon, and it holds {total} "
            f"label{'' if total == 1 else 's'} inside it."
        )
    if containers.get("not_measured"):
        return containers["not_measured"]
    return (
        "no complete-ring polygon holds this entity's insertion point in this "
        "layout. This is the result of a check, not the absence of one."
    )


@app.get("/analyses", tags=["uplift"])
def list_analyses() -> dict[str, Any]:
    """Analysis recipe catalogue: name, when used, parameters, output shape.

    Deliberately NOT under `/drawings/{id}`: the catalogue is the same for
    every drawing, and a catalogue that asked for a drawing_id would invite
    its reader to think its contents change per drawing.
    """
    return recipes.catalog()


@app.post("/drawings/{drawing_id}/analyses/{recipe}", tags=["uplift"])
def run_analysis(
    drawing_id: str,
    recipe: str,
    layout: str = Query(..., description="Required for almost every recipe."),
    params: dict[str, Any] | None = Body(default=None),
) -> dict[str, Any]:
    """Run one already-reviewed analysis recipe, server-side.

    An unrecognised recipe name returns the CATALOGUE, not a 404: a caller
    handed "unknown recipe" will force the tool that looks closest and produce
    a wrong number in the right tone.

    There is no way to hand logic in through here. What is accepted is only a
    recipe name and its parameter values, and that limit is deliberate: the
    drawing's contents come from a contractor, and this process holds the
    shared database credentials.
    """
    _require_drawing(drawing_id)
    _require_layout(drawing_id, layout)
    units = _units_for(drawing_id, layout)
    result = recipes.run(drawing_id, recipe, params or {}, layout=layout, units=units)
    return _retry_if_nothing_established(result, drawing_id, layout, units)


#: How many times one call may retry itself. Re-exported so the constant has
#: one home: the rule and its execution live in `recipes.selfheal`, which the
#: composition recipe uses too — a retry that happens on one route and not the
#: other is the same recipe giving two answers to the same question.
MAX_AUTO_RETRIES: Final[int] = selfheal.MAX_AUTO_RETRIES


def _retry_if_nothing_established(
    result: dict[str, Any], drawing_id: str, layout: str, units: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-run once, server-side, when a recipe settled nothing and said how.

    The rule itself is in `recipes.selfheal`, because this route is not the
    only caller of a recipe: `combine_findings` runs two of them in-process,
    and while the retry lived here alone the composition received the un-healed
    answer and intersected it. See that module for what a retry may and may not
    do, and why it is executed rather than described to a model.
    """
    return selfheal.heal(
        result, drawing_id=drawing_id, layout=layout, units=units, run=recipes.run
    )


# ---------------------------------------------------------------------------
# PROVISIONAL — the deck.gl map views
# ---------------------------------------------------------------------------
#
# One include line and one module (`geometry_map.py`), kept apart from the
# route families above on purpose: this was written by the frontend to unblock
# the map work and has not been reviewed by the backend team. Reviewing it is
# reading one file; rejecting it is deleting one file and this line. The
# contract it implements, and the questions it leaves open, are in
# docs/DECKGL-GEOMETRY-CONTRACT.md.
#
# Imported here at the BOTTOM rather than beside the others because
# `geometry_map` raises `ApiError`, which is defined above — the module
# imports it inside its own function bodies so that the cycle never forms at
# import time, and mounting it last keeps that ordering visible.
from . import geometry_map  # noqa: E402  (see the note above)

app.include_router(geometry_map.router)
