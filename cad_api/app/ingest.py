"""Ingest DXF files: extract -> store -> render -> profile -> cache.

Run against every file in the mounted DXF directory:

    python -m app.ingest --all

Or one file, forcing a re-render:

    python -m app.ingest --file "architectural_example-imperial.dxf" --force

Idempotent. `drawing_id` is the content hash, so re-running over the same
files overwrites each drawing's own documents and creates nothing new; files
already stored at the current `INGEST_VERSION` are skipped unless `--force`.

A failure on one file is reported and the run continues — with 18 files, one
bad DXF must not cost the other 17.

The arrival path (DOSSIER Phase 7)
----------------------------------
Until Phase 7 a drawing ingested today sat in the store with **no Dossier and
no drafted land-use config** until somebody remembered to run two scripts by
hand. Every tool answered honestly — "not computed" — and every one of those
answers was useless. The owner's requirement was never the diff engine; it was
that *a new version that arrives is understood without anyone remembering to
run anything*. So ingest now finishes by:

1.  **building the drawing's Dossier** (`scripts/dossier_backfill.py` when this
    machine has it, and the same `app.dossier.build_dossier` call directly when
    it does not — see `profile_on_arrival`);
2.  **drafting its land-use config** into the review directory
    (`app/landuse_draft.py`, Phase 3), which classifies nothing on its own and
    is not written anywhere the loader reads;
3.  **looking for a predecessor** by the stated rule — same drawing name,
    different content hash — and, when there is exactly one, running the
    revision diff and caching its headline with the new Dossier. Zero
    candidates or several are RECORDED as such and listed. Ambiguity is
    reported here, never resolved by picking one.

Everything in that block is idempotent: the Dossier is keyed by the content
hash and replaced in place, and the draft file is named after the same hash and
overwritten. Re-ingesting the same file produces the same three artefacts.

**None of it may sink an ingest.** A drawing that is stored but not yet
profiled is a recoverable state — the next run, or the backfill, completes it.
An ingest that dies because a profiler raised has cost the entity data as well,
and that is the expensive half. So each step is caught on its own, the failure
is put in the result, in the warnings, and in the printed report, and the run
continues. Silence would be the one unacceptable outcome; failure is not.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from . import ingest_runs, mongo, store
from .config import get_settings
from .convert import ConversionError, convert, is_available
from .extract import ExtractionError, extract, is_block_layout, load_document
from .logging_conf import configure_logging
from .export import read_embedded_comments
from .mongo import COLL_DOSSIERS, coll
from .render import EmptyRenderError, render_layout_svg

log = logging.getLogger(__name__)

#: Environment variable naming `scripts/dossier_backfill.py` explicitly. Read
#: first, so an operator whose layout matches none of the guesses below can say
#: where it is instead of losing the containment sample silently.
BACKFILL_ENV: str = "CAD_DOSSIER_BACKFILL"

#: Where the backfill script is looked for, in order. The image copies `app`
#: and nothing else (`cad_api/Dockerfile`), so inside the service this list
#: finds nothing unless the operator mounted `scripts/` — which is exactly what
#: `scripts/onboard_drawing.py` already documents. Missing is a supported
#: state, not an error: see `_build_dossier`.
BACKFILL_CANDIDATES: tuple[str, ...] = (
    "/scripts/dossier_backfill.py",
    "//scripts/dossier_backfill.py",
    "/src/scripts/dossier_backfill.py",
)


@dataclass
class FileResult:
    """What happened to one file. Failures are first-class, not swallowed."""

    filename: str
    ok: bool
    drawing_id: str | None = None
    entity_count: int = 0
    layer_count: int = 0
    block_count: int = 0
    layouts_rendered: list[dict[str, Any]] = field(default_factory=list)
    units: str | None = None
    extents: dict[str, list[float]] | None = None
    audit_errors: int = 0
    skipped: bool = False
    error: str | None = None
    elapsed_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    #: What the arrival path managed to do for this file (DOSSIER Phase 7).
    #: `None` means it was not attempted — `--no-profile`, or a file that
    #: failed before it had a drawing id. It never means "nothing to report":
    #: an attempt that failed comes back as a block whose `ok` is false and
    #: whose `error` says why, so a failure cannot wear the same clothes as a
    #: step that was never asked for.
    arrival: dict[str, Any] | None = None


#: How many BLOCK-definition layouts one ingest will render. A block layout is
#: a convenience view of a definition, not a sheet anyone drew, and the real
#: layouts are never capped. Sedra is why the cap exists: it brings 487 block
#: layouts against 21 real ones, and rendering all of them on a 500 MB
#: document is what turned a large ingest into an unbounded one.
MAX_BLOCK_LAYOUT_RENDERS: Final[int] = 50

#: How many entities one layout may EXPAND to before rendering it is refused.
#:
#: A layout's own entity count says nothing about the cost of drawing it: an
#: INSERT is one entity and can carry half a million. Sedra's model space is
#: 19 entities that expand to roughly 990,000, and rendering it took the
#: process past a 15.47 GiB ceiling and got it killed by the OOM killer, after
#: extract and store had both succeeded.
#:
#: Refusing loudly is better than dying: a killed process takes the whole
#: ingest with it and leaves a run that says `running` for ever, while a
#: refusal costs one sheet and says which one and why. Janadriyah's largest
#: layout expands to about 46,754, so this clears every drawing measured so
#: far by more than five times.
MAX_RENDER_EXPANSION: Final[int] = 250_000


def _renderable_layouts(
    drawing_doc: dict[str, Any],
) -> tuple[list[str], int, list[tuple[str, int]]]:
    """Layouts worth rendering, what was capped, and what was too big to draw.

    An empty layout renders to a blank SVG, which in the drawing picker is
    indistinguishable from a broken render, so a layout with no entities is
    not offered.

    Real layouts come FIRST and are never capped: they are the sheets somebody
    drew, and they are what a reviewer opens. Block-definition layouts follow
    and are capped, because a drawing whose content is all inside blocks can
    carry hundreds of them and each one costs a full render of a document that
    may be 500 MB. The number left out is returned rather than discarded, so
    the caller can say it instead of the drawing quietly having fewer sheets
    than it should.
    """
    real: list[str] = []
    blocks: list[str] = []
    too_big: list[tuple[str, int]] = []
    for layout in drawing_doc.get("layouts", []):
        if layout.get("entity_count", 0) <= 0:
            continue
        name = str(layout["name"])
        # Measured at extract time. Absent on a drawing ingested before that,
        # where the layout's own count is the best available guide and the
        # old behaviour is kept.
        cost = layout.get("expanded_entity_count", layout.get("entity_count", 0))
        if cost > MAX_RENDER_EXPANSION:
            too_big.append((name, int(cost)))
            continue
        (blocks if is_block_layout(name) else real).append(name)
    kept_blocks = blocks[:MAX_BLOCK_LAYOUT_RENDERS]
    return real + kept_blocks, len(blocks) - len(kept_blocks), too_big


# =============================================================================
# The arrival path — DOSSIER Phase 7
# =============================================================================


def _backfill_module() -> tuple[Any, str]:
    """`scripts/dossier_backfill.py`, when this machine has it, and where from.

    Loaded by PATH rather than imported, because `scripts/` is not a package
    and nothing about being callable from here should force it to become one.
    The script is integrator-owned and is not restructured to suit this call:
    loading it and calling its `build_for` is how the arrival path and the
    catch-up backfill stay ONE profiler. A second implementation would drift,
    and nothing in either Dossier would say which one wrote it.

    Returns `(module, where)` or `(None, why_not)`. Absent is a supported
    state — the image ships `app` and nothing else — and the caller degrades
    to `app.dossier.build_dossier` directly, saying what that costs.
    """
    named = os.environ.get(BACKFILL_ENV, "").strip()
    candidates: list[Path] = [Path(named)] if named else []
    candidates += [Path(p) for p in BACKFILL_CANDIDATES]
    # A checkout: app/ingest.py -> app -> cad_api -> <repo>/scripts/...
    candidates.append(Path(__file__).resolve().parents[2] / "scripts" / "dossier_backfill.py")
    candidates.append(Path.cwd() / "scripts" / "dossier_backfill.py")

    tried: list[str] = []
    for candidate in candidates:
        try:
            if not candidate.is_file():
                tried.append(str(candidate))
                continue
        except OSError:  # a path this process may not even stat
            tried.append(str(candidate))
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                "dossier_backfill_for_ingest", candidate
            )
            if spec is None or spec.loader is None:
                tried.append(f"{candidate} (no loader)")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "build_for"):
                tried.append(f"{candidate} (no build_for)")
                continue
            return module, str(candidate)
        except Exception as exc:  # noqa: BLE001 -- reported, never fatal
            tried.append(f"{candidate} ({type(exc).__name__}: {exc})")
    return None, (
        "scripts/dossier_backfill.py was not found on this machine, so the "
        "Dossier was assembled directly from app.dossier instead. Looked in: "
        + ", ".join(tried[:6])
        + f". Set {BACKFILL_ENV} to point at it, or mount scripts/ into the "
        "container as scripts/onboard_drawing.py already documents."
    )


#: The entity fields the profiler reads, mirroring
#: `scripts/dossier_backfill.PROJECTION`. It is only used on the fallback path
#: — when the script IS present its own projection is what runs — and it is
#: written out rather than reaching into the script so that the fallback works
#: when the script is the very thing that is missing. Fetched explicitly and
#: not as whole documents: an entity carries ring vertex arrays no profiler
#: looks at, and 46,754 of those is the difference between a query that fits in
#: memory and one that does not.
_FALLBACK_PROJECTION: dict[str, int] = {
    "handle": 1,
    "type": 1,
    "layer": 1,
    "layout": 1,
    "block_name": 1,
    "text": 1,
    "length": 1,
    "area": 1,
    "bbox": 1,
    "bbox_centre": 1,
    "ring_status": 1,
    "shape_key": 1,
    "anchor_point": 1,
    "polygon_centroid": 1,
    "attribs": 1,
}


def _build_dossier_without_the_script(drawing_doc: dict[str, Any]) -> dict[str, Any]:
    """The Dossier, assembled here, for a machine that has no `scripts/`.

    It is the same `app.dossier.build_dossier` call the backfill makes, over
    the same rows, with one thing missing and said out loud: the backfill also
    resolves a small SAMPLE of each annotation layer's labels to the region
    layers they sit inside, which is the routing fact that lets a question
    naming a plot number be recognised as a question about a parcel. That step
    is Mongo-backed, integrator-owned, and not duplicated here.

    So a drawing profiled on this path is fully profiled — every bucket, role,
    measure, anomaly, census and the coverage invariant — and carries a stated
    gap where the sample would be, with the one command that fills it.
    """
    # The engine now lives in the package, so the container HAS it and this
    # fallback should almost never run. Try it first anyway: a Dossier that
    # differs depending on who built it is worse than one that is merely late,
    # and the sampled label-to-parcel routing was the part that went missing
    # when the engine could only be reached through `scripts/`.
    try:  # noqa: PLC0415 -- deferred with the rest
        from . import dossier_backfill

        return dossier_backfill.build_for(drawing_doc)
    except Exception:  # noqa: BLE001 -- fall through, never fail an ingest
        log.debug("in-package backfill unavailable; assembling here", exc_info=True)

    from . import dossier as dossier_mod  # noqa: PLC0415 -- deferred
    from .mongo import COLL_ENTITIES  # noqa: PLC0415

    drawing_id = str(drawing_doc.get("_id"))
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in coll(COLL_ENTITIES).find(
        {"drawing_id": drawing_id}, _FALLBACK_PROJECTION
    ):
        key = (str(row.get("layer") or ""), str(row.get("layout") or ""))
        buckets.setdefault(key, []).append(row)

    units: dict[str, Any] = {}
    for entry in drawing_doc.get("layouts") or []:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if not name:
            continue
        try:
            units[str(name)] = store._unit_names(dict(drawing_doc), str(name))
        except Exception as exc:  # noqa: BLE001 -- reported, never guessed
            units[str(name)] = {
                "name": None,
                "why_no_unit": f"could not be read: {exc}",
            }

    document = dossier_mod.build_dossier(
        drawing_doc,
        buckets,
        entity_total=drawing_doc.get("entity_count"),
        units_by_layout=units,
        computed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    document["label_containment_sample_status"] = {
        "computed": False,
        "why": (
            "this Dossier was built by the ingest arrival path on a machine "
            "without scripts/dossier_backfill.py, which is where the label "
            "containment sample is computed. Every bucket, role, measure and "
            "the coverage invariant are complete; what is missing is the "
            "layer-level routing fact that says which region layers an "
            "annotation layer's labels sit inside."
        ),
        "how_to_add": (
            "python scripts/dossier_backfill.py --drawing "
            f"{drawing_id} — it reads the entities already in Mongo and "
            "rewrites this document in place. No DXF is re-read."
        ),
    }
    return document


def _revision_block(
    drawing_id: str, dossier: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """Predecessor detection and, when there is exactly one, the diff headline.

    The rule and its outcome are recorded on the Dossier whatever happens —
    including "no candidate" and "several candidates", because a drawing whose
    predecessor was ambiguous and a drawing that was never checked are
    different states and must not read the same.

    Only the HEADLINE of the diff is cached (G7): the full comparison is
    recomputed on demand by the `revision_diff` recipe from the two Dossiers,
    which are the source of truth. That also means a backfill re-run, which
    replaces this document wholesale, costs nothing but the cache.
    """
    from .recipes import revision as rev  # noqa: PLC0415 -- deferred; heavy

    found = rev.predecessor_for(drawing_id)
    block: dict[str, Any] = {
        "detected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": found.get("rule"),
        "rule_limits": found.get("rule_limits"),
        "drawing_name": found.get("drawing_name"),
        "predecessor": found.get("predecessor"),
        "candidate_count": found.get("candidate_count"),
        "candidates": [
            str(row.get("drawing_id")) for row in found.get("candidates") or []
        ],
        "ambiguous": bool(found.get("ambiguous")),
        "why": found.get("why"),
        "diff_summary": None,
        "diff_is_a_cache": (
            "only the headline is stored. `revision_diff` recomputes the whole "
            "comparison from the two Dossiers on demand, so this block is a "
            "convenience and never the source of truth — and a backfill "
            "re-run, which replaces this document, costs nothing but the cache."
        ),
    }

    other_id = found.get("predecessor")
    if not other_id:
        return block, None

    other = coll(COLL_DOSSIERS).find_one({"_id": str(other_id)})
    if other is None:
        block["diff_not_run"] = (
            f"the predecessor {other_id!r} has no Dossier, so there was "
            "nothing to compare against. That is NOT a statement that it is "
            "empty."
        )
        return block, block["diff_not_run"]

    diff = rev.diff_dossiers(other, dossier)
    totals = diff["totals"]
    block["diff_summary"] = {
        "against": str(other_id),
        "entity_total": totals["entity_total"],
        "buckets": totals["buckets"],
        "layers_added": totals["layers"]["added"][:20],
        "layers_removed": totals["layers"]["removed"][:20],
        "units_agree": diff["units"]["agree"],
        "measure_arithmetic": diff["units"]["measure_arithmetic"],
        "file_facts_changed": diff["file_facts_changed"]["changed_total"],
        "verdict": diff["verdict"],
    }
    return block, None


def profile_on_arrival(
    drawing_id: str,
    *,
    draft_dir: Path | None = None,
    rebuild: bool = True,
    run: Any | None = None,
) -> dict[str, Any]:
    """Build the Dossier, draft the land-use config, look for a predecessor.

    The whole of DOSSIER Phase 7's second half, in one function, so that the
    same three things happen whether a drawing arrives through `--file`,
    through `--all`, or through `scripts/onboard_drawing.py`.

    `rebuild=False` builds the Dossier only when the drawing has none. That is
    the skip path: a file already ingested at the current version is not
    re-profiled for nothing, but a file that was ingested before this hook
    existed gains its Dossier the next time anything sweeps it.

    **This function never raises.** Each step is caught on its own and reported
    in the block it returns; the caller decides nothing except whether to print
    it. A drawing stored without a profile is recoverable — the backfill, or
    the next ingest, completes it. An ingest that died inside a profiler has
    lost the entity data too, and that is the expensive half.
    """
    started = time.perf_counter()
    out: dict[str, Any] = {
        "drawing_id": drawing_id,
        "ok": False,
        "dossier": {"ok": False, "built": False, "why": None, "source": None},
        "landuse_draft": {"ok": False, "path": None, "why": None},
        "revision": {"ok": False, "why": None},
        "errors": [],
    }

    drawing_doc: dict[str, Any] | None = None
    try:
        drawing_doc = store.get_drawing(drawing_id)
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"could not read the drawing document: {exc!r}")
        log.exception("arrival: drawing read failed", extra={"drawing_id": drawing_id})
    if drawing_doc is None:
        out["dossier"]["why"] = (
            f"there is no drawing document for {drawing_id!r}, so nothing "
            "could be profiled."
        )
        out["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return out

    # Each step says it has STARTED before it does its work.
    #
    # Without this the arrival stages were written only once every one of them
    # had finished, so for the whole time they were running the run reported
    # them `pending` with no `started_at`. On Sedra over Atlas that was 3 h
    # 21 min of real work -- a Dossier over 990,790 rows, then 679,897 H3
    # cells -- during which the run was indistinguishable from a dead one:
    # no timestamps, no log lines, nothing moving. It was read as a silent
    # death and was not one. A stage that is working must say so.
    def _begin(name: str) -> None:
        if run is not None:
            try:
                run.start(name)
            except Exception:  # noqa: BLE001 -- telemetry must never break work
                log.exception("could not mark stage running", extra={"stage": name})

    # --- 1. the Dossier ------------------------------------------------------
    _begin("dossier")
    dossier: dict[str, Any] | None = None
    try:
        existing = coll(COLL_DOSSIERS).find_one({"_id": drawing_id}, {"_id": 1})
        if existing is not None and not rebuild:
            out["dossier"] = {
                "ok": True,
                "built": False,
                "source": "already stored",
                "why": (
                    "a Dossier already exists for this content hash and the "
                    "file was not re-ingested, so it was left alone. "
                    "Re-running is safe — the id is a content hash — it is "
                    "simply not needed."
                ),
            }
            dossier = coll(COLL_DOSSIERS).find_one({"_id": drawing_id})
        else:
            module, where = _backfill_module()
            if module is not None:
                dossier = module.build_for(drawing_doc)
                source = f"scripts/dossier_backfill.build_for ({where})"
                degraded = None
            else:
                dossier = _build_dossier_without_the_script(drawing_doc)
                source = "app.dossier.build_dossier (arrival path)"
                degraded = where
            dossier["_id"] = drawing_id
            dossier["drawing_id"] = drawing_id
            coll(COLL_DOSSIERS).replace_one(
                {"_id": drawing_id}, dossier, upsert=True
            )
            coverage = dossier.get("coverage") or {}
            out["dossier"] = {
                "ok": True,
                "built": True,
                "source": source,
                "why": degraded,
                "buckets": len(dossier.get("layers") or []),
                "coverage_complete": coverage.get("complete"),
                "accounted": coverage.get("accounted"),
                "total": coverage.get("total"),
            }
            if coverage.get("complete") is not True:
                out["errors"].append(
                    "the Dossier was built but does not account for every "
                    f"entity in the drawing: {coverage.get('verdict')}"
                )
            if degraded:
                out["errors"].append(degraded)
    except Exception as exc:  # noqa: BLE001 -- an ingest must not die here
        out["dossier"] = {
            "ok": False,
            "built": False,
            "source": None,
            "why": f"{type(exc).__name__}: {exc}",
        }
        out["errors"].append(f"Dossier build failed: {type(exc).__name__}: {exc}")
        log.exception("arrival: dossier build failed", extra={"drawing_id": drawing_id})

    # --- 2. the predecessor, and the diff when there is exactly one ----------
    #
    # Done before the draft is written and folded into the SAME document write,
    # so a drawing never carries a Dossier that has not been asked the revision
    # question at all.
    if dossier is not None:
        try:
            block, why_no_diff = _revision_block(drawing_id, dossier)
            coll(COLL_DOSSIERS).update_one(
                {"_id": drawing_id}, {"$set": {"revision": block}}
            )
            out["revision"] = {
                "ok": True,
                "predecessor": block.get("predecessor"),
                "candidate_count": block.get("candidate_count"),
                "candidates": block.get("candidates"),
                "ambiguous": block.get("ambiguous"),
                "why": block.get("why"),
                "diff_summary": block.get("diff_summary"),
            }
            if why_no_diff:
                out["errors"].append(why_no_diff)
        except Exception as exc:  # noqa: BLE001
            out["revision"] = {
                "ok": False,
                "why": f"{type(exc).__name__}: {exc}",
            }
            out["errors"].append(
                f"predecessor detection failed: {type(exc).__name__}: {exc}"
            )
            log.exception(
                "arrival: predecessor detection failed",
                extra={"drawing_id": drawing_id},
            )

    # --- 3. the drafted land-use config --------------------------------------
    _begin("landuse_draft")
    #
    # After the Dossier on purpose: the draft reads it for each layer's
    # geometric role, and a draft written first would have every layer
    # name-only. It classifies nothing either way — its `layers:` block is
    # empty and `write_draft` refuses the config directory by name — so the
    # worst case is a review file with less in it than it could have had.
    try:
        from . import landuse_draft  # noqa: PLC0415 -- deferred

        draft = landuse_draft.draft_for_drawing(drawing_doc)
        path = landuse_draft.write_draft(draft, out_dir=draft_dir)
        counts = draft.get("counts") or {}
        out["landuse_draft"] = {
            "ok": True,
            "path": str(path),
            "layers": counts.get("layers"),
            "ready_to_accept": counts.get("ready_to_accept"),
            "needs_a_decision": counts.get("needs_a_decision"),
            "why": (
                "nothing in this file classifies anything: its `layers:` block "
                "is empty and every proposal beside it is commented out. A "
                "human uncommenting a line and moving the file into "
                "app/landuse/ is what makes any of it live."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        out["landuse_draft"] = {
            "ok": False,
            "path": None,
            "why": f"{type(exc).__name__}: {exc}",
        }
        out["errors"].append(
            f"land-use draft failed: {type(exc).__name__}: {exc}"
        )
        log.exception("arrival: landuse draft failed", extra={"drawing_id": drawing_id})

    # --- 4. H3 cells --------------------------------------------------------
    _begin("h3")
    #
    # Last, and after the entities are stored, because it reads them back. A
    # drawing with no coordinate system refuses here and that is the ordinary
    # case — seventeen of the eighteen in the store cannot be placed — so a
    # refusal is recorded with its reason and does NOT make the arrival fail.
    #
    # Run on every ingest rather than only on the first, because
    # `store.replace_entities` writes with `ReplaceOne`: re-ingesting a drawing
    # removes these fields, and the geometry may have changed underneath them
    # anyway.
    try:
        from . import geo_backfill  # noqa: PLC0415 -- deferred

        cells = geo_backfill.assign_cells(drawing_id)
        out["h3"] = {
            "ok": cells["ok"],
            "reason": cells.get("reason"),
            "resolution": cells.get("resolution"),
            "assigned": cells.get("assigned"),
            "world_placed": cells.get("world_placed"),
            "excluded": cells.get("excluded"),
            "excluded_by_reason": {
                reason: block["count"]
                for reason, block in (cells.get("excluded_by_reason") or {}).items()
            },
            "partial_coverage": len(cells.get("partial_coverage") or []),
        }
    except Exception as exc:  # pragma: no cover - depends on the environment
        out["h3"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        out["errors"].append(f"h3 cell assignment failed: {exc}")
        log.exception("arrival: h3 assignment failed", extra={"drawing_id": drawing_id})

    out["ok"] = (
        out["dossier"]["ok"] and out["landuse_draft"]["ok"] and out["revision"]["ok"]
    )
    out["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    log.info(
        "arrival profile",
        extra={
            "drawing_id": drawing_id,
            "dossier_ok": out["dossier"]["ok"],
            "draft_ok": out["landuse_draft"]["ok"],
            "revision_ok": out["revision"]["ok"],
            "errors": len(out["errors"]),
            "elapsed_ms": out["elapsed_ms"],
        },
    )
    return out


def _attach_arrival(result: FileResult, arrival: dict[str, Any]) -> None:
    """Put the arrival outcome where a reader will actually meet it.

    On the result, in the warnings, and in the log. A profiler failure that
    lived only in a dict nobody prints would be a silent failure with extra
    steps — and this whole phase exists because "not computed" was being
    reported politely and going unread.
    """
    result.arrival = arrival
    for message in arrival.get("errors") or []:
        result.warnings.append(f"arrival: {message}")


def _record_arrival_stages(
    run: ingest_runs.Recorder, arrival: Mapping[str, Any]
) -> None:
    """Copy the arrival report into three stages of the run history.

    `profile_on_arrival` already returns per-part results and catches its own
    failures; this only persists what it returned. Nothing is re-derived, so
    the history cannot disagree with the report the CLI prints.

    The revision block is not a stage. It is a fact ABOUT the drawing rather
    than a step in ingesting it, so it lands on the run document as
    `revision_of` instead.
    """
    dossier = arrival.get("dossier") or {}
    if dossier.get("ok"):
        run.finish(
            "dossier",
            built=dossier.get("built"),
            source=dossier.get("source"),
            accounted=dossier.get("accounted"),
            total=dossier.get("total"),
            coverage_complete=dossier.get("coverage_complete"),
        )
    else:
        run.fail("dossier", str(dossier.get("why") or "the Dossier was not built"))

    draft = arrival.get("landuse_draft") or {}
    if draft.get("ok"):
        run.finish(
            "landuse_draft",
            path=str(draft.get("path")) if draft.get("path") else None,
            layers=draft.get("layers"),
            ready_to_accept=draft.get("ready_to_accept"),
            needs_a_decision=draft.get("needs_a_decision"),
        )
    else:
        run.fail("landuse_draft", str(draft.get("why") or "no draft was written"))

    h3 = arrival.get("h3") or {}
    if h3.get("ok"):
        run.finish(
            "h3",
            resolution=h3.get("resolution"),
            assigned=h3.get("assigned"),
            excluded=h3.get("excluded"),
            excluded_by_reason=h3.get("excluded_by_reason"),
            world_placed=h3.get("world_placed"),
        )
    else:
        # Recorded as skipped rather than failed on purpose: H3 assignment is
        # deliberately excluded from the arrival `ok` (see `profile_on_arrival`),
        # because a drawing with no CRS has nothing to assign and that is a
        # correct outcome, not a fault. The reason is carried either way.
        run.skip("h3", str(h3.get("reason") or "no cells were assigned"))

    revision = arrival.get("revision") or {}
    run.set_revision_of(predecessor_block(revision.get("predecessor")))


def predecessor_block(drawing_id: Any) -> dict[str, Any] | None:
    """The run history's `revision_of` shape for a predecessor drawing id.

    `recipes.revision.find_predecessors` answers with an ID and nothing else,
    because the rule it implements is about identity: same name, different
    content hash. A person reading a progress panel needs a name and a date
    before that means anything, so those are LOOKED UP here rather than being
    re-derived from the rule. The rule stays the only thing that decides which
    drawing is the predecessor.
    """
    if not drawing_id:
        return None
    doc = store.get_drawing(str(drawing_id)) or {}
    return {
        "drawing_id": str(drawing_id),
        "filename": doc.get("original_filename"),
        "ingested_at": doc.get("ingested_at"),
    }


def ingest_file(
    path: Path,
    *,
    force: bool = False,
    profile: bool = True,
    recorder: ingest_runs.Recorder | None = None,
) -> FileResult:
    """Run the full pipeline for one DXF file.

    Args:
        path: the `.dxf` file to ingest.
        force: re-extract and re-render even if already stored.
        profile: run the arrival path afterwards — build the Dossier, draft
            the land-use config, look for a predecessor (DOSSIER Phase 7).
            `False` restores the behaviour from before that phase, which is
            useful for a bulk re-ingest that will be followed by one backfill
            over everything. It is not a way to make an ingest safer: the
            arrival path cannot fail an ingest either way.
        recorder: where to write the stage history. Defaults to a recorder
            that writes nothing, so every existing caller behaves exactly as
            before. The stages reported here are a TRANSCRIPT of the control
            flow below, not a second copy of it: nothing in this function
            changes order, and no branch exists only to be recorded.
    """
    settings = get_settings()
    started = time.perf_counter()
    result = FileResult(filename=path.name, ok=False)
    run = recorder or ingest_runs.NullRecorder()

    # A .dwg is converted in-process now; the pipeline no longer depends on
    # somebody having run a converter by hand beforehand.
    source_path = str(path)
    if path.suffix.lower() == ".dwg":
        run.start("convert", converter="pending")
        try:
            conversion = convert(path, settings.svg_dir.parent / "converted")
        except ConversionError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
            log.error("dwg conversion failed", extra={"dxf_file": path.name})
            # The I1 refusal already names both converters and the next human
            # step, so it is carried through verbatim rather than summarised.
            run.failed("convert", str(exc))
            return result
        if conversion.repaired_lines:
            result.warnings.append(
                f"repaired {conversion.repaired_lines} wrapped value line(s) in "
                "the converted DXF"
            )
        path = conversion.dxf_path
        run.finish(
            "convert",
            converter=conversion.converter,
            elapsed_ms=conversion.elapsed_ms,
            repaired_lines=conversion.repaired_lines,
            bytes_out=path.stat().st_size if path.exists() else None,
        )
    else:
        run.skip("convert", "already a .dxf, nothing to convert")

    run.start("extract")
    try:
        drawing, entities = extract(path, source_path=source_path)
    except ExtractionError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.error("ingest failed", extra={"dxf_file": path.name, "error": result.error})
        run.failed("extract", result.error)
        return result
    except Exception as exc:  # noqa: BLE001 - report, never hide
        result.error = f"unexpected {type(exc).__name__}: {exc}"
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.exception("ingest crashed", extra={"dxf_file": path.name})
        run.failed("extract", result.error)
        return result

    result.drawing_id = drawing._id
    result.entity_count = drawing.entity_count
    result.layer_count = len(drawing.layers)
    result.block_count = len(drawing.blocks)
    result.units = drawing.units_name
    result.extents = drawing.extents
    result.audit_errors = drawing.audit_errors
    result.warnings = list(drawing.warnings)

    run.set_drawing(drawing._id)
    run.finish(
        "extract",
        entities=drawing.entity_count,
        layers=len(drawing.layers),
        blocks=len(drawing.blocks),
        layouts=len(drawing.layouts),
        units=drawing.units_name,
        audit_errors=drawing.audit_errors,
        warnings=list(drawing.warnings),
    )

    # "Current" means both stored AND rendered. Checking only ingest_version
    # made a half-finished ingest permanent: the drawing document is written
    # before rendering, so a crash mid-render left a drawing that every later
    # run reported as SKIP/OK while GET /svg returned RENDER_NOT_FOUND for
    # ever.
    if not force and store.drawing_is_current(drawing._id, settings.svg_dir):
        result.ok = True
        result.skipped = True
        # A skipped file is still profiled when it has never been profiled.
        # The 18 drawings already in the store were ingested before this hook
        # existed, so without this branch a sweep would keep reporting them as
        # SKIP/OK while they stayed unprofiled for ever — the exact shape of
        # the bug the `drawing_is_current` comment above records. `rebuild` is
        # False, so a drawing that already has a Dossier costs one indexed
        # lookup and nothing else.
        if profile:
            _attach_arrival(
                result, profile_on_arrival(drawing._id, rebuild=False, run=run)
            )
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info("already ingested, skipping", extra={"dxf_file": path.name})
        # Recorded as `skipped`, not `done`: the drawing is usable, but
        # nothing after extract actually ran, and a history that said
        # otherwise would be a lie the next reader has to un-learn.
        run.already_current(
            drawing_id=drawing._id, elapsed_ms=result.elapsed_ms
        )
        return result

    run.start("store")
    try:
        store.upsert_drawing(drawing)
        # Between this line and the next the drawing exists and its entities
        # do not. Flagged for exactly that window so a failure cannot leave a
        # drawing in the picker that announces its entity count and draws
        # nothing.
        store.mark_ingest_incomplete(drawing._id)
        stored = store.replace_entities(drawing._id, entities)
        store.clear_ingest_incomplete(drawing._id)
    except Exception as exc:  # noqa: BLE001 - one bad file, not the whole run
        result.error = f"storage failed: {type(exc).__name__}: {exc}"
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.exception("storage failed", extra={"dxf_file": path.name})
        run.failed("store", result.error)
        return result
    run.finish("store", entities_written=stored)

    # Rendering re-opens the document. Extraction and rendering are separate
    # passes on purpose: extraction must survive a renderer failure, because
    # the entity data is what the agent and the comment layer depend on. A
    # drawing that is queryable but not yet drawable is a usable state.
    drawing_doc = store.get_drawing(drawing._id) or {}
    handle_types = {e.handle: e.type for e in entities}

    # The extracted entity objects have been stored and are not needed again,
    # and the very next step opens a SECOND copy of the same document. On
    # Sedra that meant holding 990,790 objects while a 500 MB file was parsed
    # again, and the ingest was OOM-killed at exactly this point (measured:
    # OOMKilled=true against a 15.47 GiB ceiling). Releasing them here is what
    # makes the render stage survivable on a large drawing.
    entities = []

    layout_names, blocks_skipped, too_big = _renderable_layouts(drawing_doc)
    if not layout_names:
        result.warnings.append(
            "no layout contains entities; nothing to render "
            "(content may live only in block definitions)"
        )
    for name, cost in too_big:
        result.warnings.append(
            f"layout {name!r} would draw {cost:,} entities once its blocks are "
            f"expanded, over the {MAX_RENDER_EXPANSION:,} render limit, so no "
            "sheet was produced for it. Every entity is still stored and "
            "queryable, and the map views read the geometry directly"
        )
    if blocks_skipped:
        result.warnings.append(
            f"{blocks_skipped} block-definition layout(s) were not rendered; "
            f"the cap is {MAX_BLOCK_LAYOUT_RENDERS}. The drawing's own sheets "
            "are all rendered, and every entity is stored and queryable either "
            "way"
        )

    # Opened once and reused across layouts. Janadriyah takes ~58 s just to
    # parse, and it has three layouts -- re-opening per layout would turn a
    # 75 s ingest into a 190 s one for no benefit. The renderer only reads.
    doc = None
    if layout_names:
        try:
            doc, _errors, _fixes = load_document(path)
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(
                f"could not reopen for rendering: {type(exc).__name__}: {exc}"
            )
            log.exception("reopen for render failed", extra={"dxf_file": path.name})

    # Round-trip: a file that was exported by this system (or a copy of it)
    # carries its comments as ROSHN_COMMENTS XDATA. Read them back in, so
    # "give the file to someone else" and "they see the previous comments"
    # holds even when the file re-enters through ingest. Idempotent: the
    # deterministic client_request_id collapses re-imports of the same
    # comment on re-ingest.
    if doc is not None:
        try:
            embedded = read_embedded_comments(doc)
        except Exception as exc:  # noqa: BLE001 - import must not sink ingest
            embedded = {}
            log.exception("embedded comment read failed", extra={"dxf_file": path.name})
        imported = 0
        for handle, comments in embedded.items():
            for comment in comments:
                digest = hashlib.sha1(
                    f"{comment.get('author','')}|{comment.get('body','')}".encode()
                ).hexdigest()[:12]
                try:
                    store.add_comment(
                        drawing._id,
                        handle,
                        comment.get("body", ""),
                        author=comment.get("author", "import:xdata"),
                        client_request_id=f"xdata:{drawing._id}:{handle}:{digest}",
                        provenance={"tool": "ingest", "source": "xdata-import"},
                    )
                    imported += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not import embedded comment: %s", exc)
        if imported:
            result.warnings.append(
                f"imported {imported} comment(s) embedded in the file's XDATA"
            )
            log.info(
                "imported embedded comments",
                extra={"dxf_file": path.name, "count": imported},
            )

    run.start("render", layouts_to_render=len(layout_names))
    for layout_name in layout_names if doc is not None else []:
        try:
            render = render_layout_svg(
                doc,
                layout_name=layout_name,
                handle_types=handle_types,
                max_entities=settings.render_max_entities,
            )
            store.save_svg(
                settings.svg_dir,
                drawing._id,
                layout_name,
                render.svg,
                render.summary(),
            )
            result.layouts_rendered.append(render.summary())
        except EmptyRenderError as exc:
            # Expected for paperspace sheets holding only a viewport frame.
            result.warnings.append(str(exc))
            log.info(
                "layout has no drawable geometry",
                extra={"dxf_file": path.name, "layout": layout_name},
            )
        except Exception as exc:  # noqa: BLE001 - one bad layout, not one bad file
            message = f"layout {layout_name!r}: {type(exc).__name__}: {exc}"
            result.warnings.append(message)
            log.exception(
                "layout render failed",
                extra={"dxf_file": path.name, "layout": layout_name},
            )

    result.ok = True
    if doc is None:
        run.skip("render", "the document could not be reopened for rendering")
    else:
        run.finish(
            "render",
            layouts_rendered=len(result.layouts_rendered),
            layouts_offered=len(layout_names),
            warnings=[w for w in result.warnings if w.startswith("layout ")],
        )

    # The arrival path (DOSSIER Phase 7). Placed here — after the drawing and
    # its entities are safely stored, after the render — so that everything
    # expensive and irreplaceable has already happened. `profile_on_arrival`
    # catches its own failures and returns them, so there is no `try` around
    # this call and there must not be one: an exception escaping here would be
    # a bug in that function, and swallowing it twice would hide it.
    if profile:
        arrival = profile_on_arrival(drawing._id, run=run)
        _attach_arrival(result, arrival)
        _record_arrival_stages(run, arrival)
    else:
        for name in ("dossier", "landuse_draft", "h3"):
            run.skip(name, "profiling was switched off for this run")

    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "ingested",
        extra={
            "dxf_file": path.name,
            "drawing_id": drawing._id,
            "entities": drawing.entity_count,
            "layouts_rendered": len(result.layouts_rendered),
            "profiled": bool(result.arrival and result.arrival.get("ok")),
            "elapsed_ms": result.elapsed_ms,
        },
    )
    run.succeeded(
        drawing_id=drawing._id,
        entities=drawing.entity_count,
        layouts_rendered=len(result.layouts_rendered),
        elapsed_ms=result.elapsed_ms,
        warnings=list(result.warnings),
    )
    return result


def ingest_directory(
    directory: Path, *, force: bool = False, profile: bool = True
) -> list[FileResult]:
    """Ingest every `.dxf` in a directory, smallest first.

    Smallest first so that a run that is cut short has still produced the most
    complete set of working drawings, rather than dying inside the 113 MB one.
    """
    # Case-insensitive: the container filesystem is Linux, so `glob("*.dxf")`
    # would silently skip a `.DXF` file without reporting it as anything.
    # `.dwg` as well as `.dxf`: conversion happens in ingest_file(). A `.dxf`
    # sitting beside its own `.dwg` wins, so a hand-converted file can still
    # override the automatic one.
    by_stem: dict[str, Path] = {}
    for candidate in directory.iterdir():
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix not in (".dxf", ".dwg"):
            continue
        existing = by_stem.get(candidate.stem)
        if existing is None or (existing.suffix.lower() == ".dwg" and suffix == ".dxf"):
            by_stem[candidate.stem] = candidate
    files = sorted(by_stem.values(), key=lambda p: p.stat().st_size)
    if not files:
        log.warning("no .dxf files found", extra={"directory": str(directory)})
    results: list[FileResult] = []
    for path in files:
        results.append(
            ingest_file(
                path, force=force, profile=profile, recorder=cli_recorder(path)
            )
        )
    return results


def cli_recorder(path: Path) -> ingest_runs.Recorder:
    """A run recorder for a file ingested from the command line.

    The CLI and the upload route are two doors onto the same pipeline, so they
    write the same history; the only difference is `source`. If the history
    collection cannot be opened the ingest still goes ahead unrecorded, which
    is the same trade the recorder makes for every other write: losing the
    progress line beats losing the ingest.
    """
    try:
        run_id = ingest_runs.create(
            filename=path.name,
            source="cli",
            bytes_received=path.stat().st_size if path.exists() else None,
        )
    except Exception as exc:  # noqa: BLE001 - history must never block ingest
        log.warning("could not open an ingest run record: %s", exc)
        return ingest_runs.NullRecorder()
    recorder = ingest_runs.Recorder(run_id)
    recorder.finish("upload", source="cli", path=str(path))
    return recorder


#: Characters this codebase's prose uses that a Windows console in cp1252
#: cannot encode, and the ASCII each becomes on the way to a terminal. The
#: STORED text keeps its typography — this is a transliteration at the print
#: site, not a rewrite of the data.
_TERMINAL_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("→", "->"),
    ("←", "<-"),
    ("—", "-"),
    ("–", "-"),
    ("…", "..."),
    ("Σ", "sum"),
    ("×", "x"),
    ("≠", "!="),
    ("≈", "~"),
    ("²", "2"),
    ("‘", "'"),
    ("’", "'"),
    ("“", '"'),
    ("”", '"'),
)


def _say(line: str) -> None:
    """`print`, but a report is never lost to the console's encoding.

    This is a lesson already written down in `landuse_draft.report_lines`:
    *"a report that raises UnicodeEncodeError on someone's console is a report
    that did not get read."* It was learned again here — an em dash in a diff
    verdict took down the whole ingest report on a cp1252 terminal AFTER every
    drawing had been stored and profiled correctly, which is the worst shape a
    failure can take: nothing was wrong and everything looked broken.

    The known characters are transliterated so the line stays readable;
    anything else falls back to the encoder's own replacement rather than
    raising.
    """
    text = str(line)
    for source, target in _TERMINAL_SUBSTITUTIONS:
        text = text.replace(source, target)
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = text.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
    print(text)


def _arrival_lines(arrival: dict[str, Any] | None) -> list[str]:
    """The arrival outcome for one file, as lines under its row.

    Printed for every file rather than only on failure. A profile that ran is
    the claim this phase makes; a reader who only ever sees the line when
    something broke has no way to tell "it worked" from "it was never tried".
    """
    if arrival is None:
        return ["     profile : not attempted (--no-profile, or no drawing id)"]
    out: list[str] = []
    dossier = arrival.get("dossier") or {}
    if dossier.get("ok"):
        if dossier.get("built"):
            out.append(
                f"     dossier : built, {dossier.get('buckets')} buckets, "
                f"coverage {dossier.get('accounted')}/{dossier.get('total')}"
                + ("" if dossier.get("coverage_complete") else "  INCOMPLETE")
            )
        else:
            out.append("     dossier : already stored, left alone")
    else:
        out.append(f"  !! dossier : NOT BUILT — {dossier.get('why')}")

    draft = arrival.get("landuse_draft") or {}
    if draft.get("ok"):
        out.append(
            f"     draft   : {draft.get('path')}  "
            f"({draft.get('ready_to_accept')} ready, "
            f"{draft.get('needs_a_decision')} need a decision)"
        )
    else:
        out.append(f"  !! draft   : NOT WRITTEN — {draft.get('why')}")

    revision = arrival.get("revision") or {}
    if not revision.get("ok"):
        out.append(f"  !! revision: NOT CHECKED — {revision.get('why')}")
    elif revision.get("predecessor"):
        summary = revision.get("diff_summary") or {}
        out.append(
            f"     revision: predecessor {revision['predecessor']} — "
            + str(summary.get("verdict") or "diff not run")
        )
    elif revision.get("ambiguous"):
        out.append(
            f"  !! revision: {revision.get('candidate_count')} drawings share "
            f"this name — {', '.join(revision.get('candidates') or [])}. "
            "Ambiguous, so nothing was compared and nothing was chosen."
        )
    else:
        out.append("     revision: no earlier version of this name in the store")
    return out


def _print_report(results: list[FileResult]) -> None:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    _say(f"\n{'file':<52}{'entities':>9}{'layers':>7}{'blocks':>7}{'svg':>5}{'sec':>7}  status")
    _say("-" * 104)
    for r in sorted(results, key=lambda r: r.filename):
        status = "SKIP" if r.skipped else ("OK" if r.ok else "FAIL")
        _say(
            f"{r.filename[:51]:<52}{r.entity_count:>9}{r.layer_count:>7}"
            f"{r.block_count:>7}{len(r.layouts_rendered):>5}{r.elapsed_ms/1000:>7.1f}  {status}"
        )
        if r.error:
            _say(f"{'':<52}  -> {r.error}")
        if r.ok:
            for line in _arrival_lines(r.arrival):
                _say(line)
    _say("-" * 104)
    _say(f"{len(ok)} ok ({sum(1 for r in ok if r.skipped)} skipped), {len(failed)} failed")
    unprofiled = [
        r for r in ok if r.arrival is not None and not (r.arrival.get("ok"))
    ]
    if unprofiled:
        _say(
            f"{len(unprofiled)} file(s) ingested but NOT fully profiled — the "
            "drawings are stored and queryable; what is missing is listed "
            "above and is recoverable by re-running this or "
            "scripts/dossier_backfill.py."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest DXF files into MongoDB.")
    parser.add_argument("--all", action="store_true", help="ingest every DXF in CAD_DXF_DIR")
    parser.add_argument("--file", help="ingest a single file (name or path)")
    parser.add_argument("--force", action="store_true", help="re-ingest even if current")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help=(
            "skip the arrival path: do not build the Dossier, do not draft the "
            "land-use config, do not look for a predecessor. For a bulk "
            "re-ingest that will be followed by one dossier_backfill run over "
            "everything. A drawing ingested this way answers 'not computed' "
            "to every question about what it contains until something profiles "
            "it."
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    mongo.ensure_indexes()
    profile = not args.no_profile

    if args.file:
        # Both directories, the same two `--all` sweeps.
        #
        # A bare name used to be resolved against `dxf_dir` alone, so
        # `--file "SOMETHING.dwg"` on a drawing that exists only as a .dwg
        # reported `no such file: /data/dxf/SOMETHING.dwg` -- a true sentence
        # about the wrong directory, and the reason a re-ingest of the largest
        # drawing on this stack was attempted three times before anyone looked
        # at which path it had been given.
        path = Path(args.file)
        searched: list[Path] = []
        if not path.is_absolute():
            for base in (settings.dxf_dir, settings.dwg_dir):
                candidate = base / args.file
                searched.append(candidate)
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            print(f"no such file: {args.file}", file=sys.stderr)
            for candidate in searched or [path]:
                print(f"  looked in: {candidate}", file=sys.stderr)
            print(
                "  a name is resolved against the DXF directory and then the "
                "DWG directory; pass an absolute path to skip both.",
                file=sys.stderr,
            )
            return 2
        results = [
            ingest_file(
                path,
                force=args.force,
                profile=profile,
                recorder=cli_recorder(path),
            )
        ]
    elif args.all:
        # Both directories. A drawing that exists only as a .dwg is never seen
        # by a sweep that reads dxf_dir alone, and the report still reads
        # "n ok, 0 failed" -- right about the files it saw, silent about the
        # ones it did not.
        results = ingest_directory(
            settings.dxf_dir, force=args.force, profile=profile
        )
        if settings.dwg_dir.exists() and settings.dwg_dir != settings.dxf_dir:
            results += ingest_directory(
                settings.dwg_dir, force=args.force, profile=profile
            )
    else:
        parser.print_help()
        return 2

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    else:
        _print_report(results)

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
