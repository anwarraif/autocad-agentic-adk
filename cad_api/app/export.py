"""Export a derived DXF that carries its comments inside the file.

The requirement, in the owner's words: *"even when we export or give it to
some other person those comments will be there with that structure."*

AutoCAD has no built-in per-object comment popup, so one comment is written
into the file in **three complementary forms**, each covering a different way
the recipient will meet it:

1. **XDATA** on the entity (appid ``ROSHN_COMMENTS``) — the machine-readable
   record. It travels *with the entity*: copy the object, and the comment
   copies too. This is what lets our own ingest read comments back out of a
   file someone returns to us (round-trip, see `read_embedded_comments`).
2. **A hyperlink with a description** on the entity — the closest thing DWG
   has to "hover and a popup appears": AutoCAD natively shows the hyperlink
   description as a tooltip when the cursor rests on the object. No plugin.
3. **Visible MTEXT on a dedicated ``AI_MARKUP`` layer** — readable in any
   viewer at all, even ones that show neither XDATA nor hyperlinks. On its own
   layer so the recipient can switch the markup off in one click.

The original file is **never modified**. Round-tripping DWG→DXF is not
lossless (this very corpus lost its PDF-underlay definitions on the way in),
so the export is always a *derived* document with a name that says so.

Sizes for planning: the Janadriyah DXF is 113 MB and takes ~60 s to load and
~40 s to save, so exports are cached and regenerated only when the comment set
changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field as dataclasses_field
from pathlib import Path
from typing import Any, Iterator

from ezdxf.document import Drawing
from ezdxf.entities import DXFGraphic

from . import store
from .extract import is_block_layout

log = logging.getLogger(__name__)

#: Registered application id for our XDATA. One fixed name, never configurable:
#: a recipient's tooling must be able to rely on it.
APPID = "ROSHN_COMMENTS"

#: First XDATA string record — lets a reader reject foreign/newer payloads.
XDATA_VERSION = "ROSHN_COMMENTS_V1"

#: XDATA group-1000 strings are capped at 255 bytes; JSON is chunked to stay
#: comfortably under it.
_CHUNK = 240

#: Layer that carries the visible markup text. The name is part of the
#: contract with reviewers ("switch off AI_MARKUP to see the bare drawing").
MARKUP_LAYER = "AI_MARKUP"

#: ACI colour for the markup layer. 1 = red, the conventional redline colour.
_MARKUP_COLOR = 1

#: Dedicated text style for the markup notes, and the font it points at.
#:
#: The notes must never inherit the drawing's own `Standard` style. Janadriyah
#: opens in AutoCAD with a "Missing SHX Files" prompt because it references
#: Arabic shape fonts (`xarb.shx`, `C:/ACAD/ESA/X-ARAB1b.SHX`) that are not
#: distributed with it. There `Standard` happens to point at `simplex.shx`, a
#: stock font, so the note survives -- but that is luck, not design: in a file
#: whose `Standard` points at a missing SHX, the comment itself would be the
#: thing that disappears.
#:
#: A TrueType font is chosen over an SHX one deliberately: TrueType is
#: resolved by the operating system rather than by AutoCAD's SHX search path,
#: so it cannot go missing the way a shape font can.
MARKUP_STYLE = "AI_MARKUP_TEXT"
MARKUP_FONT = "arial.ttf"


@dataclass
class EmbedResult:
    """What was written into the derived document — and what did not survive.

    `entities_lost_on_save` exists because of a measured failure, not
    paranoia: the five 3DSOLIDs in `visualization_-_aerial` vanish when ezdxf
    re-saves the LibreDWG-converted file (their ACIS payload does not survive
    the round trip), while the thirty in `visualization_-_conference_room` are
    fine. Per-file, unpredictable — so every export build reloads its own
    output and counts. A derived file that silently lost content would poison
    the whole "comments travel with the file" story.
    """

    comments_embedded: int
    entities_annotated: int
    markup_notes: int
    skipped_missing_handles: list[str]
    entities_lost_on_save: int = 0
    lost_sample: list[str] = dataclasses_field(default_factory=list)
    corrupt_objects_removed: int = 0
    corrupt_sample: list[str] = dataclasses_field(default_factory=list)


def _chunks(text: str) -> Iterator[str]:
    for start in range(0, len(text), _CHUNK):
        yield text[start : start + _CHUNK]


def _comment_payload(comments: list[dict[str, Any]]) -> str:
    """The JSON stored in XDATA: only the fields a recipient can use."""
    return json.dumps(
        [
            {
                "author": c.get("author", ""),
                "body": c.get("body", ""),
                "created_at": str(c.get("created_at", "")),
            }
            for c in comments
        ],
        ensure_ascii=False,
    )


def _tooltip(comments: list[dict[str, Any]]) -> str:
    """One-line hover text. AutoCAD shows this next to the cursor."""
    first = comments[0]
    head = f"{first.get('author', '?')}: {first.get('body', '')}"
    if len(comments) > 1:
        head += f"  (+{len(comments) - 1} more)"
    # Tooltips are single-line; keep them readable rather than complete.
    return head[:220]


def _owner_metrics(
    owner: Any, cache: dict[int, tuple[float, tuple[float, float]] | None]
) -> tuple[float, tuple[float, float]] | None:
    """(text height, centre point) for the layout a note is drawn into.

    Height is ~1/200 of the layout's larger extent — a fixed height would be
    invisible on Janadriyah (11.7 km wide, metres) and monstrous on a title
    block (millimetres). The centre decides which way the leader text points:
    **inward**, so notes never poke past the sheet border and stretch the
    recipient's zoom-to-extents view. Cached per layout: computing extents is
    the expensive part, and it must be read *before* the first note lands.
    """
    key = id(owner)
    if key in cache:
        return cache[key]
    result: tuple[float, tuple[float, float]] | None = None
    try:
        from ezdxf import bbox as _bbox

        box = _bbox.extents(owner, fast=True)
        if box.has_data:
            span = max(box.extmax.x - box.extmin.x, box.extmax.y - box.extmin.y)
            if span > 0:
                result = (
                    span / 200.0,
                    (
                        (box.extmin.x + box.extmax.x) / 2.0,
                        (box.extmin.y + box.extmax.y) / 2.0,
                    ),
                )
    except Exception as exc:  # noqa: BLE001 - a note is best-effort
        log.debug("owner metrics failed: %s", exc)
    cache[key] = result
    return result


def _entity_center(entity: DXFGraphic) -> tuple[float, float] | None:
    """The middle of the entity — where a reader's eye says the comment *is*.

    The first version anchored notes at the bbox *corner*, which for a long
    line is its far end: the text appeared to float at the edge of the sheet,
    visually disconnected from the object it annotates, and the user read it
    as the drawing having shifted. The centre of the bbox is the line's
    midpoint, and the leader arrow drawn from it touches the object itself.
    """
    try:
        from ezdxf import bbox as _bbox

        box = _bbox.extents([entity], fast=True)
        if box.has_data:
            return (
                (box.extmin.x + box.extmax.x) / 2.0,
                (box.extmin.y + box.extmax.y) / 2.0,
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("entity center failed for %s: %s", entity.dxf.get("handle"), exc)
    return None


def _draw_leader_note(
    owner: Any,
    anchor: tuple[float, float],
    center: tuple[float, float],
    height: float,
    comments: list[dict[str, Any]],
) -> None:
    """One redline: arrowhead on the entity, leader line, text pulled inward.

    Built from primitives (LINE + SOLID + MTEXT) rather than a MULTILEADER on
    purpose: a MULTILEADER needs an MLEADERSTYLE that a foreign drawing may
    not carry, and a missing style is an exception at export time. Primitives
    render identically in AutoCAD, TrueView and every web viewer.
    """
    import math

    ax, ay = anchor
    direction_x = 1.0 if center[0] >= ax else -1.0
    direction_y = 1.0 if center[1] >= ay else -1.0

    length = 3.0 * height
    tip_x, tip_y = ax + direction_x * length, ay + direction_y * length

    attribs = {"layer": MARKUP_LAYER}
    owner.add_line((ax, ay), (tip_x, tip_y), dxfattribs=attribs)

    # Arrowhead: a small solid triangle whose tip sits ON the entity.
    unit_x, unit_y = direction_x / math.sqrt(2), direction_y / math.sqrt(2)
    base_x, base_y = ax + unit_x * height * 0.9, ay + unit_y * height * 0.9
    perp_x, perp_y = -unit_y * height * 0.3, unit_x * height * 0.3
    owner.add_solid(
        [
            (ax, ay),
            (base_x + perp_x, base_y + perp_y),
            (base_x - perp_x, base_y - perp_y),
        ],
        dxfattribs=attribs,
    )

    text = "\\P".join(
        f"[{c.get('author', '?')}] {c.get('body', '')}" for c in comments
    )
    # MIDDLE_LEFT when the text flows right of the leader tip, MIDDLE_RIGHT
    # when it flows left — so the words always run further *into* the sheet.
    mtext = owner.add_mtext(
        text,
        dxfattribs={
            "layer": MARKUP_LAYER,
            "char_height": height,
            "style": MARKUP_STYLE,
            "attachment_point": 4 if direction_x > 0 else 6,
        },
    )
    mtext.set_location((tip_x + direction_x * 0.5 * height, tip_y))


def embed_comments(
    doc: Drawing,
    drawing_id: str,
    comments_by_handle: dict[str, list[dict[str, Any]]],
    layout_hints: dict[str, tuple[float, tuple[float, float]]] | None = None,
) -> EmbedResult:
    """Write every comment into the open document, three ways per comment.

    Args:
        doc: the document to annotate (a copy of the original — never the
            original itself; the caller owns that guarantee).
        drawing_id: our content-hash id, recorded in each XDATA payload so a
            returned file can be matched to its source drawing.
        comments_by_handle: handle -> list of comment documents.
        layout_hints: optional per-layout (dense span, dense centre), measured
            from stored entity bbox centres. See `_layout_hints` for why raw
            extents cannot be trusted for sizing.
    """
    if APPID not in doc.appids:
        doc.appids.add(APPID)
    if MARKUP_LAYER not in doc.layers:
        doc.layers.add(MARKUP_LAYER, color=_MARKUP_COLOR)
    if MARKUP_STYLE not in doc.styles:
        doc.styles.add(MARKUP_STYLE, font=MARKUP_FONT)

    # Handle -> entity over every layout and block, one pass.
    wanted = set(comments_by_handle)
    found: dict[str, DXFGraphic] = {}
    containers = list(doc.layouts) + [
        b for b in doc.blocks if not b.name.startswith("*")
    ]
    for container in containers:
        if not wanted:
            break
        for entity in container:
            handle = entity.dxf.get("handle")
            if handle in wanted:
                found[handle] = entity
                wanted.discard(handle)

    metrics_cache: dict[int, tuple[float, tuple[float, float]] | None] = {}
    layout_hints = layout_hints or {}
    embedded = 0
    notes = 0

    for handle, entity in found.items():
        comments = comments_by_handle[handle]

        # 1. XDATA — the durable, machine-readable anchor.
        payload = json.dumps(
            {"drawing_id": drawing_id, "handle": handle}, ensure_ascii=False
        )
        records: list[tuple[int, str]] = [(1000, XDATA_VERSION), (1000, payload)]
        records += [(1000, chunk) for chunk in _chunks(_comment_payload(comments))]
        entity.set_xdata(APPID, records)

        # 2. Hyperlink — AutoCAD's native hover tooltip.
        entity.set_hyperlink(
            f"comments://{drawing_id}/{handle}", description=_tooltip(comments)
        )

        # 3. Visible note. Placed in the entity's own space; entities that live
        #    inside a block definition get XDATA and the tooltip but no note —
        #    a note in modelspace coordinates would land nowhere near them.
        owner = entity.get_layout()
        if owner is not None and not is_block_layout(getattr(owner, "name", "")):
            metrics = layout_hints.get(
                getattr(owner, "name", "")
            ) or _owner_metrics(owner, metrics_cache)
            anchor = _entity_center(entity)
            if metrics is not None and anchor is not None:
                height, center = metrics
                _draw_leader_note(owner, anchor, center, height, comments)
                notes += 1

        embedded += len(comments)

    missing = sorted(wanted)
    if missing:
        log.warning(
            "comments reference handles absent from the document",
            extra={"drawing_id": drawing_id, "missing": missing[:10]},
        )
    return EmbedResult(
        comments_embedded=embedded,
        entities_annotated=len(found),
        markup_notes=notes,
        skipped_missing_handles=missing,
    )


def read_embedded_comments(doc: Drawing) -> dict[str, list[dict[str, Any]]]:
    """Read `ROSHN_COMMENTS` XDATA back out of a document.

    This is the other half of the round trip: someone returns a file we (or a
    colleague's copy of this tool) exported, we ingest it, and the comments
    reappear in the database attached to the same handles — because handles
    survive the export (verified by test).
    """
    out: dict[str, list[dict[str, Any]]] = {}
    containers = list(doc.layouts) + [
        b for b in doc.blocks if not b.name.startswith("*")
    ]
    seen: set[str] = set()
    for container in containers:
        for entity in container:
            handle = entity.dxf.get("handle")
            if not handle or handle in seen:
                continue
            seen.add(handle)
            try:
                records = entity.get_xdata(APPID)
            except Exception:
                continue  # the common case: no XDATA for our appid
            strings = [value for code, value in records if code == 1000]
            if not strings or strings[0] != XDATA_VERSION:
                continue
            # strings[1] is the {drawing_id, handle} header; the rest is the
            # chunked comment JSON.
            try:
                comments = json.loads("".join(strings[2:]))
            except (ValueError, IndexError) as exc:
                log.warning(
                    "unreadable comment payload on handle %s: %s", handle, exc
                )
                continue
            if isinstance(comments, list) and comments:
                out[handle] = comments
    return out


def _layout_hints(
    drawing_id: str,
) -> dict[str, tuple[float, tuple[float, float]]]:
    """Per-layout (dense span, dense centre) from the stored entity bbox centres.

    Raw layout extents lie about scale whenever a drawing carries outlier
    geometry. Janadriyah's modelspace measures 15,129 m across, but 90% of
    its 20,328 entities sit inside a 1,486 m region — the actual site; the
    rest is a surroundings block. Sizing note text from the raw span produced
    76 m tall letters over 20 m buildings, which is what the reviewer saw in
    TrueView. The 5th–95th percentile of bbox centres is outlier-resistant and
    costs one indexed Mongo query against data ingest already computed.
    """
    points: dict[str, list[list[float]]] = {}
    try:
        from .mongo import COLL_ENTITIES, coll

        cursor = coll(COLL_ENTITIES).find(
            {"drawing_id": drawing_id, "bbox_centre": {"$ne": None}},
            {"bbox_centre": 1, "layout": 1},
        )
        for row in cursor:
            points.setdefault(row.get("layout", "Model"), []).append(row["bbox_centre"])
    except Exception as exc:  # noqa: BLE001 - hints are an optimisation
        log.warning("layout hints unavailable: %s", exc)
        return {}

    hints: dict[str, tuple[float, tuple[float, float]]] = {}
    for layout_name, pts in points.items():
        if len(pts) < 5:
            continue
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        lo = int(0.05 * (len(pts) - 1))
        hi = int(0.95 * (len(pts) - 1))
        span = max(xs[hi] - xs[lo], ys[hi] - ys[lo])
        if span <= 0:
            continue
        hints[layout_name] = (
            span / 200.0,
            ((xs[lo] + xs[hi]) / 2.0, (ys[lo] + ys[hi]) / 2.0),
        )
    return hints


#: Native object classes ODA cannot restore from DXF and then writes broken
#: into a DWG. Removed together with every ACAD_PROXY_OBJECT — see below.
UNCONVERTIBLE_CLASSES = frozenset({"RAPIDRTRENDERSETTINGS", "ACDBSECTIONVIEWSTYLE"})


def strip_unconvertible_objects(doc: Drawing) -> list[str]:
    """Remove objects that poison a DXF -> DWG conversion. DWG path only.

    Two groups go, both measured on Janadriyah with TrueView 2027 as the
    judge:

    - The RapidRT/SectionViewStyle render presets (typed or proxy-wrapped):
      ODA writes them "improperly" and its own reader rejects the DWG.
    - EVERY ACAD_PROXY_OBJECT (4,948 in Janadriyah — Civil 3D application
      data): with them present the DWG demands AutoCAD *recovery* on open;
      with them removed it opens clean, drawing and comments intact. Proxy
      objects are app-private blobs that already lost their function the
      moment the file left the original DWG for DXF — no converter can
      rebuild them, and AutoCAD's auditor flags their re-encoded husks as
      damage.

    The DXF export keeps everything; only the derived DWG is sanitized, the
    count is surfaced on the response, and the original file is never
    touched. Deletion is cascade-safe (an object's deletion can destroy
    others in the list) and dictionary entries pointing at removed handles
    are purged so nothing dangles.
    """
    class_names = {
        500 + i: c.dxf.name for i, c in enumerate(doc.classes.classes.values())
    }

    def proxy_class(obj: Any) -> str:
        try:
            for sc in obj.xtags.subclasses:
                if sc and sc[0].value == "AcDbProxyObject":
                    for t in sc:
                        if t.code == 91:
                            return class_names.get(t.value, "")
        except Exception:  # noqa: BLE001 - unresolvable proxy = leave it be
            pass
        return ""

    db = doc.entitydb
    handles: list[str] = []
    for o in list(doc.objects):
        if not o.is_alive:
            continue
        t = o.dxftype()
        if t in UNCONVERTIBLE_CLASSES or t == "ACAD_PROXY_OBJECT":
            try:
                handles.append(o.dxf.handle)
            except Exception:  # noqa: BLE001 - no handle = nothing to remove
                continue

    removed: set[str] = set()
    for h in handles:
        obj = db.get(h)
        if obj is None or not obj.is_alive:
            removed.add(h)  # died in a cascade; still gone from the file
            continue
        try:
            doc.objects.delete_entity(obj)
            removed.add(h)
        except Exception:  # noqa: BLE001 - leave what cannot be removed
            pass

    for obj in doc.objects:
        if obj.is_alive and obj.dxftype() in (
            "DICTIONARY", "ACDBDICTIONARYWDFLT"
        ):
            for key, val in list(obj.items()):
                try:
                    h = val.dxf.handle if hasattr(val, "dxf") else val
                except Exception:  # noqa: BLE001
                    continue
                if h in removed:
                    obj.discard(key)

    if removed:
        log.warning(
            "stripped unconvertible objects for dwg",
            extra={"count": len(removed), "sample": sorted(removed)[:12]},
        )
    return sorted(removed)


def build_dwg_export(
    dxf_export: Path,
    export_dir: Path,
) -> tuple[Path, int]:
    """Derive a DWG from an already-built comment-carrying DXF export.

    Returns (dwg_path, unconvertible_objects_removed). Cached beside the DXF
    under the same cache key, so it regenerates exactly when the DXF does.
    """
    from .convert import dxf_to_dwg
    from .extract import load_document

    target = dxf_export.with_suffix(".dwg")
    if target.exists():
        return target, 0

    doc, _errors, _fixes = load_document(dxf_export)
    removed = strip_unconvertible_objects(doc)

    with tempfile.TemporaryDirectory(prefix="dwg_stage_") as tmp:
        staged = Path(tmp) / dxf_export.name
        doc.saveas(staged)
        produced = dxf_to_dwg(staged, Path(tmp) / "out")
        shutil.move(str(produced), target)
    return target, len(removed)


def collect_handles(doc: Drawing) -> set[str]:
    """Every entity handle in the document: layouts and named blocks."""
    handles: set[str] = set()
    for container in list(doc.layouts) + [
        b for b in doc.blocks if not b.name.startswith("*")
    ]:
        for entity in container:
            handle = entity.dxf.get("handle")
            if handle:
                handles.add(handle)
    return handles


def export_path(export_dir: Path, drawing: dict[str, Any], cache_key: str) -> Path:
    stem = Path(drawing.get("original_filename", "drawing")).stem
    return export_dir / f"{stem}__comments-{cache_key}.dxf"


# Bump when the embedding logic changes, so cached exports built by the old
# code are regenerated on next download (e.g. v2 = dense-extents text sizing).
EXPORT_VERSION = 4


def _is_corrupt_text(s: str) -> bool:
    """True when a string carries bytes no DXF parser can have written on
    purpose: U+FFFD replacement characters or lone surrogates."""
    return any(c == "�" or "\ud800" <= c <= "\udfff" for c in s)


def strip_corrupt_xrecords(doc: Drawing) -> list[str]:
    """Drop the unreadable-garbage string tags from corrupt XRECORDs.

    Janadriyah carries seven 3DSVIZ material XRECORDs (handles 22C1..22D3)
    whose XML chunks come out as interleaved mojibake from BOTH converters —
    LibreDWG and ODA produce the same garbage — so the corruption is in how
    those records exist in DXF form, and nothing downstream can read them:
    Autodesk's web-viewer extractor dies with 0xC0000409 and ODA itself
    refuses to read the file back ("Bad Dxf"). The records are
    render-material metadata, not drawing geometry, and they are ALREADY
    unreadable in every DXF representation; the original DWG in documents/
    keeps them untouched.

    Only the poisoned string tags are removed — the XRECORD objects stay, so
    every dictionary reference to them remains valid (destroying the objects
    left dangling entries that crashed ezdxf's own audit). Counted, logged,
    and surfaced on the export response — never silent.
    """
    from ezdxf.lldxf.tags import Tags

    handles = []
    for obj in doc.objects:
        if obj.dxftype() != "XRECORD":
            continue
        try:
            corrupt = any(
                isinstance(t.value, str) and _is_corrupt_text(t.value)
                for t in obj.tags
            )
        except Exception:  # noqa: BLE001 - unreadable tags = corrupt too
            corrupt = True
        if not corrupt:
            continue
        obj.tags = Tags(
            t for t in obj.tags
            if not (isinstance(t.value, str) and _is_corrupt_text(t.value))
        )
        handles.append(obj.dxf.handle)
    if handles:
        log.warning(
            "sanitized corrupt xrecords",
            extra={"count": len(handles), "handles": handles[:10]},
        )
    return handles


def cache_key_for(comments: list[dict[str, Any]], drawing_id: str = "") -> str:
    """Changes iff the comment set, the drawing, or the code version changes.

    `drawing_id` is the source file's content hash, so a re-ingest (new
    conversion, new source) invalidates the cache even when the comments are
    identical. Measured failure without it: after the ODA re-ingest, an
    export request served the pre-ODA artifact because the comment set alone
    hashed the same — the reviewer re-uploaded a stale file and the fix
    looked broken. Regenerating unconditionally is not the answer either:
    Janadriyah costs ~2 minutes per download.
    """
    digest = hashlib.sha256()
    digest.update(str(EXPORT_VERSION).encode())
    digest.update(drawing_id.encode())
    for comment in sorted(comments, key=lambda c: str(c.get("_id", ""))):
        digest.update(str(comment.get("_id", "")).encode())
        digest.update(str(comment.get("body", "")).encode())
    return digest.hexdigest()[:12]


def build_export(
    source_dxf: Path,
    export_dir: Path,
    drawing: dict[str, Any],
) -> tuple[Path, EmbedResult]:
    """Produce (or reuse) the derived, comment-carrying DXF for one drawing.

    Args:
        source_dxf: the DXF the drawing was ingested from. Opened read-only.
        export_dir: where derived files live; created if needed.
        drawing: the stored drawing document (for id and filename).

    Returns the path to the derived file and what was embedded in it.
    """
    from .extract import load_document  # local import to avoid a cycle

    drawing_id = drawing["_id"]
    comments = store.list_comments(drawing_id)
    key = cache_key_for(comments, drawing_id)
    target = export_path(export_dir, drawing, key)

    by_handle: dict[str, list[dict[str, Any]]] = {}
    for comment in comments:
        by_handle.setdefault(comment["entity_handle"], []).append(comment)

    if target.exists():
        log.info(
            "export cache hit",
            extra={"drawing_id": drawing_id, "path": str(target)},
        )
        return target, EmbedResult(
            comments_embedded=len(comments),
            entities_annotated=len(by_handle),
            markup_notes=-1,  # unknown without regenerating; -1 = "cached"
            skipped_missing_handles=[],
        )

    started = time.perf_counter()
    doc, _errors, _fixes = load_document(source_dxf)
    corrupt_handles = strip_corrupt_xrecords(doc)
    result = embed_comments(
        doc, drawing_id, by_handle, layout_hints=_layout_hints(drawing_id)
    )
    result.corrupt_objects_removed = len(corrupt_handles)
    result.corrupt_sample = corrupt_handles[:10]
    handles_before = collect_handles(doc)

    export_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    doc.saveas(tmp)
    tmp.replace(target)

    # Trust nothing about the save: reload the file we just wrote and compare.
    # Costs one extra load (~60 s on Janadriyah, cached thereafter); the
    # alternative is shipping a file that silently lost entities.
    import ezdxf as _ezdxf

    handles_after = collect_handles(_ezdxf.readfile(str(target)))
    lost = sorted(handles_before - handles_after)
    if lost:
        result.entities_lost_on_save = len(lost)
        result.lost_sample = lost[:10]
        log.warning(
            "entities did not survive the export save",
            extra={"drawing_id": drawing_id, "lost": len(lost), "sample": lost[:5]},
        )

    log.info(
        "built export",
        extra={
            "drawing_id": drawing_id,
            "path": str(target),
            "comments": result.comments_embedded,
            "entities": result.entities_annotated,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        },
    )
    return target, result
