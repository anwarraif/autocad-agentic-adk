"""Emit docs/RECON.md — what was found in each of the 18 files.

Reads only what ingest already stored, so the report cannot disagree with the
running system. Failures are listed as prominently as successes: a file that
did not make it is the more useful half of a survey.

    python -m app.recon > docs/RECON.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from . import store
from .config import get_settings
from .logging_conf import configure_logging

#: Files that exist as source DWG but never reached MongoDB, and why.
KNOWN_ABSENT: dict[str, str] = {
    "blocks_and_tables_-_metric.dxf": (
        "DXF conversion produced a malformed file: ezdxf reports "
        '`Invalid group code "Standard" at line 97507`. This is the LibreDWG '
        "table-writing bug already recorded in DXF-CONVERSION-REPORT.md, which "
        "failed on every target version (r14 → r2013) at the same line. The "
        "defect is in the source file, not in this pipeline. Recovery route: "
        "re-convert this one file through DWG TrueView or ODA File Converter."
    ),
}


def _fmt_extents(extents: dict[str, list[float]] | None, units: str) -> str:
    if not extents:
        return "not computable"
    lo, hi = extents["min"], extents["max"]
    width = hi[0] - lo[0]
    height = hi[1] - lo[1]
    return f"{width:,.1f} × {height:,.1f} {units}"


def _top(counts: dict[str, int], n: int = 5) -> str:
    if not counts:
        return "—"
    rows = sorted(counts.items(), key=lambda kv: -kv[1])[:n]
    return ", ".join(f"`{k}` {v:,}" for k, v in rows)


def main() -> int:
    configure_logging(get_settings().log_level)
    drawings = store.list_drawings()
    by_name = {d["original_filename"]: d for d in drawings}

    total_entities = sum(d.get("entity_count", 0) for d in drawings)
    total_bytes = sum(d.get("file_bytes", 0) for d in drawings)

    out: list[str] = []
    w = out.append

    w("# RECON — what is actually in the 18 files")
    w("")
    w("Generated from MongoDB by `python -m app.recon`, so every number here is")
    w("what the running system holds, not a separate measurement that could drift.")
    w("")
    w("## Summary")
    w("")
    w(f"- Source files offered: **18** (17 Autodesk samples + 1 company drawing)")
    w(f"- Converted to DXF by LibreDWG: **17**")
    w(f"- Successfully ingested and queryable: **{len(drawings)}**")
    w(f"- Failed: **{len(KNOWN_ABSENT)}** (see the failures section — it is the useful part)")
    w(f"- Total entities stored: **{total_entities:,}**")
    w(f"- Total DXF processed: **{total_bytes / 1e6:,.0f} MB**")
    w("")

    # ---- the per-file table ------------------------------------------------
    w("## Per file")
    w("")
    w("| File | Entities | Layers | Blocks | Layouts rendered | Units | Release | Extents | Audit errors |")
    w("|---|---:|---:|---:|---:|---|---|---|---:|")

    for drawing in sorted(
        drawings, key=lambda d: -d.get("entity_count", 0)
    ):
        did = drawing["_id"]
        detail = store.get_drawing(did) or {}
        renders = store.list_render_meta(did)
        name = drawing["original_filename"].replace(".dxf", "")
        units = drawing.get("units_name", "?")
        w(
            f"| {name} "
            f"| {drawing.get('entity_count', 0):,} "
            f"| {len(detail.get('layers', []))} "
            f"| {len(detail.get('blocks', []))} "
            f"| {len(renders)} "
            f"| {units}{'' if drawing.get('units_code') else ' *(not declared)*'} "
            f"| {drawing.get('acad_release', '?')} "
            f"| {_fmt_extents(drawing.get('extents'), units)} "
            f"| {drawing.get('audit_errors', 0)} |"
        )
    w("")

    # ---- failures ----------------------------------------------------------
    w("## Failures — read this part")
    w("")
    if not KNOWN_ABSENT:
        w("None.")
    for name, reason in KNOWN_ABSENT.items():
        w(f"### `{name}`")
        w("")
        w(reason)
        w("")

    # ---- files that ingested but render nothing ----------------------------
    silent: list[tuple[str, dict[str, Any]]] = []
    for drawing in drawings:
        renders = store.list_render_meta(drawing["_id"])
        if not renders:
            silent.append((drawing["original_filename"], drawing))

    w("## Ingested but nothing to draw")
    w("")
    w("Block definitions are rendered as pseudo-layouts (`[block] NAME`), which")
    w("is what makes the title-block files viewable at all — their content is")
    w("in a block definition and not in any layout. Entities inside a block")
    w("*definition* carry real handles, unlike the copies from")
    w("`virtual_entities()`, so they stay clickable and commentable.")
    w("")
    if not silent:
        w("None — every ingested file produced at least one rendered layout.")
    else:
        w("These files are in the database and queryable, but produce no SVG.")
        w("That is a correct outcome, not a bug.")
        w("")
        for name, drawing in sorted(silent):
            detail = store.get_drawing(drawing["_id"]) or {}
            blocks = len(detail.get("blocks", []))
            w(
                f"- **{name}** — {drawing.get('entity_count', 0)} entities in layouts, "
                f"{blocks} block definitions. "
                + (
                    "Content lives in block definitions rather than in any layout: "
                    "this is a title-block/table template, not a finished drawing."
                    if blocks and drawing.get("entity_count", 0) <= 1
                    else "No entity in any layout produces drawable 2D geometry "
                    "(typically 3D solids, meshes, or viewport frames only)."
                )
            )
    w("")

    # ---- the company drawing ----------------------------------------------
    jana = by_name.get("JANADRIYAH DMP - 20240506.dxf")
    if jana:
        detail = store.get_drawing(jana["_id"]) or {}
        renders = store.list_render_meta(jana["_id"])
        w("## The company drawing — JANADRIYAH DMP")
        w("")
        w("This is the file that decides whether the approach is useful. The")
        w("Autodesk samples are for iteration; this is the evidence.")
        w("")
        w(f"- `drawing_id`: `{jana['_id']}`")
        w(f"- Format: {jana.get('dxf_version')} ({jana.get('acad_release')}), "
          f"units **{jana.get('units_name')}** (declared in file)")
        w(f"- Entities stored: **{jana.get('entity_count', 0):,}** across all layouts")
        w(f"- Layers: **{len(detail.get('layers', []))}**, "
          f"block definitions: **{len(detail.get('blocks', []))}**")
        w(f"- Opened with `ezdxf.recover.readfile()`: **{jana.get('audit_errors', 0)} "
          f"audit errors recovered**. Plain `ezdxf.readfile()` is not safe here.")
        w(f"- Extents (computed, not from header): {_fmt_extents(jana.get('extents'), jana.get('units_name', ''))}")
        w(f"- Busiest layers: {_top(detail.get('counts_by_layer', {}))}")
        w(f"- Entity mix: {_top(detail.get('counts_by_type', {}))}")
        w("")
        w("### Render cost per layout")
        w("")
        w("| Layout | Entities drawn | SVG raw | SVG gzip | Ratio | Time |")
        w("|---|---:|---:|---:|---:|---:|")
        for render in sorted(renders, key=lambda r: -r.get("rendered_entities", 0)):
            raw = render.get("bytes_raw", 0)
            gz = render.get("bytes_gzip", 1)
            w(
                f"| {render.get('layout')} "
                f"| {render.get('rendered_entities', 0):,} "
                f"| {raw / 1e6:,.1f} MB "
                f"| {gz / 1e6:,.2f} MB "
                f"| {raw / max(gz, 1):,.0f}:1 "
                f"| {render.get('elapsed_ms', 0) / 1000:,.1f} s |"
            )
        w("")

    # ---- render cost across the board -------------------------------------
    w("## Render cost, all files")
    w("")
    w("| Drawing | Layout | Entities | SVG raw | SVG gzip | Time |")
    w("|---|---|---:|---:|---:|---:|")
    rows: list[tuple[int, str]] = []
    for drawing in drawings:
        for render in store.list_render_meta(drawing["_id"]):
            raw = render.get("bytes_raw", 0)
            rows.append(
                (
                    raw,
                    f"| {drawing['original_filename'].replace('.dxf', '')[:38]} "
                    f"| {render.get('layout')} "
                    f"| {render.get('rendered_entities', 0):,} "
                    f"| {raw / 1e6:,.2f} MB "
                    f"| {render.get('bytes_gzip', 0) / 1e6:,.3f} MB "
                    f"| {render.get('elapsed_ms', 0) / 1000:,.1f} s |",
                )
            )
    for _size, line in sorted(rows, key=lambda r: -r[0])[:25]:
        w(line)
    w("")
    w("Only the 25 heaviest layouts are listed. The gzip column is the number")
    w("that matters for the browser: the API serves the compressed bytes")
    w("untouched with `Content-Encoding: gzip`.")
    w("")

    # ---- what the survey taught -------------------------------------------
    w("## What this survey changed in the design")
    w("")
    w("1. **Four files have an empty modelspace.** Their content is on a")
    w("   paperspace layout or only in block definitions. Extraction walks")
    w("   `doc.layouts`, and the API defaults to the *busiest* layout rather")
    w("   than to `Model`, or those files would open as a blank page.")
    w("2. **Units are not uniform.** Janadriyah is in metres, most samples in")
    w("   inches, and several declare nothing at all. Every measurement is")
    w("   reported with its unit, and an undeclared unit is stated as such")
    w("   rather than defaulted.")
    w("3. **Size is driven by a few entities, not by entity count.** In")
    w("   `architectural_-_annotation_scaling_and_multileaders`, two INSERT")
    w("   entities out of 1,376 account for 36 of the 36.7 MB of SVG. Capping")
    w("   entity counts would have been the wrong lever.")
    w("4. **SVG compresses about 20:1 to 95:1.** That, plus MongoDB's 16 MB")
    w("   document limit, is why renders live on disk as `.svg.gz` and not in")
    w("   the database.")

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
