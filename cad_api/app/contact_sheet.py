"""Rasterise every cached render into one contact sheet, so it can be *looked at*.

Why this exists
---------------
A whole night's work shipped a viewer in which every line was drawn ~700x too
thick, turning 20,000 entities into five coloured blobs. Every automated check
passed: the SVG was well-formed, carried 20,312 `data-handle` groups, had a
sane viewBox and compressed normally. None of them looked at the picture.

Counting attributes is not seeing. This makes seeing a single command:

    python -m app.contact_sheet /data/svg/contact-sheet.png

Requires cairosvg (see requirements-dev.txt); it is a verification tool, not a
runtime dependency of the service.
"""

from __future__ import annotations

import argparse
import gzip
import re
import io
import logging
import sys
from pathlib import Path

from . import store
from .config import get_settings
from .logging_conf import configure_logging

log = logging.getLogger(__name__)

TILE_W = 460
TILE_H = 320
COLS = 4
PAD = 10
LABEL_H = 30


def _for_raster(svg: bytes) -> bytes:
    """Make the SVG rasterisable by cairosvg without changing what it shows.

    The served SVG relies on `vector-effect: non-scaling-stroke`, so its
    stroke widths are in *screen pixels* (1-4). cairosvg does not implement
    that property and reads them as viewBox units instead -- against a
    1,000,000-unit viewBox a 1-unit line is a millionth of the width, i.e.
    invisible. Filled shapes still appear, which is worse than nothing: a
    line-only drawing rasterises to a blank sheet and looks broken when it is
    fine in the browser.

    So for rasterisation only, the property is dropped and stroke width is
    made proportional to the viewBox. Line weights are then indicative rather
    than exact -- fine for a contact sheet, whose job is "is the drawing
    there and does it look right", not exact plotting.
    """
    text = svg.decode("utf-8")
    match = re.search(r'viewBox="[\d.-]+ [\d.-]+ ([\d.]+) ', text)
    span = float(match.group(1)) if match else 1_000_000.0
    width = max(span / 1200.0, 1.0)
    text = text.replace("path{vector-effect:non-scaling-stroke}", "")
    text = text.replace(
        "</style>",
        f"path{{stroke-width:{width:.1f} !important}}</style>",
        1,
    )
    return text.encode("utf-8")


def _render_tile(svg_bytes: bytes, width: int) -> "Image.Image | None":
    import cairosvg  # imported lazily: dev-only dependency
    from PIL import Image

    try:
        png = cairosvg.svg2png(bytestring=_for_raster(svg_bytes), output_width=width)
    except Exception as exc:  # noqa: BLE001 - a bad tile must not kill the sheet
        log.warning("rasterisation failed: %s", exc)
        return None
    return Image.open(io.BytesIO(png)).convert("RGB")


def build(output: Path, layouts_per_drawing: int = 1) -> int:
    """Write a contact sheet of the busiest layout of every ingested drawing."""
    from PIL import Image, ImageDraw

    settings = get_settings()
    drawings = store.list_drawings()

    tiles: list[tuple[str, "Image.Image | None"]] = []
    for drawing in sorted(drawings, key=lambda d: -d.get("entity_count", 0)):
        renders = store.list_render_meta(drawing["_id"])
        if not renders:
            tiles.append((f"{drawing['original_filename']}\n(nothing drawable)", None))
            continue
        for meta in sorted(
            renders, key=lambda r: -r.get("rendered_entities", 0)
        )[:layouts_per_drawing]:
            payload = store.load_svg_gzip(
                settings.svg_dir, drawing["_id"], meta["layout"]
            )
            if payload is None:
                tiles.append((f"{drawing['original_filename']}\n(no cached file)", None))
                continue
            svg = gzip.decompress(payload)
            label = (
                f"{drawing['original_filename']}\n"
                f"{meta['layout']} · {meta.get('rendered_entities', 0):,} entities"
            )
            tiles.append((label, _render_tile(svg, TILE_W)))

    if not tiles:
        print("no renders to show", file=sys.stderr)
        return 1

    rows = (len(tiles) + COLS - 1) // COLS
    sheet_w = COLS * (TILE_W + PAD) + PAD
    sheet_h = rows * (TILE_H + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), "#1b1b1b")
    draw = ImageDraw.Draw(sheet)

    for index, (label, tile) in enumerate(tiles):
        col, row = index % COLS, index // COLS
        x = PAD + col * (TILE_W + PAD)
        y = PAD + row * (TILE_H + LABEL_H + PAD)

        if tile is None:
            draw.rectangle([x, y, x + TILE_W, y + TILE_H], fill="#2a1414")
            draw.text((x + 8, y + TILE_H // 2), "no render", fill="#f87171")
        else:
            tile.thumbnail((TILE_W, TILE_H))
            # Centre the thumbnail in its cell so aspect ratios stay honest.
            sheet.paste(
                tile,
                (x + (TILE_W - tile.width) // 2, y + (TILE_H - tile.height) // 2),
            )
            draw.rectangle(
                [x, y, x + TILE_W, y + TILE_H], outline="#3a4a5c", width=1
            )

        for line_no, line in enumerate(label.split("\n")):
            draw.text((x + 2, y + TILE_H + 2 + line_no * 13), line, fill="#c9d4de")

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"wrote {output} ({sheet_w}x{sheet_h}, {len(tiles)} tiles)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", nargs="?", default="/data/svg/contact-sheet.png", type=Path
    )
    parser.add_argument("--layouts", type=int, default=1)
    args = parser.parse_args(argv)

    configure_logging(get_settings().log_level)
    return build(args.output, layouts_per_drawing=args.layouts)


if __name__ == "__main__":
    raise SystemExit(main())
