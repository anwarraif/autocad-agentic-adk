"""DWG -> DXF, inside our own image.

Two converters, tried in order:

**ODA File Converter first.** Autodesk's web-viewer extractor crashes
(exit 0xC0000409) on LibreDWG's output for Janadriyah — raw or re-saved,
comments or not — while the original DWG uploads fine, so the conversion is
what breaks the file for that translator. ODA writes Autodesk-faithful DXF,
and it is also simply better at these files: Janadriyah in 10 s instead of
324 s with 6 more modelspace entities recovered, and both blocks_and_tables
drawings — unopenable from LibreDWG output since the start (T-4) — convert
cleanly. It is a Qt app that only ships the xcb platform plugin, hence xvfb
in the image.

**LibreDWG as fallback**, so a missing/broken ODA install never takes DWG
support down with it.

Two things this module does beyond shelling out to a converter.

**It repairs a specific LibreDWG defect.** LibreDWG sometimes wraps a long
value -- typically MTEXT note bodies -- onto a second physical line. DXF is
strictly alternating group-code / value lines, so one wrapped value shifts
every pair after it and the file becomes unreadable, usually surfacing as
`Invalid group code "Standard" at line N`. Rejoining the orphaned tail fixes
it. Measured on the sample set: this is what stood between `truetype.dwg` and
its 90 entities, which the viewer had been showing as an empty drawing.

**It is honest about what it cannot fix.** `blocks_and_tables_-_imperial` and
`blocks_and_tables_-_metric` have a second, deeper defect in the same output
(`Expected DXF entity LINE or SEQEND` -- a broken POLYLINE/SEQEND sequence)
that this repair does not touch. They convert, and they still fail to open.
That is reported, not smoothed over: see `docs/TECH-DEBT.md` T-4.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

log = logging.getLogger(__name__)

#: Name of the LibreDWG converter installed in the image.
DWG2DXF = "dwg2dxf"

#: ODA File Converter binary (see Dockerfile). Preferred when present: its DXF
#: is faithful enough for Autodesk's own web-viewer extractor, which crashes
#: on LibreDWG's output for Janadriyah (docs/TECH-DEBT.md T-13).
ODA = "ODAFileConverter"

#: DXF flavour ODA writes; matches the AC1032/R2018 the pipeline already uses.
ODA_VERSION = "ACAD2018"

#: A large drawing is slow: Janadriyah is 13 MB of DWG and produces 113 MB of
#: DXF. The cap exists so a pathological file cannot wedge an ingest run.
CONVERT_TIMEOUT_S = 900


class ConversionError(RuntimeError):
    """Raised when a DWG cannot be turned into a DXF at all."""


#: What a human can actually do next when both converters have refused. A
#: refusal that only says "conversion failed" makes the operator guess
#: between a broken install, a broken file and a broken transfer; this names
#: the two moves that resolve all three.
NEXT_HUMAN_STEP: Final[str] = (
    "Next step: convert the drawing to DXF on a machine with AutoCAD or ODA "
    "File Converter and ingest that .dxf instead, or ask the sender to "
    "confirm the file size and re-share it (a size mismatch means the "
    "transfer truncated it)."
)


def _refusal(filename: str, converter_lines: Sequence[str]) -> str:
    """The refusal contract: file, every converter tried and why, next step.

    Every intake refusal in this module is built here so that the four parts
    cannot drift apart or go missing one at a time. The converter lines are
    plural on purpose: the pipeline tries two converters, and a message that
    names only the last one to fail sends the operator after the wrong one.
    """
    tried = " ".join(line.rstrip(".") + "." for line in converter_lines)
    return f"{filename}: could not be converted to DXF. {tried} {NEXT_HUMAN_STEP}"


@dataclass
class ConversionResult:
    """What happened, in enough detail to explain a failure."""

    dxf_path: Path
    elapsed_ms: int
    repaired_lines: int
    converter: str = "libredwg"


def is_available() -> bool:
    """True if at least one converter is present in this image."""
    return shutil.which(DWG2DXF) is not None or shutil.which(ODA) is not None


#: How much of a DXF's tail is read to decide whether it is complete. A
#: well-formed DXF ends with the group pair `0 / EOF`; 1 KiB is far more than
#: that needs and is a constant cost regardless of the file's size, which
#: matters when the file is Sedra's 500 MB.
_TAIL_BYTES = 1024

#: First bytes of a binary DXF. The `EOF` tail check below is an ASCII-DXF
#: rule, so binary output is exempted rather than wrongly failed.
_BINARY_DXF_SENTINEL = b"AutoCAD Binary DXF"


def _truncation_reason(target: Path) -> str | None:
    """Why `target` does not look like a complete DXF, or None if it does.

    A converter that dies midway still leaves the bytes it managed to write.
    Those bytes parse as a valid prefix, so an ingest that only checks
    "the file exists and is non-empty" accepts a drawing with an arbitrary
    part of itself missing and reports success. That is the one failure mode
    this whole module exists to prevent, so completeness is checked rather
    than assumed: a DXF ends with the group pair `0` / `EOF`, and a file
    whose last bytes are anything else was cut short.
    """
    size = target.stat().st_size
    if size == 0:
        return "the converter wrote an empty file"
    with target.open("rb") as handle:
        head = handle.read(len(_BINARY_DXF_SENTINEL))
        # Binary DXF has no textual EOF marker; do not judge it by this rule.
        if head.startswith(_BINARY_DXF_SENTINEL):
            return None
        handle.seek(max(0, size - _TAIL_BYTES))
        tail = handle.read()
    if not tail.rstrip(b"\x00 \t\r\n").endswith(b"EOF"):
        return (
            f"the output stops without the closing EOF marker after "
            f"{size} bytes, so it is truncated"
        )
    return None


def _convert_with_oda(dwg_path: Path, out_dir: Path) -> tuple[Path | None, str | None]:
    """Try ODA File Converter.

    Returns `(path, None)` on success and `(None, reason)` on failure, where
    `reason` is a sentence naming what actually went wrong. Returning a bare
    `None` was the defect this signature fixes: every ODA failure looked
    identical to "ODA is not installed", so when the better converter broke
    the user was told only about the fallback that broke afterwards.

    ODA converts folders, not files, so the source's parent is the input
    folder and the exact filename is the filter — nothing else in the folder
    is touched. A Qt app needs a display; the offscreen platform plugin
    stands in for one, with xvfb-run as the fallback for ODA builds that
    reject it.

    ODA reports a bad read by writing `<name>.dxf.err` beside the output
    instead of failing the process, so the exit code alone cannot be trusted
    — `dxf_to_dwg` has always checked that channel and this direction had
    not.
    """
    if shutil.which(ODA) is None:
        return None, f"{ODA} is not installed in this image"
    target = out_dir / f"{dwg_path.stem}.dxf"
    err_file = out_dir / f"{dwg_path.stem}.dxf.err"
    # Stale files from an earlier run would otherwise be read as this run's
    # verdict: a leftover .err fails a good conversion, a leftover .dxf
    # passes a failed one.
    for stale in (target, err_file):
        if stale.exists():
            stale.unlink()
    base = [
        ODA, str(dwg_path.parent), str(out_dir),
        ODA_VERSION, "DXF", "0", "1", dwg_path.name,
    ]
    # The shipped ODA build carries only the xcb platform plugin (measured:
    # offscreen aborts with "Available platform plugins are: xcb"), so xvfb
    # is the path that actually works and goes first. The offscreen attempt
    # stays as a fallback for future ODA builds that do ship it.
    xcb_env = dict(os.environ)
    xcb_env.pop("QT_QPA_PLATFORM", None)
    attempts: list[tuple[list[str], dict[str, str]]] = []
    if shutil.which("xvfb-run") is not None:
        attempts.append((["xvfb-run", "-a", *base], xcb_env))
    attempts.append((base, dict(os.environ, QT_QPA_PLATFORM="offscreen")))

    reasons: list[str] = []
    for cmd, env in attempts:
        label = cmd[0]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=CONVERT_TIMEOUT_S, env=env,
            )
        except subprocess.TimeoutExpired:
            reasons.append(f"{label} exceeded {CONVERT_TIMEOUT_S}s")
            log.warning("ODA attempt timed out", extra={"dwg_file": dwg_path.name})
            continue
        except OSError as exc:
            reasons.append(f"{label} could not be started ({exc})")
            log.warning("ODA attempt failed: %s", exc)
            continue

        if err_file.exists():
            detail = err_file.read_text(errors="replace").strip()[:300]
            reasons.append(f"{label} rejected the drawing ({detail or 'no detail'})")
            # The partial output must not survive to be mistaken for a good
            # conversion by this run's fallback or by a later one.
            if target.exists():
                target.unlink()
            err_file.unlink()
            continue

        if not target.exists():
            noise = (completed.stderr or completed.stdout or "").strip()[:300]
            reasons.append(
                f"{label} exited {completed.returncode} and wrote no output"
                + (f" ({noise})" if noise else "")
            )
            continue

        truncated = _truncation_reason(target)
        if truncated is not None:
            reasons.append(f"{label} produced an unusable file: {truncated}")
            target.unlink()
            continue

        return target, None

    if not reasons:
        reasons.append("no usable way to run it was found in this image")
    return None, "; ".join(reasons)


def repair_wrapped_values(text: str) -> tuple[str, int]:
    """Rejoin DXF value lines that were wrapped onto a line of their own.

    A DXF file is pairs of lines: a group code, then its value. A group code
    is always an integer. So a line that appears where a code is expected and
    is *not* an integer can only be the tail of the previous value, and gluing
    it back restores the alternation for everything that follows.

    Returns the repaired text and the number of joins made. Zero joins means
    the file was already well-formed and nothing was touched.
    """
    lines = text.split("\n")
    out: list[str] = []
    joins = 0
    expect_code = True
    index = 0

    while index < len(lines):
        line = lines[index]
        if expect_code:
            stripped = line.strip()
            if stripped and not stripped.lstrip("+-").isdigit() and out:
                out[-1] = out[-1] + line
                joins += 1
                index += 1
                continue
            out.append(line)
            expect_code = False
        else:
            out.append(line)
            expect_code = True
        index += 1

    return "\n".join(out), joins


def dxf_to_dwg(dxf_path: Path, out_dir: Path) -> Path:
    """Convert one `.dxf` to `.dwg` with ODA (the only converter that writes
    DWG; LibreDWG's writer is experimental and not trusted here).

    Raises ConversionError when ODA is missing or reports a read error — ODA
    writes a `.err` file beside the output instead of failing the process.
    """
    if shutil.which(ODA) is None:
        raise ConversionError(
            f"{ODA} is not installed in this image; rebuild cad-api"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{dxf_path.stem}.dwg"
    err_file = out_dir / f"{dxf_path.stem}.dwg.err"
    for stale in (target, err_file):
        if stale.exists():
            stale.unlink()
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)
    try:
        subprocess.run(
            ["xvfb-run", "-a", ODA, str(dxf_path.parent), str(out_dir),
             ODA_VERSION, "DWG", "0", "1", dxf_path.name],
            capture_output=True, text=True,
            timeout=CONVERT_TIMEOUT_S, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConversionError(
            f"{dxf_path.name}: DWG conversion exceeded {CONVERT_TIMEOUT_S}s"
        ) from exc
    if err_file.exists():
        detail = err_file.read_text(errors="replace")[:300]
        raise ConversionError(f"{dxf_path.name}: ODA rejected the DXF. {detail}")
    if not target.exists() or target.stat().st_size == 0:
        raise ConversionError(f"{dxf_path.name}: ODA produced no DWG output")
    return target


def convert(dwg_path: Path, out_dir: Path) -> ConversionResult:
    """Convert one `.dwg` to `.dxf`, repairing wrapped values on the way.

    Args:
        dwg_path: the source drawing.
        out_dir: directory the `.dxf` is written into.

    Raises:
        ConversionError: if the converter is missing, fails, or writes nothing.
    """
    if not is_available():
        raise ConversionError(
            _refusal(
                dwg_path.name,
                [
                    f"{ODA} (ODA File Converter): not installed in this image",
                    f"{DWG2DXF} (LibreDWG): not installed in this image",
                ],
            )
        )

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    oda_target, oda_reason = _convert_with_oda(dwg_path, out_dir)
    if oda_target is not None:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "converted dwg via ODA",
            extra={
                "dwg_file": dwg_path.name,
                "bytes_out": oda_target.stat().st_size,
                "elapsed_ms": elapsed_ms,
            },
        )
        return ConversionResult(
            dxf_path=oda_target, elapsed_ms=elapsed_ms,
            repaired_lines=0, converter="oda",
        )

    # ODA did not produce a file. Its reason is carried from here on so that
    # the refusal can name BOTH converters: being told only how the fallback
    # failed hides which converter the operator actually needs to fix.
    oda_line = f"{ODA} (ODA File Converter): {oda_reason}"
    target = out_dir / f"{dwg_path.stem}.dxf"
    if target.exists():
        target.unlink()

    try:
        completed = subprocess.run(
            [DWG2DXF, "-o", str(target), str(dwg_path)],
            capture_output=True,
            text=True,
            timeout=CONVERT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConversionError(
            _refusal(
                dwg_path.name,
                [oda_line, f"{DWG2DXF} (LibreDWG): exceeded {CONVERT_TIMEOUT_S}s"],
            )
        ) from exc
    except OSError as exc:
        raise ConversionError(
            _refusal(
                dwg_path.name,
                [oda_line, f"{DWG2DXF} (LibreDWG): could not be started ({exc})"],
            )
        ) from exc

    if not target.exists():
        detail = (completed.stderr or completed.stdout or "").strip()[:300]
        raise ConversionError(
            _refusal(
                dwg_path.name,
                [
                    oda_line,
                    f"{DWG2DXF} (LibreDWG): exited {completed.returncode} and "
                    f"wrote no output"
                    + (f" ({detail})" if detail else ""),
                ],
            )
        )

    # LibreDWG exits non-zero on almost every file while still writing a good
    # DXF, so the file is the test rather than the return code (this is why
    # the exit code above is only ever quoted, never trusted). What the file
    # must not be is half-written.
    truncated = _truncation_reason(target)
    if truncated is not None:
        detail = (completed.stderr or completed.stdout or "").strip()[:300]
        target.unlink()
        raise ConversionError(
            _refusal(
                dwg_path.name,
                [
                    oda_line,
                    f"{DWG2DXF} (LibreDWG): produced an unusable file, "
                    f"{truncated}" + (f" ({detail})" if detail else ""),
                ],
            )
        )

    # LibreDWG writes warnings to stderr for almost every file; a non-zero exit
    # with a usable file is normal, so the file itself is the test, not the
    # return code.
    raw = target.read_text(encoding="utf-8", errors="surrogateescape")
    repaired, joins = repair_wrapped_values(raw)
    if joins:
        target.write_text(repaired, encoding="utf-8", errors="surrogateescape")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "converted dwg",
        extra={
            "dwg_file": dwg_path.name,
            "bytes_out": target.stat().st_size,
            "repaired_lines": joins,
            "elapsed_ms": elapsed_ms,
        },
    )
    return ConversionResult(
        dxf_path=target, elapsed_ms=elapsed_ms, repaired_lines=joins
    )
