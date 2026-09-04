"""Tables drawn into a drawing, read as tables.

An `ACAD_TABLE` is a real table: rows, columns, and a text value per cell. It
renders as lines and letters, so to a viewer it is indistinguishable from any
other geometry, and to a text search its contents are 4,898 unrelated strings.
Read as a table it is a **schedule** -- in the reference drawing, 60 of them
holding 2,449 rows of plot number and plot area, written by the engineer who
drew the plots.

That matters more than it sounds. Until now every area in this system was
*measured* from geometry. A schedule is the same quantity *stated* by the
author, which makes two things possible that one source alone cannot do:

- answer "what is the area of plot 2043" with what the drawing says, not only
  with what its outline computes to;
- compare the two, everywhere, and report where they disagree. A plot whose
  drawn outline does not match its scheduled area is a defect in the drawing,
  and finding those is the work itself.

Nothing here knows what a plot is. A table is a grid of strings; which column
identifies a thing and which measures it is worked out from the values, and
the meaning of the measure is then **confirmed against geometry** rather than
assumed. A drawing whose tables are a door schedule or a pipe list parses
exactly the same way and simply finds no agreement to report.

Pure: give it a path, get back grids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

#: The entity that holds a table. `TABLE` is deliberately absent: in DXF that
#: word introduces a symbol-table section (layers, styles, views), not a
#: drawn table, and matching it would read the drawing's dictionary as data.
TABLE_TYPES = frozenset({"ACAD_TABLE"})

#: Group codes inside `AcDbTable`.
CODE_ROWS = "91"
CODE_COLS = "92"
CODE_CELL_TEXT = "302"
CODE_SUBCLASS = "100"
SUBCLASS_TABLE = "AcDbTable"

#: MTEXT formatting, which a cell value carries and a reader does not want:
#: a number written in colour 7 arrives wrapped in a colour code. The
#: leading backslash is matched one-or-more times because DXF escapes it,
#: so the same code appears with one backslash or with two depending on who
#: wrote the file -- and a pattern that assumes one leaves the other behind,
#: turning 3401 into a string that is not a number at all.
_MTEXT_CODE = re.compile(r"\\+[A-Za-z][^;\\]*;")
_MTEXT_BREAK = re.compile(r"\\+P")
_MTEXT_BRACE = re.compile(r"[{}]")

#: A cap, because a runaway parse on a 136 MB file should stop rather than
#: fill memory. Far above the largest schedule seen (2,449 rows).
MAX_CELLS = 500_000


@dataclass(slots=True)
class Table:
    """One drawn table: where it is, how big it is, what it says."""

    handle: str
    layer: str | None = None
    block: str | None = None
    rows: int | None = None
    cols: int | None = None
    cells: list[str] = field(default_factory=list)

    @property
    def grid(self) -> list[list[str]]:
        """Cells as rows, using the column count the table declares.

        Falls back to one row when the count is missing or nonsensical: a
        wrong shape is worse than a flat list, because it silently pairs the
        wrong values together.
        """
        width = self.cols if self.cols and self.cols > 0 else None
        if width is None or width > len(self.cells):
            return [list(self.cells)]
        return [self.cells[i : i + width] for i in range(0, len(self.cells), width)]

    def as_dict(self) -> dict[str, object]:
        return {
            "handle": self.handle,
            "layer": self.layer,
            "block": self.block,
            "rows": self.rows,
            "cols": self.cols,
            "cells": len(self.cells),
        }


def clean_cell(value: str) -> str:
    """A cell's text without the formatting wrapped around it."""
    out = _MTEXT_BREAK.sub(" ", value)
    out = _MTEXT_CODE.sub("", out)
    out = _MTEXT_BRACE.sub("", out)
    return out.strip()


def read(path: Path) -> list[Table]:
    """Every drawn table in the file, in the order the file lists them.

    Streamed for the same reason as the embedded payloads: the reference
    drawing is 136 MB and `ezdxf` would hold all of it to answer a question
    about sixty entities.
    """
    return list(_tables(path))


def pairs(path: Path) -> Iterator[tuple[str, str]]:
    """A DXF as the alternating (code, value) lines it actually is.

    Reading it any other way is a trap this cost an hour to. A parser that
    decides "the previous line was a group code because it looked like a
    number" will read the *value* `58` as a code, and then read the layer
    name `0` as the code that starts a new entity -- which closes every table
    two lines after it opens and reports sixty empty ones. Codes and values
    strictly alternate; nothing else needs guessing.
    """
    code: str | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            text = line.rstrip("\r\n")
            if code is None:
                code = text.strip()
                continue
            yield code, text.strip()
            code = None


def _tables(path: Path) -> Iterator[Table]:
    current: Table | None = None
    block: str | None = None
    in_block_record = False
    in_table_subclass = False
    total_cells = 0

    for code, value in pairs(path):
        if code == "0":
            if current is not None:
                yield current
                current = None
            in_table_subclass = False
            if value == "BLOCK":
                in_block_record = True  # its name arrives on the next code 2
                block = None
            elif value == "ENDBLK":
                in_block_record = False
                block = None
            elif value in TABLE_TYPES:
                current = Table(handle="?", block=block)
            continue

        if in_block_record and code == "2" and block is None and value:
            # The block being read. A table inside one belongs to it, which is
            # how a schedule split across sheets keeps its place.
            block = value
            continue

        if current is None:
            continue

        if code == "5" and current.handle == "?":
            current.handle = value
        elif code == "8" and current.layer is None:
            current.layer = value
        elif code == CODE_SUBCLASS:
            in_table_subclass = value == SUBCLASS_TABLE
        elif in_table_subclass and code == CODE_ROWS and current.rows is None:
            current.rows = _as_int(value)
        elif in_table_subclass and code == CODE_COLS and current.cols is None:
            current.cols = _as_int(value)
        elif code == CODE_CELL_TEXT and value and total_cells < MAX_CELLS:
            current.cells.append(clean_cell(value))
            total_cells += 1

    if current is not None:
        yield current


def _as_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


# --- reading a grid as a schedule --------------------------------------------


def as_number(text: str) -> float | None:
    """A cell's value as a number, or `None` if it is not one.

    Tolerant of the separators a draughtsman types -- `1,234.5`, `1 234,5` --
    and of a unit written after the figure, because a schedule is typed by a
    person and a stricter reader would drop real rows.
    """
    raw = text.strip()
    if not raw:
        return None
    raw = raw.replace(" ", " ").replace(" ", "")
    if "," in raw and "." in raw:
        raw = raw.replace(",", "")
    elif raw.count(",") == 1 and len(raw.split(",")[-1]) != 3:
        raw = raw.replace(",", ".")
    else:
        raw = raw.replace(",", "")
    match = re.match(r"^[-+]?\d*\.?\d+", raw)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Column:
    """What one column of a schedule turned out to be."""

    index: int
    kind: str  # "identifier" | "measure" | "text"
    reason: str
    filled: int
    numeric: int
    unique: int
    minimum: float | None = None
    maximum: float | None = None


def describe_columns(grid: list[list[str]]) -> list[Column]:
    """What each column is, argued from its values.

    An identifier is a column of whole numbers that never repeats and that
    covers its own range densely -- 1 to 2,449 with no gaps is a numbering,
    not a measurement. A measure is any other numeric column. Everything else
    is text. No column name is needed and none is trusted: the reference
    drawing's schedule has no header row at all.
    """
    width = max((len(row) for row in grid), default=0)
    out: list[Column] = []
    for index in range(width):
        values = [row[index] for row in grid if index < len(row)]
        filled = [v for v in values if v.strip()]
        numbers = [n for n in (as_number(v) for v in filled) if n is not None]
        unique = len(set(filled))

        if not numbers:
            out.append(
                Column(
                    index=index,
                    kind="text",
                    reason="no value here can be read as a number",
                    filled=len(filled),
                    numeric=0,
                    unique=unique,
                )
            )
            continue

        whole = all(float(n).is_integer() for n in numbers)
        distinct = len(set(numbers)) == len(numbers)
        span = max(numbers) - min(numbers) + 1 if numbers else 0
        dense = bool(numbers) and span > 0 and len(numbers) / span >= 0.9

        if whole and distinct and dense and len(numbers) >= 2:
            kind = "identifier"
            reason = (
                f"{len(numbers)} distinct whole numbers covering the range "
                f"{int(min(numbers))}-{int(max(numbers))} with no real gaps"
            )
        else:
            kind = "measure"
            why = []
            if not whole:
                why.append("some values are fractional")
            if not distinct:
                why.append("some values repeat")
            if not dense:
                why.append("the values do not cover their own range")
            reason = "; ".join(why) or "it holds numbers"
        out.append(
            Column(
                index=index,
                kind=kind,
                reason=reason,
                filled=len(filled),
                numeric=len(numbers),
                unique=unique,
                minimum=min(numbers),
                maximum=max(numbers),
            )
        )
    return out


def merge_grids(tables: list[Table]) -> list[list[str]]:
    """Tables of the same shape read as one schedule.

    A schedule too long for a sheet is split across several tables, and the
    parts are the same table. Only tables that agree on column count are
    merged; a table of a different width is a different thing and is left
    alone.
    """
    widths: dict[int, int] = {}
    for table in tables:
        width = len(table.grid[0]) if table.grid else 0
        widths[width] = widths.get(width, 0) + len(table.grid)
    if not widths:
        return []
    common = max(widths, key=lambda w: widths[w])
    rows: list[list[str]] = []
    for table in tables:
        grid = table.grid
        if grid and len(grid[0]) == common:
            rows.extend(grid)
    return rows
