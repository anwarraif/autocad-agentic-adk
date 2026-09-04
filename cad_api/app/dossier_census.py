"""Naming what is inside a bucket — DOSSIER Phase 1, lane 4.

The failure this campaign exists to end was "how many roads?" answered "0"
while 1,233 road entities sat in a bucket nobody had counted. The same hole
has two more costumes, and both of them are this module's job:

* **A points layer counted but not named.** "412 inserts" does not answer
  "how many trees / hydrants / light poles" — the answer is in the
  `block_name`, so `block_census` counts per name.
* **An annotation layer counted but not read.** "2,449 texts" does not answer
  "what is plot 2092?" — the answer is in the CONTENT, so
  `annotation_census` classifies by content and publishes the numeric range,
  the Arabic readings and a sample of everything left over.

Both functions are **pure over already-fetched rows**: a list of entity dicts
in, a plain dict out. No MongoDB, no FastAPI, no I/O. `app.shx` is imported
read-only because it is dependency-free and because its `PHRASES` table is
human-verified evidence, not a heuristic.

Three rules are load-bearing here:

1. **Content decides, never a layer name** (rule G1). A layer called
   `TEXT-ARABIC` full of plot numbers is classified as plot numbers. The test
   suite runs the same rows under two contradictory layer names and demands
   byte-identical classes.
2. **Nothing is silently truncated** (rule G7). Every cap is a named constant,
   published in the response beside the number of things it dropped.
3. **Absence is reported, never zeroed** (rule G8). Empty input returns an
   empty-but-well-formed document. Rows carrying no text are counted and given
   an example rather than quietly skipped.

### Why every count is stamped with its layout

The label `2092` exists twice in the reference drawing: handle `27321CB` in
`Model`, and handle `2600C0A` inside the block definition `[block] shml`. Those
are **one label drawn once and placed once**, not two plots. A census that
answers "2 plots numbered 2092" would be repeating the mistake in miniature.

So the shape refuses to be flattened:

* every class and every block name carries `by_layout`, so a count can never
  be read without the layouts it came from;
* every `example` carries its own `layout`, so a quoted example drags its
  provenance along;
* `scope` states the layers and layouts the rows actually covered, and
  `scope_warnings` says so out loud when there is more than one;
* `repeated_values` — the only field that could ever produce the sentence
  "the same label appears N times" — is computed **within one layout** and
  never across layouts. Feed both `2092` rows to this function and it reports
  no repeat, because there is none.

The natural way to combine two of these documents is therefore a merge keyed by
layout, not an addition of two integers.
"""

from __future__ import annotations

import re
from typing import Any, Final

from . import shx

__all__ = [
    "block_census",
    "annotation_census",
    "classify_text",
    "KINDS",
    "KIND_NUMERIC",
    "KIND_DIMENSION",
    "KIND_ARABIC",
    "KIND_OTHER",
    "MAX_BLOCK_NAMES",
    "MAX_SAMPLES",
    "MAX_READINGS",
    "MAX_REPEATED_VALUES",
    "MAX_HANDLES_PER_VALUE",
    "MAX_ATTRIBUTE_KEYS",
    "LAYOUT_UNKNOWN",
]


# --- stated limits (rule G7) ------------------------------------------------
#
# Each one is published in the response it constrains, together with the count
# of what it dropped. A cap that is not reported is a silent truncation.

#: Block names reported by `block_census`, keeping the most numerous.
MAX_BLOCK_NAMES: Final[int] = 200

#: Distinct attribute tags listed per block name.
MAX_ATTRIBUTE_KEYS: Final[int] = 20

#: Distinct sample strings kept per annotation class.
MAX_SAMPLES: Final[int] = 5

#: Distinct Arabic readings listed for the `arabic_shx` class.
MAX_READINGS: Final[int] = 10

#: Repeated label values reported, within one layout.
MAX_REPEATED_VALUES: Final[int] = 50

#: Handles listed per repeated value.
MAX_HANDLES_PER_VALUE: Final[int] = 10

#: Written where a row carries no layout. A named placeholder, not a guess:
#: `None` silently sorts and merges with everything else.
LAYOUT_UNKNOWN: Final[str] = "(layout not recorded)"


# --- the four content classes ----------------------------------------------

KIND_NUMERIC: Final[str] = "numeric_label"
KIND_DIMENSION: Final[str] = "dimension_value"
KIND_ARABIC: Final[str] = "arabic_shx"
KIND_OTHER: Final[str] = "other"

#: Publication order of the classes. Fixed so two runs read the same.
KINDS: Final[tuple[str, ...]] = (
    KIND_NUMERIC,
    KIND_DIMENSION,
    KIND_ARABIC,
    KIND_OTHER,
)

#: How `kind` is decided, shipped inside every annotation census.
#:
#: In the response because a reader who cannot see the rule cannot check the
#: verdict, and because it names the one thing the rule never looks at.
CLASSIFICATION_BASIS: Final[str] = (
    "kind is decided from the text CONTENT alone and never from the layer "
    "name (rule G1), first match wins: "
    "1) numeric_label - the whole string is one integer, no fraction and no "
    "unit; "
    "2) dimension_value - the whole string is a number carrying a measurement "
    "mark: a decimal fraction, a unit suffix, a diameter or radius prefix, a "
    "tolerance, or an 'x'-separated pair; "
    "3) arabic_shx - app.shx reads it, through its human-verified phrase table "
    "or through the keyboard map; "
    "4) other - everything else, always published with an example and samples."
)

#: Why no range ever carries a unit here.
UNIT_BASIS: Final[str] = (
    "null, and deliberately: the census reads the text as typed and is never "
    "told the drawing's unit, so it states none rather than defaulting to one "
    "(rule G2). The numbers are the characters in the file."
)

#: The sentence that keeps `repeated_values` honest.
REPEATED_BASIS: Final[str] = (
    "counted WITHIN one layout only. The same string in model space and inside "
    "a block definition is one piece of geometry seen twice, not two things, "
    "so it is never reported as a repeat."
)

#: The sentence that keeps a total from being added to another total.
SCOPE_BASIS: Final[str] = (
    "counts are per layer x layout. Totals from two censuses must be merged "
    "per layout via `by_layout`, never added into one figure."
)


# --- content patterns -------------------------------------------------------
#
# Generic CAD annotation shapes, not conventions of any one drawing (rule G1).

#: A number as a draughtsman types it: optional sign, optional thousands
#: separators, optional decimal fraction.
_NUM: Final[str] = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

#: The whole string is one integer. Checked FIRST, because the dimension
#: pattern below would otherwise swallow a bare plot number.
_INTEGER_ONLY: Final[re.Pattern[str]] = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
)

#: Unit suffixes and dimension prefixes seen on annotation in any CAD file.
_UNIT: Final[str] = r"(?:mm|cm|dm|km|m|in|ft|yd|sqm|m2|'|\"|°|deg)"
_DIM_PREFIX: Final[str] = r"(?:[Øø⌀∅Φφ]|R|DIA\.?|RAD\.?)"

_DIM_SINGLE: Final[re.Pattern[str]] = re.compile(
    rf"\s*{_DIM_PREFIX}?\s*{_NUM}\s*{_UNIT}?\s*"
    rf"(?:(?:±|\+/-)\s*{_NUM}\s*{_UNIT}?\s*)?",
    re.IGNORECASE,
)

#: `12.00 x 25.00` — a size, not two labels.
_DIM_PAIR: Final[re.Pattern[str]] = re.compile(
    rf"\s*{_NUM}\s*{_UNIT}?\s*[x×]\s*{_NUM}\s*{_UNIT}?\s*",
    re.IGNORECASE,
)

_NUMBER_TOKEN: Final[re.Pattern[str]] = re.compile(_NUM)
_NON_DIGIT: Final[re.Pattern[str]] = re.compile(r"\D")


# --- small helpers ----------------------------------------------------------


def _layout_of(row: Any) -> str:
    """The layout of a row, or a named placeholder when it carries none."""
    value = row.get("layout") if hasattr(row, "get") else None
    text = "" if value is None else str(value).strip()
    return text or LAYOUT_UNKNOWN


def _numbers_in(text: str) -> list[float]:
    """Every number in a string, as typed. Never raises on a broken token."""
    out: list[float] = []
    for token in _NUMBER_TOKEN.findall(text):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:  # pragma: no cover - the pattern already excludes it
            continue
    return out


def _digit_count(text: str) -> int:
    """How many digits a label has — `1,234` and `1234` are both four."""
    return len(_NON_DIGIT.sub("", text))


def _font_of(row: Any) -> tuple[str | None, str | None, bool]:
    """The row's resolved text-style font, and whether it *was* resolved.

    `style_known` is not `font is not None`: "the style is unknown" and "the
    style is known and declares no font" are different states, and `app.shx`
    treats them differently on purpose. A row that carries the key at all —
    even holding `None` — has been resolved by the caller.
    """
    has_key = ("text_style_font" in row) or ("text_style_bigfont" in row)
    if not has_key:
        return None, None, False
    font = row.get("text_style_font")
    bigfont = row.get("text_style_bigfont")
    return (
        None if font is None else str(font),
        None if bigfont is None else str(bigfont),
        True,
    )


def classify_text(
    text: str,
    *,
    font: str | None = None,
    bigfont: str | None = None,
    style_known: bool = False,
) -> tuple[str, shx.Reading | None]:
    """The class of one annotation string, decided from the string.

    Exposed because the rule is worth testing on its own, and because a caller
    that wants to explain one entity should be able to reach the same verdict
    the census reached. Returns `(kind, reading)`; `reading` is populated only
    for `arabic_shx`.
    """
    stripped = text.strip()
    if not stripped:
        return KIND_OTHER, None
    if _INTEGER_ONLY.fullmatch(stripped):
        return KIND_NUMERIC, None
    if _DIM_PAIR.fullmatch(stripped) or _DIM_SINGLE.fullmatch(stripped):
        return KIND_DIMENSION, None
    reading = shx.read(text, font=font, bigfont=bigfont, style_known=style_known)
    if reading.reading:
        return KIND_ARABIC, reading
    return KIND_OTHER, None


def _scope(rows: list[Any]) -> tuple[dict[str, Any], list[str]]:
    """The layers and layouts these rows actually covered, plus any warning."""
    layers = sorted(
        {
            str(r.get("layer")).strip()
            for r in rows
            if r.get("layer") is not None and str(r.get("layer")).strip()
        }
    )
    layouts = sorted({_layout_of(r) for r in rows})
    unnamed_layers = sum(
        1
        for r in rows
        if r.get("layer") is None or not str(r.get("layer")).strip()
    )

    warnings: list[str] = []
    if len(layouts) > 1:
        warnings.append(
            f"these rows span {len(layouts)} layouts ({', '.join(layouts)}). "
            "Read the counts per layout through `by_layout`; adding them "
            "together would report one thing drawn twice as two things."
        )
    if len(layers) > 1:
        warnings.append(
            f"these rows span {len(layers)} layers ({', '.join(layers)}); a "
            "census is defined per layer x layout."
        )
    if unnamed_layers:
        warnings.append(
            f"{unnamed_layers} row(s) carry no layer, so the scope of this "
            "census is not fully determined."
        )

    scope = {
        "layers": layers,
        "layouts": layouts,
        "single_scope": len(layers) == 1 and len(layouts) == 1,
        "basis": SCOPE_BASIS,
    }
    return scope, warnings


def _annotation_example(row: Any, reading: shx.Reading | None) -> dict[str, Any]:
    """One quotable row. Always carries its layout, so it cannot be misquoted.

    `contained_by` is passed through when the caller has already attached it.
    This module cannot compute containment — that lives in the spatial store
    and needs MongoDB — but a label whose parcel is known should not lose it on
    the way through the census, because "plot 2092 is in parcel X" is the
    answer and "there is a label reading 2092" is not.
    """
    out: dict[str, Any] = {
        "handle": row.get("handle"),
        "text": row.get("text"),
        "type": row.get("type"),
        "layer": row.get("layer"),
        "layout": _layout_of(row),
    }
    if reading is not None and reading.reading:
        out["text_reading"] = reading.reading
        out["reading_method"] = reading.method
    contained_by = row.get("contained_by")
    if contained_by:
        out["contained_by"] = contained_by
    return out


def _block_example(row: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "handle": row.get("handle"),
        "block_name": row.get("block_name"),
        "type": row.get("type"),
        "layer": row.get("layer"),
        "layout": _layout_of(row),
    }
    return out


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items()))


# --- block census -----------------------------------------------------------


def block_census(rows: list[dict]) -> dict:
    """Counts per block name for a points layer, with its truncation stated.

    `rows` are entity dicts for one layer x layout, carrying at least
    `block_name`; `handle`, `type`, `layout`, `layer` and `attribs` are used
    when present.

    Returns::

        {"blocks": [{"block_name", "count", "by_layout", "example",
                     "attribute_keys", "attribute_keys_omitted"}],
         "total", "named", "distinct_names", "without_block_name",
         "scope", "scope_warnings", "truncated", "truncation"}

    `blocks` is ordered by count descending, then by name, so the answer to
    "how many of X" is the first line and two runs read alike. Rows with no
    block name are never dropped: they are counted by type in
    `without_block_name`, because a non-INSERT on a points layer is ordinary
    while an INSERT without a name is an anomaly, and the two must stay
    distinguishable.

    An empty input returns this same document with empty parts (rule G8).
    """
    rows = list(rows or [])

    counts: dict[str, int] = {}
    by_layout: dict[str, dict[str, int]] = {}
    examples: dict[str, dict[str, Any]] = {}
    attribute_keys: dict[str, set[str]] = {}

    unnamed_count = 0
    unnamed_by_type: dict[str, int] = {}
    unnamed_example: dict[str, Any] | None = None

    for row in rows:
        raw_name = row.get("block_name")
        name = "" if raw_name is None else str(raw_name).strip()
        if not name:
            unnamed_count += 1
            kind = str(row.get("type") or "(type not recorded)")
            unnamed_by_type[kind] = unnamed_by_type.get(kind, 0) + 1
            if unnamed_example is None:
                unnamed_example = _block_example(row)
            continue

        counts[name] = counts.get(name, 0) + 1
        layout = _layout_of(row)
        bucket = by_layout.setdefault(name, {})
        bucket[layout] = bucket.get(layout, 0) + 1
        if name not in examples:
            examples[name] = _block_example(row)
        attribs = row.get("attribs")
        if isinstance(attribs, dict):
            attribute_keys.setdefault(name, set()).update(
                str(k) for k in attribs.keys()
            )

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = ordered[:MAX_BLOCK_NAMES]
    dropped = ordered[MAX_BLOCK_NAMES:]

    attribute_keys_omitted = 0
    blocks: list[dict[str, Any]] = []
    for name, count in kept:
        keys = sorted(attribute_keys.get(name, set()))
        omitted_keys = max(0, len(keys) - MAX_ATTRIBUTE_KEYS)
        attribute_keys_omitted += omitted_keys
        blocks.append(
            {
                "block_name": name,
                "count": count,
                "by_layout": _sorted_counts(by_layout.get(name, {})),
                "example": examples.get(name),
                "attribute_keys": keys[:MAX_ATTRIBUTE_KEYS],
                "attribute_keys_omitted": omitted_keys,
            }
        )

    names_omitted = len(dropped)
    entities_omitted = sum(count for _, count in dropped)
    scope, warnings = _scope(rows)

    return {
        "blocks": blocks,
        "total": len(rows),
        "named": sum(counts.values()),
        "distinct_names": len(counts),
        "without_block_name": {
            "count": unnamed_count,
            "by_type": _sorted_counts(unnamed_by_type),
            "example": unnamed_example,
            "reason": (
                "these rows carry no `block_name`. A non-INSERT on a points "
                "layer simply is not a placed symbol; an INSERT without a name "
                "is an anomaly worth chasing. Counted here rather than dropped."
            ),
        },
        "scope": scope,
        "scope_warnings": warnings,
        "truncated": bool(names_omitted or attribute_keys_omitted),
        "truncation": {
            "block_name_cap": MAX_BLOCK_NAMES,
            "attribute_key_cap": MAX_ATTRIBUTE_KEYS,
            "names_omitted": names_omitted,
            "entities_omitted": entities_omitted,
            "attribute_keys_omitted": attribute_keys_omitted,
            "basis": (
                f"the {MAX_BLOCK_NAMES} most numerous names are reported; any "
                "omitted name and the entities behind it are counted above, "
                "never dropped in silence (rule G7)."
            ),
        },
    }


# --- annotation census ------------------------------------------------------


def _class_document(
    kind: str,
    count: int,
    by_layout: dict[str, int],
    example: dict[str, Any] | None,
    texts: list[str],
    numbers: list[float],
    digits: list[int],
    readings: list[dict[str, Any]],
) -> dict[str, Any]:
    """One class, in the shape the contract fixed, plus its evidence."""
    distinct = sorted(set(texts))
    samples = distinct[:MAX_SAMPLES]
    doc: dict[str, Any] = {
        "kind": kind,
        "count": count,
        "distinct_texts": len(distinct),
        "example": example,
        "samples": samples,
        "samples_omitted": max(0, len(distinct) - len(samples)),
        "by_layout": _sorted_counts(by_layout),
    }

    if kind in (KIND_NUMERIC, KIND_DIMENSION) and numbers:
        doc["range"] = {
            "min": min(numbers),
            "max": max(numbers),
            "unit": None,
            "unit_basis": UNIT_BASIS,
            # Only a label has a meaningful digit count. `12.00` has four
            # digits and the fact says nothing, so it is null there rather
            # than a number waiting to be misread.
            "digits_min": min(digits) if digits else None,
            "digits_max": max(digits) if digits else None,
        }
        doc["range_basis"] = (
            "min and max of the numbers exactly as they are typed in the "
            "drawing; the range is what turns a count into an answer."
        )
    else:
        doc["range"] = None
        doc["range_basis"] = (
            "null: this class carries no numeric value to range over."
        )

    if kind == KIND_ARABIC:
        unique: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in readings:
            key = (str(item["text"]), str(item["reading"]))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        doc["readings"] = unique[:MAX_READINGS]
        doc["readings_omitted"] = max(0, len(unique) - MAX_READINGS)
        doc["reading_note"] = shx.READING_NOTE

    return doc


def annotation_census(rows: list[dict]) -> dict:
    """Content classes for an annotation layer.

    `rows` are entity dicts for one layer x layout, carrying at least `text`;
    `handle`, `type`, `layout`, `layer` and — when the caller has resolved the
    text style — `text_style_font` / `text_style_bigfont` are used when present.

    Returns::

        {"classes": [{"kind", "count", "example", "range", ...}],
         "total", "truncated", "classified", "without_text",
         "repeated_values", "scope", "scope_warnings", "truncation",
         "classification_basis"}

    `kind` is one of `numeric_label`, `dimension_value`, `arabic_shx`, `other`,
    computed from CONTENT — never from the layer name (rule G1). Only classes
    that actually occur are listed, and each of them carries an example, so no
    bucket is ever silent.

    `total` is the number of rows given; `classified + without_text["count"]`
    always equals it, which is the coverage arithmetic of the whole campaign in
    miniature. An empty input returns this document with empty parts, never
    `None` (rule G8).
    """
    rows = list(rows or [])

    counts: dict[str, int] = {}
    by_layout: dict[str, dict[str, int]] = {}
    examples: dict[str, dict[str, Any]] = {}
    texts: dict[str, list[str]] = {}
    numbers: dict[str, list[float]] = {}
    digits: dict[str, list[int]] = {}
    readings: dict[str, list[dict[str, Any]]] = {}

    # Keyed by (layout, text) and NEVER by text alone. This is the field that
    # could produce "2 plots numbered 2092", and the key is what stops it.
    per_value: dict[tuple[str, str], list[Any]] = {}

    without_text = 0
    without_text_by_type: dict[str, int] = {}
    without_text_example: dict[str, Any] | None = None

    # Two different facts, kept apart. A row with no text and a row holding
    # only spaces are both unclassifiable, but they are not the same finding:
    # the second one DOES carry text, and saying "no text at all" about it is
    # a false sentence. Found by the Phase 4 eval on a drawing where 33 rows
    # were reported as carrying no text while `distinct` independently
    # reported 15 values -- which were 15 different lengths of blank string.
    # Both answers were wrong in opposite directions.
    blank_text = 0
    blank_lengths: set[int] = set()

    for row in rows:
        raw = row.get("text")
        text = "" if raw is None else str(raw)
        if not text.strip():
            without_text += 1
            if text:
                blank_text += 1
                blank_lengths.add(len(text))
            kind_name = str(row.get("type") or "(type not recorded)")
            without_text_by_type[kind_name] = (
                without_text_by_type.get(kind_name, 0) + 1
            )
            if without_text_example is None:
                without_text_example = _annotation_example(row, None)
            continue

        font, bigfont, style_known = _font_of(row)
        kind, reading = classify_text(
            text, font=font, bigfont=bigfont, style_known=style_known
        )
        stripped = text.strip()

        counts[kind] = counts.get(kind, 0) + 1
        layout = _layout_of(row)
        bucket = by_layout.setdefault(kind, {})
        bucket[layout] = bucket.get(layout, 0) + 1
        if kind not in examples:
            examples[kind] = _annotation_example(row, reading)
        texts.setdefault(kind, []).append(stripped)
        if kind in (KIND_NUMERIC, KIND_DIMENSION):
            numbers.setdefault(kind, []).extend(_numbers_in(stripped))
        if kind == KIND_NUMERIC:
            # Digit width is a fact about a label — "four-digit labels running
            # 1990-2110". On `12.00` the same count would be an unlabelled
            # figure inviting the wrong reading, so it is not computed there.
            digits.setdefault(kind, []).append(_digit_count(stripped))
        if kind == KIND_ARABIC and reading is not None:
            readings.setdefault(kind, []).append(
                {
                    "text": stripped,
                    "reading": reading.reading,
                    "method": reading.method,
                }
            )

        per_value.setdefault((layout, stripped), []).append(row)

    classes = [
        _class_document(
            kind,
            counts[kind],
            by_layout.get(kind, {}),
            examples.get(kind),
            texts.get(kind, []),
            numbers.get(kind, []),
            digits.get(kind, []),
            readings.get(kind, []),
        )
        for kind in KINDS
        if counts.get(kind)
    ]

    repeats = [
        (layout, value, members)
        for (layout, value), members in per_value.items()
        if len(members) > 1
    ]
    repeats.sort(key=lambda item: (-len(item[2]), item[0], item[1]))
    kept_repeats = repeats[:MAX_REPEATED_VALUES]

    handles_omitted = 0
    repeated_values: list[dict[str, Any]] = []
    for layout, value, members in kept_repeats:
        handles = [m.get("handle") for m in members]
        omitted = max(0, len(handles) - MAX_HANDLES_PER_VALUE)
        handles_omitted += omitted
        repeated_values.append(
            {
                "text": value,
                "layout": layout,
                "count": len(members),
                "handles": handles[:MAX_HANDLES_PER_VALUE],
                "handles_omitted": omitted,
            }
        )

    repeats_omitted = max(0, len(repeats) - len(kept_repeats))
    samples_omitted = sum(c["samples_omitted"] for c in classes)
    readings_omitted = sum(c.get("readings_omitted", 0) for c in classes)
    scope, warnings = _scope(rows)

    return {
        "classes": classes,
        "total": len(rows),
        "classified": sum(counts.values()),
        "without_text": {
            "count": without_text,
            "by_type": _sorted_counts(without_text_by_type),
            "example": without_text_example,
            "blank_text": blank_text,
            "blank_text_lengths": sorted(blank_lengths)[:MAX_SAMPLES],
            "reason": (
                (
                    "these rows carry nothing this census can classify. "
                    f"{without_text - blank_text} hold no text at all — a "
                    "DIMENSION whose measurement string is generated at draw "
                    f"time, or an empty TEXT — and {blank_text} hold text that "
                    "is ONLY whitespace, which is not the same thing: a tool "
                    "counting distinct values will report those as values, and "
                    "reporting them here as 'no text' would contradict it. "
                    "Neither is content."
                )
                if blank_text
                else (
                    "these rows carry no text at all — a DIMENSION whose "
                    "measurement string is generated at draw time, or an empty "
                    "TEXT."
                )
            )
            + (
                " Counted and shown rather than dropped, so that "
                "classified + without_text always equals total (rule G8)."
            ),
        },
        "repeated_values": repeated_values,
        "repeated_values_basis": REPEATED_BASIS,
        "classification_basis": CLASSIFICATION_BASIS,
        "scope": scope,
        "scope_warnings": warnings,
        "truncated": bool(
            samples_omitted or readings_omitted or repeats_omitted or handles_omitted
        ),
        "truncation": {
            "sample_cap": MAX_SAMPLES,
            "reading_cap": MAX_READINGS,
            "repeated_value_cap": MAX_REPEATED_VALUES,
            "handles_per_value_cap": MAX_HANDLES_PER_VALUE,
            "samples_omitted": samples_omitted,
            "readings_omitted": readings_omitted,
            "repeated_values_omitted": repeats_omitted,
            "handles_omitted": handles_omitted,
            "basis": (
                "counts are never capped — only the illustrative lists are, "
                "and every one of them says how much it left out (rule G7)."
            ),
        },
    }
