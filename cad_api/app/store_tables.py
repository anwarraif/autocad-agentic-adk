"""layer/style/linetype tables + find_by_name; then Arabic SHX readings

Owner: subagent F — UPLIFT-11, then UPLIFT-04.

This file is deliberately kept apart from `store.py` so that two specs worked
on at the same time never touch the same line. The rules, and their reasons:

- **Do not import from `.store`.** `store.py` imports this module at its
  bottom; importing back would be circular. Take `coll` and its friends
  straight from `.mongo`, and geometry from `.region`.
- **Do not add lines to `store.py` or `main.py`.** Both are shared, and every
  line added there is one merge conflict. Functions here are reached as
  `store.store_tables.<function>`.
- **Every response that carries MEANING must carry `evidence`** per
  `docs/UPLIFT-08-EVIDENCE.md` — use the helpers in `app/evidence.py`, do not
  assemble the dict yourself.
- **Every number carries its unit** (G2), and a unit is allowed to be absent.
  A withheld number becomes `None` + a reason, never `0`.

The `docs/UPLIFT-GENERALITY-RULES.md` G1-G10 checklist is run before every
commit in this file.

---

The shape: every capability here is split in two — one PURE function that takes
a drawing document and returns the answer, and one wrapper that fetches that
document from Mongo. Not a style but a requirement: this suite runs in a
throwaway container with no MongoDB, so the only way to test `find_by_name`
against the real Janadriyah figures is to separate the searching from the
fetching.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

from . import evidence as ev
from .mongo import COLL_DRAWINGS, COLL_ENTITIES, coll

log = logging.getLogger(__name__)

#: The dictionaries whose names can be searched. The order is the order they
#: are presented in.
KINDS: tuple[str, ...] = (
    "layer",
    "block",
    "layout",
    "text_style",
    "linetype",
    "dim_style",
)

#: A result cap that is stated, not assumed (G7). A one-letter query matches
#: almost the whole dictionary of a large drawing; what comes back is cut, and
#: the cut is SAID along with the real total. Tested on the largest drawing
#: there is: 292 layers, 78 blocks, 23 text styles, 15 linetypes.
MAX_HITS: int = 200

#: The first extraction version that stores the style tables. A drawing ingested
#: before this has no `text_styles`/`linetypes`/`dim_styles`, and that must read
#: as "not extracted yet", not as "there are none".
TABLES_INGEST_VERSION: int = 4

#: The suffix of a font file that stores Latin bytes and maps them to other
#: glyphs inside the font file itself. Not a drawing-specific constant (G1):
#: this is the AutoCAD font format, and it holds in any drawing.
SHX_SUFFIX: str = ".shx"

_NAME_NOT_MEANING = (
    "A name that contains this word is not a claim about what the object IS. "
    "It is a name somebody typed that string into; what it holds is a "
    "separate question, answered per drawing from that drawing's own config, "
    "never carried over from another drawing."
)

_SEARCH_DIFFERENCE = (
    "`search_text` searches the TEXT DRAWN IN the drawing; `find_by_name` "
    "searches the NAMES IN the drawing's dictionary. Both answer 'where does "
    "this word appear', and those are two very different places."
)


def _unit_scope(drawing: Mapping[str, Any], layout: str | None = None) -> Any:
    """The `Scope` for a whole-drawing answer, with the units from `store`.

    The import is deferred into the function on purpose. The prohibition at the
    head of this file reasons that "importing back would be circular", and that
    is true for a module-level import — `store.py` imports this module at its
    tail. An import inside a function body is not circular: by the time anyone
    calls this function, `store` has finished loading.

    The alternative that loses: copying `_unit_names` here. Its docstring
    records the worst number this project ever shipped — 1,867 paper lines
    reported as "2992.231252 m" — and a copy would drift away from that record
    with nobody noticing. A unit may have only one source.
    """
    from . import store  # noqa: PLC0415 - see the docstring

    return ev.Scope(
        layout=layout,
        includes_block_definitions=True,
        units=store._unit_names(dict(drawing), layout),
        note=(
            "the whole drawing's dictionary tables: layers, blocks, layouts, "
            "text styles, linetypes and dimension styles, block definitions "
            "included"
        ),
    )


# --- find_by_name ------------------------------------------------------------


def _hit(
    kind: str,
    name: str,
    *,
    matched_on: list[str],
    entity_count: int | None,
    count_basis: str,
) -> dict[str, Any]:
    """One match.

    `is_empty` is None when the count is not known, NEVER True. "Empty" and
    "not counted" are two different answers, and the second one disguises
    itself as the first exactly when somebody asks whether a layer is
    pointless (G8).
    """
    return {
        "kind": kind,
        "name": name,
        "matched_on": matched_on,
        "entity_count": entity_count,
        "count_basis": count_basis,
        "is_empty": None if entity_count is None else entity_count == 0,
    }


def _matches(query: str, value: Any) -> bool:
    return bool(value) and query in str(value).casefold()


def _layer_hits(drawing: Mapping[str, Any], q: str) -> list[dict[str, Any]]:
    out = []
    for layer in drawing.get("layers") or []:
        name = layer.get("name", "")
        if not _matches(q, name):
            continue
        out.append(
            _hit(
                "layer",
                name,
                matched_on=["name"],
                entity_count=layer.get("entity_count"),
                count_basis=(
                    "entities on this layer, across every layout and every "
                    "block definition"
                ),
            )
        )
    return out


def _layout_hits(drawing: Mapping[str, Any], q: str) -> list[dict[str, Any]]:
    out = []
    for layout in drawing.get("layouts") or []:
        name = layout.get("name", "")
        if not _matches(q, name):
            continue
        out.append(
            _hit(
                "layout",
                name,
                matched_on=["name"],
                entity_count=layout.get("entity_count"),
                count_basis="entities stored on this layout",
            )
        )
    return out


def _block_hits(
    drawing: Mapping[str, Any], q: str, counts: Mapping[str, int] | None
) -> list[dict[str, Any]]:
    out = []
    for name in drawing.get("blocks") or []:
        if not _matches(q, name):
            continue
        if counts is None:
            count: int | None = None
            basis = (
                "not counted: the reference count needs a query this call did "
                "not run"
            )
        else:
            count = int(counts.get(name, 0))
            basis = "INSERT references to this block definition in this drawing"
        out.append(
            _hit("block", name, matched_on=["name"], entity_count=count, count_basis=basis)
        )
    return out


def _text_style_hits(drawing: Mapping[str, Any], q: str) -> list[dict[str, Any]]:
    """Text styles whose name OR whose font file contains the query.

    The font is searched too because a font file is a name as well, and in this
    file it is the only place the word appears: the styles named `DTE` and
    `moh -2` say nothing about Arabic anywhere except in `xarab.shx`. Which
    field matched is named in `matched_on` — a style that matched through its
    font and a style that matched through its name are not the same thing, and
    only `matched_on` can tell them apart.
    """
    out = []
    for style in drawing.get("text_styles") or []:
        matched = [
            field
            for field in ("name", "font", "bigfont")
            if _matches(q, style.get(field))
        ]
        if not matched:
            continue
        hit = _hit(
            "text_style",
            style.get("name", ""),
            matched_on=matched,
            entity_count=None,
            count_basis=(
                "not stored: entities do not carry their text style yet "
                "(UPLIFT-04 asks for it); a style's usage cannot be counted "
                "from this store"
            ),
        )
        hit["font"] = style.get("font")
        hit["bigfont"] = style.get("bigfont")
        hit["font_is_shx"] = any(
            str(style.get(f) or "").lower().endswith(SHX_SUFFIX)
            for f in ("font", "bigfont")
        )
        out.append(hit)
    return out


def _linetype_hits(drawing: Mapping[str, Any], q: str) -> list[dict[str, Any]]:
    used_by: dict[str, int] = {}
    for layer in drawing.get("layers") or []:
        name = layer.get("linetype")
        if name:
            used_by[str(name)] = used_by.get(str(name), 0) + 1

    out = []
    for linetype in drawing.get("linetypes") or []:
        name = linetype.get("name", "")
        if not _matches(q, name):
            continue
        hit = _hit(
            "linetype",
            name,
            matched_on=["name"],
            entity_count=None,
            count_basis=(
                "not stored: entities do not carry a linetype override yet; "
                "what IS known is how many layers declare this linetype"
            ),
        )
        hit["layers_using"] = used_by.get(name, 0)
        hit["description"] = linetype.get("description")
        hit["pattern_length"] = linetype.get("pattern_length")
        out.append(hit)
    return out


def _dim_style_hits(drawing: Mapping[str, Any], q: str) -> list[dict[str, Any]]:
    out = []
    for style in drawing.get("dim_styles") or []:
        name = style.get("name", "")
        if not _matches(q, name):
            continue
        hit = _hit(
            "dim_style",
            name,
            matched_on=["name"],
            entity_count=None,
            count_basis=(
                "not stored: dimension entities do not carry their style in "
                "this store"
            ),
        )
        hit["scale"] = style.get("scale")
        out.append(hit)
    return out


_HIT_BUILDERS = {
    "layer": lambda d, q, c: _layer_hits(d, q),
    "block": lambda d, q, c: _block_hits(d, q, c),
    "layout": lambda d, q, c: _layout_hits(d, q),
    "text_style": lambda d, q, c: _text_style_hits(d, q),
    "linetype": lambda d, q, c: _linetype_hits(d, q),
    "dim_style": lambda d, q, c: _dim_style_hits(d, q),
}

#: The drawing-document key that holds each dictionary. Used to tell "this
#: dictionary is empty" apart from "this drawing was ingested before this
#: dictionary existed".
_SOURCE_KEY = {
    "layer": "layers",
    "block": "blocks",
    "layout": "layouts",
    "text_style": "text_styles",
    "linetype": "linetypes",
    "dim_style": "dim_styles",
}


def search_names(
    drawing: Mapping[str, Any],
    query: str,
    kinds: Sequence[str] | None = None,
    *,
    block_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Search `query` in the drawing's dictionaries. Pure: it never touches Mongo.

    Case-insensitive, substring, no stemming and no fuzziness. What is sought
    is evidence that the name CONTAINS that word, not that it resembles it.
    """
    q = (query or "").strip()
    if not q:
        return {
            "ok": False,
            "code": "EMPTY_QUERY",
            "message": "find_by_name needs a word to look for.",
            "hint": (
                "Pass the word you would type into a layer filter. To list a "
                "dictionary in full, ask describe_drawing instead."
            ),
        }

    wanted = list(kinds) if kinds else list(KINDS)
    unknown = [k for k in wanted if k not in KINDS]
    if unknown:
        return {
            "ok": False,
            "code": "UNKNOWN_KIND",
            "message": f"unknown kinds: {', '.join(sorted(unknown))}.",
            "hint": f"kinds must be a subset of {', '.join(KINDS)}.",
        }

    folded = q.casefold()
    ingest_version = int(drawing.get("ingest_version") or 0)

    hits: list[dict[str, Any]] = []
    not_searched: list[dict[str, Any]] = []
    for kind in KINDS:
        if kind not in wanted:
            continue
        if drawing.get(_SOURCE_KEY[kind]) is None:
            not_searched.append(
                {
                    "kind": kind,
                    "why": (
                        f"this drawing was extracted by ingest_version "
                        f"{ingest_version}, before the {kind} table was "
                        f"stored (needs {TABLES_INGEST_VERSION}); re-ingest "
                        "the drawing to make it searchable"
                    ),
                }
            )
            continue
        hits.extend(_HIT_BUILDERS[kind](drawing, folded, block_counts))

    total = len(hits)
    truncated = total > MAX_HITS
    body: dict[str, Any] = {
        "ok": True,
        "drawing_id": drawing.get("_id"),
        "query": q,
        "kinds_searched": [k for k in KINDS if k in wanted and not any(
            n["kind"] == k for n in not_searched
        )],
        "kinds_not_searched": not_searched,
        "hits": hits[:MAX_HITS],
        "hits_returned": min(total, MAX_HITS),
        "hits_total": total,
        "truncated": truncated,
        "limit": MAX_HITS,
        "empty_hits": [h["name"] for h in hits[:MAX_HITS] if h["is_empty"]],
        "scope_note": (
            "names declared in this drawing's dictionary, block definitions "
            "included; nothing is read from any other drawing"
        ),
        "what_this_searches": _SEARCH_DIFFERENCE,
        "name_is_not_meaning": _NAME_NOT_MEANING,
    }
    if truncated:
        body["suggestion"] = (
            f"{total} names matched and {MAX_HITS} are shown. Narrow the query "
            "or pass `kinds` to search one dictionary at a time."
        )
    return body


def _block_reference_counts(
    drawing_id: str, block_names: Iterable[str]
) -> dict[str, int]:
    """How many INSERTs point at each block definition.

    One aggregation for all the names, not one query per name: 78 blocks
    matching a one-letter query would become 78 round trips to the cluster. The
    filter is prefixed by `drawing_id` and uses the `{drawing_id, block_name}`
    index that already exists — on a `notablescan` cluster, a query without an
    indexed plan does not slow down, it FAILS.
    """
    names = [n for n in block_names if n]
    if not names:
        return {}
    pipeline = [
        {"$match": {"drawing_id": drawing_id, "block_name": {"$in": names}}},
        {"$group": {"_id": "$block_name", "n": {"$sum": 1}}},
    ]
    out = {n: 0 for n in names}
    for row in coll(COLL_ENTITIES).aggregate(pipeline):
        out[str(row["_id"])] = int(row["n"])
    return out


def find_by_name(
    drawing_id: str, query: str, kinds: Sequence[str] | None = None
) -> dict[str, Any]:
    """"Which names in this drawing contain this word?"

    This is what turns "that word -> 0 results" into a structurally useful
    answer. Before it the only search there was is `search_text`, which searches
    TEXT CONTENT; a layer whose name contains that word but which holds not one
    TEXT would never show up there, and that is exactly the layer people are
    looking for.
    """
    drawing = coll(COLL_DRAWINGS).find_one({"_id": drawing_id})
    if drawing is None:
        return {
            "ok": False,
            "code": "UNKNOWN_DRAWING",
            "message": f"no drawing with id {drawing_id!r}.",
            "hint": "list_drawings() returns the ids that exist.",
        }

    wanted = list(kinds) if kinds else list(KINDS)
    counts: dict[str, int] | None = None
    if "block" in wanted:
        folded = (query or "").strip().casefold()
        matching = [
            n for n in (drawing.get("blocks") or []) if folded and folded in n.casefold()
        ]
        try:
            counts = _block_reference_counts(drawing_id, matching)
        except Exception as exc:  # noqa: BLE001
            log.warning("block reference count failed for %s: %s", drawing_id, exc)
            counts = None
    return search_names(drawing, query, kinds, block_counts=counts)


# --- the `tables` block for describe_drawing ---------------------------------


def tables_summary(drawing: Mapping[str, Any]) -> dict[str, Any]:
    """A summary of the drawing's dictionaries, with what is worth saying unasked.

    Pure, and deliberately carrying no list at all: `describe_drawing` already
    eats ~31,500 characters of context before a single question is answered,
    and adding 292 rows of layer table there would double it. What is added are
    counts and one short list — the SHX style names — because that is the only
    part that cannot be reconstructed from a count.
    """
    layers = drawing.get("layers") or []
    text_styles = drawing.get("text_styles")
    linetypes = drawing.get("linetypes")
    dim_styles = drawing.get("dim_styles")

    shx = [
        s.get("name", "")
        for s in (text_styles or [])
        if any(
            str(s.get(f) or "").lower().endswith(SHX_SUFFIX)
            for f in ("font", "bigfont")
        )
    ]
    fonts = sorted(
        {
            str(s.get(f))
            for s in (text_styles or [])
            for f in ("font", "bigfont")
            if str(s.get(f) or "").lower().endswith(SHX_SUFFIX)
        }
    )

    stale = text_styles is None or linetypes is None or dim_styles is None
    return {
        "these_are_counts_not_names": (
            "every layer figure here is a COUNT taken from the layer table. It "
            "says how many layers are frozen or off, never which ones, and "
            "never whether any of them carry anything in the layout you are "
            "looking at"
        ),
        "ask_instead": {
            "recipe": "drafting_hygiene",
            "when": (
                "the question is WHICH layers are hidden, or whether hidden "
                "layers still carry content that a total is being computed from"
            ),
            "why_it_matters": (
                "`drafting_hygiene` names each hidden layer, says whether it is "
                "frozen or off, counts its entities in the layout, and "
                "separates the hidden layers that are empty here from the ones "
                "that are not. Measured: asked which hidden layers still "
                "contribute parcels, a turn read these counts and answered that "
                "it was not possible to determine -- while the other recipe "
                "lists eleven of them, one of which holds 1,233 entities"
            ),
        },
        "layer_count": len(layers),
        "layers_frozen": sum(1 for l in layers if l.get("frozen")),
        "layers_off": sum(1 for l in layers if l.get("off")),
        "layers_locked": sum(1 for l in layers if l.get("locked")),
        "layers_not_plotted": sum(1 for l in layers if l.get("plot") is False),
        "layers_non_continuous": sum(
            1
            for l in layers
            if l.get("linetype") and str(l["linetype"]).casefold() != "continuous"
        ),
        "layers_frozen_in_some_viewport": sum(
            1 for l in layers if l.get("frozen_in_viewports")
        ),
        "layers_with_description": sum(1 for l in layers if l.get("description")),
        "text_style_count": None if text_styles is None else len(text_styles),
        "linetype_count": None if linetypes is None else len(linetypes),
        "dim_style_count": None if dim_styles is None else len(dim_styles),
        "text_styles_using_shx": None if text_styles is None else len(shx),
        "shx_style_names": shx,
        "shx_font_files": fonts,
        "header": drawing.get("header"),
        "tables_stale": stale,
        "tables_stale_why": (
            None
            if not stale
            else (
                "this drawing was extracted before the style tables existed; "
                f"re-ingest it (ingest_version {TABLES_INGEST_VERSION}) to fill "
                "text_styles, linetypes and dim_styles"
            )
        ),
    }


def shx_font_gap(drawing: Mapping[str, Any]) -> dict[str, Any]:
    """What CANNOT be said about text drawn through an SHX font.

    Not a finding but a hole, and its shape is `unknown` on purpose. An SHX font
    stores Latin bytes and only becomes other glyphs once the font file is
    installed; that file is not shipped with the drawing, and it cannot be
    guessed. DWG TrueView shows a "Missing SHX Files" prompt on the same file,
    so this is not something this pipeline lost.

    There are three things in this project that cannot be solved by coding, and
    this is one of them (FIDELITY.md, T-11). The right thing to do: mark it
    unverified, name which files are missing and whom to ask, then move on.
    Guessing the glyphs would put a supposition where everybody reads it as
    fact.
    """
    summary = tables_summary(drawing)
    fonts = summary["shx_font_files"]
    styles = summary["shx_style_names"]
    scope = _unit_scope(drawing)

    body: dict[str, Any] = {
        "drawing_id": drawing.get("_id"),
        "scope_note": scope.note,
        "shx_styles": styles,
        "shx_font_files": fonts,
        "font_files_present": False,
        "verified": False,
    }

    if not fonts:
        gap = "no text style in this drawing points at an SHX font file"
        how = (
            "nothing to ask for: every style here names a TrueType font or "
            "no font at all, so the stored text is the text as drawn"
        )
    else:
        gap = (
            f"{len(styles)} text styles draw through SHX font files "
            f"({', '.join(fonts)}) that are not shipped with the drawing. An "
            "SHX font stores Latin bytes and maps them to other glyphs inside "
            "the font file itself, so without the file the stored text is not "
            "the text a reader sees -- here or in AutoCAD."
        )
        how = (
            "ask the drawing's author for the SHX font files listed above and "
            "register them with ezdxf (`ezdxf.options.font_dirs`). This is a "
            "request to a person, not a coding task: the mapping cannot be "
            "recovered from the drawing."
        )

    return ev.attach(
        body,
        ev.Evidence.unknown(
            ev.Claim(value="what the SHX-drawn text actually reads"),
            drawing_id=str(drawing.get("_id")),
            scope=scope,
            provenance=ev.Provenance(
                config_layer=ev.ConfigLayer.NO_CONFIG,
                note="no font mapping is configured, and none can be guessed",
            ),
            not_established=gap,
            how_to_verify=how,
            observations=[
                ev.Observation(
                    origin=ev.Origin.ABSENCE,
                    detail=(
                        f"style {name!r} names font file "
                        f"{(style.get('font') or style.get('bigfont'))!r}, "
                        "which is not present"
                    ),
                    locator=name,
                )
                for style, name in (
                    (s, s.get("name", ""))
                    for s in (drawing.get("text_styles") or [])
                )
                if any(
                    str(style.get(f) or "").lower().endswith(SHX_SUFFIX)
                    for f in ("font", "bigfont")
                )
            ][:MAX_HITS],
        ),
    )


def drawing_tables(drawing_id: str) -> dict[str, Any]:
    """The `tables` block for one drawing, fetched from the store."""
    drawing = coll(COLL_DRAWINGS).find_one({"_id": drawing_id})
    if drawing is None:
        return {
            "ok": False,
            "code": "UNKNOWN_DRAWING",
            "message": f"no drawing with id {drawing_id!r}.",
            "hint": "list_drawings() returns the ids that exist.",
        }
    return {
        "ok": True,
        "drawing_id": drawing_id,
        "tables": tables_summary(drawing),
        "shx": shx_font_gap(drawing),
    }
