"""cad-mcp — MCP tools over cad-api.

The docstrings in this file are the interface: they are what the model reads
to decide which tool to call and with what. They are written for a reader who
has never seen the drawing, and each says *when this tool is the right choice*,
not only what it does. A wrong tool call is far more often a vague docstring
than a weak model.

Design rules, from the `cad-agent-tools` skill:

- **Three separate query paths, never one generic `search()`.** Merged into
  one, a model defaults to text search, which for CAD is usually the wrong and
  most expensive answer. Structural filtering is right almost every time.
- **Every list response carries `total_matches` and `truncated`.** Without
  them a model cannot tell "there are 12 results" from "there are 40,000 and
  you are seeing 100", and that difference changes the answer a user gets.
- **Results above a threshold return a breakdown instead of rows** — a way
  forward rather than a wall.
- **Errors carry a `hint`.** A model told only "not found" retries the same
  wrong thing.
- **Nothing is trimmed in silence.** Every response is measured before it is
  returned, and one that is over budget names and counts the rows it dropped
  in `response_budget`. A tool that quietly loses the end of its own answer is
  worse than one that refuses.
- **Absent is never returned as empty.** A block that was not computed says
  the words "not computed". Silence about a drawing gets read as a statement
  about the drawing, and that mistake — "there are no roads" said about a
  drawing holding 84 km of them — is the one this file works hardest to
  prevent.

This process holds no CAD logic of its own: it calls cad-api over HTTP. One
engine, one set of numbers, whether the caller is the agent or the browser.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import httpx
from fastmcp import FastMCP

logging.basicConfig(
    level=os.environ.get("CAD_LOG_LEVEL", "INFO"),
    format='{"severity":"%(levelname)s","name":"%(name)s","message":"%(message)s"}',
)
log = logging.getLogger("cad_mcp")

CAD_API_URL = os.environ.get("CAD_API_URL", "http://cad-api:8000")
TIMEOUT = float(os.environ.get("CAD_MCP_TIMEOUT", "60"))

#: Hard server-side cap. A model asking for 5,000 rows gets 100.
MAX_ROWS = 100

#: Handles `describe_selection` will accept. Matches MAX_SELECTION_HANDLES
#: in cad-api, so the tool never forwards a list the API would refuse -- and,
#: more importantly, never silently describes a subset of a selection as
#: though it were the whole thing.
MAX_SELECTION_HANDLES = 5_000

#: Characters one tool response may occupy before it is trimmed. This is a
#: NET, not a diet. Measured against the recorded baseline (`dossier_baseline.
#: json`, 25 August 2026): `/land-use` on the reference drawing returns 59,972
#: bytes and `/drawings/{id}` 107,548 before this file trims it down. A cap set
#: anywhere near those figures would start shortening answers that ship correct
#: today, which is the opposite of the job.
#:
#: It exists because the Dossier blocks are PER LAYER and therefore grow with
#: the drawing -- 176 (layer x layout) buckets on the reference drawing, more
#: on a bigger one. The economy is done upstream by top-N; this only catches
#: what escapes it, and it catches it out loud.
MAX_RESPONSE_CHARS = int(os.environ.get("CAD_MCP_MAX_RESPONSE_CHARS", "90000"))

#: A trimmed list is never cut below this. A block trimmed to nothing reads as
#: a block that found nothing, and keeping those two apart is what this whole
#: file is for.
MIN_KEPT_ROWS = 3

#: A list smaller than this is never trimmed at all. Halving a ten-name sample
#: to recover 150 characters cannot be what rescues a 90,000-character
#: response, and a trim record listing eight blocks that saved nothing buries
#: the one block that did.
MIN_TRIMMABLE_CHARS = 1_000

mcp = FastMCP("cad-mcp")

_client = httpx.Client(base_url=CAD_API_URL, timeout=TIMEOUT)


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET from cad-api, translating transport failures into tool errors."""
    try:
        response = _client.get(path, params={k: v for k, v in (params or {}).items() if v is not None})
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": "CAD_API_UNREACHABLE",
            "message": f"Could not reach cad-api: {type(exc).__name__}.",
            "hint": "The drawing service may be down. Ask the user to check it.",
        }
    if response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = {}
        return {
            "ok": False,
            "error": body.get("error", f"HTTP_{response.status_code}"),
            "message": body.get("message", response.text[:200]),
            "hint": body.get("hint", ""),
        }
    return response.json()


def _post(
    path: str, params: dict[str, Any] | None, payload: Any
) -> dict[str, Any]:
    """POST to cad-api, translating transport failures into tool errors.

    The same contract as `_get`, and it exists for the same reason: an
    exception raised inside this process reaches the model as a stack trace,
    and a model cannot act on one. Every failure leaves here as a dict with
    `error`, `message` and `hint`.

    Scope goes in the query string and the recipe's parameters go in the body,
    because cad-api reads them from those two places and they are different
    kinds of thing: `layout` says which drawing this is about, `params` says
    what the calculation was asked for.

    This function was called by `run_analysis` before it existed -- the tool
    raised `NameError` on every invocation while looking, from the catalogue,
    exactly like a tool that works.
    """
    try:
        response = _client.post(
            path,
            params={k: v for k, v in (params or {}).items() if v is not None},
            json=payload,
        )
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": "CAD_API_UNREACHABLE",
            "message": f"Could not reach cad-api: {type(exc).__name__}.",
            "hint": "The drawing service may be down. Ask the user to check it.",
        }
    if response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = {}
        return {
            "ok": False,
            "error": body.get("error", f"HTTP_{response.status_code}"),
            "message": body.get("message", response.text[:200]),
            "hint": body.get("hint", ""),
        }
    return response.json()


@mcp.tool
def list_drawings() -> dict[str, Any]:
    """List every CAD drawing available, with its id and size.

    Call this first when you do not already know a `drawing_id`. Every other
    tool needs one, and ids are content hashes that cannot be guessed from a
    filename.

    Returns a list of drawings, each with `drawing_id`, `filename`,
    `entity_count`, `units` and the layouts it contains.
    """
    data = _get("/drawings")
    if data.get("ok") is False:
        return data

    def summarise(d: dict[str, Any]) -> dict[str, Any]:
        populated = [l for l in d.get("layouts", []) if l.get("entity_count", 0) > 0]
        # Real layouts and block definitions were listed together, so
        # Janadriyah appeared to have 81 layouts when it has 3. A block
        # definition is a component, not a view, and a caller that cannot tell
        # them apart cannot scope a question correctly.
        real = [l["name"] for l in populated if not l.get("is_block")]
        blocks = sum(1 for l in populated if l.get("is_block"))
        summary = {
            "drawing_id": d["_id"],
            "filename": d.get("original_filename"),
            "entity_count": d.get("entity_count", 0),
            "units": d.get("units_name"),
            "layouts": real,
        }
        if blocks:
            summary["block_definitions_with_geometry"] = blocks
        return summary

    return {
        "total": data.get("total", 0),
        "note": (
            "`layouts` are the drawing's real views. Scope every question to "
            "one of them: entity counts differ enormously between model space, "
            "a paper sheet, and the whole file."
        ),
        "drawings": [summarise(d) for d in data.get("drawings", [])],
    }


@mcp.tool
def describe_drawing(drawing_id: str) -> dict[str, Any]:
    """Summarise one drawing: layers, blocks, layouts, units and extents.

    This is the correct entry point for any question about a drawing you have
    not looked at yet. Read this before running detail queries — it tells you
    which layer names and block names actually exist, so you can filter with
    real values instead of guessing.

    **Read `dossier` first.** It is the drawing's own account of itself,
    computed once over every entity in the file, and it answers "what is in
    this file" without a scan and without a guess: the ROLES present, the
    coverage verdict — whether every entity in the drawing landed in a bucket,
    46,754 of 46,754 on the reference drawing — and the largest layers with
    their role and their native measure. A question that used to cost six
    exploratory calls is answered by reading this block.

    **Roles have NATIVE MEASURES, and the role decides which question a layer
    can even be asked.** A `region` layer answers in a count and an AREA. A
    `network` layer answers in LENGTH — it is open paths, it has no parcels to
    count, and asking it for a parcel count is asking it the wrong question;
    the zero that comes back is about the question, not about the drawing.
    `points` answers in a count per block name, `annotation` in content
    classes, and `mixed` answers in a count and says that it could not decide.
    Read the role, then choose the tool: `measure(measure="length")` for a
    network, `measure(measure="area")` or `land_use_summary` for a region,
    `query_entities(block_name=...)` for points.

    **`dossier.status == "not_computed"` does not mean the drawing is empty.**
    It means nobody has built the Dossier for this drawing yet. Every other
    field in this response is still true, and `distinct_values` and `measure`
    still answer one layer at a time. Say "not computed"; never say "nothing
    there".

    The layer list inside it is capped and says where it stopped. The Dossier
    holds a bucket per (layer × layout) — 176 of them on the reference drawing
    — and pasting all of them into an overview would cost more context than
    the overview is worth.

    Args:
        drawing_id: id from `list_drawings()`.

    Returns format and unit information, computed extents, the layers with
    their entity counts, the block definitions, the layouts, any unresolved
    external references, and `dossier`: the roles present, the coverage
    verdict, and the top layers by entity count with role and native measure.
    """
    data = _get(f"/drawings/{drawing_id}")
    if data.get("ok") is False:
        return data

    layers = sorted(
        data.get("layers", []), key=lambda l: l.get("entity_count", 0), reverse=True
    )
    populated = [l for l in data.get("layouts", []) if l.get("entity_count", 0) > 0]
    real_layouts = [l for l in populated if not l.get("is_block")]
    block_layouts = [l for l in populated if l.get("is_block")]
    counts_by_type = data.get("counts_by_type", {}) or {}

    # Size matters as much as correctness here. This call and list_drawings are
    # the two a model makes before it can answer anything, and together they
    # used to cost ~31,500 characters of context -- more than the questions and
    # answers combined -- because they carried 40 block names and every entity
    # type in the file. Everything trimmed below is one query away.
    out: dict[str, Any] = {
        "drawing_id": data["_id"],
        "filename": data.get("original_filename"),
        "format": f"{data.get('dxf_version')} ({data.get('acad_release')})",
        "units": data.get("units_name"),
        "units_declared_in_file": data.get("units_code", 0) != 0,
        "units_scope": (
            "MODEL space only. $INSUNITS is a drawing-header field and DXF "
            "carries no per-layout unit, so a measurement on a paper layout "
            "has no unit at all -- `measure` returns null there rather than "
            "stamping this one on page geometry."
        ),
        "entity_count": data.get("entity_count", 0),
        "entity_count_scope": (
            "every layout AND every block definition. A per-layout count is "
            "query_entities(layout=...) with limit=1."
        ),
        # The Dossier: the drawing's account of itself, computed once over
        # every entity rather than assembled out of whatever the current
        # question happened to ask for.
        #
        # It is projected HERE, and it has to be. This function builds its
        # response key by key, so anything cad-api returns that this dict does
        # not name never reaches the agent at all -- the block would exist in
        # Mongo, exist in the HTTP response, and be invisible where it counts.
        "dossier": _dossier_block(data.get("dossier")),
        "extents": data.get("extents"),
        "extents_scope": (
            "model space only, computed from geometry rather than read from "
            "the file header. A paper sheet has its own, much smaller, extent."
        ),
        "layouts": [
            {"name": l["name"], "entity_count": l.get("entity_count", 0)}
            for l in real_layouts
        ],
        "block_definitions_with_geometry": len(block_layouts),
        "layer_count": len(layers),
        "layer_count_scope": (
            "layers DECLARED in the file, which is not the same as layers "
            "holding anything: most drawings carry a long tail of empty ones "
            "inherited from a template. For layers actually in use, call "
            "distinct_values(field='layer', layout=...)."
        ),
        # Long tail of near-empty layers is noise for a summary; the full list
        # is one distinct_values(field="layer") call away.
        "layers_top_20": [
            {"name": l["name"], "entity_count": l.get("entity_count", 0)}
            for l in layers[:20]
        ],
        # These counts are drawing-wide, and saying so is not pedantry. Asked
        # for "top 20 layers by entity count" on model space, the agent quoted
        # this list and labelled it "in the Model layout": DIM came back as
        # 19,477 when model space holds 9,738. The same question answered via
        # distinct_values(layout="Model") gave the right figure, so the two
        # routes to one question disagreed by a factor of two. `entity_count`
        # and `extents` beside this already carry a scope note; this list did
        # not, and that omission is the whole defect.
        "layers_top_20_scope": (
            "counted across EVERY layout and every block definition, not the "
            "layout on screen. For one layout use "
            "distinct_values(field='layer', layout=...)."
        ),
        "block_count": len(data.get("blocks", [])),
        "block_count_scope": (
            "block DEFINITIONS in the file. Being defined is not being "
            "placed; several of these are typically never inserted."
        ),
        # Every name, not the alphabetical first fifteen.
        #
        # The docstring above promises this call tells you which block names
        # exist "so you can filter with real values instead of guessing". A
        # truncated alphabetical slice breaks that promise silently: on the
        # reference drawing the four tree blocks sit at positions 47-52 of 78,
        # so "how many trees are there" could not be answered from the metadata
        # the agent was told to trust, and it answered "there are none".
        #
        # The whole list is 78 names and about 1.3 kB. Trimming it saved
        # nothing worth the wrong answer.
        "blocks_all": sorted(data.get("blocks", [])),
        "blocks_all_order": (
            "alphabetical, and complete - every block defined in the drawing. "
            "Being defined is not being placed: use "
            "query_entities(block_name=..., layout=...) to see where a block is "
            "actually inserted."
        ),
        "counts_by_type_top_15": dict(
            sorted(counts_by_type.items(), key=lambda kv: -kv[1])[:15]
        ),
        "counts_by_type_top_15_scope": (
            "drawing-wide, as layers_top_20_scope. For one layout use "
            "query_entities(type=..., layout=..., limit=1) and read "
            "total_matches."
        ),
        # The path, not the class name. This projected `block_name`, which for
        # an image or PDF attachment is the DXF class -- so six distinct missing
        # files arrived as ["IMAGEDEF", "IMAGEDEF", "IMAGEDEF", "IMAGEDEF",
        # "PDFDEFINITION", "PDFDEFINITION"], reading as two xrefs with silly
        # names rather than six files the recipient will be prompted for.
        "unresolved_xrefs": [
            {"path": x.get("path"), "kind": x.get("kind")}
            for x in data.get("xrefs", [])
            if not x.get("resolved")
        ],
        "comment_count": sum(data.get("comment_counts", {}).values()),
        "comment_count_scope": "the whole drawing, every layout.",
        "warnings": data.get("warnings", []),
        "matching_rules": (
            "layer and layout names are matched EXACTLY, case included; `type` "
            "is upper-cased for you. Take names from this response rather than "
            "typing them."
        ),
    }
    # The Dossier grows with the drawing, so the one block in here that is not
    # already capped by a constant above gets measured before it leaves.
    return _budget(out, trim_inside=("dossier",))


@mcp.tool
def query_entities(
    drawing_id: str,
    layer: str | None = None,
    type: str | None = None,
    block_name: str | None = None,
    layout: str | None = None,
    text_contains: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Find entities by structure: layer, DXF type, block name or layout.

    This is the cheapest and most often correct way to answer a question about
    a drawing — prefer it over `search_text`. Use it for "how many doors",
    "what is on the ELECTRICAL layer", "list the block references".

    **The viewer marks every object this call returns.** That is the mechanism
    behind "show me where the X are", and it has two consequences worth
    holding on to.

    First, when someone asks to SEE or LOCATE a set, raise `limit` until it
    covers `total_matches` -- up to a few hundred. The default of 50 is for
    counting questions. Asked to show 133 plots and called with the default,
    this marks 50 of them and the drawing then tells the reader something
    untrue about its own contents.

    Second, what gets marked is `total_matches`, **not** `returned`. The rows
    are capped so this response stays readable; the marks are not. So never
    tell the reader that "the first 100" or "100 of them" are highlighted --
    you are reading your own page size and reporting it as the picture. Say
    how many there ARE and stop.

    Third, you do NOT need to list handles for objects to be highlighted.
    Quote a handful when they are useful to click -- roughly ten -- and let the
    marks do the rest. Fifty handles in a sentence is not an answer anyone
    reads.

    **Say which layout you mean.** "How many entities" has four different
    correct answers on the same file — model space, one paper sheet, all
    layouts, or all layouts plus every block definition — and they differ by a
    factor of eight on Janadriyah. Omitting `layout` gives you the last of
    those, which is rarely the question.

    **Zero matches is not the same as "there are none".** If a filter matches
    nothing, this tool returns a `why_empty` explanation: whether the layer
    exists at all, whether it exists with different capitals, which layouts it
    lives in, and which DXF types are actually present. Read it before
    reporting an absence — a real case: "no polylines on layer 'fram'" was
    literally true and completely misleading, because that layer holds 1,100
    LWPOLYLINE and zero POLYLINE, in a layout the question had not asked for.

    Args:
        drawing_id: id from `list_drawings()`.
        layer: exact layer name, **case-sensitive**. Take names from
            `describe_drawing` rather than typing them.
        type: DXF type, e.g. "INSERT", "LINE", "MTEXT", "TEXT". Upper-cased
            for you, unlike `layer` and `layout`. "INSERT" means a block
            reference — that is what a door or a window usually is. Note that
            what people call a polyline is usually "LWPOLYLINE".
        block_name: exact block name, only meaningful together with INSERT.
        layout: restrict to one layout, e.g. "Model". Case-sensitive.
        limit: rows to return, capped at 100.
        offset: rows to skip, for paging through a large result.

    Returns `total_matches`, `truncated`, and up to `limit` compact rows. If
    the match count is very large it returns a per-layer and per-type
    breakdown instead of rows, so you can narrow the filter. To count
    *different* values rather than rows, use `distinct_values`.
    """
    data = _get(
        f"/drawings/{drawing_id}/entities",
        {
            "layer": layer,
            "type": type,
            "block_name": block_name,
            "layout": layout,
            "text_contains": text_contains,
            "limit": min(limit, MAX_ROWS),
            "offset": offset,
        },
    )
    if data.get("ok") is False:
        return data
    return _compact(data)


@mcp.tool
def spatial_query(
    drawing_id: str,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    layout: str | None = None,
    layer: str | None = None,
    type: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Find entities inside a rectangle, in drawing coordinates.

    Use this for location questions — "what is next to the stairs", "what is
    in the north-east corner". No other tool can answer those: a text search
    would return a confident-sounding guess.

    **Pass `layout`.** Without it the rectangle is tested against model space,
    every paper sheet and every block definition at once, and those do not
    share a coordinate frame in any meaningful way. Measured on Janadriyah:
    "what is within 100 m of this label" returned 437 unscoped and 238 in
    model space. The unscoped number is not a broader answer, it is a wrong
    one.

    Get the coordinate range from `describe_drawing`'s `extents` first, and
    remember the units it reports; Janadriyah is in metres and the Autodesk
    samples are mostly in inches.

    Args:
        drawing_id: id from `list_drawings()`.
        minx, miny, maxx, maxy: rectangle corners, in drawing units.
        layout: restrict to one layout, e.g. "Model". Strongly recommended.
        layer: restrict to one layer, matched exactly.
        type: restrict to one DXF type, e.g. "TEXT".
        limit: rows to return, capped at 100.

    Returns entities whose bounding box overlaps the rectangle. The test is on
    bounding boxes, not exact geometry: a long diagonal line can be returned
    by a rectangle that only clips the empty corner of its box.
    """
    data = _get(
        f"/drawings/{drawing_id}/spatial",
        {
            "minx": minx,
            "miny": miny,
            "maxx": maxx,
            "maxy": maxy,
            "layout": layout,
            "layer": layer,
            "type": type,
            "limit": min(limit, MAX_ROWS),
        },
    )
    if data.get("ok") is False:
        return data
    return _compact(data)


@mcp.tool
def distinct_values(
    drawing_id: str,
    field: str,
    layer: str | None = None,
    layout: str | None = None,
    type: str | None = None,
    top: int = 20,
) -> dict[str, Any]:
    """Count how many *different* values a field takes, and list the commonest.

    Use for "how many different X are there" — distinct text strings, layers
    actually in use on one sheet, block names in a layout. Answering that by
    paging through rows costs dozens of calls and hundreds of thousands of
    characters to produce one integer, and the count comes out wrong the
    moment a page is missed.

    Args:
        drawing_id: id from `list_drawings()`.
        field: one of "text", "layer", "type", "layout", "block_name".
        layer, layout, type: narrow the population first. Scope matters:
            distinct text strings in model space and in the whole file are
            different numbers and both are correct.
        top: how many of the commonest values to list, capped at 50.

    Returns `distinct_values`, `entities_with_a_value`, and `most_common`.
    Entities whose field is empty are excluded from both counts.

    For `layer` and `block_name` it also returns **`defined_but_not_present`**:
    the values this drawing DEFINES that do not appear in the scope you asked
    about. A drawing declares its layers and its block definitions up front, and
    how many of them anything is actually drawn on is a separate question.
    Grouping entities can only ever return values that occur, so without this
    the declared-and-unused half is invisible -- invisible in a way that reads
    as non-existent, because a complete-looking list came back.

    Each entry says whether that value is drawn anywhere else and where, which
    separates three different answers to "is there any": defined and never used
    at all; used, but not in this scope; and used only inside something that is
    itself never placed. "How many of X are in this drawing" needs both halves,
    and one call now returns both.

    `top` does not apply to it: the absent half is listed up to its own cap and
    says when it truncates.
    """
    return _get(
        f"/drawings/{drawing_id}/distinct",
        {
            "field": field,
            "layer": layer,
            "layout": layout,
            "type": type,
            "top": min(top, 50),
        },
    )


@mcp.tool
def describe_selection(
    drawing_id: str,
    selection_id: str | None = None,
    handles: list[str] | None = None,
    sample: int = 3,
) -> dict[str, Any]:
    """Describe the user's current selection in aggregate.

    This is the tool for a region the user drew in the viewer. It turns a
    selection into its *shape* — how many objects, on which layers, of which
    types, their total length and area, the layouts involved, and the box they
    occupy — in one call.

    **Use `selection_id`.** Your context carries one whenever the user has a
    selection active. It refers to the selection stored server-side, so this
    tool reads it **whole**, however large it is: a 20,000-object selection is
    described completely and the answer says so. Do not ask the user to
    re-select in order to shrink it.

    `handles` exists only for the rare case where you were handed a literal
    list and no id. Prefer the id; a list you assembled yourself is a list you
    can get wrong.

    Call this **before** reaching for `get_entity`. A selection of a thousand
    objects fetched one at a time is a thousand calls and an answer nobody can
    check; the aggregate is one call, and it tells you which handful of
    objects deserve a closer look.

    Everything reported here is safe to state as fact, and every claim can be
    pinned to a handle — which is what lets the user check it in the viewer.
    `units` and `units_declared_in_file` come back with it, so quote them
    rather than guessing.

    Args:
        drawing_id: id from `list_drawings()`.
        selection_id: reference to the user's stored selection. Preferred.
        handles: explicit handles, only when no id is available.
        sample: example rows per (layer, type) group, capped at 10.
    """
    if not selection_id and not handles:
        return {
            "ok": False,
            "error": "EMPTY_SELECTION",
            "message": "No selection_id and no handles were given.",
            "hint": (
                "If the user has a selection, its selection_id is in your "
                "context. Otherwise ask them to draw a region in the viewer."
            ),
        }

    body: dict[str, Any] = {"sample": min(sample, 10)}
    if selection_id:
        body["selection_id"] = selection_id
    else:
        body["handles"] = handles

    try:
        response = _client.post(
            f"/drawings/{drawing_id}/selection/describe", json=body
        )
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": "CAD_API_UNREACHABLE",
            "message": f"Could not reach cad-api: {type(exc).__name__}.",
            "hint": "The drawing service may be down. Ask the user to check it.",
        }
    if response.status_code >= 400:
        try:
            body_json = response.json()
        except ValueError:
            body_json = {}
        return {
            "ok": False,
            "error": body_json.get("error", f"HTTP_{response.status_code}"),
            "message": body_json.get("message", response.text[:200]),
            "hint": body_json.get("hint", ""),
        }

    data = response.json()
    if data.get("missing_handles"):
        data["note"] = (
            f"{len(data['missing_handles'])} of the handles given do not exist "
            "in this drawing and were ignored; the totals cover the rest."
        )
    return data


@mcp.tool
def measure(
    drawing_id: str,
    measure: str,
    layout: str,
    layer: str | None = None,
    type: str | None = None,
    block_name: str | None = None,
    group_by: str | None = None,
    top: int = 25,
) -> dict[str, Any]:
    """Total the length or area of everything matching a filter, in one call.

    The quantity take-off tool: "total length of the LWPOLYLINEs on layer VL2",
    "total area on layer X", "a per-layer recap for the bill of quantities".
    Reach for it whenever the answer is a NUMBER rather than a list of objects.

    Do not compute a total by paging `query_entities` and adding rows. A real
    session tried exactly that on one layer, spent 136 calls fetching entities
    one at a time, hit an error and reported 11 of 133 -- a sum assembled by
    paging is wrong the moment one page is missed, and it is slow enough that
    the attempt is abandoned. This is one call.

    When you pass `group_by`, the rows are ranked by the MEASURE, not by how many entities each group holds -- `groups_ordered_by` says so in the response. A question about which layers hold the most ENTITIES is distinct_values(field=..., layout=...), not this. QUOTE `statement`. It already contains the number, its unit, the scope it
    covers and the denominator it was summed from. Two wrong figures shipped on
    this project were both composed in prose around a correct number: one
    reported "1,928 DIMENSIONs, about 72%" when the truth was 9,867 and 14.1%,
    and another reported 812 where the truth was 818. Do not compose a
    percentage this response does not contain.

    **`duplicate_warning` — never quote the total without it.** When the layer
    you measured holds identical clusters, the total is the truth MULTIPLIED by
    the number of copies, and every other field in this response is
    arithmetically correct while the number is three times wrong. The reference
    case: the road centreline layer totals 83,962.565 m, and that is three
    copies of 27,987.522 m stacked on one another — the block definition states
    27,987.52167 itself, so the per-copy figure is corroborated rather than
    guessed. The warning names the cluster count and the per-copy figure. Quote
    both, and say which is which: "83,962.565 m as drawn, but the layer holds 3
    identical copies, so the distinct length is 27,987.522 m."

    The warning is one-sided by construction. It appears when copies were
    detected; its ABSENCE is not a certificate that a layer is clean — the
    detector can only miss, never invent, and it is silent altogether on a
    drawing whose Dossier has not been built. `find_duplicates` is the tool
    that asks the question directly.

    Read the denominator before quoting anything. Most entities cannot be
    measured -- in one model space only 4,205 of 20,334 carry a length -- so
    `total_for_all_matched` is a number ONLY when every matched entity was
    measurable. Whenever it is `null`, `sum_measured_only` is a FLOOR and not a
    total, and you must say so. `null` means "not measurable"; `0` means
    "measured as zero", and `zero_valued_entities` counts the second.

    What carries a measurement: LINE, ARC, CIRCLE and straight-segment
    LWPOLYLINE have a length -- LWPOLYLINE is the one that matters most, since
    road centrelines and plot boundaries are drawn with it; CIRCLE and closed
    LWPOLYLINE have an area; an LWPOLYLINE with any curved (bulge)
    segment has NEITHER, so an arc-drawn road centreline contributes nothing to
    a road total; HATCH carries no area at all. When something is skipped,
    `unmeasurable_by_type` says what it was.

    Args:
        drawing_id: the drawing on screen.
        measure: "length" or "area".
        layout: REQUIRED, e.g. "Model". Unscoped, a total spans model space,
            every paper sheet and every block definition at once and counts the
            same geometry more than once. Case-sensitive; `describe_drawing`
            lists the layouts.
        layer: exact layer name, case-sensitive. A LAYER, not a DXF type -- this
            drawing has a layer named `DIM` and a type named `DIMENSION`, and
            passing one for the other is refused rather than answered.
        type: DXF type, e.g. "LWPOLYLINE". Upper-cased for you. Omitting it sums
            every linear type together, which is ordinary for a road total but
            wrong if CIRCLEs are present -- their length is a circumference. The
            response warns when that happens.
        block_name: exact block name. Useful with type="INSERT" for counting;
            note that INSERTs carry no length or area of their own.
        group_by: "layer" or "type" for a per-group breakdown. The totals stay
            over everything even when the group list is capped.
        top: groups listed, max 200.

    Returns `statement`, `total_matches`, `measured_entities`,
    `unmeasurable_entities`, `zero_valued_entities`, `measured_fraction`,
    `sum_measured_only`, `total_for_all_matched`,
    `total_for_all_matched_withheld`, `unmeasurable_by_type`, `units`, `scope`,
    `scope_label` and `warnings`; with `group_by`, also `by_layer` or `by_type`
    where every row carries its own denominator; and `duplicate_warning` when
    the layer holds identical clusters. On an empty match it returns
    `why_empty` and `why_empty_facts` -- read those before reporting an absence.
    """
    data = _get(
        f"/drawings/{drawing_id}/measure",
        {
            "measure": measure,
            "layout": layout,
            "layer": layer,
            "type": type,
            "block_name": block_name,
            "group_by": group_by,
            "top": min(top, 200),
        },
    )
    if data.get("ok") is False:
        return data
    # `by_layer` and `by_type` are already bounded by `top` and are the answer
    # the caller asked for, so they are not offered for trimming; only the
    # anomaly block, which is the part that grows on its own, is.
    return _budget(data, trim_inside=("duplicate_warning",))


@mcp.tool
def search_text(drawing_id: str, query: str, limit: int = 20) -> dict[str, Any]:
    """Search the annotation text: MTEXT, TEXT, dimensions, block attributes.

    A last resort, not a first move. Reach for it only when the question is
    genuinely about wording — a room name, a plot number, a note on the
    drawing. For anything structural (`how many`, `which layer`, `what type`)
    use `query_entities`, which is exact and much cheaper.

    Args:
        drawing_id: id from `list_drawings()`.
        query: substring to look for; matching is case-insensitive and
            partial, so "A-1" finds "A-101".
        limit: rows to return, capped at 100.
    """
    data = _get(
        f"/drawings/{drawing_id}/search",
        {"q": query, "limit": min(limit, MAX_ROWS)},
    )
    if data.get("ok") is False:
        return data
    return _compact(data)


@mcp.tool
def get_entity(drawing_id: str, handle: str) -> dict[str, Any]:
    """Everything about one entity, including comments already attached to it.

    Use after a query has narrowed things down to a specific object. The list
    tools return compact rows on purpose; this is where the full geometry,
    block attributes and comment history live.

    Args:
        drawing_id: id from `list_drawings()`.
        handle: entity handle taken from a query result. Handles are unique
            only within one drawing, and must never be invented.
    """
    data = _get(f"/drawings/{drawing_id}/entities/{handle}")
    if data.get("ok") is False:
        return data
    return {
        "handle": data.get("handle"),
        "type": data.get("type"),
        "layer": data.get("layer"),
        "layout": data.get("layout"),
        "block_name": data.get("block_name"),
        "text": data.get("text"),
        "attribs": data.get("attribs"),
        "bbox": data.get("bbox"),
        "bbox_centre": data.get("bbox_centre"),
        "length": data.get("length"),
        "area": data.get("area"),
        "comments": [
            {"body": c["body"], "author": c["author"], "created_at": c["created_at"]}
            for c in data.get("comments", [])
        ],
    }


@mcp.tool
def add_comment(
    drawing_id: str,
    handle: str,
    body: str,
    dry_run: bool = True,
    client_request_id: str | None = None,
) -> dict[str, Any]:
    """Attach a comment to one entity. The original DWG file is never modified.

    Comments are stored separately from the CAD file, keyed by
    `(drawing_id, handle)`, and appear in the viewer when the user clicks that
    object.

    Run with `dry_run=True` first and read back which entity you are about to
    annotate. This is not about file safety — comments never touch the DWG —
    but about catching an invented handle before thirty comments end up on the
    wrong objects. Then repeat with `dry_run=False` to save.

    Args:
        drawing_id: id from `list_drawings()`.
        handle: entity handle from a query result, never constructed by hand.
        body: the comment text.
        dry_run: when True, report the target entity and save nothing.
        client_request_id: optional idempotency key. Reusing the same value
            returns the original comment instead of creating a duplicate, so a
            retry after a timeout is safe.
    """
    entity = _get(f"/drawings/{drawing_id}/entities/{handle}")
    if entity.get("ok") is False:
        return entity

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "would_annotate": {
                "handle": entity.get("handle"),
                "type": entity.get("type"),
                "layer": entity.get("layer"),
                "text": entity.get("text"),
                "existing_comments": len(entity.get("comments", [])),
            },
            "next_step": "Repeat this call with dry_run=False to save it.",
        }

    try:
        response = _client.post(
            "/comments",
            json={
                "drawing_id": drawing_id,
                "handle": handle,
                "body": body,
                "author": "agent:cad-agent",
                "client_request_id": client_request_id or str(uuid.uuid4()),
                "provenance": {"tool": "add_comment", "source": "cad-mcp"},
            },
        )
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": "CAD_API_UNREACHABLE",
            "message": f"Could not save the comment: {type(exc).__name__}.",
            "hint": "The drawing service may be down.",
        }

    if response.status_code >= 400:
        try:
            body_json = response.json()
        except ValueError:
            body_json = {}
        return {
            "ok": False,
            "error": body_json.get("error", f"HTTP_{response.status_code}"),
            "message": body_json.get("message", response.text[:200]),
            "hint": body_json.get("hint", ""),
        }

    saved = response.json()
    return {
        "ok": True,
        "dry_run": False,
        "saved": {
            "handle": saved.get("entity_handle"),
            "body": saved.get("body"),
            "author": saved.get("author"),
        },
        "note": "Stored in MongoDB. The original DWG file was not modified.",
    }


# ---------------------------------------------------------------------------
# UPLIFT tools.
#
# Everything below is a thin pass-through to cad-api, which is the rule this
# server has always followed: no CAD logic lives here. What is new is that the
# agent can now ask questions it previously could not, and each docstring says
# when the tool is NOT the right one -- because a model that reaches for the
# wrong tool confidently is the failure this whole series exists to fix.
# ---------------------------------------------------------------------------


@mcp.tool
def land_use_summary(drawing_id: str, layout: str) -> dict[str, Any]:
    """What the drawing is FOR, per land use, with how firmly each is known.

    Use this for "how many houses / schools / mosques are there", and for any
    question about what a part of the drawing is rather than how much of it
    there is.

    **A zero with a `residual` beside it is NOT an absence.** This is the rule
    this whole tool exists to protect, and it is the one most easily thrown
    away in the last sentence of an answer. When a use reports 0 parcels while
    unclassified layers hold geometry whose NAME speaks for that use, the row
    carries a `residual` naming those layers with their role, their entity
    count and their native measure. On the reference drawing with no road
    config, "road: 0 parcels" arrives beside a residual reading
    `00_Prop - Road - CL_`, role network, 1,233 entities, 83,962.565 m. The
    road PARCELS are not there. THE ROADS ARE. Relay the residual every time,
    in the same breath as the zero -- dropping it is exactly how "there are no
    roads" got said about a drawing carrying 84 km of them.

    **Name the frame around every zero.** "0 road parcels" is a truth INSIDE
    the parcel frame: it counts closed rings on layers a config maps to a use,
    and `parcel_basis` in this response spells out what that includes. Repeated
    as a truth about the world it is false, and the reader cannot see the frame
    you dropped. Say what was counted and what was found instead: "no road
    PARCELS are configured; the road geometry sits on `<layer>` as a network of
    1,233 entities totalling 83,962.565 m."

    **A missing residual is not a residual of nothing.** A zero row with no
    `residual` at all means the question "is this use present as something
    other than parcels" was never asked -- the Dossier for this drawing has not
    been built. `residual_status` says so when that happens. A residual that IS
    present and lists nothing is the other answer, and it was asked.

    The residual also tells a real layer from a template one, because it
    carries geometry rather than a name. Sixty `C-ROAD-*` sheet-layout layers
    match the word "road" on this drawing and come back with few or no entities
    and no measure; the one road centreline layer comes back with 1,233 and a
    length. The name proposes; the geometry disposes. And because a name search cannot see a
    layer whose name says nothing, the same block carries
    `largest_unclassified`: the biggest unclassified layers ranked by what
    they hold, one per geometric role, flagged `name_matched: false`. On
    the reference drawing that is where the right-of-way corridors appear
    for a question about roads. Read both lists before answering. Rank by what was
    measured, not by what a layer is called.

    `unclassified_layers` carries `profiles`: role, entity count and native
    measure for the layers it names. "A layer that is unclassified is not a
    layer that is empty" was always the note on that block; the profiles are
    what finally make it checkable rather than a disclaimer.

    Every row carries `evidence` with one of four grades. Read it and pass it
    on, do not flatten it: `stated` means the file says so, `inferred` means we
    concluded it and the file never says the word. Asked how many houses a
    drawing held, an agent once answered "maybe there are no houses" about
    2,380 residential plots -- because nothing in any response told it that
    residential is a conclusion drawn from plot geometry rather than a label
    written in the file.

    **`by_layer` already answers "count and average area per type" — do not
    follow this call with one `stats` call per layer.** Each row carries
    `parcels`, `parcels_measured`, `area_total`, `area_mean` and `area_unit`,
    plus `area_not_measured` naming what was left out and why. Measured on 24
    August 2026, an agent made thirteen extra `stats` calls for numbers that
    were already in the response it had just read, and on one run the seventh
    of those came back as a garbled tool name and cost the entire answer.

    **Asked WHERE these things are, name the layers and stop.** Do not follow
    this with a `query_entities` sweep per layer to fetch handles for the
    viewer -- the layers you name here are resolved for it already, whole and
    server-side. Measured on 24 August 2026: an agent that paged all thirteen
    typology layers itself took 31 tool calls and two and a half minutes to
    produce marks that were on screen either way.

    NOT the tool for counting entities on a named layer -- that is
    `query_entities`. NOT the tool for totals -- that is `measure`. Reach for
    `stats` only when you need median, mode or standard deviation, which this
    does not carry.
    """
    data = _get(f"/drawings/{drawing_id}/land-use", {"layout": layout})
    if data.get("ok") is False:
        return data
    _mark_uncomputed_dossier_blocks(data)
    # The largest response this server returns -- 59,972 bytes on the reference
    # drawing before Phase 2 added anything to it -- so it is the one that
    # actually needs the net. The Dossier-fed blocks are offered for trimming
    # ahead of `uses`, because `uses` is the answer.
    return _budget(data, trim_inside=("dossier", "unclassified_layers", "uses"))


@mcp.tool
def join_labels(
    drawing_id: str,
    layout: str,
    target_layer: str | None = None,
    label_layer: str | None = None,
    label_type: str = "TEXT",
    limit: int = 50,
) -> dict[str, Any]:
    """Which text sits INSIDE which polygon, by shape rather than by box.

    Use this to attach plot numbers, room names or any annotation to the parcel
    it labels.

    The difference from a bounding-box test is not small: on one parcel the box
    pulls in 42 neighbouring texts where point-in-polygon leaves the 1 that is
    genuinely inside. Only polygons whose ring is complete are eligible;
    curved and open ones are skipped AND counted, split by cause. A skipped
    target is not a target that failed to match, and the response says which
    is which.

    NOT the tool for "what is near this" -- that is `proximity_count`.
    """
    return _get(
        f"/drawings/{drawing_id}/join-labels",
        {
            "layout": layout,
            "target_layer": target_layer,
            "label_layer": label_layer,
            "label_type": label_type,
            "limit": min(limit, 500),
        },
    )


@mcp.tool
def distance_matrix(
    drawing_id: str,
    layout: str,
    layers: str | None = None,
    handles: str | None = None,
    mode: str = "centroid",
    cluster_threshold: float | None = None,
) -> dict[str, Any]:
    """Distance between every pair, in one call rather than one call per pair.

    **This answers every distance question this drawing can support, including
    the ones between two different kinds of thing.** "How far is each school
    from all the mosques" is one call with both sets of layers in `layers` --
    schools and mosques together -- and the response already carries:

    - `matrix`: every pair;
    - `nearest`: for each item, its closest counterpart and that distance;
    - `stats`: `min`, `max`, `mean`, `median` and `pairs_measured` over the
      whole set;
    - `skipped_*`: how many parcels could not be measured, by cause.

    So there is never a reason to say the calculation is unavailable. Measured
    on 24 August 2026 an agent answered "there are no tools available to
    directly calculate the distance from each school to all mosques" while
    holding this one; the numbers were a single call away.

    Use this for "how far apart", "nearest", "furthest", "average distance",
    "shortest", and anything of that shape. `layers` and `handles` are
    comma-separated.

    These are STRAIGHT-LINE distances and the response says so. They are not
    walking distances: that would need a road network, which this drawing does
    not carry. Do not describe them as walking distances even if asked in those
    words -- say what was measured.

    **Report the `skipped_*` counts. A parcel that could not be measured is
    part of the answer.** Asked how far each school was from all the mosques,
    an agent listed four mosques and said nothing about the other two, whose
    rings carry arcs and therefore have no centroid -- `skipped_bulge: 2` was
    right there in the response. Nine schools times four mosques reads exactly
    like a complete matrix, and the reader has no way to tell it covers two
    thirds of the mosques. Say "4 of the 6" and say why.

    Measured from the true polygon centroid, not the centre of the bounding
    box. For a large or L-shaped parcel those two points can be tens of metres
    apart. There is a cap on how many items may be compared, and going over it
    returns a refusal with a way forward rather than hanging.
    """
    return _get(
        f"/drawings/{drawing_id}/distance-matrix",
        {
            "layout": layout,
            "layers": layers,
            "handles": handles,
            "mode": mode,
            "cluster_threshold": cluster_threshold,
        },
    )


@mcp.tool
def proximity_count(
    drawing_id: str,
    layout: str,
    around_layers: str,
    count_layers: str,
    radius: float,
    measure_from: str = "centroid",
) -> dict[str, Any]:
    """How many of one thing sit within a radius of each of another thing.

    Use this for "how many houses are near each school" -- the question that
    was asked on 23 August and never answered, because the session hung first.
    Both layer arguments are comma-separated.

    `radius` is in DRAWING UNITS and the response echoes which unit that is.
    Eleven of the eighteen drawings here are in inches and three declare no
    unit at all, so a radius of 400 does not mean 400 metres unless the drawing
    says metres. Check the echoed unit before repeating the number.

    Every row carries more than a count: `nearest`, `farthest` and
    `total_area` for the objects inside the radius. **Report them.** "How many
    houses are near this school" and "how close are they" are the same
    question asked twice, and the second one is already answered in the row
    you are holding. A table of counts alone throws away two thirds of what
    was measured.

    **Never refuse to answer because nobody named a radius.** "Vicinity",
    "nearby" and "around" have no fixed value, so pick one, answer with it,
    and say in one clause that you picked it and it can be changed. A walking
    catchment of 400 m is the usual planning figure where the drawing is in
    metres; scale it to whatever unit the drawing declares. Measured on 25
    August 2026: asked how many houses are near each school, the agent asked
    the user for a radius and returned no figures at all -- the same
    ask-back that lost this question on 23 August, one step later in the
    conversation.
    """
    return _get(
        f"/drawings/{drawing_id}/proximity-count",
        {
            "layout": layout,
            "around_layers": around_layers,
            "count_layers": count_layers,
            "radius": radius,
            "measure_from": measure_from,
        },
    )


@mcp.tool
def stats(
    drawing_id: str,
    layout: str,
    measure: str = "area",
    layers: str | None = None,
    type: str | None = None,
) -> dict[str, Any]:
    """Mean, median, mode and standard deviation, per layer rather than pooled.

    Use this for "average plot area per type" and for any spread question.

    **Pass EVERY layer you care about in one call, comma-separated. Do not call
    this once per layer.** The response already carries one row per layer plus
    a combined row over the same population, so thirteen calls produce thirteen
    answers a reader then has to add up -- which is the work this tool exists to
    do. It is also thirteen chances for the turn to die: measured on 24 August
    2026, an answer that made thirteen separate calls emitted a garbled tool
    name on the seventh, and the whole reply was lost even though twelve calls
    had already succeeded.

    Omitting `layers` covers every layer in the drawing.

    Objects that cannot be measured never enter as zero. Each row names its own
    denominator -- "133 of 20,000" -- and counts what was left out. A mean that
    silently averaged in unmeasurable objects as 0 would be lower than the
    truth and would look perfectly reasonable.

    NOT the tool for a plain total -- that is `measure`.
    """
    return _get(
        f"/drawings/{drawing_id}/stats",
        {"layout": layout, "measure": measure, "layers": layers, "type": type},
    )


@mcp.tool
def find_duplicates(
    drawing_id: str,
    layout: str,
    kind: str = "geometry",
    tolerance: float = 0.01,
    limit: int = 50,
) -> dict[str, Any]:
    """Polygons drawn twice, and shapes that simply repeat. Two questions.

    `kind="geometry"` finds copies sitting on top of each other -- a real
    defect, and the reason a plot number can land inside two polygons at once.
    `kind="shape"` finds the repeated plot modules, which in a masterplan is
    not a defect at all but the point of the design. The response says which
    question it answered.

    `tolerance` is in drawing units and is echoed back. Note that it does two
    things: it bounds how far apart copies may sit, and it also controls how
    coarsely shapes are bucketed together -- measured on the reference drawing,
    the bucketing is what moves the answer, not the distance.
    """
    return _get(
        f"/drawings/{drawing_id}/duplicates",
        {
            "layout": layout,
            "kind": kind,
            "tolerance": tolerance,
            "limit": min(limit, 500),
        },
    )


@mcp.tool
def shape_fingerprint(drawing_id: str, layout: str, limit: int = 50) -> dict[str, Any]:
    """The plot modules a drawing uses, read from geometry and nothing else.

    Use this to answer "what plot sizes are in here" without trusting a single
    layer name. That is the point: the next drawing from a different contractor
    will name its layers differently, and a 12x25 plot is still a 12x25 plot.

    Fingerprints are only comparable WITHIN one drawing. They are sensitive to
    scale but blind to units, so a 12x25 metre plot and a 12x25 inch component
    produce the same key.
    """
    return _get(
        f"/drawings/{drawing_id}/shapes", {"layout": layout, "limit": min(limit, 500)}
    )


@mcp.tool
def find_by_name(
    drawing_id: str, query: str, kinds: str | None = None
) -> dict[str, Any]:
    """Search the drawing's dictionary -- layers, styles, linetypes, blocks.

    Different from `search_text`, and the difference matters: `search_text`
    searches annotation that was drawn, this searches names that were declared.
    A layer that exists in the table and was never drawn on cannot be found by
    the first and is found by the second. Asked whether something exists in a
    drawing, an absence from `search_text` is not an absence from the file.

    `kinds` is comma-separated.
    """
    return _get(f"/drawings/{drawing_id}/find-by-name", {"q": query, "kinds": kinds})


@mcp.tool
def drawing_tables(drawing_id: str) -> dict[str, Any]:
    """The drawing's own dictionary: layers with their real state, styles, units.

    Use this when a question turns on how the drawing is SET UP rather than what
    is in it -- which layers are switched off, what the text styles point at,
    what the plot settings say.

    Layer state is the part worth reading. A layer that is off or set not to
    plot still holds its entities, so "why can I not see these labels" and "are
    these labels missing" are different questions with different answers.
    """
    return _get(f"/drawings/{drawing_id}/tables")


@mcp.tool
def list_layer_tags(drawing_id: str) -> dict[str, Any]:
    """Business names people have given to layers in this drawing.

    These are human assertions, not statements the file makes, and their
    evidence says so. Someone naming a typology on screen does not turn a
    conclusion into a fact -- it records who said it and on what basis. Report
    the name and the source together, never the name alone.
    """
    return _get(f"/drawings/{drawing_id}/tags")


@mcp.tool
def list_analyses() -> dict[str, Any]:
    """Every named calculation this drawing service can run, and what it is for.

    Read this BEFORE deciding a question cannot be answered. The catalogue is
    the honest boundary of what exists: each entry says what it answers, when
    to use it, what parameters it takes, and what it returns.

    It also carries `no_recipe_sentence`. When nothing in the catalogue fits,
    say that sentence — name the calculation that is missing, name the closest
    thing that does exist, and say it can be added as an analysis recipe. Do
    not force the nearest recipe: a recipe answering the wrong question
    produces a wrong number in a confident voice, and that is the most
    expensive failure in a technical document.
    """
    return _get("/analyses")


@mcp.tool
def run_analysis(
    drawing_id: str,
    recipe: str,
    layout: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one reviewed, named calculation over a drawing.

    You choose the recipe and fill in its parameters; you never write the
    logic. That is deliberate and it is not going to change: drawing content
    comes from contractors, and the service that would run the code also holds
    the credentials to a shared database.

    Call `list_analyses` first for names and parameters. An unknown name
    returns the catalogue rather than an error, so a wrong guess costs one call
    and tells you the right one.

    Answers carry the same contract as every other tool here: `scope_note`,
    `evidence` with its grade, and `not_measured` for what was left out.
    Quote the grade — `inferred` means the drawing did not say it, and
    `coverage_gap` in particular reports a shortfall as an OBSERVATION whose
    cause the drawing does not establish. Reporting it as a defect is an
    accusation, not a measurement.
    """
    return _post(
        f"/drawings/{drawing_id}/analyses/{recipe}",
        {"layout": layout},
        params or {},
    )


def _compact(data: dict[str, Any]) -> dict[str, Any]:
    """Trim a cad-api list response to the fields worth spending context on.

    Full geometry for 100 entities is tens of thousands of tokens; a handful
    of compact fields is enough for a model to decide what deserves a closer
    look via `get_entity`.

    `block_name` and `layout` are kept, and that is a correction rather than a
    preference. They were dropped here while cad-api returned them, so "which
    block is used most often" — a question every row already carried the
    answer to — could only be reached by calling `get_entity` forty-two times.
    A trim that removes the answer is not a trim.
    """
    rows = []
    for row in data.get("entities", []):
        text = row.get("text")
        compact = {
            "handle": row.get("handle"),
            "type": row.get("type"),
            "layer": row.get("layer"),
            "text": (text[:80] + "…") if text and len(text) > 80 else text,
            "bbox_centre": [round(c, 3) for c in row["bbox_centre"]]
            if row.get("bbox_centre")
            else None,
        }
        # Only when they carry information: every row of a layout-scoped query
        # repeating the same layout name is noise, and most entities are not
        # block references.
        if row.get("block_name"):
            compact["block_name"] = row["block_name"]
        if row.get("layout"):
            compact["layout"] = row["layout"]
        rows.append(compact)

    out: dict[str, Any] = {
        "total_matches": data.get("total_matches", 0),
        "returned": data.get("returned", len(rows)),
        "truncated": data.get("truncated", False),
        "entities": rows,
    }
    if data.get("scope"):
        out["scope"] = data["scope"]
    if data.get("next_offset") is not None:
        out["next_offset"] = data["next_offset"]
    if data.get("too_many"):
        # Rows were still returned, but the breakdown is the more useful
        # answer at this size; surface it so the model narrows down.
        out["too_many"] = True
        out["breakdown"] = data.get("breakdown")
        out["hint"] = data.get("hint")
    # An empty result that cad-api could explain must arrive explained. This
    # is the single most important field in the file: without it "0 matches"
    # and "you asked the wrong question" are the same response, and a model
    # reports absence as fact.
    if data.get("why_empty"):
        out["why_empty"] = data["why_empty"]
        # These facts belong with `why_empty` and had drifted onto the shadow
        # branch, so a drawing with no block definitions lost them entirely --
        # including `layouts_note`, which is a row in the label sweep's own
        # fixed table.
        #
        # Renamed from `evidence` in D-082: that key now belongs to the
        # UPLIFT-08 grade contract, and two meanings under one name in one
        # response is the defect class the label sweep exists to catch.
        out["why_empty_facts"] = data.get("why_empty_facts")
    # The one string to quote when a filter matched nothing. `why_empty` is a
    # list, and a list gets summarised; the summary is where "defined but never
    # placed" turned back into "there are none".
    if data.get("why_empty_statement"):
        out["why_empty_statement"] = data["why_empty_statement"]
    if data.get("scope_note"):
        out["scope_note"] = data["scope_note"]
    if data.get("block_definition_shadow"):
        out["block_definition_shadow"] = data["block_definition_shadow"]
    # The same filter counted across the whole drawing, present only when a
    # layout narrowed the query. It is preserved for the reason `why_empty`
    # is: without it the model holds one number where the question wanted
    # two. Asked whether anything sat on layer 0, the agent answered 10 for
    # the sheet and never mentioned the 639 in the drawing -- correct, scoped,
    # and not the figure an audit was reaching for. Dropping it here would
    # undo the fix in cad-api one layer further out.
    if data.get("total_in_drawing") is not None:
        out["total_in_drawing"] = data["total_in_drawing"]
    return out


def _dossier_block(value: Any) -> dict[str, Any]:
    """Normalise the Dossier summary so that ABSENT never arrives as EMPTY.

    cad-api returns nothing for a drawing whose Dossier has not been built, and
    from here a key that is null, a key that is `{}` and a key that was never
    mounted are the same event: silence. Silence about a drawing gets read as a
    statement about the drawing. So an absent Dossier is turned into a block
    that says, in words the model will repeat, that it was not computed, that
    the drawing is not thereby empty, and what still answers in the meantime.

    A Dossier that is present passes through untouched and is marked
    `computed`. Nothing here reshapes cad-api's numbers -- this function knows
    only the difference between something and nothing.
    """
    if not isinstance(value, dict) or not value:
        return {
            "status": "not_computed",
            "meaning": (
                "No Dossier has been built for this drawing yet. This is a "
                "MISSING ANSWER, not an empty drawing: every other field in "
                "this response is still true. Do not report it as 'nothing "
                "found', and do not conclude anything about what the drawing "
                "contains from its absence."
            ),
            "meanwhile": (
                "distinct_values(field='layer', layout=...) lists the layers "
                "that actually hold something, and measure(...) totals one of "
                "them. The Dossier is faster and covers the whole file at "
                "once; nothing here becomes unanswerable without it."
            ),
            "how_to_build": "scripts/dossier_backfill.py, per drawing_id.",
        }
    block = dict(value)
    block.setdefault("status", "computed")
    return block


def _mark_uncomputed_dossier_blocks(data: dict[str, Any]) -> None:
    """Say "not computed" in the two places where saying nothing reads as an
    answer.

    Mutates `data` in place, and only ever ADDS a sibling key -- nothing is
    renamed, removed or re-typed, because the viewer and the stream route read
    this payload by key.

    Both cases are the same defect wearing different clothes. A zero row with
    no residual is a question that was never asked, arriving as a question that
    was answered no. An `unclassified_layers` block with no profiles is a list
    of layer names with nothing to distinguish a road centreline from a sheet
    grid, which is the state of affairs Phase 2 exists to end.
    """
    block = data.get("unclassified_layers")
    if isinstance(block, dict) and not block.get("profiles"):
        block["profiles_status"] = (
            "not computed for this drawing -- no role, entity count or measure "
            "is available for these layers here. Unclassified is not empty: "
            "ask measure(layer=...) or distinct_values about one of them "
            "before saying there is nothing on it."
        )

    for row in data.get("uses") or []:
        if not isinstance(row, dict) or row.get("parcels") != 0:
            continue
        if row.get("residual") is None:
            row["residual_status"] = (
                "no residual was computed for this zero, so the question 'is "
                f"{row.get('use')!r} present in this drawing as something "
                "other than parcels' was never asked. It was NOT answered no. "
                "Report the 0 as a count of configured parcels and say the "
                "wider question is open."
            )


def _response_chars(payload: Any) -> int:
    """What this response will cost the agent, measured rather than assumed."""
    try:
        return len(json.dumps(payload, default=str, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(repr(payload))


def _lists_within(
    node: Any, path: str, depth: int = 0
) -> list[tuple[dict[str, Any], str, str]]:
    """Every list held under `node`, as (holder, key, label).

    The Dossier blocks are written by other modules and their key names are
    theirs to choose, so this finds trimmable rows by SHAPE rather than by
    name: nothing here needs updating when a block gains a field.

    Two rules keep it safe. The root itself is never a candidate -- `uses` is
    the answer to the question and is recursed INTO, never shortened. And a
    list that is offered as a candidate is not also recursed into, so no
    candidate can ever point at rows that an earlier trim has already dropped.
    """
    found: list[tuple[dict[str, Any], str, str]] = []
    if depth > 4:
        return found
    if isinstance(node, dict):
        for key, value in node.items():
            label = f"{path}.{key}"
            if isinstance(value, list):
                found.append((node, key, label))
            elif isinstance(value, dict):
                found.extend(_lists_within(value, label, depth + 1))
    elif isinstance(node, list):
        # The index goes in the label. Three `uses[].residual.layers` lines in
        # one trim record name the same block three times and say nothing about
        # which land use lost rows.
        for index, item in enumerate(node):
            found.extend(_lists_within(item, f"{path}[{index}]", depth + 1))
    return found


def _budget(
    payload: dict[str, Any], *, trim_inside: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Hold a response inside its size budget, and SAY SO when it did not fit.

    The enriched responses carry a block per layer, and a per-layer block grows
    with the drawing rather than with the question. Left alone that is how a
    tool ends up returning more than the agent can hold and losing the end of
    its own answer -- quietly, in the transport, with the visible part reading
    like the whole thing.

    Three properties, in order of importance:

    1. **Under budget, this function does nothing at all.** No key is added,
       no row is moved. Responses that ship correct today are byte-identical,
       which is what keeps the additive-only contract honest.
    2. **Over budget, every dropped row is named and counted.** `trims` says
       which block was shortened and by how much, and the shortened block
       carries its own `<key>_truncated_for_size` note beside it, so a reader
       looking at the block sees the limit without reading the footer.
    3. **What gets trimmed is chosen, not discovered.** `trim_inside` is an
       allowlist of roots; nothing outside it is eligible at any size.
       `blocks_all` in `describe_drawing` is complete on purpose -- a
       size-saving that reintroduces "there are no trees" is not a saving --
       and the `uses` list is the answer to the question, so it is recursed
       into and never shortened.

    Within the allowlist the BIGGEST block goes first, measured in characters
    rather than in rows: four hundred short rows cost less than a hundred long
    ones, and taking the expensive block first buys the characters back for the
    fewest facts dropped. Each pass halves each block once and stops the moment
    the response fits, so equally sized blocks come out equally trimmed rather
    than one surviving whole because another was flattened first.

    If everything eligible has been trimmed and the response is still too
    large, `still_over_budget` says so rather than the function pretending. A
    stated overflow can be narrowed by the next call; a silent one cannot.
    """
    if not isinstance(payload, dict) or payload.get("ok") is False:
        return payload

    size = _response_chars(payload)
    if size <= MAX_RESPONSE_CHARS:
        return payload

    found: list[tuple[dict[str, Any], str, str]] = []
    for root in trim_inside:
        node = payload.get(root)
        if isinstance(node, (dict, list)):
            found.extend(_lists_within(node, root))
    # Cost each block once, drop the ones too small to be the problem, and take
    # the most expensive first. The sort is stable, so `trim_inside` order
    # still decides between two blocks that cost the same.
    costed = [(_response_chars(h.get(k)), h, k, label) for h, k, label in found]
    costed = [c for c in costed if c[0] >= MIN_TRIMMABLE_CHARS]
    costed.sort(key=lambda c: -c[0])
    candidates = [(h, k, label) for _, h, k, label in costed]

    originals = [
        len(h[k]) if isinstance(h.get(k), list) else None for h, k, _ in candidates
    ]

    # One halving per block per pass, biggest first, stopping the moment the
    # response fits. Halving a block all the way to the floor before touching
    # the next one left two equally sized residuals showing 3 layers each while
    # a third kept 150 -- an asymmetry that reads as a finding about the
    # drawing and is nothing of the sort. `progress` ends the loop when every
    # eligible block is already at the floor.
    progress = True
    while size > MAX_RESPONSE_CHARS and progress:
        progress = False
        for holder, key, _ in candidates:
            if size <= MAX_RESPONSE_CHARS:
                break
            rows = holder.get(key)
            if not isinstance(rows, list) or len(rows) <= MIN_KEPT_ROWS:
                continue
            holder[key] = rows[: max(MIN_KEPT_ROWS, len(rows) // 2)]
            size = _response_chars(payload)
            progress = True

    trims: list[dict[str, Any]] = []
    for (holder, key, label), original in zip(candidates, originals):
        rows = holder.get(key)
        if original is None or not isinstance(rows, list) or len(rows) >= original:
            continue
        kept = len(rows)
        holder[f"{key}_truncated_for_size"] = (
            f"{original - kept} of {original} rows were dropped here to keep "
            "this response inside its size budget. This is a DISPLAY limit and "
            "not a measurement: the counts and totals beside it were computed "
            "over all of them."
        )
        trims.append({"block": label, "kept": kept, "dropped": original - kept})

    payload["response_budget"] = {
        "limit_chars": MAX_RESPONSE_CHARS,
        "size_chars": size,
        "trimmed": bool(trims),
        "trims": trims,
        "still_over_budget": size > MAX_RESPONSE_CHARS,
        "note": (
            "this response was over its size budget. Every row dropped is "
            "named and counted in `trims`, which is empty when nothing was "
            "eligible to drop; no count, total or unit was changed either way. "
            "If `still_over_budget` is true the response is over the limit "
            "anyway and the end of it may be lost in transit -- narrow the "
            "query (one layout, one layer) rather than trusting what arrived."
        ),
    }
    return payload


def main() -> None:
    transport = os.environ.get("CAD_MCP_TRANSPORT", "stdio")
    if transport == "http":
        # Deliberately NOT `PORT`. This project's .env is shared with another
        # service and already defines PORT=8092; reading it bound cad-mcp to
        # 8092 while compose published 8000, so the agent could not reach it.
        # Every variable this stack owns is CAD_-prefixed for that reason.
        port = int(os.environ.get("CAD_MCP_BIND_PORT", "8000"))
        log.info("starting cad-mcp on http port %d -> %s", port, CAD_API_URL)
        mcp.run(transport="http", host="0.0.0.0", port=port)
    else:
        log.info("starting cad-mcp on stdio -> %s", CAD_API_URL)
        mcp.run()


@mcp.tool
def scheduled_area(
    drawing_id: str,
    layout: str,
    number: int,
    label_layer: str | None = None,
) -> dict[str, Any]:
    """What the drawing's own table says a numbered thing measures, and
    whether its outline agrees.

    Use this whenever a question names a numbered thing -- "what is the area of
    plot 2043", "how big is unit 17" -- because this drawing states the answer
    as well as drawing it, and an answer backed by both is worth more than one
    computed alone. On the reference drawing plot 2043 is 300 in the table and
    299.9999992 by measurement.

    Say BOTH numbers when they agree, and say so plainly when they do not: a
    disagreement is a defect in the drawing, not a rounding detail to smooth
    over. The `evidence` on the row is `stated` when the table speaks, and it
    stops there rather than climbing to `corroborated`, because coordinates
    check a figure but never utter one.

    **Reach for this FIRST, before `search_text`.** Measured on 24 August 2026:
    asked how large plot 2043 is, an agent opened with `search_text`, then
    `get_entity`, `land_use_summary`, `join_labels`, `spatial_query` and
    `measure` -- six calls and eighty seconds -- and concluded it "cannot
    definitively determine the exact size", about a plot this tool answers in
    one call.

    NOT the tool for a total, a mean or a count -- that is `measure` or
    `stats`. NOT the tool for a thing named rather than numbered -- that is
    `search_text` or `join_labels`.
    """
    return _get(
        f"/drawings/{drawing_id}/schedule/{number}",
        {"layout": layout, "label_layer": label_layer},
    )


@mcp.tool
def schedule_check(
    drawing_id: str,
    layout: str,
    tolerance: float = 0.02,
    limit: int = 50,
    only_mismatches: bool = True,
) -> dict[str, Any]:
    """Where the drawing disagrees with itself: scheduled area against drawn
    area, for everything at once.

    Use this for "is this drawing consistent", "which plots are wrong", "check
    the areas", and for any quality question about the drawing as a whole.
    Measured on the reference drawing: 2,443 of 2,449 scheduled plots have an
    outline of their own, 2,383 agree within two percent, and 60 do not.

    Report the count first and the worst rows after it. Do not present the
    disagreements as an error in the analysis -- they are the finding. Each row
    names the polygon, so the viewer can mark them.

    Each row carries the gap twice: `difference_ratio` (0.43 means 43 %) and
    `difference_percent` (43.2 means 43.2 %). **Put `difference_percent` under
    a column headed %, and never `difference_ratio`.** Measured on 24 August
    2026: a table headed "%" was filled with the ratio, and a plot off by
    4,321 % was reported as off by 43 %.

    NOT the tool for one item -- that is `scheduled_area`. NOT the tool for
    what a plot is FOR -- that is `land_use_summary`.
    """
    return _get(
        f"/drawings/{drawing_id}/schedule/check",
        {
            "layout": layout,
            "tolerance": tolerance,
            "limit": min(limit, 500),
            "only_mismatches": only_mismatches,
        },
    )


@mcp.tool
def embedded_documents(drawing_id: str) -> dict[str, Any]:
    """Documents carried INSIDE the drawing that nothing draws.

    A DXF can hold whole files -- a workbook, a Word page, a picture pasted
    from Paint -- and no renderer shows them, so they are invisible to the
    viewer and to every text search. Use this when a question asks what a code
    or a colour MEANS, when the answer would otherwise have to be inferred, or
    when someone asks what else is in the file.

    It matters here specifically: the reference drawing carries its land-use
    key this way, as a picture stating which plot numbers are commercial,
    which are educational, and that the rest are residential villas. That was
    inferred by elimination for weeks while the drawing said it outright.

    Each row reports what the payload IS -- `bmp`, `xlsx`, `pdf`, `wmf` -- read
    from the bytes rather than assumed, and a row with no `file` is an object
    whose content is linked rather than stored. Point the reader at the handle;
    the viewer can open it.
    """
    return _get(f"/drawings/{drawing_id}/embedded")


if __name__ == "__main__":
    main()
