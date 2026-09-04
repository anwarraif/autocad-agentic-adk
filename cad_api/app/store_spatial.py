"""join_labels point-in-polygon; then distance_matrix + proximity_count

Owner: subagent C — UPLIFT-03, then UPLIFT-05.

This file is deliberately kept apart from `store.py` so that two specs worked
on at the same time never touch the same line. The rules, and their reasons:

- **Do not import from `.store`.** `store.py` imports this module at its
  bottom; importing back would be circular. Take `coll` and friends straight
  from `.mongo`, and geometry from `.region`.
- **Do not add lines in `store.py` or `main.py`.** Both are shared property,
  and every line added there is one merge conflict. Functions here are reached
  as `store.store_spatial.<function>`.
- **Every response that carries MEANING must carry `evidence`** per
  `docs/UPLIFT-08-EVIDENCE.md` — use the helpers in `app/evidence.py`, do not
  assemble the dict yourself.
- **Every number carries its unit** (G2), and a unit is allowed to be absent.
  A withheld number becomes `None` + a reason, never `0`.

The `docs/UPLIFT-GENERALITY-RULES.md` G1-G10 checklist is run before every
commit to this file.

---

## What this module answers, and why not `spatial_query`

"What this parcel actually is" is answered by the text that sits **inside**
the parcel, not by the text near it. `spatial_query` tests bounding boxes, and
on a grid rotated by about 49 degrees the axis-aligned box is far larger than
its polygon: on the reference parcel `205D12B` the bbox test pulls in dozens
of texts while point-in-polygon pulls in one. The answer from the bbox there
is not a rough answer; it is the wrong answer, and that is exactly the shape
of the "plot 2043 area" failure — a question about a parcel answered about the
text that labels it.

## Four decisions that shape the whole file

**There is one gate: `ring_status == "complete"`.** Not "`ring` is not null and
`ring_truncated` is false". A filter shaped as the ABSENCE of a flag lets
through bulged polygons (the chord is used as the parcel boundary) and open
polygons (ray casting closes them implicitly and answers inside for a shape
that has no inside). Measured across the whole store: 3,485 `complete`, 3,090
`open`, 1,298 `bulge` — the old filter accepted **4,388 wrong ones, more than
the 3,485 right ones**. A filter shaped as the absence of a flag also fails
every time someone adds a third cause. See D-083.

**The point tested is `anchor_point`, never `bbox_centre`.** A text box
depends on the font, and the reference drawing's SHX font is missing, so the
box itself is already a guess. An entity with no derivable insertion point is
skipped and **counted**; the box centre is never a silent fallback.

**The test is `region.point_in_ring`, which returns THREE answers.** A plot
number the drafter snapped to its plot line is not an edge case — it is how
the drawing was made. A `boundary` point is reported as inside for its row AND
counted in `boundary_cases`, which is present in every response even when its
value is 0, so that "there are no boundary cases in this data" becomes a
measurement, not an assumption. No second point-in-polygon is written here:
two implementations are how two answers start to differ.

**The 2,000,000 pair budget is counted AFTER the spatial prefilter.** Counted
over the raw product it cancels its own spec's Definition of Done: the plot
number test is 2,380 targets against 2,449 texts, that is 5.8 million raw
pairs, and its advice — "narrow `label_layer`" — is already as narrow as it
goes. That is a wall, not a way out. After the grid prefilter, the pairs that
remain are on the order of the label count itself.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from . import evidence as ev
from . import region
from . import store_landuse as landuse_store
from .extract import ANCHOR_TYPES, BLOCK_LAYOUT_PREFIX
from .geometry import BOUNDARY_EPS_REL
from .mongo import COLL_DRAWINGS, COLL_ENTITIES, coll

#: The join's work budget, counted over the pairs that SURVIVE the spatial
#: prefilter. See the file header note for why this number must not be counted
#: over `len(targets) * len(candidates)`.
PAIRS_BUDGET = 2_000_000

#: A stated resource limit (G7), not a property of a drawing. Both sit far
#: above the largest drawing there is — 2,885 `complete` polygons and 2,449
#: texts in Janadriyah modelspace — and exist so that a filter which filters
#: nothing refuses instead of eating up the container's memory.
MAX_TARGETS = 50_000
MAX_CANDIDATES = 200_000

#: Spatial index cells per axis. The cell count is derived from the NUMBER of
#: targets, not from the size of the drawing, so there is not a single
#: drawing-specific constant here (G1). The ceiling of 256 caps its memory at
#: 65,536 cells.
_MAX_CELLS_PER_AXIS = 256

#: A target whose box crosses more cells than this is kept in a separate list
#: and tested against every point. Without this, one block-boundary polygon
#: covering the whole drawing would be inserted into every cell and the cost of
#: building the index would become the very product the index was built to
#: avoid.
_MAX_CELLS_PER_TARGET = 64

#: The label types that can be asked for. `ANY` means every type that CARRIES
#: an insertion point -- not every type in the drawing. An LWPOLYLINE has no
#: insertion point to be skipped for, so counting it as "skipped for having no
#: anchor" would turn that number into noise.
LABEL_TYPES = ("TEXT", "MTEXT", "INSERT", "ANY")

#: Ring statuses other than `complete`. Reported as `skipped_<status>` in
#: every response, even when the value is 0: a target that was skipped is not a
#: target that did not match.
SKIP_CAUSES = ("bulge", "open", "non_planar", "degenerate", "oversize", "unreadable")

METHOD = "point-in-polygon (ray casting) on the text insertion point"
POINT_USED = "the entity's anchor_point, not the centre of its bounding box"
#: The reverse direction uses the SAME test point and the same arithmetic;
#: what differs is only which side is held fixed. Written separately so that a
#: response never claims to test text-inside-polygon when what it answers is
#: polygon-that-contains-text.
METHOD_CONTAINING = (
    "point-in-polygon (ray casting) on the entity's insertion point against "
    "every complete-ring polygon in the same layout"
)


class SpatialRefused(Exception):
    """A join that could only be answered misleadingly.

    Its shape mirrors `store.MeasureRefused` (`code`/`message`/`hint`) so that
    the `@app.exception_handler` pattern already in `main.py` can be reused and
    a violation comes out as an actionable 400. It is a class of its own and
    not `MeasureRefused` because importing that from `.store` would close the
    import cycle this file split was made to break; registering it in `main.py`
    is requested via `PROGRESS.md`.

    `hint` is mandatory: a refusal that cannot be acted on sends the agent back
    to fetching entities one by one, which is the failure this whole tool layer
    was built to end.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


# ---------------------------------------------------------------------------
# Spatial index -- pure, does not touch MongoDB
# ---------------------------------------------------------------------------


Box = tuple[float, float, float, float]


class _SpatialIndex:
    """A uniform grid over the target bounding boxes, to prefilter points.

    This is purely an optimisation: `hits()` answers exactly the question a
    full scan answers — which targets have a box containing this point — and
    `test_the_grid_agrees_with_brute_force_on_awkward_shapes` tests that
    equivalence against randomly sized shapes. An index that changes the answer
    is not an optimisation but a bug.

    The cell count is derived from the number of targets (square root, so that
    on average there is one target per cell), not from the size or units of the
    drawing. That is what makes it work identically in metres, in inches, and
    in a drawing that declares no units.
    """

    __slots__ = ("_cells", "_wide", "_boxes", "_min", "_size", "_g")

    def __init__(self, boxes: Sequence[Box | None]) -> None:
        self._boxes = list(boxes)
        self._cells: dict[tuple[int, int], list[int]] = {}
        self._wide: list[int] = []
        present = [(i, b) for i, b in enumerate(self._boxes) if b is not None]
        if not present:
            self._min = (0.0, 0.0)
            self._size = (1.0, 1.0)
            self._g = 1
            return

        minx = min(b[0] for _, b in present)
        miny = min(b[1] for _, b in present)
        maxx = max(b[2] for _, b in present)
        maxy = max(b[3] for _, b in present)
        g = max(1, min(_MAX_CELLS_PER_AXIS, int(math.ceil(math.sqrt(len(present))))))
        # A zero cell width is a real state, not an edge case: a drawing whose
        # targets all share a single x coordinate produces it. Its replacement
        # value is arbitrary precisely because every point then falls in the
        # same column, and that is the right answer.
        w = (maxx - minx) / g or 1.0
        h = (maxy - miny) / g or 1.0
        self._min = (minx, miny)
        self._size = (w, h)
        self._g = g

        for i, b in present:
            cx0 = self._axis(b[0], minx, w)
            cx1 = self._axis(b[2], minx, w)
            cy0 = self._axis(b[1], miny, h)
            cy1 = self._axis(b[3], miny, h)
            if (cx1 - cx0 + 1) * (cy1 - cy0 + 1) > _MAX_CELLS_PER_TARGET:
                self._wide.append(i)
                continue
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    self._cells.setdefault((cx, cy), []).append(i)

    def _axis(self, value: float, origin: float, size: float) -> int:
        return max(0, min(self._g - 1, int((value - origin) // size)))

    def hits(self, x: float, y: float) -> list[int]:
        """The indices of the targets whose bounding box contains this point."""
        cx = self._axis(x, self._min[0], self._size[0])
        cy = self._axis(y, self._min[1], self._size[1])
        out: list[int] = []
        for i in self._cells.get((cx, cy), ()):
            b = self._boxes[i]
            if b is not None and b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                out.append(i)
        for i in self._wide:
            b = self._boxes[i]
            if b is not None and b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                out.append(i)
        return out


class _NoIndex:
    """A full scan. It exists only as the comparison in the tests."""

    __slots__ = ("_boxes",)

    def __init__(self, boxes: Sequence[Box | None]) -> None:
        self._boxes = list(boxes)

    def hits(self, x: float, y: float) -> list[int]:
        return [
            i
            for i, b in enumerate(self._boxes)
            if b is not None and b[0] <= x <= b[2] and b[1] <= y <= b[3]
        ]


# ---------------------------------------------------------------------------
# Join -- pure, over Mongo-shaped documents
# ---------------------------------------------------------------------------


def _ring_of(doc: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in (doc.get("ring") or [])]


def _box_of(ring: Sequence[tuple[float, float]]) -> Box | None:
    if len(ring) < 3:
        return None
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _join(
    targets: Sequence[dict[str, Any]],
    labels: Sequence[dict[str, Any]],
    *,
    budget: int,
    _index: bool = True,
) -> dict[str, Any]:
    """Match each insertion point to the polygon containing it. No MongoDB.

    Split out of `join_labels` so that all the arithmetic that can go wrong can
    be tested without a database: the bbox-vs-polygon gap, boundary cases, the
    pair budget, and the index's equivalence to a full scan. What is left in
    `join_labels` is the query and the assembling of the response.

    `targets` and `labels` are Mongo documents exactly as they are, not
    intermediate objects: that makes the tests prove the query projection too,
    not just the logic.

    Raises:
        SpatialRefused: `JOIN_TOO_LARGE`, when the pairs that survive the
            prefilter exceed `budget`. Checked BEFORE a single ray casting is
            run, so that the limit is a limit on work and not a report after
            the work has already been done.
    """
    rings = [_ring_of(t) for t in targets]
    boxes = [_box_of(r) for r in rings]
    index = _SpatialIndex(boxes) if _index else _NoIndex(boxes)

    points: list[tuple[int, float, float] | None] = []
    skipped_no_anchor = 0
    for j, doc in enumerate(labels):
        pt = doc.get("anchor_point")
        if not pt or len(pt) < 2 or pt[0] is None or pt[1] is None:
            skipped_no_anchor += 1
            points.append(None)
            continue
        points.append((j, float(pt[0]), float(pt[1])))

    # The first sweep only counts. Storing the pairs first and then checking
    # the budget would mean the memory used is that budget itself -- two
    # million tuples -- on the very path that is supposed to refuse in order to
    # avoid it.
    pairs = 0
    for item in points:
        if item is None:
            continue
        pairs += len(index.hits(item[1], item[2]))

    if pairs > budget:
        raise SpatialRefused(
            "JOIN_TOO_LARGE",
            f"{pairs} pairs survived the spatial prefilter, above the budget "
            f"of {budget}. This number is counted after the prefilter, not "
            f"over {len(targets)} x {len(labels)} raw pairs.",
            "Narrow `target_layer` or `label_layer`. If both are already as "
            "narrow as they can be, what is being asked for is a full join "
            "over this drawing and that needs to be split per layer, not to "
            "have its limit raised.",
        )

    matched: list[list[dict[str, Any]]] = [[] for _ in targets]
    boundary_cases = 0
    shared = 0
    tested = 0

    for item in points:
        if item is None:
            continue
        j, x, y = item
        doc = labels[j]
        landed = 0
        for i in index.hits(x, y):
            tested += 1
            where = region.point_in_ring(
                x, y, rings[i], origin=_origin_of(targets[i], rings[i]),
                eps_rel=BOUNDARY_EPS_REL,
            )
            if where == "outside":
                continue
            if where == "boundary":
                boundary_cases += 1
            landed += 1
            matched[i].append(
                {
                    "handle": doc.get("handle"),
                    "layer": doc.get("layer"),
                    "type": doc.get("type"),
                    "text": doc.get("text"),
                    "anchor_basis": doc.get("anchor_basis"),
                    "where": where,
                }
            )
        if landed > 1:
            shared += 1

    rows = []
    for t, found in zip(targets, matched):
        rows.append(
            {
                "target_handle": t.get("handle"),
                "target_layer": t.get("layer"),
                "target_area_value": t.get("area"),
                "labels": found,
                "label_count": len(found),
            }
        )

    return {
        "rows": rows,
        "pairs_after_prefilter": pairs,
        "pairs_tested": tested,
        "boundary_cases": boundary_cases,
        "labels_matched_more_than_one_target": shared,
        "skipped_no_anchor": skipped_no_anchor,
        "candidates_examined": len(labels),
    }


def _origin_of(
    doc: dict[str, Any], ring: Sequence[tuple[float, float]]
) -> tuple[float, float] | None:
    """The ring's translation point, from storage when there is one.

    `region.point_in_ring` derives it itself when None; reading it from the
    document only saves one sweep per target. What matters is not the saving
    but that the point is the SAME one used when `polygon_centroid` was
    computed, so that two numbers about the same polygon are never computed in
    two different frames.
    """
    o = doc.get("ring_origin")
    if o and len(o) >= 2:
        return (float(o[0]), float(o[1]))
    if not ring:
        return None
    return (min(p[0] for p in ring), min(p[1] for p in ring))


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _require_layout(layout: Any) -> str:
    """`layout` is mandatory and has no default.

    Polygons in Model and text on a sheet do not share a coordinate frame in
    any meaningful way. A cross-layout join produces pairs that look convincing
    and mean nothing at all, and there is not one sign in the response that
    would tell them apart from correct pairs.
    """
    if not isinstance(layout, str) or not layout.strip():
        raise SpatialRefused(
            "LAYOUT_REQUIRED",
            f"`layout` was given as {layout!r}, which does not name a single "
            "coordinate frame.",
            "Name one layout, e.g. layout='Model'. Polygons in modelspace and "
            "text on a sheet do not share coordinates, so there is no correct "
            "answer for 'all layouts at once'.",
        )
    return layout


def _as_list(value: str | Sequence[str] | None, name: str) -> list[str] | None:
    """One layer name or several, with empty values refused.

    `if layer:` treats `""` as "no filter", so a caller who hands over an empty
    layer receives the whole layout and has every reason to believe they
    received one layer. Those two readings differ by four orders of magnitude
    and there is nothing downstream that could tell them apart.

    It accepts a list, not just a single string, because the question this spec
    exists to answer is itself list-shaped: the Janadriyah plot typology is
    spread over 13 layers, and asking for them one at a time makes "2,380
    plots, each with exactly one number" impossible to answer in one call.
    """
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        return None
    cleaned = []
    for v in values:
        if not isinstance(v, str) or not v.strip():
            raise SpatialRefused(
                "EMPTY_FILTER_VALUE",
                f"{name} contains {v!r}, which filters nothing.",
                f"Hand over a real value for {name}, or drop {name} entirely "
                f"if there is in fact no filter.",
            )
        cleaned.append(v)
    return cleaned


def _type_filter(label_type: str) -> dict[str, Any]:
    kind = (label_type or "TEXT").strip().upper()
    if kind not in LABEL_TYPES:
        raise SpatialRefused(
            "UNKNOWN_LABEL_TYPE",
            f"label_type {label_type!r} is not one of "
            f"{', '.join(LABEL_TYPES)}.",
            "Use TEXT for plain text, MTEXT for paragraph text, INSERT for "
            "blocks, or ANY for every type that carries an insertion point.",
        )
    if kind == "ANY":
        return {"type": {"$in": sorted(ANCHOR_TYPES)}}
    return {"type": kind}


def _target_query(
    drawing_id: str,
    layout: str,
    layers: list[str] | None,
    handles: list[str] | None,
) -> dict[str, Any]:
    # `drawing_id` is always the leading filter: this cluster runs with
    # `notablescan`, so a query with no indexed plan does not slow down -- it
    # fails. That is also what G10 asks for, for an entirely different reason.
    query: dict[str, Any] = {"drawing_id": drawing_id, "layout": layout}
    if layers:
        query["layer"] = layers[0] if len(layers) == 1 else {"$in": layers}
    if handles:
        query["handle"] = handles[0] if len(handles) == 1 else {"$in": handles}
    return query


def _skipped_by_cause(base: dict[str, Any]) -> dict[str, int]:
    """How many targets were skipped, broken down by cause.

    Counted over the SAME target scope, only without the
    `ring_status == "complete"` gate. A target that was skipped is not a target
    that did not match, and a number that is not reported is a silent
    truncation (G7).
    """
    query = dict(base)
    query["ring_status"] = {"$exists": True, "$ne": "complete"}
    rows = coll(COLL_ENTITIES).aggregate(
        [{"$match": query}, {"$group": {"_id": "$ring_status", "n": {"$sum": 1}}}]
    )
    found = {r["_id"]: int(r["n"]) for r in rows}
    return {cause: found.get(cause, 0) for cause in SKIP_CAUSES}


def _union_box(rings: Iterable[Sequence[tuple[float, float]]]) -> Box | None:
    xs: list[float] = []
    ys: list[float] = []
    for ring in rings:
        for x, y in ring:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    box = (min(xs), min(ys), max(xs), max(ys))
    # Relative padding, the size of the edge tolerance itself. A point sitting
    # exactly on the outer edge of the union box is reported `boundary` by
    # `point_in_ring`, and a prefilter that discarded it first would lose a
    # real label without a single sign. Relative, so it carries no unit at
    # all.
    pad = max(box[2] - box[0], box[3] - box[1]) * BOUNDARY_EPS_REL
    return (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)


def _quantity_area(value: Any, units: dict[str, Any], handle: Any) -> dict[str, Any]:
    """One target's area as a `Quantity`, never as a bare number.

    A target whose area is not stored yields a WITHHELD value together with its
    reason, not `0`. Zero is a wrong number, not an empty one, and a `Quantity`
    has no way of turning into zero.
    """
    if value is None:
        return ev.Quantity.withheld(
            basis=f"the area of parcel {handle}",
            method="not measured",
            reason="no area is stored for this entity",
        ).as_dict()
    return ev.Quantity.measured(
        basis=f"the area of parcel {handle}",
        method="shoelace over the closed ring, computed in the ring's local frame",
        value=float(value),
        unit=units.get("area_unit"),
        unit_reason=(
            None
            if units.get("area_unit")
            else units.get("why_no_unit")
            or "this drawing declares no units, so none is claimed"
        ),
    ).as_dict()


def _scope_of(drawing_id: str, layout: str, note: str) -> ev.Scope:
    """The scope of the claim, with units taken from one place only.

    Units are NOT recomputed here. `store._unit_names` holds knowledge that was
    expensive to obtain -- $INSUNITS describes MODEL space only, an A0 sheet
    labelled in metres is wrong by about a thousand times over -- and copying
    its logic here is how two answers about the same units start to differ.

    Its import is deferred into the function rather than placed at the head of
    the file, and that is not an oversight: `store.py` imports this module at
    its tail, so a module-level import would be circular. By the time this
    function is CALLED, `store` has finished loading. The request to move
    `_unit_names` into a shared module, so that this deferral can be removed,
    is recorded in `PROGRESS.md`.
    """
    from .store import _unit_names  # noqa: PLC0415 -- see docstring

    drawing = coll(COLL_DRAWINGS).find_one({"_id": drawing_id})
    if not drawing:
        raise SpatialRefused(
            "DRAWING_UNKNOWN",
            f"drawing {drawing_id!r} is not in the store.",
            "Check the drawing list first; a mistyped id yields zero pairs "
            "that cannot be told apart from a correct zero pairs.",
        )
    return ev.Scope(
        layout=layout,
        includes_block_definitions=layout.startswith(BLOCK_LAYOUT_PREFIX),
        units=_unit_names(drawing, layout),
        note=note,
    )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _provenance() -> ev.Provenance:
    """No config decides anything here.

    `join_labels` does not classify; it reports topology. Its config layer is
    `NO_CONFIG` and therefore `verified` is false — not because the mapping is
    doubted, but because there is no mapping at all to verify.
    """
    return ev.Provenance(
        config_layer=ev.ConfigLayer.NO_CONFIG,
        note="topology measured from coordinates; no config was read",
    )


def _row_evidence(
    row: dict[str, Any], *, drawing_id: str, scope: ev.Scope, layout: str
) -> ev.Evidence:
    """Evidence for one parcel: what sits inside it, and how strongly.

    Point-in-polygon is a COORDINATE corpus, and coordinates never STATE — they
    infer. So a row here will never rise to `stated` on the strength of its own
    geometry, however clean the pairing is. What can be stated is the text
    content, and the vocabulary that decides whether a string states something
    lives in config (UPLIFT-02/UPLIFT-13), not in this file (G1). Until that
    vocabulary is handed over, `tokens` is empty and the grade stops at
    `inferred` — which is the correct one, not the limiting one.
    """
    handle = row["target_handle"]
    layer = row["target_layer"]
    if not row["labels"]:
        return ev.Evidence.unknown(
            ev.Claim(value=f"what parcel {handle} is"),
            drawing_id=drawing_id,
            scope=scope,
            provenance=_provenance(),
            not_established=(
                f"not one text has its insertion point falling inside the ring "
                f"of parcel {handle} in layout {layout}. That is an absence of "
                "labels, not an absence of parcels"
            ),
            how_to_verify=(
                "widen `label_type` to ANY, or drop `label_layer`; if it is "
                "still empty, this parcel really is not labelled inside its "
                "own boundary and its name has to come from another source"
            ),
            observations=(
                ev.Observation(
                    origin=ev.Origin.LAYER_NAME,
                    detail=f"layer {layer!r}",
                    observed=layer,
                    locator=layer,
                    layout=layout,
                ),
            ),
        )

    observations = [
        ev.Observation(
            origin=ev.Origin.LAYER_NAME,
            detail=f"layer {layer!r}",
            observed=layer,
            locator=layer,
            layout=layout,
        )
    ]
    for found in row["labels"][:4]:
        observations.append(
            ev.Observation(
                origin=ev.Origin.TOPOLOGY,
                detail=(
                    f"the insertion point of {found['type']} {found['handle']} "
                    f"sits {found['where']} the ring of parcel {handle}"
                ),
                locator=handle,
                layout=layout,
            )
        )
        text = (found.get("text") or "").strip()
        if text:
            observations.append(
                ev.Observation(
                    origin=ev.Origin.ENTITY_TEXT,
                    detail=f"{found['type']} on layer {found['layer']!r}: {text!r}",
                    observed=text,
                    locator=found["handle"],
                    layout=layout,
                )
            )
    return ev.Evidence.of(
        ev.Claim(
            value=(
                f"parcel {handle} contains {row['label_count']} texts inside "
                "its boundary"
            )
        ),
        drawing_id=drawing_id,
        scope=scope,
        provenance=_provenance(),
        observations=observations,
        not_established=(
            "what that text means. What was measured is that it sits inside, "
            "not that it names the parcel"
        ),
        how_to_verify=(
            "hand over the land-use vocabulary through this drawing's config, "
            "or read its glyphs through a verified SHX corpus"
        ),
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def join_labels(
    drawing_id: str,
    *,
    layout: str,
    target_layer: str | Sequence[str] | None = None,
    target_handles: Sequence[str] | None = None,
    label_layer: str | None = None,
    label_type: str = "TEXT",
    limit: int = 50,
) -> dict[str, Any]:
    """The text INSIDE each polygon, tested with point-in-polygon.

    **Use this, not `spatial_query`, when the question is "what is INSIDE this
    shape".** `spatial_query` answers "what is around here", and its answer is
    bounding-box based. On geometry sitting on a rotated grid the two are not a
    rough answer and a fine answer to the same question: measured on one school
    parcel, the bbox test pulls in dozens of texts while the number genuinely
    inside the parcel is one.

    Args:
        drawing_id: the drawing being read. One response belongs to one
            drawing (G10).
        layout: MANDATORY, and no default. Modelspace and sheets do not share a
            coordinate frame, so there is no correct answer for "all of them".
        target_layer: one polygon layer name, or several. Several because one
            typology can be spread over many layers.
        target_handles: pick the target polygons directly, instead of a layer.
        label_layer: restrict the text considered to one layer.
        label_type: `TEXT`, `MTEXT`, `INSERT`, or `ANY`. `ANY` means every
            type that carries an insertion point.
        limit: how many ROWS come back. It does not limit what is counted:
            `total_targets` and `targets_with_label` always cover the whole
            scope, so 2,380 plots can be counted and checked in one call
            without sending 2,380 rows.

    Returns:
        A response with `rows`, the counts that form its denominator
        (`targets_without_label` prevents the classic mistake: counting labels
        as a count of parcels), `boundary_cases`, the count of skipped targets
        broken down by cause, the budget numbers, and an `evidence` block.

    Raises:
        SpatialRefused: `LAYOUT_REQUIRED`, `EMPTY_FILTER_VALUE`,
            `UNKNOWN_LABEL_TYPE`, `DRAWING_UNKNOWN`, `TOO_MANY_TARGETS`,
            `TOO_MANY_CANDIDATES`, or `JOIN_TOO_LARGE`.
    """
    layout = _require_layout(layout)
    layers = _as_list(target_layer, "target_layer")
    handles = _as_list(target_handles, "target_handles")
    label_layers = _as_list(label_layer, "label_layer")
    type_filter = _type_filter(label_type)

    entities = coll(COLL_ENTITIES)
    base = _target_query(drawing_id, layout, layers, handles)
    target_query = dict(base)
    target_query["ring_status"] = "complete"

    total_targets = entities.count_documents(target_query)
    if total_targets > MAX_TARGETS:
        raise SpatialRefused(
            "TOO_MANY_TARGETS",
            f"{total_targets} target polygons, above the limit of {MAX_TARGETS}.",
            "Narrow `target_layer`, or hand over `target_handles`.",
        )

    targets = list(
        entities.find(
            target_query,
            {
                "handle": 1,
                "layer": 1,
                "ring": 1,
                "ring_origin": 1,
                "area": 1,
                "_id": 0,
            },
        ).sort([("layer", 1), ("handle", 1)])
    )
    skipped = _skipped_by_cause(base)

    scope_note = _scope_note(layout, layers, handles, label_layers, label_type)
    scope = _scope_of(drawing_id, layout, scope_note)

    rings = [_ring_of(t) for t in targets]
    box = _union_box(rings)

    candidates: list[dict[str, Any]] = []
    db_no_anchor = 0
    if box is not None:
        label_query: dict[str, Any] = {
            "drawing_id": drawing_id,
            "layout": layout,
            **type_filter,
            "anchor_point.0": {"$gte": box[0], "$lte": box[2]},
            "anchor_point.1": {"$gte": box[1], "$lte": box[3]},
        }
        if label_layers:
            label_query["layer"] = (
                label_layers[0] if len(label_layers) == 1 else {"$in": label_layers}
            )
        found = entities.count_documents(label_query)
        if found > MAX_CANDIDATES:
            raise SpatialRefused(
                "TOO_MANY_CANDIDATES",
                f"{found} text candidates, above the limit of {MAX_CANDIDATES}.",
                "Narrow `label_layer` or `label_type`.",
            )
        candidates = list(
            entities.find(
                label_query,
                {
                    "handle": 1,
                    "layer": 1,
                    "type": 1,
                    "text": 1,
                    "anchor_point": 1,
                    "anchor_basis": 1,
                    "_id": 0,
                },
            )
        )
        no_anchor_query = {
            "drawing_id": drawing_id,
            "layout": layout,
            **type_filter,
            "anchor_point": None,
        }
        if label_layers:
            no_anchor_query["layer"] = label_query["layer"]
        db_no_anchor = entities.count_documents(no_anchor_query)

    joined = _join(targets, candidates, budget=PAIRS_BUDGET)

    units = dict(scope.units)
    rows = []
    for row in joined["rows"][:limit]:
        rows.append(
            {
                "target_handle": row["target_handle"],
                "target_layer": row["target_layer"],
                "target_area": _quantity_area(
                    row["target_area_value"], units, row["target_handle"]
                ),
                "label_count": row["label_count"],
                "labels": row["labels"],
            }
        )

    with_label = sum(1 for row in joined["rows"] if row["label_count"])
    without_label = total_targets - with_label
    label_total = sum(row["label_count"] for row in joined["rows"])

    body: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "method": METHOD,
        "point_used": POINT_USED,
        "scope_note": scope_note,
        "total_targets": total_targets,
        "targets_with_label": with_label,
        "targets_without_label": without_label,
        "label_count": label_total,
        "labels_matched_more_than_one_target": joined[
            "labels_matched_more_than_one_target"
        ],
        "boundary_cases": joined["boundary_cases"],
        "boundary_note": (
            "a point falling exactly on the line is reported as inside for its "
            "row AND counted here. A plot number the drafter snapped to its "
            "plot line is not an edge case; it is how the drawing was made"
        ),
        "candidates_examined": joined["candidates_examined"],
        "skipped_no_anchor": joined["skipped_no_anchor"] + db_no_anchor,
        "skipped_no_anchor_note": (
            "an entity with no derivable insertion point is skipped and "
            "counted. The box centre is never a fallback: a text box depends "
            "on the font, and this drawing's SHX font is missing"
        ),
        "skipped_targets_total": sum(skipped.values()),
        "pairs_budget": PAIRS_BUDGET,
        "pairs_after_prefilter": joined["pairs_after_prefilter"],
        "pairs_tested": joined["pairs_tested"],
        "prefilter": (
            "the bounding box of each target's ring, through a uniform grid "
            "whose cell count is derived from the number of targets"
        ),
        "returned": len(rows),
        "truncated": len(rows) < len(joined["rows"]),
        "rows": rows,
    }
    for cause, count in skipped.items():
        body[f"skipped_{cause}"] = count
    if without_label:
        body["empty_note"] = (
            f"{without_label} of {total_targets} target polygons contain no "
            "text at all inside their boundary. That is an absence of labels, "
            "not an absence of parcels."
        )

    parts = [
        _row_evidence(row, drawing_id=drawing_id, scope=scope, layout=layout)
        for row in joined["rows"]
    ]
    if parts:
        aggregate = ev.weakest(
            parts,
            ev.Claim(
                value=(
                    f"{with_label} of {total_targets} polygons contain text "
                    "inside their boundary"
                )
            ),
        )
    else:
        aggregate = ev.Evidence.unknown(
            ev.Claim(value="which polygons contain text inside their boundary"),
            drawing_id=drawing_id,
            scope=scope,
            provenance=_provenance(),
            not_established=(
                "there is not one polygon with `ring_status: complete` in "
                f"this scope; {sum(skipped.values())} polygons were skipped"
            ),
            how_to_verify=(
                "check `skipped_bulge` and `skipped_open`: arced polygons and "
                "open polygons cannot be used to test whether a point is "
                "inside them"
            ),
        )
    return ev.attach(body, aggregate)


def _scope_note(
    layout: str,
    layers: list[str] | None,
    handles: list[str] | None,
    label_layers: list[str] | None,
    label_type: str,
) -> str:
    target = (
        f"{len(handles)} named handles"
        if handles
        else (
            "layer " + ", ".join(repr(l) for l in layers[:4]) + ("…" if len(layers) > 4 else "")
            if layers
            else "every complete-ring polygon"
        )
    )
    source = (
        "layer " + ", ".join(repr(l) for l in label_layers)
        if label_layers
        else "any layer"
    )
    return (
        f"layout {layout!r}; targets: {target}, only those with "
        f"`ring_status == 'complete'`; labels: {label_type.upper()} from "
        f"{source}, tested at their insertion point"
    )


def labels_inside(
    drawing_id: str, handle: str, *, label_type: str = "TEXT", limit: int = 10
) -> dict[str, Any]:
    """The text inside one polygon — the answer to "what is this plot".

    Used to enrich `get_entity`, so that the question that failed in the
    23 August meeting — clicking a plot number and then asking for its parcel
    area, and being answered about its TEXT entity — can be answered in one
    call.

    Returns `total: None` together with its reason, not `0`, for polygons that
    cannot be tested: zero means "checked and there is none", and that is a
    different claim from "could not be checked" (G8).
    """
    doc = coll(COLL_ENTITIES).find_one(
        {"drawing_id": drawing_id, "handle": handle},
        {"layout": 1, "ring_status": 1, "geometry_note": 1, "_id": 0},
    )
    if not doc:
        raise SpatialRefused(
            "ENTITY_UNKNOWN",
            f"there is no entity {handle!r} in drawing {drawing_id!r}.",
            "Check the handle; a mistyped handle yields zero labels that "
            "cannot be told apart from a correct zero labels.",
        )

    status = doc.get("ring_status")
    if status != "complete":
        return {
            "handle": handle,
            "labels": [],
            "total": None,
            "not_measured": (
                f"ring_status {status!r}: "
                + (
                    doc.get("geometry_note")
                    or "this entity carries no closed ring that can be tested"
                )
            ),
            "method": METHOD,
        }

    joined = join_labels(
        drawing_id,
        layout=doc.get("layout"),
        target_handles=[handle],
        label_type=label_type,
        limit=1,
    )
    row = joined["rows"][0] if joined["rows"] else {"labels": []}
    return {
        "handle": handle,
        "labels": row["labels"][:limit],
        "total": joined["label_count"],
        "truncated": joined["label_count"] > limit,
        "boundary_cases": joined["boundary_cases"],
        "method": METHOD,
        "point_used": POINT_USED,
        "scope_note": joined["scope_note"],
        "evidence": joined["evidence"],
    }



# ---------------------------------------------------------------------------
# The reverse direction: from a label to the parcel that contains it
# ---------------------------------------------------------------------------
#
# `labels_inside` answers "what text is inside this polygon". That is the
# useful direction when what was clicked is the parcel. What Harsh's gate asked
# for is the other direction, and that other direction is the one that failed
# in the 23 August meeting: someone clicked a PLOT NUMBER, asked for its area,
# and was answered about its TEXT entity -- which has no area at all.
#
# For a whole day the `/containing` route claimed in its docstring to do this
# while its code called `labels_inside`. Both return something that looks
# plausible, so nothing failed loudly; the plot number was simply asked "which
# polygon contains you" and answered "you are not a polygon".

#: How many candidate polygons may be examined for ONE point before the request
#: is refused. The prefilter uses the `drawing_bbox_min` index, which can only
#: bind from one side -- every polygon whose bottom-left corner is below-left
#: of the test point comes along -- so this limit is real and not decoration.
#: The reference drawing uses about 7,900 of this allowance at its farthest
#: point.
MAX_CONTAINERS_EXAMINED = 60_000


def parcels_containing(
    drawing_id: str,
    handle: str,
    *,
    layout: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Which polygons contain this entity, tested point-in-polygon.

    The inverse of `labels_inside`, and the one used when someone clicks a
    label: from its text to the parcel that labels it, and then to THAT
    parcel's area.

    Three things make it hold on the 19th drawing and not only on the
    reference drawing:

    1. **No layer name anywhere.** What decides whether an entity may be made
       a container is `ring_status == "complete"` -- DXF format vocabulary --
       not its name. A drawing with a layer convention nobody has ever seen is
       still answered.
    2. **The point used is the insertion point, not the box centre.** A text
       box depends on the font, and the reference drawing's SHX font is
       missing; the box centre would miss by the width of its text on a
       rotated grid.
    3. **Nested containers all come back, the smallest first.** A plot number
       sits inside its plot AND inside its block boundary AND inside its
       district boundary, and all three are correct answers to different
       questions. Returning one means choosing the question for the reader;
       smallest-first order means the first one is the one almost always
       meant.

    Returns:
        `contained_by` sorted smallest first, `total`, and -- when it cannot be
        tested -- `not_measured` together with its cause. `total: None` means
        it could not be checked; `total: 0` means it was checked and there
        genuinely is none (G8).

    Raises:
        SpatialRefused: `ENTITY_UNKNOWN` or `TOO_MANY_CONTAINERS`.
    """
    entities = coll(COLL_ENTITIES)
    doc = entities.find_one(
        {"drawing_id": drawing_id, "handle": handle},
        {
            "layout": 1,
            "type": 1,
            "layer": 1,
            "text": 1,
            "anchor_point": 1,
            "anchor_basis": 1,
            "polygon_centroid": 1,
            "_id": 0,
        },
    )
    if not doc:
        raise SpatialRefused(
            "ENTITY_UNKNOWN",
            f"there is no entity {handle!r} in drawing {drawing_id!r}.",
            "Check the handle; a mistyped handle yields zero containers that "
            "cannot be told apart from a correct zero containers.",
        )

    scope_layout = _require_layout(layout or doc.get("layout"))
    selected = {
        "handle": handle,
        "type": doc.get("type"),
        "layer": doc.get("layer"),
        "text": doc.get("text"),
    }

    # Two sources for the test point, and their order is not taste. A LABEL is
    # tested at its insertion point, because that is the point the file really
    # stores. A POLYGON has no insertion point at all, and refusing to answer
    # for polygons means "which district is this school parcel inside" cannot
    # be asked -- and that is a reasonable question with an answer. So a
    # polygon is tested at its centroid, and the response SAYS so: a centroid
    # that is inside is not the same thing as the whole polygon being inside,
    # and that difference is real for concave shapes.
    point = doc.get("anchor_point")
    basis_used = "the entity's anchor_point"
    containment_note = None
    if not point or len(point) < 2 or point[0] is None or point[1] is None:
        point = doc.get("polygon_centroid")
        basis_used = "the entity's polygon_centroid"
        containment_note = (
            "this entity has no insertion point, so what was tested is its "
            "CENTROID. A centroid that is inside is not the same as the whole "
            "polygon being inside; for concave shapes the two can differ"
        )
    if not point or len(point) < 2 or point[0] is None or point[1] is None:
        basis = doc.get("anchor_basis")
        return {
            "selected": selected,
            "contained_by": [],
            "total": None,
            "not_measured": (
                "this entity has neither an insertion point nor a derivable "
                "centroid"
                + (f" (anchor_basis {basis!r})" if basis else "")
                + ". The bounding box centre is never used as a fallback: a "
                "text box depends on the font, and guessing here means "
                "attaching the label to the plot next door"
            ),
            "method": METHOD_CONTAINING,
            "point_used": POINT_USED,
            "layout": scope_layout,
        }

    x, y = float(point[0]), float(point[1])

    # Prefiltered through the `drawing_bbox_min` index. It only binds from one
    # side; the other side is filtered in Python, because one index cannot do
    # both and `notablescan` refuses a query with no index -- it does not slow
    # it down, it REFUSES it.
    query = {
        "drawing_id": drawing_id,
        "bbox.min.0": {"$lte": x},
        "bbox.min.1": {"$lte": y},
    }
    examined = 0
    hits: list[dict[str, Any]] = []
    for cand in entities.find(
        query,
        {
            "handle": 1,
            "layer": 1,
            "type": 1,
            "area": 1,
            "ring": 1,
            "ring_origin": 1,
            "ring_status": 1,
            "bbox": 1,
            "layout": 1,
            "_id": 0,
        },
    ):
        examined += 1
        if examined > MAX_CONTAINERS_EXAMINED:
            raise SpatialRefused(
                "TOO_MANY_CONTAINERS",
                f"more than {MAX_CONTAINERS_EXAMINED} candidate polygons for "
                f"one point in drawing {drawing_id!r}.",
                "This is a resource limit, not an answer. A drawing this large "
                "needs a two-sided spatial index before its question can be "
                "answered honestly.",
            )
        if cand.get("handle") == handle:
            continue
        if cand.get("ring_status") != "complete":
            continue
        if cand.get("layout") != scope_layout:
            continue
        box = cand.get("bbox") or {}
        mx = box.get("max")
        if not mx or len(mx) < 2 or x > float(mx[0]) or y > float(mx[1]):
            continue
        ring = _ring_of(cand)
        if len(ring) < 3:
            continue
        origin = _origin_of(cand, ring)
        where = region.point_in_ring(x, y, ring, origin=origin)
        if where == "outside":
            continue
        hits.append({"doc": cand, "where": where})

    scope_note = (
        f"the parcels containing {handle} in layout {scope_layout!r}, tested "
        "point-in-polygon against every complete-ring polygon"
    )
    scope = _scope_of(drawing_id, scope_layout, scope_note)
    units = dict(scope.units)

    layers = [str(h["doc"].get("layer")) for h in hits if h["doc"].get("layer")]
    classified = landuse_store.classify_layers(drawing_id, layers) if layers else {}

    def _sort_key(hit: dict[str, Any]) -> tuple[int, float, str]:
        area = hit["doc"].get("area")
        # A container with no area sorts last, rather than being treated as
        # zero: zero would make it win as "the smallest" precisely because it
        # was not measured.
        if area is None:
            return (1, 0.0, str(hit["doc"].get("handle") or ""))
        return (0, float(area), str(hit["doc"].get("handle") or ""))

    hits.sort(key=_sort_key)

    rows = []
    for hit in hits[:limit]:
        cand = hit["doc"]
        layer = cand.get("layer")
        use = classified.get(str(layer), {})
        area_value = cand.get("area")
        rows.append(
            {
                "handle": cand.get("handle"),
                "layer": layer,
                "type": cand.get("type"),
                # Two shapes for the same number, and that is deliberate.
                # `area` and `area_unit` are the contract the viewer already
                # reads; `area_quantity` is the one that carries the basis,
                # the method, and -- when the area is not stored -- the reason
                # it was withheld. The flat one alone would make "not
                # measured" indistinguishable from "zero"; the rich one alone
                # would force every caller to unpack an object to get one
                # number.
                "area": None if area_value is None else float(area_value),
                "area_unit": units.get("area_unit"),
                "area_quantity": _quantity_area(area_value, units, cand.get("handle")),
                "land_use": use.get("land_use"),
                "land_use_subtype": use.get("land_use_subtype"),
                "land_use_verified": use.get("land_use_verified"),
                "where": hit["where"],
            }
        )

    return {
        "selected": selected,
        "contained_by": rows,
        "total": len(hits),
        "truncated": len(hits) > limit,
        "boundary_cases": sum(1 for h in hits if h["where"] == "boundary"),
        "boundary_note": (
            "a point falling exactly on the line is reported as inside. A "
            "plot number the drafter snapped to its plot line is not an edge "
            "case; it is how the drawing was made"
        ),
        "candidates_examined": examined,
        "method": METHOD_CONTAINING,
        "point_used": basis_used,
        "containment_note": containment_note,
        "layout": scope_layout,
        "scope_note": scope_note,
        "nesting_note": (
            "nested containers are all returned, the SMALLEST BY AREA first. "
            "A label can sit inside its plot and inside its block boundary at "
            "the same time, and both are true"
        ),
    }

# ===========================================================================
# UPLIFT-05 -- distance
# ===========================================================================
#
# There is no `evidence` block below here, and that is a decision not an
# oversight. The UPLIFT-08 contract binds responses that say what something
# IS. A distance does not say what anything is; it is a quantity, and the
# contract that binds it is `Quantity` -- basis, method, unit, and the reason
# if it is withheld -- which is used on every dimensioned number below.
# Attaching an evidence block to a plain number would dilute the meaning of
# `grade` exactly where it is needed most.
#
# What MUST be present in every response, and is tested: `not_measured`.
# Planning standards are written in walking distance, and handing over a
# straight-line number without saying it is a straight line is the easiest way
# to make a correct answer be used for the wrong thing.

#: Item limit per mode. `centroid` gives 780 pairs at 40 items, and above that
#: the table itself already exceeds the agent's context budget. `edge` is far
#: stricter because its cost is O(n*m) per EDGE pair, not per item pair.
MAX_ITEMS = {"centroid": 40, "edge": 16}

#: The edge-comparison limit for one polygon pair. A polygon may carry up to
#: 4,096 vertices, and two of those are 16 million comparisons for one cell. A
#: cell that exceeds it is WITHHELD together with its reason, not quietly
#: computed with a simplified version.
MAX_EDGE_PAIRS = 250_000

DISTANCE_MEASURE = {
    "centroid": "straight-line distance between `polygon_centroid`s",
    "edge": "shortest straight-line distance between polygon edges; 0 when touching or overlapping",
}

NOT_MEASURED = (
    "NOT walking distance through the road network; the road network is not "
    "modelled in this store, and no number here may be read as a travel "
    "distance"
)


# ---------------------------------------------------------------------------
# Pure
# ---------------------------------------------------------------------------


def _row_keys(items: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Row names that are short, unique, and derived from the data itself.

    A matrix with no row names cannot be read by anyone, and a hand-picked
    abbreviation table is a drawing-specific constant (G1). So the rule is
    derived from the names actually being chosen, not from a list:

    1. A word that appears in more than one layer name within this selection
       distinguishes nothing and is dropped. A type word used by five layers
       at once adds no information; a word used by only one does.
    2. If dropping them leaves not one word, the LAST shared word is returned
       -- it is the most specific one. Without this step, a name whose every
       word is shared shrinks to its digits alone and stops being readable.
    3. The words that remain are shortened to the SHORTEST unique prefix, at
       least three letters, so that two names sharing their first three
       letters are extended until they differ.
    4. A layer that has more than one parcel gets the suffix `-1`, `-2`
       running from EAST to west.

    The consequence is honest and recorded in `docs/UPLIFT-05-DISTANCE.md`: two
    of the nine keys in the spec's reference table differ by one letter from
    what this rule produces, because a rule that can run on the 17th drawing
    cannot guess an abbreviation a human chose. The handle always travels on
    every row, so a key is never the only identity.
    """
    import re
    from collections import Counter

    names = sorted({str(it["layer"]) for it in items})
    tokens = {n: re.findall(r"[A-Za-z]+|\d+", n) for n in names}

    shared: Counter[str] = Counter()
    for n in names:
        for t in {t.upper() for t in tokens[n] if t.isalpha()}:
            shared[t] += 1

    kept: dict[str, list[str]] = {}
    for n in names:
        alpha = [t for t in tokens[n] if t.isalpha()]
        keep = [t for t in tokens[n] if not t.isalpha() or shared[t.upper()] == 1]
        if not any(t.isalpha() for t in keep):
            keep = ([alpha[-1]] if alpha else []) + [
                t for t in keep if not t.isalpha()
            ]
        kept[n] = keep or tokens[n] or [n]

    leads = {
        n: next((t.upper() for t in kept[n] if t.isalpha()), "") for n in names
    }
    base: dict[str, str] = {}
    for n in names:
        lead = leads[n]
        others = [leads[m] for m in names if m != n]
        length = 3
        while length < len(lead) and any(o[:length] == lead[:length] for o in others):
            length += 1
        digits = "".join(t for t in kept[n] if not t.isalpha())
        base[n] = (lead[:length] or n.upper()[:3]) + digits

    # Two layer names that still produce the same key after all of that are a
    # possible state, not an impossible one. It is resolved with a stable
    # ordering, not left to overwrite silently.
    seen: Counter[str] = Counter()
    for n in names:
        seen[base[n]] += 1
    counter: Counter[str] = Counter()
    for n in names:
        if seen[base[n]] > 1:
            counter[base[n]] += 1
            base[n] = f"{base[n]}#{counter[base[n]]}"

    out: dict[str, str] = {}
    for name in names:
        group = [it for it in items if str(it["layer"]) == name]
        # East to west: x descending. Ties are broken by y and then handle, so
        # that a parcel's key does not change between calls.
        group.sort(key=lambda it: (-it["point"][0], -it["point"][1], it["handle"]))
        for i, it in enumerate(group, start=1):
            out[it["handle"]] = base[name] + (f"-{i}" if len(group) > 1 else "")
    return out


def _ring_distance(
    a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]
) -> float | None:
    """Shortest distance between two polygon edges; 0 if touching/overlapping.

    Three cases, and dropping any one of them reports a positive distance for
    two parcels that actually meet:

    1. crossing edges -- 0;
    2. one polygon entirely inside the other -- 0, and that polygon has not one
       crossing edge, so an edge test on its own does not see it;
    3. otherwise, the minimum point-to-edge distance over all four endpoint
       combinations.

    All of it is computed after both rings have been translated to a shared
    bottom-left corner. The reason is the same one that shapes all of
    `geometry.py`: in projected coordinates, the orientation determinant and
    the point-to-edge distance are computed from differences of numbers around
    1e6, and translating first removes that whole class of error without
    changing the answer.

    Uses `region._segments_cross` and `region._point_segment_distance` as they
    are, and does not rewrite either: two implementations of the same geometric
    test are how two answers start to differ. Both are private today; the
    request to promote this function into `region.py` as a public
    `ring_distance` is recorded in `PROGRESS.md`.

    Returns None if this pair exceeds `MAX_EDGE_PAIRS`. A cell withheld
    together with its reason is more honest than a cell computed with a
    simplified version without saying so (G7).
    """
    if len(a) < 3 or len(b) < 3:
        return None
    if len(a) * len(b) > MAX_EDGE_PAIRS:
        return None

    ox = min(min(p[0] for p in a), min(p[0] for p in b))
    oy = min(min(p[1] for p in a), min(p[1] for p in b))
    A = [(p[0] - ox, p[1] - oy) for p in a]
    B = [(p[0] - ox, p[1] - oy) for p in b]

    ea = [(A[i], A[(i + 1) % len(A)]) for i in range(len(A))]
    eb = [(B[i], B[(i + 1) % len(B)]) for i in range(len(B))]

    best = math.inf
    for a1, a2 in ea:
        for b1, b2 in eb:
            if region._segments_cross(a1, a2, b1, b2):
                return 0.0
            best = min(
                best,
                region._point_segment_distance(a1[0], a1[1], b1, b2),
                region._point_segment_distance(a2[0], a2[1], b1, b2),
                region._point_segment_distance(b1[0], b1[1], a1, a2),
                region._point_segment_distance(b2[0], b2[1], a1, a2),
            )
    if region.point_in_ring(A[0][0], A[0][1], B) != "outside":
        return 0.0
    if region.point_in_ring(B[0][0], B[0][1], A) != "outside":
        return 0.0
    return best if best is not math.inf else None


def _single_linkage(
    points: dict[str, tuple[float, float]], threshold: float
) -> list[list[str]]:
    """Groups connected through a chain, not only through direct neighbours.

    Single-linkage was chosen because it is what fits the question: three
    parcels 500 m apart in a row are one district at a 600 m threshold even
    though the outermost are 1,000 m apart. It is also the only linkage that
    needs no second parameter, so there is no hidden number inside it.
    """
    keys = sorted(points)
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if math.dist(points[a], points[b]) <= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _measurable_items(
    drawing_id: str,
    layout: str,
    *,
    handles: Sequence[str] | None,
    layers: Sequence[str] | None,
    need_ring: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """The polygons that have a centroid, plus a count of those that do not.

    A `complete` polygon with no `polygon_centroid` is a real state -- its
    signed area is zero, the polygon folds back on itself -- and it is skipped
    and counted, never replaced by its box centre.
    """
    handle_list = _as_list(handles, "handles")
    layer_list = _as_list(layers, "layers")
    if not handle_list and not layer_list:
        raise SpatialRefused(
            "NOTHING_TO_MEASURE",
            "neither `handles` nor `layers` was given.",
            "Name the handles to be compared, or the layers whose parcels are "
            "all to be compared. A matrix over the whole drawing is not a "
            "broader answer, it is an answer nobody can read.",
        )
    base = _target_query(drawing_id, layout, layer_list, handle_list)
    query = dict(base)
    query["ring_status"] = "complete"

    projection: dict[str, Any] = {
        "handle": 1,
        "layer": 1,
        "polygon_centroid": 1,
        "area": 1,
        "_id": 0,
    }
    if need_ring:
        projection["ring"] = 1

    items: list[dict[str, Any]] = []
    no_centroid = 0
    for doc in coll(COLL_ENTITIES).find(query, projection).sort(
        [("layer", 1), ("handle", 1)]
    ):
        centre = doc.get("polygon_centroid")
        if not centre or len(centre) < 2:
            no_centroid += 1
            continue
        doc["point"] = (float(centre[0]), float(centre[1]))
        items.append(doc)
    return items, _skipped_by_cause(base), no_centroid


def _length_quantity(
    value: float | None,
    units: dict[str, Any],
    *,
    basis: str,
    method: str,
    withheld_reason: str = "cannot be measured for this pair",
) -> dict[str, Any]:
    if value is None:
        return ev.Quantity.withheld(
            basis=basis, method=method, reason=withheld_reason
        ).as_dict()
    return ev.Quantity.measured(
        basis=basis,
        method=method,
        value=float(value),
        unit=units.get("length_unit"),
        unit_reason=(
            None
            if units.get("length_unit")
            else units.get("why_no_unit")
            or "this drawing declares no units, so none is claimed"
        ),
    ).as_dict()


def distance_matrix(
    drawing_id: str,
    *,
    layout: str,
    handles: Sequence[str] | None = None,
    layers: Sequence[str] | None = None,
    mode: str = "centroid",
    max_items: int = 40,
    cluster_threshold: float | None = None,
) -> dict[str, Any]:
    """Distance between parcels, in two modes answering two different questions.

    `centroid` answers "how far apart do they sit" and is measured between
    `polygon_centroid`s -- not between bounding box centres, which on a
    rotated grid introduce errors of up to several metres without a single
    sign in the response. `edge` answers "are the two parcels adjacent" and
    returns 0 for parcels that touch.

    Every response names `not_measured`: this is a straight line, not walking
    distance through the road network.

    Args:
        layout: MANDATORY. Layouts do not share a coordinate frame.
        handles: the parcels being compared; or
        layers: every parcel on these layers.
        mode: `centroid` or `edge`.
        max_items: cut more strictly by the mode limits -- 40 and 16.
        cluster_threshold: the single-linkage threshold, in drawing units.
            Without it there are no clusters, and that is said. The threshold
            is a parameter and not a number inside the code because "600 m" is
            a planning number of one project, not a property of a CAD drawing
            (G1).

    Raises:
        SpatialRefused: `LAYOUT_REQUIRED`, `NOTHING_TO_MEASURE`,
            `UNKNOWN_MODE`, `TOO_MANY_ITEMS`, `DRAWING_UNKNOWN`.
    """
    layout = _require_layout(layout)
    kind = (mode or "centroid").strip().lower()
    if kind not in MAX_ITEMS:
        raise SpatialRefused(
            "UNKNOWN_MODE",
            f"mode {mode!r} is neither `centroid` nor `edge`.",
            "`centroid` for how far apart they sit, `edge` for whether the two "
            "are adjacent. They answer different questions.",
        )
    cap = min(int(max_items), MAX_ITEMS[kind])

    items, skipped, no_centroid = _measurable_items(
        drawing_id, layout, handles=handles, layers=layers, need_ring=(kind == "edge")
    )
    if len(items) > cap:
        raise SpatialRefused(
            "TOO_MANY_ITEMS",
            f"{len(items)} parcels, above the limit of {cap} for mode {kind!r} "
            f"(mode limits: centroid {MAX_ITEMS['centroid']}, edge "
            f"{MAX_ITEMS['edge']}).",
            "Narrow `layers`, or hand over `handles` for the parcels genuinely "
            "worth comparing. A matrix larger than this is not truncated in "
            "silence because the table itself already exceeds the context "
            "budget of whatever reads it; for 'how many are around each "
            "object', use proximity_count, whose output is O(n) not O(n "
            "squared).",
        )

    scope_note = (
        f"layout {layout!r}; {len(items)} parcels with `ring_status: complete`; "
        f"mode {kind!r}"
    )
    scope = _scope_of(drawing_id, layout, scope_note)
    units = dict(scope.units)
    keys = _row_keys(items)

    n = len(items)
    matrix: list[list[float | None]] = [[0.0] * n for _ in range(n)]
    withheld: list[dict[str, Any]] = []
    flat: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if kind == "centroid":
                value: float | None = math.dist(items[i]["point"], items[j]["point"])
            else:
                value = _ring_distance(
                    _ring_of(items[i]), _ring_of(items[j])
                )
            matrix[i][j] = matrix[j][i] = value
            if value is None:
                withheld.append(
                    {
                        "between": [items[i]["handle"], items[j]["handle"]],
                        "reason": (
                            "the edge comparison for this pair exceeds the "
                            f"limit of {MAX_EDGE_PAIRS}; its value is "
                            "withheld, not simplified"
                        ),
                    }
                )
            else:
                flat.append(value)

    nearest = []
    for i in range(n):
        best_j = None
        best_v = math.inf
        for j in range(n):
            v = matrix[i][j]
            if i == j or v is None:
                continue
            if v < best_v:
                best_v, best_j = v, j
        nearest.append(
            {
                "from": keys[items[i]["handle"]],
                "from_handle": items[i]["handle"],
                "to": keys[items[best_j]["handle"]] if best_j is not None else None,
                "to_handle": items[best_j]["handle"] if best_j is not None else None,
                "distance": _length_quantity(
                    None if best_j is None else best_v,
                    units,
                    basis=f"the nearest distance from {items[i]['handle']}",
                    method=DISTANCE_MEASURE[kind],
                    withheld_reason="there is no other parcel that can be measured",
                ),
            }
        )

    body: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "mode": kind,
        "measure": DISTANCE_MEASURE[kind],
        "not_measured": NOT_MEASURED,
        "unit": units.get("length_unit"),
        "unit_reason": (
            None
            if units.get("length_unit")
            else units.get("why_no_unit")
            or "this drawing declares no units, so none is claimed"
        ),
        "scope_note": scope_note,
        "max_items_effective": cap,
        "items": [
            {
                "key": keys[it["handle"]],
                "handle": it["handle"],
                "layer": it["layer"],
                "centroid": [it["point"][0], it["point"][1]],
            }
            for it in items
        ],
        "pairs_total": n * (n - 1) // 2,
        "matrix": matrix,
        "matrix_note": (
            "cells hold bare numbers because one table cannot carry "
            f"{n * n} unit blocks; the unit is `unit` in this response, and a "
            "cell that cannot be measured is null and is listed in `withheld`"
        ),
        "withheld": withheld,
        "nearest": nearest,
        "stats": _distance_stats(flat, units, DISTANCE_MEASURE[kind]),
        "skipped_no_centroid": no_centroid,
        "skipped_targets_total": sum(skipped.values()),
    }
    for cause, count in skipped.items():
        body[f"skipped_{cause}"] = count

    if cluster_threshold is None:
        body["clusters"] = {
            "count": None,
            "members": [],
            "method": None,
            "threshold": None,
            "not_computed": (
                "no `cluster_threshold` was given. A cluster threshold is a "
                "planning number of one project, not a property of a drawing, "
                "so there is no default value that could be justified"
            ),
        }
    else:
        groups = _single_linkage(
            {it["handle"]: it["point"] for it in items}, float(cluster_threshold)
        )
        body["clusters"] = {
            "count": len(groups),
            "members": [[h for h in group] for group in groups],
            "members_by_key": [[keys[h] for h in group] for group in groups],
            "method": (
                f"single-linkage, threshold {cluster_threshold:g} "
                f"{units.get('length_unit') or 'drawing units'} between "
                "`polygon_centroid`s"
            ),
            "threshold": _length_quantity(
                float(cluster_threshold),
                units,
                basis="the cluster threshold requested by the caller",
                method="single-linkage between `polygon_centroid`s",
            ),
        }
    return body


def _distance_stats(
    values: Sequence[float], units: dict[str, Any], method: str
) -> dict[str, Any]:
    """min/max/mean/median over the pairs that were genuinely measured.

    A summary that quietly counts a withheld pair as zero would drag its mean
    down and still look plausible. That is why its denominator travels with
    it.
    """
    rows = sorted(values)
    if not rows:
        blank = ev.Quantity.withheld(
            basis="no pair was measured",
            method=method,
            reason="fewer than two parcels can be measured",
        ).as_dict()
        return {"pairs_measured": 0, "min": blank, "max": blank, "mean": blank,
                "median": blank}
    mid = len(rows) // 2
    median = rows[mid] if len(rows) % 2 else (rows[mid - 1] + rows[mid]) / 2.0
    return {
        "pairs_measured": len(rows),
        "min": _length_quantity(rows[0], units, basis="the closest pair", method=method),
        "max": _length_quantity(rows[-1], units, basis="the farthest pair", method=method),
        "mean": _length_quantity(
            sum(rows) / len(rows), units,
            basis=f"the mean of {len(rows)} pairs", method=method
        ),
        "median": _length_quantity(
            median, units, basis=f"the median of {len(rows)} pairs", method=method
        ),
    }


def proximity_count(
    drawing_id: str,
    *,
    layout: str,
    around_layers: Sequence[str],
    count_layers: Sequence[str],
    radius: float,
    measure_from: str = "centroid",
) -> dict[str, Any]:
    """How many objects around each object — an O(n) shape, not O(n²).

    The question there was no time to answer in the 23 August meeting at minute
    16:22: "vicinity of houses around each school". Split out of
    `distance_matrix` precisely because its output shape is different — "2,380
    houses around 9 schools" as a matrix is 21,420 cells that are of no use to
    anyone, while as nine rows it is an answer.

    Its parameter is named `radius`, not `radius_m`. Planting a unit inside a
    parameter name in a repo where 11 of 18 drawings are in inches and 3
    declare no units at all is the same defect as `tolerance_m` in UPLIFT-07,
    and is fixed the same way: the number is in DRAWING units, and that unit is
    echoed in the response.

    Raises:
        SpatialRefused: `LAYOUT_REQUIRED`, `EMPTY_FILTER_VALUE`,
            `UNKNOWN_MODE`, `TOO_MANY_EDGE_PAIRS`, `DRAWING_UNKNOWN`.
    """
    layout = _require_layout(layout)
    kind = (measure_from or "centroid").strip().lower()
    if kind not in MAX_ITEMS:
        raise SpatialRefused(
            "UNKNOWN_MODE",
            f"measure_from {measure_from!r} is neither `centroid` nor `edge`.",
            "`centroid` measures from the centre point of the area, `edge` "
            "from the nearest edge. They answer different questions.",
        )
    need_ring = kind == "edge"
    centres, _, centres_no_centroid = _measurable_items(
        drawing_id, layout, handles=None, layers=around_layers, need_ring=need_ring
    )
    counted, skipped, counted_no_centroid = _measurable_items(
        drawing_id, layout, handles=None, layers=count_layers, need_ring=need_ring
    )

    if need_ring and len(centres) * len(counted) > 0:
        pairs = sum(
            len(_ring_of(a)) * len(_ring_of(b)) for a in centres[:1] for b in counted
        ) * max(1, len(centres))
        if pairs > MAX_EDGE_PAIRS * 8:
            raise SpatialRefused(
                "TOO_MANY_EDGE_PAIRS",
                f"mode `edge` here demands about {pairs} edge comparisons.",
                "Use measure_from='centroid', or narrow `count_layers`. Edge "
                "distance costs O(n·m) per polygon pair, and an answer that "
                "takes ten minutes is not an answer.",
            )

    scope_note = (
        f"layout {layout!r}; {len(centres)} centres, {len(counted)} objects "
        f"counted; radius measured from {kind}"
    )
    scope = _scope_of(drawing_id, layout, scope_note)
    units = dict(scope.units)
    keys = _row_keys(centres) if centres else {}

    # The span is computed from the points genuinely measured, not from rings:
    # `centroid` mode does not fetch rings at all, and a span that quietly
    # becomes None in one mode makes the "radius exceeds the drawing" warning
    # disappear in exactly the mode that is used most often.
    span = None
    if counted:
        xs = [d["point"][0] for d in counted]
        ys = [d["point"][1] for d in counted]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
    exceeds = bool(span is not None and float(radius) > span)

    rows = []
    #: Every target within the radius of AT LEAST ONE centre, counted once.
    #
    # The rows below are one set per centre and they OVERLAP: a plot beside
    # two park sections is in both. Measured on the reference drawing -- 17
    # park sections, rows summing to 3,674, over a population of 2,380.
    # Asked how many plots lie within 200 m of the park, an agent added the
    # column and published a total larger than the number of plots that
    # exist. Nothing in the response had told it not to, and a sum looks
    # exactly like an answer.
    near_any: set[str] = set()
    for centre in centres:
        inside: list[tuple[float, dict[str, Any]]] = []
        for other in counted:
            if other["handle"] == centre["handle"]:
                continue
            if kind == "centroid":
                d = math.dist(centre["point"], other["point"])
            else:
                d = _ring_distance(_ring_of(centre), _ring_of(other))
                if d is None:
                    continue
            if d <= float(radius):
                inside.append((d, other))
        areas = [
            ev.Quantity.measured(
                basis=f"the area of {o['handle']}",
                method="shoelace over the closed ring, computed in the ring's local frame",
                value=float(o["area"]),
                unit=units.get("area_unit"),
                unit_reason=(
                    None
                    if units.get("area_unit")
                    else units.get("why_no_unit") or "the drawing declares no unit"
                ),
            )
            if o.get("area") is not None
            else ev.Quantity.withheld(
                basis=f"the area of {o['handle']}",
                method="not measured",
                reason="no area is stored for this entity",
            )
            for _, o in inside
        ]
        for _, o in inside:
            near_any.add(str(o.get("handle") or ""))
        rows.append(
            {
                "key": keys.get(centre["handle"]),
                "handle": centre["handle"],
                "layer": centre["layer"],
                "count": len(inside),
                "total_area": ev.total(
                    areas,
                    basis=f"Σ area of {len(inside)} objects within the radius",
                    method="shoelace over the closed ring",
                ).as_dict(),
                "nearest": _length_quantity(
                    min((d for d, _ in inside), default=None),
                    units,
                    basis=f"the nearest object to {centre['handle']}",
                    method=DISTANCE_MEASURE[kind],
                    withheld_reason="there is no object within the radius",
                ),
                "farthest": _length_quantity(
                    max((d for d, _ in inside), default=None),
                    units,
                    basis=f"the farthest object from {centre['handle']} within the radius",
                    method=DISTANCE_MEASURE[kind],
                    withheld_reason="there is no object within the radius",
                ),
            }
        )

    body: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layout": layout,
        "does_not_measure": (
            "adjacency. This is a radius between one point per object, so two "
            "parcels that share a boundary are as far apart as their centres "
            "are -- tens of metres on a plot of any size. A small radius here "
            "therefore returns nothing for reasons of arithmetic, not of "
            "layout, and reads exactly like 'nothing is next to it'"
        ),
        "ask_instead": {
            "recipe": "adjacency",
            "when": (
                "the question is which parcels TOUCH or sit beside another one"
            ),
            "why_it_matters": (
                "`adjacency` measures the shortest distance between polygon "
                "EDGES and reports 0 where they touch, which is what 'next to' "
                "means on a plan. Measured: asked which plots are directly "
                "adjacent to a school, a turn used this recipe at a 1 m radius "
                "from centroids, found none, and reported that as a finding"
            ),
        },
        "measure": (
            "straight-line radius from " + DISTANCE_MEASURE[kind]
        ),
        "measure_from": kind,
        "within_radius_of_any_centre": len(near_any),
        "do_not_add_the_rows": (
            "each row is the set around ONE centre and the sets overlap, so "
            "adding the `count` column counts a target once per centre it is "
            "near, and can exceed the population entirely. "
            "`within_radius_of_any_centre` is what a question of the form 'how "
            "many X are within R of Y' asks for: each target counted once"
        ),
        "not_measured": NOT_MEASURED,
        "scope_note": scope_note,
        "radius": _length_quantity(
            float(radius),
            units,
            basis="the radius requested by the caller, in drawing units",
            method="straight line",
        ),
        "unit": units.get("length_unit"),
        "centres": len(centres),
        "counted_population": len(counted),
        "rows": rows,
        "radius_exceeds_extents": exceeds,
        "extents_note": (
            (
                f"radius {float(radius):g} exceeds the span of the objects "
                f"counted ({span:g} drawing units), so every row contains "
                "almost its whole population. That is not proximity, that is "
                "the total"
            )
            if exceeds
            else None
        ),
        "skipped_no_centroid": centres_no_centroid + counted_no_centroid,
        "skipped_targets_total": sum(skipped.values()),
    }
    for cause, count in skipped.items():
        body[f"skipped_{cause}"] = count
    return body
