"""What changed between two revisions of a drawing — DOSSIER Phase 7.

Owner: the Phase 7 (revision) lane. Read with `docs/DOSSIER-05-07-PLAN.md`
(Phase 7), `docs/DOSSIER-01-DESIGN.md` (what a Dossier holds) and
`docs/UPLIFT-GENERALITY-RULES.md` (G1–G10, binding).

Two things live in this file and they are deliberately separated:

* **The engine** — `diff_dossiers()` and the helpers under it. Pure functions
  over two already-fetched Dossier documents. No pymongo, no ezdxf, no
  fastapi, exactly like the four Phase 1 lane modules, which is why the whole
  of it is testable with fixtures and why `scripts/dossier_diff.py` can be a
  thin command-line wrapper rather than a second implementation.
* **The recipe** — `revision_diff`, registered at the bottom. It rides on
  `run_analysis` like every other recipe, so the MCP surface does not grow by
  a tool (`docs/DOSSIER-05-07-PLAN.md`, Phase 7).

Why a Dossier diff and not a DXF diff
-------------------------------------
`drawing_id` is a content hash. A new version of a file therefore arrives as
its own drawing with its own Dossier, and nothing has to be versioned by hand
— which is what makes "what changed" a comparison of two documents rather than
a new engine. The consequence is the honest limit at the centre of this file:

    A Dossier is an aggregate per (layer x layout) bucket. This diff can
    therefore see anything that moves an aggregate — an entity count, a type
    count, a role, a Σ length or Σ area, an anomaly, a layout, a file fact —
    and it can see NOTHING ELSE. A pure translation changes no aggregate, so
    a moved entity is invisible to it.

That limit is published in `cannot_see` on every response rather than left for
a reader to discover. An invisible change reported as "no change" would be the
same confident half-truth the DOSSIER campaign exists to end.

Units, and the arithmetic that is refused (G2)
----------------------------------------------
Eleven of the drawings in this store are in inches, one in millimetres and
three declare no unit at all. Subtracting a figure in inches from one in
metres produces a number, and that number means nothing. So:

* if the two drawings declare **different units**, every measure delta in the
  response is `compared: false` with the disagreement named. Counts are still
  compared — a count of objects carries no unit, so it is safe;
* even when the two files agree, a **bucket** whose own `measure.unit` differs
  between the revisions is refused on its own terms. A layer that moved from
  model space into a block definition loses its unit, and a delta across that
  move is a delta across two different questions;
* a measure of a different **kind** — an area against a length — is refused
  for the same reason;
* a measure whose value is **absent on either side** is refused rather than
  treated as zero (G8). "Nothing was measurable here" and "this measures zero"
  are different facts and this file never lets them collapse.

Everything refused says so in `not_compared`, with the reason and, where one
exists, the remedy.

G1 — no drawing lives in this file
----------------------------------
Not one layer name, layout name, typology code or unit is written here. The
engine reads whatever two documents it is handed; the CLI's fixture builder in
`scripts/dossier_diff.py` derives its targets from the file it is given rather
than naming them. That is what makes this work on the 17th drawing, from a
contractor whose layer convention nobody has seen yet.
"""

from __future__ import annotations

from typing import Any, Final, Iterable, Mapping, Sequence

from ..mongo import COLL_DOSSIERS, COLL_DRAWINGS, coll
from .registry import (
    Param,
    Recipe,
    RecipeRefused,
    register,
)

# --- Stated caps (G7) --------------------------------------------------------
#
# Every cap below is published in `truncation` and every one of them counts
# what it dropped. A diff that silently stops listing is a diff that reports
# "nothing else changed" about a file it stopped reading.

#: Buckets listed in each of `layers_added`, `layers_removed`, `layers_changed`.
MAX_BUCKET_ROWS: Final[int] = 100

#: DXF types listed inside one changed bucket.
MAX_TYPE_ROWS: Final[int] = 20

#: File-level facts listed in `file_facts_changed`.
MAX_FACT_ROWS: Final[int] = 40

#: Layouts listed in `layouts_changed`.
MAX_LAYOUT_ROWS: Final[int] = 40

#: Candidate predecessors listed when the rule is ambiguous. The count is
#: always exact; only the listing is cut.
MAX_CANDIDATES: Final[int] = 20

#: Characters of republished prose (a `role_basis`, a `verdict`). The
#: reference drawing's road-layer caveat alone is ~2,400 characters.
MAX_TEXT: Final[int] = 320

#: Bumped only when the shape of the response changes so that an old and a new
#: report cannot be told apart without reading them.
DIFF_VERSION: Final[int] = 1


# =============================================================================
# The predecessor rule — stated once, in one place, and never guessed
# =============================================================================

#: The whole rule, published verbatim in every response that uses it. It is a
#: constant rather than a sentence written at each call site so that the rule a
#: reader is told is provably the rule that ran.
PREDECESSOR_RULE: Final[str] = (
    "A candidate predecessor is a drawing already in the store that carries "
    "the SAME drawing name and a DIFFERENT content hash. The name is matched "
    "exactly, character for character, against `original_filename` (falling "
    "back to `name`). Nothing else is used: not the ingest time, not the file "
    "size, not the directory the file came from, not how similar the two "
    "drawings look. Exactly one candidate is treated as the predecessor. Zero "
    "or several are REPORTED as such and listed; ambiguity is never resolved "
    "by silently picking one."
)

#: What the rule cannot do, published beside it. A rule whose limits are not
#: stated is a rule people will trust past its edge.
PREDECESSOR_LIMITS: Final[tuple[str, ...]] = (
    "A file RENAMED between revisions has no candidate at all under this "
    "rule, and that reads exactly like a file the store has never seen "
    "before. Name the other drawing explicitly with `other_drawing` when that "
    "happens.",
    "A candidate is a candidate, not a proven predecessor. Two ingests of the "
    "same drawing through different routes — a DXF and the same file "
    "converted from its DWG — carry one name and two hashes, so they satisfy "
    "this rule while not being revisions of each other at all. What actually "
    "differs is in the diff; read it before calling the pair a revision.",
    "Order is not established. The rule finds the other drawing with this "
    "name; it does not prove which of the two came first. `ingested_at` is "
    "when this store saw the file, not when the drafter saved it.",
)


def _name_of(drawing: Mapping[str, Any]) -> str | None:
    """The drawing NAME the predecessor rule matches on.

    `original_filename` is what the extractor records from the file it read
    (`path.name`). `name` is the fallback for a document that predates it.
    Absence is returned as absence: a drawing with no name cannot be matched,
    and matching it against every other nameless drawing would invent a family.
    """
    for key in ("original_filename", "name"):
        value = drawing.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def find_predecessors(
    drawing: Mapping[str, Any], others: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply `PREDECESSOR_RULE` to one drawing against a list of drawings.

    Pure: the caller fetches. That is not tidiness — it is what lets the rule
    be tested against a store that does not exist, including the two cases
    that matter most (none, and several).

    Returns a block that ALWAYS carries `rule`, `candidates`, `candidate_count`
    and `why`. `predecessor` is a drawing id only when exactly one candidate
    was found; in every other case it is `None` and `why` says which case it
    was. There is no branch that picks a winner out of several.
    """
    drawing_id = str(drawing.get("_id") or drawing.get("drawing_id") or "")
    name = _name_of(drawing)

    if name is None:
        return {
            "rule": PREDECESSOR_RULE,
            "rule_limits": list(PREDECESSOR_LIMITS),
            "drawing_id": drawing_id,
            "drawing_name": None,
            "candidates": [],
            "candidate_count": 0,
            "predecessor": None,
            "ambiguous": False,
            "why": (
                "this drawing document carries no `original_filename` and no "
                "`name`, so the rule has nothing to match on. No predecessor "
                "was looked for. This is NOT a statement that the drawing is "
                "the first of its kind."
            ),
        }

    candidates: list[dict[str, Any]] = []
    for other in others:
        other_id = str(other.get("_id") or other.get("drawing_id") or "")
        if not other_id or other_id == drawing_id:
            continue
        if _name_of(other) != name:
            continue
        candidates.append(
            {
                "drawing_id": other_id,
                "drawing_name": name,
                "entity_count": other.get("entity_count"),
                "ingested_at": _plain(other.get("ingested_at")),
                "source_path": other.get("source_path"),
            }
        )
    # Deterministic: the same store must produce the same listing on every
    # call, or two sessions comparing their answers are comparing orderings.
    candidates.sort(key=lambda row: str(row["drawing_id"]))

    if len(candidates) == 1:
        why = (
            f"exactly one other drawing in the store is named {name!r}, so it "
            "is the predecessor under the rule above. It is a CANDIDATE "
            "established by name, not a proven revision: what actually "
            "differs is in the diff."
        )
    elif not candidates:
        why = (
            f"no other drawing in the store is named {name!r}. Either this is "
            "the first version of it the store has seen, or the file was "
            "renamed between revisions — the rule matches names exactly and "
            "cannot tell those two apart. Name the other drawing explicitly "
            "with `other_drawing` if you know it."
        )
    else:
        why = (
            f"{len(candidates)} other drawings in the store are named "
            f"{name!r}, so the rule does not identify one predecessor. They "
            "are listed above. This is reported, not resolved: picking one of "
            "them here would produce a confident diff against a drawing "
            "nobody chose."
        )

    listed = candidates[:MAX_CANDIDATES]
    return {
        "rule": PREDECESSOR_RULE,
        "rule_limits": list(PREDECESSOR_LIMITS),
        "drawing_id": drawing_id,
        "drawing_name": name,
        "candidates": listed,
        "candidate_count": len(candidates),
        "candidates_dropped": max(0, len(candidates) - len(listed)),
        "candidates_cap": MAX_CANDIDATES,
        "predecessor": candidates[0]["drawing_id"] if len(candidates) == 1 else None,
        "ambiguous": len(candidates) > 1,
        "why": why,
    }


def predecessor_for(drawing_id: str) -> dict[str, Any]:
    """`find_predecessors` with the store read for you.

    Every drawing document is fetched and the match is made in Python rather
    than with a `{"original_filename": name}` filter, and that is a
    requirement rather than a preference: this cluster runs with
    `notablescan`, `autocad_drawings` carries no index on that field, and a
    filtered query without an indexed plan FAILS outright rather than merely
    running slowly. An empty-filter read of a collection holding a score of
    documents is what `store.list_drawings` already does.
    """
    projection = {
        "_id": 1,
        "name": 1,
        "original_filename": 1,
        "entity_count": 1,
        "ingested_at": 1,
        "source_path": 1,
    }
    rows = list(coll(COLL_DRAWINGS).find({}, projection))
    mine = next((r for r in rows if str(r.get("_id")) == str(drawing_id)), None)
    if mine is None:
        return {
            "rule": PREDECESSOR_RULE,
            "rule_limits": list(PREDECESSOR_LIMITS),
            "drawing_id": str(drawing_id),
            "drawing_name": None,
            "candidates": [],
            "candidate_count": 0,
            "predecessor": None,
            "ambiguous": False,
            "why": (
                f"there is no drawing with id {drawing_id!r} in the store, so "
                "no name could be read and no predecessor looked for."
            ),
        }
    return find_predecessors(mine, rows)


# =============================================================================
# Small readers, all of them tolerant of a document that is not what we expect
# =============================================================================


def _plain(value: Any) -> Any:
    """A JSON-able value, without changing its meaning."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _clip(text: Any, *, limit: int = MAX_TEXT) -> tuple[str | None, bool]:
    """`(text, was_truncated)`. `None` stays `None`: absence is not a string."""
    if text is None:
        return None, False
    value = str(text)
    if not value.strip():
        return None, False
    if len(value) <= limit:
        return value, False
    return value[: limit - 1].rstrip() + "…", True


def _number(value: Any) -> float | None:
    """A real number, or None. `bool` is an `int` and is not a measurement."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _round(value: float) -> float:
    """Six decimals, matching what the Dossier itself stores.

    Rounding here rather than leaving the raw subtraction is not cosmetic:
    83962.565009 - 83962.565009 is not always exactly 0.0 in binary floating
    point, and a delta of 7.3e-12 published as a change would make this diff
    report a difference in a file where nothing moved.
    """
    return round(value, 6)


def _blocks(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = document.get("layers")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [b for b in raw if isinstance(b, Mapping)]


def _bucket_index(
    document: Mapping[str, Any], layout: str | None
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Buckets keyed by `(layer, layout)` — the Dossier's own partition key.

    `layout=None` means every layout, block definitions among them. That is
    the default on purpose: geometry inside a block definition is still
    geometry in the file, and a revision that changed only a block would
    otherwise be reported as a revision that changed nothing.
    """
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for block in _blocks(document):
        key = (str(block.get("layer") or ""), str(block.get("layout") or ""))
        if layout is not None and key[1] != layout:
            continue
        out[key] = block
    return out


def _units_of(document: Mapping[str, Any]) -> dict[str, Any]:
    """What the FILE declares. Never substituted for a bucket's own unit."""
    facts = document.get("file_facts")
    units = facts.get("units") if isinstance(facts, Mapping) else None
    if not isinstance(units, Mapping):
        return {
            "units_name": None,
            "units_code": None,
            "declared_in_file": None,
            "why": (
                "this Dossier records no unit facts, so whether the file "
                "declares a unit is unknown. No unit may be assumed from it."
            ),
        }
    return {
        "units_name": units.get("units_name"),
        "units_code": units.get("units_code"),
        "declared_in_file": units.get("declared_in_file"),
    }


def _anomaly_kinds(block: Mapping[str, Any]) -> list[str]:
    """The anomaly kinds of one bucket, from any shape a profiler wrote.

    Written tolerantly on purpose, and the reason is a shipped defect: the one
    time this repo assumed a single shape for anomalies, the producing lane
    wrote `anomalies` as a list of flags and a consuming lane read
    `anomaly_kinds`, so a recorded triplication reached a user as a bare
    length with no caveat at all. A reader that accepts several shapes costs a
    few lines; a reader that accepts one costs a wrong answer.
    """
    raw = block.get("anomalies")
    if isinstance(raw, Mapping):
        for key in ("flags", "items", "anomalies", "findings"):
            if isinstance(raw.get(key), (list, tuple)):
                raw = raw[key]
                break
        else:
            raw = []
    if not isinstance(raw, (list, tuple)):
        return []
    kinds = {
        str(item.get("kind"))
        for item in raw
        if isinstance(item, Mapping) and item.get("kind")
    }
    return sorted(kinds)


def _anomalies_examined(block: Mapping[str, Any]) -> bool | None:
    status = block.get("anomalies_status")
    if not isinstance(status, Mapping):
        return None
    computed = status.get("computed")
    return computed if isinstance(computed, bool) else None


# =============================================================================
# Comparing one bucket
# =============================================================================


def compare_measure(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    units_agree: bool,
    units_why: str | None,
) -> dict[str, Any]:
    """Two `measure` blocks, subtracted only when subtracting means something.

    Four separate grounds for refusing the arithmetic, and every one of them
    has produced a wrong number somewhere in this project:

    1. **the two files declare different units** (G2) — an inch delta printed
       against a metre total;
    2. **the two buckets carry different units** — the same layer measured in
       model space and inside a block definition, where DXF declares no unit
       at all;
    3. **different measure kinds** — an area against a length. The subtraction
       is arithmetically fine and semantically empty;
    4. **a value absent on either side** (G8) — "nothing here could be
       measured" is not "this measures zero", and treating the first as the
       second turns an unknown into a finding.

    A refusal still reports both sides, so a reader sees what the two numbers
    were and why they were not subtracted.
    """
    before = before if isinstance(before, Mapping) else {}
    after = after if isinstance(after, Mapping) else {}

    kind_from = before.get("kind")
    kind_to = after.get("kind")
    unit_from = before.get("unit")
    unit_to = after.get("unit")
    value_from = _number(before.get("value"))
    value_to = _number(after.get("value"))

    view: dict[str, Any] = {
        "kind": {"from": kind_from, "to": kind_to, "changed": kind_from != kind_to},
        "unit": {"from": unit_from, "to": unit_to, "changed": unit_from != unit_to},
        "value": {"from": before.get("value"), "to": after.get("value")},
        "measured_entities": _delta(
            _count(before.get("measured_entities")),
            _count(after.get("measured_entities")),
        ),
        "unmeasured_entities": _delta(
            _count(before.get("unmeasured_entities")),
            _count(after.get("unmeasured_entities")),
        ),
        "chains": _delta(_count(before.get("chains")), _count(after.get("chains"))),
        "compared": False,
        "delta": None,
        "delta_unit": None,
        "why_not": None,
    }

    if not units_agree:
        view["why_not"] = (
            "the two drawings do not declare the same unit, so a difference "
            "between their measures is a number with no unit at all (G2). "
            + (units_why or "")
        ).strip()
        return view
    if kind_from != kind_to:
        view["why_not"] = (
            f"this bucket measured {kind_from!r} before and {kind_to!r} after. "
            "An area and a length subtract into a number that means nothing; "
            "the change of KIND is the finding here, not a delta."
        )
        return view
    if unit_from != unit_to:
        view["why_not"] = (
            f"this bucket's measure carries unit {unit_from!r} before and "
            f"{unit_to!r} after. A layer that moves between model space and a "
            "block definition changes what its numbers are measured in, and a "
            "delta across that move compares two different questions (G2)."
        )
        return view
    if kind_from is None and kind_to is None:
        view["why_not"] = (
            "neither side records a measure kind for this bucket, so there is "
            "nothing to compare. That is not a measure of zero."
        )
        return view
    if value_from is None or value_to is None:
        missing = (
            "before" if value_from is None and value_to is not None
            else "after" if value_to is None and value_from is not None
            else "on both sides"
        )
        view["why_not"] = (
            f"no value was established {missing}. Absent is not zero (G8): "
            "subtracting a measurement from a blank would publish the whole "
            "of one side as if it were the change."
        )
        return view

    view["compared"] = True
    view["delta"] = _round(value_to - value_from)
    view["delta_unit"] = unit_to
    view["delta_unit_reason"] = (
        None
        if unit_to
        else "neither revision declares a unit for this bucket, so the "
        "difference is in drawing units"
    )
    return view


def _delta(before: int | None, after: int | None) -> dict[str, Any]:
    """`{from, to, delta}` for a COUNT. Counts carry no unit, so they always
    compare — but an absent count still refuses to become a zero (G8)."""
    return {
        "from": before,
        "to": after,
        "delta": None if before is None or after is None else after - before,
        "changed": before != after,
    }


def compare_types(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Per-DXF-type entity counts, before against after.

    A count of entities of one type is dimensionless, so it is always
    comparable — no unit question arises. What IS stated is the Dossier's own
    truncation: `types` is capped upstream, so a type that fell outside the
    cap on either side is reported as unknown rather than as zero.
    """
    before = before if isinstance(before, Mapping) else {}
    after = after if isinstance(after, Mapping) else {}
    names = sorted(set(before) | set(after))
    rows: list[dict[str, Any]] = []
    for name in names:
        was = _count(before.get(name))
        now = _count(after.get(name))
        if was == now:
            continue
        rows.append(
            {
                "type": name,
                "from": was,
                "to": now,
                "delta": None if was is None or now is None else now - (was or 0),
                "appeared": was is None,
                "disappeared": now is None,
            }
        )
    listed = rows[:MAX_TYPE_ROWS]
    return {
        "changed": listed,
        "changed_total": len(rows),
        "changed_dropped": max(0, len(rows) - len(listed)),
        "cap": MAX_TYPE_ROWS,
        "basis": (
            "Counts per DXF type, which carry no unit and so always compare. "
            "`from: null` means the type is absent from that side's map — "
            "which the Dossier caps, so it can mean 'not in the top N' as "
            "well as 'not present'. It never means zero."
        ),
    }


def _bucket_summary(block: Mapping[str, Any]) -> dict[str, Any]:
    """One bucket described on its own — used for added and removed layers."""
    measure = block.get("measure")
    measure = measure if isinstance(measure, Mapping) else {}
    return {
        "layer": block.get("layer"),
        "layout": block.get("layout"),
        "entities": _count(block.get("entities")),
        "role": block.get("role"),
        "types": dict(block.get("types") or {}),
        "measure": {
            "kind": measure.get("kind"),
            "value": measure.get("value"),
            "unit": measure.get("unit"),
            "unit_reason": measure.get("unit_reason"),
        },
        "anomaly_kinds": _anomaly_kinds(block),
    }


def compare_bucket(
    key: tuple[str, str],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    units_agree: bool,
    units_why: str | None,
) -> dict[str, Any] | None:
    """One `(layer x layout)` bucket, or `None` when nothing about it moved.

    "Nothing moved" is decided over exactly the fields a Dossier holds:
    entity count, the type map, the geometric role, the native measure, and
    the anomaly kinds together with whether the anomaly check ran at all. A
    bucket whose anomaly check ran in one revision and not in the other has
    changed — not because the layer changed, but because what is known about
    it did, and hiding that would let "not examined" quietly replace a clean
    bill of health.
    """
    entities = _delta(_count(before.get("entities")), _count(after.get("entities")))
    types = compare_types(before.get("types") or {}, after.get("types") or {})

    role_from = before.get("role")
    role_to = after.get("role")
    role_changed = role_from != role_to

    measure = compare_measure(
        before.get("measure") or {},
        after.get("measure") or {},
        units_agree=units_agree,
        units_why=units_why,
    )

    kinds_from = _anomaly_kinds(before)
    kinds_to = _anomaly_kinds(after)
    appeared = [k for k in kinds_to if k not in kinds_from]
    disappeared = [k for k in kinds_from if k not in kinds_to]
    examined_from = _anomalies_examined(before)
    examined_to = _anomalies_examined(after)

    measure_moved = bool(
        measure["kind"]["changed"]
        or measure["unit"]["changed"]
        or (measure["compared"] and measure["delta"] != 0)
        or (not measure["compared"] and measure["value"]["from"] != measure["value"]["to"])
        or measure["chains"]["changed"]
    )

    if not (
        entities["changed"]
        or types["changed_total"]
        or role_changed
        or measure_moved
        or appeared
        or disappeared
        or examined_from != examined_to
    ):
        return None

    role_basis_from, from_clipped = _clip(before.get("role_basis"))
    role_basis_to, to_clipped = _clip(after.get("role_basis"))

    return {
        "layer": key[0],
        "layout": key[1],
        "entities": entities,
        "types": types,
        "role": {
            "from": role_from,
            "to": role_to,
            "changed": role_changed,
            "from_basis": role_basis_from if role_changed else None,
            "to_basis": role_basis_to if role_changed else None,
            "basis_truncated": bool(role_changed and (from_clipped or to_clipped)),
            "note": (
                "Role is decided from DXF type and ring status alone and never "
                "from the layer name (G1). A role that changed means the "
                "GEOMETRY on this layer changed character — closed rings "
                "became open paths, or a type family crossed the dominance "
                "threshold."
            )
            if role_changed
            else None,
        },
        "measure": measure,
        "anomalies": {
            "appeared": appeared,
            "disappeared": disappeared,
            "kinds_from": kinds_from,
            "kinds_to": kinds_to,
            "examined_from": examined_from,
            "examined_to": examined_to,
            "note": (
                "`examined` false on a side means the anomaly check did not "
                "run there. A kind that 'disappeared' into a not-examined "
                "revision has not been shown to be gone."
            )
            if examined_from != examined_to
            else None,
        },
    }


# =============================================================================
# Comparing the whole document
# =============================================================================

#: What a Dossier diff structurally cannot detect. Published on EVERY response,
#: not only when it happens to bite, because the reader who needs it most is
#: the one who has just been told "nothing changed".
CANNOT_SEE: Final[tuple[dict[str, str], ...]] = (
    {
        "what": "an entity that MOVED without changing size",
        "why": "A Dossier records aggregates per (layer x layout) bucket — a "
        "count, a type map, a role, a summed length or a summed area. A pure "
        "translation "
        "changes none of them, so a feature dragged to the other side of the "
        "site is invisible to this diff.",
        "remedy": "Compare the entities themselves: query_entities for that "
        "layer in both drawings and compare handles and bounding boxes, or "
        "look at the two renders side by side.",
    },
    {
        "what": "WHICH entity changed",
        "why": "A Dossier holds no handles, so one entity deleted and another "
        "added on the same layer is indistinguishable from a bucket whose "
        "count stayed the same. The counts and the measures move; the "
        "identities are not recorded here.",
        "remedy": "get_entity / query_entities on both drawings.",
    },
    {
        "what": "a change inside TEXT",
        "why": "The annotation census records content CLASSES and their "
        "ranges, not the strings. A label whose text was corrected reads as "
        "the same census.",
        "remedy": "search the two drawings for the string in question.",
    },
    {
        "what": "which of the two is actually NEWER",
        "why": "`computed_at` is when the Dossier was built and `ingested_at` "
        "is when the store saw the file. Neither is when the drafter saved "
        "it, and a DXF carries no reliable revision number.",
        "remedy": "Ask whoever sent the files, or read the title block.",
    },
)


def diff_dossiers(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    layout: str | None = None,
) -> dict[str, Any]:
    """Two Dossiers compared. Pure — it opens nothing and fetches nothing.

    `before` and `after` are the stored documents, exactly as
    `dossier_read.dossier_for` returns them. `layout` restricts the bucket
    comparison to one layout; the default of `None` compares every layout
    including block definitions, because a revision that changed only a block
    definition changed the file.

    What comes back always carries `not_compared` and `cannot_see`. The first
    is what this run declined to subtract and why; the second is what a
    Dossier diff can never see at all. Neither is ever empty of meaning: a
    reader told "no differences" without them would take it for "the drawings
    are the same file", and it is not the same claim.
    """
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise TypeError(
            "diff_dossiers compares two stored Dossier documents. Pass the "
            "result of dossier_read.dossier_for() for each side."
        )

    id_from = str(before.get("drawing_id") or before.get("_id") or "")
    id_to = str(after.get("drawing_id") or after.get("_id") or "")

    units_from = _units_of(before)
    units_to = _units_of(after)
    units_agree = (
        units_from.get("units_name") == units_to.get("units_name")
        and units_from.get("units_code") == units_to.get("units_code")
    )
    units_why = (
        None
        if units_agree
        else (
            f"the earlier drawing declares {units_from.get('units_name')!r} "
            f"(code {units_from.get('units_code')!r}) and the later one "
            f"{units_to.get('units_name')!r} (code "
            f"{units_to.get('units_code')!r}). Lengths and areas are therefore "
            "not subtracted anywhere in this report; the two figures are "
            "printed side by side instead, so a reader can see both without "
            "being handed a difference that means nothing."
        )
    )

    index_from = _bucket_index(before, layout)
    index_to = _bucket_index(after, layout)
    keys_from = set(index_from)
    keys_to = set(index_to)

    added_keys = sorted(keys_to - keys_from)
    removed_keys = sorted(keys_from - keys_to)
    shared_keys = sorted(keys_from & keys_to)

    changed: list[dict[str, Any]] = []
    unchanged = 0
    for key in shared_keys:
        row = compare_bucket(
            key,
            index_from[key],
            index_to[key],
            units_agree=units_agree,
            units_why=units_why,
        )
        if row is None:
            unchanged += 1
        else:
            changed.append(row)

    added = [_bucket_summary(index_to[k]) for k in added_keys]
    removed = [_bucket_summary(index_from[k]) for k in removed_keys]

    not_compared = _not_compared(
        changed, units_agree=units_agree, units_why=units_why
    )

    layers_from = {k[0] for k in keys_from}
    layers_to = {k[0] for k in keys_to}

    report: dict[str, Any] = {
        "diff_version": DIFF_VERSION,
        "compared": {
            "from": _side(before, id_from, units_from),
            "to": _side(after, id_to, units_to),
            "same_drawing": bool(id_from) and id_from == id_to,
        },
        "scope": {
            "layout": layout,
            "basis": (
                f"only layout {layout!r} is compared"
                if layout is not None
                else (
                    "every layout in both drawings is compared, block "
                    "definitions among them — geometry inside a block is "
                    "still geometry in the file, and a revision that touched "
                    "only a block would otherwise read as a revision that "
                    "changed nothing"
                )
            ),
        },
        "units": {
            "from": units_from,
            "to": units_to,
            "agree": units_agree,
            "measure_arithmetic": "allowed" if units_agree else "refused",
            "why": units_why
            or (
                "both drawings declare the same unit, so measures of the same "
                "kind in the same bucket unit are subtracted. A bucket that "
                "carries a different unit from its counterpart is still "
                "refused on its own terms."
            ),
        },
        "totals": {
            "entity_total": _delta(
                _count(before.get("entity_total")), _count(after.get("entity_total"))
            ),
            "buckets": {
                "from": len(keys_from),
                "to": len(keys_to),
                "added": len(added_keys),
                "removed": len(removed_keys),
                "changed": len(changed),
                "unchanged": unchanged,
            },
            "layers": {
                "from": len(layers_from),
                "to": len(layers_to),
                "added": sorted(layers_to - layers_from),
                "removed": sorted(layers_from - layers_to),
            },
            "coverage": _coverage_delta(before, after),
        },
        "layers_added": added[:MAX_BUCKET_ROWS],
        "layers_removed": removed[:MAX_BUCKET_ROWS],
        "layers_changed": changed[:MAX_BUCKET_ROWS],
        "layouts_changed": _layouts_delta(before, after),
        "file_facts_changed": _file_facts_delta(before, after),
        "anomalies": _anomaly_rollup(changed, added, removed),
        "not_compared": not_compared,
        "cannot_see": [dict(row) for row in CANNOT_SEE],
        "truncation": {
            "bucket_rows_cap": MAX_BUCKET_ROWS,
            "added_dropped": max(0, len(added) - MAX_BUCKET_ROWS),
            "removed_dropped": max(0, len(removed) - MAX_BUCKET_ROWS),
            "changed_dropped": max(0, len(changed) - MAX_BUCKET_ROWS),
            "basis": (
                f"Each list is capped at {MAX_BUCKET_ROWS} rows (G7). The "
                "counts in `totals` are computed over EVERY bucket, not only "
                "the ones listed, so a truncated listing never understates "
                "how much changed."
            ),
        },
    }
    report["verdict"] = _verdict(report)
    return report


def _side(
    document: Mapping[str, Any], drawing_id: str, units: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "drawing_id": drawing_id or None,
        "computed_at": document.get("computed_at"),
        "dossier_version": document.get("dossier_version"),
        "entity_total": document.get("entity_total"),
        "buckets": len(_blocks(document)),
        "units": dict(units),
    }


def _coverage_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Whether each side accounted for its own file, and whether that changed.

    Published because it decides how much the rest of this report is worth: a
    diff between two Dossiers where one of them does not account for its own
    drawing is a diff over part of a file, and it must say so rather than
    letting the reader assume both sides were complete.
    """
    cov_from = before.get("coverage")
    cov_to = after.get("coverage")
    cov_from = cov_from if isinstance(cov_from, Mapping) else {}
    cov_to = cov_to if isinstance(cov_to, Mapping) else {}
    complete_from = cov_from.get("complete")
    complete_to = cov_to.get("complete")
    return {
        "accounted": _delta(
            _count(cov_from.get("accounted")), _count(cov_to.get("accounted"))
        ),
        "complete": {
            "from": complete_from,
            "to": complete_to,
            "changed": complete_from != complete_to,
        },
        "both_complete": complete_from is True and complete_to is True,
        "note": (
            "Both Dossiers account for every entity in their own drawing, so "
            "the comparison below is over whole files."
            if complete_from is True and complete_to is True
            else "At least one side does NOT account for every entity in its "
            "own drawing. Every figure below is a comparison over part of a "
            "file, and a bucket missing from an incomplete Dossier reads here "
            "exactly like a layer that was deleted."
        ),
    }


def _layouts_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Layouts added, removed, and those whose profiled entity count moved."""

    def index(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        raw = document.get("layouts")
        rows = raw if isinstance(raw, list) else []
        return {
            str(r.get("name")): r
            for r in rows
            if isinstance(r, Mapping) and r.get("name") is not None
        }

    a, b = index(before), index(after)
    changed: list[dict[str, Any]] = []
    for name in sorted(set(a) & set(b)):
        row = _delta(
            _count(a[name].get("profiled_entities")),
            _count(b[name].get("profiled_entities")),
        )
        if row["changed"]:
            changed.append({"layout": name, "profiled_entities": row})
    listed = changed[:MAX_LAYOUT_ROWS]
    return {
        "added": sorted(set(b) - set(a)),
        "removed": sorted(set(a) - set(b)),
        "changed": listed,
        "changed_total": len(changed),
        "changed_dropped": max(0, len(changed) - len(listed)),
        "cap": MAX_LAYOUT_ROWS,
    }


#: File-level facts worth comparing, each named by the path it lives at and by
#: what a reader should understand from a change in it. Written as data rather
#: than as a chain of `if`s so that the list is readable as a list.
_FACT_PATHS: Final[tuple[tuple[str, tuple[str, ...], str], ...]] = (
    (
        "xrefs.total",
        ("xrefs", "total"),
        "how many external references the file names",
    ),
    (
        "xrefs.unresolved",
        ("xrefs", "unresolved"),
        "external references that could not be found. A rise means content "
        "the drawing points at is missing from what was delivered",
    ),
    (
        "text_styles.total",
        ("text_styles", "total"),
        "text styles declared in the file",
    ),
    (
        "text_styles.shx_styles",
        ("text_styles", "shx_styles"),
        "styles pointing at an SHX font, whose text is only readable through "
        "the SHX map",
    ),
    (
        "layers.declared",
        ("layers", "declared"),
        "layers in the drawing's own layer table",
    ),
    (
        "layers.profiled",
        ("layers", "profiled"),
        "layers that actually hold at least one entity",
    ),
    (
        "layers.declared_but_unprofiled",
        ("layers", "declared_but_unprofiled"),
        "layers declared and empty",
    ),
    (
        "layouts.declared",
        ("layouts", "declared"),
        "layouts the file declares",
    ),
    (
        "layouts.disagreeing",
        ("layouts", "disagreeing"),
        "layouts whose declared and profiled entity counts differ",
    ),
    (
        "audit.errors",
        ("audit", "errors"),
        "errors ezdxf's auditor recovered when the file was opened. A file "
        "that arrives damaged is normal; a change here says the two files "
        "arrived in different condition, which is a fact about the DELIVERY "
        "and not about the design",
    ),
    (
        "audit.fixes",
        ("audit", "fixes"),
        "fixes the auditor applied on open. Same reading as `audit.errors`: "
        "a property of the file as delivered, not of what is drawn",
    ),
    (
        "units.units_name",
        ("units", "units_name"),
        "the unit the file header declares for model space. A change here "
        "changes the meaning of every length and area in the drawing",
    ),
    (
        "units.units_code",
        ("units", "units_code"),
        "the raw $INSUNITS code behind `units_name`",
    ),
    (
        "embedded_documents.total",
        ("embedded_documents", "total"),
        "documents embedded inside the file",
    ),
)


def _dig(document: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = document.get("file_facts")
    for step in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(step)
    return node


def _file_facts_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, path, meaning in _FACT_PATHS:
        was = _dig(before, path)
        now = _dig(after, path)
        if was == now:
            continue
        rows.append(
            {
                "fact": name,
                "from": _plain(was),
                "to": _plain(now),
                "delta": (
                    now - was
                    if isinstance(was, (int, float))
                    and isinstance(now, (int, float))
                    and not isinstance(was, bool)
                    and not isinstance(now, bool)
                    else None
                ),
                "meaning": meaning,
            }
        )
    listed = rows[:MAX_FACT_ROWS]
    return {
        "changed": listed,
        "changed_total": len(rows),
        "changed_dropped": max(0, len(rows) - len(listed)),
        "cap": MAX_FACT_ROWS,
        "basis": (
            "File-level facts, which are deliberately NOT entities and are "
            "not part of the entity totals above. `audit.errors` and "
            "`audit.fixes` in particular describe the condition the FILE "
            "arrived in, not the design inside it: they move when the same "
            "drawing is re-exported through a different tool."
        ),
    }


def _anomaly_rollup(
    changed: Sequence[Mapping[str, Any]],
    added: Sequence[Mapping[str, Any]],
    removed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Which anomaly kinds are newly reported, and which stopped being."""
    appeared: dict[str, int] = {}
    disappeared: dict[str, int] = {}
    for row in changed:
        block = row.get("anomalies") or {}
        for kind in block.get("appeared") or []:
            appeared[str(kind)] = appeared.get(str(kind), 0) + 1
        for kind in block.get("disappeared") or []:
            disappeared[str(kind)] = disappeared.get(str(kind), 0) + 1
    for row in added:
        for kind in row.get("anomaly_kinds") or []:
            appeared[str(kind)] = appeared.get(str(kind), 0) + 1
    for row in removed:
        for kind in row.get("anomaly_kinds") or []:
            disappeared[str(kind)] = disappeared.get(str(kind), 0) + 1
    return {
        "appeared": dict(sorted(appeared.items())),
        "disappeared": dict(sorted(disappeared.items())),
        "basis": (
            "Counted over buckets, not over findings: `duplicate_clusters: 2` "
            "means two buckets began reporting that kind, not that two copies "
            "were found. A kind that appears on a layer whose geometry did "
            "not change is usually a threshold crossed, not a new defect — "
            "the detector derives its threshold from the layer's own widest "
            "feature."
        ),
    }


def _not_compared(
    changed: Sequence[Mapping[str, Any]],
    *,
    units_agree: bool,
    units_why: str | None,
) -> list[dict[str, Any]]:
    """What this run declined to subtract, grouped by reason, with counts.

    Grouped rather than listed per bucket: on a drawing with 176 buckets, one
    line per refusal is a wall nobody reads, and the fact that matters is
    *how many* and *why*.
    """
    out: list[dict[str, Any]] = []
    if not units_agree:
        out.append(
            {
                "what": "every length and area in this report",
                "count": len(changed),
                "why": units_why,
                "remedy": (
                    "There is none that is honest inside this tool. A "
                    "conversion factor would have to be assumed, and 3 of the "
                    "drawings in this store declare no unit at all, so the "
                    "factor would sometimes be assumed out of nothing. "
                    "Compare the two figures as printed."
                ),
            }
        )
        return out

    reasons: dict[str, int] = {}
    for row in changed:
        measure = row.get("measure") or {}
        if measure.get("compared"):
            continue
        why = str(measure.get("why_not") or "no reason was recorded")
        reasons[why] = reasons.get(why, 0) + 1
    for why, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(
            {
                "what": "the measure delta for this many changed buckets",
                "count": count,
                "why": why,
                "remedy": (
                    "Both values are printed in `measure.value`; read them "
                    "side by side."
                ),
            }
        )
    return out


def _verdict(report: Mapping[str, Any]) -> str:
    """One sentence a person or an agent can relay. Generated, never handed in."""
    compared = report["compared"]
    if compared.get("same_drawing"):
        return (
            "These are the same drawing id, so there is nothing to compare. A "
            "drawing id is a content hash: two files with the same id are the "
            "same file, byte for byte."
        )
    totals = report["totals"]
    buckets = totals["buckets"]
    entity = totals["entity_total"]
    moved = buckets["added"] + buckets["removed"] + buckets["changed"]
    if not moved and not report["file_facts_changed"]["changed_total"]:
        return (
            "No difference was found in anything a Dossier records: the same "
            f"{buckets['from']} (layer x layout) buckets, the same counts, "
            "roles and measures. That is NOT the same as 'the two files are "
            "identical' — see `cannot_see`: a Dossier holds aggregates, so an "
            "entity that moved without changing size leaves no trace here."
        )
    parts = []
    if entity["delta"] is not None and entity["delta"]:
        parts.append(
            f"{abs(entity['delta'])} "
            f"{'more' if entity['delta'] > 0 else 'fewer'} entities "
            f"({entity['from']} -> {entity['to']})"
        )
    if buckets["added"]:
        parts.append(f"{buckets['added']} bucket(s) appeared")
    if buckets["removed"]:
        parts.append(f"{buckets['removed']} bucket(s) disappeared")
    if buckets["changed"]:
        parts.append(f"{buckets['changed']} bucket(s) changed")
    facts = report["file_facts_changed"]["changed_total"]
    if facts:
        parts.append(f"{facts} file-level fact(s) differ")
    head = "; ".join(parts) if parts else "something moved"
    tail = ""
    if report["units"]["measure_arithmetic"] == "refused":
        tail = (
            " Lengths and areas were NOT subtracted: the two drawings declare "
            "different units, and a difference across that is a number in no "
            "unit at all."
        )
    return (
        f"{head}. {buckets['unchanged']} bucket(s) are unchanged in everything "
        "a Dossier records." + tail
    )


# =============================================================================
# The recipe
# =============================================================================


def _dossier(drawing_id: str) -> Mapping[str, Any] | None:
    """One stored Dossier. The single seam this file has to the store."""
    return coll(COLL_DOSSIERS).find_one({"_id": str(drawing_id)})


def _refuse_no_dossier(drawing_id: str, which: str) -> RecipeRefused:
    return RecipeRefused(
        "RECIPE_PARAM_INVALID",
        f"no Dossier has been built for the {which} drawing {drawing_id!r}.",
        "This is NOT a statement that the drawing is empty — it means what "
        "the file contains has not been computed, so there is nothing to "
        "compare. Since DOSSIER Phase 7 a Dossier is built when a drawing is "
        "ingested; a drawing ingested before that gets one from `python "
        f"scripts/dossier_backfill.py --drawing {drawing_id}`.",
    )


def _run_revision_diff(
    *,
    drawing_id: str,
    layout: str | None,
    units: Mapping[str, Any],
    other_drawing: str | None,
    scope: str,
) -> dict[str, Any]:
    wanted = str(scope or "all").strip().lower()
    if wanted not in ("all", "layout"):
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"`scope` has the value {scope!r}; it must be `all` or `layout`.",
            "`all` compares every layout including block definitions, which "
            "is what a question about a revision usually means. `layout` "
            "restricts the comparison to the layout this call names.",
        )
    scoped_layout = layout if wanted == "layout" else None

    mine = _dossier(drawing_id)
    if mine is None:
        raise _refuse_no_dossier(str(drawing_id), "current")

    if other_drawing:
        other_id = str(other_drawing).strip()
        found = {
            "rule": PREDECESSOR_RULE,
            "rule_limits": list(PREDECESSOR_LIMITS),
            "drawing_id": str(drawing_id),
            "predecessor": other_id,
            "candidates": [],
            "candidate_count": 0,
            "ambiguous": False,
            "why": (
                "the caller named the other drawing explicitly, so the "
                "predecessor rule was not used. The rule is published above "
                "so it is visible that it was not what chose this pair."
            ),
            "source": "named by the caller",
        }
    else:
        found = predecessor_for(str(drawing_id))
        found["source"] = "the predecessor rule"
        other_id = found.get("predecessor")

    if not other_id:
        # Ambiguity and absence are REPORTED, never resolved. The response is
        # a well-formed answer that says which of the two happened and lists
        # what it found, because an agent handed a bare error will reach for
        # another tool and answer the question wrongly with confidence.
        return {
            "scope_note": (
                f"no comparison was made. Drawing {drawing_id!r}; "
                f"{found.get('candidate_count', 0)} candidate predecessor(s) "
                "under the stated rule."
            ),
            "compared": False,
            "why_no_comparison": found.get("why"),
            "predecessor_search": found,
            "diff": None,
            "not_measured": (
                "nothing was measured: no single predecessor was identified, "
                "so no two Dossiers were compared. "
                + str(found.get("why") or "")
            ),
            "what_to_do": (
                "Name the drawing to compare against with the "
                "`other_drawing` parameter. `list_drawings` shows every "
                "drawing in the store with its id."
                if found.get("ambiguous")
                else "If a previous version exists under a different file "
                "name, name it with the `other_drawing` parameter — the rule "
                "matches names exactly and cannot follow a rename."
            ),
        }

    if str(other_id) == str(drawing_id):
        raise RecipeRefused(
            "RECIPE_PARAM_INVALID",
            f"`other_drawing` is {other_id!r}, the same drawing as this one.",
            "A drawing id is a content hash, so a drawing compared with "
            "itself has nothing to report by construction. Name the OTHER "
            "revision.",
        )

    theirs = _dossier(str(other_id))
    if theirs is None:
        raise _refuse_no_dossier(str(other_id), "other")

    # `theirs` is the BEFORE side and `mine` the AFTER side, and that is a
    # convention rather than a finding: nothing in either document establishes
    # which file the drafter saved first (`cannot_see`). The question asked
    # through this recipe is "what changed in the drawing I am looking at",
    # so the drawing being asked about is the later one.
    report = diff_dossiers(theirs, mine, layout=scoped_layout)
    report["order_basis"] = (
        f"the drawing being asked about ({drawing_id}) is treated as the "
        f"LATER revision and {other_id} as the earlier one. That is the "
        "direction of the question, not a fact established from the files: "
        "neither Dossier records when the drafter saved anything."
    )

    return {
        "scope_note": (
            f"drawing {drawing_id!r} compared against {other_id!r}; "
            + (
                f"layout {scoped_layout!r} only"
                if scoped_layout is not None
                else "every layout in both files, block definitions among them"
            )
            + "; units "
            + (
                f"{report['units']['to'].get('units_name')!r} on both sides"
                if report["units"]["agree"]
                else "DIFFER between the two files, so no length or area was "
                "subtracted (G2)"
            )
        ),
        "compared": True,
        "predecessor_search": found,
        "diff": report,
        "verdict": report["verdict"],
        "not_measured": (
            "This compares two DOSSIERS, which are aggregates per (layer x "
            "layout) bucket. It did not measure position, entity identity, or "
            "the content of any text — see `diff.cannot_see` for what that "
            "rules out, and `diff.not_compared` for what this particular run "
            "declined to subtract."
        ),
    }


register(
    Recipe(
        name="revision_diff",
        answers=(
            "what changed between two revisions of a drawing: layers "
            "appearing and disappearing, roles changing, counts and measures "
            "moving, anomalies appearing"
        ),
        when_to_use=(
            "'what changed since the last revision', 'what is different "
            "between these two files', 'did any plot's area change'. Leave "
            "`other_drawing` empty to let the predecessor rule find the other "
            "version by name — if it finds none or several, the answer says "
            "so and lists them rather than picking one."
        ),
        params=(
            Param(
                "other_drawing",
                "text",
                "the drawing id of the other revision. Leave it out to have "
                "the predecessor found by the stated rule: same drawing name, "
                "different content hash. A drawing id is a content hash, so "
                "each version of a file has its own.",
            ),
            Param(
                "scope",
                "text",
                "`all` (the default) compares every layout, block definitions "
                "included — a revision that changed only a block still "
                "changed the file. `layout` restricts the comparison to the "
                "layout named in this call.",
                default="all",
            ),
        ),
        returns=(
            "diff.totals.entity_total",
            "diff.layers_added",
            "diff.layers_removed",
            "diff.layers_changed[].entities",
            "diff.layers_changed[].measure.delta",
            "diff.layers_changed[].role",
            "diff.anomalies",
            "diff.file_facts_changed",
            "diff.not_compared",
            "diff.cannot_see",
            "predecessor_search",
            "verdict",
        ),
        built_on=(
            "autocad_dossiers.find_one",
            "autocad_drawings.find",
            "app.recipes.revision.diff_dossiers",
        ),
        limits={
            "bucket_rows_per_list": MAX_BUCKET_ROWS,
            "types_per_bucket": MAX_TYPE_ROWS,
            "file_facts": MAX_FACT_ROWS,
            "predecessor_candidates_listed": MAX_CANDIDATES,
        },
        # It measures: counts, lengths, areas and their differences. It says
        # what something IS nowhere — a role is republished from the Dossier,
        # never decided here — so it carries `not_measured` rather than
        # `evidence`, and what it did not measure is the point of the file.
        carries_meaning=False,
        # A revision is a question about a FILE, not about a sheet. The API
        # requires a layout on every recipe call, so one arrives; `scope`
        # decides whether it restricts anything, and `scope_note` says which.
        needs_layout=False,
        run=_run_revision_diff,
    )
)
