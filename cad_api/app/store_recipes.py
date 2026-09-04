"""business tags per typology; then the analysis recipe registry

Owner: subagent H — UPLIFT-13, then UPLIFT-09.

This file is deliberately split off from `store.py` so that two specs worked
on at the same time never touch the same line. The rules, and why:

- **Do not import from `.store`.** `store.py` imports this module at its
  bottom; a reverse import would be circular. Take `coll` and friends
  straight from `.mongo`, and geometry from `.region`.
- **Do not add lines to `store.py` or `main.py`.** Both are shared property,
  and every line added there is one merge conflict. Functions here are
  reached as `store.store_recipes.<function>`.
- **Every response that carries MEANING must carry `evidence`** per
  `docs/UPLIFT-08-EVIDENCE.md` — use the helpers in `app/evidence.py`, do not
  assemble the dict yourself.
- **Every number carries its unit** (G2), and a unit is allowed to be absent.
  A withheld number becomes `None` + a reason, never `0`.

The `docs/UPLIFT-GENERALITY-RULES.md` G1-G10 checklist is run before every
commit in this file.

---

## UPLIFT-13 — what was actually asked for, and the shape that follows

> *"if we do it for C-PROP-PlotNumber, it should reflect for all the plots."*

Label once for a typology, and it applies to all of its plots. That is why the
target is a **layer**, not an entity: a typology is indeed carried by the layer
name, and 46,677 entities cannot be tagged one by one by a human.

One sentence determines all the rest: **a business tag is a claim about
MEANING, and the file states nothing about business names.** 43,109 texts were
examined; zero of them name the meaning of a typology code. Three
consequences, and all three are encoded here rather than left to the author's
discipline:

1.  **A tag claim never carries `tokens`.** `Claim.tokens` are the words
    that, when they appear in a file string, make that file STATE its claim.
    If a tag carried tokens, somebody would only have to type a business name
    containing a word from its own layer name, and the `LAYER_NAME`
    observation would "state" the claim they had just invented. That is not
    evidence; that is a circle. Empty tokens is a VALID state according to
    `evidence.py`: the claim simply can never be `stated`.

2.  **The ceiling is `inferred`, stated, not accidental.** It is already
    `inferred` today because `Origin.HUMAN` falls into `Corpus.OUTSIDE_FILE`,
    which is in `NEVER_SPEAKS`. The explicit ceiling exists so that the intent
    is readable by the next person, and so that adding tokens — by anyone —
    does not quietly raise it.

3.  **`verified` and `grade` are two axes, and here they are DEFINITELY
    different.** `grade` answers what the file says — `inferred`. `verified`
    answers whether a human has confirmed the mapping — `true`, because
    somebody pressed a button and signed it with a `source`. "Human-verified,
    not stated by the file" is the most honest description of what a business
    tag is, and it is not a contradiction.

`verified` is therefore NOT stored as a boolean. UPLIFT-08 removed it
precisely because a `verified` boolean plus `source` prose is a mechanism for
raising the evidence grade in the shape of data. What is mandatory is
`source`, and it is forwarded as the `reference` of an `Origin.HUMAN`
observation — where `evidence.py` itself refuses to publish it when empty
(`REFERENCE_REQUIRED`).

## Reading, not rewriting

```
YAML config  →  autocad_tags  →  effective value
```

A tag is never written into the entity documents and never written to the DWG
file. Editing one layer applies to every entity on that layer because the
merge happens at READ time — that is what "without re-ingest" means.

The config is handed over by the **caller** as a parameter, not fetched from
`store_landuse` here: that file belongs to another subagent and its config
shape is still moving. What is here stays correct when there is no config at
all (G3).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Final, Iterable, Mapping, Sequence

from . import evidence as ev
from .extract import BLOCK_LAYOUT_PREFIX
from .mongo import COLL_ENTITIES, coll

# --- collections -------------------------------------------------------------

#: Business decisions from the screen, separate from the config reviewed
#: through a merge request. Prefixed `autocad_` like every collection in this
#: project: this database is shared with other teams and a collection named
#: `tags` in there is damage that cannot be undone.
#:
#: Its registration in `mongo.OWNED_COLLECTIONS` is requested through
#: `docs/PROGRESS.md`, not edited from here — `mongo.py` is shared property.
COLL_TAGS: Final[str] = "autocad_tags"

#: Raised only when the shape of a tag document changes so that old and new
#: documents cannot be told apart without looking at them.
TAG_CONTRACT_VERSION: Final[int] = 1

#: The only target that can be tagged. Per-entity tagging is NOT built:
#: tagging some of 46,677 entities produces half-correct data, and that is a
#: decision of its own if it is ever needed.
TARGET_LAYER: Final[str] = "layer"
TARGET_KINDS: Final[tuple[str, ...]] = (TARGET_LAYER,)

#: The separator inside the composite key. `target` is always the LAST
#: fragment, so a layer name that happens to contain this separator is still
#: read back whole.
_KEY_SEP: Final[str] = ":"

#: A stated limit, not an assumed one (G7). A "business name" as long as a
#: paragraph is not a name; it is a note in the wrong place, and its place is
#: `source`.
MAX_BUSINESS_NAME: Final[int] = 120
MAX_SOURCE: Final[int] = 2000

#: How many layouts are broken out before the impact report stops breaking
#: them out. Above this the total stays correct and the rest are COUNTED,
#: never silently truncated.
MAX_IMPACT_LAYOUTS: Final[int] = 24


# --- refusals ----------------------------------------------------------------


#: A closed list, for the same reason as `evidence.ERROR_CODES`: a new code
#: that simply appears cannot be looked up in the documentation, and two
#: different causes using one code can no longer be told apart when read from
#: the log.
TAG_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "TAG_DRAWING_ID_INVALID",
        "TAG_UNKNOWN_TARGET_KIND",
        "TAG_TARGET_INVALID",
        "TAG_TARGET_MISMATCH",
        "TAG_WITHOUT_NAME",
        "TAG_WITHOUT_SOURCE",
        "TAG_WITHOUT_AUTHOR",
        "TAG_NAME_TOO_LONG",
        "TAG_SOURCE_TOO_LONG",
    }
)


class TagRefused(Exception):
    """An actionable refusal, not an infrastructure failure.

    Its shape mirrors `store.MeasureRefused` (`code`/`message`/`hint`) so that
    the `_refusal_handler` already present in `main.py` can be reused as is
    and the refusal comes out as a 400. It does not INHERIT `MeasureRefused`
    precisely because that class lives in `store.py`, and importing it would
    close the import cycle that this file's separation exists to break.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        if code not in TAG_ERROR_CODES:
            raise AssertionError(f"refusal code {code!r} is not registered")
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


# --- the MongoDB door --------------------------------------------------------


def _tags():
    """The tag collection. The only way there, and it goes through `coll()`."""
    return coll(COLL_TAGS)


def _entities():
    return coll(COLL_ENTITIES)


# --- document key ------------------------------------------------------------


def tag_id(drawing_id: str, target_kind: str, target: str) -> str:
    """`<drawing_id>:<target_kind>:<target>`.

    It leads with `drawing_id` not for beauty: every read of this collection
    goes through the `_id` index, and all the tags of one drawing are a single
    contiguous range inside it. This cluster runs with `notablescan` — a query
    without an indexed plan does not slow down, it fails.
    """
    did = (drawing_id or "").strip()
    if not did or _KEY_SEP in did:
        raise TagRefused(
            "TAG_DRAWING_ID_INVALID",
            f"drawing_id {drawing_id!r} cannot be used as a tag key.",
            f"drawing_id is a digest of the file contents and never contains "
            f"{_KEY_SEP!r}. A value that contains it would make two layers "
            "share one tag document, silently.",
        )
    if target_kind not in TARGET_KINDS:
        raise TagRefused(
            "TAG_UNKNOWN_TARGET_KIND",
            f"tag target {target_kind!r} is not known.",
            "The only thing that can be tagged is a `layer`. Per-entity "
            "tagging is not built: 46,677 entities cannot be tagged one by "
            "one by a human, and tagging some of them produces half-correct "
            "data. If it really is needed, that is a decision of its own.",
        )
    name = (target or "").strip()
    if not name:
        raise TagRefused(
            "TAG_TARGET_INVALID",
            "the name of the layer being tagged is empty.",
            "Name the layer being tagged. A tag without a target would apply "
            "to whatever reads it next.",
        )
    return f"{did}{_KEY_SEP}{target_kind}{_KEY_SEP}{name}"


def parse_tag_id(key: str) -> tuple[str, str, str]:
    """The inverse of `tag_id`. `target` is the remainder, so it is never ambiguous."""
    drawing_id, target_kind, target = str(key).split(_KEY_SEP, 2)
    return drawing_id, target_kind, target


def _drawing_key_range(drawing_id: str) -> dict[str, str]:
    """The `_id` range that holds exactly all the tags of one drawing.

    Its upper bound is not `\\uffff` but the character immediately after the
    separator. MongoDB compares strings as UTF-8 bytes, and a layer name that
    contains a character above the BMP (an emoji, say) begins its bytes with
    `F0`, which is greater than `EF BF BF`. A bound of "the next character
    after the separator" has no such hole at all: the comparison finishes at
    the separator position, whatever follows it.
    """
    lo = f"{drawing_id}{_KEY_SEP}"
    hi = f"{drawing_id}{chr(ord(_KEY_SEP) + 1)}"
    return {"$gte": lo, "$lt": hi}


# --- writing a tag -----------------------------------------------------------


def _required(value: Any, *, code: str, what: str, hint: str) -> str:
    text = (value or "").strip() if isinstance(value, str) else ""
    if not text:
        raise TagRefused(code, f"{what} is empty.", hint)
    return text


def put_layer_tag(
    *,
    drawing_id: str,
    layer: str,
    business_name: str,
    source: str,
    author: str,
    use: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Store one business name for one layer. It applies to every entity on that layer.

    `source` is mandatory and has no default. A tag without a source would
    become exactly the problem UPLIFT-08 exists to prevent: a guess that in
    six months cannot be told apart from a fact. The refusal happens HERE,
    before a single byte is written, so that its message names the field that
    has to be filled in — `evidence.py` would also refuse it later through
    `REFERENCE_REQUIRED`, but that happens at read time, far from the person
    who is typing.
    """
    key = tag_id(drawing_id, TARGET_LAYER, layer)
    did, _, target = parse_tag_id(key)

    name = _required(
        business_name,
        code="TAG_WITHOUT_NAME",
        what="business_name",
        hint="Write the business name people use for this typology. That is "
        "the only content of a tag; without it nothing is stored.",
    )
    if len(name) > MAX_BUSINESS_NAME:
        raise TagRefused(
            "TAG_NAME_TOO_LONG",
            f"business_name is {len(name)} characters, the limit is {MAX_BUSINESS_NAME}.",
            "A name as long as a paragraph is not a name but a note, and the "
            "place for a note is `source`.",
        )
    reason = _required(
        source,
        code="TAG_WITHOUT_SOURCE",
        what="source",
        hint="Name who decided it, when, and on what basis — a meeting, a "
        "document, or the typology key from Roshn, for example. This tag will "
        "be read by people who were not there when you typed it, and without "
        "a source it cannot be told apart from a guess.",
    )
    if len(reason) > MAX_SOURCE:
        raise TagRefused(
            "TAG_SOURCE_TOO_LONG",
            f"source is {len(reason)} characters, the limit is {MAX_SOURCE}.",
            "Condense the source into a reference other people can open.",
        )
    who = _required(
        author,
        code="TAG_WITHOUT_AUTHOR",
        what="author",
        hint="Name who is saving this tag. `source` explains its basis; "
        "`author` explains who pressed the button, and the two are not always "
        "the same person.",
    )

    stamp = now or datetime.now(timezone.utc)
    prior = _tags().find_one({"drawing_id": did, "_id": key})

    doc: dict[str, Any] = {
        "_id": key,
        "drawing_id": did,
        "target_kind": TARGET_LAYER,
        "target": target,
        "business_name": name,
        "use": (use or "").strip() or None,
        "source": reason,
        "author": who,
        "created_at": (prior or {}).get("created_at", stamp),
        "updated_at": stamp,
        "revision": int((prior or {}).get("revision", 0)) + 1,
        # Editing a tag deletes the previous business decision. What remains
        # must be enough to answer "who changed it from what"; one snapshot,
        # not an unbounded history growing in a shared database.
        "previous": _snapshot(prior),
        "tag_contract_version": TAG_CONTRACT_VERSION,
    }
    _tags().replace_one({"drawing_id": did, "_id": key}, doc, upsert=True)
    return doc


def _snapshot(prior: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not prior:
        return None
    return {
        "business_name": prior.get("business_name"),
        "use": prior.get("use"),
        "source": prior.get("source"),
        "author": prior.get("author"),
        "updated_at": prior.get("updated_at"),
        "revision": prior.get("revision"),
    }


# --- reading a tag -----------------------------------------------------------


def get_layer_tag(drawing_id: str, layer: str) -> dict[str, Any] | None:
    """One layer's tag, or None. None means not tagged yet, not failed."""
    key = tag_id(drawing_id, TARGET_LAYER, layer)
    return _tags().find_one({"drawing_id": drawing_id, "_id": key})


def list_tags(drawing_id: str) -> list[dict[str, Any]]:
    """All the tags of one drawing, ordered by their target.

    Its filter leads with `drawing_id` and is bounded by an `_id` range, so it
    always has an indexed plan without waiting for a new index to be built on
    this collection.
    """
    rows = list(
        _tags().find(
            {"drawing_id": drawing_id, "_id": _drawing_key_range(drawing_id)}
        )
    )
    return sorted(rows, key=lambda r: (r.get("target_kind", ""), r.get("target", "")))


def tags_by_layer(drawing_id: str) -> dict[str, dict[str, Any]]:
    """A `layer -> tag` map, to be folded into other modules' responses."""
    return {
        row["target"]: row
        for row in list_tags(drawing_id)
        if row.get("target_kind") == TARGET_LAYER and row.get("target")
    }


# --- how many objects are affected -------------------------------------------


def layer_impact(
    drawing_id: str, layer: str, layout: str | None = None
) -> dict[str, Any]:
    """How many objects will use this name, before anything is stored.

    Asked for by UPLIFT-13 as the first thing that has to be visible on the
    screen. It is computed with `count_documents` + `distinct`, not with an
    aggregation pipeline, because the semantics of those two are simple enough
    to fake in a test without building a pipeline interpreter that could lie
    differently from the cluster.

    The count is split between real layouts and block definitions. One
    combined number would make people think everything is drawn where they can
    see it, when geometry inside a block definition is rescaled by every
    insertion of it.
    """
    base: dict[str, Any] = {"drawing_id": drawing_id, "layer": layer}
    if layout:
        base["layout"] = layout

    entities = _entities()
    total = entities.count_documents(dict(base))
    layouts = list(entities.distinct("layout", dict(base)))

    shown = sorted(layouts)[:MAX_IMPACT_LAYOUTS]
    by_layout = {
        name: entities.count_documents({**base, "layout": name}) for name in shown
    }
    in_blocks = sum(
        n for name, n in by_layout.items() if str(name).startswith(BLOCK_LAYOUT_PREFIX)
    )
    truncated = len(layouts) - len(shown)

    if total == 0:
        note = (
            f"Layer {layer!r} carries no objects at all in this drawing"
            + (f" on layout {layout!r}" if layout else "")
            + "; tagging it will not change anything visible."
        )
    else:
        note = (
            f"{layer} — {total} objects will use this name"
            + (f" on layout {layout!r}" if layout else "")
            + "."
        )
        if in_blocks:
            note += (
                f" {in_blocks} of them sit inside a block definition, so the "
                "number drawn on the screen depends on how many times that "
                "block is inserted."
            )
        if truncated > 0:
            note += f" {truncated} other layouts are not broken out."

    return {
        "drawing_id": drawing_id,
        "layer": layer,
        "layout": layout,
        "entity_count": total,
        "by_layout": by_layout,
        "in_block_definitions": in_blocks,
        "on_real_layouts": total - in_blocks,
        "layouts_truncated": truncated if truncated > 0 else 0,
        "impact_note": note,
        "why_empty": None if total else (
            "There are no entities on this layer within the requested scope. "
            "That is an answer, not a failure: a layer can be listed in the "
            "layer table without a single object drawing it."
        ),
    }


# --- evidence ----------------------------------------------------------------


def tag_observation(tag: Mapping[str, Any]) -> ev.Observation:
    """The only sanctioned way to fold a screen tag into ANYBODY's evidence.

    Other modules call this instead of assembling an `Observation` from
    `tag["source"]` themselves. If they assembled it themselves, sooner or
    later somebody would write it as `LAYER_DESCRIPTION` — the `NAME_STRING`
    corpus, which CAN state — and a sentence typed by a person would rise into
    a statement by the file.

    `observed` is deliberately None: the `OUTSIDE_FILE` corpus is in
    `NEVER_SPEAKS`, so there is no file string that could be handed over, and
    `evidence.py` does not ask for one. What is MANDATORY is `reference`, and
    that is `source`.
    """
    return ev.Observation(
        origin=ev.Origin.HUMAN,
        detail=(
            f"screen tag for {tag.get('target_kind', TARGET_LAYER)} "
            f"{tag.get('target')!r}, saved by {tag.get('author')}"
        ),
        observed=None,
        locator=f"tag:{tag.get('_id')}",
        reference=tag.get("source"),
    )


def _layer_observation(layer: str, layout: str | None) -> ev.Observation:
    """The layer itself: the only thing in the file this tag attaches to.

    It will never STATE its business name — a tag claim carries no tokens —
    but it is what makes the claim point at something real instead of
    floating.
    """
    return ev.Observation(
        origin=ev.Origin.LAYER_NAME,
        detail=f"layer {layer!r} is in this drawing's layer table",
        observed=layer,
        locator=layer,
        layout=layout,
    )


def _scope(layer: str, layout: str | None, units: Mapping[str, Any]) -> ev.Scope:
    if layout:
        where = f"only entities on layout {layout!r}"
        includes_blocks = layout.startswith(BLOCK_LAYOUT_PREFIX)
    else:
        where = "every entity in this drawing, on any layout"
        includes_blocks = True
    return ev.Scope(
        layout=layout,
        includes_block_definitions=includes_blocks,
        units=units,
        note=(
            f"the tag applies to layer {layer!r}: {where}. One label for one "
            "typology, applying to all of its objects; there are no per-object "
            "tags."
        ),
    )


#: The ceiling of every claim that leaves this module. Stated, not relied on:
#: it is already correct today because a tag claim carries no tokens and
#: `OUTSIDE_FILE` is in `NEVER_SPEAKS`, but what states the INTENT is this
#: line, and that is what the next person reads.
TAG_CEILING: Final[ev.Grade] = ev.Grade.INFERRED

_NOT_ESTABLISHED_TAGGED = (
    "this file states no business name for any layer; the value comes from a "
    "person, not from the drawing"
)
_HOW_TO_VERIFY = (
    "ask Roshn for the typology key, or for the planning document that maps "
    "layer codes to product types; the meaning of those codes is not in any "
    "file"
)


def _provenance(
    tag: Mapping[str, Any] | None, config_entry: Mapping[str, Any] | None
) -> ev.Provenance:
    """Which layer decided this value.

    A screen tag is an override for THIS drawing, hence `DRAWING_OVERRIDE`,
    and the `verified` derived from it is true — a human did confirm the
    mapping. That does not raise `grade` in the slightest; the two are
    different axes.
    """
    if tag:
        return ev.Provenance(
            config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
            config_version=tag.get("tag_contract_version"),
            note=str(tag.get("source") or ""),
        )
    if config_entry:
        raw = str(config_entry.get("config_layer") or ev.ConfigLayer.GLOBAL_PATTERN.value)
        try:
            layer = ev.ConfigLayer(raw)
        except ValueError:
            layer = ev.ConfigLayer.GLOBAL_PATTERN
        return ev.Provenance(
            config_layer=layer,
            config_version=config_entry.get("config_version"),
            note=str(config_entry.get("note") or ""),
        )
    return ev.Provenance(config_layer=ev.ConfigLayer.NO_CONFIG)


def _claim(value: str) -> ev.Claim:
    """A claim without tokens, always. See note (1) at the head of this file."""
    return ev.Claim(value=value, tokens=())


# --- effective value ---------------------------------------------------------


def effective_layer(
    *,
    drawing_id: str,
    layer: str,
    units: Mapping[str, Any],
    layout: str | None = None,
    config_entry: Mapping[str, Any] | None = None,
    tag: Mapping[str, Any] | None = None,
    impact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The value in force for one layer: config overridden by tag, both visible.

    `units` and `config_entry` are handed over by the caller, not fetched
    here. Units because the evidence module demands them handed over verbatim
    (11 of 16 drawings in inches, 3 declaring nothing at all); config because
    the file that holds it belongs to another subagent and its shape is still
    moving. Without either of them this function still answers — with an
    `unknown` that names what is missing.
    """
    if tag:
        if tag.get("drawing_id") != drawing_id:
            raise ev.EvidenceError(
                "CROSS_DRAWING",
                f"a tag belonging to {tag.get('drawing_id')!r} was used for "
                f"drawing {drawing_id!r}.",
                "A layer name that means school in this drawing does not "
                "necessarily mean school in another drawing (G10). A source "
                "outside the file is the easiest path for moving knowledge "
                "between drawings, so that path is closed here.",
            )
        if tag.get("target") != layer:
            raise TagRefused(
                "TAG_TARGET_MISMATCH",
                f"a tag for layer {tag.get('target')!r} was used for layer {layer!r}.",
                "Fetch the tag through `tags_by_layer()` so that its target "
                "never comes loose from the layer being read.",
            )

    scope = _scope(layer, layout, units)
    provenance = _provenance(tag, config_entry)
    observations = [_layer_observation(layer, layout)]
    if tag:
        observations.append(tag_observation(tag))

    config_name = (config_entry or {}).get("business_name")
    config_use = (config_entry or {}).get("use")
    name = (tag or {}).get("business_name") or config_name or None
    use = (tag or {}).get("use") or config_use or None
    source_of = "user_tag" if tag else ("config" if config_entry else None)

    payload: dict[str, Any] = {
        "drawing_id": drawing_id,
        "layer": layer,
        "layout": layout,
        "scope_note": scope.note,
        "tag_contract_version": TAG_CONTRACT_VERSION,
        "business_tag": name,
        "tag_source": source_of,
        "tagged_by": (tag or {}).get("author"),
        "tagged_at": (tag or {}).get("updated_at"),
        "tag_note": (tag or {}).get("source"),
        "overridden": bool(tag and config_entry),
        # Both stay visible, whichever one wins. An override that hides the
        # value it overrode makes "why did the answer change" unanswerable by
        # anyone.
        "config_value": (
            {
                "business_name": config_name,
                "use": config_use,
                "config_layer": (config_entry or {}).get("config_layer"),
                "note": (config_entry or {}).get("note"),
            }
            if config_entry
            else None
        ),
        "config_available": config_entry is not None,
        "impact": dict(impact) if impact else None,
    }
    if use is not None:
        payload["use"] = use

    blocks: dict[str, ev.Evidence] = {"business_name": _name_evidence(
        drawing_id=drawing_id,
        layer=layer,
        name=name,
        scope=scope,
        provenance=provenance,
        observations=observations,
    )}
    if use is not None:
        blocks["use"] = _use_evidence(
            drawing_id=drawing_id,
            use=use,
            scope=scope,
            provenance=provenance,
            observations=observations,
            from_tag=bool(tag),
        )

    return ev.attach(payload, blocks)


def _name_evidence(
    *,
    drawing_id: str,
    layer: str,
    name: str | None,
    scope: ev.Scope,
    provenance: ev.Provenance,
    observations: Sequence[ev.Observation],
) -> ev.Evidence:
    if name is None:
        return ev.Evidence.unknown(
            _claim(f"business name for layer {layer!r}"),
            drawing_id=drawing_id,
            scope=scope,
            provenance=provenance,
            observations=tuple(observations),
            not_established=(
                "this layer has not been tagged from the screen and is not in "
                "this drawing's land use config; the file itself names no "
                "business name for any typology code"
            ),
            how_to_verify=(
                "tag it from the layer panel — its source is mandatory — or "
                "add this layer to this drawing's land use config through a "
                "merge request"
            ),
        )
    return ev.Evidence.of(
        _claim(name),
        drawing_id=drawing_id,
        scope=scope,
        provenance=provenance,
        observations=tuple(observations),
        not_established=_NOT_ESTABLISHED_TAGGED,
        how_to_verify=_HOW_TO_VERIFY,
        ceiling=TAG_CEILING,
    )


def _use_evidence(
    *,
    drawing_id: str,
    use: str,
    scope: ev.Scope,
    provenance: ev.Provenance,
    observations: Sequence[ev.Observation],
    from_tag: bool,
) -> ev.Evidence:
    """The effective land use — deliberately more conservative than UPLIFT-02.

    This block is an echo, not a computation: when the value comes from the
    config, the authority on its grade is `land_use_summary`, which reads the
    vocabulary and can reach `stated`. Here the ceiling stays `inferred`, and
    that is safe precisely because it can only move down — an echo stronger
    than its original is a grade rise in disguise.
    """
    return ev.Evidence.of(
        _claim(use),
        drawing_id=drawing_id,
        scope=scope,
        provenance=provenance,
        observations=tuple(observations),
        not_established=(
            _NOT_ESTABLISHED_TAGGED
            if from_tag
            else "an echo from the config; the authoritative grade comes from land_use_summary"
        ),
        how_to_verify=(
            _HOW_TO_VERIFY
            if from_tag
            else "read land_use_summary for this drawing; that is what reads the vocabulary"
        ),
        ceiling=TAG_CEILING,
    )


# --- export to the UPLIFT-02 config ------------------------------------------


_YAML_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _yq(value: Any) -> str:
    """A double-quoted YAML scalar, always, whatever it contains.

    A business name is text typed by a human; it will contain colons, hashes,
    quotes, and now and then a newline. Quoting only "when necessary" means
    guessing when it is necessary, and a wrong guess produces a config file
    that cannot be read — or worse, one that can be read with a different
    meaning.
    """
    if value is None:
        return "null"
    out: list[str] = []
    for ch in str(value):
        if ch in _YAML_ESCAPES:
            out.append(_YAML_ESCAPES[ch])
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _stamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def export_yaml(drawing_id: str, rows: Iterable[Mapping[str, Any]]) -> str:
    """One drawing's tags as YAML in the shape of an UPLIFT-02 config.

    The flow: tag from the screen → export → review → commit as config. That
    way business decisions finally land in the repo instead of staying in a
    single database that has no review history.

    Its output is deterministic and carries no generation timestamp: a
    timestamp at the head of the file would make every export produce a diff
    even when not a single tag changed, and a false diff is a diff people stop
    reading.
    """
    items = [dict(r) for r in rows]
    foreign = sorted({str(r.get("drawing_id")) for r in items if r.get("drawing_id") != drawing_id})
    if foreign:
        raise ev.EvidenceError(
            "CROSS_DRAWING",
            f"the export for {drawing_id!r} contains tags from {', '.join(foreign)}.",
            "One config file belongs to one drawing; the classification is "
            "re-read per drawing (G10).",
        )
    items.sort(key=lambda r: str(r.get("target", "")))

    lines = [
        f"# Business tags for drawing {drawing_id}, exported from the {COLL_TAGS} collection.",
        "# Flow: tag from the screen -> export -> review -> commit as an UPLIFT-02 config.",
        "#",
        "# There is no `verified:` key here, and that is deliberate. UPLIFT-08 made",
        "# `verified` DERIVED from the config layer that decides, not something that is",
        "# typed: a boolean that can be written by hand, accompanied by `source` prose,",
        "# IS a mechanism for raising the evidence grade in YAML form. What is mandatory",
        "# and what remains is `source` — a sentence that a person can trace.",
        "#",
        "# A business name is never stated by any DXF file, so a claim born from this",
        "# file will never rise above `inferred`.",
        f"tag_contract_version: {TAG_CONTRACT_VERSION}",
        f"drawing_id: {_yq(drawing_id)}",
    ]
    if not items:
        lines += [
            "# No tag has been recorded for any layer of this drawing yet.",
            "layers: {}",
        ]
        return "\n".join(lines) + "\n"

    lines.append("layers:")
    for row in items:
        lines.append(f"  {_yq(row.get('target'))}:")
        lines.append(f"    business_name: {_yq(row.get('business_name'))}")
        lines.append(f"    use: {_yq(row.get('use'))}")
        lines.append(f"    source: {_yq(row.get('source'))}")
        lines.append(f"    tagged_by: {_yq(row.get('author'))}")
        lines.append(f"    tagged_at: {_yq(_stamp(row.get('updated_at')))}")
    return "\n".join(lines) + "\n"
