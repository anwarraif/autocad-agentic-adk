"""Draft a land-use config for a drawing that has none — DOSSIER Phase 3.

Owner: the Phase 3 lane (`docs/DOSSIER-00-MASTER.md`, Phase 3). Read with
`docs/UPLIFT-GENERALITY-RULES.md` (G3, G4, G9, G10) and
`cad_api/app/landuse/596212db022a3397.yaml`, which is the hand-written shape
this module drafts towards.

The hole this closes
--------------------
A drawing arrives with no `landuse/<drawing_id>.yaml`. Today it is ingested,
it renders, and every one of its layers is `unclassified` — on the reference
drawing that would be 292 layers with no meaning attached to any of them. Rule
G3 says that state is correct: absence of config must be answered gracefully
and never with a guess. Rule G9 says the drawing is never rejected for it. But
"correct and useless" is where this campaign started, and the answer is not to
guess — it is to write down, for a human, everything the file itself already
says about each layer, and to mark plainly where it says nothing.

The two sources of evidence, and why they must stay apart
---------------------------------------------------------
* **The global patterns** (`landuse/_patterns.yaml`) read the layer NAME and
  propose a `use`. A name is a claim by whoever drafted the file. It is a
  proposal, never a fact — sixty Civil3D `C-ROAD-*` layers carry the word
  "road" and every one of them is sheet layout.
* **The Dossier** (Phase 1, read through `dossier_read`) reads the GEOMETRY
  and proposes a `role`. A role is a measured property of the entities on the
  layer: closed rings, open paths, block inserts, text. It says nothing about
  meaning, which is exactly why it can be trusted about shape.

The name proposes; the geometry disposes. Both travel into the draft, each
attributed to its own source, and neither is ever silently merged into the
other. There is no map anywhere in this file from a `use` to a `role` (G1).

Why nothing in the drafted file is live
---------------------------------------
`verified` is derived, not typed: `evidence.Provenance.verified` is true when —
and only when — the classification came from the `drawing_override` layer, i.e.
from `landuse/<drawing_id>.yaml`. So writing a pattern hit into that file does
not merely record it; it **promotes** it from "a pattern proposed this" to "a
human confirmed this", with no human anywhere near it. That is the exact defect
this campaign exists to end, one layer down.

Hence the shape of the output, which is the one design decision everything else
follows from:

    The drafted file's `layers:` block is EMPTY when it is written. Every
    proposal sits beside it as a commented, ready-to-uncomment entry.

While the proposals stay commented, a pattern hit keeps being served by the
GLOBAL PATTERN layer — `config_layer: global_pattern`, `verified: false`, which
is already true the day the drawing lands (G4). Uncommenting a line is a human
signing for it, and only then does it become an override worth a `verified:
true`. The 3-character prefix is the signature.

Two markers, and the difference between them matters
-----------------------------------------------------
    `#> `  a COMPLETE proposal. Delete the three characters and the loader
           will read it. Every one of these carries both what it claims and
           how it knows.
    `#? `  an INCOMPLETE template: something a human must decide before the
           layer can be written down at all. It is deliberately NOT
           acceptable by deleting a prefix, because the thing missing from it
           is a decision, and a template that could be accepted in one
           mechanical pass would be accepted in one mechanical pass.

The commonest reason for `#? ` is the loader's default role. `landuse.
DEFAULT_ROLE` is `parcel`, so an entry that omits `role:` is an entry that
claims the layer holds plots. A layer whose geometry could not name a role
therefore cannot be offered as ready: accepting it would sign for `parcel`
without anyone choosing it.

Geometry to role — the whole mapping, stated (hard rule 3)
-----------------------------------------------------------
| Dossier role | drafted `role` | why                                        |
|--------------|----------------|--------------------------------------------|
| `region`     | `parcel`       | closed rings are the only geometry a plot can be. It says the rings are CLOSED, not that they are plots — a block boundary, a hatch outline and a site boundary are closed rings too. This is the proposal a reviewer most often has to say no to, and it is offered because `parcel` is also what the loader would default to anyway. |
| `network`    | `network`      | open path geometry, whose native measure is length. The config role of the same name exists for precisely this and reports Σ length instead of a parcel count of zero. |
| `annotation` | `annotation`   | TEXT/MTEXT/DIMENSION. The role exists so that labels are never counted as plots; on the reference drawing one annotation layer once made the unit mix wrong by 416. |
| `points`     | **none**       | repeated block inserts. The config vocabulary has no `points` role, and a tree, a street light and a plot-corner marker are the same geometry. `overlay` and `parcel` are both decisions, so a human makes them. |
| `mixed`      | **none**       | no type family reached the Dossier's dominance threshold. Ambiguity IS the finding here; picking a role would erase it. |
| no profile   | **none**       | either the layer holds no entity, or no Dossier has been built. There is no geometry to read, so nothing is claimed about it. |

The `use` side has no such table and never will: a use is proposed only when a
pattern in `_patterns.yaml` matches the layer name, and a layer that matches no
pattern gets **no use at all** — it appears in the draft with its geometric
profile and an explicit marker saying a human is needed. Silence and invention
are both wrong; naming the gap is right.

G10 — nothing walks in from another drawing
-------------------------------------------
Every fact emitted for a drawing comes from that drawing's own layer table, its
own Dossier, and the global patterns, which are config that applies to any
drawing by design (G4). This module holds no memory between calls, no cache
keyed by layer name, and no corpus of "layers we have seen before".

Where the draft is written
--------------------------
Never into `cad_api/app/landuse/`. `write_draft()` refuses that directory by
name, because a file that lands there is a file the loader believes. A human
moves it — that move is the second half of the signature.

Size (G7)
---------
The YAML is deliberately NOT capped. It is a file for a person to read once,
not a response that has to fit in an agent's context, and a layer dropped for
space is a layer nobody will ever classify. What IS capped is the report
`report_lines()` prints, which is counts only.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from . import landuse

# --- Markers, and the file name ----------------------------------------------

#: Prefix of a COMPLETE proposal. Deleting it — and nothing else — turns the
#: line into config the loader reads. Three characters, chosen so that the
#: entry underneath keeps its two-space indent under `layers:`.
ACCEPT_MARKER: str = "#> "

#: Prefix of an INCOMPLETE template. Never mechanically acceptable, on purpose:
#: what is missing from it is a human decision.
DECIDE_MARKER: str = "#? "

#: Suffix of the drafted file. Deliberately NOT `<drawing_id>.yaml`, which is
#: the name `landuse.config_path()` looks for: a draft that is copied into the
#: config directory by accident must still not be read as a config.
DRAFT_SUFFIX: str = ".draft.yaml"

#: Environment variable that names the review directory. Read directly rather
#: than through `app.config.Settings`, which this module does not own.
DRAFT_DIR_ENV: str = "CAD_LANDUSE_DRAFT_DIR"

#: Directory name used under whichever root is chosen.
DRAFT_DIR_NAME: str = "landuse_drafts"


# --- The geometry-to-role mapping (hard rule 3) -------------------------------

#: The only geometric roles that name a config role. Every entry here is a
#: statement about SHAPE; not one of them says what the layer is for.
ROLE_FOR_GEOMETRY: Mapping[str, str] = {
    "region": "parcel",
    "network": "network",
    "annotation": "annotation",
}

#: Geometric roles that deliberately propose nothing, with the reason that
#: travels into the draft beside the layer. Ambiguity is reported, not resolved.
GEOMETRY_WITHOUT_A_ROLE: Mapping[str, str] = {
    "points": (
        "the Dossier profiles this layer as `points` — repeated block inserts. "
        "The config role vocabulary has no `points`, and a tree, a street light "
        "and a plot-corner marker are the same geometry: `overlay` and `parcel` "
        "are both decisions, so neither is proposed here"
    ),
    "mixed": (
        "the Dossier profiles this layer as `mixed`: no type family reached its "
        "stated dominance threshold, so the geometry itself is undecided. "
        "Ambiguity is the finding; choosing a role would erase it"
    ),
}

#: What the loader does with an entry that names no role. Quoted into the
#: drafted file, because it is the reason an unresolved role is never offered
#: as ready-to-accept.
_DEFAULT_ROLE_TRAP = (
    f"the loader defaults a missing `role:` to {landuse.DEFAULT_ROLE!r}, which "
    "is the only role counted as a plot"
)


class DraftRefused(RuntimeError):
    """A draft was about to be written somewhere it must never be written.

    Shaped like `landuse.ConfigError` (`code`/`message`/`hint`) so a caller can
    publish it as an actionable message rather than a traceback.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


# --- Status vocabulary --------------------------------------------------------

#: The status axis is EVIDENCE, not readiness, and the two are deliberately
#: separate. A layer can carry a full geometric profile and still not be
#: acceptable as written (its geometry names no role), and saying "nothing is
#: known about it" in that case would be false — the profile is right there in
#: the file. `ready` answers the other question.
#:
#: A global pattern matched the layer NAME and proposed a use.
STATUS_USE_PROPOSED: str = "use_proposed"

#: No pattern matched. The Dossier profiles the layer, so its geometry is in
#: the draft — its role, its entity count, its native measure — and no use is
#: proposed for it at all.
STATUS_GEOMETRY_ONLY: str = "geometry_only"

#: Nothing but the layer's own name is known: no pattern matched it and the
#: Dossier profiles no bucket for it (it holds no entity, or none has been
#: built). The layer is still listed, because a layer omitted from a review
#: file is a layer nobody will ever look at.
STATUS_NAME_ONLY: str = "name_only"

#: A human already wrote this layer into the drawing's config. Left alone.
STATUS_ALREADY: str = "already_in_config"


# --- Small readers ------------------------------------------------------------


def _text(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def declared_layers(drawing: Mapping[str, Any]) -> list[dict[str, Any]]:
    """`[{name, entity_count}]` from the drawing's own layer table.

    Tolerates both shapes the store has held: a list of layer documents and a
    bare list of names. A layer table that cannot be read comes back empty and
    the caller says so — it never comes back as "this drawing has no layers".
    """
    raw = drawing.get("layers")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            name = _text(item.get("name"))
            if name is None:
                continue
            out.append({"name": name, "entity_count": _count(item.get("entity_count"))})
        else:
            name = _text(item)
            if name is not None:
                out.append({"name": name, "entity_count": None})
    return out


def _profile_facts(profile: Any) -> dict[str, Any] | None:
    """The handful of fields the draft needs out of a `dossier_read` profile.

    Returns `None` for a layer the Dossier does not profile — and for a row
    that came back with `profiled: false`, which means "we stopped looking",
    not "there is nothing there". The two are distinguished by the caller
    through `looked`, below.
    """
    if not isinstance(profile, Mapping):
        return None
    if profile.get("profiled") is False:
        return {
            "looked": False,
            "why": _text(profile.get("why")),
            "role": None,
        }
    measure = profile.get("measure")
    measure = measure if isinstance(measure, Mapping) else {}
    scope = profile.get("scope")
    scope = scope if isinstance(scope, Mapping) else {}
    anomalies = profile.get("anomalies")
    kinds = (
        list(anomalies.get("kinds") or [])
        if isinstance(anomalies, Mapping)
        else []
    )
    leader = profile.get("role_leader")
    leader = leader if isinstance(leader, Mapping) else {}
    return {
        "looked": True,
        "why": None,
        "role": _text(profile.get("role")),
        "layout": _text(profile.get("layout")),
        "entities": _count(profile.get("entities")),
        "entities_all_buckets": _count(scope.get("entities_all_buckets")),
        "buckets": _count(scope.get("buckets_for_this_layer")),
        "measure_kind": _text(measure.get("kind")),
        "measure_value": measure.get("value"),
        "unit": _text(measure.get("unit")),
        "unit_reason": _text(measure.get("unit_reason")),
        "measured_entities": _count(measure.get("measured_entities")),
        "unmeasured_entities": _count(measure.get("unmeasured_entities")),
        "anomaly_kinds": [str(k) for k in kinds],
        "role_share": leader.get("share"),
        "role_runner_up": _text(leader.get("runner_up")),
    }


def measure_phrase(facts: Mapping[str, Any] | None) -> str:
    """One clause naming a bucket's native measure, with its unit (G2).

    A unit that the file does not declare is written as such. There is no
    default metre here and there must not be: 11 of the drawings in this store
    are in inches and 3 declare nothing at all. A value that was never
    established is reported as absent, never as 0 (G8).

    A `count` carries no unit and is never given one. Writing "count 40 drawing
    units" would attach a dimension to a dimensionless number — the same class
    of mistake as writing "m" on a drawing that declares no unit, in the
    opposite direction.
    """
    if not facts or not facts.get("looked"):
        return "no measure (this layer was not profiled)"
    kind = facts.get("measure_kind") or "measure"
    value = facts.get("measure_value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return (
            f"{kind}: no value was established — absent, which is not zero (G8)"
        )
    if kind == "count":
        return f"count {value} entities"
    unit = facts.get("unit")
    unit_text = unit if unit else "drawing units (this bucket declares none)"
    out = f"{kind} {value} {unit_text}"
    measured = facts.get("measured_entities")
    unmeasured = facts.get("unmeasured_entities")
    if measured is not None and unmeasured:
        out += f", from {measured} of {measured + unmeasured} entities"
    return out


def _bucket_phrase(facts: Mapping[str, Any] | None) -> str:
    """One bucket's entity count and its native measure, said once each.

    A `count` measure already IS the entity count, so writing "18 entities,
    count 18 entities" states the same figure twice and invites a reader to
    believe they are two.
    """
    if not facts or not facts.get("looked"):
        return measure_phrase(facts)
    if facts.get("measure_kind") == "count":
        return measure_phrase(facts)
    return f"{facts.get('entities')} entities, {measure_phrase(facts)}"


# --- The one decision this module makes on its own ----------------------------


def role_from_geometry(role: Any) -> tuple[str | None, str]:
    """`(config role, basis)` from a Dossier geometric role.

    The whole mapping is the table in this module's docstring, and it is the
    only place geometry becomes a config value. A role that maps to nothing
    returns `(None, why)` — the ambiguity travels into the draft as a sentence
    rather than being resolved by a tie-break nobody would ever read.

    A role is a geometric FACT and this function never sees a layer name, so it
    cannot be influenced by one (G1).
    """
    name = _text(role)
    if name is None:
        return None, (
            "the Dossier names no geometric role for this layer, so none is "
            "proposed. That is not a statement that the layer is empty"
        )
    mapped = ROLE_FOR_GEOMETRY.get(name)
    if mapped is not None:
        return mapped, (
            f"the Dossier profiles this layer's geometry as {name!r}, which is "
            f"drafted as config role {mapped!r}. Role is decided from DXF type "
            "and ring status alone; the layer name is never read (G1)"
        )
    return None, GEOMETRY_WITHOUT_A_ROLE.get(
        name,
        f"the Dossier profiles this layer as {name!r}, which names no role in "
        f"the config vocabulary ({', '.join(sorted(landuse.ROLES))}), so none "
        "is proposed",
    )


# --- One layer ----------------------------------------------------------------


def draft_entry(
    layer: str,
    *,
    profile: Any = None,
    rule: landuse.PatternRule | None = None,
    declared_entities: int | None = None,
    in_layer_table: bool = True,
) -> dict[str, Any]:
    """Everything the draft knows about ONE layer. Pure; touches no database.

    `rule` is the global pattern that matched the layer name, or `None`.
    `profile` is that layer's `dossier_read` profile, or `None`.

    The two are read separately and stay separately attributed all the way into
    the YAML: `use` may only ever come from `rule`, `role` may only ever come
    from `profile`. A layer with no `rule` gets **no use** — not `unknown`
    dressed as a decision, not the commonest use, nothing.
    """
    facts = _profile_facts(profile)
    role, role_basis = role_from_geometry(facts.get("role") if facts else None)

    use = _text(rule.use) if rule is not None else None
    subtype = _text(rule.subtype) if rule is not None else None

    entities = None
    if facts and facts.get("looked"):
        entities = facts.get("entities_all_buckets")
        if entities is None:
            entities = facts.get("entities")
    if entities is None:
        entities = declared_entities

    empty = declared_entities == 0 and not (facts and facts.get("looked"))

    profiled = bool(facts and facts.get("looked"))
    if use:
        status = STATUS_USE_PROPOSED
    elif profiled:
        status = STATUS_GEOMETRY_ONLY
    else:
        status = STATUS_NAME_ONLY

    # Two different kinds of "a human is needed", and collapsing them is how a
    # review file stops being read.
    #
    #   blocking  — the entry cannot be written down at all until someone
    #               decides. Only ever the role, and only because
    #               `landuse.DEFAULT_ROLE` turns a missing role into a claim.
    #   open      — the entry can be written and is still incomplete. A layer
    #               with no use is the case: the entry records that inside
    #               itself, as `unknown:` plus `how_to_verify:`, which is how
    #               this repo has always carried a gap.
    blocking: list[str] = []
    open_questions: list[str] = []
    if not role and not empty:
        blocking.append(f"the role — {role_basis}; {_DEFAULT_ROLE_TRAP}")
    if not use:
        open_questions.append(
            "the use — no global pattern matches this layer's name, so nothing "
            "proposes one. The entry carries that as `unknown:` rather than "
            "guessing, and it is still a question for a human"
        )

    # A name that proposes a land use over geometry that is not made of closed
    # rings. Stated without any table from a use to a role (G1): what is said
    # here is a fact about the ROLE alone, and it is the same signal that tells
    # a Civil3D sheet-layout layer from a feature layer with the same word in
    # its name.
    tension: str | None = None
    if use and role and role != landuse.PARCEL_ROLE:
        tension = (
            f"the NAME proposes a land use while the GEOMETRY profiles as "
            f"{str(facts.get('role'))!r} (config role {role!r}) — this layer is "
            "not made of closed rings, so it holds no plot to classify. Layers "
            "that carry a land-use word and draw something else are usually "
            "sheet layout or labelling; check before accepting"
        )

    # Ready means two things at once: nothing is blocking, AND the entry
    # actually claims something. A declared-but-empty layer that no pattern
    # names claims nothing at all — writing it down would add a line that says
    # only "this layer exists", which the layer table already said.
    ready = bool(use or role) and not blocking

    entry = _entry_mapping(
        layer,
        use=use,
        subtype=subtype,
        role=role,
        rule=rule,
        facts=facts,
        role_basis=role_basis,
        empty=empty,
        tension=tension,
    )

    return {
        "layer": layer,
        "status": status,
        "ready": ready,
        "blocking": blocking,
        "open_questions": open_questions,
        "tension": tension,
        "use": use,
        "subtype": subtype,
        "matched_pattern": _text(rule.match) if rule is not None else None,
        "role": role,
        "role_basis": role_basis,
        "geometric_role": facts.get("role") if facts else None,
        "declared_entities": declared_entities,
        # Two different figures, and they are kept apart on purpose: `entities`
        # is the layer's total over every bucket and is what the draft ranks
        # by, while `bucket_entities` is the ONE bucket the measure beside it
        # was computed over. Printing the first next to the second would be a
        # number labelled by the wrong question.
        "entities": entities,
        "bucket_entities": facts.get("entities") if facts and facts.get("looked") else None,
        "declared_but_empty": empty,
        "profiled": bool(facts and facts.get("looked")),
        "profile_gap": None if not facts else facts.get("why"),
        "in_layer_table": in_layer_table,
        "measure": measure_phrase(facts),
        "profile_phrase": _bucket_phrase(facts),
        "anomaly_kinds": list(facts.get("anomaly_kinds") or []) if facts else [],
        "layout": facts.get("layout") if facts and facts.get("looked") else None,
        "buckets": facts.get("buckets") if facts and facts.get("looked") else None,
        "entry": entry,
    }


def _entry_mapping(
    layer: str,
    *,
    use: str | None,
    subtype: str | None,
    role: str | None,
    rule: landuse.PatternRule | None,
    facts: Mapping[str, Any] | None,
    role_basis: str,
    empty: bool,
    tension: str | None = None,
) -> dict[str, Any]:
    """The YAML mapping for one layer: what is claimed, and how it is known.

    `sources` is the part that matters. Every claim in the entry above it is
    traceable to one line here, and a claim with no line here is not written at
    all — which is why a layer with no pattern hit carries `use: unknown`
    rather than a use: `unknown` is the loader's word for "no decision", and
    the absence source underneath says who decided nothing.
    """
    sources: list[dict[str, Any]] = []
    if rule is not None:
        sources.append(
            {
                "origin": "layer_name",
                "observed": layer,
                "locator": layer,
                "detail": (
                    f"the layer name matches the global pattern `{rule.match}` "
                    "in landuse/_patterns.yaml. A pattern match is a proposal "
                    "about the NAME and states nothing about what the layer "
                    "holds — while this entry stays commented out it is served "
                    "by the global_pattern layer at verified: false"
                ),
            }
        )
    else:
        sources.append(
            {
                "origin": "absence",
                "locator": layer,
                "detail": (
                    "no pattern in landuse/_patterns.yaml matches this layer "
                    "name, so nothing proposes a land use for it. The entry "
                    "below records what its GEOMETRY is; what it is FOR is "
                    "not established"
                ),
            }
        )

    if facts and facts.get("looked") and facts.get("role"):
        detail = (
            f"the Dossier profiles this layer as role "
            f"{str(facts.get('role'))!r} in layout {str(facts.get('layout'))!r}: "
            f"{_bucket_phrase(facts)}. "
            "Decided from DXF type and ring status alone (G1)"
        )
        if facts.get("buckets") and int(facts["buckets"]) > 1:
            detail += (
                f"; this layer holds {facts['buckets']} (layer x layout) "
                "buckets and they are never summed (G2)"
            )
        sources.append({"origin": "geometry", "locator": layer, "detail": detail})
    elif empty:
        sources.append(
            {
                "origin": "absence",
                "locator": layer,
                "detail": (
                    "the drawing's own layer table declares this layer and it "
                    "holds 0 entities, so no geometry could name a role for it"
                ),
            }
        )

    entry: dict[str, Any] = {"use": use or landuse.UNKNOWN_USE}
    if subtype:
        entry["subtype"] = subtype
    if role:
        entry["role"] = role
    entry["sources"] = sources
    if not use:
        entry["unknown"] = (
            "what this layer is used for. Its name matches no global pattern, "
            "and its geometry says what shape it is, never what it is for"
        )
        entry["how_to_verify"] = (
            "ask the drawing's author what this layer holds; or, if the name is "
            "one a whole family of drawings shares, add a pattern for it to "
            "landuse/_patterns.yaml so every drawing gains it at once"
        )
    notes: list[str] = []
    if tension:
        notes.append(tension)
    if not use and role:
        notes.append(
            "accepting this records what the layer IS and leaves what it is FOR "
            "open. That has one consequence worth knowing: a layer named in a "
            "config is no longer `unclassified`, so the residual that "
            "land_use_summary attaches to a use reporting zero will no longer "
            "surface it. Accept it when the role is worth recording — a "
            "`network` layer then answers in LENGTH instead of `0 parcels` — "
            "and leave it commented if you would rather it kept showing up as "
            "unmapped"
        )
    if empty:
        notes.append(
            "declared in the layer table and holds ZERO entities. No role is "
            f"named because there is no geometry to read one from, and "
            f"{_DEFAULT_ROLE_TRAP} — which has nothing to count here"
        )
    elif not role:
        notes.append(role_basis)
    if notes:
        entry["note"] = ". ".join(notes)
    return entry


# --- The whole drawing --------------------------------------------------------


def build_draft(
    drawing: Mapping[str, Any],
    *,
    profiles: Mapping[str, Any] | None = None,
    pattern_set: landuse.PatternSet | None = None,
    config: landuse.DrawingConfig | None = None,
    layout: str | None = None,
    profiles_note: str | None = None,
) -> dict[str, Any]:
    """The whole draft for one drawing. Pure when `profiles` is handed in.

    `drawing` is the `autocad_drawings` document — its `_id` and its own layer
    table are the only things read from it. `profiles` maps layer name to that
    layer's `dossier_read` profile; `{}` is a legitimate input and produces a
    draft in which every layer is name-only, which is exactly what a drawing
    with no Dossier deserves to look like.

    `config` is the drawing's existing override, if it has one. Layers a human
    has already written there are reported and left ALONE: re-proposing what
    someone already decided is how a draft starts arguing with its reviewer.
    """
    drawing_id = _text(drawing.get("_id")) or _text(drawing.get("drawing_id"))
    if not drawing_id:
        raise ValueError(
            "a draft is keyed by drawing_id (a content hash); without one the "
            "file could not be named, and a config that is not keyed to a file "
            "is meaning that walks between drawings (G10)."
        )

    pats = pattern_set if pattern_set is not None else landuse.patterns()
    known = dict(profiles or {})
    table = declared_layers(drawing)
    names = [row["name"] for row in table]
    entity_counts = {row["name"]: row["entity_count"] for row in table}

    # A layer the Dossier profiles but the layer table does not declare is
    # still this drawing's own layer. It is included and marked, because a
    # layer table that disagrees with the entities is a finding.
    for extra in known:
        if extra not in entity_counts:
            names.append(str(extra))
            entity_counts[str(extra)] = None

    already = set(config.layers) if config is not None else set()

    entries: list[dict[str, Any]] = []
    for name in names:
        if name in already:
            entries.append(
                {
                    "layer": name,
                    "status": STATUS_ALREADY,
                    "ready": False,
                    "needs": [],
                    "use": None,
                    "role": None,
                    "entities": entity_counts.get(name),
                    "entry": None,
                }
            )
            continue
        entries.append(
            draft_entry(
                name,
                profile=known.get(name),
                rule=pats.first_match(name),
                declared_entities=entity_counts.get(name),
                in_layer_table=name in {row["name"] for row in table},
            )
        )

    entries.sort(key=lambda e: (-(e.get("entities") or 0), str(e["layer"])))

    return {
        "drawing_id": drawing_id,
        "drawing_name": _drawing_name(drawing),
        "source_file": _text(drawing.get("original_filename")),
        "drafted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layout": layout,
        "pattern_version": pats.version,
        "patterns_available": len(pats.rules),
        "config_present": config is not None,
        "config_path": config.path if config is not None else None,
        "profiles_note": profiles_note,
        "layers_in_table": len(table),
        "entries": entries,
        "counts": counts(entries),
    }


def _drawing_name(drawing: Mapping[str, Any]) -> str:
    filename = _text(drawing.get("original_filename"))
    if filename:
        return Path(filename).stem
    return (
        _text(drawing.get("name"))
        or _text(drawing.get("_id"))
        or ""
    )


def counts(entries: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """The numbers the readiness report prints, each one labelled by its key.

    They are deliberately not one number. "34 layers unclassified" is the
    sentence this campaign exists to improve on: what a reviewer needs to know
    is how many of them a machine could propose something for, how many carry
    geometry and no meaning, and how many are nothing but a name.
    """
    rows = list(entries)
    drafted = [r for r in rows if r.get("status") != STATUS_ALREADY]
    return {
        "layers": len(rows),
        "already_in_config": len(rows) - len(drafted),
        "drafted": len(drafted),
        "use_proposed": sum(
            1 for r in drafted if r.get("status") == STATUS_USE_PROPOSED
        ),
        "geometry_only": sum(
            1 for r in drafted if r.get("status") == STATUS_GEOMETRY_ONLY
        ),
        "name_only": sum(
            1 for r in drafted if r.get("status") == STATUS_NAME_ONLY
        ),
        "ready_to_accept": sum(1 for r in drafted if r.get("ready")),
        "needs_a_decision": sum(1 for r in drafted if not r.get("ready")),
        "needs_a_human": len(drafted),
        "profiled": sum(1 for r in drafted if r.get("profiled")),
        "unprofiled": sum(1 for r in drafted if not r.get("profiled")),
    }


# --- Reading the store (guarded, exactly as the sibling lanes do) -------------


def profiles_from_store(
    drawing_id: str,
    layer_names: Sequence[str],
    *,
    layout: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """`(profiles, why_they_are_missing)` from the Dossier. Never raises.

    `dossier_read` is imported behind a guard, as every Dossier caller in this
    repo does, so a draft is still produced when the module or the store is not
    there — with every layer name-only and a sentence saying why. A drafting
    tool that refuses to run without a Dossier would be one more reason a new
    drawing gets turned away at the door (G9).

    The Dossier is read ONCE and the layer names are then asked for in chunks
    of `dossier_read.MAX_PROFILE_LAYERS`. That cap is a per-response size
    budget, and its own message says the remedy is a smaller call; a drawing
    with 292 layers is 8 such calls over one already-fetched document, not 292
    trips to Mongo.
    """
    try:
        from . import dossier_read
    except Exception as exc:  # noqa: BLE001 -- guarded on purpose, see docstring
        return {}, (
            f"dossier_read is not importable ({type(exc).__name__}), so no "
            "geometric role could be read for any layer. Every layer below is "
            "name-only; that is a gap in this report, not a fact about the "
            "drawing."
        )

    try:
        document = dossier_read.dossier_for(drawing_id)
    except Exception as exc:  # noqa: BLE001
        return {}, (
            f"the Dossier store could not be read ({type(exc).__name__}). "
            "Nothing was established about any layer's geometry either way."
        )

    if document is None:
        return {}, (
            "no Dossier has been built for this drawing, so no layer has a "
            "geometric role yet. Build it with `python "
            f"scripts/dossier_backfill.py --drawing {drawing_id}` and draft "
            "again — every layer below is name-only until then. This is NOT a "
            "statement that the drawing is empty."
        )

    chunk = max(1, int(getattr(dossier_read, "MAX_PROFILE_LAYERS", 40)))
    profiles: dict[str, Any] = {}
    names = [str(n) for n in layer_names]
    for start in range(0, len(names), chunk):
        window = names[start : start + chunk]
        try:
            profiles.update(
                dossier_read.profiles_for(
                    drawing_id, window, layout=layout, dossier=document
                )
            )
        except Exception as exc:  # noqa: BLE001
            return profiles, (
                f"the Dossier was read but profiling stopped after "
                f"{len(profiles)} of {len(names)} layers "
                f"({type(exc).__name__}). The layers after that point are "
                "name-only here because they were not examined."
            )
    return profiles, None


def draft_for_drawing(
    drawing: Mapping[str, Any],
    *,
    layout: str | None = None,
    pattern_set: landuse.PatternSet | None = None,
) -> dict[str, Any]:
    """`build_draft` with the Dossier and the drawing's config read for you.

    The store-backed convenience `scripts/onboard_drawing.py` calls. Every
    failure below degrades into a stated gap rather than an exception: a
    readiness report that crashes on a damaged drawing is a report nobody can
    use on the drawings that need it most (G8).
    """
    drawing_id = _text(drawing.get("_id")) or _text(drawing.get("drawing_id")) or ""
    names = [row["name"] for row in declared_layers(drawing)]
    profiles, note = profiles_from_store(drawing_id, names, layout=layout)

    config = None
    config_note = None
    try:
        config = landuse.for_drawing(drawing_id)
    except landuse.ConfigError as exc:
        config_note = (
            f"this drawing HAS a config file and it cannot be read: "
            f"{exc.message} Nothing here proposes a replacement for it."
        )

    draft = build_draft(
        drawing,
        profiles=profiles,
        pattern_set=pattern_set,
        config=config,
        layout=layout,
        profiles_note=note,
    )
    if config_note:
        draft["config_note"] = config_note
    return draft


# --- Rendering ----------------------------------------------------------------


def _yaml_lines(name: str, entry: Mapping[str, Any]) -> list[str]:
    """One layer entry as YAML lines, indented two spaces under `layers:`.

    Rendered by `yaml.safe_dump` rather than by string formatting, so a layer
    name carrying a quote, a colon, a backslash or Arabic text still produces a
    file that parses. Layer names are not ours to sanitise.
    """
    text = yaml.safe_dump(
        {name: dict(entry)},
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=88,
    )
    return ["  " + line if line else "" for line in text.rstrip("\n").split("\n")]


def _marked(lines: Sequence[str], marker: str) -> list[str]:
    return [(marker + line) if line else marker.rstrip() for line in lines]


def _header(draft: Mapping[str, Any]) -> list[str]:
    c = draft["counts"]
    out = [
        f"# DRAFT land-use config — {draft['drawing_name']}",
        "#",
        "# Written by cad_api/app/landuse_draft.py (DOSSIER Phase 3) on "
        f"{draft['drafted_at']}.",
        f"# Drawing id {draft['drawing_id']} · {draft['layers_in_table']} layers "
        "in this drawing's own layer table.",
        "#",
        "# THIS FILE CLASSIFIES NOTHING YET, AND THAT IS THE POINT.",
        "#",
        "# `layers:` below is empty. Every proposal sits under it, commented "
        "out. While",
        "# a proposal stays commented, a layer whose NAME matched a global "
        "pattern keeps",
        "# being served by the pattern layer — config_layer: global_pattern, "
        "verified:",
        "# false — which is what a proposal deserves (G4). Writing it into "
        "this file",
        "# instead would make it a drawing_override, and `verified` is DERIVED "
        "from that:",
        "# `evidence.Provenance.verified` is true for drawing_override. So "
        "uncommenting a",
        "# line is not a formatting step, it is a human signing for it. That "
        "is the only",
        "# way anything here becomes verified, and it is meant to be.",
        "#",
        "# HOW TO REVIEW IT",
        "#",
        f"#   1. read each proposal below. A line starting with "
        f"{ACCEPT_MARKER!r} is a COMPLETE",
        "#      entry: delete those three characters and the loader reads it "
        "as config.",
        f"#   2. a line starting with {DECIDE_MARKER!r} is an INCOMPLETE "
        "template. It is missing",
        "#      a decision nobody has made — nearly always the role, because",
        f"#      {_DEFAULT_ROLE_TRAP}. Fill it in by hand, or delete the "
        "layer. Do not",
        "#      accept it mechanically; that is why its marker is different.",
        "#   3. correct anything that is wrong. A wrong classification is "
        "worse than none.",
        f"#   4. rename this file to {draft['drawing_id']}.yaml and move it "
        f"into {landuse.CONFIG_DIR.name}/ .",
        "#      Until that move, nothing in here affects a single answer.",
        "#",
        "# WHAT PROPOSED WHAT",
        "#",
        "#   use   ← the global patterns in "
        f"{landuse.PATTERNS_FILE} read the layer NAME "
        f"({draft['patterns_available']} patterns, version "
        f"{draft['pattern_version']}).",
        "#           A name is a claim by whoever drafted the file. A layer "
        "that matches no",
        "#           pattern gets NO use here — not a likely one, not a "
        "common one, none.",
        "#   role  ← the Drawing Dossier read this drawing's GEOMETRY. Role is "
        "decided from",
        "#           DXF type and ring status alone and never from the name "
        "(G1), which is",
        "#           what tells a real feature layer from a sheet-layout layer "
        "with the",
        "#           same word in its name.",
        "#",
        "#   region → parcel      closed rings. Says the rings are CLOSED, not "
        "that they are",
        "#                        plots: boundaries and hatch outlines are "
        "closed rings too.",
        "#   network → network    open paths; its native measure is LENGTH, "
        "not a parcel count.",
        "#   annotation → annotation   TEXT/MTEXT/DIMENSION; never counted as "
        "a plot.",
        "#   points, mixed → no role proposed. A block insert may be a tree, a "
        "light or a",
        "#                        plot marker; `mixed` means the geometry "
        "itself is undecided.",
        "#",
        "# ONE CONSEQUENCE OF ACCEPTING A `use: unknown` ENTRY",
        "#",
        "# Those entries record what a layer IS — its role and its native "
        "measure — and",
        "# leave what it is FOR open. That is honest and it costs something: a "
        "layer named",
        "# in a config is no longer `unclassified`, so the residual that "
        "land_use_summary",
        "# attaches to a use reporting zero stops surfacing it. Accept it when "
        "the role is",
        "# worth recording — a `network` layer then answers in LENGTH instead "
        "of `0 parcels`",
        "# — and leave it commented if you would rather it kept showing up as "
        "unmapped.",
        "#",
        "# WHAT IS IN HERE",
        "#",
        f"#   {'layers in the drawing layer table':<36}: {c['layers']:5}",
        f"#   {'already written in a config by hand':<36}: "
        f"{c['already_in_config']:5}   (left alone below)",
        f"#   {'use proposed by a global pattern':<36}: {c['use_proposed']:5}"
        "   (verified: false — a proposal)",
        f"#   {'geometry only, no use proposed':<36}: {c['geometry_only']:5}",
        f"#   {'nothing but the layer name':<36}: {c['name_only']:5}",
        f"#   {f'ready to accept ({ACCEPT_MARKER.strip()} lines)':<36}: "
        f"{c['ready_to_accept']:5}",
        f"#   {f'need a decision first ({DECIDE_MARKER.strip()} lines)':<36}: "
        f"{c['needs_a_decision']:5}",
        "#",
        "# Nothing was dropped for size: a layer left out of this file is a "
        "layer nobody",
        "# will ever classify.",
    ]
    if draft.get("profiles_note"):
        out += ["#", "# ONE GAP, STATED:"] + [
            "#   " + line for line in _wrap(str(draft["profiles_note"]), 70)
        ]
    if draft.get("config_note"):
        out += ["#", "# CONFIG WARNING:"] + [
            "#   " + line for line in _wrap(str(draft["config_note"]), 70)
        ]
    return out


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def render_yaml(draft: Mapping[str, Any]) -> str:
    """The review file: a loadable scaffold plus every proposal, commented.

    Loadable as written — `layers:` carries nothing, so a human who moves this
    file without reading it changes not one classification. That is deliberate:
    the failure mode of a draft nobody reviewed must be "no effect", never "250
    guesses in production".
    """
    entries = list(draft["entries"])
    ready = [e for e in entries if e.get("ready")]
    decide = [
        e
        for e in entries
        if not e.get("ready") and e.get("status") != STATUS_ALREADY
    ]
    already = [e for e in entries if e.get("status") == STATUS_ALREADY]

    lines = _header(draft)
    lines += [
        "",
        "version: 1",
        f"drawing_id: {_scalar(draft['drawing_id'])}",
        f"drawing_name: {_scalar(draft['drawing_name'])}",
        "",
        "# `unknown` means no decision. It is not a land use and counts nothing.",
        "default_use: unknown",
        "",
        "# EMPTY ON PURPOSE — see the top of this file. Uncomment the proposals",
        "# below to fill it in. `layers:` must stay the LAST live key here, so"
        " that",
        "# an uncommented entry lands inside it.",
        "layers:",
    ]

    lines += ["", "# " + "=" * 70]
    lines += [
        f"# READY TO ACCEPT — {len(ready)} layer(s). Delete "
        f"{ACCEPT_MARKER.strip()!r} + one space to accept.",
        "# " + "=" * 70,
    ]
    if not ready:
        lines += [
            "#",
            "# None. Nothing here could be proposed completely from the "
            "drawing alone.",
        ]
    for entry in ready:
        lines += [""] + _comment_for(entry)
        lines += _marked(_yaml_lines(entry["layer"], entry["entry"]), ACCEPT_MARKER)

    lines += ["", "# " + "=" * 70]
    lines += [
        f"# NEEDS A HUMAN — {len(decide)} layer(s). Templates, not proposals: "
        "each one is",
        "# missing a decision nobody has made. Fill it in by hand; do NOT "
        "accept it as is.",
        "# " + "=" * 70,
    ]
    if not decide:
        lines += ["#", "# None."]
    for entry in decide:
        lines += [""] + _comment_for(entry)
        lines += _marked(_yaml_lines(entry["layer"], entry["entry"]), DECIDE_MARKER)

    if already:
        lines += ["", "# " + "=" * 70]
        lines += [
            f"# ALREADY DECIDED BY A HUMAN — {len(already)} layer(s) in "
            f"{draft.get('config_path')}.",
            "# Not re-proposed here: a draft that argues with its own reviewer "
            "wastes both.",
            "# " + "=" * 70,
        ]
        for entry in already:
            lines.append(f"#   {entry['layer']}")

    return "\n".join(lines) + "\n"


def _scalar(value: str) -> str:
    """One scalar, quoted the way YAML needs it.

    Dumped as part of a mapping and then split, rather than dumped on its own:
    PyYAML ends a bare scalar document with a `...` marker, and pasting that
    into a file produces something that no longer parses.
    """
    dumped = yaml.safe_dump(
        {"k": str(value)}, allow_unicode=True, default_flow_style=False, width=10_000
    )
    return dumped.split(":", 1)[1].strip()


def _comment_for(entry: Mapping[str, Any]) -> list[str]:
    """The prose above one proposal: what is measured, and what is missing."""
    head = f"# --- {entry['layer']} "
    head = head + "-" * max(3, 74 - len(head))
    out = [head]

    if entry.get("profiled"):
        out.append(
            f"#     geometry : role {entry.get('geometric_role')!r} in layout "
            f"{entry.get('layout')!r} — {entry.get('profile_phrase')}"
        )
        if entry.get("buckets") and int(entry["buckets"]) > 1:
            out.append(
                f"#                {entry['buckets']} (layer x layout) buckets, "
                f"{entry.get('entities')} entities over all of them; the "
                "measure above is"
            )
            out.append(
                "#                the largest bucket only, because measures "
                "are never summed across layouts (G2)"
            )
        if entry.get("anomaly_kinds"):
            out.append(
                "#     anomaly  : " + ", ".join(entry["anomaly_kinds"])
            )
    elif entry.get("declared_but_empty"):
        out.append(
            "#     geometry : none — the layer table declares this layer and "
            "it holds 0 entities"
        )
    else:
        out.append(
            "#     geometry : NOT PROFILED — "
            + (entry.get("profile_gap") or "this layer has no Dossier bucket")
        )

    if entry.get("use"):
        out.append(
            f"#     name     : matches `{entry.get('matched_pattern')}` → use "
            f"{entry['use']!r}"
            + (f", subtype {entry['subtype']!r}" if entry.get("subtype") else "")
        )
    else:
        out.append(
            "#     name     : matches no global pattern — NO USE IS PROPOSED"
        )

    if entry.get("tension"):
        out += ["#     DISAGREEMENT:"] + [
            "#       " + line for line in _wrap(str(entry["tension"]), 66)
        ]
    for need in entry.get("blocking") or []:
        out += ["#     DECIDE BEFORE WRITING THIS DOWN:"] + [
            "#       " + line for line in _wrap(need, 66)
        ]
    for question in entry.get("open_questions") or []:
        out += ["#     STILL OPEN AFTER ACCEPTING:"] + [
            "#       " + line for line in _wrap(question, 66)
        ]
    return out


def accept(text: str) -> str:
    """The file as it would read if a human accepted every ready proposal.

    Mechanical: `ACCEPT_MARKER` is removed, and nothing else changes.
    `DECIDE_MARKER` lines survive as comments, which is the whole reason there
    are two markers — an accept-all pass must never be able to sweep up an
    entry that is missing a decision.

    Used by the tests to prove the draft loads through the real loader, and
    available to any tool that wants to show a reviewer the accepted form
    before they commit to it. It writes nothing.
    """
    out: list[str] = []
    for line in str(text).splitlines():
        if line.startswith(ACCEPT_MARKER):
            out.append(line[len(ACCEPT_MARKER) :])
        elif line.rstrip() == ACCEPT_MARKER.rstrip():
            out.append("")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


# --- Writing ------------------------------------------------------------------


def default_draft_dir() -> Path:
    """Where drafts go when the caller names no directory.

    In order: the `CAD_LANDUSE_DRAFT_DIR` environment variable; the writable
    SVG cache directory the service already owns; the working directory. Not
    one of them is `landuse/`, and `write_draft` refuses that directory anyway
    — a default that could ever resolve there would be one restart away from
    publishing a draft as config.
    """
    named = os.environ.get(DRAFT_DIR_ENV, "").strip()
    if named:
        return Path(named)
    try:
        from .config import get_settings

        return Path(get_settings().svg_dir) / DRAFT_DIR_NAME
    except Exception:  # noqa: BLE001 -- config is optional outside the service
        return Path.cwd() / DRAFT_DIR_NAME


def draft_path(drawing_id: str, *, out_dir: Path | str | None = None) -> Path:
    """The file a draft for this drawing is written to."""
    directory = Path(out_dir) if out_dir is not None else default_draft_dir()
    return directory / f"{drawing_id}{DRAFT_SUFFIX}"


def _refuse_config_dir(path: Path) -> None:
    """The guard. A draft in `landuse/` is a draft the loader believes."""
    resolved = path.resolve()
    config_dir = landuse.CONFIG_DIR.resolve()
    if resolved.parent == config_dir or config_dir in resolved.parents:
        raise DraftRefused(
            "DRAFT_IN_CONFIG_DIR",
            f"refusing to write a draft into {config_dir}: {resolved}.",
            "A file in that directory is read as config by "
            "`landuse.for_drawing`, and its entries would arrive as "
            "drawing_override — which is what `verified: true` is derived "
            "from. A draft is a proposal; a human moves it there once they "
            "have signed for it.",
        )


def write_draft(
    draft: Mapping[str, Any],
    *,
    out_dir: Path | str | None = None,
    path: Path | str | None = None,
) -> Path:
    """Write the review file and return where it went. Never into `landuse/`.

    The directory is created if it does not exist. The file is overwritten if
    it does: a draft is derived entirely from the drawing and the patterns, so
    re-running produces the same document and there is nothing to lose — and
    `drawing_id` is a content hash, so a new revision writes a new file rather
    than trampling the old one.
    """
    target = (
        Path(path)
        if path is not None
        else draft_path(str(draft["drawing_id"]), out_dir=out_dir)
    )
    _refuse_config_dir(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_yaml(draft), encoding="utf-8")
    return target


# --- The readiness report -----------------------------------------------------


def report_lines(
    draft: Mapping[str, Any],
    *,
    path: Path | str | None = None,
    ok: str = "OK ",
    warn: str = "!! ",
    gap: str = "-- ",
) -> list[str]:
    """The block `scripts/onboard_drawing.py` prints (G9).

    Counts only, each one labelled by what it is a count OF. "34 layers
    unclassified" is the sentence this phase exists to improve on: a reviewer
    needs to know how many of them something could be proposed for, how many
    carry geometry and no meaning, and how many are nothing but a name.

    ASCII only, deliberately. This goes to a terminal whose encoding is not
    ours to choose; the drafted FILE is written as UTF-8 and keeps its
    typography. A report that raises `UnicodeEncodeError` on someone's console
    is a report that did not get read.
    """
    c = draft["counts"]
    lead = ok if c["ready_to_accept"] else gap
    out = [
        f"{lead}Land-use draft  : "
        + (str(path) if path is not None else "not written")
    ]
    out += [
        f"     use proposed by a pattern    : {c['use_proposed']:5} layers"
        "   (verified: false - a proposal, not a fact)",
        f"     geometry only, no use        : {c['geometry_only']:5} layers"
        "   (role measured, meaning not established)",
        f"     nothing but the layer name   : {c['name_only']:5} layers",
        f"     ready to accept ('{ACCEPT_MARKER.strip()}' lines) : "
        f"{c['ready_to_accept']:5}",
        f"     need a decision ('{DECIDE_MARKER.strip()}' lines) : "
        f"{c['needs_a_decision']:5}",
    ]
    if c["already_in_config"]:
        out.append(
            f"     already decided by a human   : "
            f"{c['already_in_config']:5} layers   (left alone)"
        )
    if draft.get("profiles_note"):
        out.append(f"{warn}     no geometry to draft from:")
        out += ["       " + line for line in _wrap(str(draft["profiles_note"]), 64)]
    if draft.get("config_note"):
        out.append(f"{warn}     " + str(draft["config_note"]))
    out.append(
        "     Nothing is classified by this file. It is NOT in "
        f"{landuse.CONFIG_DIR.name}/, and no"
    )
    out.append(
        "     answer changes until a human uncomments what they agree with "
        "and moves it there."
    )
    return out
