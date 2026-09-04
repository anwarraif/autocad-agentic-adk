"""Layer -> land use map: loader, validator, and lookup.

Why meaning lives in YAML and not in `.py`
------------------------------------------
The question "how many schools are there" is answered with zero results by a
text search, and that answer is correct: of the 43,109 TEXT/MTEXT in the
reference drawing, not one contains the word "school", "villa", or any typology
code. The information is in the LAYER NAME, and until now no tool has read a
layer name as meaning.

Three reasons for config rather than a guess inside the code:

1.  **Fixable without touching code.** Once the typology key from Roshn
    arrives, what changes is one line of YAML.
2.  **Reviewable.** A planner can read and correct this file; they cannot
    review a prompt.
3.  **Able to carry an evidence grade.** Each entry states how it knows, and
    `evidence.py` computes the grade from that.

Two layers, and their order is the contract (G4)
------------------------------------------------
``<drawing_id>.yaml`` (per-drawing override) beats ``_patterns.yaml``
(global patterns); whatever matches neither is ``unclassified`` and is
returned as ``None``, not as a guess. There is no fuzzy matching in this file
and there must not be: a guess that sounds like a fact is the most damaging
kind of failure, because it does not look wrong.

``verified`` is not in the YAML and cannot be typed
----------------------------------------------------
``{use: open_space, verified: true, source: "layer name"}`` — the old form —
is one source, one corpus, marked verified by hand. A ``verified`` boolean plus
``source`` prose IS a mechanism for raising the evidence grade, in YAML form.
Both were removed (D-082): ``verified`` is derived from the config layer in
``evidence.Provenance`` — ``drawing_override`` true, ``global_pattern`` always
false — and ``grade`` is computed from ``sources``. The loader below REFUSES
both keys by name, so that a half-finished migration explodes instead of going
quiet.

When it fails hard, and when it does not
-----------------------------------------
STRUCTURAL errors — a ``use`` outside the vocabulary, a ``role`` that is not in
``ROLES``, an ``origin`` that is not a member of ``evidence.Origin``, an
entry without ``sources``, the legacy ``verified``/``source`` keys — raise
``ConfigError`` when the file is loaded, and the file is loaded when that
drawing is asked about. So a broken config takes down ONE drawing, not the
service: D-067 records a hard failure at startup that left cad-api never
healthy, and `main.py` is not a file that may be touched from here to install a
startup hook.

A mismatch against the DRAWING — a layer the config names that is not in that
drawing's layer table — does NOT raise. It is reported inside the response as
`config_problems`, with the layer names. A drawing that was revised and lost a
layer must not make its whole land use summary disappear; what its reader needs
is the name of the layer that drifted, and that is what they get.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .. import evidence as ev
from . import crs as crs_math

log = logging.getLogger(__name__)

#: The config directory. It travels into the image via `COPY app ./app` in the
#: Dockerfile, so there is no separate deploy step for a new config.
CONFIG_DIR = Path(__file__).resolve().parent

#: File name of the global patterns. It is prefixed with an underscore so it
#: can never collide with `<drawing_id>.yaml` — a drawing_id is 16 hex digits.
PATTERNS_FILE = "_patterns.yaml"

#: Shared standards, one file per family of drawings.
#:
#: This is the layer that was missing, and the question that made it visible:
#: "what about the other Roshn drawings?" Global patterns catch anything named
#: with a word -- school, mosque, park -- and that holds in anyone's drawing.
#: Typology codes do not: `VL2` means nothing in any language, and until now it
#: had to be tagged again for EVERY drawing even though every Roshn drawing
#: uses the same codes. Measured on the reference drawing: without the override
#: file, 301 parcels get a land use and 2,584 do not.
#:
#: A standard is used only if the drawing's config names it. Without
#: `standard:` there, this file is never read -- meaning does not travel
#: between drawings by itself (G10).
STANDARDS_DIR = CONFIG_DIR / "_standards"

#: The `use` vocabulary, READ FROM `_patterns.yaml` and not typed here.
#:
#: It used to be a `frozenset` literal in this file, and that meant adding one
#: kind of building -- a warehouse, a hotel, a cemetery -- was a Python change
#: in a module already used by eighteen drawings, rather than one line in a
#: config file. A list of land uses that lives in code is a list that stops
#: growing, and the next drawing comes from a contractor who names things that
#: are not on that list.
#:
#: The source is now the `vocabulary` block in `_patterns.yaml`: a use is valid
#: if -- and only if -- there is a vocabulary entry for it. That is not a
#: loosening but a tighter binding than before, because a use without a
#: vocabulary can never be `stated` by any text; it would be quietly locked at
#: `inferred` forever. Requiring both to exist means that state cannot be
#: created without anyone noticing.
#:
#: The three values below are not land uses and have no word to search for;
#: they are machinery, so they stay in the code.
RESERVED_USES: frozenset[str] = frozenset({"structure", "annotation", "unknown"})


def _uses_from_patterns() -> frozenset[str]:
    """The valid `use` values, from the vocabulary in `_patterns.yaml`.

    Never raises. A broken patterns file is already reported clearly by
    `patterns()`; making this module fail on import would turn one mistyped
    config into a service that refuses to start.
    """
    try:
        import yaml

        raw = yaml.safe_load((CONFIG_DIR / PATTERNS_FILE).read_text(encoding="utf-8"))
        vocab = (raw or {}).get("vocabulary") or {}
        found = {str(k).strip() for k in vocab if str(k).strip()}
    except Exception:  # noqa: BLE001 -- see the docstring
        found = set()
    return frozenset(found | RESERVED_USES)


USES: frozenset[str] = _uses_from_patterns()

#: The role of a layer inside the drawing. Only `parcel` may be counted as a
#: plot -- this is what stops 8 overlay HATCHes and 416 TEXT labels from being
#: counted as plot types. One of them once made the unit mix wrong by 416.
#:
#: `network` was added by DOSSIER Phase 2 and is the only one of the five that
#: is not about EXCLUDING something. A road centreline is not a plot and never
#: will be, so under the original four the only honest thing a config could say
#: about it was `overlay` -- and an overlay is reported as "adds nothing to the
#: plot count", which is true and useless. The measured fact about that layer is
#: its LENGTH: 1,233 entities and 83,962.565 m sat unread in this drawing while
#: the tool answered "0 road parcels". A role whose native measure is length is
#: what lets a config say what such a layer IS instead of only what it is not.
#:
#: This is a role, not a land use. It says what shape the geometry has; the
#: `use` beside it still says what the layer is for, and neither is derived from
#: the other (G1). Nothing here maps a use to a role.
ROLES: frozenset[str] = frozenset(
    {"parcel", "overlay", "annotation", "structure", "network"}
)

#: The default role of an entry that does not name one.
#:
#: Deliberately unchanged by the arrival of `network`: the default is applied to
#: every entry that does not name a role, so moving it would silently
#: re-classify every existing config entry in every drawing.
DEFAULT_ROLE = "parcel"

#: The only role that is counted as a parcel.
PARCEL_ROLE = "parcel"

#: The role whose native measure is LENGTH, not a parcel count. A layer with
#: this role reports its Σ length from the Dossier; reporting `0 parcels` for it
#: would be arithmetically true and completely misleading.
NETWORK_ROLE = "network"

#: The `use` that means "no decision", not a land use.
UNKNOWN_USE = "unknown"

#: The keys REMOVED by D-082. Refused by name, because a half-migrated config
#: accepted in silence would publish a `verified: true` with nothing behind it.
RETIRED_KEYS: tuple[str, ...] = ("verified", "source")


class ConfigError(ValueError):
    """A land use config that is structurally wrong.

    Its shape mirrors `store.MeasureRefused` (`code`/`message`/`hint`) so that
    the caller can publish it as an actionable 400 -- complete with the name of
    the offending layer -- rather than a 500 that can only be stared at.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


# --- Source ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Source:
    """One `sources:` line from the YAML, already validated.

    It is deliberately NOT an `evidence.Observation` itself: an `Observation`
    needs to know the layout being answered about, and a config entry applies
    to every layout. `observation()` marries the two, per call.
    """

    origin: ev.Origin
    detail: str
    observed: str | None = None
    observed_raw: str | None = None
    locator: str | None = None
    layout: str | None = None
    reading: ev.Reading = ev.VERBATIM
    reference: str | None = None

    def observation(self, *, scope_layout: str | None) -> ev.Observation:
        return ev.Observation(
            origin=self.origin,
            detail=self.detail,
            observed=self.observed,
            observed_raw=self.observed_raw,
            locator=self.locator,
            layout=self.layout if self.layout is not None else scope_layout,
            reading=self.reading,
            reference=self.reference,
        )


@dataclass(frozen=True, slots=True)
class LandUse:
    """The land use decision for ONE layer in ONE drawing."""

    layer: str
    use: str
    subtype: str | None
    role: str
    config_layer: ev.ConfigLayer
    config_version: int | None
    sources: tuple[Source, ...] = ()
    note: str = ""
    not_established: str | None = None
    how_to_verify: str | None = None
    matched_pattern: str | None = None

    @property
    def is_parcel(self) -> bool:
        return self.role == PARCEL_ROLE

    @property
    def is_network(self) -> bool:
        """Whether this layer's native measure is length rather than a count.

        Read by `store_landuse` to decide which non-parcel rows are asked for a
        Dossier length. It is deliberately a property of the ROLE and of nothing
        else -- no land use implies a network here, and no network implies a
        land use (G1).
        """
        return self.role == NETWORK_ROLE

    def provenance(self) -> ev.Provenance:
        return ev.Provenance(
            config_layer=self.config_layer,
            config_version=self.config_version,
            note=self.note,
        )

    def as_row(self) -> dict[str, Any]:
        """The compact form for a response row. Does NOT carry `verified`.

        `verified` is only born from `Provenance`, and copying it here would
        give two homes to a value that must have exactly one.
        """
        return {
            "layer": self.layer,
            "land_use": self.use,
            "land_use_subtype": self.subtype,
            "role": self.role,
            "config_layer": self.config_layer.value,
            "matched_pattern": self.matched_pattern,
            "note": self.note or None,
            "not_established": self.not_established,
            "how_to_verify": self.how_to_verify,
        }


@dataclass(frozen=True, slots=True)
class PatternRule:
    """One `patterns:` line, with its regex already compiled."""

    match: str
    regex: re.Pattern[str]
    use: str
    subtype: str | None
    role: str
    note: str


@dataclass(frozen=True, slots=True)
class PatternSet:
    version: int | None
    rules: tuple[PatternRule, ...]
    vocabulary: Mapping[str, tuple[str, ...]]
    vocabulary_problems: tuple[str, ...]

    def first_match(self, layer_name: str) -> PatternRule | None:
        for rule in self.rules:
            if rule.regex.search(layer_name or ""):
                return rule
        return None

    def tokens_for(self, use: str) -> tuple[str, ...]:
        return tuple(self.vocabulary.get(use, ()))


#: The H3 resolution used when a drawing's config does not name one. Cells at
#: this level average 43.87 m2 — fine enough that the smallest parcel on the
#: reference drawing gets several, and coarse enough that a whole
#: neighbourhood is thousands of cells rather than millions. It is a default
#: for ANY drawing, not a fact about one, which is what keeps it out of the
#: drawing-specific-constant gate (G1).
DEFAULT_H3_RESOLUTION = 13

#: H3 defines resolutions 0 through 15. Anything else is not a coarser or
#: finer answer; it is not an answer.
_H3_RESOLUTIONS = range(0, 16)


@dataclass(frozen=True, slots=True)
class DrawingConfig:
    """One drawing's override."""

    drawing_id: str
    drawing_name: str
    version: int | None
    default_use: str
    layers: Mapping[str, LandUse]
    #: The name of the shared standard this drawing declares it follows, or
    #: `None`.
    standard: str | None
    empty_but_declared: tuple[str, ...]
    path: str
    #: This drawing's coordinate system, or `None`. `None` means no lat/long
    #: may be published — there is no default and no guessed zone (G3, and
    #: UPLIFT-14 "what is not done").
    crs: crs_math.Crs | None = None
    #: The H3 resolution this drawing's cells are indexed at. Config rather
    #: than a literal in the assignment code, because the right resolution is
    #: a property of what a drawing CONTAINS: 43 m2 cells resolve the 200 m2
    #: plots on the reference drawing and would be absurd on a site plan whose
    #: smallest object is a hectare. Defaulted rather than required — a
    #: drawing that says nothing gets a resolution, not an error.
    h3_resolution: int = DEFAULT_H3_RESOLUTION
    # Mismatches against the drawing's layer table are NOT stored here: the
    # loader must not need a database in order to read a file.
    # `problems_against_drawing()` computes them, per call.


# --- Loading -----------------------------------------------------------------

#: path -> (mtime_ns, parse result). Loaded once, reloaded if the file changes.
#: Deliberately without a TTL: a config edited during debugging must be read on
#: the next request, and a config that was not edited must not be re-read 46
#: thousand times.
_CACHE: dict[str, tuple[int, Any]] = {}


def _read_yaml(path: Path) -> Mapping[str, Any] | None:
    """Read YAML with an mtime-based cache. `None` if the file is not there."""
    try:
        stat = path.stat()
    except OSError:
        _CACHE.pop(str(path), None)
        return None

    cached = _CACHE.get(str(path))
    if cached is not None and cached[0] == stat.st_mtime_ns:
        return cached[1]

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(
            "LAND_USE_CONFIG_UNREADABLE",
            f"{path.name} is not valid YAML: {exc}.",
            "Fix the file. A config that cannot be read had better stop one "
            "drawing than classify half of it.",
        ) from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(
            "LAND_USE_CONFIG_UNREADABLE",
            f"{path.name} does not contain a mapping at the top level.",
            "A config file starts with the keys `version:` and `layers:`.",
        )
    _CACHE[str(path)] = (stat.st_mtime_ns, raw)
    return raw


def clear_cache() -> None:
    """Drop the cache. For tests; not used on the request path."""
    _CACHE.clear()


# --- Validation --------------------------------------------------------------


def _require_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(
            "LAND_USE_ENTRY_MALFORMED",
            f"{where} is not a mapping, but a {type(value).__name__}.",
            "Every layer entry has the form `\"Layer Name\": {use: ..., sources: [...]}`.",
        )
    return value


def _check_retired_keys(entry: Mapping[str, Any], *, where: str) -> None:
    present = [k for k in RETIRED_KEYS if k in entry]
    if not present:
        return
    raise ConfigError(
        "LAND_USE_RETIRED_KEY",
        f"{where} uses keys that have been removed: {', '.join(present)}.",
        "`verified:` and `source:` were removed by D-082. `verified` is now "
        "derived from the config layer and cannot be typed; `source:` prose is "
        "demoted to `note:` and the real source is written in `sources:` with "
        "an `origin` and an `observed`.",
    )


def _build_reading(raw: Any, *, where: str) -> ev.Reading:
    if raw is None:
        return ev.VERBATIM
    raw = _require_mapping(raw, where=f"{where} the `reading` block")
    missing = [k for k in ("verbatim", "method", "corpus_verified") if k not in raw]
    if missing:
        raise ConfigError(
            "LAND_USE_READING_INCOMPLETE",
            f"{where} the `reading` block is missing: {', '.join(missing)}.",
            "A reading states whether it is bytes as they are (`verbatim`), "
            "how it was read (`method`), and whether its raw string is in the "
            "verified corpus (`corpus_verified`). Without all three, its "
            "evidence ceiling cannot be computed.",
        )
    return ev.Reading(
        verbatim=bool(raw["verbatim"]),
        method=str(raw["method"]),
        corpus_verified=bool(raw["corpus_verified"]),
    )


def _build_source(raw: Any, *, where: str) -> Source:
    raw = _require_mapping(raw, where=f"{where} a `sources` entry")
    origin_name = str(raw.get("origin") or "").strip()
    try:
        origin = ev.Origin(origin_name)
    except ValueError as exc:
        raise ConfigError(
            "LAND_USE_UNKNOWN_ORIGIN",
            f"{where} uses origin {origin_name!r}, which is not a member of "
            "evidence.Origin.",
            "Valid values: " + ", ".join(sorted(o.value for o in ev.Origin)) + ". "
            "An origin without a corpus decision would count as independent "
            "and raise the evidence grade with nothing behind it.",
        ) from exc

    source = Source(
        origin=origin,
        detail=str(raw.get("detail") or ""),
        observed=_opt_str(raw.get("observed")),
        observed_raw=_opt_str(raw.get("observed_raw")),
        locator=_opt_str(raw.get("locator")),
        layout=_opt_str(raw.get("layout")),
        reading=_build_reading(raw.get("reading"), where=where),
        reference=_opt_str(raw.get("reference")),
    )
    # Not a second validation: `Observation` holds the rule "a corpus that can
    # speak must have an `observed`" and the rule "a source outside the file
    # must have a `reference`". Copying those rules here would create two
    # copies that can drift apart. So it is built once, here, so that a
    # defective entry explodes when the config is loaded -- not when someone
    # asks a question.
    try:
        source.observation(scope_layout=None)
    except ev.EvidenceError as exc:
        raise ConfigError(
            "LAND_USE_SOURCE_INVALID",
            f"{where}: {exc.message}",
            exc.hint,
        ) from exc
    return source


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_land_use(
    layer: str,
    entry: Any,
    *,
    config_layer: ev.ConfigLayer,
    config_version: int | None,
    where: str,
    matched_pattern: str | None = None,
    sources: Sequence[Source] | None = None,
) -> LandUse:
    entry = _require_mapping(entry, where=where)
    _check_retired_keys(entry, where=where)

    use = str(entry.get("use") or "").strip()
    if use not in USES:
        raise ConfigError(
            "LAND_USE_UNKNOWN_USE",
            f"{where} uses use {use!r}, which is not in the vocabulary.",
            "Valid values: " + ", ".join(sorted(USES)) + ".",
        )

    role = str(entry.get("role") or DEFAULT_ROLE).strip()
    if role not in ROLES:
        raise ConfigError(
            "LAND_USE_UNKNOWN_ROLE",
            f"{where} uses role {role!r}.",
            "Valid values: " + ", ".join(sorted(ROLES)) + ". Only `parcel` "
            "is counted as a plot; `overlay` and `annotation` exist exactly "
            "so that hatches and labels are never counted in; `network` says "
            "the layer's native measure is LENGTH, so it is reported in "
            "length instead of as a parcel count of zero.",
        )

    if sources is None:
        raw_sources = entry.get("sources")
        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
            raise ConfigError(
                "LAND_USE_WITHOUT_SOURCES",
                f"{where} has no `sources` list.",
                "Every entry states how it knows. Without that its evidence "
                "grade is `unknown`, and a classification with no basis is "
                "more dangerous than no classification at all.",
            )
        built = tuple(
            _build_source(s, where=f"{where} source {i + 1}")
            for i, s in enumerate(raw_sources)
        )
        if not built:
            raise ConfigError(
                "LAND_USE_WITHOUT_SOURCES",
                f"{where} has an empty `sources`.",
                "Name at least one concrete thing seen in the file.",
            )
    else:
        built = tuple(sources)

    not_established = _opt_str(entry.get("unknown"))
    how_to_verify = _opt_str(entry.get("how_to_verify"))
    if not_established and not how_to_verify:
        raise ConfigError(
            "LAND_USE_UNKNOWN_WITHOUT_ROUTE",
            f"{where} names `unknown:` without `how_to_verify:`.",
            "A gap that does not say where to ask will be skipped by its "
            "reader. Name the document or the person.",
        )

    return LandUse(
        layer=layer,
        use=use,
        subtype=_opt_str(entry.get("subtype")),
        role=role,
        config_layer=config_layer,
        config_version=config_version,
        sources=built,
        note=str(entry.get("note") or "").strip(),
        not_established=not_established,
        how_to_verify=how_to_verify,
        matched_pattern=matched_pattern,
    )


# --- Global patterns ---------------------------------------------------------


def patterns() -> PatternSet:
    """The patterns that apply to any drawing, plus the literal-test vocabulary."""
    raw = _read_yaml(CONFIG_DIR / PATTERNS_FILE)
    if raw is None:
        return PatternSet(version=None, rules=(), vocabulary={}, vocabulary_problems=())

    rules: list[PatternRule] = []
    for i, item in enumerate(raw.get("patterns") or []):
        where = f"{PATTERNS_FILE} pattern {i + 1}"
        item = _require_mapping(item, where=where)
        _check_retired_keys(item, where=where)
        expr = str(item.get("match") or "")
        try:
            regex = re.compile(expr)
        except re.error as exc:
            raise ConfigError(
                "LAND_USE_PATTERN_INVALID",
                f"{where} is not a valid regex: {exc}.",
                "Matching is `re.search`, so a trailing `.*` is unnecessary; "
                "what is needed instead is `\\b` so that `CAR PARKING` does "
                "not become a park.",
            ) from exc
        use = str(item.get("use") or "").strip()
        if use not in USES:
            raise ConfigError(
                "LAND_USE_UNKNOWN_USE",
                f"{where} uses use {use!r}, which is not in the vocabulary.",
                "Valid values: " + ", ".join(sorted(USES)) + ".",
            )
        role = str(item.get("role") or DEFAULT_ROLE).strip()
        if role not in ROLES:
            raise ConfigError(
                "LAND_USE_UNKNOWN_ROLE",
                f"{where} uses role {role!r}.",
                "Valid values: " + ", ".join(sorted(ROLES)) + ".",
            )
        rules.append(
            PatternRule(
                match=expr,
                regex=regex,
                use=use,
                subtype=_opt_str(item.get("subtype")),
                role=role,
                note=str(item.get("note") or "").strip(),
            )
        )

    vocabulary, problems = ev.load_vocabulary(raw.get("vocabulary") or {})
    if problems:
        # Reported, not raised: a rejected token makes one claim unable to be
        # `stated`, it does not take the service down (D-067).
        log.warning("the land use vocabulary rejected %d tokens", len(problems))
    return PatternSet(
        version=_opt_int(raw.get("version")),
        rules=tuple(rules),
        vocabulary=vocabulary,
        vocabulary_problems=tuple(problems),
    )


def _opt_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --- Per-drawing override ----------------------------------------------------


def standard_path(name: str) -> Path:
    """The file of a shared standard."""
    return STANDARDS_DIR / f"{name}.yaml"


@lru_cache(maxsize=8)
def standard_layers(name: str) -> Mapping[str, LandUse]:
    """The layer mapping of a shared standard.

    Its file has exactly the same shape as a per-drawing override, and that is
    deliberate: a mapping that is promoted from one drawing to a standard does
    not have to be rewritten, only moved. The only difference is its config
    layer, and through that, `verified`.

    The `sources` inside it must not name the geometry of any drawing. A plot
    module of 12 x 25 m measured in one file is a fact about that file; what
    travels to another file is only the mapping, and its origin is the human
    who decided it.
    """
    path = standard_path(name)
    raw = _read_yaml(path)
    if raw is None:
        raise ConfigError(
            "LAND_USE_STANDARD_MISSING",
            f"standard {name!r} is not in {STANDARDS_DIR.name}.",
            "Create `landuse/_standards/" + name + ".yaml`, or remove the "
            "`standard:` line from the drawing's config.",
        )
    version = _opt_int(raw.get("version"))
    out: dict[str, LandUse] = {}
    for layer_name, entry in (raw.get("layers") or {}).items():
        layer = str(layer_name)
        out[layer] = _build_land_use(
            layer,
            entry,
            config_layer=ev.ConfigLayer.SHARED_STANDARD,
            config_version=version,
            where=f"{path.name} layer {layer!r}",
        )
    return out


#: A writable directory searched BEFORE the one baked into the image.
#:
#: Until this existed, giving a drawing a coordinate system meant editing a
#: file inside the image and rebuilding it. That is not a step a user of the
#: viewer can take, so every newly uploaded drawing stopped at the same place:
#: ingested, stored, dossiered, and unmappable, with no way forward that did
#: not involve a developer. Sedra reached exactly that state on its first
#: hands-off run.
#:
#: The image tree stays the home of the reviewed, committed configs. This
#: overlay is where an operator's accepted answer lands, and it wins, because
#: a decision taken about THIS store is newer than one baked in weeks ago.
ACCEPTED_DIR = Path(
    os.environ.get("CAD_LANDUSE_ACCEPTED_DIR", "/data/svg/landuse_configs")
)


def _config_search_dirs() -> tuple[Path, ...]:
    """Every directory a per-drawing config may live in, in priority order.

    This tuple is the ONE list of config locations. `config_path` walks it to
    find a drawing's file and `configured_drawing_ids` walks it to enumerate
    them, so the two questions -- "where is this drawing's config" and "which
    drawings have one" -- cannot drift apart. They did drift once: a test
    helper globbed the image tree while the loader had learned to read the
    accepted overlay first, and eight guards fired at behaviour that was
    correct. Anyone adding a third location edits this tuple and both answers
    move together.
    """
    return (ACCEPTED_DIR, CONFIG_DIR)


def configured_drawing_ids() -> set[str]:
    """Every drawing id that ships a per-drawing config, wherever it lives.

    The authoritative answer to "which drawings are configured". Tests and
    tools must call this rather than reading a directory themselves, because
    a copy of this logic is a copy that will drift the moment a location is
    added -- which is exactly how the first red gate of this campaign
    happened.
    """
    ids: set[str] = set()
    for directory in _config_search_dirs():
        try:
            for path in directory.glob("*.yaml"):
                if not path.stem.startswith("_"):
                    ids.add(path.stem)
        except OSError:  # pragma: no cover - an unreadable mount
            continue
    return ids


def config_path(drawing_id: str) -> Path:
    """Where a drawing's override SHOULD be, whether or not it exists.

    Returned as it is so that the response for a drawing without a config can
    name the file that needs to be created, rather than merely saying there is
    none. The first search directory holding the file wins; when none holds
    it, the image tree's path is returned, which is also the right thing to
    name when nothing exists yet.
    """
    for directory in _config_search_dirs():
        candidate = directory / f"{drawing_id}.yaml"
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover - an unreadable mount
            continue
    return CONFIG_DIR / f"{drawing_id}.yaml"


def for_drawing(drawing_id: str) -> DrawingConfig | None:
    """One drawing's override, or `None` if the file is not there (G3)."""
    path = config_path(drawing_id)
    raw = _read_yaml(path)
    if raw is None:
        return None

    version = _opt_int(raw.get("version"))
    declared_id = _opt_str(raw.get("drawing_id"))
    if declared_id and declared_id != drawing_id:
        raise ConfigError(
            "LAND_USE_CONFIG_MISFILED",
            f"{path.name} names drawing_id {declared_id!r}.",
            "The file name is the key. A misfiled config would move meaning "
            "between drawings, and a layer name that means school in one "
            "drawing does not necessarily mean school in another (G10).",
        )

    default_use = str(raw.get("default_use") or UNKNOWN_USE).strip()
    if default_use not in USES:
        raise ConfigError(
            "LAND_USE_UNKNOWN_USE",
            f"{path.name} uses default_use {default_use!r}.",
            "Valid values: " + ", ".join(sorted(USES)) + ".",
        )

    standard = _opt_str(raw.get("standard"))
    if standard and not standard_path(standard).is_file():
        raise ConfigError(
            "LAND_USE_STANDARD_MISSING",
            f"{path.name} names standard {standard!r} whose file does not exist.",
            "Its file goes in `landuse/_standards/<name>.yaml`. A missing "
            "standard is allowed to fail hard rather than be ignored in "
            "silence: ignored, this drawing would lose its whole typology "
            "classification and look like a drawing that simply has none.",
        )

    layers: dict[str, LandUse] = {}
    for name, entry in (raw.get("layers") or {}).items():
        layer = str(name)
        layers[layer] = _build_land_use(
            layer,
            entry,
            config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
            config_version=version,
            where=f"{path.name} layer {layer!r}",
        )

    empty = tuple(str(n) for n in (raw.get("empty_but_declared") or []))
    return DrawingConfig(
        drawing_id=drawing_id,
        drawing_name=str(raw.get("drawing_name") or ""),
        version=version,
        default_use=default_use,
        layers=layers,
        empty_but_declared=empty,
        standard=standard,
        path=path.name,
        crs=_build_crs(raw.get("crs"), where=f"{path.name} the `crs` block"),
        h3_resolution=_build_h3_resolution(
            raw.get("h3"), where=f"{path.name} the `h3` block"
        ),
    )


def _build_h3_resolution(raw: Any, *, where: str) -> int:
    """The `h3:` block's resolution, or the default.

    An absent block is not an error — the default applies and the drawing is
    indexed. A block that is present and wrong IS an error: somebody wrote a
    resolution on purpose, and quietly substituting the default for it would
    produce cells at a level nobody asked for, which is indistinguishable from
    working correctly until the counts are compared against something.
    """
    if raw is None:
        return DEFAULT_H3_RESOLUTION
    raw = _require_mapping(raw, where=where)
    value = raw.get("resolution")
    if value is None:
        return DEFAULT_H3_RESOLUTION
    try:
        resolution = int(value)
    except (TypeError, ValueError):
        raise ConfigError(
            "H3_RESOLUTION_NOT_A_NUMBER",
            f"{where} has resolution {value!r}.",
            "It is an integer from 0 to 15.",
        ) from None
    if resolution not in _H3_RESOLUTIONS:
        raise ConfigError(
            "H3_RESOLUTION_OUT_OF_RANGE",
            f"{where} has resolution {resolution}.",
            "H3 defines resolutions 0 to 15; 13 averages 43.87 m2 per cell "
            "and is the default.",
        )
    return resolution


def _build_crs(raw: Any, *, where: str) -> crs_math.Crs | None:
    """A drawing's `crs:` block, or `None` if there is none.

    There is no default and no guessed zone. A lat/long from the wrong zone
    still looks like a correct lat/long: on the reference drawing, zone 37N
    puts it in the Red Sea and nothing in the numbers shouts.
    """
    if raw is None:
        return None
    raw = _require_mapping(raw, where=where)
    _check_retired_keys(raw, where=where)

    epsg = _opt_int(raw.get("epsg"))
    if epsg is None:
        raise ConfigError(
            "CRS_WITHOUT_EPSG",
            f"{where} without an `epsg`.",
            "Name its EPSG code. A projection name without a code cannot be "
            "checked and cannot be computed.",
        )
    try:
        crs_math.zone_of(epsg)
    except crs_math.CrsUnsupported as exc:
        raise ConfigError(exc.code, f"{where}: {exc.message}", exc.hint) from exc

    if "declared_in_file" not in raw:
        raise ConfigError(
            "CRS_WITHOUT_DECLARED_FLAG",
            f"{where} without `declared_in_file`.",
            "That field prevents the easiest mistake in this whole feature: a "
            "lat/long that looks official while the file never stated its "
            "coordinate system. It is required, whatever its value.",
        )

    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise ConfigError(
            "CRS_WITHOUT_SOURCES",
            f"{where} without a `sources` list.",
            "A CRS states how it is known, exactly like every layer entry. "
            "Without that its evidence grade is `unknown` and the lat/long "
            "born from it cannot be checked by anyone.",
        )
    sources = tuple(
        _build_source(s, where=f"{where} source {i + 1}")
        for i, s in enumerate(raw_sources)
    )
    if not sources:
        raise ConfigError(
            "CRS_WITHOUT_SOURCES",
            f"{where} has an empty `sources`.",
            "Name at least one concrete thing seen in the file.",
        )

    not_established = _opt_str(raw.get("unknown"))
    how_to_verify = _opt_str(raw.get("how_to_verify"))
    if not_established and not how_to_verify:
        raise ConfigError(
            "LAND_USE_UNKNOWN_WITHOUT_ROUTE",
            f"{where} names `unknown:` without `how_to_verify:`.",
            "Name the document or the person.",
        )

    return crs_math.Crs(
        epsg=epsg,
        name=str(raw.get("name") or f"EPSG:{epsg}"),
        declared_in_file=bool(raw.get("declared_in_file")),
        note=str(raw.get("note") or "").strip(),
        not_established=not_established,
        how_to_verify=how_to_verify,
        sources=sources,
    )


# --- Lookup ------------------------------------------------------------------


def _pattern_sources(layer_name: str, rule: PatternRule) -> tuple[Source, ...]:
    """The sources for a pattern-based classification.

    One observation, and only one: the layer name. A pattern that matches is
    NOT a second source -- it is a way of reading the same source. That is why
    a pattern hit can never be `corroborated`, and it is not forced through a
    ceiling; it falls out of this list on its own.
    """
    return (
        Source(
            origin=ev.Origin.LAYER_NAME,
            detail=f"layer name matches global pattern {rule.match!r}",
            observed=layer_name,
            locator=layer_name,
        ),
    )


def classify(
    drawing_id: str,
    layer_name: str,
    *,
    config: DrawingConfig | None = None,
    pattern_set: PatternSet | None = None,
) -> LandUse | None:
    """A layer's land use, or `None` if nothing decides it.

    The order is per-drawing override -> global pattern -> `default_use` ->
    `None`. The first match wins, and the response names which layer was used.

    `config` and `pattern_set` can be handed in by the caller so that a summary
    over 292 layers does not load the same file 292 times.
    """
    cfg = config if config is not None else for_drawing(drawing_id)
    if cfg is not None:
        hit = cfg.layers.get(layer_name)
        if hit is not None:
            return hit

    # The shared standard, and only if this drawing declares that it follows
    # it. Placed below the override so that a drawing can always refuse its
    # standard for one particular layer, and above the global patterns so that
    # a code that means nothing in any language still loses to a human
    # decision and beats nothing at all.
    if cfg is not None and cfg.standard:
        hit = standard_layers(cfg.standard).get(layer_name)
        if hit is not None:
            return hit

    pats = pattern_set if pattern_set is not None else patterns()
    rule = pats.first_match(layer_name)
    if rule is not None:
        return LandUse(
            layer=layer_name,
            use=rule.use,
            subtype=rule.subtype,
            role=rule.role,
            config_layer=ev.ConfigLayer.GLOBAL_PATTERN,
            config_version=pats.version,
            sources=_pattern_sources(layer_name, rule),
            note=rule.note,
            matched_pattern=rule.match,
        )

    if cfg is not None and cfg.default_use != UNKNOWN_USE:
        return LandUse(
            layer=layer_name,
            use=cfg.default_use,
            subtype=None,
            role=DEFAULT_ROLE,
            config_layer=ev.ConfigLayer.DEFAULT_USE,
            config_version=cfg.version,
            sources=(
                Source(
                    origin=ev.Origin.ABSENCE,
                    detail=(
                        f"no override and no pattern matched layer "
                        f"{layer_name!r}; this drawing's config sets "
                        f"default_use {cfg.default_use!r}"
                    ),
                    locator=layer_name,
                ),
            ),
            note=(
                "the result of default_use, not the result of reading this "
                "layer. A default that applies to every technical layer is a "
                "blunt instrument; check it before quoting it"
            ),
        )
    return None


def problems_against_drawing(
    config: DrawingConfig | None, declared_layers: Iterable[str]
) -> list[str]:
    """Mismatches between the config and the drawing's layer table.

    Deliberately returns a list rather than raising. A drawing that was revised
    and lost a layer must not delete its whole land use summary; what its
    reader needs is the name of the layer that drifted.
    """
    if config is None:
        return []
    known = {str(n) for n in declared_layers}
    if not known:
        return []
    problems = [
        f"layer {name!r} is in {config.path} and is not in this drawing's layer table"
        for name in sorted(config.layers)
        if name not in known
    ]
    problems += [
        f"layer {name!r} is listed as empty_but_declared and is not in this "
        "drawing's layer table"
        for name in sorted(config.empty_but_declared)
        if name not in known
    ]
    return problems
