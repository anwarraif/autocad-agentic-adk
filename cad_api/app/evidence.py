"""Evidence grades for claims that carry MEANING.

Counting entities is one thing; saying WHAT an entity is is another, and the
second has an evidence grade. This module holds that grade, and one decision
determines its whole shape:

    `grade` is COMPUTED, never typed.

There is no `grade` parameter, no `override_grade`, no `assume`, no confidence
number, and no `strongest()`/`max()` -- only `weakest()` is exported. Numbers
invite averages, and an average is a grade increase in disguise. If `grade`
could be a parameter, the rule "two sources of the same kind are not
corroboration" would fall to the author's discipline, and the 17th drawing
would be handled by someone who has never read UPLIFT-08.

Two properties that must not be lost when this file is edited:

1.  **Pure.** Zero imports from `app.*`; stdlib only. `store.py` imports
    `store_landuse` and friends at the BOTTOM of its file precisely to break
    the import cycle; one import from here into `.store` closes that cycle
    again. That is why units are **handed in by the caller** through
    `Scope.units`, never fetched here from `store._unit_names()`.

2.  **Zero drawing-specific constants** (G1). The vocabulary below is the
    vocabulary of DXF structure -- layer table, annotation, coordinates -- not
    the vocabulary of one drawing. The tokens that decide whether a string
    STATES something live in config, not here. The moment someone adds
    `ZEROLOT_TERM` to one of the enums, G1 breaks and this module becomes a
    Janadriyah-only solution.

The decision is recorded in `docs/DECISIONS-LOG.md` D-082, together with the
two alternatives that lost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, NoReturn, Sequence

#: Raised only when the emitted shape changes such that an old and a new
#: response cannot be told apart without looking at them.
EVIDENCE_CONTRACT_VERSION: int = 1


# --- Grade -------------------------------------------------------------------


class Grade(str, Enum):
    """The four UPLIFT-08 grades. Their order is the contract."""

    UNKNOWN = "unknown"
    INFERRED = "inferred"
    STATED = "stated"
    CORROBORATED = "corroborated"


#: Weakest to strongest. Used by `_weaker`; do not reorder.
GRADES: tuple[Grade, ...] = (
    Grade.UNKNOWN,
    Grade.INFERRED,
    Grade.STATED,
    Grade.CORROBORATED,
)

_RANK: Mapping[Grade, int] = MappingProxyType({g: i for i, g in enumerate(GRADES)})


def _weaker(a: Grade, b: Grade) -> Grade:
    """The weaker of two grades.

    Deliberately without a counterpart. There is no `_stronger`, `max_grade`,
    or `strongest` in this module, and none may be added: the only way a grade
    goes up is by handing in a source that really does state the claim.
    """
    return a if _RANK[a] <= _RANK[b] else b


# --- Origin and corpus -------------------------------------------------------


class Origin(str, Enum):
    """WHERE in the file the observation was seen. Not what it means."""

    LAYER_NAME = "layer_name"
    BLOCK_NAME = "block_name"
    LAYER_DESCRIPTION = "layer_description"
    ENTITY_TEXT = "entity_text"
    DIMENSION_TEXT = "dimension_text"
    BLOCK_ATTRIBUTE = "block_attribute"
    XDATA = "xdata"
    GEOMETRY = "geometry"
    TOPOLOGY = "topology"
    ABSENCE = "absence"
    EXTERNAL_KEY = "external_key"
    HUMAN = "human"


class Corpus(str, Enum):
    """Provenance: the axis that decides corroboration.

    Two sources corroborate each other only if both can be wrong for
    DIFFERENT reasons. That is a property of their corpus, not a property of
    their field name.
    """

    NAME_STRING = "name_string"
    ANNOTATION = "annotation"
    STRUCTURED = "structured"
    COORDINATES = "coordinates"
    ABSENCE = "absence"
    OUTSIDE_FILE = "outside_file"


#: The mapping that carries the entire weight of the rule "two sources of the
#: same kind are not corroboration". Three of its lines are the decisive ones,
#: and each has its reason:
#:
#: - LAYER_NAME + BLOCK_NAME + LAYER_DESCRIPTION collapse into NAME_STRING.
#:   Naming a layer "Mosque" and naming a block "Mosque" is one naming habit;
#:   both are wrong together if the drafter reused the term. This is the
#:   two-layer-names pair that UPLIFT-08 itself names as its example of false
#:   corroboration.
#: - GEOMETRY + TOPOLOGY collapse into COORDINATES. The plot module and the
#:   tiling of the block boundary are ONE act of drawing read two ways.
#: - ABSENCE stands alone and never states: elimination infers.
CORPUS_OF: Mapping[Origin, Corpus] = MappingProxyType(
    {
        Origin.LAYER_NAME: Corpus.NAME_STRING,
        Origin.BLOCK_NAME: Corpus.NAME_STRING,
        Origin.LAYER_DESCRIPTION: Corpus.NAME_STRING,
        Origin.ENTITY_TEXT: Corpus.ANNOTATION,
        Origin.DIMENSION_TEXT: Corpus.ANNOTATION,
        Origin.BLOCK_ATTRIBUTE: Corpus.STRUCTURED,
        Origin.XDATA: Corpus.STRUCTURED,
        Origin.GEOMETRY: Corpus.COORDINATES,
        Origin.TOPOLOGY: Corpus.COORDINATES,
        Origin.ABSENCE: Corpus.ABSENCE,
        Origin.EXTERNAL_KEY: Corpus.OUTSIDE_FILE,
        Origin.HUMAN: Corpus.OUTSIDE_FILE,
    }
)

# At module level, not inside a function: a new Origin added without a corpus
# decision will bring down the import, rather than quietly becoming an
# independent corpus and raising the grade of an answer that does not deserve
# raising.
assert set(CORPUS_OF) == set(Origin), (
    "every Origin must have a corpus decision; without one it would count as "
    "an independent origin and raise the evidence grade with nothing behind it"
)

#: The corpora that never STATE. Coordinates and elimination infer; a source
#: outside the file is the easiest route for smuggling knowledge between
#: drawings (G10), so it is closed until a spec asks for it.
NEVER_SPEAKS: frozenset[Corpus] = frozenset(
    {Corpus.COORDINATES, Corpus.ABSENCE, Corpus.OUTSIDE_FILE}
)


class ConfigLayer(str, Enum):
    """The config layer that decided the classification (G4).

    This is a DIFFERENT axis from `Grade`. `grade` answers what the file says;
    `verified` answers whether a human has confirmed the mapping. The two can
    differ and that is not a contradiction: a layer named literally
    "MOSQUE-PARCEL-OUTLINE" really is stated by the file (`stated`) while
    nobody has yet confirmed that at this contractor that layer means a parcel
    (`verified: false`).
    """

    DRAWING_OVERRIDE = "drawing_override"
    #: A mapping that applies to a FAMILY of drawings, not to one drawing. It
    #: is used only if the drawing's own config declares `standard: <name>`, so
    #: the claim "this drawing follows that standard" stays a recorded human
    #: decision, not meaning that walks between files on its own (G10).
    #:
    #: `verified` deliberately does NOT cover it. Someone has confirmed that
    #: `VL2` means residential in this standard; nobody has necessarily checked
    #: that layer `VL2` in THIS drawing is the same layer. The classification
    #: still holds and can still be counted; what is missing is only the tick
    #: that means "someone has looked at it here".
    SHARED_STANDARD = "shared_standard"
    GLOBAL_PATTERN = "global_pattern"
    DEFAULT_USE = "default_use"
    NO_CONFIG = "no_config"


# --- Errors ------------------------------------------------------------------


ERROR_CODES: frozenset[str] = frozenset(
    {
        "UNKNOWN_ORIGIN",
        "OBSERVATION_WITHOUT_DETAIL",
        "OBSERVATION_WITHOUT_OBSERVED",
        "REFERENCE_REQUIRED",
        "SCOPE_WITHOUT_UNITS",
        "UNKNOWN_WITHOUT_GAP",
        "CROSS_DRAWING",
        "MEANING_WITHOUT_SCOPE",
        "EVIDENCE_KEY_TAKEN",
        "UNCOVERED_MEANING",
        "QUANTITY_NOT_A_NUMBER",
        "QUANTITY_WITHOUT_METHOD",
        "QUANTITY_WITHOUT_REASON",
        "CEILING_WOULD_RAISE",
    }
)


class EvidenceError(ValueError):
    """A violation of the evidence contract.

    Its shape deliberately mirrors `store.MeasureRefused` (`code`/`message`/
    `hint`) so that the `@app.exception_handler` pattern already in `main.py`
    can be reused, and a violation comes out as an actionable 400 rather than a
    500 that can only be stared at.
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


# --- Scope -------------------------------------------------------------------


_UNIT_KEYS = ("name", "declared_in_file", "length_unit", "area_unit", "space")


@dataclass(frozen=True, slots=True)
class Scope:
    """The scope of a claim: which layout, whether block definitions are in,
    and which units.

    `units` is handed in VERBATIM by the caller from `store._unit_names()` or
    `store._unit_names_for_layouts()`. This module does not fetch them itself
    -- see the purity note at the head of the file. It is required, with no
    default, because 11 of the 16 drawings are in inches and 3 state no unit at
    all: a response that could be published without naming a unit would write
    "m" merely because the drawing that happens to be open uses metres (G2).
    """

    layout: str | None
    includes_block_definitions: bool
    units: Mapping[str, Any]
    note: str

    def __post_init__(self) -> None:
        missing = [k for k in _UNIT_KEYS if k not in self.units]
        if missing:
            raise EvidenceError(
                "SCOPE_WITHOUT_UNITS",
                f"Scope.units is missing keys: {', '.join(missing)}.",
                "Hand in the result of store._unit_names() or "
                "_unit_names_for_layouts() as it is. A unit the drawing does "
                "not state still has to come through here as null plus its "
                "reason, not be dropped.",
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "includes_block_definitions": self.includes_block_definitions,
            "units": dict(self.units),
            "note": self.note,
        }


# --- Reading -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reading:
    """How `observed` came to be read, and how strongly it may speak.

    It exists because UPLIFT-04 rules that an SHX reading is always `inferred`,
    while E3 gives `corroborated` to the mosque on the strength of an Arabic
    label inside its parcel -- a label read through that same SHX map. A blind
    ceiling would drop the mosque to `stated` and break the contract that
    demands it. So the ceiling is conditional: a reading whose raw string is in
    the U04 verified corpus can still reach `stated`; outside that it drops to
    `inferred`, so corroboration on top of guessed glyphs stays impossible.
    """

    verbatim: bool
    method: str
    corpus_verified: bool

    @property
    def ceiling(self) -> Grade:
        if self.verbatim or self.corpus_verified:
            return Grade.STATED
        return Grade.INFERRED

    def as_dict(self) -> dict[str, Any]:
        return {
            "verbatim": self.verbatim,
            "method": self.method,
            "corpus_verified": self.corpus_verified,
        }


#: Bytes as they are in the file. The default for almost every observation.
VERBATIM: Reading = Reading(verbatim=True, method="verbatim", corpus_verified=True)


# --- Observation -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    """One concrete thing seen inside the file.

    `detail` is for humans; `observed` is for machines. Both exist because
    prose cannot be tested: `speaks_for` tests `observed`, never `detail`.
    Filling `observed` with the same sentence as `detail` makes the literal
    test meaningless -- structure cannot prevent that, and it is recorded in
    D-082 as a known hole.
    """

    origin: Origin
    detail: str
    observed: str | None = None
    observed_raw: str | None = None
    locator: str | None = None
    layout: str | None = None
    reading: Reading = VERBATIM
    reference: str | None = None

    def __post_init__(self) -> None:
        if self.origin not in CORPUS_OF:
            raise EvidenceError(
                "UNKNOWN_ORIGIN",
                f"origin {self.origin!r} has no corpus decision.",
                "Add it to CORPUS_OF with a written reason; an origin without "
                "a corpus would count as independent and raise the grade.",
            )
        if not (self.detail or "").strip():
            raise EvidenceError(
                "OBSERVATION_WITHOUT_DETAIL",
                "Observation.detail is empty.",
                "Write the concrete thing that was seen, not the conclusion "
                "drawn from it. A source without a detail cannot be checked by "
                "anyone and only raises the grade for free.",
            )
        speaks_possible = CORPUS_OF[self.origin] not in NEVER_SPEAKS
        if speaks_possible and not (self.observed or "").strip():
            raise EvidenceError(
                "OBSERVATION_WITHOUT_OBSERVED",
                f"observation {self.origin.value} without an `observed`.",
                "Hand in the string that was actually read in the file. This "
                "corpus can STATE something, and what decides whether it "
                "really states it is the content of that string -- not the "
                "caller's intent.",
            )
        if self.origin in (Origin.EXTERNAL_KEY, Origin.HUMAN) and not (
            self.reference or ""
        ).strip():
            raise EvidenceError(
                "REFERENCE_REQUIRED",
                f"observation {self.origin.value} without a `reference`.",
                "A source outside the file must name the document or the "
                "person, so that its claim can be traced back to something "
                "real.",
            )

    @property
    def corpus(self) -> Corpus:
        return CORPUS_OF[self.origin]

    def as_dict(self, *, speaks_token: str | None = None) -> dict[str, Any]:
        return {
            "origin": self.origin.value,
            "corpus": self.corpus.value,
            "speaks": speaks_token is not None,
            "speaks_token": speaks_token,
            "detail": self.detail,
            "observed": self.observed,
            "observed_raw": self.observed_raw,
            "locator": self.locator,
            "layout": self.layout,
            "reading": self.reading.as_dict(),
            "reference": self.reference,
        }


@dataclass(frozen=True, slots=True)
class Contradiction:
    """Something in the file that does not agree with the claim.

    Recorded, not detected: detecting it needs point-in-polygon (U03) and
    duplicate detection (U07) to exist first. What matters now is that it
    LOWERS the grade -- because two sources from different corpora that
    disagree must not read as corroboration.
    """

    detail: str
    observation: Observation


# --- Claim and config provenance ---------------------------------------------


@dataclass(frozen=True, slots=True)
class Claim:
    """What is claimed, and which words count as stating it.

    `tokens` come from config (`_patterns.yaml`, the `vocabulary` block), never
    from `.py` -- G1. Empty tokens are a VALID state: that claim simply can
    never be `stated`, because there is no word to look for that would prove
    the file states it.
    """

    value: str
    tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        folded: list[str] = []
        for t in self.tokens:
            f = str(t).casefold().strip()
            if f and f not in folded:
                folded.append(f)
        object.__setattr__(self, "tokens", tuple(folded))

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "tokens": list(self.tokens)}


@dataclass(frozen=True, slots=True)
class Provenance:
    """Which config layer decided this classification.

    `verified` is DERIVED and cannot be typed. UPLIFT-02 used to write
    `{use: open_space, verified: true, source: "layer name"}` -- one source,
    one corpus, marked verified. A `verified` boolean plus `source` prose IS a
    mechanism for raising the evidence grade in YAML form, and this module
    removes it rather than wrapping it. The `source:` prose is demoted to
    `note`; it is a remark, not a source.
    """

    config_layer: ConfigLayer
    config_version: int | None = None
    note: str = ""

    @property
    def verified(self) -> bool:
        return self.config_layer is ConfigLayer.DRAWING_OVERRIDE

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_layer": self.config_layer.value,
            "config_version": self.config_version,
            "verified": self.verified,
            "note": self.note,
        }


# --- Literal test ------------------------------------------------------------


_HAS_LATIN = re.compile(r"[A-Za-z]")
_LOOKS_LIKE_CODE = re.compile(r"^[A-Za-z]{1,3}\d+$")
_MIN_LATIN_TOKEN = 4


def speaks_for(observed: str | None, tokens: Sequence[str]) -> str | None:
    """The token that string really does STATE, or None.

    This is the only place where the module inspects CONTENT, not only origin.
    Without it, "the layer name matches the regex" automatically becomes "the
    file states it", and a global pattern gets promoted into a statement.

    Latin tokens are tested with word boundaries, so `CAR PARKING AREA` does
    NOT state `park`. Non-Latin tokens -- Arabic, for instance -- are tested
    with plain containment, because word boundaries cannot be relied on there.
    No stemming, no fuzzy matching, no edit distance: what is being looked for
    is evidence that the file uses that word, not that it resembles it.
    """
    if not observed:
        return None
    # An underscore is treated as a separator, not a letter. It is a WORD
    # character to the regex engine, so without this line
    # `CS-Land use-00_Education` would not state "education" -- the word
    # boundary would never be found. The two most widely used layer naming
    # conventions, AIA/NCS and ISO 13567, both use the underscore as a field
    # separator, so this is not an edge case but the normal shape of a layer
    # name.
    hay = str(observed).casefold().replace("_", " ")
    for token in tokens:
        t = str(token).casefold().strip()
        if not t:
            continue
        if _HAS_LATIN.search(t):
            if re.search(rf"\b{re.escape(t)}\b", hay):
                return t
        elif t in hay:
            return t
    return None


def load_vocabulary(
    raw: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """Read the `vocabulary` block from config. NEVER raises.

    Refused and dropped: tokens that are not strings; Latin tokens shorter than
    four letters; tokens that are code-shaped, such as `vl2`. Without those
    refusals, someone adds `vl` to the residential vocabulary and 2,380 plots
    climb to `stated` -- E1 and E8 collapse without one line of Python
    changing. Config is this module's structural hole, and this is the fence
    that can be put across it.

    It does not raise because a wrong config must not become an outage: D-067
    records a hard failure at startup that left cad-api never healthy. What is
    refused is reported through `problems`, and a claim that loses all of its
    tokens remains valid -- it simply can never be `stated`.
    """
    vocabulary: dict[str, tuple[str, ...]] = {}
    problems: list[str] = []

    for use, spec in (raw or {}).items():
        raw_tokens: Any = spec
        if isinstance(spec, Mapping):
            raw_tokens = spec.get("tokens", [])
        if isinstance(raw_tokens, str) or not isinstance(raw_tokens, Iterable):
            problems.append(f"{use}: the tokens block is not a list; the whole entry is skipped")
            vocabulary[str(use)] = ()
            continue

        kept: list[str] = []
        for token in raw_tokens:
            if not isinstance(token, str):
                problems.append(f"{use}: token {token!r} is not a string; dropped")
                continue
            t = token.casefold().strip()
            if not t:
                problems.append(f"{use}: empty token; dropped")
                continue
            if _LOOKS_LIKE_CODE.match(t):
                problems.append(
                    f"{use}: token {token!r} is code-shaped. A code is not a "
                    "definition of itself; accepting it would make a typology "
                    "code read as a statement of land use. Dropped."
                )
                continue
            if _HAS_LATIN.search(t) and len(t) < _MIN_LATIN_TOKEN:
                problems.append(
                    f"{use}: Latin token {token!r} is shorter than "
                    f"{_MIN_LATIN_TOKEN} letters; too easy to match by chance. Dropped."
                )
                continue
            if t not in kept:
                kept.append(t)
        vocabulary[str(use)] = tuple(kept)

    return vocabulary, problems


# --- Evidence ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Evidence:
    """Evidence for ONE claim.

    Evidence attaches to a claim, not to an object. Evidence that is
    `corroborated` for "this parcel is a place of worship" is not evidence for
    "there are six of them"; there is deliberately no `retag()` or
    `narrowed_to()`, so reusing this object for a second claim forces its
    author to list the sources again -- and that is where they realise they do
    not have any.
    """

    claim: Claim
    drawing_id: str
    scope: Scope
    provenance: Provenance
    observations: tuple[Observation, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    not_established: str | None = None
    how_to_verify: str | None = None
    _ceiling: Grade = Grade.CORROBORATED
    _forced_unknown: str | None = None
    _spread: Mapping[str, int] | None = None
    #: Filled only by `weakest()`. An aggregate has no observations of its own
    #: -- if its grade were computed from observations it would always fall to
    #: `unknown`, and a summary whose every row is strong would report knowing
    #: nothing.
    _floor: Grade | None = None

    # -- computation ----------------------------------------------------------

    @property
    def speaking(self) -> tuple[Observation, ...]:
        """The observations that really do STATE this claim."""
        return tuple(
            o
            for o in self.observations
            if o.corpus not in NEVER_SPEAKS
            and speaks_for(o.observed, self.claim.tokens) is not None
        )

    @property
    def _stating(self) -> tuple[Observation, ...]:
        return tuple(o for o in self.speaking if o.reading.ceiling is Grade.STATED)

    @property
    def independent_corpora(self) -> frozenset[Corpus]:
        return frozenset(o.corpus for o in self._stating)

    @property
    def grade(self) -> Grade:
        if self._forced_unknown is not None:
            return Grade.UNKNOWN

        if self._floor is not None:
            return _weaker(self._floor, self._ceiling)

        stating = self._stating
        corpora = {o.corpus for o in stating}
        # Places, not corpora: one entity read twice -- once as text, once as
        # an attribute -- is one piece of evidence, not two. If this were keyed
        # on (corpus, place), that pair would climb to corroborated and the
        # whole rule would leak through a single handle.
        places = {o.locator for o in stating}

        if len(corpora) >= 2 and len(places) >= 2:
            g = Grade.CORROBORATED
        elif corpora:
            g = Grade.STATED
        elif self.observations:
            g = Grade.INFERRED
        else:
            g = Grade.UNKNOWN

        if self.contradictions:
            g = _weaker(g, Grade.INFERRED)
        return _weaker(g, self._ceiling)

    @property
    def verified(self) -> bool:
        return self.provenance.verified

    # -- constructors ---------------------------------------------------------

    @classmethod
    def of(
        cls,
        claim: Claim,
        *,
        drawing_id: str,
        scope: Scope,
        provenance: Provenance,
        observations: Sequence[Observation],
        contradictions: Sequence[Contradiction] = (),
        not_established: str | None = None,
        how_to_verify: str | None = None,
        ceiling: Grade = Grade.CORROBORATED,
    ) -> "Evidence":
        return cls(
            claim=claim,
            drawing_id=drawing_id,
            scope=scope,
            provenance=provenance,
            observations=tuple(observations),
            contradictions=tuple(contradictions),
            not_established=not_established,
            how_to_verify=how_to_verify,
            _ceiling=ceiling,
        )

    @classmethod
    def unknown(
        cls,
        claim: Claim,
        *,
        drawing_id: str,
        scope: Scope,
        provenance: Provenance,
        not_established: str,
        how_to_verify: str,
        observations: Sequence[Observation] = (),
    ) -> "Evidence":
        """An "I don't know" that must name what is missing and where to ask.

        Both are required and non-empty. An empty `unknown` is an answer that
        is easy to skip past, and E1 tests precisely whether the agent can say
        it does not know in a way that is useful.
        """
        if not (not_established or "").strip() or not (how_to_verify or "").strip():
            raise EvidenceError(
                "UNKNOWN_WITHOUT_GAP",
                "grade unknown without `not_established` or `how_to_verify`.",
                "Name what was looked for and not found, then where the answer "
                "might be. Without both, 'I don't know' is only an emptiness "
                "that its reader will skip past.",
            )
        return cls(
            claim=claim,
            drawing_id=drawing_id,
            scope=scope,
            provenance=provenance,
            observations=tuple(observations),
            not_established=not_established,
            how_to_verify=how_to_verify,
            _forced_unknown=not_established,
        )

    @classmethod
    def for_layer(
        cls,
        claim: Claim,
        *,
        drawing_id: str,
        scope: Scope,
        provenance: Provenance,
        layer_name: str,
        extra: Sequence[Observation] = (),
        not_established: str | None = None,
        how_to_verify: str | None = None,
    ) -> "Evidence":
        """A layer-name-based classification, which therefore cannot be
        zero-source."""
        first = Observation(
            origin=Origin.LAYER_NAME,
            detail=f"layer {layer_name!r}",
            observed=layer_name,
            locator=layer_name,
            layout=scope.layout,
        )
        return cls.of(
            claim,
            drawing_id=drawing_id,
            scope=scope,
            provenance=provenance,
            observations=(first, *extra),
            not_established=not_established,
            how_to_verify=how_to_verify,
        )

    # -- transformations that can only lower ----------------------------------

    def unresolved(self, because: str, how_to_verify: str) -> "Evidence":
        """Force `unknown`, whatever the observations say (G3)."""
        if not (because or "").strip() or not (how_to_verify or "").strip():
            raise EvidenceError(
                "UNKNOWN_WITHOUT_GAP",
                "unresolved() without a reason or without a way to verify.",
                "Name what makes this claim impossible to establish, and where "
                "the answer might be.",
            )
        return replace(
            self,
            not_established=because,
            how_to_verify=how_to_verify,
            _forced_unknown=because,
        )

    def with_ceiling(self, ceiling: Grade) -> "Evidence":
        """Lower the ceiling. Raising it raises an error."""
        if _RANK[ceiling] > _RANK[self._ceiling]:
            raise EvidenceError(
                "CEILING_WOULD_RAISE",
                f"ceiling {self._ceiling.value} cannot be raised to {ceiling.value}.",
                "An evidence grade can only go down through this route. To "
                "raise it, hand in a source that really does state the claim.",
            )
        return replace(self, _ceiling=ceiling)

    def combined_with(self, other: "Evidence") -> "Evidence":
        """Combine two pieces of evidence; the result is never stronger than
        the weaker one."""
        if other.drawing_id != self.drawing_id:
            raise EvidenceError(
                "CROSS_DRAWING",
                f"evidence from {self.drawing_id} combined with {other.drawing_id}.",
                "A layer name that means school in this drawing does not "
                "necessarily mean school in another (G10). Classification is "
                "read again per drawing.",
            )
        merged = replace(
            self,
            observations=self.observations + other.observations,
            contradictions=self.contradictions + other.contradictions,
        )
        return merged.with_ceiling(_weaker(merged._ceiling, other.grade))

    # -- prose ----------------------------------------------------------------

    def why_this_grade(self) -> str:
        g = self.grade
        if g is Grade.UNKNOWN:
            if self._forced_unknown:
                return (
                    f"Forced unknown: {self._forced_unknown} Nothing here may be "
                    "filled in from how this term is usually used elsewhere."
                )
            return "Not a single observation was offered for this claim."
        corpora = sorted(c.value for c in self.independent_corpora)
        if g is Grade.CORROBORATED:
            return (
                f"Two independent corpora state it: {', '.join(corpora)}. "
                "Two sources from the same corpus will never be corroboration, "
                "because both can be wrong for the same reason."
            )
        if g is Grade.STATED:
            why = f"One corpus states it: {', '.join(corpora)}."
            if self.contradictions:
                why += " A contradiction is on record, so the grade is held back."
            return why
        speak = len(self.speaking)
        return (
            f"No source STATES this claim ({speak} observations carry its "
            f"word, {len(self.observations)} observations in all). Independent "
            "bases do not accumulate into corroboration: what infers keeps "
            "inferring, however many of them there are."
        )

    def statement(self) -> str:
        """One sentence the agent can quote.

        Generated, never handed in by the caller. On `unknown` it never carries
        a candidate answer -- saying "in other projects this term usually means
        X" is bait whose negation a paraphrase can simply delete, and E8 tests
        exactly that.
        """
        g = self.grade
        value = self.claim.value
        if g is Grade.UNKNOWN:
            gap = self.not_established or "not established by this drawing"
            return (
                f"Not known from this drawing: {value}. {gap}. "
                f"To establish it: {self.how_to_verify}."
            )
        if g is Grade.CORROBORATED:
            return (
                f"{value} — stated by the file and corroborated. "
                + self._sources_sentence()
            )
        if g is Grade.STATED:
            base = f"{value} — stated by the file. " + self._sources_sentence()
            if self.not_established:
                base += f" Not yet established: {self.not_established}."
            return base
        base = (
            f"{value} — INFERRED, not stated. " + self._sources_sentence()
        )
        if self.not_established:
            base += f" Not yet established: {self.not_established}."
        return base

    def _sources_sentence(self) -> str:
        bits = [o.detail for o in self.observations[:4]]
        if not bits:
            return "No sources are attached."
        more = len(self.observations) - len(bits)
        tail = f", and {more} further bases" if more > 0 else ""
        return "Its bases: " + "; ".join(bits) + tail + "."

    # -- serialisation --------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return self.to_json()

    def to_json(self, *, max_sources: int = 6, max_detail: int = 160) -> dict[str, Any]:
        """Every contract key, always, including the ones whose value is null.

        There is no `exclude_none`. A missing `not_established` in a response
        must read as a visible bug, not as a claim that the answer has no
        limits.
        """
        shown = self.observations[:max_sources]
        detail_truncated = False
        sources: list[dict[str, Any]] = []
        for o in shown:
            row = o.as_dict(speaks_token=speaks_for(o.observed, self.claim.tokens))
            if len(row["detail"]) > max_detail:
                row["detail"] = row["detail"][: max_detail - 1] + "…"
                detail_truncated = True
            sources.append(row)

        out: dict[str, Any] = {
            "contract_version": EVIDENCE_CONTRACT_VERSION,
            "claim": self.claim.as_dict(),
            "drawing_id": self.drawing_id,
            "grade": self.grade.value,
            "verified": self.verified,
            "independent_corpora": sorted(c.value for c in self.independent_corpora),
            "why_this_grade": self.why_this_grade(),
            "sources": sources,
            "contradictions": [
                {"detail": c.detail, "observation": c.observation.as_dict()}
                for c in self.contradictions
            ],
            "sources_total": len(self.observations),
            "sources_truncated": len(self.observations) > len(shown),
            "detail_truncated": detail_truncated,
            "not_established": self.not_established,
            "how_to_verify": self.how_to_verify,
            "provenance": self.provenance.as_dict(),
            "scope": self.scope.as_dict(),
            "statement": self.statement(),
        }
        if self._spread is not None:
            out["grade_spread"] = dict(self._spread)
        return out


# --- Aggregation -------------------------------------------------------------


def grade_spread(parts: Iterable[Evidence]) -> dict[str, int]:
    """How many rows sit at each grade. Must accompany every aggregate."""
    counts = {g.value: 0 for g in GRADES}
    for p in parts:
        counts[p.grade.value] += 1
    return counts


def weakest(parts: Iterable[Evidence], claim: Claim) -> Evidence:
    """Aggregate grade = the weakest of its members, AND the spread travels too.

    "Weakest wins" on its own destroys the signal: one `inferred` row among
    forty makes the whole summary read `inferred`, and within two days its
    reader stops reading the grade at all. That is why `_spread` always travels
    with it, and the response-level sentence is generated from it.
    """
    items = list(parts)
    if not items:
        raise EvidenceError(
            "UNKNOWN_WITHOUT_GAP",
            "weakest() was called without a single piece of evidence.",
            "An empty summary still has to say why it is empty; use "
            "Evidence.unknown() with its reason.",
        )
    first = items[0]
    for other in items:
        if other.drawing_id != first.drawing_id:
            raise EvidenceError(
                "CROSS_DRAWING",
                "weakest() is combining evidence from more than one drawing.",
                "One response belongs to one drawing (G10).",
            )
    floor = items[0].grade
    for p in items[1:]:
        floor = _weaker(floor, p.grade)
    spread = grade_spread(items)

    merged = Evidence(
        claim=claim,
        drawing_id=first.drawing_id,
        scope=first.scope,
        provenance=first.provenance,
        observations=(),
        contradictions=(),
        not_established=next((p.not_established for p in items if p.not_established), None),
        how_to_verify=next((p.how_to_verify for p in items if p.how_to_verify), None),
        _floor=floor,
        _forced_unknown=(
            "not a single row could be established" if floor is Grade.UNKNOWN else None
        ),
        _spread=MappingProxyType(dict(spread)),
    )
    return merged


# --- Quantity ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Quantity:
    """A number that knows what it is a number OF, and how it was measured.

    It deliberately has no `__float__`, `__int__`, `or_zero()`, or
    `value_or()`. A HATCH carries no area; reporting `0 m²` for it is the WRONG
    number, not an empty one, and the only way to make sure that never happens
    is to give the withheld value no route to turn into a zero.
    """

    basis: str
    method: str
    value: float | None
    unit: str | None
    unit_reason: str | None = None
    withheld_reason: str | None = None

    def __post_init__(self) -> None:
        if not (self.method or "").strip():
            raise EvidenceError(
                "QUANTITY_WITHOUT_METHOD",
                f"quantity {self.basis!r} without a `method`.",
                "Say how it was measured. A 'distance' that does not say it is "
                "a straight line will be read as a walking distance.",
            )
        if self.value is None and not (self.withheld_reason or "").strip():
            raise EvidenceError(
                "QUANTITY_WITHOUT_REASON",
                f"quantity {self.basis!r} withheld without a reason.",
                "A withheld number is always accompanied by its reason. "
                "Without it, it cannot be told apart from a missing number.",
            )
        if self.value is not None and self.unit is None and not (
            self.unit_reason or ""
        ).strip():
            raise EvidenceError(
                "QUANTITY_WITHOUT_REASON",
                f"quantity {self.basis!r} without a unit and without a reason.",
                "A drawing that states no unit may still give a number, but it "
                "has to say that its unit is not stated (G2).",
            )

    @classmethod
    def measured(
        cls,
        *,
        basis: str,
        method: str,
        value: float,
        unit: str | None,
        unit_reason: str | None = None,
    ) -> "Quantity":
        return cls(
            basis=basis,
            method=method,
            value=value,
            unit=unit,
            unit_reason=unit_reason,
        )

    @classmethod
    def withheld(cls, *, basis: str, method: str, reason: str) -> "Quantity":
        return cls(
            basis=basis,
            method=method,
            value=None,
            unit=None,
            withheld_reason=reason,
        )

    @property
    def is_withheld(self) -> bool:
        return self.value is None

    def __add__(self, other: "Quantity") -> "Quantity":
        if not isinstance(other, Quantity):
            raise EvidenceError(
                "QUANTITY_NOT_A_NUMBER",
                f"a quantity cannot be added to a {type(other).__name__}.",
                "Use evidence.total(...). Adding through bare numbers throws "
                "away the basis, the unit, and the reason it was withheld.",
            )
        basis = f"{self.basis} + {other.basis}"
        if self.is_withheld or other.is_withheld:
            reasons = [
                r
                for r in (self.withheld_reason, other.withheld_reason)
                if r
            ]
            return Quantity.withheld(
                basis=basis, method=self.method, reason="; ".join(reasons)
            )
        return Quantity(
            basis=basis,
            method=self.method,
            value=(self.value or 0.0) + (other.value or 0.0),
            unit=self.unit if self.unit == other.unit else None,
            unit_reason=(
                None
                if self.unit == other.unit
                else "summed from different units"
            ),
        )

    def __radd__(self, other: Any) -> NoReturn:
        raise EvidenceError(
            "QUANTITY_NOT_A_NUMBER",
            "a quantity was used inside sum() or added to a number.",
            "Use evidence.total(...). sum() starts from the int 0, and a zero "
            "that slips into a total is exactly the wrong number this contract "
            "was built to prevent.",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "not_zero": True if self.is_withheld else None,
            "withheld_reason": self.withheld_reason,
            "basis": self.basis,
            "method": self.method,
            "unit": self.unit,
            "unit_reason": self.unit_reason,
        }


def total(items: Iterable[Quantity], *, basis: str, method: str) -> Quantity:
    """Add quantities up without ever passing through a bare zero."""
    rows = list(items)
    if not rows:
        return Quantity.withheld(
            basis=basis,
            method=method,
            reason="there is not a single value that could be added up",
        )
    withheld = [r for r in rows if r.is_withheld]
    measured = [r for r in rows if not r.is_withheld]
    if not measured:
        return Quantity.withheld(
            basis=basis,
            method=method,
            reason="; ".join(dict.fromkeys(r.withheld_reason or "" for r in withheld)),
        )
    units = {r.unit for r in measured}
    unit = next(iter(units)) if len(units) == 1 else None
    q = Quantity(
        basis=basis,
        method=method,
        value=sum(r.value or 0.0 for r in measured),
        unit=unit,
        unit_reason=None if unit is not None else "summed from different units",
    )
    if withheld:
        # A total that hides the fact that part of it was not measured is a
        # total that lies about its scope (G7).
        return replace(
            q,
            basis=f"{basis} ({len(measured)} measured, {len(withheld)} withheld)",
        )
    return q


# --- Seam and audit ----------------------------------------------------------


MEANING_BEARING_KEYS: frozenset[str] = frozenset(
    {
        "land_use",
        "land_use_subtype",
        "use",
        "uses",
        "subtype",
        "classification",
        "typology",
        "text_reading",
        "label_for",
        "matched_parcel",
        "duplicate_of",
        "crs",
        "lat",
        "lon",
        "business_tag",
    }
)

QUANTITY_KEYS: frozenset[str] = frozenset(
    {
        "total_area",
        "total_length",
        "area",
        "length",
        "distance",
        "perimeter",
        "mean",
        "median",
        "sd",
    }
)

_CONTRACT_KEYS = ("grade", "claim", "not_established", "how_to_verify")


def _is_contract_evidence(value: Any) -> bool:
    return isinstance(value, Mapping) and all(k in value for k in _CONTRACT_KEYS)


def attach(
    payload: dict[str, Any],
    evidence: Evidence | Mapping[str, Evidence],
    *,
    scope_key: str = "scope_note",
) -> dict[str, Any]:
    """Attach evidence to a response, and refuse to publish it if the contract
    is not complete.

    It raises rather than quietly repairing, because everything raised here is
    an answer that would look correct while being incomplete.
    """
    if not (payload.get(scope_key) or "").strip():
        raise EvidenceError(
            "MEANING_WITHOUT_SCOPE",
            f"the response carries meaning without {scope_key!r}.",
            "Name which layout, whether block definitions are included, and "
            "which units. A classification without a scope cannot be checked "
            "by anyone.",
        )

    existing = payload.get("evidence")
    if existing is not None and not _is_contract_evidence(existing):
        raise EvidenceError(
            "EVIDENCE_KEY_TAKEN",
            "the 'evidence' key in this response is already used for another shape.",
            "That key belongs to the UPLIFT-08 contract. The old shape of the "
            "facts behind why_empty is called 'why_empty_facts'; free prose "
            "may not use this name.",
        )

    one = evidence if isinstance(evidence, Evidence) else None
    many = None if one is not None else dict(evidence)  # type: ignore[arg-type]

    ids = (
        {one.drawing_id}
        if one is not None
        else {e.drawing_id for e in (many or {}).values()}
    )
    if len(ids) > 1:
        raise EvidenceError(
            "CROSS_DRAWING",
            "one response carries evidence from more than one drawing.",
            "Classification is read again per drawing; one response belongs to "
            "one drawing (G10).",
        )
    if payload.get("drawing_id") and ids and payload["drawing_id"] not in ids:
        raise EvidenceError(
            "CROSS_DRAWING",
            f"evidence for {next(iter(ids))} was attached to response {payload['drawing_id']}.",
            "The wrong drawing is a wrong answer, not a small mismatch.",
        )

    if one is not None:
        payload["evidence"] = one.to_json()
    else:
        for key, ev in (many or {}).items():
            payload.setdefault("evidence_by_use", {})[key] = ev.to_json()

    uncovered = sorted(
        k
        for k in payload
        if k in MEANING_BEARING_KEYS
        and not payload.get("evidence")
        and not payload.get("evidence_by_use")
    )
    if uncovered:
        raise EvidenceError(
            "UNCOVERED_MEANING",
            f"meaning-bearing keys without evidence: {', '.join(uncovered)}.",
            "Every claim about WHAT something is carries its evidence grade.",
        )
    return payload


def forward(src: Mapping[str, Any], dst: dict[str, Any]) -> dict[str, Any]:
    """The only legitimate route for this contract across the cad-mcp seam.

    `_compact()` in `cad_mcp/server.py` is an allowlist built from nothing;
    anything it does not copy by name is lost, and its own comment records that
    a field has already been lost there once. One function that can be tested
    is better than a copy list that keeps getting longer.
    """
    for key in ("evidence", "evidence_by_use", "scope_note", "why_empty_facts"):
        if key in src:
            dst[key] = src[key]
    return dst


def audit_response(payload: Mapping[str, Any]) -> list[str]:
    """The list of contract violations in a response. Empty means clean.

    Used in two places: the invariant test across every drawing in the store,
    and dev mode. It is what turns "the contract" from something that is read
    into something that is run.
    """
    problems: list[str] = []
    ev = payload.get("evidence")
    by_use = payload.get("evidence_by_use") or {}
    blocks = [b for b in [ev, *by_use.values()] if isinstance(b, Mapping)]

    meaning = sorted(k for k in payload if k in MEANING_BEARING_KEYS)
    if meaning and not blocks:
        problems.append(
            f"V1 meaning-bearing keys without evidence: {', '.join(meaning)}"
        )
    if meaning and not (payload.get("scope_note") or "").strip():
        problems.append("V5 a response carrying meaning without scope_note")

    if ev is not None and not _is_contract_evidence(ev):
        problems.append("V7 the 'evidence' value is not the UPLIFT-08 contract shape")

    for block in blocks:
        missing = [k for k in _CONTRACT_KEYS if k not in block]
        if missing:
            problems.append(f"V2 evidence block is missing keys: {', '.join(missing)}")
        g = block.get("grade")
        if g is not None and g not in {x.value for x in GRADES}:
            problems.append(f"V3 grade outside the four values: {g!r}")
        if "not_established" not in block or "how_to_verify" not in block:
            problems.append("V7 evidence block emits contract keys selectively")

    declared = None
    for block in blocks:
        units = (block.get("scope") or {}).get("units") or {}
        if "declared_in_file" in units:
            declared = units["declared_in_file"]

    for key, value in payload.items():
        if key not in QUANTITY_KEYS:
            continue
        if isinstance(value, Mapping):
            if not (value.get("method") or "").strip():
                problems.append(f"V9 quantity {key!r} without a method")
            if value.get("value") is None and not value.get("withheld_reason"):
                problems.append(f"V9 quantity {key!r} withheld without a reason")
            if declared is False and value.get("unit"):
                problems.append(
                    f"V6 unit {value['unit']!r} on {key!r} while the drawing "
                    "states no unit"
                )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            problems.append(
                f"V4 {key!r} is a bare number {value!r}; use Quantity so that "
                "its basis, method, and unit travel with it"
            )

    ids = {b.get("drawing_id") for b in blocks if b.get("drawing_id")}
    if len(ids) > 1:
        problems.append(f"V8 one response carries {len(ids)} different drawing_ids")

    return problems
