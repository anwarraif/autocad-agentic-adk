"""UPLIFT-13 — business tags per typology, given from the screen.

Written from the UPLIFT-13 Definition of Done, before its implementation.

One sentence determines the shape of this whole file: **a business tag is a
claim about MEANING, and the file states nothing about business names.** So
what is tested is not only "the value is stored and read back" — but that this
value can never rise to `stated`, that its source is mandatory, and that it
never crosses between drawings.

Two kinds of test, both mandatory (G5):

- **Number tests** — 133 objects on one Janadriyah layer, the number UPLIFT-13
  itself uses on screen: *"VL2 — 133 objects will use this name."*
- **Invariant tests** — run over several different shapes of units, including
  a drawing that declares no units at all and a paper layout. The Janadriyah
  number will never catch a break there.

The whole file runs **without MongoDB**: the only things that touch the
database are `_tags()` and `_entities()`, and both are replaced with a fake
collection in `fake_store`. That is also why `put_layer_tag` can be tested
without a single real write to the shared cluster.
"""

from __future__ import annotations

import pathlib
import re
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import evidence as ev  # noqa: E402
from app import mongo  # noqa: E402
from app import store_recipes as sr  # noqa: E402


# --- tooling -----------------------------------------------------------------


DRAWING = "596212db022a3397"
OTHER_DRAWING = "b19dba2ed16edaf6"

#: The Janadriyah typology layer. This name lives in the TEST, never in `.py`
#: — G1. The number 133 comes from docs/GROUND-TRUTH-2.md.
TYPOLOGY_LAYER = "VL2"
TYPOLOGY_LAYER_COUNT = 133

METRE_UNITS = {
    "name": "m",
    "declared_in_file": True,
    "length_unit": "m",
    "area_unit": "m2",
    "space": "model",
}

UNITLESS = {
    "name": "unitless",
    "declared_in_file": False,
    "length_unit": None,
    "area_unit": None,
    "space": "model",
}

INCH_UNITS = {
    "name": "in",
    "declared_in_file": True,
    "length_unit": "in",
    "area_unit": "in2",
    "space": "model",
}

SHEET_UNITS = {
    "name": "sheet coordinates",
    "declared_in_file": False,
    "length_unit": None,
    "area_unit": None,
    "space": "paper",
    "why_no_unit": "DXF carries no per-layout units.",
}

ALL_UNIT_SHAPES = [
    pytest.param(METRE_UNITS, id="metre"),
    pytest.param(INCH_UNITS, id="inch"),
    pytest.param(UNITLESS, id="undeclared"),
    pytest.param(SHEET_UNITS, id="sheet"),
]

NOW = datetime(2026, 8, 23, 9, 0, 0, tzinfo=timezone.utc)

SOURCE = "set by Harsh Ratde, 2026-08-23, sync on Cad integration"
AUTHOR = "user:harsh"


class FakeCollection:
    """Just enough MongoDB for this file, and not one line more.

    Deliberately does NOT interpret aggregation pipelines. A fake that
    interprets a pipeline is a fake that can lie differently from the real
    cluster, and `layer_impact` is written using `count_documents` +
    `distinct` precisely so that its semantics are simple enough to fake
    without that risk.
    """

    def __init__(self, docs=()):
        self.docs = [dict(d) for d in docs]
        self.filters: list[dict] = []

    def _match(self, doc, flt):
        for key, want in flt.items():
            if isinstance(want, dict):
                got = doc.get(key)
                for op, arg in want.items():
                    if op == "$gte" and not (got is not None and got >= arg):
                        return False
                    if op == "$lt" and not (got is not None and got < arg):
                        return False
                    if op not in ("$gte", "$lt"):
                        raise AssertionError(f"operator {op} has not been faked")
            elif doc.get(key) != want:
                return False
        return True

    def _found(self, flt):
        self.filters.append(dict(flt))
        return [d for d in self.docs if self._match(d, flt)]

    def find_one(self, flt, *args, **kwargs):
        rows = self._found(flt)
        return dict(rows[0]) if rows else None

    def find(self, flt, *args, **kwargs):
        return [dict(d) for d in self._found(flt)]

    def count_documents(self, flt):
        return len(self._found(flt))

    def distinct(self, field, flt=None):
        rows = self._found(flt or {})
        return sorted({r.get(field) for r in rows if r.get(field) is not None})

    def replace_one(self, flt, doc, upsert=False):
        rows = self._found(flt)
        if rows:
            for i, existing in enumerate(self.docs):
                if self._match(existing, flt):
                    self.docs[i] = dict(doc)
                    break
        elif upsert:
            self.docs.append(dict(doc))
        return None


@pytest.fixture()
def fake_store(monkeypatch):
    """Replace both MongoDB doors with a fake collection.

    `_tags`/`_entities` are replaced, not `mongo.coll`: `coll()` is the prefix
    gate, and a test that bypasses that gate stops proving that the production
    code goes through it.
    """
    tags = FakeCollection()
    entities = FakeCollection(
        [
            {
                "drawing_id": DRAWING,
                "handle": f"H{i}",
                "layer": TYPOLOGY_LAYER,
                "layout": "Model",
            }
            for i in range(TYPOLOGY_LAYER_COUNT)
        ]
        + [
            {
                "drawing_id": DRAWING,
                "handle": "P1",
                "layer": TYPOLOGY_LAYER,
                "layout": "Layout1",
            },
            {
                "drawing_id": DRAWING,
                "handle": "B1",
                "layer": TYPOLOGY_LAYER,
                "layout": "[block] TITLEBLOCK",
            },
            # Another drawing, a layer with the same name. If it gets counted
            # too, G10 has already leaked at the very bottom layer.
            {
                "drawing_id": OTHER_DRAWING,
                "handle": "X1",
                "layer": TYPOLOGY_LAYER,
                "layout": "Model",
            },
        ]
    )
    monkeypatch.setattr(sr, "_tags", lambda: tags)
    monkeypatch.setattr(sr, "_entities", lambda: entities)
    return {"tags": tags, "entities": entities}


def a_tag(**over):
    """A tag document as `put_layer_tag` writes it, without MongoDB."""
    doc = {
        "_id": sr.tag_id(DRAWING, sr.TARGET_LAYER, TYPOLOGY_LAYER),
        "drawing_id": DRAWING,
        "target_kind": sr.TARGET_LAYER,
        "target": TYPOLOGY_LAYER,
        "business_name": "Villa 12m frontage",
        "use": "residential",
        "source": SOURCE,
        "author": AUTHOR,
        "created_at": NOW,
        "updated_at": NOW,
        "revision": 1,
        "previous": None,
        "tag_contract_version": sr.TAG_CONTRACT_VERSION,
    }
    doc.update(over)
    return doc


def a_config(**over):
    """An UPLIFT-02 land use config entry, as the caller hands it over."""
    entry = {
        "use": "unclassified_residential",
        "business_name": None,
        "config_layer": "global_pattern",
        "config_version": 1,
        "note": "pattern match on layer name",
    }
    entry.update(over)
    return entry


def blocks_of(payload):
    """Every evidence block in a response, whatever shape it is attached in."""
    out = []
    if isinstance(payload.get("evidence"), dict):
        out.append(payload["evidence"])
    out.extend((payload.get("evidence_by_use") or {}).values())
    return out


# --- collections: the MongoDB fence ------------------------------------------


def test_tag_collection_is_inside_the_prefix_fence():
    """`autocad_tags` obeys the same rule as every other collection.

    290 collections in this cluster, 285 of them belonging to other teams. A
    collection named `tags` in there is damage that cannot be undone.
    """
    assert sr.COLL_TAGS == "autocad_tags"
    assert sr.COLL_TAGS.startswith(mongo.PREFIX)


def test_tag_collection_registration_is_actually_requested():
    """Registration in `OWNED_COLLECTIONS` is a handover, not good intentions.

    `mongo.py` is shared property, so this subagent does not edit it itself —
    it writes its request in `docs/PROGRESS.md`. This test is what makes that
    request CHECKABLE rather than trusted. It passes two ways: the request is
    still written down, or the coordinator has already registered it.
    """
    if sr.COLL_TAGS in mongo.OWNED_COLLECTIONS:
        return
    path = pathlib.Path(__file__).resolve().parents[2] / "docs" / "PROGRESS.md"
    if not path.exists():
        # This suite is run through a throwaway container that only mounts
        # `cad_api/`, so `docs/` really is absent there. Said out loud: a
        # silent skip is a gate that has stopped guarding.
        pytest.skip(f"{path} is not mounted; run pytest from the repo root")
    progress = path.read_text(encoding="utf-8")
    assert sr.COLL_TAGS in progress and "OWNED_COLLECTIONS" in progress, (
        f"{sr.COLL_TAGS} is not registered in mongo.OWNED_COLLECTIONS and there "
        "is no request to register it in docs/PROGRESS.md"
    )


def test_module_never_imports_store():
    """`store.py` imports this module below; a reverse import would be circular."""
    source = pathlib.Path(sr.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*from \.store import", source, re.M)
    assert not re.search(r"^\s*from \. import .*\bstore\b(?!_)", source, re.M)


def test_every_query_leads_with_drawing_id(fake_store):
    """This cluster refuses queries without an index, it does not merely slow down.

    Every filter must carry `drawing_id` as its FIRST key — that is the prefix
    of every index that exists, and it is also what G10 asks for, for a
    completely different reason.
    """
    sr.put_layer_tag(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        business_name="Villa 12m frontage",
        use="residential",
        source=SOURCE,
        author=AUTHOR,
        now=NOW,
    )
    sr.get_layer_tag(DRAWING, TYPOLOGY_LAYER)
    sr.list_tags(DRAWING)
    sr.layer_impact(DRAWING, TYPOLOGY_LAYER)

    seen = fake_store["tags"].filters + fake_store["entities"].filters
    assert seen, "not a single query was recorded"
    for flt in seen:
        assert next(iter(flt)) == "drawing_id", f"filter does not lead with drawing_id: {flt}"
        assert flt["drawing_id"] == DRAWING


# --- document key ------------------------------------------------------------


def test_tag_id_round_trips_even_when_the_target_carries_a_colon():
    """A composite key must be readable back without ambiguity.

    A layer name containing `:` is unusual, but "unusual" is exactly the class
    of event G8 refuses to call an edge case: two layers whose keys collide
    would overwrite each other's tag, silently.
    """
    weird = "PLOT:TYPE:A"
    key = sr.tag_id(DRAWING, sr.TARGET_LAYER, weird)
    assert sr.parse_tag_id(key) == (DRAWING, sr.TARGET_LAYER, weird)


def test_tag_id_refuses_a_drawing_id_that_would_break_the_key():
    with pytest.raises(sr.TagRefused) as excinfo:
        sr.tag_id("has:colon", sr.TARGET_LAYER, TYPOLOGY_LAYER)
    assert excinfo.value.code == "TAG_DRAWING_ID_INVALID"


def test_only_layers_can_be_tagged():
    """Per-entity tagging is NOT built, and the refusal has to say so.

    46,677 entities cannot be tagged one by one by a human, and tagging some
    of them produces half-correct data.
    """
    with pytest.raises(sr.TagRefused) as excinfo:
        sr.tag_id(DRAWING, "entity", "5A3F")
    assert excinfo.value.code == "TAG_UNKNOWN_TARGET_KIND"
    assert "layer" in excinfo.value.hint


# --- writing a tag: the source is mandatory ----------------------------------


def test_a_tag_without_a_source_is_refused_with_a_usable_message(fake_store):
    """DoD: `source` is mandatory; saving without one is refused with a clear message.

    A tag without a source would become exactly the problem UPLIFT-08 exists
    to prevent: a guess that in six months cannot be told apart from a fact.
    """
    for bad in ("", "   ", None):
        with pytest.raises(sr.TagRefused) as excinfo:
            sr.put_layer_tag(
                drawing_id=DRAWING,
                layer=TYPOLOGY_LAYER,
                business_name="Villa 12m frontage",
                source=bad,
                author=AUTHOR,
                now=NOW,
            )
        assert excinfo.value.code == "TAG_WITHOUT_SOURCE"
        assert excinfo.value.hint.strip(), "a refusal without a suggestion cannot be acted on"
    assert fake_store["tags"].docs == [], "a tag without a source must not be stored"


@pytest.mark.parametrize(
    "field,code",
    [("business_name", "TAG_WITHOUT_NAME"), ("author", "TAG_WITHOUT_AUTHOR")],
)
def test_the_other_mandatory_fields(fake_store, field, code):
    kwargs = dict(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        business_name="Villa 12m frontage",
        source=SOURCE,
        author=AUTHOR,
        now=NOW,
    )
    kwargs[field] = "  "
    with pytest.raises(sr.TagRefused) as excinfo:
        sr.put_layer_tag(**kwargs)
    assert excinfo.value.code == code
    assert fake_store["tags"].docs == []


def test_size_limits_are_stated_not_assumed(fake_store):
    """G7: limits are stated, and fail with a suggestion, not with a silent truncation."""
    with pytest.raises(sr.TagRefused) as excinfo:
        sr.put_layer_tag(
            drawing_id=DRAWING,
            layer=TYPOLOGY_LAYER,
            business_name="x" * (sr.MAX_BUSINESS_NAME + 1),
            source=SOURCE,
            author=AUTHOR,
            now=NOW,
        )
    assert excinfo.value.code == "TAG_NAME_TOO_LONG"
    assert str(sr.MAX_BUSINESS_NAME) in excinfo.value.message
    assert fake_store["tags"].docs == []


def test_writing_a_tag_stores_it_under_the_composite_key(fake_store):
    doc = sr.put_layer_tag(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        business_name="  Villa 12m frontage  ",
        use="residential",
        source=SOURCE,
        author=AUTHOR,
        now=NOW,
    )
    assert doc["_id"] == f"{DRAWING}:layer:{TYPOLOGY_LAYER}"
    assert doc["business_name"] == "Villa 12m frontage", "the stored value has been tidied"
    assert doc["drawing_id"] == DRAWING
    assert doc["revision"] == 1
    assert doc["previous"] is None
    assert sr.get_layer_tag(DRAWING, TYPOLOGY_LAYER)["business_name"] == "Villa 12m frontage"


def test_a_tag_document_never_carries_a_typed_verified_boolean(fake_store):
    """UPLIFT-08 removed the typable `verified`, and that applies here too.

    A `verified` boolean plus `source` prose IS a mechanism for raising the
    evidence grade in the shape of data. `verified` is now DERIVED from
    `Provenance.config_layer`; what is stored is only the mandatory `source`.
    """
    doc = sr.put_layer_tag(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        business_name="Villa 12m frontage",
        source=SOURCE,
        author=AUTHOR,
        now=NOW,
    )
    assert "verified" not in doc
    assert "supersedes_config" not in doc, (
        "overriding is a property of the reading, not a flag that can be "
        "switched off while the tag stays stored"
    )


def test_overwriting_a_tag_keeps_the_one_it_replaced(fake_store):
    """Never drop anything silently.

    Editing a tag deletes the previous business decision. What remains must be
    enough to answer "who changed it from what".
    """
    sr.put_layer_tag(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        business_name="Villa 12m frontage",
        source=SOURCE,
        author=AUTHOR,
        now=NOW,
    )
    later = datetime(2026, 8, 24, 9, 0, 0, tzinfo=timezone.utc)
    doc = sr.put_layer_tag(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        business_name="Villa 15m frontage",
        source="corrected by Harsh Ratde, 2026-08-24",
        author=AUTHOR,
        now=later,
    )
    assert doc["revision"] == 2
    assert doc["created_at"] == NOW, "the creation time does not change when it is edited"
    assert doc["updated_at"] == later
    assert doc["previous"]["business_name"] == "Villa 12m frontage"
    assert doc["previous"]["author"] == AUTHOR
    assert len(fake_store["tags"].docs) == 1, "one layer, one document"


# --- how many objects are affected -------------------------------------------


def test_one_layer_tag_reaches_every_object_on_that_layer(fake_store):
    """DoD: editing one layer applies to 133 objects without re-ingest.

    The number 133 is the count of layer VL2 on layout Model
    (GROUND-TRUTH-2.md), and it is what appears on screen before anything is
    saved.
    """
    impact = sr.layer_impact(DRAWING, TYPOLOGY_LAYER, layout="Model")
    assert impact["entity_count"] == TYPOLOGY_LAYER_COUNT
    assert impact["by_layout"] == {"Model": TYPOLOGY_LAYER_COUNT}
    assert str(TYPOLOGY_LAYER_COUNT) in impact["impact_note"]
    assert TYPOLOGY_LAYER in impact["impact_note"]


def test_impact_counts_never_cross_into_another_drawing(fake_store):
    """G10: a layer with the same name in another drawing is not an affected object."""
    impact = sr.layer_impact(OTHER_DRAWING, TYPOLOGY_LAYER)
    assert impact["entity_count"] == 1
    assert impact["drawing_id"] == OTHER_DRAWING


def test_impact_says_out_loud_how_much_of_it_is_inside_block_definitions(fake_store):
    """A drawing-wide tag also touches geometry inside block definitions.

    One combined number would make people think 135 parcels are drawn where
    they can see them. The truth: 134 on real layouts, 1 inside a block
    definition that every insertion of it rescales.
    """
    impact = sr.layer_impact(DRAWING, TYPOLOGY_LAYER)
    assert impact["entity_count"] == TYPOLOGY_LAYER_COUNT + 2
    assert impact["in_block_definitions"] == 1
    assert impact["on_real_layouts"] == TYPOLOGY_LAYER_COUNT + 1
    assert "block" in impact["impact_note"].lower()


def test_a_layer_with_nothing_on_it_says_zero_and_why(fake_store):
    """G8: what happens when the data is not there."""
    impact = sr.layer_impact(DRAWING, "LAYER-THAT-DOES-NOT-EXIST")
    assert impact["entity_count"] == 0
    assert impact["by_layout"] == {}
    assert impact["why_empty"], "a zero without a reason cannot be told apart from a failure"


# --- effective value: config overridden by tag -------------------------------


def test_a_tag_overrides_config_and_both_stay_visible():
    """DoD: the tag overrides the config; both stay visible."""
    payload = sr.effective_layer(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        units=METRE_UNITS,
        config_entry=a_config(),
        tag=a_tag(),
    )
    assert payload["business_tag"] == "Villa 12m frontage"
    assert payload["use"] == "residential"
    assert payload["tag_source"] == "user_tag"
    assert payload["tagged_by"] == AUTHOR
    assert payload["overridden"] is True
    assert payload["config_value"]["use"] == "unclassified_residential"
    assert payload["config_value"]["config_layer"] == "global_pattern"


def test_without_a_tag_the_config_value_is_the_effective_one():
    payload = sr.effective_layer(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        units=METRE_UNITS,
        config_entry=a_config(business_name="Residential plot"),
    )
    assert payload["tag_source"] == "config"
    assert payload["business_tag"] == "Residential plot"
    assert payload["overridden"] is False
    assert payload["tagged_by"] is None


def test_no_config_and_no_tag_answers_gracefully_and_guesses_nothing():
    """G3: the absence of a config is not an error and not a guess."""
    payload = sr.effective_layer(
        drawing_id=DRAWING, layer=TYPOLOGY_LAYER, units=METRE_UNITS
    )
    assert payload["business_tag"] is None
    assert payload["tag_source"] is None
    block = payload["evidence_by_use"]["business_name"]
    assert block["grade"] == "unknown"
    assert block["not_established"].strip()
    assert block["how_to_verify"].strip()
    # E8: an `unknown` that names a candidate answer is bait whose negation a
    # paraphrase only has to drop.
    assert "villa" not in block["statement"].lower()
    assert "residential" not in block["statement"].lower()


# --- evidence grade: a business name is never stated by the file -------------


def test_a_business_tag_is_never_stated_by_the_file():
    """This is the decision that shapes the whole of this spec.

    43,109 texts examined, zero of them naming the meaning of a typology code.
    The file states nothing about business names, so the grade must not be
    `stated` — however sure the person who typed it was.
    """
    payload = sr.effective_layer(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        units=METRE_UNITS,
        tag=a_tag(),
    )
    block = payload["evidence_by_use"]["business_name"]
    assert block["grade"] == "inferred"
    assert block["independent_corpora"] == []


def test_a_human_tag_cannot_manufacture_a_stated_grade_by_echoing_the_layer_name():
    """The easiest path of abuse, closed and tested.

    If a tag claim carried `tokens`, somebody would only have to type a
    business name containing a word from its own layer name — and the
    LAYER_NAME observation would "state" the claim they had just invented.
    That is not evidence; that is a circle. That is why a business tag claim
    NEVER carries tokens.
    """
    payload = sr.effective_layer(
        drawing_id=DRAWING,
        layer="VILLA-PLOT-OUTLINE",
        units=METRE_UNITS,
        tag=a_tag(target="VILLA-PLOT-OUTLINE", business_name="Villa compound"),
    )
    block = payload["evidence_by_use"]["business_name"]
    assert block["claim"]["tokens"] == []
    assert block["grade"] == "inferred"


def test_verified_and_grade_are_two_different_axes():
    """Human-verified, but not stated by the file — and that is consistent.

    `grade` answers what the FILE says; `verified` answers whether a human has
    confirmed the mapping. A business tag is the only place in this stack
    where the two are certain to differ, and that is the most honest
    description of what a tag is.
    """
    payload = sr.effective_layer(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        units=METRE_UNITS,
        tag=a_tag(),
    )
    block = payload["evidence_by_use"]["business_name"]
    assert block["verified"] is True
    assert block["provenance"]["config_layer"] == "drawing_override"
    assert block["grade"] == "inferred"


def test_the_source_prose_travels_as_the_reference_of_a_human_observation():
    """A source outside the file must name its person, so that it can be traced."""
    payload = sr.effective_layer(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        units=METRE_UNITS,
        tag=a_tag(),
    )
    block = payload["evidence_by_use"]["business_name"]
    human = [s for s in block["sources"] if s["origin"] == "human"]
    assert len(human) == 1
    assert human[0]["reference"] == SOURCE
    assert human[0]["corpus"] == "outside_file"
    assert human[0]["speaks"] is False


def test_tag_observation_is_the_only_sanctioned_way_to_fold_a_tag_in():
    """The seam for other modules: they receive an Observation, not a free string.

    If `store_landuse` assembled its own observation from `tag["source"]`,
    sooner or later somebody would write it as `Origin.LAYER_DESCRIPTION` and
    a sentence typed by a person would rise into a statement by the file.
    """
    obs = sr.tag_observation(a_tag())
    assert obs.origin is ev.Origin.HUMAN
    assert obs.corpus is ev.Corpus.OUTSIDE_FILE
    assert obs.corpus in ev.NEVER_SPEAKS
    assert obs.reference == SOURCE


def test_a_tag_from_another_drawing_is_refused_not_merged():
    """G10: a layer name that means school here does not necessarily mean it there.

    A source outside the file is the easiest path for smuggling knowledge
    between drawings, so its refusal is tested directly.
    """
    with pytest.raises(ev.EvidenceError) as excinfo:
        sr.effective_layer(
            drawing_id=DRAWING,
            layer=TYPOLOGY_LAYER,
            units=METRE_UNITS,
            tag=a_tag(drawing_id=OTHER_DRAWING),
        )
    assert excinfo.value.code == "CROSS_DRAWING"


def test_a_tag_for_another_layer_is_refused_too():
    """A tag on the wrong target is a wrong answer, not a small mismatch.

    Its code is deliberately NOT `CROSS_DRAWING`: `ERROR_CODES` in
    `evidence.py` is a closed list, and using the wrong name for a different
    cause makes those two causes impossible to tell apart when read from the
    log.
    """
    with pytest.raises(sr.TagRefused) as excinfo:
        sr.effective_layer(
            drawing_id=DRAWING,
            layer="ROADS",
            units=METRE_UNITS,
            tag=a_tag(),
        )
    assert excinfo.value.code == "TAG_TARGET_MISMATCH"


# --- invariants over different shapes of drawing (G5) ------------------------


@pytest.mark.parametrize("units", ALL_UNIT_SHAPES)
@pytest.mark.parametrize("with_tag", [True, False], ids=["tagged", "untagged"])
def test_the_response_always_passes_the_evidence_audit(units, with_tag):
    payload = sr.effective_layer(
        drawing_id=DRAWING,
        layer=TYPOLOGY_LAYER,
        units=units,
        tag=a_tag() if with_tag else None,
    )
    assert ev.audit_response(payload) == []


@pytest.mark.parametrize("units", ALL_UNIT_SHAPES)
def test_the_response_never_invents_a_unit(units):
    """G2: units are handed over by the caller as they are, never filled in here."""
    payload = sr.effective_layer(
        drawing_id=DRAWING, layer=TYPOLOGY_LAYER, units=units, tag=a_tag()
    )
    for block in blocks_of(payload):
        assert block["scope"]["units"] == dict(units)


@pytest.mark.parametrize("units", ALL_UNIT_SHAPES)
def test_the_grade_never_rises_above_inferred_whatever_the_drawing(units):
    for tag, config in (
        (a_tag(), None),
        (a_tag(), a_config()),
        (None, a_config(business_name="Residential plot")),
        (None, a_config(config_layer="drawing_override")),
    ):
        payload = sr.effective_layer(
            drawing_id=DRAWING,
            layer=TYPOLOGY_LAYER,
            units=units,
            tag=tag,
            config_entry=config,
        )
        for name, block in payload["evidence_by_use"].items():
            assert block["grade"] in ("unknown", "inferred"), name


def test_every_response_carries_a_scope_note():
    payload = sr.effective_layer(
        drawing_id=DRAWING, layer=TYPOLOGY_LAYER, units=METRE_UNITS, tag=a_tag()
    )
    assert payload["scope_note"].strip()
    assert TYPOLOGY_LAYER in payload["scope_note"]


def test_no_drawing_specific_constant_lives_in_the_module():
    """G1, tested rather than only grepped.

    The tagging code works over any layer name; a typology code appearing in
    `.py` means this module has just become a solution specific to one
    drawing.
    """
    source = pathlib.Path(sr.__file__).read_text(encoding="utf-8")
    for token in ("VL2", "DP4", "LP1", "TH3", "Primary School", "32638"):
        assert token not in source, token


# --- export to the UPLIFT-02 config ------------------------------------------


def test_export_produces_config_shaped_yaml_sorted_and_deterministic():
    """DoD: the export produces YAML usable directly as a config.

    The flow: tag from the screen → export → review → commit as config.
    Business decisions finally land in the repo instead of staying in a single
    database.
    """
    rows = [
        a_tag(target="ZZ", business_name="Retail strip", use="commercial"),
        a_tag(),
    ]
    text = sr.export_yaml(DRAWING, rows)
    assert text == sr.export_yaml(DRAWING, list(reversed(rows))), (
        "the output must be deterministic, or every export becomes a false diff"
    )
    assert text.index(f'"{TYPOLOGY_LAYER}"') < text.index('"ZZ"'), "sorted"
    assert "layers:" in text
    assert f'drawing_id: "{DRAWING}"' in text


def test_exported_yaml_carries_the_source_and_omits_a_typed_verified():
    text = sr.export_yaml(DRAWING, [a_tag()])
    assert SOURCE in text
    assert AUTHOR in text
    assert not re.search(r"^\s*verified:", text, re.M), (
        "a boolean that can be written by hand is an evidence grade rise in "
        "YAML form"
    )


def test_exported_yaml_survives_a_business_name_that_would_break_it():
    """A business name is text typed by a human; it will contain anything."""
    yaml = pytest.importorskip("yaml")
    nasty = 'Villa: "12m" #1 \\ frontage\nline two'
    text = sr.export_yaml(DRAWING, [a_tag(business_name=nasty)])
    loaded = yaml.safe_load(text)
    assert loaded["layers"][TYPOLOGY_LAYER]["business_name"] == nasty
    assert loaded["drawing_id"] == DRAWING


def test_exported_yaml_round_trips_arabic_and_a_layer_name_that_needs_quoting():
    yaml = pytest.importorskip("yaml")
    text = sr.export_yaml(
        DRAWING,
        [a_tag(target="0: default", business_name="مسجد الحي")],
    )
    loaded = yaml.safe_load(text)
    assert loaded["layers"]["0: default"]["business_name"] == "مسجد الحي"


def test_exporting_nothing_says_so_rather_than_emitting_an_empty_mapping():
    """G8: `layers: {}` reads as "there are no layers", not "not tagged yet"."""
    text = sr.export_yaml(DRAWING, [])
    assert "layers: {}" in text
    assert "no tag" in text.lower()


def test_export_refuses_rows_from_another_drawing():
    with pytest.raises(ev.EvidenceError) as excinfo:
        sr.export_yaml(DRAWING, [a_tag(), a_tag(drawing_id=OTHER_DRAWING, target="ZZ")])
    assert excinfo.value.code == "CROSS_DRAWING"


# =============================================================================
# UPLIFT-09 — the analysis recipe registry
# =============================================================================
#
# Written from the UPLIFT-09 Definition of Done. Two kinds of test, both
# mandatory (G5):
#
# - **Number tests** — the exact Janadriyah values for all eight recipes,
#   measured from the store on 24 August 2026. These numbers are also what
#   makes the recipes trustworthy: a recipe without a single ground-truth
#   number only proves that it runs.
# - **Invariant tests** — run over EVERY drawing that exists, not only
#   Janadriyah, and over various shapes of units. That is what catches a break
#   on the 17th drawing; the Janadriyah numbers will never catch it.
#
# Most of this file runs **without MongoDB**: the only things that touch the
# database are the eight recipe functions, and the registry engine itself is
# pure.

import json  # noqa: E402

from app import recipes  # noqa: E402
from app.recipes import library as rl  # noqa: E402
from app.recipes import registry as rr  # noqa: E402

#: Every registered recipe, pinned here so that one lost to a broken import
#: shows up as a red test rather than as a catalogue that quietly shrinks.
#:
#: The list is sorted, because `known()` sorts. It grew past the original
#: eight when DOSSIER added the topology and truth recipes, and the name kept
#: its history: this tuple exists precisely so that growth is a deliberate
#: edit rather than something that happens to a catalogue while nobody looks.
#: The roster, in the order `known()` returns it. The name is historical —
#: there were eight once — and the list is the point: a recipe that appears
#: without anyone editing this tuple is a recipe nobody reviewed, and one that
#: disappears is a capability lost in silence. Adding a line here is the
#: deliberate act; GEO-H3 G4 added `h3_vicinity`.
EIGHT_RECIPES = (
    "adjacency",
    "catchment_coverage",
    "combine_findings",
    "containment",
    "coverage_gap",
    "cross_check_classification",
    "dimension_truth",
    "drafting_hygiene",
    "frontage_check",
    "h3_vicinity",
    "junction_census",
    "label_coverage",
    "outliers",
    "overlap_scan",
    "parcel_inventory",
    "revision_diff",
    "road_hierarchy",
    "size_module",
)

#: The reference layout. "2,380 parcels" means nothing without the layout that
#: produced it: modelspace and paper sheets do not share a coordinate frame.
LAYOUT = "Model"

#: The Janadriyah plot-number layer. It lives in the TEST, never in production
#: `.py` — G1. A layer name is one drawing's trait.
NUMBER_LAYER = "C-PROP-PlotNumber"

#: The Janadriyah block boundary layer, used as the OUTER polygon in
#: `containment`.
BLOCK_LAYER = "BlockBoundary"

#: Measured from the store on 24 August 2026, after UPLIFT-01, UPLIFT-02,
#: UPLIFT-03 and UPLIFT-07 landed.
GT_NUMBERS = 2449            # text-bearing entities on the plot-number layer, layout Model
GT_PARCELS_RESIDENTIAL = 2380
GT_GAP = 69                  # GROUND-TRUTH-2 Q2.6, its cause has never been established
GT_RESIDENTIAL_AREA = 712977.389756
GT_EDUCATION_PARCELS = 9
GT_EDUCATION_AREA = 41110.456509
GT_MODULE_12x25 = 652
GT_MODULE_10x25 = 417
GT_BLOCK_BOUNDARIES = 197
GT_CODED_TOTAL = 2380
GT_CODED_PASSING = 2312
GT_CODED_FAILING = 68
#: Measured 77, while the UPLIFT-07 Definition of Done writes 80. See
#: `test_cross_check_classification_reproduces_the_sweep` and
#: docs/PROGRESS-R.md.
GT_OUTSIDE_MEASURED = 77
GT_OUTSIDE_BY_LAYER = {"0": 52, "Open Spaces": 13, "Linear Park": 11, "Pocket Park": 1}

#: The UPLIFT-07 shape test ranges. In the test, because they are this
#: drawing's traits (G1).
EDGE_RANGE = (24.0, 26.0)
AREA_RANGE = (150.0, 650.0)


def _live_or_skip():
    """The reference drawing from the store, or skip.

    Its helper is rewritten here instead of imported from
    `test_duplicates.py`: a shared helper living in a file owned by another
    subagent is a dependency that is invisible until that file is moved.
    """
    from app import store as live_store

    try:
        drawing = live_store.get_drawing(DRAWING)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip(
            f"reference drawing {DRAWING} has not been ingested; the UPLIFT-09 "
            "acceptance numbers cannot be checked"
        )
    return live_store, drawing


def _config_layers(use=None, role="parcel"):
    """Layers from the reviewed land use config, not from the data under test.

    This is what separates a sweep that checks something from a sweep that
    checks itself: if the layer list were derived from the geometry being
    checked, whatever number came out would match.
    """
    from app import landuse as live_landuse

    config = live_landuse.for_drawing(DRAWING)
    if config is None:
        pytest.skip("the reference drawing has no land use config yet")
    names = sorted(
        name
        for name, entry in config.layers.items()
        if entry.role == role and (use is None or entry.use == use)
    )
    if not names:
        pytest.skip(f"the config names no layer with role={role} use={use}")
    return names


def _run(name, params, *, layout=LAYOUT):
    live_store, drawing = _live_or_skip()
    return recipes.run(
        DRAWING, name, params, layout=layout,
        units=live_store._unit_names(drawing, layout),
    )


# --- the engine: the catalogue and the honest limits -------------------------


def test_the_catalogue_lists_all_eight_recipes():
    """DoD: eight recipes implemented, and it is the catalogue that says so."""
    assert recipes.known() == EIGHT_RECIPES
    cat = recipes.catalog()
    assert cat["count"] == len(EIGHT_RECIPES)
    assert [row["recipe"] for row in cat["recipes"]] == list(EIGHT_RECIPES)


def test_every_recipe_says_when_to_use_it_and_what_it_stands_on():
    """The catalogue is the real interface: the model reads it, not the code."""
    for row in recipes.catalog()["recipes"]:
        assert row["answers"].strip(), row["recipe"]
        assert row["when_to_use"].strip(), row["recipe"]
        assert row["returns"], row["recipe"]
        # `built_on` names the store functions that are really called, so that
        # a recipe quietly rewriting geometry is visible from its own
        # catalogue entry.
        assert row["built_on"], row["recipe"]
        # Limits are stated, not assumed (G7).
        assert row["limits"], row["recipe"]
        for param in row["params"]:
            assert param["about"].strip(), (row["recipe"], param["name"])


def test_an_unknown_recipe_returns_the_catalogue_not_a_bare_error():
    """DoD: an unknown recipe name returns the catalogue.

    An agent that receives "unknown recipe" will force the wrong tool and
    produce the wrong number in the right tone. One that receives a catalogue
    can say what does not exist yet.
    """
    out = recipes.run(DRAWING, "berapa_rumah", {}, layout=LAYOUT, units=METRE_UNITS)
    assert out["error"] == "RECIPE_UNKNOWN"
    assert out["count"] == len(EIGHT_RECIPES)
    assert [r["recipe"] for r in out["recipes"]] == list(EIGHT_RECIPES)
    assert out["no_recipe_sentence"] == rr.NO_RECIPE_SENTENCE


def test_the_no_recipe_sentence_travels_with_the_catalogue_verbatim():
    """DoD: the agent instructions carry the "not available yet" sentence verbatim.

    It lives in the code AND travels in every catalogue, not only in the
    prompt: a sentence that lives only in a prompt is lost at the first
    paraphrase, and what is lost is precisely its negative half.
    """
    sentence = recipes.catalog()["no_recipe_sentence"]
    filled = sentence.format(what="walking distance", nearest="straight-line distance")
    assert "not available yet" in filled
    assert "can be added as an analysis recipe" in filled
    assert "{" not in filled and "}" not in filled


def test_the_honest_limits_are_part_of_the_catalogue_not_only_the_document():
    """An agent asked "what can you do" answers from the SAME list a human
    reads."""
    limits = recipes.catalog()["honest_limits"]
    assert any("write a new computation on the spot" in row["cannot"] for row in limits)
    for row in limits:
        assert row["because"].strip() and row["fixable"].strip()


# --- the engine: no code execution path --------------------------------------


def test_the_registry_has_no_code_execution_path():
    """The most important boundary in this spec, tested and not promised.

    A drawing's contents come from contractors and consultants; text inside a
    DWG enters the agent's context. A tool that runs code gives that text an
    execution path, and `cad-api` holds the database credentials other teams
    use.
    """
    package = pathlib.Path(recipes.__file__).parent
    sources = sorted(package.glob("*.py"))
    assert len(sources) >= 3, "the recipe package could not be read"
    # `re.compile` is a regex, not a code path, and `truth.py` needs several to
    # read what a dimension label prints. It is stripped BEFORE the scan rather
    # than dropped from the list, so a bare `compile(` -- the builtin, which
    # does turn text into code -- still fails. The rule being defended is that
    # a contractor's drawing text can never become something this process runs;
    # narrowing the check to keep that true is not the same as relaxing it.
    for path in sources:
        text = path.read_text(encoding="utf-8").replace("re.compile(", "")
        for forbidden in ("eval(", "exec(", "compile(", "__import__(", "subprocess"):
            assert forbidden not in text, f"{path.name} contains {forbidden}"


def test_a_recipe_cannot_be_added_at_request_time():
    """A new recipe = a merge request. That is the only thing separating this
    registry from free code execution."""
    import inspect

    assert not hasattr(recipes, "unregister")
    assert not hasattr(rr, "unregister")
    # `run` accepts a name and values. No parameter accepts logic.
    names = set(inspect.signature(recipes.run).parameters)
    assert names == {"drawing_id", "recipe", "params", "layout", "units"}
    for row in recipes.catalog()["recipes"]:
        for param in row["params"]:
            assert param["kind"] in rr.PARAM_KINDS, param


def test_no_recipe_reads_the_clock_or_rolls_a_die():
    """Determinism is tested as a property of the file, not as the author's intent.

    Same input → same output, forever. Two sessions can only compare numbers
    if the numbers do not move by themselves.
    """
    package = pathlib.Path(recipes.__file__).parent
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("datetime.now", "time.time", "random.", "requests.", "urlopen"):
            assert forbidden not in text, f"{path.name} contains {forbidden}"


def test_every_query_in_the_recipes_leads_with_drawing_id():
    """This cluster runs with `notablescan`: a query without an indexed plan
    does not slow down, it FAILS on the request path. G10 asks for the same
    thing for a completely different reason."""
    source = pathlib.Path(rl.__file__).read_text(encoding="utf-8")
    assert '"drawing_id": drawing_id' in source
    # Not a single filter opens with `layout` as its first key: every query
    # like that would be refused by the cluster.
    assert '{\n        "layout"' not in source
    assert '{"layout"' not in source


# --- the engine: parameters --------------------------------------------------


def test_an_unknown_parameter_is_refused_not_ignored():
    """A misspelled parameter that is silently dropped produces an answer over
    the default scope, and there is nothing in the response that says this was
    not what was asked for."""
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        recipes.run(
            DRAWING, "coverage_gap",
            {"number_layer": "X", "parcel_layer": ["Y"]},
            layout=LAYOUT, units=METRE_UNITS,
        )
    assert excinfo.value.code == "RECIPE_PARAM_UNKNOWN"
    assert "parcel_layers" in excinfo.value.hint


def test_a_required_parameter_has_no_default_and_says_why():
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        recipes.run(DRAWING, "coverage_gap", {}, layout=LAYOUT, units=METRE_UNITS)
    assert excinfo.value.code == "RECIPE_PARAM_REQUIRED"
    assert "G1" in excinfo.value.hint


def test_layout_is_required_because_layouts_do_not_share_a_frame():
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        recipes.run(
            DRAWING, "coverage_gap", {"number_layer": "X"},
            layout=None, units=METRE_UNITS,
        )
    assert excinfo.value.code == "RECIPE_LAYOUT_REQUIRED"


@pytest.mark.parametrize(
    "kind,value,expected",
    [
        ("layers", "A, B , A", ("A", "B")),
        ("layers", ["A", "B"], ("A", "B")),
        ("layer", " A ", "A"),
        ("number", "2.5", 2.5),
        ("number", 3, 3.0),
        ("range", [1, 2], (1.0, 2.0)),
        ("range", {"min": 1, "max": 2}, (1.0, 2.0)),
        ("flag", True, True),
        ("text", " x ", "x"),
    ],
)
def test_parameters_are_cleaned_deterministically(kind, value, expected):
    assert rr.coerce(rr.Param("p", kind, "about"), value) == expected


@pytest.mark.parametrize(
    "kind,value",
    [
        ("layers", 5),
        ("layers", []),
        ("layer", ""),
        ("number", "abc"),
        ("number", True),
        ("range", [2, 1]),
        ("range", "1,2"),
        ("flag", "yes"),
    ],
)
def test_a_parameter_is_never_coerced_into_shape_by_guessing(kind, value):
    """A forced value produces an answer to a question different from the one
    the person typed."""
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        rr.coerce(rr.Param("p", kind, "about"), value)
    assert excinfo.value.code in ("RECIPE_PARAM_INVALID", "RECIPE_PARAM_TOO_MANY")


def test_size_limits_are_stated_and_actually_bind():
    """G7. A limit that does not bind is a limit that does not exist yet."""
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        rr.coerce(
            rr.Param("layers", "layers", "about"),
            [f"L{i}" for i in range(rr.MAX_LAYERS_PER_PARAM + 1)],
        )
    assert excinfo.value.code == "RECIPE_PARAM_TOO_MANY"

    with pytest.raises(recipes.RecipeRefused) as excinfo:
        rr.refuse_oversize(what="polygons", size=10, limit=5, hint="narrow it")
    assert excinfo.value.code == "RECIPE_INPUT_TOO_LARGE"
    assert excinfo.value.hint == "narrow it"


def test_every_refusal_code_is_registered():
    """The reason is the same as `evidence.ERROR_CODES`: a code that simply
    appears cannot be looked up in the documentation."""
    with pytest.raises(AssertionError):
        recipes.RecipeRefused("BUKAN_KODE", "message", "hint")


# --- the engine: the envelope and the evidence contract ----------------------


def _fake_recipe(body, **over):
    spec = dict(
        name="_fake",
        answers="a",
        when_to_use="b",
        params=(),
        returns=("x",),
        built_on=("none",),
        limits={"x": 1},
        run=lambda **kw: dict(body),
    )
    spec.update(over)
    return rr.Recipe(**spec)


def test_the_envelope_refuses_a_meaningful_answer_without_evidence():
    """UPLIFT-08 binds a response that says what something IS."""
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        rr.envelope(
            _fake_recipe({}), drawing_id=DRAWING, layout=LAYOUT,
            params_used={}, body={"scope_note": "present"},
        )
    assert excinfo.value.code == "RECIPE_WITHOUT_EVIDENCE"


def test_the_envelope_refuses_a_measurement_that_never_says_what_it_did_not_measure():
    """A straight-line distance that does not say it is a straight line will be
    read as a walking distance."""
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        rr.envelope(
            _fake_recipe({}, carries_meaning=False), drawing_id=DRAWING,
            layout=LAYOUT, params_used={}, body={"scope_note": "present"},
        )
    assert excinfo.value.code == "RECIPE_WITHOUT_EVIDENCE"


def test_the_envelope_refuses_an_answer_without_a_scope_note():
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        rr.envelope(
            _fake_recipe({}), drawing_id=DRAWING, layout=LAYOUT,
            params_used={}, body={"not_measured": "x"},
        )
    assert excinfo.value.code == "RECIPE_WITHOUT_SCOPE"


def test_the_envelope_publishes_the_parameters_that_were_actually_used():
    """Determinism that cannot be checked is not determinism."""
    out = rr.envelope(
        _fake_recipe({}, carries_meaning=False),
        drawing_id=DRAWING, layout=LAYOUT,
        params_used={"layers": ("A", "B"), "z": 2.0},
        body={"scope_note": "present", "not_measured": "x"},
    )
    assert out["params_used"] == {"layers": ["A", "B"], "z": 2.0}
    assert out["deterministic"] is True
    assert out["built_on"] == ["none"]


def test_no_drawing_specific_constant_lives_in_the_recipe_package():
    """G1, tested rather than only grepped. All eight recipes work over any
    layer name; a typology code in `.py` means this package has just become a
    solution specific to one drawing."""
    package = pathlib.Path(recipes.__file__).parent
    for path in sorted(package.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for token in ("VL2", "DP4", "LP1", "TH3", "Primary School", "32638",
                      NUMBER_LAYER, BLOCK_LAYER):
            assert token not in source, f"{path.name}: {token}"


# --- Janadriyah numbers: coverage_gap ----------------------------------------


def test_coverage_gap_reproduces_the_sixty_nine():
    """DoD: `coverage_gap` reproduces the difference of 69 and reports it as an
    OBSERVATION, not as an explanation.

    The question that has hung in `GROUND-TRUTH-2.md` Q2.6 since 23 August and
    has never been settled. And it is not an academic question: asked how many
    houses there were, the agent once answered 2,449 — the count of TEXT
    entities on the plot-number layer — while the number of parcels is 2,380.
    That difference, exactly 69, is the most expensive mistake in a technical
    document: the wrong number in the right tone.
    """
    r = _run("coverage_gap", {"number_layer": NUMBER_LAYER, "parcel_use": "residential"})
    assert r["numbers_total"] == GT_NUMBERS
    assert r["parcels_total"] == GT_PARCELS_RESIDENTIAL
    assert r["gap"] == GT_GAP
    assert r["answer_to_how_many"] == GT_PARCELS_RESIDENTIAL

    # And what makes it more than a subtraction: every residential parcel
    # contains EXACTLY one number inside its boundary, none is shared, and no
    # parcel is empty. So the 69 numbers left over cannot possibly sit on any
    # residential parcel — they fall outside them.
    assert r["join"]["targets_tested"] == GT_PARCELS_RESIDENTIAL
    assert r["join"]["targets_with_number"] == GT_PARCELS_RESIDENTIAL
    assert r["join"]["targets_without_number"] == 0
    assert r["join"]["numbers_matched_more_than_one_parcel"] == 0
    assert r["numbers_without_parcel"] == GT_GAP
    assert r["parcels_without_number"] == 0
    assert r["sanity"]["targets_tested_vs_parcels_total"] == 0


def test_coverage_gap_says_which_of_its_two_counts_answers_how_many_houses():
    """The sentence that has never been said to anyone, and the cause of the
    whole 23 August mistake: the two counts count DIFFERENT things."""
    r = _run("coverage_gap", {"number_layer": NUMBER_LAYER, "parcel_use": "residential"})
    what = r["what_each_number_counts"]
    assert "LABEL" in what["numbers_total"]
    assert "not a parcel count" in what["numbers_total"]
    assert "role=parcel" in what["parcels_total"]
    assert r["answer_to_how_many"] == r["parcels_total"]


def test_coverage_gap_reports_the_shortfall_as_an_observation_not_a_cause():
    """A difference presented as a defect finding before the plot list from the
    drawing's owner confirms it is an accusation, not a measurement."""
    r = _run("coverage_gap", {"number_layer": NUMBER_LAYER, "parcel_use": "residential"})
    block = r["evidence"]
    # There is no word in any file that STATES a coverage gap, so this claim
    # can never rise above `inferred`.
    assert block["grade"] == "inferred"
    assert block["claim"]["tokens"] == []
    assert "not established" in block["not_established"]
    assert block["how_to_verify"].strip()
    assert "an observation, not an explanation" in r["is_an_observation_not_an_explanation"]
    assert ev.audit_response(r) == []


def test_widening_the_parcel_set_moves_the_gap_and_says_so():
    """The way to verify it, written in its own response, actually run.

    With EVERY layer with role parcel — schools, mosques, parks, utilities —
    the number of polygons exceeds the number of numbers, and the direction of
    the difference reverses. That is what makes "69" something that must not
    be read as one tidy cause.
    """
    narrow = _run("coverage_gap", {"number_layer": NUMBER_LAYER, "parcel_use": "residential"})
    wide = _run("coverage_gap", {"number_layer": NUMBER_LAYER})
    assert wide["parcels_total"] > narrow["parcels_total"]
    assert wide["gap"] < narrow["gap"]
    assert wide["gap_direction"] == "more polygons than numbers"
    # Some numbers fall inside more than one polygon once the facility parcels
    # take part — overlapping polygons. The number of placements is therefore
    # not the number of numbers, and the figure is WITHHELD rather than
    # guessed.
    if wide["join"]["numbers_matched_more_than_one_parcel"]:
        assert wide["numbers_without_parcel"] is None
        assert "WITHHELD" in wide["numbers_without_parcel_note"]


# --- Janadriyah numbers: the other seven recipes -----------------------------


def test_parcel_inventory_counts_the_two_thousand_three_hundred_and_eighty():
    """The number once answered with "there may be no houses here"."""
    r = _run("parcel_inventory", {})
    by_use = {row["use"]: row for row in r["uses"]}
    assert by_use["residential"]["parcels"] == GT_PARCELS_RESIDENTIAL
    assert by_use["residential"]["total_area"]["value"] == pytest.approx(
        GT_RESIDENTIAL_AREA, abs=1e-4
    )
    assert by_use["education"]["parcels"] == GT_EDUCATION_PARCELS
    assert by_use["education"]["total_area"]["value"] == pytest.approx(
        GT_EDUCATION_AREA, abs=1e-4
    )
    assert ev.audit_response(r) == []


def test_parcel_inventory_size_band_filters_without_recounting():
    """The band FILTERS; it does not rewrite the total. A band that changes its
    parent number makes two questions share one answer."""
    r = _run("parcel_inventory", {"min_area": 200.0, "max_area": 400.0})
    band = {row["use"]: row for row in r["size_band"]["by_use"]}
    row = band["residential"]
    assert row["parcels"] == GT_PARCELS_RESIDENTIAL
    assert row["parcels_in_band"] + row["parcels_outside_band"] == row["parcels_measured"]
    assert 0 < row["parcels_in_band"] < GT_PARCELS_RESIDENTIAL
    # A parcel with no measured area enters neither side; it is counted on its own.
    assert row["parcels_not_measured"] == row["parcels"] - row["parcels_measured"]


def test_label_coverage_finds_a_number_inside_every_residential_parcel():
    """2,380 of 2,380, and not one label falls in two parcels."""
    r = _run(
        "label_coverage",
        {"target_layers": _config_layers(use="residential"), "label_layer": NUMBER_LAYER},
    )
    assert r["total_targets"] == GT_PARCELS_RESIDENTIAL
    assert r["targets_with_label"] == GT_PARCELS_RESIDENTIAL
    assert r["targets_without_label"] == 0
    assert r["coverage_fraction"] == 1.0
    assert r["labels_matched_more_than_one_target"] == 0
    # `boundary_cases` is present even at zero: a plot number the drafter
    # snapped to its own plot line is not an edge case, it is how the drawing
    # was made.
    assert r["boundary_cases"] == 0
    assert ev.audit_response(r) == []


def test_size_module_reads_the_plot_modules_without_one_layer_name():
    """The UPLIFT-07 DoD turned into a recipe: plot shapes are found from their
    own geometry. That is what makes it survive into the 17th drawing."""
    r = _run("size_module", {"top": 8})
    modules = {}
    for group in r["groups"]:
        lens = sorted(round(v) for v in group["modal_edge_lengths"])
        if len(lens) == 4 and lens[2] == lens[3] == 25 and lens[0] == lens[1]:
            modules.setdefault(lens[0], group["count"])
    assert modules.get(12) == GT_MODULE_12x25
    assert modules.get(10) == GT_MODULE_10x25
    assert modules[12] > modules[10]
    assert r["no_tolerance_note"].strip()


def test_adjacency_measures_edge_to_edge_and_says_it_is_not_a_walk():
    """Nine education parcels fall into TWO clusters at a threshold of 500
    drawing units — the same number UPLIFT-05 measured."""
    r = _run("adjacency", {"layers": _config_layers(use="education"), "gap_max": 500.0})
    assert r["clusters"]["count"] == 2
    assert sum(len(m) for m in r["clusters"]["members"]) == GT_EDUCATION_PARCELS
    # `not_measured` here is store_spatial's sentence, not this recipe's own.
    assert "walking distance" in r["not_measured"].lower()
    assert r["limits"]["items"] == rl.MAX_ADJACENCY_ITEMS


def test_adjacency_refuses_a_population_it_cannot_answer_for_and_names_the_tool_that_can():
    """G7. The limit is stated AND binds, and its refusal names the way out."""
    with pytest.raises(Exception) as excinfo:
        _run("adjacency", {"layers": _config_layers(use="residential"), "gap_max": 1.0})
    exc = excinfo.value
    assert getattr(exc, "code", "") in ("TOO_MANY_ITEMS", "RECIPE_INPUT_TOO_LARGE")
    assert "proximity_count" in getattr(exc, "hint", "")


def test_containment_puts_every_residential_parcel_in_exactly_one_block():
    """197 block boundaries, 2,380 residential parcels, and not one of them
    falls in two blocks at once. That is also what makes the numbers
    trustworthy: a parcel counted in two blocks would make every per-block
    count overstate."""
    r = _run(
        "containment",
        {"outer_layer": BLOCK_LAYER, "inner_layers": _config_layers(use="residential")},
    )
    assert r["outer_total"] == GT_BLOCK_BOUNDARIES
    assert r["outer_with_inner"] == GT_BLOCK_BOUNDARIES
    assert r["outer_without_inner"] == 0
    assert r["inner_total"] == GT_PARCELS_RESIDENTIAL
    assert r["inner_placed"] == GT_PARCELS_RESIDENTIAL
    assert r["inner_matched_more_than_one_outer"] == 0
    assert r["inner_without_outer"] == 0
    assert ev.audit_response(r) == []


def test_containment_never_claims_to_be_a_full_containment_test():
    """Its test point is `polygon_centroid`, and an inner polygon that sticks
    out still counts as 'inside'. That is admitted in its response, not hidden
    in the documentation."""
    r = _run(
        "containment",
        {"outer_layer": BLOCK_LAYER, "inner_layers": _config_layers(use="residential")},
    )
    assert "polygon_centroid" in r["point_used"]
    assert "not a full containment test" in r["evidence"]["not_established"]
    # Coordinates imply, they never state.
    assert r["evidence"]["grade"] == "inferred"


def test_outliers_finds_the_one_education_parcel_that_is_nothing_like_the_rest():
    """Eight education parcels range 2,749–5,821; one measures 16,029. At z = 1
    only that one leaves the band, and that is correct."""
    r = _run("outliers", {"layers": _config_layers(use="education"), "field": "area", "z": 1.0})
    assert r["measured"] == GT_EDUCATION_PARCELS
    assert len(r["outliers"]) == 1
    row = r["outliers"][0]
    assert row["value"] > 16000
    assert row["direction"] == "large"
    assert row["z_score"] > 2.5
    assert row["unit"] == "m2"
    # "Deviating" is a POSITION within the distribution, not a defect.
    assert "not a defect" in r["not_measured"]


def test_outliers_refuses_a_threshold_that_would_flag_everything():
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        recipes.run(
            DRAWING, "outliers", {"layers": ["A"], "z": 0},
            layout=LAYOUT, units=METRE_UNITS,
        )
    assert excinfo.value.code == "RECIPE_PARAM_INVALID"


def test_cross_check_classification_reproduces_the_sweep():
    """DoD: 2,312 pass / 68 fail the shape test.

    The list of coded layers comes from the UPLIFT-02 land use config — the
    one reviewed through a merge request — not derived from the geometry being
    checked. If it were derived from the data, this sweep would be checking
    itself and whatever number came out would match. That is also what blocked
    this test: until the UPLIFT-02 config landed, these three numbers could not
    be reproduced from any side without inventing the layer list.

    The THIRD number of the UPLIFT-07 Definition of Done did not survive
    meeting the store. It writes 80 plot-shaped polygons outside the coded
    layers, with a spread of {0: 52, Open Spaces: 15, Linear Park: 12, Pocket
    Park: 1}. Measured: 77, with {0: 52, Open Spaces: 13, Linear Park: 11,
    Pocket Park: 1}. The difference is three polygons, all of them on two park
    layers, and the mechanism can be explained: UPLIFT-01 removes bulged
    polygons and open polygons from the candidates, and parks are drawn with
    arcs far more often than plots are. The `0` figure, which did not move at
    all, reinforces that — a copy on layer 0 is a copy of a plot, and plots
    have no arcs.

    Tested as MEASURED, with the difference recorded in docs/PROGRESS-R.md —
    not loosened to `>= 70`, which would stop catching anything.
    """
    r = _run(
        "cross_check_classification",
        {
            "layers": _config_layers(use="residential"),
            "edge_range": list(EDGE_RANGE),
            "area_range": list(AREA_RANGE),
        },
    )
    assert r["coded_total"] == GT_CODED_TOTAL
    assert r["coded_passing_shape_test"] == GT_CODED_PASSING
    assert r["coded_failing_shape_test"] == GT_CODED_FAILING
    assert r["shape_matches_outside_coded_layers"] == GT_OUTSIDE_MEASURED
    assert {row["layer"]: row["count"] for row in r["outside_by_layer"]} == (
        GT_OUTSIDE_BY_LAYER
    )
    assert ev.audit_response(r) == []


def test_cross_check_refuses_to_invent_the_shape_test():
    """`edge_range` and `area_range` are traits of ONE drawing. A default
    inside the code makes this sweep a solution specific to that drawing
    (G1)."""
    with pytest.raises(recipes.RecipeRefused) as excinfo:
        recipes.run(
            DRAWING, "cross_check_classification", {},
            layout=LAYOUT, units=METRE_UNITS,
        )
    assert excinfo.value.code == "RECIPE_PARAM_REQUIRED"
    assert "size_module" in excinfo.value.hint


# --- invariants: every drawing, not one (G5) ---------------------------------


def _every_drawing():
    from app import store as live_store

    try:
        drawings = live_store.list_drawings()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawings:
        pytest.skip("no drawing has been ingested")
    return live_store, drawings


def test_invariant_no_recipe_ever_invents_a_unit():
    """G2 + G5. 11 of 16 drawings are in inches and 3 declare no units at all;
    a response that writes "m" because the drawing that happens to be open
    uses metres is wrong for most drawings."""
    live_store, drawings = _every_drawing()
    checked = 0
    for d in drawings:
        did = d.get("drawing_id") or d.get("_id")
        units = live_store._unit_names(d, LAYOUT)
        try:
            r = recipes.run(did, "size_module", {"top": 3}, layout=LAYOUT, units=units)
        except Exception:
            # A drawing with no `Model` layout, or with no polygons at all.
            # That is a normal state in this store and is not what is being
            # tested here.
            continue
        problems = ev.audit_response(r)
        assert problems == [], f"{did}: {problems}"
        if units.get("declared_in_file") is False:
            text = json.dumps(r, default=str)
            assert '"unit": "m"' not in text, did
        checked += 1
    assert checked > 0


def test_invariant_a_drawing_without_land_use_config_is_refused_gracefully():
    """G3. A drawing without a config does not error 500 and does not guess;
    it refuses with a suggestion that is actionable in two directions — create
    the config, or name the layers directly."""
    live_store, drawings = _every_drawing()
    seen = 0
    for d in drawings:
        did = d.get("drawing_id") or d.get("_id")
        units = live_store._unit_names(d, LAYOUT)
        try:
            recipes.run(
                did, "coverage_gap", {"number_layer": "apa pun"},
                layout=LAYOUT, units=units,
            )
        except recipes.RecipeRefused as exc:
            if exc.code == "RECIPE_NO_CONFIG":
                assert ".yaml" in exc.hint or "parameter" in exc.hint
                seen += 1
        except Exception:
            continue
    if seen == 0:
        pytest.skip("every drawing that exists already has a land use config")


@pytest.mark.parametrize("units", ALL_UNIT_SHAPES)
def test_invariant_the_envelope_holds_for_every_shape_of_units(units):
    """The envelope is the same whatever the units, including a drawing that
    does not declare them and a paper layout."""
    scope = rr.scope_for(layout=LAYOUT, units=units, note="test scope")
    assert scope.units == units
    out = rr.envelope(
        _fake_recipe({}, carries_meaning=False),
        drawing_id=DRAWING, layout=LAYOUT, params_used={},
        body={"scope_note": scope.note, "not_measured": "x"},
    )
    assert out["scope_note"] == "test scope"
    assert out["recipe"] == "_fake"


def test_invariant_the_catalogue_is_never_empty_however_it_is_imported():
    """`library` is imported for its side effect in `__init__`, not left to the
    caller. An empty catalogue reads as "this agent cannot do anything", and
    that is a wrong answer in the right tone."""
    import importlib

    module = importlib.import_module("app.recipes")
    assert module.catalog()["count"] == len(EIGHT_RECIPES)
