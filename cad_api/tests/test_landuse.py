"""Acceptance tests for UPLIFT-02 — the land use config + `land_use_summary`.

Two parts, deliberately separated, following the habit of `test_measure.py`.

The first part is pure: the loader, the validator, and the evidence grade are
ordinary functions over dicts and are tested as such. It runs without a
database and is never skipped, so the rules that are the reason this module
exists — a pattern hit is always `verified: false`, two sources of the same
kind are never `corroborated`, a typology code is never its own definition —
have a guard that is always alive.

The second part runs on top of already ingested data and **skips while naming
its reason** if the database is not there. The number that is the reason this
feature was built (2,380 house plots on a drawing that was once answered with
"there may be no houses here") is a property of the stored data, and a test
that faked the aggregation would be green while the aggregation was wrong.
`pytest -rs` names what was skipped; read that before trusting a green run.
"""

from __future__ import annotations

import types

import pytest

from app import evidence as ev
from app import landuse, store, store_landuse
from app.landuse import crs as crs_math

# ---------------------------------------------------------------------------
# Reference figures, pinned
# ---------------------------------------------------------------------------

JANADRIYAH = "596212db022a3397"
MODEL = "Model"

#: Parcel count per land use in model space. All three were measured from DXF.
REF_PARCELS = {"residential": 2380, "education": 9, "religious": 6}

#: Σ area of the parcels that HAVE a closed ring, m2. `religious` is not
#: 16,728.46: two of the five `Local Mosque` parcels have bulges, so their area
#: is withheld. See the contradiction note in docs/PROGRESS.md.
REF_AREA = {"residential": 712977.389756, "education": 41110.456509,
            "religious": 13049.319316}

#: The number that must NOT appear: it can only be obtained by summing the
#: chord polygons of two bulged parcels, which is exactly what
#: `ring_status: bulge` exists to prevent.
CHORD_INFLATED_RELIGIOUS_AREA = 16728.46

REF_LAYERS = {"residential": 13, "education": 6, "religious": 2}

EMPTY_LAND_USE_LAYERS = [
    "CS-Land use-00_Commercial",
    "CS-Land use-00_Education",
    "CS-Land use-00_Open Space",
    "CS-Land use-00_Religious_Mosque",
    "CS-Land use-00_Roads and Parking",
    "CS-Land use-00_Utilities",
]

#: Synthetic units for the pure tests. `Scope` refuses to stand up without
#: units, and that is deliberate: 11 of the 18 drawings here are in inches.
UNITS = {
    "name": "m",
    "declared_in_file": True,
    "length_unit": "m",
    "area_unit": "m2",
    "space": "model",
}


@pytest.fixture(autouse=True)
def _fresh_config_cache():
    """Config is cached per mtime; a test that writes a temporary file has to start clean."""
    landuse.clear_cache()
    yield
    landuse.clear_cache()


def _scope(layout: str | None = MODEL) -> ev.Scope:
    return ev.Scope(
        layout=layout,
        includes_block_definitions=False,
        units=UNITS,
        note="pure test",
    )


def _grade_of(land_use: landuse.LandUse, tokens=()) -> ev.Grade:
    """The evidence grade of one config entry, without touching the database."""
    claim = ev.Claim(value=land_use.use, tokens=tuple(tokens))
    evidence = ev.Evidence.of(
        claim,
        drawing_id=JANADRIYAH,
        scope=_scope(),
        provenance=land_use.provenance(),
        observations=[s.observation(scope_layout=MODEL) for s in land_use.sources],
        not_established=land_use.not_established,
        how_to_verify=land_use.how_to_verify,
    )
    return evidence.grade


# ---------------------------------------------------------------------------
# Pure: loader and validator
# ---------------------------------------------------------------------------


def test_golden_config_loads_and_passes_the_validator():
    """Definition of Done 1. A config that does not pass the validator never
    reaches an answer; this one has to pass it whole."""
    config = landuse.for_drawing(JANADRIYAH)
    assert config is not None, "the Janadriyah config was not found"
    assert config.version == 1
    assert config.default_use == landuse.UNKNOWN_USE
    assert len(config.layers) >= 42
    assert sorted(config.empty_but_declared) == EMPTY_LAND_USE_LAYERS


def test_every_golden_layer_uses_the_locked_vocabulary():
    config = landuse.for_drawing(JANADRIYAH)
    for name, entry in config.layers.items():
        assert entry.use in landuse.USES, name
        assert entry.role in landuse.ROLES, name
        assert entry.sources, f"{name} without a source"


def test_the_retired_verified_and_source_keys_are_refused_by_name():
    """D-082. `verified: true` + `source: "layer name"` is one source, one
    corpus, marked verified by hand — a mechanism for raising the evidence
    grade in YAML form. A half-migrated config accepted in silence would
    publish a `verified: true` with nothing behind it."""
    for key, value in (("verified", True), ("source", "layer name")):
        with pytest.raises(landuse.ConfigError) as caught:
            landuse._build_land_use(
                "L",
                {"use": "education", key: value,
                 "sources": [{"origin": "layer_name", "observed": "L",
                              "detail": "layer name"}]},
                config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
                config_version=1,
                where="test",
            )
        assert caught.value.code == "LAND_USE_RETIRED_KEY"
        assert key in caught.value.message


def test_a_use_outside_the_vocabulary_is_refused_with_the_layer_named():
    with pytest.raises(landuse.ConfigError) as caught:
        landuse._build_land_use(
            "Mystery Layer",
            {"use": "shopping_mall",
             "sources": [{"origin": "layer_name", "observed": "Mystery Layer",
                          "detail": "layer name"}]},
            config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
            config_version=1,
            where="test layer 'Mystery Layer'",
        )
    assert caught.value.code == "LAND_USE_UNKNOWN_USE"
    assert "Mystery Layer" in caught.value.message


def test_a_role_outside_the_four_is_refused():
    with pytest.raises(landuse.ConfigError) as caught:
        landuse._build_land_use(
            "L",
            {"use": "education", "role": "plot",
             "sources": [{"origin": "layer_name", "observed": "L",
                          "detail": "layer name"}]},
            config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
            config_version=1,
            where="test",
        )
    assert caught.value.code == "LAND_USE_UNKNOWN_ROLE"


def test_an_entry_without_sources_is_refused():
    """A classification with no basis is more dangerous than no classification
    at all: it reads as a fact and cannot be checked by anyone."""
    with pytest.raises(landuse.ConfigError) as caught:
        landuse._build_land_use(
            "L", {"use": "education"},
            config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
            config_version=1, where="test",
        )
    assert caught.value.code == "LAND_USE_WITHOUT_SOURCES"


def test_a_source_with_an_origin_evidence_does_not_know_is_refused():
    with pytest.raises(landuse.ConfigError) as caught:
        landuse._build_source(
            {"origin": "vibes", "observed": "L", "detail": "a hunch"}, where="test"
        )
    assert caught.value.code == "LAND_USE_UNKNOWN_ORIGIN"


def test_a_speaking_source_without_observed_is_refused_by_the_evidence_contract():
    """The validator does not copy the rules of `evidence.py`; it builds an
    `Observation` and lets the contract do the refusing. Two copies of a rule
    would differ once one of them is edited."""
    with pytest.raises(landuse.ConfigError) as caught:
        landuse._build_source(
            {"origin": "layer_name", "detail": "layer name"}, where="test"
        )
    assert caught.value.code == "LAND_USE_SOURCE_INVALID"


def test_unknown_without_a_route_to_an_answer_is_refused():
    with pytest.raises(landuse.ConfigError) as caught:
        landuse._build_land_use(
            "VLX",
            {"use": "residential", "unknown": "what the VL code means is not stated",
             "sources": [{"origin": "layer_name", "observed": "VLX",
                          "detail": "layer name"}]},
            config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
            config_version=1, where="test",
        )
    assert caught.value.code == "LAND_USE_UNKNOWN_WITHOUT_ROUTE"


# ---------------------------------------------------------------------------
# Pure: the two config layers (G4)
# ---------------------------------------------------------------------------


def test_a_drawing_with_no_config_still_gets_pattern_coverage_and_is_never_verified():
    """G4. The 17th drawing has partial coverage on the day it arrives, without
    a single guess masquerading as a fact."""
    hit = landuse.classify("0000000000000000", "Primary School Plot")
    assert hit is not None
    assert hit.use == "education"
    assert hit.config_layer is ev.ConfigLayer.GLOBAL_PATTERN
    assert hit.provenance().verified is False
    assert hit.matched_pattern


def test_a_pattern_hit_can_never_be_corroborated():
    """A pattern that matches is not a second source — it is a way of reading
    the same source. That is not forced through a ceiling; it falls out of the
    source list on its own."""
    hit = landuse.classify("0000000000000000", "Local Mosque Parcel")
    tokens = landuse.patterns().tokens_for("religious")
    assert _grade_of(hit, tokens) is ev.Grade.STATED
    assert len(hit.sources) == 1


def test_a_drawing_override_beats_a_pattern():
    """The order is the contract: per-drawing override -> global pattern ->
    unclassified."""
    hit = landuse.classify(JANADRIYAH, "SchoolHatch")
    assert hit.config_layer is ev.ConfigLayer.DRAWING_OVERRIDE
    assert hit.role == "overlay"


def test_a_layer_nothing_decides_is_none_and_not_a_guess():
    """There is no fuzzy matching. A layer with no entry returns None and is
    counted as `unclassified`, not guessed."""
    assert landuse.classify(JANADRIYAH, "C-ROAD-CNTR") is None
    assert landuse.classify(JANADRIYAH, "DIM") is None


def test_car_parking_is_not_a_park():
    """`speaks_for` tests Latin tokens with word boundaries, and the pattern
    order puts parking above parks. Both have to be right at once."""
    hit = landuse.classify("0000000000000000", "CAR PARKING AREA")
    assert hit.use == "parking"
    tokens = landuse.patterns().tokens_for("open_space")
    assert ev.speaks_for("CAR PARKING AREA", tokens) is None


def test_a_misfiled_config_is_refused(tmp_path, monkeypatch):
    """G10. The file name is the key; a misfiled config would move meaning
    between drawings."""
    (tmp_path / "1111111111111111.yaml").write_text(
        'version: 1\ndrawing_id: "2222222222222222"\nlayers: {}\n', encoding="utf-8"
    )
    monkeypatch.setattr(landuse, "CONFIG_DIR", tmp_path)
    with pytest.raises(landuse.ConfigError) as caught:
        landuse.for_drawing("1111111111111111")
    assert caught.value.code == "LAND_USE_CONFIG_MISFILED"


def test_a_missing_config_file_is_none_not_an_error(tmp_path, monkeypatch):
    """G3. The absence of a config has to be graceful."""
    monkeypatch.setattr(landuse, "CONFIG_DIR", tmp_path)
    assert landuse.for_drawing("3333333333333333") is None


# ---------------------------------------------------------------------------
# Pure: the evidence grades that come out of the golden config
# ---------------------------------------------------------------------------


def test_residential_is_inferred_and_can_never_be_stated():
    """UPLIFT-08 E2. Houses are never stated by this file: zero of the 43,109
    TEXT/MTEXT contain 'residential', 'villa', or 'house'. The bases are many
    and still infer — the plot module and the block boundary tiling are one act
    of drawing read two ways."""
    config = landuse.for_drawing(JANADRIYAH)
    tokens = landuse.patterns().tokens_for("residential")
    rows = [e for e in config.layers.values() if e.use == "residential"]
    assert len(rows) == REF_LAYERS["residential"]
    for entry in rows:
        assert _grade_of(entry, tokens) is ev.Grade.INFERRED, entry.layer
        assert entry.not_established, entry.layer
        assert entry.how_to_verify, entry.layer


def test_a_typology_code_can_never_become_its_own_definition():
    """Without this fence, someone adds `vl` to the vocabulary and 2,380 plots
    climb to `stated` without one line of Python changing."""
    vocabulary, problems = ev.load_vocabulary(
        {"residential": {"tokens": ["vl2", "vl", "villa"]}}
    )
    assert vocabulary["residential"] == ("villa",)
    assert len(problems) == 2
    assert landuse.patterns().tokens_for("residential")
    assert ev.speaks_for("VL2", landuse.patterns().tokens_for("residential")) is None


def test_two_sources_of_the_same_corpus_never_reach_corroborated():
    """UPLIFT-08 Definition of Done. A layer name and a block name are both
    names; both can be wrong for the same reason."""
    entry = landuse._build_land_use(
        "Jumaa Mosque",
        {"use": "religious",
         "sources": [
             {"origin": "layer_name", "observed": "Jumaa Mosque",
              "locator": "Jumaa Mosque", "detail": "layer name"},
             {"origin": "block_name", "observed": "Mosque Block",
              "locator": "Mosque Block", "detail": "block name"},
         ]},
        config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
        config_version=1, where="test",
    )
    tokens = landuse.patterns().tokens_for("religious")
    assert _grade_of(entry, tokens) is ev.Grade.STATED


def test_the_layer_named_mosque_with_an_arabic_label_is_corroborated():
    """UPLIFT-08 E3. The layer name and the Arabic label were written on
    different occasions by different tools, so the two cannot be wrong for the
    same reason."""
    config = landuse.for_drawing(JANADRIYAH)
    tokens = landuse.patterns().tokens_for("religious")
    assert _grade_of(config.layers["Local Mosque"], tokens) is ev.Grade.CORROBORATED
    # ...and the one with no second label stays `stated`. The difference is
    # real: there is not a single 'مسجد جامع' text in this file.
    assert _grade_of(config.layers["Jumaa Mosque"], tokens) is ev.Grade.STATED


def test_the_school_layers_with_a_label_are_corroborated_and_the_one_without_is_not():
    config = landuse.for_drawing(JANADRIYAH)
    tokens = landuse.patterns().tokens_for("education")
    assert _grade_of(config.layers["Primary School"], tokens) is ev.Grade.CORROBORATED
    assert _grade_of(config.layers["Private School"], tokens) is ev.Grade.STATED


def test_the_annotation_and_overlay_layers_are_not_parcels():
    """Traps 1 and 2. 416 TEXT were once counted as a 9th plot type, and 8
    hatches were once used as a school counter — wrong in both directions,
    because there are nine education parcels."""
    config = landuse.for_drawing(JANADRIYAH)
    assert config.layers["T4"].is_parcel is False
    assert config.layers["T4"].role == "annotation"
    assert config.layers["SchoolHatch"].is_parcel is False
    assert config.layers["SchoolHatch"].role == "overlay"
    assert config.layers["BlockBoundary"].is_parcel is False


# ---------------------------------------------------------------------------
# On top of already ingested data
# ---------------------------------------------------------------------------


def _mongo_or_skip(drawing_id: str = JANADRIYAH):
    try:
        drawing = store.get_drawing(drawing_id)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    if not drawing:
        pytest.skip(
            f"the reference drawing {drawing_id} has not been ingested; these "
            "acceptance figures cannot be checked"
        )
    return drawing


@pytest.fixture(scope="module")
def summary():
    _mongo_or_skip()
    return store_landuse.land_use_summary(JANADRIYAH, MODEL)


def _row(summary, use: str) -> dict:
    for row in summary["uses"]:
        if row["use"] == use:
            return row
    raise AssertionError(f"land use {use!r} is not in the response")


def test_the_three_reference_figures(summary):
    """Definition of Done 2, the part that can be reached.

    `religious` is deliberately NOT 16,728.46: that number can only be obtained
    by summing the chord polygons of two bulged `Local Mosque` parcels, and
    `ring_status: bulge` exists precisely to refuse that. What is reported is
    the 4 of 6 parcels that were measured, with 2 withheld and their reason
    travelling with them.
    """
    for use, parcels in REF_PARCELS.items():
        row = _row(summary, use)
        assert row["parcels"] == parcels, use
        assert row["layers"] == REF_LAYERS[use], use
        assert row["total_area"]["value"] == pytest.approx(REF_AREA[use], abs=0.01), use
        assert row["total_area"]["unit"] == "m2", use

    religious = _row(summary, "religious")
    assert religious["parcels_measured"] == 4
    assert religious["parcels_area_withheld"] == 2
    assert religious["total_area"]["value"] != pytest.approx(
        CHORD_INFLATED_RELIGIOUS_AREA, abs=1.0
    ), "the chord area of the bulged polygons was summed in"


def test_the_denominator_travels_with_every_area(summary):
    """A total without the count that makes it up is the second shape of wrong
    number this project has already shipped."""
    for row in summary["uses"]:
        assert row["parcels"] == row["parcels_measured"] + row["parcels_area_withheld"]
        area = row["total_area"]
        assert area["method"]
        if area["value"] is None:
            assert area["withheld_reason"]
            assert area["not_zero"] is True


def test_the_road_label_layer_is_never_a_residential_parcel(summary):
    """Definition of Done 3. The trap that once made the unit mix wrong by
    416."""
    residential = _row(summary, "residential")
    layers = {r["layer"] for r in residential["by_layer"]}
    assert "T4" not in layers
    for row in summary["uses"]:
        assert "T4" not in {r["layer"] for r in row["by_layer"]}

    not_counted = {r["layer"]: r for r in summary["roles_not_counted"]}
    assert not_counted["T4"]["role"] == "annotation"
    assert not_counted["T4"]["entities"] == 416
    assert "not a parcel" in not_counted["T4"]["why_not_counted"]


def test_the_hatch_overlay_adds_neither_area_nor_parcels(summary):
    """Definition of Done 4, and UPLIFT-08 E6/E7 at the same time. 8 hatches, 9
    education parcels: using the hatches as a school counter is wrong in both
    directions."""
    education = _row(summary, "education")
    assert "SchoolHatch" not in {r["layer"] for r in education["by_layer"]}
    assert education["parcels"] == 9
    assert education["total_area"]["value"] == pytest.approx(
        REF_AREA["education"], abs=0.01
    )

    not_counted = {r["layer"]: r for r in summary["roles_not_counted"]}
    assert not_counted["SchoolHatch"]["role"] == "overlay"
    assert not_counted["SchoolHatch"]["entities"] == 8


def test_the_hatch_layer_on_a_parcel_layer_is_reported_not_counted(summary):
    """The commercial layer contains 3 parcel polygons and 15 HATCHes. What is
    not counted is still named, so that "why only three" has an answer."""
    commercial = _row(summary, "commercial")
    excluded = {
        r["type"]: r["entities"]
        for r in commercial["types_on_these_layers_not_counted_as_parcels"]
    }
    assert excluded.get("HATCH") == 15


def test_the_declared_but_empty_land_use_layers_are_named(summary):
    """Definition of Done 5. A query to these layers returns nil, and that nil
    reads as 'there are no schools in this drawing'.

    The list is longer than six, and that is a finding: `Parking Area`,
    `Living Plaza`, `District Park`, `Neighbourhood Park` and `STP` are also
    declared and empty. What is tested here is that all six ARE in it, not that
    only those six are.
    """
    named = summary["empty_but_declared"]
    assert set(EMPTY_LAND_USE_LAYERS) <= set(named)
    for fact in summary["empty_but_declared_facts"]:
        assert fact["in_layer_table"] is True, fact["layer"]
        assert fact["entities_in_file"] == 0, fact["layer"]
    assert "NOT" in summary["empty_but_declared_note"]


def test_the_empty_layer_list_is_derived_and_not_only_hand_written(summary):
    """A hand-written list stops being right at the next revision. An empty land
    use layer is derived from the layer table, so it does not have to wait for
    someone to remember to add it."""
    facts = {f["layer"]: f for f in summary["empty_but_declared_facts"]}
    derived_only = [
        f for f in facts.values() if f["from"] == "derived from the layer table"
    ]
    assert derived_only, "not one of them is derived; the check is dead"
    for name in ("District Park", "Neighbourhood Park", "STP"):
        assert name in facts, name
        assert facts[name]["land_use"], name


def test_an_empty_land_use_layer_still_names_its_use(summary):
    """P0. 'how many schools are there' is answered nil by a query to an empty
    land use layer. The response has to name nine parcels on six OTHER layers
    while also acknowledging that empty layer."""
    education = _row(summary, "education")
    assert education["parcels"] == 9
    assert education["layers"] == 6
    assert "CS-Land use-00_Education" in education["layers_with_no_entity_in_scope"]
    assert education["layers_configured"] == 7


def test_residential_carries_its_inferred_grade_to_every_row(summary):
    """Definition of Done 6, in the shape it has after D-082.

    `verified` is now derived from the config layer, so the Janadriyah override
    entries come out `verified: true` — and what carries "this is a conclusion,
    not a statement by the file" is `grade`, not `verified`. What is tested
    here is that this property reaches EVERY residential row and is not lost
    along the way.
    """
    residential = _row(summary, "residential")
    assert residential["evidence"]["grade"] == "inferred"
    assert residential["evidence"]["grade_spread"] == {
        "unknown": 0, "inferred": 13, "stated": 0, "corroborated": 0,
    }
    for row in residential["by_layer"]:
        assert row["grade"] == "inferred", row["layer"]
        assert row["not_established"], row["layer"]
        assert row["how_to_verify"], row["layer"]
    assert summary["inferred_note"]
    assert residential["evidence"]["not_established"]
    assert "villa" not in residential["evidence"]["statement"].lower()


def test_the_use_grade_is_the_weakest_layer_that_actually_carries_parcels(summary):
    """Not `corroborated`, and that is a finding, not a regression.

    `Jumaa Mosque` has no second label — there is not a single 'مسجد جامع' text
    in this file — and `Private School` has neither a SchoolHatch nor a T4
    label. An aggregate grade must not be stronger than its weakest member, so
    both are `stated`, and it is `grade_spread` that shows that the majority
    are corroborated.
    """
    education = _row(summary, "education")
    assert education["evidence"]["grade"] == "stated"
    assert education["evidence"]["grade_spread"]["corroborated"] == 5
    assert education["evidence"]["grade_spread"]["stated"] == 1

    religious = _row(summary, "religious")
    assert religious["evidence"]["grade"] == "stated"
    assert religious["evidence"]["grade_spread"]["corroborated"] == 1
    assert religious["evidence"]["grade_spread"]["stated"] == 1


def test_an_empty_declared_layer_does_not_weaken_the_parcels_that_exist(summary):
    """`CS-Land use-00_Education` contributes zero parcels. Letting it drag nine
    education parcels down one grade would make the evidence sound weaker
    precisely because an empty layer exists in the file."""
    education = _row(summary, "education")
    assert education["evidence"]["grade_spread"]["inferred"] == 0
    assert sum(education["evidence"]["grade_spread"].values()) == education["layers"]


def test_the_bases_of_an_inferred_answer_can_be_quoted(summary):
    """UPLIFT-08 E2: "how do you know that is a house" has to be answerable with
    at least two bases. The `weakest()` aggregate deliberately has no
    observations of its own, so its bases are collected separately."""
    residential = _row(summary, "residential")
    origins = {s["origin"] for s in residential["distinct_sources"]}
    assert {"topology", "absence", "geometry", "layer_name"} <= origins
    assert len(residential["distinct_sources"]) >= 4


def test_every_typology_code_keeps_its_own_open_question(summary):
    """UPLIFT-08 E1. `evidence.not_established` carries only one
    representative; if that were all there was, twelve other codes would lose
    their answer."""
    residential = _row(summary, "residential")
    assert len(residential["open_questions"]) == REF_LAYERS["residential"]
    for question in residential["open_questions"]:
        assert question["how_to_verify"]
    assert any(q["use"] == "residential" for q in summary["open_questions"])


def test_a_pattern_layer_would_be_reported_unverified(summary):
    """G4, seen from the response side: the config layer that decided travels
    out, so its reader can tell what was reviewed by hand from what was
    patterned."""
    assert summary["config_layers_used"] == ["drawing_override"]
    for row in summary["uses"]:
        for layer_row in row["by_layer"]:
            assert layer_row["config_layer"] in {"drawing_override", "global_pattern"}
            if layer_row["config_layer"] == "global_pattern":
                assert layer_row["verified"] is False


def test_the_golden_config_matches_the_drawing_it_is_written_for(summary):
    assert summary["config_problems"] == []
    assert summary["land_use_available"] is True
    assert summary["config_version"] == 1


def test_the_response_passes_the_evidence_contract_audit(summary):
    assert ev.audit_response(summary) == []
    assert summary["scope_note"]
    assert summary["evidence"]["drawing_id"] == JANADRIYAH


def test_an_unknown_drawing_is_empty_not_an_error():
    _mongo_or_skip()
    assert store_landuse.land_use_summary("ffffffffffffffff") == {}


def test_a_whole_drawing_summary_refuses_to_name_a_unit():
    """G2. A total that adds sheet coordinates to model coordinates is not a
    quantity in any unit, and saying so is the only honest option."""
    _mongo_or_skip()
    whole = store_landuse.land_use_summary(JANADRIYAH, None)
    assert whole["area_unit"] is None
    assert whole["area_unit_reason"]
    for row in whole["uses"]:
        assert row["total_area"]["unit"] is None
        if row["total_area"]["value"] is not None:
            assert row["total_area"]["unit_reason"]
    assert ev.audit_response(whole) == []


# --- Invariants across ALL drawings (G5) ------------------------------------


def _all_drawing_ids() -> list[str]:
    try:
        return [d["_id"] for d in store.list_drawings()]
    except Exception:  # pragma: no cover - depends on the environment
        return []


ALL_DRAWINGS = _all_drawing_ids()


@pytest.mark.parametrize("drawing_id", ALL_DRAWINGS or ["<no database>"])
def test_every_ingested_drawing_answers_gracefully(drawing_id):
    """G5 + G3. All 18 drawings, not one.

    A drawing without a land use config returns `land_use_available: false`
    together with its reason and the file that needs to be created — not a 500,
    not `{}`, and not a guessed classification.
    """
    if not ALL_DRAWINGS:
        pytest.skip("no database: the cross-drawing invariant cannot be run")

    result = store_landuse.land_use_summary(drawing_id, None)
    assert result, drawing_id
    assert ev.audit_response(result) == [], drawing_id
    assert isinstance(result["land_use_available"], bool)
    if not result["land_use_available"]:
        assert result["uses"] == []
        assert result["evidence"]["grade"] == "unknown"
        assert result["evidence"]["not_established"]
        assert result["evidence"]["how_to_verify"]
        assert result["config_file_expected_at"].startswith(drawing_id)
    else:
        assert result["uses"]



def configured_drawings() -> set[str]:
    """Asked of `landuse`, which owns where configs live. See tests/conftest.py
    for the drift this replaces; this file had its own second copy of the same
    glob and it failed the same way on the same day."""
    return landuse.configured_drawing_ids()


def test_only_configured_drawings_have_a_land_use_config():
    """Definition of Done 7. A drawing with no config is honestly reported as
    having none, and not one of its layer names is wrongly pulled into a land
    use by a global pattern.

    The configured set is read from the config tree rather than named here,
    so adding a config for a new drawing is a deliberate act that this test
    follows instead of a change that breaks it.
    """
    if not ALL_DRAWINGS:
        pytest.skip("no database")
    configured = configured_drawings()
    available = {
        d: store_landuse.availability(d)["land_use_available"] for d in ALL_DRAWINGS
    }
    assert available.get(JANADRIYAH) is True, "the reference drawing must stay configured"
    others = {d: v for d, v in available.items() if d not in configured}
    assert others, "every drawing in the store is configured; this tests nothing"
    assert not any(others.values()), [d for d, v in others.items() if v]


# --- Enrichment of existing tools -------------------------------------------


def test_layer_enrichment_leaves_unclassified_layers_absent():
    """`null` for all three, not `"unknown"` — which reads as a statement that
    this drawing says its land use is unknown."""
    rows = store_landuse.classify_layers(JANADRIYAH, ["VL2", "DIM", "VL2"])
    assert "DIM" not in rows
    assert rows["VL2"]["land_use"] == "residential"
    assert rows["VL2"]["land_use_subtype"] == "typology:VL2"
    assert rows["VL2"]["land_use_verified"] is True


# ---------------------------------------------------------------------------
# UPLIFT-14 — georeferencing
# ---------------------------------------------------------------------------

#: Cross-references anyone can check, from `UPLIFT-14`. Both were recomputed
#: with pyproj outside this repo, so they test the Krüger series in
#: `landuse/crs.py` against an entirely different implementation.
GEOREF = {
    "205D0EB": (24.85142, 46.89358),   # Jumaa Mosque
    "205D153": (24.85975, 46.89974),   # Primary School
}

JANADRIYAH_EPSG = 32638


def test_the_utm_inverse_agrees_with_pyproj_on_the_reference_points():
    """A tolerance of 1e-5 degrees, about 1.1 m. The reference figures
    themselves are rounded to five decimals, so anything tighter than this
    would be testing their rounding."""
    _mongo_or_skip()
    for handle, (lat, lon) in GEOREF.items():
        row = store.get_entity(JANADRIYAH, handle)
        got_lat, got_lon = crs_math.to_lat_lon(
            row["polygon_centroid"][0], row["polygon_centroid"][1],
            epsg=JANADRIYAH_EPSG,
        )
        assert got_lat == pytest.approx(lat, abs=1e-5), handle
        assert got_lon == pytest.approx(lon, abs=1e-5), handle


def test_the_wrong_zone_is_wrong_by_six_degrees_and_says_nothing_about_it():
    """Why there must be no default zone. Zone 37N puts the reference drawing
    in the Red Sea, and nothing in the numbers shouts — it is still a plausible
    looking lat/long."""
    lat38, lon38 = crs_math.to_lat_lon(687513.9187, 2747693.4530, epsg=32638)
    lat37, lon37 = crs_math.to_lat_lon(687513.9187, 2747693.4530, epsg=32637)
    assert lat38 == pytest.approx(lat37, abs=1e-9)
    assert lon38 - lon37 == pytest.approx(6.0, abs=1e-9)
    assert -90 <= lat37 <= 90 and -180 <= lon37 <= 180


def test_a_projection_this_module_cannot_invert_is_refused_not_approximated():
    with pytest.raises(crs_math.CrsUnsupported) as caught:
        crs_math.to_lat_lon(0.0, 0.0, epsg=4326)
    assert caught.value.code == "CRS_NOT_SUPPORTED"


def test_the_reference_config_declares_the_crs_as_inferred():
    """DoD 1. `declared_in_file: false` is the field that prevents the easiest
    mistake of all: a lat/long that looks official while the file never stated
    its coordinate system."""
    config = landuse.for_drawing(JANADRIYAH)
    assert config.crs is not None
    assert config.crs.epsg == JANADRIYAH_EPSG
    assert config.crs.declared_in_file is False
    assert config.crs.not_established and config.crs.how_to_verify
    assert len(config.crs.sources) >= 2


def test_a_crs_block_without_the_declared_flag_is_refused():
    with pytest.raises(landuse.ConfigError) as caught:
        landuse._build_crs(
            {"epsg": 32638, "name": "x",
             "sources": [{"origin": "geometry", "detail": "where the extents sit"}]},
            where="test",
        )
    assert caught.value.code == "CRS_WITHOUT_DECLARED_FLAG"


def test_a_crs_block_without_sources_is_refused():
    with pytest.raises(landuse.ConfigError) as caught:
        landuse._build_crs(
            {"epsg": 32638, "name": "x", "declared_in_file": False}, where="test"
        )
    assert caught.value.code == "CRS_WITHOUT_SOURCES"


def test_a_crs_block_with_an_epsg_this_module_cannot_invert_is_refused():
    with pytest.raises(landuse.ConfigError) as caught:
        landuse._build_crs(
            {"epsg": 27700, "declared_in_file": True,
             "sources": [{"origin": "geometry", "detail": "where the extents sit"}]},
            where="test",
        )
    assert caught.value.code == "CRS_NOT_SUPPORTED"


def test_the_crs_claim_can_never_climb_above_inferred():
    """Where it sits and the absence of GEODATA are both corpora that never
    state. This test is what holds "convincing" back from turning into
    "stated"."""
    _mongo_or_skip()
    block = store_landuse.describe_crs(JANADRIYAH)
    assert block["crs"]["known"] is True
    assert block["crs"]["grade"] == "inferred"
    assert block["evidence"]["independent_corpora"] == []


def test_every_lat_lon_carries_the_sentence_that_says_it_was_inferred():
    """DoD 4. A lat/long from the wrong zone still looks like a correct
    lat/long, so the warning attaches to the number — not to the
    documentation."""
    _mongo_or_skip()
    for handle in GEOREF:
        got = store_landuse.lat_lon_for_entity(JANADRIYAH, handle)
        assert got is not None, handle
        assert "INFERRED" in got["caveat"]
        assert "before it is confirmed" in got["caveat"]
        assert got["crs_declared_in_file"] is False
        assert got["crs_grade"] == "inferred"
        assert got["crs_confirmed_by"] is None
        assert got["from_field"] == "polygon_centroid"
        assert got["evidence"]["how_to_verify"]

    block = store_landuse.describe_crs(JANADRIYAH)
    assert "INFERRED" in block["crs"]["caveat"]


def test_the_caveat_switches_on_who_confirmed_it_not_on_who_typed_it():
    """After D-082, `verified` answers "did someone type an override" — and for
    a CRS inferred from where the extents sit the answer is always yes. What
    its reader is asking is whether Roshn has confirmed the zone, and that can
    only come from a source outside the file — which the evidence contract
    requires to name the document or the person."""
    inferred = landuse._build_crs(
        {"epsg": 32638, "name": "WGS 84 / UTM zone 38N", "declared_in_file": False,
         "sources": [{"origin": "geometry", "detail": "where the extents sit"}]},
        where="test",
    )
    assert crs_math.confirmed_by(inferred) is None
    assert "INFERRED" in crs_math.caveat(inferred)

    confirmed = landuse._build_crs(
        {"epsg": 32638, "name": "WGS 84 / UTM zone 38N", "declared_in_file": False,
         "sources": [
             {"origin": "geometry", "detail": "where the extents sit"},
             {"origin": "human", "detail": "confirmed by email",
              "observed": "UTM 38N", "reference": "GIS Roshn, 1 Sep 2026"},
         ]},
        where="test",
    )
    assert crs_math.confirmed_by(confirmed) == "GIS Roshn, 1 Sep 2026"
    assert "GIS Roshn, 1 Sep 2026" in crs_math.caveat(confirmed)
    assert "INFERRED" not in crs_math.caveat(confirmed)


def test_lat_lon_matches_the_cross_reference_through_the_store():
    """DoD 3, through the route that is actually used."""
    _mongo_or_skip()
    for handle, (lat, lon) in GEOREF.items():
        got = store_landuse.lat_lon_for_entity(JANADRIYAH, handle)
        assert got["lat"] == pytest.approx(lat, abs=1e-5), handle
        assert got["lon"] == pytest.approx(lon, abs=1e-5), handle


def test_the_extents_box_reports_four_corners_not_two():
    """On a grid rotated by 49°, the south-west corner is NOT the point with the
    smallest latitude and longitude at once. `UPLIFT-14` computed the box from
    two corners and therefore cut off part of the drawing in the north."""
    _mongo_or_skip()
    bounds = store_landuse.describe_crs(JANADRIYAH)["crs"]["bounds"]
    assert len(bounds["corners"]) == 4
    south_west = bounds["corners"][0]
    assert south_west["lat"] == pytest.approx(24.83265, abs=1e-5)
    assert south_west["lon"] == pytest.approx(46.85552, abs=1e-5)
    # ...and that corner is not its smallest latitude.
    assert bounds["lat_min"] < south_west["lat"]


def test_a_drawing_without_a_crs_config_never_returns_a_lat_lon():
    """DoD 5, and the most important rule in UPLIFT-14. The 17 Autodesk samples
    use local coordinates and some of them are in inches; applying a UTM zone to
    them produces nonsense shaped like a lat/long."""
    if not ALL_DRAWINGS:
        pytest.skip("no database")
    configured = configured_drawings()
    for drawing_id in ALL_DRAWINGS:
        if drawing_id in configured:
            continue
        block = store_landuse.describe_crs(drawing_id)
        assert block["crs"]["known"] is False, drawing_id
        assert block["crs"]["epsg"] is None, drawing_id
        assert block["crs"]["not_established"], drawing_id
        assert block["crs"]["how_to_verify"], drawing_id
        assert block["evidence"]["grade"] == "unknown", drawing_id
        assert ev.audit_response(block) == [], drawing_id

        rows = store.query_entities(drawing_id, limit=1)["entities"]
        if rows:
            assert store_landuse.lat_lon_for_entity(
                drawing_id, rows[0]["handle"]
            ) is None, drawing_id


def test_the_land_use_summary_says_whether_the_answer_can_be_put_on_a_map(summary):
    assert summary["crs"]["known"] is True
    assert summary["crs"]["epsg"] == JANADRIYAH_EPSG
    assert summary["crs"]["evidence"]["grade"] == "inferred"
    assert ev.audit_response(summary) == []


def test_a_text_search_that_finds_nothing_can_still_name_the_land_use():
    """The most expensive failure in this project: `search_text("school")`
    returns zero rows on a drawing containing nine education parcels, and those
    zero rows are correct."""
    _mongo_or_skip()
    hits = {row["use"]: row for row in store_landuse.uses_matching(JANADRIYAH, "SCHOOL")}
    assert "education" in hits
    assert hits["education"]["matched_token"] == "school"
    assert "Primary School" in hits["education"]["layers"]


# --- per-layer count and mean area, inside ONE response ---------------------
#
# Added 24 August 2026 after watching the agent answer "how many plots per type
# and what is their mean area" with thirteen separate `stats` calls -- after a
# `land_use_summary` call that already contained the counts. On one of those
# runs the seventh call came out with a broken tool name and the whole answer
# was lost even though the other twelve calls had succeeded.


def test_every_layer_row_carries_its_own_count_and_mean_area(summary):
    """One call answers the question, or it is of no use."""
    rows = [r for use in summary["uses"] for r in use.get("by_layer", [])]
    assert rows, "the test drawing has not a single classified layer"
    for row in rows:
        assert "parcels" in row
        assert "parcels_measured" in row
        assert "area_mean" in row
        assert "area_total" in row
        assert "area_unit" in row


def test_the_mean_divides_by_what_was_measured_not_by_everything(summary):
    """A mean that divides by the full population while its numerator only sums
    the measured ones will be lower than the truth, and will look
    plausible."""
    for use in summary["uses"]:
        for row in use.get("by_layer", []):
            if not row.get("parcels_measured"):
                assert row["area_mean"] is None
                assert row["area_total"] is None
                continue
            expected = row["area_total"] / row["parcels_measured"]
            # Both are published at six decimals, so an agreement tighter than
            # that is not a claim the already rounded numbers can support. What
            # is tested here is the divisor, not the float precision.
            assert row["area_mean"] == pytest.approx(expected, abs=1e-6)


def test_a_layer_with_unmeasurable_parcels_says_which_ones_it_dropped(summary):
    """A zero that is not explained is a zero that is believed."""
    for use in summary["uses"]:
        for row in use.get("by_layer", []):
            if row["parcels_measured"] == row["parcels"]:
                assert row["area_not_measured"] is None
            else:
                assert row["area_not_measured"], row["layer"]
                assert str(row["parcels"]) in row["area_not_measured"]


# ---------------------------------------------------------------------------
# DOSSIER Phase 2, lane B — the Dossier answers where the config is silent
# ---------------------------------------------------------------------------
#
# The campaign's origin is one sentence this module used to produce: **"0 road
# parcels"**, while 1,233 road entities and 83,962.565 m of centreline sat in
# `unclassified_layers` as a name in a list. The count was correct. Everything
# around it was missing.
#
# Everything in this section is PURE unless it takes the `summary` fixture. The
# Dossier profiles below are FIXTURES: shaped like what lane A returns and
# written to the figures Phase 1 measured, so a reader recognises them. They
# are inputs to these tests, never evidence that anything was read from Mongo.
# The checks that take `summary` are the live ones, and they skip while naming
# their reason when there is no database.

ROAD_LAYER = "00_Prop - Road - CL_"
ROAD_ENTITIES = 1233
ROAD_LENGTH_M = 83962.565009

#: A Civil3D sheet-layout layer. It matches the word "road" exactly as well as
#: the layer above does, and it is a grid on a drawing sheet. Sixty of these
#: are why `_patterns.yaml` refuses a road pattern, and telling them apart by
#: name is impossible — which is the whole point of publishing the geometry.
SHEET_TEMPLATE_LAYER = "C-ROAD-PROF-GRID-MINR"


def _road_profile() -> dict:
    return {
        "layer": ROAD_LAYER,
        "layout": MODEL,
        "role": "network",
        "role_basis": "the network family holds 98% of this bucket, above 0.60",
        "entities": ROAD_ENTITIES,
        "measure": {
            "kind": "length",
            "value": ROAD_LENGTH_M,
            "unit": "m",
            "unit_reason": None,
            "measured_entities": 1191,
            "unmeasured_entities": 42,
            "basis": "Sigma length over 1,191 of 1,233 entities",
        },
        "anomalies": [{"kind": "duplicate_clusters", "detail": "3 clusters of 411"}],
    }


def _sheet_template_profile() -> dict:
    return {
        "layer": SHEET_TEMPLATE_LAYER,
        "layout": MODEL,
        "role": "annotation",
        "entities": 4,
        "measure": {
            "kind": "count",
            "value": None,
            "unit": None,
            "unit_reason": "a count of entities has no unit",
            "measured_entities": 0,
            "unmeasured_entities": 4,
            "basis": "an annotation bucket is counted, not measured",
        },
    }


#: `residual=_UNDEFINED` means the module exists but has not written the
#: function yet — the half-written lane, which is a different failure from an
#: absent one and has to read differently.
_UNDEFINED = object()


def _fake_dossier(*, profiles=None, residual=_UNDEFINED, calls=None):
    """A stand-in for lane A, which is written in parallel with this file.

    `profiles` and `residual` are either a value to return or a callable. A
    value of `None` is lane A's contractual way of saying "this drawing has no
    Dossier", and it is deliberately easy to write here, because the difference
    between that and an empty answer is what these tests exist to pin.
    """

    def profiles_for(drawing_id, layer_names, *, layout=None):
        names = [str(n) for n in layer_names]
        if calls is not None:
            calls.append(("profiles_for", drawing_id, names, layout))
        return profiles(names) if callable(profiles) else profiles

    namespace = types.SimpleNamespace(profiles_for=profiles_for)

    if residual is not _UNDEFINED:

        def residual_for(drawing_id, use, tokens, classified_layers, *, layout=None):
            if calls is not None:
                calls.append(
                    (
                        "residual_for",
                        drawing_id,
                        use,
                        tuple(tokens),
                        tuple(classified_layers),
                        layout,
                    )
                )
            return residual(use) if callable(residual) else residual

        namespace.residual_for = residual_for

    return namespace


@pytest.fixture
def lane_a(monkeypatch):
    """Install a fake lane A. Returns the installer so a test can choose it."""

    def install(**kwargs):
        fake = _fake_dossier(**kwargs)
        monkeypatch.setattr(store_landuse, "_DOSSIER", fake, raising=False)
        return fake

    return install


# --- 1. `network` joins the role vocabulary ---------------------------------


def test_network_joins_the_role_vocabulary_without_moving_the_default():
    """The four roles that existed still exist, and `parcel` is still what an
    entry that names no role gets. Moving the default would silently
    re-classify every entry in every config in this repo."""
    assert {"parcel", "overlay", "annotation", "structure"} <= landuse.ROLES
    assert "network" in landuse.ROLES
    assert landuse.NETWORK_ROLE == "network"
    assert landuse.DEFAULT_ROLE == "parcel"
    assert landuse.PARCEL_ROLE == "parcel"


def test_a_network_layer_is_not_a_parcel_and_says_which_it_is():
    """`network` is not a loosening of `parcel`. It is a layer that HAS a
    measured size which is simply not a size in plots."""
    entry = landuse._build_land_use(
        "Road Centreline",
        {"use": "road", "role": "network",
         "sources": [{"origin": "layer_name", "observed": "Road Centreline",
                      "detail": "layer name"}]},
        config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
        config_version=1, where="test",
    )
    assert entry.role == "network"
    assert entry.is_network is True
    assert entry.is_parcel is False
    assert entry.as_row()["role"] == "network"

    plot = landuse.classify(JANADRIYAH, "VL2")
    assert plot.is_network is False
    assert plot.is_parcel is True


def test_a_role_outside_the_vocabulary_is_still_refused_after_network_joined():
    """Adding one role must not turn the check into a rubber stamp: `highway`
    is a word a config author would plausibly type, and it means nothing."""
    with pytest.raises(landuse.ConfigError) as caught:
        landuse._build_land_use(
            "L",
            {"use": "road", "role": "highway",
             "sources": [{"origin": "layer_name", "observed": "L",
                          "detail": "layer name"}]},
            config_layer=ev.ConfigLayer.DRAWING_OVERRIDE,
            config_version=1, where="test",
        )
    assert caught.value.code == "LAND_USE_UNKNOWN_ROLE"
    assert "network" in caught.value.hint


def test_a_network_layer_reports_its_length_instead_of_a_parcel_count():
    """The reason `network` was added. Every other non-parcel role answers "it
    adds nothing to the plot count", which is true and tells its reader
    nothing at all."""
    profiles = {ROAD_LAYER: store_landuse._profile_row(ROAD_LAYER, _road_profile())}
    got = store_landuse._network_keys(ROAD_LAYER, profiles=profiles, why=None)

    assert got["native_measure_status"]["status"] == "computed"
    assert got["native_measure"]["kind"] == "length"
    assert got["native_measure"]["value"] == pytest.approx(ROAD_LENGTH_M)
    assert got["native_measure"]["unit"] == "m"
    assert got["dossier_entities"] == ROAD_ENTITIES
    assert got["dossier_role"] == "network"
    # G8: the 42 that could not be measured are counted beside the total, not
    # summed in as zero and not dropped out of the denominator.
    assert got["native_measure"]["measured_entities"] == 1191
    assert got["native_measure"]["unmeasured_entities"] == 42
    assert "LENGTH" in got["native_measure_note"]


def test_a_network_layer_without_a_dossier_reports_absent_and_not_zero():
    """A length that quietly became 0.0 because no Dossier existed would be
    indistinguishable from a layer that really holds nothing (G8)."""
    got = store_landuse._network_keys(
        ROAD_LAYER, profiles={}, why="dossier_read is not available"
    )
    assert "native_measure" not in got
    assert got["native_measure_status"]["status"] == "not_computed"
    assert got["native_measure_status"]["why"] == "dossier_read is not available"


def test_the_config_and_the_geometry_are_allowed_to_disagree_out_loud():
    """The config says what a layer is FOR; the Dossier says what shape it IS.
    Different questions, so a disagreement is a finding to publish rather than
    a conflict to settle behind the reader's back."""
    profile = store_landuse._profile_row(
        SHEET_TEMPLATE_LAYER, _sheet_template_profile()
    )
    got = store_landuse._network_keys(
        SHEET_TEMPLATE_LAYER, profiles={SHEET_TEMPLATE_LAYER: profile}, why=None
    )
    assert "network" in got["role_disagreement"]
    assert "annotation" in got["role_disagreement"]


# --- 2. `unclassified_layers` gains profiles --------------------------------


def test_the_unclassified_profiles_tell_a_road_layer_from_a_sheet_grid(lane_a):
    """`count` and `sample` were a dead end: two names, and no way to tell
    1,233 road entities from an empty template. The profiles are what make the
    block's own note — "a layer that is unclassified is not a layer that is
    empty" — checkable rather than a disclaimer."""
    lane_a(profiles={SHEET_TEMPLATE_LAYER: _sheet_template_profile(),
                     ROAD_LAYER: _road_profile()})
    got = store_landuse._unclassified_profile_keys(
        JANADRIYAH, [ROAD_LAYER, SHEET_TEMPLATE_LAYER], layout=MODEL
    )

    assert got["profiles_status"]["status"] == "computed"
    assert got["profiles_status"]["why"] is None
    assert got["profiles_status"]["layers_the_dossier_knows"] == 2

    # Ranked by what was measured, not by what a layer is called.
    first, second = got["profiles"]
    assert first["layer"] == ROAD_LAYER
    assert first["role"] == "network"
    assert first["entities"] == ROAD_ENTITIES
    assert first["measure"]["value"] == pytest.approx(ROAD_LENGTH_M)
    assert first["measure"]["unit"] == "m"
    assert second["layer"] == SHEET_TEMPLATE_LAYER
    assert second["entities"] == 4
    assert second["measure"]["value"] is None
    # The one line that separates them at a glance.
    assert first["reads"] != second["reads"]
    assert "83962.565009 m" in first["reads"]


def test_an_unclassified_profile_carries_the_anomaly_that_would_triple_a_total():
    """83,962.565 m on the reference road layer is three copies of 27,987.52 m.
    A profile that offered the total without the flag would be handing over a
    number that is three times the truth."""
    row = store_landuse._profile_row(ROAD_LAYER, _road_profile())
    assert row["anomaly_kinds"] == ["duplicate_clusters"]
    assert "measure" in row["anomaly_note"]


def test_a_drawing_with_no_dossier_says_not_computed_rather_than_nothing(lane_a):
    """G8, and this campaign's whole thesis in one assertion: `None` from lane
    A means the Dossier has not been built, which is not the same answer as
    "these layers hold nothing"."""
    lane_a(profiles=None)
    got = store_landuse._unclassified_profile_keys(
        JANADRIYAH, [ROAD_LAYER], layout=MODEL
    )
    assert got["profiles"] == []
    assert got["profiles_status"]["status"] == "not_computed"
    assert "no Dossier" in got["profiles_status"]["why"]
    # Not `0`. Nobody counted, so there is no count to report.
    assert got["profiles_status"]["layers_the_dossier_knows"] is None
    assert got["profiles_status"]["layers_asked_about"] == 1


def test_a_dossier_that_knows_nothing_is_a_different_answer_from_no_dossier(lane_a):
    """Both produce an empty list and they must not read the same. One was
    asked and answered; the other was never asked."""
    lane_a(profiles={})
    answered = store_landuse._unclassified_profile_keys(
        JANADRIYAH, [ROAD_LAYER], layout=MODEL
    )
    lane_a(profiles=None)
    unasked = store_landuse._unclassified_profile_keys(
        JANADRIYAH, [ROAD_LAYER], layout=MODEL
    )

    assert answered["profiles"] == unasked["profiles"] == []
    assert answered["profiles_status"]["status"] == "computed"
    assert answered["profiles_status"]["layers_the_dossier_knows"] == 0
    assert unasked["profiles_status"]["status"] == "not_computed"


def test_a_missing_lane_a_module_degrades_and_names_itself(monkeypatch):
    """The module is written in parallel with this file, so its absence is a
    state this code has to ANSWER in, not a state it may crash in."""
    monkeypatch.setattr(store_landuse, "_DOSSIER", None, raising=False)
    monkeypatch.setattr(
        store_landuse, "_DOSSIER_WHY", "ModuleNotFoundError: dossier_read",
        raising=False,
    )
    got = store_landuse._unclassified_profile_keys(
        JANADRIYAH, [ROAD_LAYER], layout=MODEL
    )
    why = got["profiles_status"]["why"]
    assert got["profiles_status"]["status"] == "not_computed"
    assert "dossier_read" in why
    assert "ModuleNotFoundError" in why


def test_a_half_written_lane_a_reads_differently_from_an_absent_one(lane_a):
    """A module that exists and has not defined this function yet is a
    different situation from a module that is not there, and the reason string
    is where that difference is allowed to live."""
    fake = lane_a(profiles={})
    del fake.profiles_for
    got = store_landuse._unclassified_profile_keys(
        JANADRIYAH, [ROAD_LAYER], layout=MODEL
    )
    assert got["profiles_status"]["status"] == "not_computed"
    assert "does not define it" in got["profiles_status"]["why"]


def test_lane_a_blowing_up_does_not_take_the_land_use_summary_with_it(lane_a):
    def explode(_names):
        raise RuntimeError("no mongo here")

    lane_a(profiles=explode)
    got = store_landuse._unclassified_profile_keys(
        JANADRIYAH, [ROAD_LAYER], layout=MODEL
    )
    assert got["profiles_status"]["status"] == "not_computed"
    assert "RuntimeError" in got["profiles_status"]["why"]
    assert "no mongo here" in got["profiles_status"]["why"]


def test_the_profile_list_states_its_cap_and_counts_what_it_dropped(lane_a):
    """G7. This drawing declares 251 unclassified layers and a profile is a
    nested block rather than a name. What is dropped is counted and stated."""
    many = {
        f"LAYER-{i:03d}": {"role": "mixed", "entities": i, "measure": None}
        for i in range(30)
    }
    lane_a(profiles=many)
    got = store_landuse._unclassified_profile_keys(
        JANADRIYAH, sorted(many), layout=MODEL
    )
    cap = store_landuse.MAX_UNCLASSIFIED_PROFILES
    assert len(got["profiles"]) == cap
    assert got["profiles_status"]["cap"] == cap
    assert got["profiles_status"]["truncated"] is True
    assert got["profiles_status"]["layers_the_dossier_knows"] == 30
    assert got["profiles_status"]["layers_shown"] == cap
    # Biggest first, so the cap drops the tail rather than the answer.
    assert got["profiles"][0]["entities"] == 29


def test_a_layer_the_dossier_never_measured_is_absent_not_present_with_nulls(lane_a):
    """Lane A's contract. A row of nulls reads as "measured, and it is
    nothing"; absence reads as "not measured", which is what is true."""
    lane_a(profiles={ROAD_LAYER: _road_profile()})
    got = store_landuse._unclassified_profile_keys(
        JANADRIYAH, [ROAD_LAYER, "DIM"], layout=MODEL
    )
    assert [p["layer"] for p in got["profiles"]] == [ROAD_LAYER]
    assert got["profiles_status"]["layers_asked_about"] == 2
    assert got["profiles_status"]["layers_the_dossier_knows"] == 1


def test_every_published_measure_carries_a_unit_or_the_reason_it_has_none(lane_a):
    """G2. Eleven of the eighteen drawings here are in inches and three state
    nothing at all, so a number that quietly defaults to metres is wrong on
    most of this corpus without ever looking wrong."""
    unitless = {"kind": "length", "value": 12.5, "unit": None, "unit_reason": None}
    lane_a(profiles={
        ROAD_LAYER: _road_profile(),
        "NO-UNITS": {"role": "network", "entities": 3, "measure": unitless},
        "NO-MEASURE": {"role": "mixed", "entities": 1},
    })
    got = store_landuse._unclassified_profile_keys(
        JANADRIYAH, [ROAD_LAYER, "NO-UNITS", "NO-MEASURE"], layout=MODEL
    )
    for profile in got["profiles"]:
        measure = profile["measure"]
        assert measure["unit"] or measure["unit_reason"], profile["layer"]
    by_layer = {p["layer"]: p for p in got["profiles"]}
    assert by_layer["NO-UNITS"]["measure"]["unit_reason"]
    assert "no unit stated" in by_layer["NO-UNITS"]["reads"]


# --- 3. The no-dead-end-zero rule -------------------------------------------


def _residual_for_test(use="road", tokens=("road", "roads", "street"),
                       classified=("VL2", "Local Mosque")):
    return store_landuse._residual_keys(
        JANADRIYAH, use=use, tokens=tokens,
        classified_layers=classified, layout=MODEL,
    )


def test_a_zero_gains_a_residual_naming_what_the_drawing_actually_holds(lane_a):
    """THE fix this whole campaign was started for. "0 road parcels" was
    correct, and it was reported over 1,233 road entities and 84 km of
    centreline sitting in `unclassified_layers` as a name in a list."""
    lane_a(residual={"layers": [_sheet_template_profile(), _road_profile()]})
    got = _residual_for_test()

    assert got["residual_status"]["status"] == "computed"
    residual = got["residual"]
    assert residual["layers_matched"] == 2
    first = residual["layers"][0]
    assert first["layer"] == ROAD_LAYER
    assert first["role"] == "network"
    assert first["entities"] == ROAD_ENTITIES
    assert first["measure"]["value"] == pytest.approx(ROAD_LENGTH_M)
    assert first["measure"]["unit"] == "m"
    # ...and the sheet-layout layer matching the same word is visibly a
    # different kind of thing in the same list.
    second = residual["layers"][1]
    assert second["layer"] == SHEET_TEMPLATE_LAYER
    assert second["measure"]["value"] is None


def test_the_residual_ranks_by_geometry_and_not_by_the_order_it_was_given(lane_a):
    """The name proposes; the geometry disposes. Sixty `C-ROAD-*` layers match
    the word "road" on this drawing, so an unranked list buries the one layer
    that is a road."""
    lane_a(residual={"layers": [
        _sheet_template_profile(),
        {"layer": "C-ROAD-UNCOUNTED", "role": "mixed"},
        _road_profile(),
    ]})
    names = [row["layer"] for row in _residual_for_test()["residual"]["layers"]]
    assert names[0] == ROAD_LAYER
    # A layer the Dossier could not count sorts last, not as though it were
    # empty.
    assert names[-1] == "C-ROAD-UNCOUNTED"


def test_a_residual_of_a_drawing_with_no_dossier_is_absent_and_says_why(lane_a):
    """A zero row with no `residual` at all means the question was never
    asked — the Dossier has not been built. `residual_status` says so."""
    lane_a(residual=None)
    got = _residual_for_test()
    assert "residual" not in got
    assert got["residual_status"]["status"] == "not_computed"
    assert "no Dossier" in got["residual_status"]["why"]


def test_a_residual_that_matched_nothing_is_not_an_unasked_question(lane_a):
    """The two answers this campaign exists to keep apart, side by side."""
    lane_a(residual={"layers": []})
    answered = _residual_for_test()
    lane_a(residual=None)
    unasked = _residual_for_test()

    assert answered["residual"]["layers"] == []
    assert answered["residual"]["layers_matched"] == 0
    assert answered["residual_status"]["status"] == "computed"
    assert "residual" not in unasked
    assert unasked["residual_status"]["status"] == "not_computed"


def test_the_residual_degrades_when_lane_a_is_missing_entirely(monkeypatch):
    monkeypatch.setattr(store_landuse, "_DOSSIER", None, raising=False)
    monkeypatch.setattr(store_landuse, "_DOSSIER_WHY", None, raising=False)
    got = _residual_for_test()
    assert "residual" not in got
    assert "dossier_read" in got["residual_status"]["why"]


def test_the_residual_states_its_cap_and_counts_the_matches_it_dropped(lane_a):
    """G7, per `uses` row rather than per response: a summary can carry several
    zero rows and each of them brings a list."""
    lane_a(residual={"layers": [
        {"layer": f"C-ROAD-{i:02d}", "role": "annotation", "entities": i}
        for i in range(9)
    ]})
    residual = _residual_for_test()["residual"]
    cap = store_landuse.MAX_RESIDUAL_LAYERS
    assert residual["cap"] == cap
    assert len(residual["layers"]) == cap
    assert residual["layers_matched"] == 9
    assert residual["truncated"] is True


def test_the_residual_passes_the_use_and_its_config_tokens_and_maps_nothing(
    monkeypatch,
):
    """G1, and the decision that keeps this general. There is no table from
    land use to geometric role in this code and there must never be one: what
    goes to lane A is the use and the words `_patterns.yaml` already lists for
    it, so the seventeenth drawing needs a config line and not a code change."""
    calls: list = []
    monkeypatch.setattr(
        store_landuse, "_DOSSIER",
        _fake_dossier(residual={"layers": []}, calls=calls), raising=False,
    )
    tokens = landuse.patterns().tokens_for("road")
    assert tokens, "the road vocabulary is empty; this test would prove nothing"

    got = store_landuse._residual_keys(
        JANADRIYAH, use="road", tokens=tokens,
        classified_layers=("VL2",), layout=MODEL,
    )
    assert calls == [
        ("residual_for", JANADRIYAH, "road", tuple(tokens), ("VL2",), MODEL)
    ]
    assert got["residual"]["matched_on"] == list(tokens)
    assert "speaks_for" in got["residual_status"]["matched_on_source"]
    # `_patterns.yaml` deliberately has no road PATTERN — the vocabulary is a
    # different thing, and this is the route that works without one.
    assert landuse.patterns().first_match(ROAD_LAYER) is None


def test_a_use_with_no_vocabulary_tokens_says_so_instead_of_guessing(monkeypatch):
    """There is no hardcoded fallback behind the vocabulary. A use with no
    words to look for produces a named gap, not an invented match."""
    calls: list = []
    monkeypatch.setattr(
        store_landuse, "_DOSSIER",
        _fake_dossier(residual={"layers": [_road_profile()]}, calls=calls),
        raising=False,
    )
    got = _residual_for_test(use="road", tokens=())

    assert "residual" not in got
    assert got["residual_status"]["status"] == "not_computed"
    assert "vocabulary" in got["residual_status"]["why"]
    assert calls == [], "lane A was called for a use with nothing to match on"


def test_nothing_lane_a_said_is_dropped_in_silence(lane_a):
    """G7 in its other direction: a cap that silently ate a field lane A
    published would be the same defect in miniature."""
    lane_a(residual={
        "layers": [_road_profile()],
        "basis": "layer names tested with evidence.speaks_for",
        "sampled": False,
        "tokens_rejected": ["rd"],
    })
    residual = _residual_for_test()["residual"]
    assert residual["from_dossier"]["basis"]
    assert residual["from_dossier"]["sampled"] is False
    assert residual["from_dossier_keys_not_shown"] == ["tokens_rejected"]


def test_a_residual_list_under_an_unexpected_key_is_still_found_and_named(lane_a):
    """Lane A is written in parallel with this file; the contract fixes the
    SIGNATURE of `residual_for`, not the field names inside its block."""
    lane_a(residual={"candidates": [_road_profile()]})
    got = _residual_for_test()
    assert got["residual"]["layers"][0]["layer"] == ROAD_LAYER
    assert got["residual_status"]["dossier_list_key"] == "candidates"


# --- The additive-only contract ---------------------------------------------
#
# The viewer on 4310 and `web/app/api/agent/stream/route.ts` read this response
# BY KEY. A rename is invisible to pytest and fatal on screen, which is exactly
# why the old shape is pinned here by name AND by type rather than trusted.

OLD_TOP_LEVEL_KEYS: dict[str, type] = {
    "drawing_id": str,
    "layout": str,
    "scope_note": str,
    "land_use_available": bool,
    "config_version": int,
    "config_file": str,
    "config_file_expected_at": str,
    "config_layers_used": list,
    "config_problems": list,
    "vocabulary_problems": list,
    "uses": list,
    "open_questions": list,
    "roles_not_counted": list,
    "parcel_basis": str,
    "unclassified_layers": dict,
    "empty_but_declared": list,
    "empty_but_declared_facts": list,
    "empty_but_declared_note": str,
    "unverified_note": str,
    "inferred_note": str,
    "crs": dict,
    "area_unit": str,
    "area_scope": str,
    "limits": dict,
    "evidence": dict,
}

OLD_USES_ROW_KEYS: dict[str, type] = {
    "use": str,
    "parcels": int,
    "parcels_measured": int,
    "parcels_area_withheld": int,
    "total_area": dict,
    "layers": int,
    "layers_configured": int,
    "layers_with_no_entity_in_scope": list,
    "types_on_these_layers_not_counted_as_parcels": list,
    "by_layer": list,
    "by_layer_truncated": bool,
    "distinct_sources": list,
    "open_questions": list,
    "evidence": dict,
}

OLD_UNCLASSIFIED_KEYS: dict[str, type] = {
    "count": int,
    "count_basis": str,
    "in_scope_count": int,
    "in_scope_basis": str,
    "sample": list,
    "sample_truncated": bool,
    "note": str,
}

OLD_NOT_COUNTED_KEYS: dict[str, type] = {
    "layer": str,
    "land_use": str,
    "role": str,
    "config_layer": str,
    "entities": int,
    "why_not_counted": str,
    "grade": str,
    "verified": bool,
}

OLD_LIMITS_KEYS: dict[str, type] = {
    "layer_rows_per_use": int,
    "unclassified_sample": int,
}


def test_the_new_blocks_only_add_keys_and_never_shadow_an_old_one(lane_a):
    """Pure, so it is alive on every run rather than only when a database is.
    Every key these helpers emit is checked against the recorded shape of the
    row it is merged into."""
    lane_a(profiles={ROAD_LAYER: _road_profile()},
           residual={"layers": [_road_profile()]})
    residual_keys = set(_residual_for_test())
    profile_keys = set(
        store_landuse._unclassified_profile_keys(
            JANADRIYAH, [ROAD_LAYER], layout=MODEL
        )
    )
    network_keys = set(
        store_landuse._network_keys(
            ROAD_LAYER,
            profiles={
                ROAD_LAYER: store_landuse._profile_row(ROAD_LAYER, _road_profile())
            },
            why=None,
        )
    )

    assert residual_keys == {"residual", "residual_status"}
    assert residual_keys & set(OLD_USES_ROW_KEYS) == set()
    assert profile_keys & set(OLD_UNCLASSIFIED_KEYS) == set()
    assert network_keys & set(OLD_NOT_COUNTED_KEYS) == set()


def test_the_stated_caps_exist_and_are_the_ones_the_response_publishes():
    """G7 asks for a limit to be STATED, not merely applied."""
    assert store_landuse.MAX_UNCLASSIFIED_PROFILES > 0
    assert store_landuse.MAX_RESIDUAL_LAYERS > 0
    assert store_landuse.MAX_ANOMALY_KINDS > 0


# --- The whole response, assembled, with no database ------------------------
#
# The live checks above skip wherever there is no Mongo, and a lane whose only
# end-to-end proof skips on the machine it is written on is a lane that is not
# proven. The store reads of `land_use_summary` are two functions, so they are
# stood in for here and the ASSEMBLY -- which is what this lane changed -- runs
# for real: the config is loaded and validated, layers are classified, the
# aggregation is grouped, the residual is attached to the row that reports zero,
# and the evidence contract is audited on the result.


@pytest.fixture
def synthetic_summary(tmp_path, monkeypatch):
    """`land_use_summary` over a made-up drawing, with the store stood in for.

    The drawing is a miniature of the reference one: a road use whose only
    parcel layer is empty, a road CENTRELINE the config declares `role=network`,
    an unclassified road layer carrying the geometry, and a Civil3D sheet-layout
    layer that matches the same word and carries nothing.
    """
    drawing_id = "4444444444444444"
    (tmp_path / "_patterns.yaml").write_text(
        (landuse.CONFIG_DIR / landuse.PATTERNS_FILE).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / f"{drawing_id}.yaml").write_text(
        "version: 3\n"
        "layers:\n"
        '  "Road Reserve":\n'
        "    use: road\n"
        "    role: parcel\n"
        "    sources:\n"
        '      - {origin: layer_name, observed: "Road Reserve", detail: "layer name"}\n'
        '  "Road Centreline":\n'
        "    use: road\n"
        "    role: network\n"
        "    sources:\n"
        '      - {origin: layer_name, observed: "Road Centreline", detail: "layer name"}\n'
        '  "VL2":\n'
        "    use: residential\n"
        "    role: parcel\n"
        "    sources:\n"
        '      - {origin: layer_name, observed: "VL2", detail: "layer name"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(landuse, "CONFIG_DIR", tmp_path)
    landuse.clear_cache()

    drawing = {
        "_id": drawing_id,
        "units_name": "m",
        "layouts": [{"name": MODEL, "entity_count": 2477}],
        "layers": [
            {"name": "Road Reserve", "entity_count": 0},
            {"name": "Road Centreline", "entity_count": ROAD_ENTITIES},
            {"name": "VL2", "entity_count": 4},
            {"name": ROAD_LAYER, "entity_count": ROAD_ENTITIES},
            {"name": SHEET_TEMPLATE_LAYER, "entity_count": 4},
            {"name": "DIM", "entity_count": 3},
        ],
    }
    groups = [
        {"_id": {"layer": "VL2", "type": "LWPOLYLINE"},
         "n": 4, "n_measured": 4, "area_sum": 1200.0},
        {"_id": {"layer": "Road Centreline", "type": "LINE"},
         "n": ROAD_ENTITIES, "n_measured": 0, "area_sum": 0.0},
        {"_id": {"layer": ROAD_LAYER, "type": "LINE"},
         "n": ROAD_ENTITIES, "n_measured": 0, "area_sum": 0.0},
        {"_id": {"layer": SHEET_TEMPLATE_LAYER, "type": "TEXT"},
         "n": 4, "n_measured": 0, "area_sum": 0.0},
        {"_id": {"layer": "DIM", "type": "DIMENSION"},
         "n": 3, "n_measured": 0, "area_sum": 0.0},
    ]
    monkeypatch.setattr(
        store_landuse, "_drawing",
        lambda did: drawing if did == drawing_id else None,
    )
    monkeypatch.setattr(store_landuse, "_groups_in_scope", lambda did, lay: groups)

    known = {
        ROAD_LAYER: _road_profile(),
        SHEET_TEMPLATE_LAYER: _sheet_template_profile(),
        "Road Centreline": dict(_road_profile(), layer="Road Centreline"),
    }
    monkeypatch.setattr(
        store_landuse, "_DOSSIER",
        _fake_dossier(
            profiles=lambda names: {n: known[n] for n in names if n in known},
            residual=lambda use: (
                {"layers": [_sheet_template_profile(), _road_profile()],
                 "basis": "layer names tested with evidence.speaks_for"}
                if use == "road"
                else {"layers": []}
            ),
        ),
        raising=False,
    )
    return store_landuse.land_use_summary(drawing_id, MODEL)


def test_a_zero_road_row_arrives_with_the_road_geometry_beside_it(synthetic_summary):
    """The Definition of Done, assembled through the real function. "0 road
    parcels" is still the right count of road PARCELS, and it now arrives with
    the 1,233 entities and the 83,962.565 m that were always in the file."""
    road = _row(synthetic_summary, "road")
    assert road["parcels"] == 0
    assert road["residual_status"]["status"] == "computed"

    layers = road["residual"]["layers"]
    assert layers[0]["layer"] == ROAD_LAYER
    assert layers[0]["role"] == "network"
    assert layers[0]["entities"] == ROAD_ENTITIES
    assert layers[0]["measure"]["value"] == pytest.approx(ROAD_LENGTH_M)
    assert layers[0]["measure"]["unit"] == "m"
    # ...and the sheet-layout layer that matches the same word is in the same
    # list, visibly a different kind of thing.
    assert layers[1]["layer"] == SHEET_TEMPLATE_LAYER
    assert layers[1]["measure"]["value"] is None
    assert "not an absence" in synthetic_summary["residual_note"]


def test_a_row_that_has_parcels_is_not_given_a_residual(synthetic_summary):
    """A row with parcels is already an answer, and spending the response's
    size budget restating it would make the block noise (G7)."""
    residential = _row(synthetic_summary, "residential")
    assert residential["parcels"] == 4
    assert "residual" not in residential
    assert "residual_status" not in residential


def test_the_network_layer_of_a_zero_row_reports_length_in_the_whole_response(
    synthetic_summary,
):
    """`Road Centreline` is `role=network`, so it is not a parcel and never
    will be. Under the four old roles the only thing the response could say
    about it was that it adds nothing to the plot count."""
    rows = {r["layer"]: r for r in synthetic_summary["roles_not_counted"]}
    centreline = rows["Road Centreline"]
    assert centreline["role"] == "network"
    assert centreline["native_measure_status"]["status"] == "computed"
    assert centreline["native_measure"]["kind"] == "length"
    assert centreline["native_measure"]["value"] == pytest.approx(ROAD_LENGTH_M)
    assert centreline["native_measure"]["unit"] == "m"
    # The sentence beside every parcel figure now names this exclusion too.
    assert "network" in synthetic_summary["parcel_basis"]


def test_the_whole_response_still_passes_the_evidence_audit_with_the_new_blocks(
    synthetic_summary,
):
    """The new blocks are nested inside `uses` rows, inside
    `roles_not_counted` rows and inside `unclassified_layers`, so none of them
    is a top-level meaning-bearing key needing evidence of its own."""
    assert ev.audit_response(synthetic_summary) == []
    assert synthetic_summary["scope_note"]


def test_the_assembled_response_keeps_every_old_key_and_type(synthetic_summary):
    """Additive-only, proven on a response this lane assembled end to end
    rather than only on the helpers in isolation."""
    for key, kind in OLD_TOP_LEVEL_KEYS.items():
        assert key in synthetic_summary, key
        assert isinstance(synthetic_summary[key], kind), key
    for key, kind in OLD_LIMITS_KEYS.items():
        assert isinstance(synthetic_summary["limits"][key], kind), key
    for row in synthetic_summary["uses"]:
        for key, kind in OLD_USES_ROW_KEYS.items():
            assert isinstance(row[key], kind), (row["use"], key)
    for key, kind in OLD_UNCLASSIFIED_KEYS.items():
        assert isinstance(synthetic_summary["unclassified_layers"][key], kind), key
    for row in synthetic_summary["roles_not_counted"]:
        for key, kind in OLD_NOT_COUNTED_KEYS.items():
            assert isinstance(row[key], kind), (row["layer"], key)


def test_the_unclassified_block_profiles_the_layers_it_samples(synthetic_summary):
    """`sample` and `profiles` describe the same population, so the two can be
    read together instead of against each other."""
    block = synthetic_summary["unclassified_layers"]
    assert ROAD_LAYER in block["sample"]
    profiled = {p["layer"]: p for p in block["profiles"]}
    assert profiled[ROAD_LAYER]["entities"] == ROAD_ENTITIES
    assert "83962.565009 m" in profiled[ROAD_LAYER]["reads"]
    assert profiled[SHEET_TEMPLATE_LAYER]["measure"]["value"] is None
    # `DIM` is unclassified too and the Dossier has no entry for it, so it is
    # absent from the profiles rather than present as a row of nulls.
    assert "DIM" in block["sample"]
    assert "DIM" not in profiled


def test_the_assembled_response_degrades_without_lane_a(
    synthetic_summary, monkeypatch
):
    """The same drawing, with lane A gone. Every zero still says something —
    it just says "not computed" instead of naming layers, and it never says
    nothing at all."""
    monkeypatch.setattr(store_landuse, "_DOSSIER", None, raising=False)
    degraded = store_landuse.land_use_summary("4444444444444444", MODEL)

    road = _row(degraded, "road")
    assert road["parcels"] == 0
    assert "residual" not in road
    assert road["residual_status"]["status"] == "not_computed"
    assert "dossier_read" in road["residual_status"]["why"]

    block = degraded["unclassified_layers"]
    assert block["profiles"] == []
    assert block["profiles_status"]["status"] == "not_computed"
    assert block["count"] == synthetic_summary["unclassified_layers"]["count"]

    centreline = {r["layer"]: r for r in degraded["roles_not_counted"]}[
        "Road Centreline"
    ]
    assert "native_measure" not in centreline
    assert centreline["native_measure_status"]["status"] == "not_computed"
    assert ev.audit_response(degraded) == []


def test_the_pre_existing_response_keys_all_survive_with_their_old_types(summary):
    """The additive-only rule, checked against the whole live response rather
    than only against the blocks this lane touched."""
    for key, kind in OLD_TOP_LEVEL_KEYS.items():
        assert key in summary, key
        assert isinstance(summary[key], kind), key
    assert summary["area_unit_reason"] is None

    for key, kind in OLD_LIMITS_KEYS.items():
        assert isinstance(summary["limits"][key], kind), key

    for row in summary["uses"]:
        for key, kind in OLD_USES_ROW_KEYS.items():
            assert key in row, (row.get("use"), key)
            assert isinstance(row[key], kind), (row.get("use"), key)

    block = summary["unclassified_layers"]
    for key, kind in OLD_UNCLASSIFIED_KEYS.items():
        assert key in block, key
        assert isinstance(block[key], kind), key

    for row in summary["roles_not_counted"]:
        for key, kind in OLD_NOT_COUNTED_KEYS.items():
            assert key in row, (row.get("layer"), key)
            assert isinstance(row[key], kind), (row.get("layer"), key)


def test_the_live_response_carries_the_profiles_block(summary):
    block = summary["unclassified_layers"]
    assert isinstance(block["profiles"], list)
    assert len(block["profiles"]) <= store_landuse.MAX_UNCLASSIFIED_PROFILES
    status = block["profiles_status"]
    assert status["status"] in {"computed", "not_computed"}
    if status["status"] == "not_computed":
        assert status["why"], "an uncomputed block must name its reason"
    else:
        assert status["why"] is None
    assert block["profiles_note"]
    assert summary["residual_note"]


def test_every_live_zero_parcel_row_is_asked_the_residual_question(summary):
    """The no-dead-end-zero rule as the response actually leaves the API. A
    zero that carries neither a residual nor a reason is the dead end."""
    for row in summary["uses"]:
        if row["parcels"]:
            assert "residual" not in row, row["use"]
            assert "residual_status" not in row, row["use"]
            continue
        status = row["residual_status"]
        assert status["status"] in {"computed", "not_computed"}, row["use"]
        assert status["use"] == row["use"]
        if status["status"] == "computed":
            assert "residual" in row, row["use"]
            assert isinstance(row["residual"]["layers"], list)
        else:
            assert status["why"], row["use"]
            assert "residual" not in row, row["use"]


def test_the_live_response_still_passes_the_evidence_contract_audit(summary):
    """The new blocks are nested inside `uses` rows and inside
    `unclassified_layers`, so none of them is a top-level meaning-bearing key
    that would need evidence of its own — and this is what proves it."""
    assert ev.audit_response(summary) == []


def test_a_network_layer_in_the_live_response_carries_its_length(summary):
    """Any layer this drawing's config declares `role=network` reports a
    measured length or names the reason it has none. It never reports a parcel
    count of zero and stops there."""
    rows = [r for r in summary["roles_not_counted"] if r["role"] == "network"]
    if not rows:
        pytest.skip("this drawing's config declares no role=network layer")
    for row in rows:
        status = row["native_measure_status"]
        assert status["status"] in {"computed", "not_computed"}, row["layer"]
        if status["status"] == "computed":
            measure = row["native_measure"]
            assert measure["unit"] or measure["unit_reason"], row["layer"]
        else:
            assert status["why"], row["layer"]
            assert "native_measure" not in row, row["layer"]
