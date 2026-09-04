"""The UPLIFT-08 evidence contract: a grade is computed, never typed.

Two kinds of test in this file, and both are required:

- **Number tests** — the Janadriyah cases that UPLIFT-08 names itself: the
  mosque `corroborated`, the houses `inferred`, the meaning of the VL code
  `unknown`.
- **Invariant tests** — properties that must hold for any drawing, including
  the 17th drawing whose layer conventions nobody has ever seen. The Janadriyah
  numbers will never catch a break there.

What is tested here is the shape of the contract, not the data; the `evidence`
module is deliberately pure, so this whole file runs without MongoDB.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import evidence as ev  # noqa: E402


# --- tooling -----------------------------------------------------------------


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

DRAWING = "596212db022a3397"


def scope(units=None) -> ev.Scope:
    return ev.Scope(
        layout="Model",
        includes_block_definitions=False,
        units=units or METRE_UNITS,
        note="only entities on layout Model; block definitions are not included",
    )


def override() -> ev.Provenance:
    return ev.Provenance(config_layer=ev.ConfigLayer.DRAWING_OVERRIDE, config_version=1)


def pattern() -> ev.Provenance:
    return ev.Provenance(config_layer=ev.ConfigLayer.GLOBAL_PATTERN, config_version=1)


RELIGIOUS = ev.Claim(value="religious", tokens=("mosque", "masjid", "مسجد"))
RESIDENTIAL = ev.Claim(
    value="residential", tokens=("residential", "villa", "house", "dwelling")
)


# --- N1-N5: the grades for the cases UPLIFT-08 names --------------------------


def test_mosque_is_corroborated_by_two_independent_corpora():
    """The layer name AND a label inside its parcel: two corpora, two places."""
    e = ev.Evidence.of(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        observations=[
            ev.Observation(
                origin=ev.Origin.LAYER_NAME,
                detail="layer 'Jumaa Mosque'",
                observed="Jumaa Mosque",
                locator="Jumaa Mosque",
            ),
            ev.Observation(
                origin=ev.Origin.ENTITY_TEXT,
                detail="TEXT inside the parcel, point-in-polygon on the insertion point",
                observed="مسجد جامع",
                observed_raw="ls{] {hlU",
                locator="70C91A4",
                reading=ev.Reading(
                    verbatim=False, method="shx-map:xarb.shx", corpus_verified=True
                ),
            ),
        ],
    )
    assert e.grade is ev.Grade.CORROBORATED
    assert e.independent_corpora == frozenset(
        {ev.Corpus.NAME_STRING, ev.Corpus.ANNOTATION}
    )


def test_residential_stays_inferred_however_many_bases_agree():
    """Four bases that agree are still not corroboration.

    The plot module and the BlockBoundary tiling are one act of drawing read
    two ways; elimination infers; and the layer name is a code that contains no
    word at all for its claim.
    """
    e = ev.Evidence.of(
        RESIDENTIAL,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        observations=[
            ev.Observation(
                origin=ev.Origin.GEOMETRY, detail="plot module 12x25 on 13 layers"
            ),
            ev.Observation(
                origin=ev.Origin.TOPOLOGY,
                detail="2,380 plots fill exactly 197 BlockBoundary polygons",
            ),
            ev.Observation(
                origin=ev.Origin.LAYER_NAME,
                detail="layer 'DP4 ZEROLOT'",
                observed="DP4 ZEROLOT",
                locator="DP4 ZEROLOT",
            ),
            ev.Observation(
                origin=ev.Origin.ABSENCE,
                detail="18 non-residential land uses have an explicit layer name",
            ),
        ],
        not_established="the file never states a residential land use",
        how_to_verify="ask Roshn for the typology key",
    )
    assert e.grade is ev.Grade.INFERRED
    assert e.independent_corpora == frozenset()


def test_two_layer_names_never_reach_corroborated():
    """The rule UPLIFT-08 names itself: SchoolHatch + Primary School."""
    education = ev.Claim(value="education", tokens=("school",))
    e = ev.Evidence.of(
        education,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        observations=[
            ev.Observation(
                origin=ev.Origin.LAYER_NAME,
                detail="layer 'SchoolHatch'",
                observed="SchoolHatch",
                locator="SchoolHatch",
            ),
            ev.Observation(
                origin=ev.Origin.LAYER_NAME,
                detail="layer 'Primary School'",
                observed="Primary School",
                locator="Primary School",
            ),
        ],
    )
    assert e.grade is ev.Grade.STATED
    assert len(e.independent_corpora) == 1


def test_a_layer_name_and_a_block_name_are_one_corpus():
    """Naming a layer and naming a block are one naming habit.

    This is what distinguishes the corpus axis from the origin axis: splitting
    the layer table and the block table into two origins would raise this pair
    to corroborated, and that is exactly the shape of mistake that is
    forbidden.
    """
    e = ev.Evidence.of(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        observations=[
            ev.Observation(
                origin=ev.Origin.LAYER_NAME,
                detail="layer 'Mosque'",
                observed="Mosque",
                locator="Mosque",
            ),
            ev.Observation(
                origin=ev.Origin.BLOCK_NAME,
                detail="block 'Mosque'",
                observed="Mosque",
                locator="BLK-Mosque",
            ),
        ],
    )
    assert e.grade is ev.Grade.STATED


def test_one_entity_read_twice_is_one_source():
    """Two different corpora but the same place do not become corroboration."""
    e = ev.Evidence.of(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        observations=[
            ev.Observation(
                origin=ev.Origin.ENTITY_TEXT,
                detail="TEXT 70C91A4",
                observed="masjid",
                locator="70C91A4",
            ),
            ev.Observation(
                origin=ev.Origin.BLOCK_ATTRIBUTE,
                detail="an attribute on the same entity",
                observed="masjid",
                locator="70C91A4",
            ),
        ],
    )
    assert e.grade is ev.Grade.STATED


# --- N6: an "I don't know" that is useful ------------------------------------


def test_unknown_must_say_what_is_missing_and_where_to_ask():
    with pytest.raises(ev.EvidenceError) as exc:
        ev.Evidence.unknown(
            ev.Claim(value="typology:VL2"),
            drawing_id=DRAWING,
            scope=scope(),
            provenance=override(),
            not_established="",
            how_to_verify="",
        )
    assert exc.value.code == "UNKNOWN_WITHOUT_GAP"


def test_unknown_statement_never_offers_a_candidate_answer():
    """E1 and E8: an answer whose negation can simply be deleted is bait."""
    e = ev.Evidence.unknown(
        ev.Claim(value="the meaning of the layer code 'VL'"),
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        not_established="what 'VL' stands for is not stated in this file",
        how_to_verify="ask Roshn for the typology key",
    )
    assert e.grade is ev.Grade.UNKNOWN
    said = e.statement().casefold()
    for bait in ("villa", "usually", "probably", "likely means"):
        assert bait not in said, f"the statement offers a candidate: {bait!r}"


# --- N7-N8: the literal test -------------------------------------------------


def test_a_word_boundary_stops_car_parking_from_stating_park():
    assert ev.speaks_for("CAR PARKING AREA", ("park",)) is None
    assert ev.speaks_for("Central Park", ("park",)) == "park"


def test_an_underscore_separates_words_like_every_layer_convention_says():
    """Found when UPLIFT-02 landed, on a real layer in the reference drawing.

    `_` is a WORD character to the regex engine, so the word boundary is never
    found in `00_Education` and a layer that plainly names its land use reads
    as naming nothing. AIA/NCS and ISO 13567 both use the underscore as a field
    separator.
    """
    assert ev.speaks_for("CS-Land use-00_Education", ("education",)) == "education"
    assert ev.speaks_for("A-FLOR_MOSQUE_01", ("mosque",)) == "mosque"
    # And it does not loosen the word boundary for anything else.
    assert ev.speaks_for("CAR_PARKING_AREA", ("park",)) is None


def test_non_latin_tokens_match_by_containment():
    assert ev.speaks_for("مسجد جامع", ("مسجد",)) == "مسجد"


# --- N9: SHX readings --------------------------------------------------------


def test_an_unverified_shx_reading_cannot_corroborate():
    """Corroboration on top of guessed glyphs stays impossible.

    A reading whose raw string is not in the UPLIFT-04 verified corpus is
    ceilinged to inferred, so it cannot become a second corpus.
    """
    guessed = ev.Reading(
        verbatim=False, method="shx-map:unverified", corpus_verified=False
    )
    assert guessed.ceiling is ev.Grade.INFERRED
    e = ev.Evidence.of(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        observations=[
            ev.Observation(
                origin=ev.Origin.LAYER_NAME,
                detail="layer 'Jumaa Mosque'",
                observed="Jumaa Mosque",
                locator="Jumaa Mosque",
            ),
            ev.Observation(
                origin=ev.Origin.ENTITY_TEXT,
                detail="an SHX reading that is not yet verified",
                observed="مسجد",
                locator="70C91A4",
                reading=guessed,
            ),
        ],
    )
    assert e.grade is ev.Grade.STATED


# --- N10-N11: quantities -----------------------------------------------------


def test_a_withheld_area_can_never_become_zero():
    """A HATCH carries no area; 0 m2 is the WRONG number, not an empty one."""
    q = ev.Quantity.withheld(
        basis="8 HATCHes on layer SchoolHatch",
        method="not measured",
        reason="a HATCH carries no area in this store",
    )
    row = q.as_dict()
    assert row["value"] is None
    assert row["not_zero"] is True
    assert row["withheld_reason"]

    with pytest.raises(TypeError):
        float(q)
    with pytest.raises(ev.EvidenceError) as exc:
        sum([q])
    assert exc.value.code == "QUANTITY_NOT_A_NUMBER"


def test_withheld_is_contagious_through_addition():
    good = ev.Quantity.measured(
        basis="parcels", method="shoelace", value=10.0, unit="m2"
    )
    bad = ev.Quantity.withheld(basis="hatch", method="not measured", reason="HATCH")
    assert (good + bad).is_withheld
    assert (bad + good).is_withheld


def test_a_quantity_without_a_method_is_refused():
    """E5: a distance that does not say it is a straight line reads as a walking
    distance."""
    with pytest.raises(ev.EvidenceError) as exc:
        ev.Quantity.measured(basis="distance", method="", value=1.0, unit="m")
    assert exc.value.code == "QUANTITY_WITHOUT_METHOD"


def test_total_says_how_much_it_could_not_measure():
    rows = [
        ev.Quantity.measured(basis="a", method="shoelace", value=3.0, unit="m2"),
        ev.Quantity.withheld(basis="b", method="not measured", reason="HATCH"),
    ]
    t = ev.total(rows, basis="area", method="shoelace")
    assert "1 measured" in t.basis and "1 withheld" in t.basis


# --- N14-N17: a grade can only go down ---------------------------------------


def test_the_module_offers_no_way_to_raise_a_grade():
    """E8: this contract has to withstand a request to break it."""
    forbidden = {"override_grade", "assume", "confidence", "strongest", "max_grade"}
    assert forbidden.isdisjoint(dir(ev))
    assert forbidden.isdisjoint(dir(ev.Evidence))
    assert "_stronger" not in dir(ev)


def test_a_ceiling_can_only_be_lowered():
    e = ev.Evidence.of(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        observations=[
            ev.Observation(
                origin=ev.Origin.LAYER_NAME,
                detail="layer 'Jumaa Mosque'",
                observed="Jumaa Mosque",
                locator="Jumaa Mosque",
            )
        ],
    )
    assert e.with_ceiling(ev.Grade.INFERRED).grade is ev.Grade.INFERRED
    lowered = e.with_ceiling(ev.Grade.INFERRED)
    with pytest.raises(ev.EvidenceError) as exc:
        lowered.with_ceiling(ev.Grade.CORROBORATED)
    assert exc.value.code == "CEILING_WOULD_RAISE"


def test_unresolved_forces_unknown_whatever_the_sources_say():
    e = ev.Evidence.for_layer(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        layer_name="Jumaa Mosque",
    )
    assert e.grade is ev.Grade.STATED
    assert e.unresolved("its config was withdrawn", "ask the contractor").grade is (
        ev.Grade.UNKNOWN
    )


def test_evidence_never_crosses_drawings():
    a = ev.Evidence.for_layer(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        layer_name="Jumaa Mosque",
    )
    b = ev.Evidence.for_layer(
        RELIGIOUS,
        drawing_id="9c2f8a1e",
        scope=scope(),
        provenance=override(),
        layer_name="Jumaa Mosque",
    )
    with pytest.raises(ev.EvidenceError) as exc:
        a.combined_with(b)
    assert exc.value.code == "CROSS_DRAWING"


def test_a_pattern_match_can_never_report_itself_verified():
    """G4, enforced by structure: `verified` is derived, it cannot be typed."""
    assert pattern().verified is False
    assert override().verified is True
    with pytest.raises(Exception):
        pattern().verified = True  # type: ignore[misc]


# --- N18: the vocabulary comes from config ----------------------------------


def test_a_code_shaped_token_is_dropped_and_reported_not_raised():
    """One line of YAML must not raise 2,380 plots to `stated`.

    And a wrong config must not become an outage: D-067 records a hard failure
    at startup that left cad-api never healthy.
    """
    vocab, problems = ev.load_vocabulary(
        {"residential": {"tokens": ["vl", "vl2", "abc", "residential", "villa"]}}
    )
    assert vocab["residential"] == ("residential", "villa")
    assert len(problems) == 3


def test_a_claim_with_no_tokens_is_legal_and_can_never_be_stated():
    e = ev.Evidence.of(
        ev.Claim(value="typology:VL2"),
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        observations=[
            ev.Observation(
                origin=ev.Origin.LAYER_NAME,
                detail="layer 'VL2'",
                observed="VL2",
                locator="VL2",
            )
        ],
    )
    assert e.grade is ev.Grade.INFERRED


# --- aggregation -------------------------------------------------------------


def test_an_aggregate_carries_its_spread_not_only_its_floor():
    """Weakest-wins on its own destroys the signal; the spread must travel."""
    strong = ev.Evidence.for_layer(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        layer_name="Jumaa Mosque",
    )
    weak = ev.Evidence.of(
        RESIDENTIAL,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        observations=[
            ev.Observation(origin=ev.Origin.GEOMETRY, detail="plot module 12x25")
        ],
    )
    agg = ev.weakest([strong, strong, weak], ev.Claim(value="land use"))
    assert agg.grade is ev.Grade.INFERRED
    body = agg.to_json()
    assert body["grade_spread"]["stated"] == 2
    assert body["grade_spread"]["inferred"] == 1


# --- shape invariants --------------------------------------------------------


def test_no_sources_means_unknown_and_never_the_other_way_round():
    e = ev.Evidence.of(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        observations=[],
    )
    assert e.grade is ev.Grade.UNKNOWN


def test_the_contract_keys_are_always_emitted_even_when_null():
    """A missing `not_established` must read as a bug, not as a claim that the
    answer has no limits."""
    e = ev.Evidence.for_layer(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        layer_name="Jumaa Mosque",
    )
    body = e.to_json()
    for key in ("grade", "claim", "not_established", "how_to_verify", "scope"):
        assert key in body
    assert body["not_established"] is None


def test_scope_cannot_be_built_without_units():
    """G2: 11 of the 16 drawings are in inches, 3 state no unit."""
    with pytest.raises(ev.EvidenceError) as exc:
        ev.Scope(
            layout="Model",
            includes_block_definitions=False,
            units={"name": "m"},
            note="x",
        )
    assert exc.value.code == "SCOPE_WITHOUT_UNITS"


def test_every_origin_has_a_corpus_decision():
    """A new Origin without a corpus decision would count as independent."""
    assert set(ev.CORPUS_OF) == set(ev.Origin)


def test_the_module_holds_no_drawing_specific_constant():
    """G1, actually run against the file itself."""
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "app" / "evidence.py"
    ).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    banned = re.compile(
        r"\b(VL2|DP4|LP1|TH3|ZEROLOT|SchoolHatch|BlockBoundary|32638)\b"
    )
    assert not banned.search(code), banned.search(code).group(0)  # type: ignore[union-attr]


# --- the seam ----------------------------------------------------------------


def test_meaning_cannot_be_published_without_a_scope():
    e = ev.Evidence.for_layer(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        layer_name="Jumaa Mosque",
    )
    with pytest.raises(ev.EvidenceError) as exc:
        ev.attach({"drawing_id": DRAWING, "land_use": "religious"}, e)
    assert exc.value.code == "MEANING_WITHOUT_SCOPE"


def test_the_old_meaning_of_the_key_explodes_in_a_test_not_in_an_answer():
    e = ev.Evidence.for_layer(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        layer_name="Jumaa Mosque",
    )
    payload = {
        "drawing_id": DRAWING,
        "scope_note": "layout Model",
        "evidence": {"layer_did_you_mean": ["Green"]},
    }
    with pytest.raises(ev.EvidenceError) as exc:
        ev.attach(payload, e)
    assert exc.value.code == "EVIDENCE_KEY_TAKEN"


def test_forward_carries_the_contract_across_the_mcp_seam():
    src = {
        "evidence": {"grade": "stated"},
        "scope_note": "layout Model",
        "why_empty_facts": {"layouts": {}},
        "ignored": 1,
    }
    out = ev.forward(src, {})
    assert set(out) == {"evidence", "scope_note", "why_empty_facts"}


def test_audit_catches_a_bare_zero_where_a_quantity_belongs():
    problems = ev.audit_response(
        {"scope_note": "layout Model", "land_use": "education", "total_area": 0.0}
    )
    assert any(p.startswith("V4") for p in problems)
    assert any(p.startswith("V1") for p in problems)


def test_audit_catches_a_unit_on_a_drawing_that_declares_none():
    """G2: a unit leaking onto a drawing that does not state one."""
    e = ev.Evidence.for_layer(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(UNITLESS),
        provenance=override(),
        layer_name="Jumaa Mosque",
    )
    payload = {"drawing_id": DRAWING, "scope_note": "layout Model"}
    ev.attach(payload, e)
    payload["total_area"] = {
        "value": 10.0,
        "unit": "m2",
        "basis": "parcels",
        "method": "shoelace",
    }
    problems = ev.audit_response(payload)
    assert any(p.startswith("V6") for p in problems)


def test_a_clean_response_audits_clean():
    e = ev.Evidence.for_layer(
        RELIGIOUS,
        drawing_id=DRAWING,
        scope=scope(),
        provenance=override(),
        layer_name="Jumaa Mosque",
    )
    payload: dict = {"drawing_id": DRAWING, "scope_note": "layout Model"}
    ev.attach(payload, e)
    payload["land_use"] = "religious"
    payload["total_area"] = ev.Quantity.measured(
        basis="parcels", method="shoelace", value=16728.46, unit="m2"
    ).as_dict()
    assert ev.audit_response(payload) == []
