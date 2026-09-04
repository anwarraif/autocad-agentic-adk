"""Naming what is inside a bucket — DOSSIER Phase 1, lane 4.

Two functions, both pure over rows that somebody else fetched, so every test
here is a plain list of dicts and there is no MongoDB anywhere in the file.

What is guarded is not "does it count" — counting is easy. It is:

* that a points bucket says **what** was placed, not merely how many;
* that an annotation bucket says **what the text says**, with its numeric
  range, its Arabic reading, and a sample of the leftovers;
* that the classification never once looks at the layer name (rule G1) — the
  same rows are run under two contradictory layer names and must classify
  identically;
* that the shape cannot be flattened into "2 plots numbered 2092". The label
  `2092` exists twice in the reference drawing, once in `Model` and once
  inside the block definition `[block] shml`, and that is one label placed
  once, not two plots.
"""

from __future__ import annotations

import pytest

from app import dossier_census as census
from app import shx


# --- fixtures, hand-written -------------------------------------------------


def text_row(handle, text, *, layout="Model", layer="LABELS", type="TEXT", **extra):
    row = {
        "handle": handle,
        "text": text,
        "layout": layout,
        "layer": layer,
        "type": type,
    }
    row.update(extra)
    return row


def insert_row(handle, block_name, *, layout="Model", layer="SYMBOLS", **extra):
    row = {
        "handle": handle,
        "block_name": block_name,
        "layout": layout,
        "layer": layer,
        "type": "INSERT",
    }
    row.update(extra)
    return row


def class_of(result, kind):
    for item in result["classes"]:
        if item["kind"] == kind:
            return item
    return None


# --- annotation census: numbers become an answer ----------------------------


def test_numeric_labels_report_their_range_not_just_a_count():
    """"2,449 texts" is not an answer. "labels running 1990-2110" is."""
    rows = [
        text_row("A1", "1990"),
        text_row("A2", "2092"),
        text_row("A3", "2110"),
    ]
    out = census.annotation_census(rows)

    numeric = class_of(out, census.KIND_NUMERIC)
    assert numeric["count"] == 3
    assert numeric["range"]["min"] == 1990.0
    assert numeric["range"]["max"] == 2110.0
    assert numeric["range"]["digits_min"] == 4
    assert numeric["range"]["digits_max"] == 4
    assert numeric["distinct_texts"] == 3
    assert out["total"] == 3
    assert out["classified"] == 3


def test_a_numeric_range_never_invents_a_unit():
    """Rule G2: the census is not told the drawing's unit, so it states none."""
    out = census.annotation_census([text_row("A1", "12.00")])
    dimension = class_of(out, census.KIND_DIMENSION)
    assert dimension["range"]["unit"] is None
    assert "G2" in dimension["range"]["unit_basis"]


def test_thousands_separators_are_still_one_integer():
    out = census.annotation_census([text_row("A1", "1,234")])
    numeric = class_of(out, census.KIND_NUMERIC)
    assert numeric["count"] == 1
    assert numeric["range"]["min"] == 1234.0
    assert numeric["range"]["digits_min"] == 4


# --- annotation census: dimension values ------------------------------------


def test_dimension_values_are_their_own_class_with_a_range():
    rows = [
        text_row("D1", "12.00"),
        text_row("D2", "25.00"),
        text_row("D3", "25.00"),
    ]
    out = census.annotation_census(rows)

    dimension = class_of(out, census.KIND_DIMENSION)
    assert dimension["count"] == 3
    assert dimension["distinct_texts"] == 2
    assert dimension["range"]["min"] == 12.0
    assert dimension["range"]["max"] == 25.0
    assert class_of(out, census.KIND_NUMERIC) is None


def test_a_dimension_has_no_digit_width_because_the_figure_would_mislead():
    """`12.00` has four digits and the fact means nothing.

    An unlabelled figure is how the wrong reading gets made; the digit width
    is published for labels, where "four-digit labels running 1990-2110" is an
    answer, and left null everywhere else.
    """
    out = census.annotation_census([text_row("D1", "12.00"), text_row("A1", "2092")])
    assert class_of(out, census.KIND_DIMENSION)["range"]["digits_min"] is None
    assert class_of(out, census.KIND_DIMENSION)["range"]["digits_max"] is None
    assert class_of(out, census.KIND_NUMERIC)["range"]["digits_min"] == 4


@pytest.mark.parametrize(
    "raw",
    ["12.00", "25.00", "-3.5", "25m", "1200mm", "R12", "12.00 x 25.00", "12.5+/-0.2"],
)
def test_measurement_shapes_read_as_dimension_values(raw):
    assert census.classify_text(raw)[0] == census.KIND_DIMENSION


@pytest.mark.parametrize("raw", ["2092", "1", "1,234", "0042"])
def test_bare_integers_read_as_labels_not_dimensions(raw):
    """A plot number is a name, not a measurement — the fraction is the tell."""
    assert census.classify_text(raw)[0] == census.KIND_NUMERIC


# --- annotation census: the Arabic reading ----------------------------------


def test_an_arabic_shx_string_is_read_and_the_reading_is_published():
    """`ls{] lpgD` is one of the twelve human-verified phrases in `app.shx`."""
    out = census.annotation_census([text_row("S1", "ls{] lpgD")])

    arabic = class_of(out, census.KIND_ARABIC)
    assert arabic["count"] == 1
    assert arabic["readings"][0]["reading"] == "مسجد محلي"
    assert arabic["readings"][0]["method"] == "phrase"
    assert arabic["example"]["text"] == "ls{] lpgD"
    assert arabic["example"]["text_reading"] == "مسجد محلي"


def test_every_arabic_class_carries_the_note_that_the_letters_are_derived():
    """`app.shx` rule 1: the reading is never presented as the stored bytes."""
    out = census.annotation_census([text_row("S1", "ls{] lpgD")])
    arabic = class_of(out, census.KIND_ARABIC)
    assert arabic["reading_note"] == shx.READING_NOTE
    assert "not shipped with the drawing" in arabic["reading_note"]


def test_the_stored_text_is_never_replaced_by_its_reading():
    out = census.annotation_census([text_row("S1", "ls{] lpgD")])
    arabic = class_of(out, census.KIND_ARABIC)
    assert arabic["samples"] == ["ls{] lpgD"]
    assert arabic["example"]["text"] == "ls{] lpgD"


@pytest.mark.parametrize("raw", ["sad", "hall", "Villa", "PLOT 2092"])
def test_a_latin_word_typed_on_the_same_keys_is_not_read_as_arabic(raw):
    """The trap `app.shx` exists to avoid, re-guarded at the census boundary.

    `sad` and `hall` are made entirely of keys that exist in the Arabic
    layout, so a careless per-character map reads them into Arabic nonsense.
    """
    assert census.classify_text(raw)[0] == census.KIND_OTHER


def test_a_resolved_style_pointing_at_an_arabic_font_unlocks_the_charmap():
    rows = [text_row("S1", "lpgD", text_style_font="xarb.shx")]
    out = census.annotation_census(rows)
    arabic = class_of(out, census.KIND_ARABIC)
    assert arabic["readings"][0]["reading"] == "محلي"
    assert arabic["readings"][0]["method"] == "charmap"


def test_a_resolved_style_declaring_no_font_is_evidence_it_is_not_arabic():
    """"Style unknown" and "style known, declares nothing" are different states.

    A row that carries the key at all has been resolved by the caller, and a
    resolved style with no Arabic font means there is nothing to re-read.
    """
    rows = [text_row("S1", "lpgD", text_style_font=None)]
    out = census.annotation_census(rows)
    assert class_of(out, census.KIND_ARABIC) is None
    assert class_of(out, census.KIND_OTHER)["count"] == 1


# --- annotation census: mixed content, and nothing silent -------------------


def test_mixed_content_splits_into_classes_that_add_up_to_the_total():
    rows = [
        text_row("A1", "2092"),
        text_row("A2", "2093"),
        text_row("D1", "25.00"),
        text_row("S1", "ls{] lpgD"),
        text_row("O1", "SETBACK LINE"),
        text_row("O2", "", type="DIMENSION"),
    ]
    out = census.annotation_census(rows)

    assert out["total"] == 6
    assert out["classified"] == 5
    assert out["without_text"]["count"] == 1
    assert out["classified"] + out["without_text"]["count"] == out["total"]
    assert {c["kind"] for c in out["classes"]} == {
        census.KIND_NUMERIC,
        census.KIND_DIMENSION,
        census.KIND_ARABIC,
        census.KIND_OTHER,
    }


def test_classes_are_published_in_a_fixed_order():
    rows = [
        text_row("O1", "SETBACK LINE"),
        text_row("S1", "ls{] lpgD"),
        text_row("D1", "25.00"),
        text_row("A1", "2092"),
    ]
    out = census.annotation_census(rows)
    assert [c["kind"] for c in out["classes"]] == list(census.KINDS)


def test_other_is_never_a_silent_bucket():
    """"everything else: 812" answers nothing. An example and samples do."""
    rows = [
        text_row("O1", "SETBACK LINE"),
        text_row("O2", "SITE PLAN"),
        text_row("O3", "SETBACK LINE"),
    ]
    out = census.annotation_census(rows)
    other = class_of(out, census.KIND_OTHER)

    assert other["count"] == 3
    assert other["example"]["handle"] == "O1"
    assert other["example"]["text"] == "SETBACK LINE"
    assert set(other["samples"]) == {"SETBACK LINE", "SITE PLAN"}


def test_rows_with_no_text_are_counted_and_shown_never_dropped():
    """Rule G8: a DIMENSION whose string is generated at draw time still exists."""
    rows = [
        text_row("A1", "2092"),
        text_row("N1", None, type="DIMENSION"),
        text_row("N2", "   ", type="MTEXT"),
    ]
    out = census.annotation_census(rows)

    missing = out["without_text"]
    assert missing["count"] == 2
    assert missing["by_type"] == {"DIMENSION": 1, "MTEXT": 1}
    assert missing["example"]["handle"] == "N1"
    assert missing["example"]["layout"] == "Model"


def test_an_empty_annotation_census_is_well_formed_and_never_none():
    out = census.annotation_census([])
    assert out is not None
    assert out["classes"] == []
    assert out["total"] == 0
    assert out["classified"] == 0
    assert out["without_text"]["count"] == 0
    assert out["repeated_values"] == []
    assert out["truncated"] is False
    assert out["scope"]["layouts"] == []


# --- rule G1: content decides, the layer name never does --------------------


def test_the_same_rows_classify_identically_under_contradictory_layer_names():
    """The `C-ROAD-*` trap, at the census boundary.

    A layer called `TEXT-ARABIC` full of plot numbers is full of plot numbers.
    """
    texts = ["2092", "25.00", "ls{] lpgD", "SETBACK LINE"]
    honest = [text_row(f"H{i}", t, layer="LABELS") for i, t in enumerate(texts)]
    lying = [text_row(f"H{i}", t, layer="TEXT-ARABIC") for i, t in enumerate(texts)]

    a = census.annotation_census(honest)
    b = census.annotation_census(lying)

    assert [(c["kind"], c["count"]) for c in a["classes"]] == [
        (c["kind"], c["count"]) for c in b["classes"]
    ]


# --- the acceptance case: "2092" exists twice, in two layouts ---------------


def test_the_same_label_in_two_layouts_is_never_reported_as_two_things():
    """The measured anchor: `27321CB` in `Model`, `2600C0A` in `[block] shml`.

    One label, drawn once and placed once. A census that answered "2 plots
    numbered 2092" would be repeating in miniature the mistake this campaign
    exists to end, so `repeated_values` is keyed by (layout, text) and finds
    no repeat here — while `by_layout` keeps both layouts visible.
    """
    rows = [
        text_row("27321CB", "2092", layout="Model"),
        text_row("2600C0A", "2092", layout="[block] shml"),
    ]
    out = census.annotation_census(rows)

    assert out["repeated_values"] == []
    numeric = class_of(out, census.KIND_NUMERIC)
    assert numeric["by_layout"] == {"Model": 1, "[block] shml": 1}
    assert sum(numeric["by_layout"].values()) == numeric["count"] == 2


def test_a_census_spanning_two_layouts_says_so_out_loud():
    rows = [
        text_row("27321CB", "2092", layout="Model"),
        text_row("2600C0A", "2092", layout="[block] shml"),
    ]
    out = census.annotation_census(rows)

    assert out["scope"]["single_scope"] is False
    assert out["scope"]["layouts"] == ["Model", "[block] shml"]
    assert out["scope_warnings"]
    assert "never" in out["scope"]["basis"]


def test_one_layer_and_one_layout_is_a_single_scope_with_no_warning():
    rows = [text_row("27321CB", "2092", layout="Model", layer="LABELS")]
    out = census.annotation_census(rows)
    assert out["scope"]["single_scope"] is True
    assert out["scope_warnings"] == []


def test_every_example_carries_its_own_layout():
    """An example that travels without its layout can be misquoted."""
    rows = [text_row("2600C0A", "2092", layout="[block] shml")]
    out = census.annotation_census(rows)
    assert class_of(out, census.KIND_NUMERIC)["example"]["layout"] == "[block] shml"


def test_a_known_containing_parcel_survives_the_census():
    """The census cannot compute containment — that needs MongoDB — but it
    must not throw it away either. "Plot 2092 is in parcel 205C7CC" is the
    answer; "there is a label reading 2092" is not."""
    rows = [
        text_row(
            "27321CB",
            "2092",
            contained_by=[{"handle": "205C7CC", "layer": "PARCELS", "area": 300.0}],
        )
    ]
    out = census.annotation_census(rows)
    example = class_of(out, census.KIND_NUMERIC)["example"]
    assert example["contained_by"][0]["handle"] == "205C7CC"


def test_a_row_without_containment_simply_omits_the_key():
    out = census.annotation_census([text_row("H1", "2092")])
    assert "contained_by" not in class_of(out, census.KIND_NUMERIC)["example"]


def test_a_genuine_repeat_inside_one_layout_is_reported_with_its_handles():
    """Two labels reading `2092` in the SAME layout is a real finding."""
    rows = [
        text_row("H1", "2092", layout="Model"),
        text_row("H2", "2092", layout="Model"),
        text_row("H3", "2093", layout="Model"),
    ]
    out = census.annotation_census(rows)

    assert len(out["repeated_values"]) == 1
    repeat = out["repeated_values"][0]
    assert repeat["text"] == "2092"
    assert repeat["layout"] == "Model"
    assert repeat["count"] == 2
    assert repeat["handles"] == ["H1", "H2"]


def test_a_row_without_a_layout_gets_a_named_placeholder_not_none():
    rows = [text_row("H1", "2092", layout=None)]
    out = census.annotation_census(rows)
    assert out["scope"]["layouts"] == [census.LAYOUT_UNKNOWN]
    assert class_of(out, census.KIND_NUMERIC)["by_layout"] == {
        census.LAYOUT_UNKNOWN: 1
    }


# --- annotation census: stated truncation (rule G7) -------------------------


def test_sample_lists_state_what_they_left_out():
    rows = [text_row(f"O{i}", f"NOTE {i}") for i in range(census.MAX_SAMPLES + 3)]
    out = census.annotation_census(rows)
    other = class_of(out, census.KIND_OTHER)

    assert len(other["samples"]) == census.MAX_SAMPLES
    assert other["samples_omitted"] == 3
    assert other["count"] == census.MAX_SAMPLES + 3  # the COUNT is never capped
    assert out["truncated"] is True
    assert out["truncation"]["sample_cap"] == census.MAX_SAMPLES


def test_handles_of_a_repeat_state_what_they_left_out():
    n = census.MAX_HANDLES_PER_VALUE + 4
    rows = [text_row(f"H{i}", "2092") for i in range(n)]
    out = census.annotation_census(rows)

    repeat = out["repeated_values"][0]
    assert repeat["count"] == n
    assert len(repeat["handles"]) == census.MAX_HANDLES_PER_VALUE
    assert repeat["handles_omitted"] == 4
    assert out["truncation"]["handles_omitted"] == 4


# --- block census -----------------------------------------------------------


def test_a_points_layer_is_counted_per_block_name():
    """"412 inserts" does not answer "how many trees". The name does."""
    rows = (
        [insert_row(f"T{i}", "TREE-PALM") for i in range(5)]
        + [insert_row(f"H{i}", "HYDRANT") for i in range(2)]
        + [insert_row("L1", "LIGHT-POLE")]
    )
    out = census.block_census(rows)

    assert out["total"] == 8
    assert out["named"] == 8
    assert out["distinct_names"] == 3
    assert [(b["block_name"], b["count"]) for b in out["blocks"]] == [
        ("TREE-PALM", 5),
        ("HYDRANT", 2),
        ("LIGHT-POLE", 1),
    ]


def test_block_counts_are_stamped_with_their_layouts():
    """A symbol placed in model space and one inside a block definition are
    not two symbols in one place — the same lie as "2 plots numbered 2092"."""
    rows = [
        insert_row("B1", "TREE-PALM", layout="Model"),
        insert_row("B2", "TREE-PALM", layout="[block] shml"),
    ]
    out = census.block_census(rows)

    block = out["blocks"][0]
    assert block["by_layout"] == {"Model": 1, "[block] shml": 1}
    assert out["scope"]["single_scope"] is False
    assert out["scope_warnings"]


def test_rows_without_a_block_name_are_counted_by_type_never_dropped():
    """A LINE on a points layer is ordinary; a nameless INSERT is an anomaly."""
    rows = [
        insert_row("B1", "TREE-PALM"),
        {"handle": "L1", "type": "LINE", "layer": "SYMBOLS", "layout": "Model"},
        {"handle": "B2", "type": "INSERT", "layer": "SYMBOLS", "layout": "Model"},
    ]
    out = census.block_census(rows)

    assert out["named"] == 1
    assert out["without_block_name"]["count"] == 2
    assert out["without_block_name"]["by_type"] == {"INSERT": 1, "LINE": 1}
    assert out["without_block_name"]["example"]["handle"] == "L1"
    assert out["named"] + out["without_block_name"]["count"] == out["total"]


def test_block_attributes_are_surfaced_because_they_are_the_data_record():
    rows = [
        insert_row("B1", "TREE-PALM", attribs={"SPECIES": "Phoenix", "AGE": "3"}),
        insert_row("B2", "TREE-PALM", attribs={"SPECIES": "Phoenix"}),
    ]
    out = census.block_census(rows)
    assert out["blocks"][0]["attribute_keys"] == ["AGE", "SPECIES"]
    assert out["blocks"][0]["attribute_keys_omitted"] == 0


def test_block_census_states_its_cap_and_counts_what_it_dropped():
    """Rule G7: the ceiling is published, and so is everything above it."""
    rows = [
        insert_row(f"B{i}", f"BLOCK-{i:04d}")
        for i in range(census.MAX_BLOCK_NAMES + 7)
    ]
    out = census.block_census(rows)

    assert len(out["blocks"]) == census.MAX_BLOCK_NAMES
    assert out["distinct_names"] == census.MAX_BLOCK_NAMES + 7
    assert out["truncated"] is True
    assert out["truncation"]["block_name_cap"] == census.MAX_BLOCK_NAMES
    assert out["truncation"]["names_omitted"] == 7
    assert out["truncation"]["entities_omitted"] == 7
    # The arithmetic still closes: nothing vanished, it was only not listed.
    reported = sum(b["count"] for b in out["blocks"])
    assert reported + out["truncation"]["entities_omitted"] == out["named"]


def test_block_census_orders_by_count_then_name_so_two_runs_read_alike():
    rows = [
        insert_row("B1", "ZEBRA"),
        insert_row("B2", "ALPHA"),
        insert_row("B3", "MIKE"),
        insert_row("B4", "MIKE"),
    ]
    out = census.block_census(rows)
    assert [b["block_name"] for b in out["blocks"]] == ["MIKE", "ALPHA", "ZEBRA"]


def test_an_empty_block_census_is_well_formed_and_never_none():
    out = census.block_census([])
    assert out is not None
    assert out["blocks"] == []
    assert out["total"] == 0
    assert out["named"] == 0
    assert out["distinct_names"] == 0
    assert out["without_block_name"]["count"] == 0
    assert out["without_block_name"]["example"] is None
    assert out["truncated"] is False


def test_a_block_name_that_is_blank_is_not_a_block_name():
    rows = [insert_row("B1", "   "), insert_row("B2", "TREE-PALM")]
    out = census.block_census(rows)
    assert out["distinct_names"] == 1
    assert out["without_block_name"]["count"] == 1


# --- invariants, over both functions ----------------------------------------


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [text_row("A1", "2092")],
        [text_row("A1", "2092"), text_row("N1", None)],
        [text_row(f"M{i}", t) for i, t in enumerate(["1", "2.5", "ls{] lpgD", "x"])],
    ],
)
def test_annotation_coverage_arithmetic_always_closes(rows):
    """classified + without_text == total, on every input. The campaign's
    coverage invariant, in miniature."""
    out = census.annotation_census(rows)
    assert out["classified"] + out["without_text"]["count"] == out["total"]
    assert sum(c["count"] for c in out["classes"]) == out["classified"]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [insert_row("B1", "TREE-PALM")],
        [insert_row("B1", "TREE-PALM"), {"handle": "L1", "type": "LINE"}],
    ],
)
def test_block_coverage_arithmetic_always_closes(rows):
    out = census.block_census(rows)
    assert out["named"] + out["without_block_name"]["count"] == out["total"]
    reported = sum(b["count"] for b in out["blocks"])
    assert reported + out["truncation"]["entities_omitted"] == out["named"]


@pytest.mark.parametrize("func", [census.block_census, census.annotation_census])
def test_neither_function_touches_the_rows_it_is_given(func):
    rows = [
        {
            "handle": "X1",
            "type": "INSERT",
            "block_name": "TREE-PALM",
            "text": "2092",
            "layer": "L",
            "layout": "Model",
        }
    ]
    before = [dict(r) for r in rows]
    func(rows)
    assert rows == before


@pytest.mark.parametrize("func", [census.block_census, census.annotation_census])
def test_every_by_layout_map_sums_to_its_own_count(func):
    rows = [
        insert_row("B1", "TREE-PALM", layout="Model") | {"text": "2092"},
        insert_row("B2", "TREE-PALM", layout="[block] shml") | {"text": "2092"},
        insert_row("B3", "HYDRANT", layout="Model") | {"text": "25.00"},
    ]
    out = func(rows)
    items = out.get("blocks") or out.get("classes")
    for item in items:
        assert sum(item["by_layout"].values()) == item["count"]
