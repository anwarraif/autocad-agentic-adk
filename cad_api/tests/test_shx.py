"""Arabic SHX readings — UPLIFT-04.

What is guarded here is not only "is the reading right", but **does it know
when it must not read at all**. The per-character map accepts every key on the
Arabic keyboard layout, and those keys are ordinary Latin letters — so without
a guard, `sad` and `hall` would "read" into Arabic nonsense that looks exactly
like a legitimate reading.

That is the same class of defect found twice already this session: an answer
that looks plausible and has no way of telling its reader that it is wrong.
"""

from __future__ import annotations

import pytest

from app import shx


# --- the phrase layer: twelve strings a human has read ---------------------


@pytest.mark.parametrize("raw,expected", sorted(shx.PHRASES.items()))
def test_every_verified_phrase_reads_exactly(raw, expected):
    """DoD UPLIFT-04: the 12 strings in the table give the exact `phrase` reading."""
    r = shx.read(raw)
    assert r.reading == expected[0]
    assert r.method == "phrase"
    assert r.reason is None
    assert "not shipped with the drawing" in r.note


def test_a_verified_phrase_survives_sloppy_whitespace():
    """`ahvU uvQ  15   ljv` comes from the file with double spaces.

    Another drawing will type it with a single space, and losing its reading
    because of that is an invisible loss.
    """
    r = shx.read("ahvU uvQ 15 ljv")
    assert r.reading == "شارع عرض ١٥ متر"
    assert r.method == "phrase"


def test_the_reading_never_replaces_the_stored_text():
    """UPLIFT-04's first rule. `text` stays the bytes from the file."""
    out = shx.describe("ls{] lpgD")
    assert out["text"] == "ls{] lpgD"
    assert out["text_reading"] == "مسجد محلي"
    assert out["reading_meaning"] == "local mosque"


# --- the charmap layer: reaching what has never been seen -------------------


def test_an_unseen_string_reads_by_charmap_and_says_so():
    # `lpgD` = محلي, a fragment of an already verified phrase, but not a
    # `PHRASES` key of its own. It must read through the map and CARRY the
    # not-yet-verified warning.
    r = shx.read("lpgD", font="xarb.shx", style_known=True)
    assert r.reading == "محلي"
    assert r.method == "charmap"
    assert "never been verified by a human" in r.note


def test_one_unmapped_key_kills_the_whole_reading():
    """A partial reading is never published.

    A word half of which was guessed reads exactly like a word that is whole,
    and its reader has no way of telling them apart.
    """
    r = shx.read("ls{]~", font="xarb.shx", style_known=True)
    assert r.reading is None
    assert r.method is None
    assert "a partial reading is not published" in r.reason


def test_digits_map_to_arabic_indic_digits():
    r = shx.read("15", font="xarb.shx", style_known=True)
    assert r.reading == "١٥"


# --- the easiest thing to get wrong: refusing to read -----------------------


def test_plain_latin_text_is_never_read():
    """`text_reading` is null for all ordinary Latin text — DoD UPLIFT-04."""
    for raw in ("ROOM 101", "Primary School", "NTS", "Scale 1:500"):
        r = shx.read(raw)
        assert r.reading is None, f"{raw!r} must not be read"
        assert r.reason


def test_a_latin_word_made_only_of_mapped_keys_is_still_refused():
    """The heart of this file.

    `sad`, `hall`, `flat` and `stad` are made entirely of keys that exist in
    this layout. The per-character map will happily turn them into Arabic
    letters, and the result is a word that never existed in any drawing,
    presented in the same tone as a word that is real.
    """
    for raw in ("sad", "hall", "flat", "stad", "grad"):
        r = shx.read(raw)
        assert r.reading is None, f"{raw!r} was read as {r.reading!r}"
        assert "Latin" in r.reason


def test_a_style_that_is_not_arabic_refuses_before_anything_else():
    """The text style decides, not a guess about the letters.

    Even a string that is EXACTLY in the phrase dictionary must be refused if
    its text style points at a Latin font: the same string typed in simplex is
    a coincidence, not Arabic.
    """
    r = shx.read("ls{] lpgD", font="simplex.shx", style_known=True)
    assert r.reading is None
    assert "not an Arabic SHX font" in r.reason


def test_a_truetype_arabic_style_is_not_our_business():
    """TrueType stores Arabic letters as Arabic letters. There is nothing to
    re-read, and re-reading it would break it."""
    assert not shx.is_arabic_shx("arabtype.ttf")
    assert shx.is_arabic_shx("xarb.shx")
    assert shx.is_arabic_shx(None, "Y-ARAB1b.SHX")


def test_font_detection_is_a_substring_rule_not_a_closed_list():
    """The next drawing comes from another contractor.

    A closed list would answer "there is no Arabic text in this drawing" with
    confidence, for a file that is full of it.
    """
    for font in ("X-ARAB1b.SHX", "arabic.shx", "ARB__.SHX", "myArabFont.shx"):
        assert shx.is_arabic_shx(font), font
    for font in ("simplex.shx", "romanc", "MONOTXT", "swisscb.ttf"):
        assert not shx.is_arabic_shx(font), font


def test_an_unknown_style_still_reads_keyboard_shaped_strings():
    """An unknown style is not a reason to refuse everything.

    A string carrying a key outside letters-and-digits, or a change of letter
    case inside a word, cannot be an ordinary Latin word.
    """
    assert shx.read("p]drm").reading == "حديقة"      # has a `]`
    assert shx.read("lvhtR").reading == "مرافق"      # a change of case
    assert shx.read("hall").reading is None          # neither of the two


def test_empty_and_missing_text_answer_with_a_reason():
    for raw in (None, "", "   "):
        r = shx.read(raw)
        assert r.reading is None
        assert r.reason


# --- searching in both directions -------------------------------------------


def test_an_arabic_word_maps_back_to_every_key_that_spells_it():
    """The map is many-to-one, so a search that tries a single form would miss
    half the drawing without a single sign that it had."""
    forms = shx.shx_forms("مسجد")
    assert "ls{]" in forms
    assert len(forms) >= 1


def test_searching_for_the_mosque_word_finds_the_stored_form():
    """DoD UPLIFT-04: `search_text("مسجد")` finds the mosque parcel through the
    reverse map. What is tested here is the map; the query path lives in the
    store."""
    forms = set(shx.shx_forms("مسجد محلي"))
    assert "ls{] lpgD" in forms


def test_a_word_outside_the_layout_returns_empty_not_a_guess():
    """Empty means no text in any drawing could spell it — a different answer
    from "not found"."""
    assert shx.shx_forms("ژپچ") == []
    assert shx.shx_forms("") == []


def test_reverse_forms_stay_within_a_stated_ceiling():
    long_word = "مسجدمسجدمسجدمسجد"
    assert len(shx.shx_forms(long_word)) <= shx.MAX_FORMS


# --- the shape of the response ----------------------------------------------


def test_describe_keeps_the_same_shape_whether_it_read_anything_or_not():
    read_ok = shx.describe("ls{] lpgD")
    refused = shx.describe("Primary School")
    assert set(read_ok) == set(refused)
    assert refused["text_reading"] is None
    assert refused["reading_not_measured"]
    assert read_ok["reading_not_measured"] is None


def test_every_charmap_entry_is_traceable_to_a_verified_phrase():
    """The rule that makes this map trustworthy: not one letter comes from
    outside the corpus a human has read.

    Digits and the space are excluded — both are unambiguous and neither
    appears in the phrases as a letter.
    """
    corpus = "".join(shx.PHRASES)
    for key in shx.CHARMAP:
        if key.isdigit() or key == " ":
            continue
        assert key in corpus, f"{key!r} appears in no verified phrase"
