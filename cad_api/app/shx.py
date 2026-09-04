"""Reading Arabic text stored as Latin bytes — UPLIFT-04.

A drawing typed with an Arabic SHX font stores **the keystrokes**, not the
letters. `مسجد محلي` lands in the file as
`ls{] lpgD`, and every tool that reads that file as it stands reports exactly
that — right about the bytes, useful to nobody.

**The font files are missing, and this module works precisely because they are
not needed.** `xarb.shx`, `xarab.shx`, `X-ARAB1b.SHX` and `Y-ARAB1b.SHX` are
not shipped with the drawing, and a note from an earlier session concluded that
an SHX reading was therefore impossible. That conclusion was mistaken, and it
stood long enough for `UPLIFT-04` to be reported as work that could not be
done. What is needed is not the glyphs but the keyboard layout, and that layout
is derived from a corpus that has already been verified by a human — the twelve
phrases in `UPLIFT-04-SHX-ARABIC.md`, every letter of which can be traced back
to a pairing somebody has read.

Four rules that make this module trustworthy:

1. **`text` never changes.** The reading is a second field beside it, never a
   replacement for it. What is stored in the file stays checkable by anyone.
2. **A partial reading is never published.** One character outside the map
   makes the whole reading `None`. A word half of which was guessed reads
   exactly like a word that is whole.
3. **The method is always named.** `phrase` means a person has read this exact
   string; `charmap` means it was assembled letter by letter and nobody has
   ever seen it. Treating the two as equal is raising the evidence grade, which
   the UPLIFT-08 contract forbids.
4. **The text style decides, not a guess about the letters.** This is the part
   that is easiest to get wrong: `sad`, `hall` and `TH` are made entirely of
   keys that exist in this layout, so a per-character map will happily "read"
   an ordinary Latin word into Arabic nonsense. What determines that an entity
   was written with an Arabic font is its text style, and that style is stored
   — 7,854 entities in the reference drawing carry it.

A pure module: no MongoDB and no `app.*`. Every `CHARMAP` entry below carries
the phrase that proves it.
"""

from __future__ import annotations

import re
from typing import Final, NamedTuple

#: The sentence that MUST accompany every reading that reaches a reader.
#:
#: Not politeness. A reader who sees Arabic letters in a CAD tool will assume
#: those letters are in the file, and here they are not.
READING_NOTE: Final[str] = (
    "The Arabic reading is derived from the SHX font's keyboard mapping, not "
    "from the stored value. The original font files (xarb.shx, xarab.shx, "
    "X-ARAB1b.SHX) are not shipped with the drawing. Present the reading "
    "together with its raw string."
)

#: An addition for a reading assembled letter by letter.
CHARMAP_NOTE: Final[str] = (
    "This string has never been verified by a human; it was assembled from the "
    "per-character map and can be wrong on a word nobody has seen."
)


class Reading(NamedTuple):
    """One reading, or its absence together with the cause.

    `reading` being None means NOT READ, and `reason` says why. Never an empty
    string: an empty string reads like "it was read, and the result really is
    nothing at all".
    """

    reading: str | None
    method: str | None
    reason: str | None
    note: str | None


#: Twelve human-verified phrases, from the T4 layer corpus.
#:
#: The keys are the bytes exactly as they are in the file. This is the most
#: accurate and the narrowest layer; anything outside it falls through to
#: `CHARMAP`.
PHRASES: Final[dict[str, tuple[str, str]]] = {
    "ahvU uvQ  15   ljv": ("شارع عرض ١٥ متر", "15 m wide street"),
    "ahvU uvQ  18   ljv": ("شارع عرض ١٨ متر", "18 m wide street"),
    "ahvU uvQ  10   ljv": ("شارع عرض ١٠ متر", "10 m wide street"),
    "lvhtR": ("مرافق", "facilities"),
    "uhlm": ("عامة", "public"),
    "p]drm": ("حديقة", "garden"),
    "j{hvD": ("تجاري", "commercial"),
    "l,HrT sdhvHJ": ("مواقف سيارات", "car parking"),
    "lvtR jugdlD": ("مرفق تعليمي", "educational facility"),
    "ls{] lpgD": ("مسجد محلي", "local mosque"),
    "ls{] {hlU": ("مسجد جامع", "congregational mosque"),
    "ahvU hgskfG": ("شارع السنبل", "Al-Sunbul Street"),
}

#: `PHRASES` with its whitespace flattened, built once.
#:
#: The keys come from the file as it stands, and that file holds double and
#: triple spaces -- `ahvU uvQ  15   ljv`. The next drawing will type the same
#: phrase with a single space, and losing its reading because of that is an
#: invisible loss: it produces no error, only fewer answers.
PHRASES_NORMALISED: Final[dict[str, tuple[str, str]]] = {
    " ".join(key.split()): value for key, value in PHRASES.items()
}

#: The per-character map, DERIVED from `PHRASES` and from nowhere else.
#:
#: An Arabic keyboard layout maps several keys to the same letter — the
#: initial, medial, final and isolated forms of one letter sit on different
#: keys — so this map is deliberately many-to-one.
CHARMAP: Final[dict[str, str]] = {
    "a": "ش",   # ahvU      -> شارع
    "h": "ا",   # ahvU      -> شارع
    "H": "ا",   # l,HrT     -> مواقف
    "v": "ر",   # ahvU      -> شارع
    "U": "ع",   # ahvU      -> شارع
    "u": "ع",   # uvQ       -> عرض
    "Q": "ض",   # uvQ       -> عرض
    "l": "م",   # ljv       -> متر
    "j": "ت",   # ljv       -> متر
    "J": "ت",   # sdhvHJ    -> سيارات
    "t": "ف",   # lvhtR     -> مرافق
    "T": "ف",   # l,HrT     -> مواقف
    "R": "ق",   # lvhtR     -> مرافق
    "r": "ق",   # p]drm     -> حديقة
    "m": "ة",   # uhlm      -> عامة
    "p": "ح",   # p]drm     -> حديقة
    "]": "د",   # p]drm     -> حديقة
    "d": "ي",   # p]drm     -> حديقة
    "D": "ي",   # j{hvD     -> تجاري
    "{": "ج",   # ls{]      -> مسجد
    ",": "و",   # l,HrT     -> مواقف
    "s": "س",   # ls{]      -> مسجد
    "g": "ل",   # jugdlD    -> تعليمي
    "G": "ل",   # hgskfG    -> السنبل
    "k": "ن",   # hgskfG    -> السنبل
    "f": "ب",   # hgskfG    -> السنبل
    "0": "٠",
    "1": "١",
    "2": "٢",
    "3": "٣",
    "4": "٤",
    "5": "٥",
    "6": "٦",
    "7": "٧",
    "8": "٨",
    "9": "٩",
    " ": " ",
}

#: Characters that may appear without cancelling a reading but that contribute
#: no letter. Kept apart from `CHARMAP` so that "ignored" cannot be confused
#: with "this is a letter".
IGNORED: Final[frozenset[str]] = frozenset({"\t", "\n", "\r"})

#: The font file names that mark an Arabic text style.
#:
#: Matched as a SUBSTRING of the name, not as a closed list, and that is what
#: makes it hold for the next drawing: another contractor uses
#: `X-ARAB1b.SHX`, `arabic.shx`, or `ARB__.SHX`, and a closed list would answer
#: "there is no Arabic text in this drawing" with confidence.
ARABIC_FONT_MARKS: Final[tuple[str, ...]] = ("arab", "arb")


def is_arabic_shx(font: str | None, bigfont: str | None = None) -> bool:
    """Whether this text style points at an Arabic SHX font.

    Decided from the file name, because the file itself is not there. A style
    that points at a `.ttf` is none of this module's business: TrueType stores
    Arabic letters as Arabic letters, and there is nothing to re-read.
    """
    for value in (font, bigfont):
        if not value:
            continue
        name = str(value).casefold()
        if name.endswith(".ttf") or name.endswith(".otf"):
            continue
        if any(mark in name for mark in ARABIC_FONT_MARKS):
            return True
    return False


#: The marks that a string was written in keyboard order rather than in the
#: Latin alphabet.
#:
#: Used ONLY when the text style is unknown. `sad` and `hall` are made entirely
#: of keys that exist in the map, so without this guard the per-character map
#: would read an ordinary English word into Arabic nonsense. Two marks, neither
#: of which appears in a normal Latin word: a key outside letters-and-digits,
#: or a change of letter case INSIDE a word.
_NON_ALNUM_KEY = re.compile(r"[^A-Za-z0-9\s]")
_CASE_SHIFT_INSIDE_WORD = re.compile(r"[a-z][A-Z]|[A-Z][a-z0-9]*[a-z][A-Z]")


def _looks_like_keyboard_order(raw: str) -> bool:
    return bool(_NON_ALNUM_KEY.search(raw) or _CASE_SHIFT_INSIDE_WORD.search(raw))


def _has_unmapped_letter(raw: str) -> bool:
    return any(c.isalpha() and c not in CHARMAP for c in raw)


def read(
    raw: str | None,
    *,
    font: str | None = None,
    bigfont: str | None = None,
    style_known: bool = False,
) -> Reading:
    """Read one SHX string. Never raises, never guesses a part of it.

    Args:
        raw: the text exactly as it is in the file.
        font, bigfont: the font files of this entity's text style, if known.
        style_known: True when the caller has already resolved the text style
            and `font` really describes it. Kept apart from `font is None`
            because "the style is unknown" and "the style is known and declares
            no font" are two different states, and the second one is evidence
            that it is NOT a recognised Arabic font.
    """
    if raw is None:
        return Reading(None, None, "there is no text on this entity", None)
    if not raw.strip():
        return Reading(None, None, "the text is empty", None)

    arabic_style = is_arabic_shx(font, bigfont)
    if style_known and not arabic_style:
        return Reading(
            None,
            None,
            f"the text style points at font {font or bigfont or '(not declared)'!r}, "
            "not an Arabic SHX font; there is nothing to re-read",
            None,
        )

    hit = PHRASES.get(raw) or PHRASES_NORMALISED.get(" ".join(raw.split()))
    if hit is not None:
        return Reading(hit[0], "phrase", None, READING_NOTE)

    if _has_unmapped_letter(raw):
        return Reading(
            None,
            None,
            "it contains a letter that is not on this keyboard layout, so it "
            "is not Arabic text typed with this font",
            None,
        )

    if not arabic_style and not _looks_like_keyboard_order(raw):
        return Reading(
            None,
            None,
            "the text style is unknown and this string reads like ordinary "
            "Latin text; a reading here would invent an Arabic word out of a "
            "word that is not one",
            None,
        )

    out: list[str] = []
    for ch in raw:
        if ch in IGNORED:
            continue
        mapped = CHARMAP.get(ch)
        if mapped is None:
            # One unknown character cancels the whole thing.
            return Reading(
                None,
                None,
                f"character {ch!r} is not in the keyboard map; a partial "
                "reading is not published",
                None,
            )
        out.append(mapped)

    reading = "".join(out).strip()
    if not reading:
        return Reading(None, None, "no letter is left after mapping", None)
    return Reading(reading, "charmap", None, f"{READING_NOTE} {CHARMAP_NOTE}")


#: The reverse map: one Arabic letter to every key that produces it.
#:
#: Used by search. Because the map is many-to-one, a search that tries a single
#: key would miss half the drawing without a single sign that it had.
REVERSE: Final[dict[str, tuple[str, ...]]] = {}
for _key, _letter in CHARMAP.items():
    REVERSE[_letter] = REVERSE.get(_letter, ()) + (_key,)

#: The cap on the combinatorial explosion in `shx_forms`.
MAX_FORMS: Final[int] = 512


def shx_forms(arabic: str) -> list[str]:
    """Every SHX form that could produce this Arabic word.

    A list, not one string, because the layout is many-to-one: `مسجد`
    can be typed in more than one way, and a search that tries only one of them
    would report zero with confidence.

    An EMPTY list means this word contains a letter that is not in this layout
    at all — no text in any drawing could spell it, and that is a different
    answer from "not found".
    """
    if not arabic:
        return []
    options: list[tuple[str, ...]] = []
    for ch in arabic:
        if ch == " ":
            options.append((" ",))
            continue
        keys = REVERSE.get(ch)
        if not keys:
            return []
        options.append(keys)

    out = [""]
    for keys in options:
        if len(out) * len(keys) > MAX_FORMS:
            break
        out = [prefix + k for prefix in out for k in keys]
    return sorted(set(out))


def describe(
    raw: str | None,
    *,
    font: str | None = None,
    bigfont: str | None = None,
    style_known: bool = False,
) -> dict[str, object]:
    """A ready-to-paste reading block for a response.

    Always the same shape, whatever the outcome. A shape that disappears when
    there is no reading forces every caller to guess, and a guess like that has
    already failed in this repo — see D-085.
    """
    r = read(raw, font=font, bigfont=bigfont, style_known=style_known)
    meaning = PHRASES_NORMALISED.get(" ".join((raw or "").split()), (None, None))[1]
    return {
        "text": raw,
        "text_reading": r.reading,
        "reading_method": r.method,
        "reading_not_measured": r.reason,
        "reading_note": r.note,
        "reading_meaning": meaning if r.method == "phrase" else None,
    }
