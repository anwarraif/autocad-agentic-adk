"""The in-file comment contract: embed, survive a save, read back.

This is the feature the owner described as "even when we export or give it to
some other person those comments will be there with that structure". Every
test here runs the full write -> save -> reload cycle, because the only thing
that matters is what is in the file after it leaves us.
"""

from __future__ import annotations

import ezdxf
import pytest
from ezdxf.lldxf.types import dxftag

from app.export import (
    APPID,
    MARKUP_LAYER,
    XDATA_VERSION,
    cache_key_for,
    embed_comments,
    read_embedded_comments,
)


def _doc_with_line():
    doc = ezdxf.new("R2010", setup=True)
    line = doc.modelspace().add_line((0, 0), (100, 50))
    return doc, line


def _roundtrip(doc, tmp_path):
    path = tmp_path / "exported.dxf"
    doc.saveas(path)
    return ezdxf.readfile(str(path))


COMMENTS = [
    {"author": "user:kurnia", "body": "Check wall alignment", "created_at": "2026-08-18"},
    {"author": "agent:cad-agent", "body": "Setback below minimum", "created_at": "2026-08-18"},
]


def test_xdata_survives_save_and_reload(tmp_path):
    """The machine-readable record must still be there after the file leaves."""
    doc, line = _doc_with_line()
    handle = line.dxf.handle
    embed_comments(doc, "abc123", {handle: COMMENTS})

    reloaded = _roundtrip(doc, tmp_path)
    found = read_embedded_comments(reloaded)
    assert handle in found, "comments vanished on save/reload"
    assert [c["body"] for c in found[handle]] == [c["body"] for c in COMMENTS]
    assert found[handle][0]["author"] == "user:kurnia"


def test_handle_is_preserved_by_the_export(tmp_path):
    """The whole design keys on the handle staying identical in the copy."""
    doc, line = _doc_with_line()
    handle = line.dxf.handle
    embed_comments(doc, "abc123", {handle: COMMENTS})
    reloaded = _roundtrip(doc, tmp_path)
    handles = {e.dxf.handle for e in reloaded.modelspace()}
    assert handle in handles


def test_hyperlink_gives_autocad_its_hover_tooltip(tmp_path):
    """AutoCAD shows the hyperlink description on hover — no plugin needed.

    This is the mechanism behind "hover over any building and a comment pops
    up" for a recipient opening the file in real AutoCAD.
    """
    doc, line = _doc_with_line()
    handle = line.dxf.handle
    embed_comments(doc, "abc123", {handle: COMMENTS})

    reloaded = _roundtrip(doc, tmp_path)
    entity = next(e for e in reloaded.modelspace() if e.dxf.handle == handle)
    link, description, _ = entity.get_hyperlink()
    assert "user:kurnia: Check wall alignment" in description
    assert "+1 more" in description, "multiple comments must be signalled"


def test_visible_note_lands_on_the_markup_layer(tmp_path):
    """A viewer that shows neither XDATA nor hyperlinks still sees the note,
    and the recipient can switch all markup off via one layer."""
    doc, line = _doc_with_line()
    embed_comments(doc, "abc123", {line.dxf.handle: COMMENTS})

    reloaded = _roundtrip(doc, tmp_path)
    assert MARKUP_LAYER in reloaded.layers
    notes = [
        e
        for e in reloaded.modelspace()
        if e.dxf.layer == MARKUP_LAYER and e.dxftype() == "MTEXT"
    ]
    assert len(notes) == 1
    assert "Check wall alignment" in notes[0].text
    assert "Setback below minimum" in notes[0].text


def test_uncommented_entities_are_untouched(tmp_path):
    """The export must not spray metadata over the rest of the drawing."""
    doc, line = _doc_with_line()
    other = doc.modelspace().add_circle((5, 5), radius=2)
    embed_comments(doc, "abc123", {line.dxf.handle: COMMENTS})

    reloaded = _roundtrip(doc, tmp_path)
    circle = next(e for e in reloaded.modelspace() if e.dxftype() == "CIRCLE")
    with pytest.raises(Exception):
        circle.get_xdata(APPID)


def test_missing_handles_are_reported_not_silently_dropped(tmp_path):
    """A comment pointing at a handle absent from the file must be surfaced."""
    doc, line = _doc_with_line()
    result = embed_comments(
        doc, "abc123", {line.dxf.handle: COMMENTS, "DEAD01": COMMENTS}
    )
    assert result.skipped_missing_handles == ["DEAD01"]
    assert result.entities_annotated == 1


def test_long_comment_bodies_survive_chunking(tmp_path):
    """XDATA strings cap at 255 bytes; a long body must chunk and rejoin."""
    doc, line = _doc_with_line()
    handle = line.dxf.handle
    long_body = "verify this segment " * 40  # ~800 chars
    embed_comments(doc, "abc123", {handle: [{"author": "a", "body": long_body, "created_at": ""}]})

    reloaded = _roundtrip(doc, tmp_path)
    found = read_embedded_comments(reloaded)
    assert found[handle][0]["body"] == long_body


def test_cache_key_changes_with_the_comment_set():
    """Exports regenerate exactly when the comments change — not more often
    (Janadriyah costs ~2 minutes) and never less."""
    a = cache_key_for([{"_id": "1", "body": "x"}])
    b = cache_key_for([{"_id": "1", "body": "x"}, {"_id": "2", "body": "y"}])
    c = cache_key_for([{"_id": "1", "body": "CHANGED"}])
    assert a != b and a != c and b != c
    assert a == cache_key_for([{"_id": "1", "body": "x"}])


def test_reader_rejects_foreign_xdata(tmp_path):
    """XDATA from another application must not be misread as our comments."""
    doc, line = _doc_with_line()
    doc.appids.add(APPID)
    line.set_xdata(APPID, [(1000, "SOMETHING_ELSE_V9"), (1000, "not ours")])
    reloaded = _roundtrip(doc, tmp_path)
    assert read_embedded_comments(reloaded) == {}


def test_markup_note_never_inherits_the_drawing_font(tmp_path):
    """The comment must not vanish because the drawing's font is missing.

    Janadriyah prompts "Missing SHX Files" in AutoCAD: it references Arabic
    shape fonts that were not distributed with it. If the note borrowed the
    drawing's `Standard` style and that style pointed at one of those, the
    comment would be the thing that disappears. It gets its own style, on a
    TrueType font the OS resolves rather than AutoCAD's SHX search path.
    """
    from app.export import MARKUP_FONT, MARKUP_STYLE

    doc, line = _doc_with_line()
    # A drawing whose Standard style points at a font nobody has.
    doc.styles.get("Standard").dxf.font = "totally-missing-font.shx"
    embed_comments(doc, "abc123", {line.dxf.handle: COMMENTS})

    reloaded = _roundtrip(doc, tmp_path)
    note = next(
        e
        for e in reloaded.modelspace()
        if e.dxf.layer == MARKUP_LAYER and e.dxftype() == "MTEXT"
    )
    assert note.dxf.style == MARKUP_STYLE, "note fell back to the drawing's style"
    assert reloaded.styles.get(MARKUP_STYLE).dxf.font == MARKUP_FONT


def test_leader_arrow_touches_the_entity_and_text_points_inward(tmp_path):
    """The note must read as attached to its object, inside the sheet.

    The first version anchored plain text at the entity's bbox *corner* — for
    a long line, its far end — with no visual link. In TrueView the notes
    floated at the sheet corners, extents stretched to include them, and the
    user reported the whole drawing as "shifted". The redline contract now:
    an arrow whose tip sits on the entity's midpoint, and text pulled toward
    the middle of the sheet so zoom-to-extents stays the drawing's own frame.
    """
    doc, line = _doc_with_line()  # line from (0,0) to (100,50): midpoint (50,25)
    # More geometry so the sheet centre is meaningfully placed.
    doc.modelspace().add_line((0, 100), (100, 100))
    embed_comments(doc, "abc123", {line.dxf.handle: COMMENTS})

    reloaded = _roundtrip(doc, tmp_path)
    marks = [e for e in reloaded.modelspace() if e.dxf.layer == MARKUP_LAYER]
    kinds = {e.dxftype() for e in marks}
    assert kinds == {"LINE", "SOLID", "MTEXT"}, f"redline incomplete: {kinds}"

    leader = next(e for e in marks if e.dxftype() == "LINE")
    start = (leader.dxf.start.x, leader.dxf.start.y)
    assert start == pytest.approx((50.0, 25.0)), (
        "the arrow must touch the entity midpoint, not a bbox corner"
    )

    # Text sits between the entity and the sheet interior, not past the edge.
    mtext = next(e for e in marks if e.dxftype() == "MTEXT")
    x, y = mtext.dxf.insert.x, mtext.dxf.insert.y
    assert 0 <= x <= 100 and 0 <= y <= 100, "note escaped the sheet extents"


def test_collect_handles_covers_layouts_and_blocks(tmp_path):
    """The post-save verifier must see everything a drawing can hold.

    It exists because visualization_-_aerial's five 3DSOLIDs vanish when
    ezdxf re-saves the converted file — measured, not hypothetical. A verifier
    that missed block contents or paperspace would report zero loss falsely.
    """
    from app.export import collect_handles

    doc, line = _doc_with_line()
    block = doc.blocks.new(name="PART")
    inner = block.add_line((0, 0), (1, 1))
    paper = doc.layouts.get("Layout1")
    pline = paper.add_line((0, 0), (5, 5))

    handles = collect_handles(doc)
    assert line.dxf.handle in handles
    assert inner.dxf.handle in handles
    assert pline.dxf.handle in handles


def test_layout_hints_override_raw_extents(tmp_path):
    """Outlier geometry must not inflate the note text.

    Janadriyah's modelspace spans 15,129 m because of a surroundings block,
    but 90% of its entities sit in a 1,486 m site — sizing from raw extents
    drew 76 m letters over 20 m buildings. A hint measured from the dense
    content region must win over the layout's own extents.
    """
    doc, line = _doc_with_line()
    # An outlier far away inflates the raw extents 100x.
    doc.modelspace().add_line((10_000, 10_000), (10_001, 10_001))
    hint = {"Model": (2.5, (50.0, 25.0))}

    embed_comments(doc, "abc123", {line.dxf.handle: COMMENTS}, layout_hints=hint)
    reloaded = _roundtrip(doc, tmp_path)
    note = next(
        e
        for e in reloaded.modelspace()
        if e.dxf.layer == MARKUP_LAYER and e.dxftype() == "MTEXT"
    )
    assert note.dxf.char_height == pytest.approx(2.5), (
        "the dense-extents hint was ignored; raw extents sized the text"
    )


def test_cache_key_changes_when_drawing_changes():
    """A re-ingested drawing must not be served another drawing's cache.

    Measured failure: after the ODA re-ingest changed the drawing_id, the
    identical comment set produced the same key and the endpoint served the
    stale pre-ODA artifact.
    """
    comments = [{"_id": "1", "body": "x"}]
    assert cache_key_for(comments, "aaaa") != cache_key_for(comments, "bbbb")
    assert cache_key_for(comments, "aaaa") == cache_key_for(comments, "aaaa")


def test_strip_corrupt_xrecords_removes_only_garbage(tmp_path):
    """Corrupt string tags go; the XRECORD object and healthy tags stay.

    Janadriyah's seven 3DSVIZ material records come out of BOTH converters as
    unreadable garbage and crash Autodesk's web extractor; healthy records
    must never be touched by the same sweep.
    """
    from app.export import strip_corrupt_xrecords

    doc, _ = _doc_with_line()
    good = doc.objects.add_xrecord(owner=doc.rootdict.dxf.handle)
    good.tags.append(dxftag(300, "<Material>perfectly readable</Material>"))
    bad = doc.objects.add_xrecord(owner=doc.rootdict.dxf.handle)
    bad.tags.append(dxftag(300, "<Material>ok start"))
    bad.tags.append(dxftag(300, "then \ufffd\ufffd garbage"))

    good_handle, bad_handle = good.dxf.handle, bad.dxf.handle

    removed = strip_corrupt_xrecords(doc)

    assert removed == [bad_handle]
    kept = doc.entitydb.get(bad_handle)
    assert kept is not None and kept.is_alive
    texts = [t.value for t in kept.tags if isinstance(t.value, str)]
    assert "<Material>ok start" in texts
    assert all("�" not in t for t in texts)
    live = doc.entitydb.get(good_handle)
    assert live is not None and live.is_alive


def test_strip_unconvertible_objects_noop_on_clean_doc():
    """A document without render-preset objects must pass through untouched."""
    from app.export import strip_unconvertible_objects

    doc, _ = _doc_with_line()
    before = len(list(doc.objects))
    assert strip_unconvertible_objects(doc) == []
    assert len(list(doc.objects)) == before
