"""Extraction: handles, extents, units, and the paperspace trap.

These build their own DXF documents in memory rather than reading fixtures, so
the tests state exactly the condition they check and cannot drift when the
sample files change.
"""

from __future__ import annotations

import math
from pathlib import Path

import ezdxf
import pytest

from app import extract


@pytest.fixture
def simple_doc(tmp_path: Path) -> Path:
    """A DXF with content in modelspace, on two layers, plus a block."""
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 6  # metres
    doc.layers.add("WALL")
    doc.layers.add("TEXT-L")

    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0), dxfattribs={"layer": "WALL"})
    msp.add_line((10, 0), (10, 5), dxfattribs={"layer": "WALL"})
    msp.add_circle((5, 5), radius=2, dxfattribs={"layer": "WALL"})
    msp.add_text("ROOM 101", dxfattribs={"layer": "TEXT-L"}).set_placement((1, 1))

    block = doc.blocks.new(name="DOOR")
    block.add_line((0, 0), (1, 0))
    msp.add_blockref("DOOR", (3, 3), dxfattribs={"layer": "WALL"})

    path = tmp_path / "simple.dxf"
    doc.saveas(path)
    return path


@pytest.fixture
def paperspace_only_doc(tmp_path: Path) -> Path:
    """The trap: nothing in modelspace, everything on a paperspace layout.

    Four of the 17 sample files are shaped like this. An extractor that walks
    `doc.modelspace()` reports them as empty.
    """
    doc = ezdxf.new("R2010", setup=True)
    layout = doc.layouts.get("Layout1")
    layout.add_line((0, 0), (100, 0), dxfattribs={"layer": "0"})
    layout.add_line((0, 0), (0, 50), dxfattribs={"layer": "0"})
    path = tmp_path / "paperspace.dxf"
    doc.saveas(path)
    return path


def test_drawing_id_is_content_addressed(simple_doc: Path, tmp_path: Path):
    """Same bytes -> same id; different bytes -> different id."""
    copy = tmp_path / "renamed.dxf"
    copy.write_bytes(simple_doc.read_bytes())
    assert extract.compute_drawing_id(copy) == extract.compute_drawing_id(simple_doc)

    changed = tmp_path / "changed.dxf"
    changed.write_bytes(simple_doc.read_bytes() + b"\n999\nX\n")
    assert extract.compute_drawing_id(changed) != extract.compute_drawing_id(simple_doc)

    assert len(extract.compute_drawing_id(simple_doc)) == 16


def test_every_entity_gets_a_handle(simple_doc: Path):
    """A handle is the primary key; an entity without one is unusable."""
    _drawing, entities = extract.extract(simple_doc)
    assert entities
    for entity in entities:
        assert entity.handle, f"{entity.type} has no handle"
        assert isinstance(entity.handle, str)
        assert entity._id == f"{entity.drawing_id}:{entity.handle}"


def test_handles_are_unique(simple_doc: Path):
    """`(drawing_id, handle)` is the real key; duplicates would collide."""
    _drawing, entities = extract.extract(simple_doc)
    handles = [e.handle for e in entities]
    assert len(handles) == len(set(handles))


def test_paperspace_content_is_not_missed(paperspace_only_doc: Path):
    """The regression this whole design exists to prevent."""
    drawing, entities = extract.extract(paperspace_only_doc)
    assert drawing.entity_count > 0, (
        "content in a paperspace layout was reported as an empty drawing -- "
        "extraction is walking doc.modelspace() instead of doc.layouts"
    )
    assert any(e.layout != "Model" for e in entities)


def test_insert_records_its_block_name(simple_doc: Path):
    """One INSERT = one handle = one commentable object."""
    _drawing, entities = extract.extract(simple_doc)
    inserts = [e for e in entities if e.type == "INSERT"]
    assert len(inserts) == 1
    assert inserts[0].block_name == "DOOR"

    # Non-INSERT entities must not carry a block name.
    for entity in entities:
        if entity.type != "INSERT":
            assert entity.block_name is None


def test_text_is_extracted(simple_doc: Path):
    """Annotation text is what most real questions are actually about."""
    _drawing, entities = extract.extract(simple_doc)
    texts = [e.text for e in entities if e.text]
    assert "ROOM 101" in texts


def test_extents_are_computed_and_finite(simple_doc: Path):
    """Never the 1e+20 header sentinel, which empties the SVG viewBox."""
    drawing, _entities = extract.extract(simple_doc)
    assert drawing.extents is not None
    for value in drawing.extents["min"] + drawing.extents["max"]:
        assert math.isfinite(value)
        assert abs(value) < 1e19, "the 1e+20 sentinel reached the extents"
    assert drawing.extents["min"][0] < drawing.extents["max"][0]


def test_header_sentinel_is_rejected(tmp_path: Path):
    """A file whose header claims 1e+20 must still yield sane extents."""
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$EXTMIN"] = (1e20, 1e20, 1e20)
    doc.header["$EXTMAX"] = (-1e20, -1e20, -1e20)
    doc.modelspace().add_line((0, 0), (4, 3))
    path = tmp_path / "sentinel.dxf"
    doc.saveas(path)

    drawing, _entities = extract.extract(path)
    assert drawing.extents is not None
    assert drawing.extents["max"][0] == pytest.approx(4, abs=0.01)
    assert drawing.extents["max"][1] == pytest.approx(3, abs=0.01)


def test_units_are_reported_by_name(simple_doc: Path):
    drawing, _entities = extract.extract(simple_doc)
    assert drawing.units_code == 6
    assert drawing.units_name == "m"


def test_undeclared_units_are_reported_not_guessed(tmp_path: Path):
    """`$INSUNITS = 0` means "not stated", which is information, not a default."""
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 0
    doc.modelspace().add_line((0, 0), (1, 1))
    path = tmp_path / "unitless.dxf"
    doc.saveas(path)

    drawing, _entities = extract.extract(path)
    assert drawing.units_code == 0
    assert drawing.units_name == "unitless"


def test_bbox_and_measurements(simple_doc: Path):
    """Geometry that feeds spatial queries and the entity panel."""
    _drawing, entities = extract.extract(simple_doc)

    line = next(
        e for e in entities if e.type == "LINE" and e.length and e.length > 9
    )
    assert line.length == pytest.approx(10.0, abs=0.001)
    assert line.bbox is not None
    assert line.bbox_centre is not None

    circle = next(e for e in entities if e.type == "CIRCLE")
    assert circle.area == pytest.approx(math.pi * 4, rel=0.001)
    assert circle.length == pytest.approx(2 * math.pi * 2, rel=0.001)


def test_counts_are_consistent(simple_doc: Path):
    """The summary counts must add up to the entities actually stored."""
    drawing, entities = extract.extract(simple_doc)
    assert drawing.entity_count == len(entities)
    assert sum(drawing.counts_by_type.values()) == len(entities)
    assert sum(drawing.counts_by_layer.values()) == len(entities)


def test_unreadable_file_raises_extraction_error(tmp_path: Path):
    """A bad file must fail loudly with a typed error, never silently."""
    path = tmp_path / "broken.dxf"
    path.write_text("this is not a DXF at all", encoding="utf-8")
    with pytest.raises(extract.ExtractionError):
        extract.extract(path)


def test_drawing_id_hashes_the_original_source_not_the_converted_dxf(tmp_path):
    """Re-converting the same DWG must not fork a new drawing identity.

    Converter output is not byte-deterministic (ODA stamps time + GUID), so
    hashing the converted DXF orphaned the comments on every re-ingest.
    """
    import ezdxf
    from app.extract import compute_drawing_id, extract

    dxf_a = tmp_path / "conv_a.dxf"
    dxf_b = tmp_path / "conv_b.dxf"
    for p, note in ((dxf_a, "a"), (dxf_b, "b")):
        doc = ezdxf.new("R2018")
        doc.modelspace().add_text(note)  # different bytes per "conversion"
        doc.saveas(p)
    original = tmp_path / "original.dwg"
    original.write_bytes(b"stable dwg bytes")

    d1, _ = extract(dxf_a, source_path=str(original))
    d2, _ = extract(dxf_b, source_path=str(original))
    assert d1._id == d2._id == compute_drawing_id(original)
