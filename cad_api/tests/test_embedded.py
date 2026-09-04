"""What a drawing carries inside itself.

Every case here is built from bytes rather than from the reference drawing,
because the point of the module is that it knows nothing about any particular
file. The one measured fact from the real drawing -- an ``Ole10Native`` stream
holding a 24-bit DIB with no file header -- is reproduced as a fixture so the
shape that actually occurred is the shape under test.
"""

from __future__ import annotations

import io
import struct

import pytest

from app import embedded

OLE_NATIVE = "\x01Ole10Native"


def _dib(width: int, height: int, bits: int = 24) -> bytes:
    """A ``BITMAPINFOHEADER`` and enough pixels to match it."""
    row = ((width * bits + 31) // 32) * 4
    header = struct.pack(
        "<IiihhIIiiII", 40, width, height, 1, bits, 0, row * height, 0, 0, 0, 0
    )
    return header + b"\xc8" * (row * height)


def _bmp(width: int, height: int, bits: int = 24) -> bytes:
    body = _dib(width, height, bits)
    return b"BM" + struct.pack("<IHHI", 14 + len(body), 0, 0, 54) + body


def _minimal_ole(streams: dict[str, bytes]) -> bytes:
    """Build a compound document without a writer library.

    ``olefile`` reads but does not write, so the fixture is assembled from the
    format itself: a 512-byte header, one FAT sector, a directory, and the
    stream data. Only the single-stream case above the mini-stream cutoff is
    needed, which keeps this to the parts the reader actually walks.
    """
    sector = 512
    payload = b""
    entries = []
    for name, data in streams.items():
        start = len(payload) // sector
        payload += data + b"\x00" * (-len(data) % sector)
        entries.append((name, start, len(data)))

    n_payload = len(payload) // sector
    fat: list[int] = []
    for _name, start, size in entries:
        count = max(1, (size + sector - 1) // sector)
        for i in range(count):
            fat.append(start + i + 1 if i + 1 < count else 0xFFFFFFFE)
    dir_sector = n_payload
    fat.append(0xFFFFFFFE)  # the directory sector
    fat.append(0xFFFFFFFD)  # the FAT sector describes itself
    fat += [0xFFFFFFFF] * (sector // 4 - len(fat))
    fat_bytes = b"".join(struct.pack("<I", v) for v in fat)

    def entry(name: str, kind: int, start: int, size: int, child: int = -1) -> bytes:
        raw = name.encode("utf-16-le") + b"\x00\x00"
        block = bytearray(128)
        block[0 : len(raw)] = raw
        struct.pack_into("<H", block, 64, len(raw))
        block[66] = kind  # 2 stream, 5 root
        block[67] = 1  # black
        struct.pack_into("<i", block, 68, -1)
        struct.pack_into("<i", block, 72, -1)
        struct.pack_into("<i", block, 76, child)
        struct.pack_into("<I", block, 116, start)
        struct.pack_into("<I", block, 120, size)
        return bytes(block)

    directory = bytearray()
    directory += entry("Root Entry", 5, 0xFFFFFFFE, 0, child=1)
    for name, start, size in entries:
        directory += entry(name, 2, start, size)
    if len(directory) % sector:
        directory += b"\x00" * (sector - len(directory) % sector)

    header = bytearray(512)
    header[0:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    struct.pack_into("<H", header, 24, 0x003E)
    struct.pack_into("<H", header, 26, 0x0003)
    struct.pack_into("<H", header, 28, 0xFFFE)
    struct.pack_into("<H", header, 30, 9)  # 512-byte sectors
    struct.pack_into("<H", header, 32, 6)  # 64-byte mini sectors
    struct.pack_into("<I", header, 44, 1)  # one FAT sector
    struct.pack_into("<I", header, 48, dir_sector)
    struct.pack_into("<I", header, 56, 4096)  # mini-stream cutoff
    struct.pack_into("<I", header, 60, 0xFFFFFFFE)
    struct.pack_into("<I", header, 64, 0)
    struct.pack_into("<I", header, 68, 0xFFFFFFFE)
    struct.pack_into("<I", header, 72, 0)
    struct.pack_into("<I", header, 76, n_payload + 1)

    return bytes(header) + payload + bytes(directory) + fat_bytes


def _dxf(entities: list[tuple[str, str, bytes | None]]) -> str:
    """A DXF fragment holding the given entities, hex-encoded like AutoCAD."""
    out = ["0", "SECTION", "2", "ENTITIES"]
    for kind, handle, blob in entities:
        out += ["0", kind, "5", handle]
        if blob is not None:
            hexed = blob.hex().upper()
            for i in range(0, len(hexed), 254):
                out += ["310", hexed[i : i + 254]]
    out += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(out) + "\n"


# --- what the bytes are ------------------------------------------------------


def test_sniff_reads_the_signature_not_the_extension():
    assert embedded.sniff(b"\x89PNG\r\n\x1a\n rest") == ("png", "image/png")
    assert embedded.sniff(_bmp(4, 4))[0] == "bmp"
    assert embedded.sniff(b"nothing familiar") == ("bin", "application/octet-stream")


def test_container_signature_wins_over_bm():
    """``BM`` is two bytes and would shadow anything ordered after it."""
    assert embedded.sniff(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")[0] == "cfb"


def test_a_workbook_is_reported_as_a_workbook_not_as_a_container():
    """The type is what the bytes are, not the box they arrived in.

    A compound document holding a ``Workbook`` stream is an Excel file. Saying
    "cfb" would be true and useless; saying "an image" -- which an extractor
    that hunts for pictures ends up doing -- would be false.
    """
    olefile = pytest.importorskip("olefile")
    container = _minimal_ole({"Workbook": b"\x09\x08" + b"\x00" * 600})
    if not olefile.isOleFile(io.BytesIO(container)):
        pytest.skip("the hand-built container is not readable by this olefile")
    assert embedded.sniff(container) == ("xls", "application/vnd.ms-excel")


def test_an_xlsx_is_told_apart_from_an_ordinary_zip():
    import zipfile

    book = io.BytesIO()
    with zipfile.ZipFile(book, "w") as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
    assert embedded.sniff(book.getvalue())[0] == "xlsx"

    plain = io.BytesIO()
    with zipfile.ZipFile(plain, "w") as archive:
        archive.writestr("notes.txt", "hello")
    assert embedded.sniff(plain.getvalue())[0] == "zip"


def test_emf_is_recognised_where_the_format_says_so():
    """Its first four bytes are a record type, not a distinctive signature."""
    emf = struct.pack("<I", 1) + b"\x00" * 36 + b" EMF" + b"\x00" * 20
    assert embedded.sniff(emf) == ("emf", "image/emf")
    assert embedded.sniff(b"\x00" * 60)[0] == "bin"


def test_a_workbook_container_is_not_unpacked_into_a_stream(tmp_path):
    """The container IS the file; a stream taken out of it opens in nothing."""
    olefile = pytest.importorskip("olefile")
    container = _minimal_ole({"Workbook": b"\x09\x08" + b"\x00" * 600})
    if not olefile.isOleFile(io.BytesIO(container)):
        pytest.skip("the hand-built container is not readable by this olefile")
    path = tmp_path / "book.dxf"
    path.write_text(
        _dxf([("OLE2FRAME", "XL", b"\x00" * 128 + container)]), encoding="utf-8"
    )
    found = embedded.payloads(path)
    assert len(found) == 1
    assert found[0].fmt == "xls"
    assert found[0].data.startswith(b"\xd0\xcf\x11\xe0")
    assert found[0].width is None


def test_bitmap_size_is_read_not_guessed():
    assert embedded.bitmap_size(_bmp(1749, 2066)) == (1749, 2066)


def test_bitmap_size_refuses_an_absurd_header():
    """A scan for ``BM`` at a guessed offset once claimed 1,296,498,688 px."""
    broken = bytearray(_bmp(4, 4))
    struct.pack_into("<i", broken, 18, 1_296_498_688)
    assert embedded.bitmap_size(bytes(broken)) == (None, None)


def test_bitmap_size_of_a_non_bitmap_is_unknown():
    assert embedded.bitmap_size(b"%PDF-1.7") == (None, None)


def test_negative_height_is_a_top_down_bitmap_not_an_error():
    raw = bytearray(_bmp(10, 20))
    struct.pack_into("<i", raw, 22, -20)
    assert embedded.bitmap_size(bytes(raw)) == (10, 20)


# --- the header-less DIB -----------------------------------------------------


def test_dib_becomes_an_openable_bmp():
    out = embedded.dib_to_bmp(_dib(30, 40))
    assert out is not None
    assert out.startswith(b"BM")
    assert embedded.bitmap_size(out) == (30, 40)


def test_dib_pixels_are_where_the_header_says_they_are():
    out = embedded.dib_to_bmp(_dib(30, 40))
    assert out is not None
    offset = struct.unpack_from("<I", out, 10)[0]
    row = ((30 * 24 + 31) // 32) * 4
    assert len(out) - offset >= row * 40


def test_palette_is_counted_for_low_bit_depths():
    """8 bpp carries 256 palette entries between the header and the pixels."""
    out = embedded.dib_to_bmp(_dib(8, 8, bits=8))
    assert out is not None
    assert struct.unpack_from("<I", out, 10)[0] == 14 + 40 + 256 * 4


def test_a_bmp_is_not_wrapped_twice():
    assert embedded.dib_to_bmp(_bmp(4, 4)) is None


def test_random_bytes_are_not_promoted_to_an_image():
    assert embedded.dib_to_bmp(b"\x28\x00\x00\x00" + b"\xff" * 100) is None
    assert embedded.dib_to_bmp(b"short") is None


# --- reading the file --------------------------------------------------------


def test_payload_is_found_under_the_entity(tmp_path):
    path = tmp_path / "one.dxf"
    path.write_text(_dxf([("OLE2FRAME", "AB12", _bmp(12, 9))]), encoding="utf-8")
    found = embedded.payloads(path)
    assert [(p.handle, p.fmt, p.width, p.height) for p in found] == [
        ("AB12", "bmp", 12, 9)
    ]


def test_only_payload_entities_are_read(tmp_path):
    path = tmp_path / "mixed.dxf"
    path.write_text(
        _dxf(
            [
                ("LWPOLYLINE", "1", None),
                ("OLE2FRAME", "2", _bmp(4, 4)),
                ("TEXT", "3", None),
            ]
        ),
        encoding="utf-8",
    )
    assert [p.handle for p in embedded.payloads(path)] == ["2"]


def test_several_payloads_keep_their_own_handles(tmp_path):
    path = tmp_path / "two.dxf"
    path.write_text(
        _dxf([("OLE2FRAME", "A", _bmp(4, 4)), ("OLEFRAME", "B", _bmp(6, 6))]),
        encoding="utf-8",
    )
    assert [(p.handle, p.width) for p in embedded.payloads(path)] == [
        ("A", 4),
        ("B", 6),
    ]


def test_a_frame_with_no_bytes_is_reported_not_dropped(tmp_path):
    """The reference drawing has one: a frame whose content is linked.

    Silently dropping it would make the entity count disagree with this list,
    and leave a reader to work out which of the two was lying.
    """
    path = tmp_path / "linked.dxf"
    path.write_text(
        _dxf([("OLE2FRAME", "EMPTY", None), ("OLE2FRAME", "FULL", _bmp(4, 4))]),
        encoding="utf-8",
    )
    found = embedded.payloads(path)
    assert [p.handle for p in found] == ["EMPTY", "FULL"]
    assert found[0].data == b""
    assert "linked" in (found[0].note or "")


def test_the_last_entity_in_the_file_is_not_lost(tmp_path):
    """Nothing follows it to trigger the flush, so the end of file must."""
    path = tmp_path / "last.dxf"
    text = "\n".join(["0", "OLE2FRAME", "5", "Z9", "310", _bmp(5, 5).hex().upper()])
    path.write_text(text + "\n", encoding="utf-8")
    assert [p.handle for p in embedded.payloads(path)] == ["Z9"]


def test_hex_split_across_lines_is_rejoined(tmp_path):
    """AutoCAD writes 254 characters a line; an 11 MB picture is 43k lines."""
    blob = _bmp(200, 50)
    path = tmp_path / "split.dxf"
    path.write_text(_dxf([("OLE2FRAME", "S", blob)]), encoding="utf-8")
    found = embedded.payloads(path)
    assert len(found) == 1
    assert found[0].data == blob


def test_odd_length_hex_is_skipped_rather_than_crashing(tmp_path):
    path = tmp_path / "odd.dxf"
    path.write_text(
        "\n".join(
            ["0", "SECTION", "0", "OLE2FRAME", "5", "X", "310", "ABC", "0", "EOF"]
        ),
        encoding="utf-8",
    )
    found = embedded.payloads(path)
    assert [p.handle for p in found] == ["X"]
    assert found[0].data == b""


def test_a_file_with_nothing_embedded_yields_nothing(tmp_path):
    path = tmp_path / "plain.dxf"
    path.write_text(
        _dxf([("LINE", "1", None), ("CIRCLE", "2", None)]), encoding="utf-8"
    )
    assert embedded.payloads(path) == []


# --- the container -----------------------------------------------------------


def test_container_offset_is_searched_not_assumed():
    """AutoCAD writes its own header first; in the real file it is 128 bytes."""
    body = b"\x00" * 128 + b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    assert embedded._container_offset(body) == 128
    assert embedded._container_offset(b"no container here") == -1


def test_container_is_not_searched_past_the_cap():
    body = (
        b"\x00" * (embedded.CONTAINER_SEARCH_BYTES + 10)
        + b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    )
    assert embedded._container_offset(body) == -1


def test_ole_stream_holding_a_dib_comes_out_as_a_bitmap(tmp_path):
    """The shape measured in the reference drawing, rebuilt from bytes."""
    olefile = pytest.importorskip("olefile")
    inner = _dib(64, 48)
    container = _minimal_ole({OLE_NATIVE: struct.pack("<I", len(inner)) + inner})
    if not olefile.isOleFile(io.BytesIO(container)):
        pytest.skip("the hand-built container is not readable by this olefile")
    blob = b"\x00" * 128 + container
    path = tmp_path / "ole.dxf"
    path.write_text(_dxf([("OLE2FRAME", "OLE1", blob)]), encoding="utf-8")
    found = embedded.payloads(path)
    assert len(found) == 1
    assert found[0].fmt == "bmp"
    assert (found[0].width, found[0].height) == (64, 48)
    assert "OLE container" in (found[0].note or "")


def test_a_container_that_will_not_open_is_reported_not_hidden(tmp_path):
    blob = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600
    path = tmp_path / "broken.dxf"
    path.write_text(_dxf([("OLE2FRAME", "BAD", blob)]), encoding="utf-8")
    found = embedded.payloads(path)
    assert len(found) == 1
    assert found[0].note and "OLE" in found[0].note
    assert found[0].data == blob


# --- writing them out --------------------------------------------------------


def test_write_all_names_files_by_handle_and_format(tmp_path):
    path = tmp_path / "w.dxf"
    path.write_text(_dxf([("OLE2FRAME", "7F3", _bmp(8, 8))]), encoding="utf-8")
    rows = embedded.write_all(path, tmp_path / "out")
    assert rows[0]["file"] == "7F3.bmp"
    written = (tmp_path / "out" / "7F3.bmp").read_bytes()
    assert embedded.bitmap_size(written) == (8, 8)


def test_write_all_writes_no_file_for_an_empty_frame(tmp_path):
    path = tmp_path / "e.dxf"
    path.write_text(_dxf([("OLE2FRAME", "NONE", None)]), encoding="utf-8")
    rows = embedded.write_all(path, tmp_path / "out")
    assert rows[0]["file"] is None
    assert list((tmp_path / "out").iterdir()) == []


def test_row_carries_what_a_reader_needs_to_decide_to_open_it(tmp_path):
    path = tmp_path / "r.dxf"
    path.write_text(_dxf([("OLE2FRAME", "R1", _bmp(20, 10))]), encoding="utf-8")
    row = embedded.write_all(path, tmp_path / "out")[0]
    assert row["bytes"] == len(_bmp(20, 10))
    assert row["media_type"] == "image/bmp"
    assert (row["width"], row["height"]) == (20, 10)


def test_image_entities_are_deliberately_not_claimed():
    """``IMAGE`` references an external raster; those bytes are not in here."""
    assert "IMAGE" not in embedded.PAYLOAD_TYPES
    assert embedded.PAYLOAD_TYPES == {"OLE2FRAME", "OLEFRAME"}


def test_an_oversized_payload_stops_growing(tmp_path, monkeypatch):
    monkeypatch.setattr(embedded, "MAX_PAYLOAD_BYTES", 64)
    path = tmp_path / "big.dxf"
    path.write_text(_dxf([("OLE2FRAME", "BIG", _bmp(40, 40))]), encoding="utf-8")
    found = embedded.payloads(path)
    assert len(found) == 1
    assert len(found[0].data) <= 64
