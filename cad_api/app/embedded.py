"""Documents that live inside a drawing and that no DXF renderer draws.

A DXF can carry whole files inside itself. `OLE2FRAME` holds an embedded OLE
object -- a workbook, a Word page, a metafile, a picture pasted from Paint --
and every renderer in this stack skips it, because rendering it would mean
implementing the application that made it. The viewer has always said so, in
those words: *"the payload lives outside the drawing geometry"*.

It said so for three objects in the reference drawing, and nobody opened them.
Two of them are 11 MB bitmaps: the approvals sheet, and the **land-use key** --
the page that says, in Arabic, that plots 1-15 are commercial, 22-29 are
educational, and the remaining plots are residential villas. The pipeline had
spent weeks inferring that same fact by elimination and grading it `inferred`,
while the drawing stated it in a picture nobody had looked at.

That is the general lesson, and it is why this module has no drawing-specific
anything in it. What a layer code means is not written in the layer name; it is
written on a sheet, and the sheet is often pasted in. Any drawing can carry
one.

What comes out is reported as **what it is** -- `xlsx`, `xls`, `bmp`, `wmf`,
`cfb`, `pdf` -- never as "an image". The first payload here happened to be a
bitmap; assuming the next one is too would be a guess dressed as a fact, and
the whole point of opening a container is to stop guessing about it.

Pure: no MongoDB, no configuration, nothing from `app.*`. Give it a path, get
back what is inside.
"""

from __future__ import annotations

import binascii
import io
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

#: Entities whose payload is an embedded document rather than geometry.
#:
#: `IMAGE` is deliberately absent. It REFERENCES an external raster through an
#: `IMAGEDEF`, so those bytes are not in the file at all -- listing it would
#: promise content the drawing does not carry. What this module extracts is
#: only what the file itself holds.
PAYLOAD_TYPES = frozenset({"OLE2FRAME", "OLEFRAME"})

#: Group code carrying binary chunks inside an entity.
BINARY_CODE = "310"

#: Format signatures, longest first so a two-byte one cannot shadow a
#: container. Read from the bytes, never from a name: an extension inside a
#: container is a label the author typed.
SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "cfb", "application/x-cfb"),
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"PK\x03\x04", "zip", "application/zip"),
    (b"\xd7\xcd\xc6\x9a", "wmf", "image/wmf"),
    (b"{\\rtf", "rtf", "application/rtf"),
    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
    (b"GIF8", "gif", "image/gif"),
    (b"II*\x00", "tif", "image/tiff"),
    (b"MM\x00*", "tif", "image/tiff"),
    (b"%PDF-", "pdf", "application/pdf"),
    (b"\x01\x00\x09\x00", "wmf", "image/wmf"),
    (b"BM", "bmp", "image/bmp"),
)

#: An enhanced metafile is recognised by a leading record type of 1 together
#: with the signature ` EMF` at offset 40 -- not by its first bytes, which
#: carry no distinctive value and would match any zero-padded blob.
EMF_SIGNATURE_OFFSET = 40
EMF_SIGNATURE = b" EMF"

#: What a compound document holds, decided by the streams inside it rather
#: than by anything written on the outside.
CFB_CONTENTS: tuple[tuple[str, str, str], ...] = (
    ("Workbook", "xls", "application/vnd.ms-excel"),
    ("Book", "xls", "application/vnd.ms-excel"),
    ("WordDocument", "doc", "application/msword"),
    ("PowerPoint Document", "ppt", "application/vnd.ms-powerpoint"),
    ("VisioDocument", "vsd", "application/vnd.visio"),
)

#: What a zip holds. An `xlsx` is a zip with an `xl/` folder in it; nothing
#: else distinguishes the Office formats from an ordinary archive.
ZIP_CONTENTS: tuple[tuple[str, str, str], ...] = (
    (
        "xl/",
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    (
        "word/",
        "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    (
        "ppt/",
        "pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
)

#: Streams that hold an embedded file rather than describing one.
CONTENT_STREAMS = ("\x01Ole10Native", "CONTENTS", "Package")

#: How far in to look for a container signature.
#:
#: An `OLE2FRAME` payload does not begin at the document: AutoCAD writes its
#: own header first, and in the reference drawing the compound-document
#: signature sits 128 bytes in. Testing only offset zero reads an ordinary
#: embedded object as "not a container", which is how two 11 MB documents
#: stayed shut.
CONTAINER_SEARCH_BYTES = 64 * 1024

#: `BITMAPINFOHEADER` size, and the bit depths a DIB may declare.
DIB_HEADER_BYTES = 40
DIB_BIT_DEPTHS = frozenset({1, 4, 8, 16, 24, 32})

#: Cap on one payload. A drawing that embeds something larger than this is
#: telling us something we should report rather than load into memory.
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Payload:
    """One embedded document, ready to be written out or served."""

    handle: str
    kind: str
    fmt: str
    media_type: str
    data: bytes
    width: int | None = None
    height: int | None = None
    note: str | None = None

    @property
    def filename(self) -> str:
        return f"{self.handle}.{self.fmt}"

    def as_dict(self) -> dict[str, object]:
        return {
            "handle": self.handle,
            "kind": self.kind,
            "format": self.fmt,
            "media_type": self.media_type,
            "bytes": len(self.data),
            "width": self.width,
            "height": self.height,
            "note": self.note,
        }


# --- what the bytes are ------------------------------------------------------


def sniff(data: bytes) -> tuple[str, str]:
    """Format and media type from the bytes themselves.

    Containers are opened one level further, because "a compound document" is
    not a useful answer when the compound document is a workbook.
    """
    if _is_emf(data):
        return "emf", "image/emf"
    for sig, fmt, media in SIGNATURES:
        if data.startswith(sig):
            if fmt == "cfb":
                return _refine_cfb(data)
            if fmt == "zip":
                return _refine_zip(data)
            return fmt, media
    return "bin", "application/octet-stream"


def _is_emf(data: bytes) -> bool:
    """An enhanced metafile, recognised where the format actually says so."""
    if len(data) < EMF_SIGNATURE_OFFSET + 4:
        return False
    if struct.unpack_from("<I", data, 0)[0] != 1:  # EMR_HEADER
        return False
    return data[EMF_SIGNATURE_OFFSET : EMF_SIGNATURE_OFFSET + 4] == EMF_SIGNATURE


def _refine_cfb(data: bytes) -> tuple[str, str]:
    """A compound document, named by what is inside it."""
    try:
        import olefile

        ole = olefile.OleFileIO(io.BytesIO(data))
    except Exception:  # noqa: BLE001 -- an unreadable container is still a cfb
        return "cfb", "application/x-cfb"
    try:
        names = {"/".join(entry) for entry in ole.listdir()}
    finally:
        ole.close()
    for stream, fmt, media in CFB_CONTENTS:
        if stream in names:
            return fmt, media
    return "cfb", "application/x-cfb"


def _refine_zip(data: bytes) -> tuple[str, str]:
    """A zip, named by the folders inside it."""
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
    except Exception:  # noqa: BLE001 -- a truncated archive is still a zip
        return "zip", "application/zip"
    for prefix, fmt, media in ZIP_CONTENTS:
        if any(name.startswith(prefix) for name in names):
            return fmt, media
    return "zip", "application/zip"


def bitmap_size(data: bytes) -> tuple[int | None, int | None]:
    """Pixel size of a BMP, or `(None, None)`.

    Read rather than assumed, and validated: a scan for `BM` at a guessed
    offset once produced a header claiming 1,296,498,688 pixels across, which
    is what a coincidence looks like when nobody checks it.
    """
    if not data.startswith(b"BM") or len(data) < 30:
        return None, None
    try:
        width, height = struct.unpack_from("<ii", data, 18)
    except struct.error:
        return None, None
    if not (0 < width < 100_000 and 0 < abs(height) < 100_000):
        return None, None
    return width, abs(height)


def dib_to_bmp(data: bytes) -> bytes | None:
    """A header-less DIB given the 14 bytes that make it a `.bmp`, or `None`.

    Paint hands OLE a device-independent bitmap: the pixel description and the
    pixels, with no file header, because inside a container there is no file.
    Written to disk unchanged it is a `.bin` that nothing opens -- the
    difference between "this drawing embeds something" and being able to read
    what the drawing says.

    This is the one transformation this module performs, and it is not a
    guess: every field of the header is parsed and must agree before the bytes
    are called a bitmap.
    """
    if len(data) < DIB_HEADER_BYTES:
        return None
    header_size = struct.unpack_from("<I", data, 0)[0]
    if header_size != DIB_HEADER_BYTES:
        return None
    width, height, planes, bits = struct.unpack_from("<iihh", data, 4)
    if not (0 < width < 100_000 and 0 < abs(height) < 100_000):
        return None
    if planes != 1 or bits not in DIB_BIT_DEPTHS:
        return None
    palette = 0 if bits > 8 else (1 << bits) * 4
    offset = 14 + DIB_HEADER_BYTES + palette
    return b"BM" + struct.pack("<IHHI", 14 + len(data), 0, 0, offset) + data


def _promote(data: bytes) -> tuple[bytes, str | None]:
    """Bytes in the most openable form they can honestly take."""
    if sniff(data)[0] != "bin":
        return data, None
    bmp = dib_to_bmp(data)
    if bmp is not None:
        return bmp, "a BMP file header was added to a DIB stored without one"
    return data, None


# --- the container -----------------------------------------------------------


def _container_offset(data: bytes) -> int:
    """Where a compound document starts inside a payload, or `-1`."""
    return data.find(SIGNATURES[0][0], 0, CONTAINER_SEARCH_BYTES)


def _unwrap_ole(data: bytes) -> tuple[bytes, str | None]:
    """The document inside an OLE container, or the payload unchanged.

    Three shapes occur and none of them is preferred. A packaged file arrives
    in `Ole10Native` behind a four-byte length prefix. An embedded application
    object -- a workbook, a Word page -- arrives as a compound document that
    IS the file, so the container itself is what comes out. Anything else is
    reported as the compound document it is, rather than being searched until
    something looks like a picture.
    """
    at = _container_offset(data)
    if at < 0:
        return _promote(data)

    try:
        import olefile
    except ImportError:  # pragma: no cover -- declared in requirements
        return data, "olefile is not installed, so the OLE container was not opened"

    body = data[at:]
    try:
        ole = olefile.OleFileIO(io.BytesIO(body))
    except Exception as exc:  # noqa: BLE001 -- a broken container is not fatal
        return data, f"the OLE container could not be opened: {type(exc).__name__}"

    try:
        names = {"/".join(entry) for entry in ole.listdir()}
    except Exception:  # noqa: BLE001
        ole.close()
        return body, "the contents of the OLE container could not be read"

    # A workbook or a document is not something to unpack: the container is
    # the file. Handing back a stream from inside it would produce bytes that
    # no application opens.
    for stream, _fmt, _media in CFB_CONTENTS:
        if stream in names:
            ole.close()
            return body, None

    try:
        for name in CONTENT_STREAMS:
            if name not in names:
                continue
            raw = ole.openstream(name).read()
            if not raw:
                continue
            if name == "\x01Ole10Native" and len(raw) > 4:
                declared = struct.unpack_from("<I", raw, 0)[0]
                raw = raw[4 : 4 + declared] if 0 < declared <= len(raw) - 4 else raw[4:]
            return _promote(raw)
        return body, "the OLE container has no recognised content stream"
    finally:
        ole.close()


# --- reading the file --------------------------------------------------------


def _chunks_from_dxf(path: Path) -> Iterator[tuple[str, bytes]]:
    """Every binary payload in the file, streamed.

    The DXF is read as text and never loaded whole: the reference drawing is
    136 MB and holds 22 MB of embedded documents. `ezdxf` is not used here
    because it exposes no blob for `OLE2FRAME` -- the bytes are reachable only
    as group-310 hex under the entity.
    """
    handle: str | None = None
    inside = False
    hex_parts: list[str] = []
    size = 0

    def flush() -> tuple[str, bytes]:
        """What this entity holds -- possibly nothing, which is an answer."""
        if not hex_parts:
            return (handle or "?", b"")
        try:
            return (handle or "?", binascii.unhexlify("".join(hex_parts)))
        except binascii.Error:
            return (handle or "?", b"")

    for code, value in _pairs(path):
        if code == "0":
            if inside:
                yield flush()
                hex_parts, handle, size = [], None, 0
            inside = value in PAYLOAD_TYPES
        elif inside and code == "5" and handle is None:
            handle = value
        elif inside and code == BINARY_CODE:
            if re.fullmatch(r"[0-9A-Fa-f]*", value) and len(value) % 2 == 0:
                size += len(value) // 2
                if size <= MAX_PAYLOAD_BYTES:
                    hex_parts.append(value)

    if inside:
        yield flush()


def _pairs(path: Path) -> Iterator[tuple[str, str]]:
    """A DXF as the alternating (code, value) lines it actually is.

    Not "the previous line looked numeric, so it was a group code": a payload
    chunk of `0000000000` is all digits and would then be read as a code,
    silently truncating the document it belongs to.
    """
    code: str | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            text = line.rstrip("\r\n")
            if code is None:
                code = text.strip()
                continue
            yield code, text.strip()
            code = None


def payloads(path: Path) -> list[Payload]:
    """Every embedded document in a drawing, unwrapped and identified."""
    out: list[Payload] = []
    for handle, raw in _chunks_from_dxf(path):
        if not raw:
            # An OLE frame whose content is linked rather than stored. Saying
            # so is the honest answer; dropping it would make the count on the
            # entities disagree with this list and leave a reader to work out
            # which of the two was lying.
            out.append(
                Payload(
                    handle=handle,
                    kind="ole",
                    fmt="none",
                    media_type="application/octet-stream",
                    data=b"",
                    note=(
                        "this entity declares an embedded object but carries no "
                        "bytes inside the drawing -- its content is linked, "
                        "not stored"
                    ),
                )
            )
            continue
        was_container = _container_offset(raw) >= 0
        body, note = _unwrap_ole(raw)
        fmt, media = sniff(body)
        width, height = bitmap_size(body)
        parts = [
            "extracted from an OLE container" if was_container else "",
            note or "",
        ]
        out.append(
            Payload(
                handle=handle,
                kind="ole",
                fmt=fmt,
                media_type=media,
                data=body,
                width=width,
                height=height,
                note=" ".join(p for p in parts if p) or None,
            )
        )
    return out


def write_all(path: Path, into: Path) -> list[dict[str, object]]:
    """Extract and cache to disk. Returns one row per payload.

    Writing is what makes this usable: an 11 MB document is not something to
    hold in a response, and a reader wants to open it rather than be told it
    exists.
    """
    into.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for payload in payloads(path):
        row = payload.as_dict()
        if payload.data:
            target = into / payload.filename
            target.write_bytes(payload.data)
            row["file"] = target.name
        else:
            row["file"] = None
        rows.append(row)
    return rows
