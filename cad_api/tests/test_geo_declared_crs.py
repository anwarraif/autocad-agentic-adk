"""GEO-H3 G1, BINDING 1 — a file that states its coordinate system.

None of the drawings in the store declares one. Janadriyah does not, which is
why a zone had to be inferred for it and why every position it produces
carries a warning. So the declared path has no real file to be tested against,
and the choice is between testing it on something synthetic or shipping it
untested until the drawing that needs it arrives — at which point the failure
would look like a problem with that drawing.

Everything here is therefore built in a temp directory: a real DXF, written by
ezdxf, carrying a real GEODATA object with the XML coordinate-system
definition AutoCAD writes, read back through the real extractor. What it does
NOT do is write to Mongo. The shared cluster is not a fixture, and the claim
under test — "a declared CRS needs no configuration" — is a claim about
extraction and resolution, both of which take a drawing document rather than a
database.

The point on the ground is deliberately the mosque's own UTM centroid, so the
expected lat/long is a figure this repo has already cross-checked against
pyproj rather than one invented here.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

ezdxf = pytest.importorskip("ezdxf")

from app import extract, landuse, store_landuse
from app.landuse import crs as crs_math

#: The mosque's stored centroid, in drawing coordinates, and where UPLIFT-14
#: proved it lands. Reused rather than re-derived: a second reference point
#: invented for this test would be a second thing that can be wrong.
MOSQUE_UTM = (691332.0613834372, 2749824.7711861087)
MOSQUE_LAT_LON = (24.851418547, 46.893576218)

UTM_38N = 32638


def _csd(epsg: int, *, first_axis: str = "E") -> str:
    """The XML AutoCAD stores in a GEODATA object.

    Not a WKT string — that is the other convention, and reaching for it is
    how this gets written twice. ezdxf reads the EPSG out of an `Alias` whose
    namespace is "EPSG Code", and the axis order out of the first
    `CoordinateSystemAxis`; both are reproduced here exactly because both are
    what the parser looks for.
    """
    name = {"E": ("Easting", "east"), "N": ("Northing", "north")}[first_axis]
    other = ("Northing", "north") if first_axis == "E" else ("Easting", "east")
    other_abbr = "N" if first_axis == "E" else "E"
    return f"""<Dictionary version="1.0" xmlns="http://www.osgeo.org/mapguide/coordinatesystem">
<Alias id="{epsg}" type="CoordinateSystem"><ObjectId>SYNTHETIC</ObjectId><Namespace>EPSG Code</Namespace></Alias>
<Axis uom="METER">
<CoordinateSystemAxis><AxisOrder>1</AxisOrder><AxisName>{name[0]}</AxisName><AxisAbbreviation>{first_axis}</AxisAbbreviation><AxisDirection>{name[1]}</AxisDirection></CoordinateSystemAxis>
<CoordinateSystemAxis><AxisOrder>2</AxisOrder><AxisName>{other[0]}</AxisName><AxisAbbreviation>{other_abbr}</AxisAbbreviation><AxisDirection>{other[1]}</AxisDirection></CoordinateSystemAxis>
</Axis>
</Dictionary>"""


def _write_dxf(tmp_path: Path, csd: str | None, name: str = "declared.dxf") -> Path:
    """A minimal DXF around the mosque, optionally declaring a CRS."""
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 6  # metres, so nothing here depends on a default
    msp = doc.modelspace()
    east, north = MOSQUE_UTM
    msp.add_lwpolyline(
        [
            (east - 10, north - 10),
            (east + 10, north - 10),
            (east + 10, north + 10),
            (east - 10, north + 10),
        ],
        close=True,
        dxfattribs={"layer": "PARCEL"},
    )
    if csd is not None:
        geodata = msp.new_geodata()
        geodata.dxf.coordinate_type = 2  # local grid
        geodata.coordinate_system_definition = csd
    path = tmp_path / name
    doc.saveas(str(path))
    return path


@pytest.fixture
def declared(tmp_path):
    """The drawing document of a file that states EPSG:32638."""
    path = _write_dxf(tmp_path, _csd(UTM_38N))
    drawing, _entities = extract.extract(path)
    return drawing


# ---------------------------------------------------------------------------
# The extractor reads what the file says
# ---------------------------------------------------------------------------


def test_the_extractor_reads_a_declared_coordinate_system(declared):
    assert declared.declared_crs is not None
    assert declared.declared_crs["epsg"] == UTM_38N
    assert declared.declared_crs["xy_ordering"] is True
    assert declared.declared_crs["unreadable"] is None


def test_a_file_that_declares_nothing_claims_nothing(tmp_path):
    """`None`, not a zone and not an empty dict. Every drawing in the store
    today takes this path, and the whole reason the inferred-CRS caveat exists
    is that this path must never quietly acquire a default."""
    path = _write_dxf(tmp_path, None, name="silent.dxf")
    drawing, _ = extract.extract(path)
    assert drawing.declared_crs is None


def test_geodata_that_cannot_be_read_is_a_lead_not_a_silence(tmp_path):
    """G8. "This file says nothing" and "this file says something I could not
    read" are different facts, and only the second is worth chasing."""
    path = _write_dxf(tmp_path, "<Dictionary>not the schema</Dictionary>", name="broken.dxf")
    drawing, _ = extract.extract(path)
    assert drawing.declared_crs is not None
    assert drawing.declared_crs["epsg"] is None
    assert drawing.declared_crs["unreadable"]


# ---------------------------------------------------------------------------
# Zero configuration
# ---------------------------------------------------------------------------


def test_a_declared_crs_places_the_drawing_with_no_config_at_all(declared):
    """BINDING 1, and the whole point of the phase.

    `config=None` is not a convenience here — it is the claim. A drawing that
    arrives tomorrow has no entry in `landuse/`, nobody has looked at where
    its extents land, and it still has to come out on the map."""
    choice = store_landuse.crs_for_drawing(
        declared._id, drawing=vars(declared), config=None
    )
    assert choice.layer == "declared_in_file"
    assert choice.note is None
    assert choice.crs is not None
    assert choice.crs.epsg == UTM_38N
    assert choice.crs.declared_in_file is True

    lat, lon = choice.crs.to_lat_lon(*MOSQUE_UTM)
    assert lat == pytest.approx(MOSQUE_LAT_LON[0], abs=1e-5)
    assert lon == pytest.approx(MOSQUE_LAT_LON[1], abs=1e-5)


def test_a_declared_crs_drops_the_inferred_warning(declared):
    """The caveat is the difference between the two paths, so it is the thing
    worth asserting. A position from a zone the file STATES must not carry a
    sentence telling its reader not to rely on it."""
    choice = store_landuse.crs_for_drawing(
        declared._id, drawing=vars(declared), config=None
    )
    caveat = crs_math.caveat(choice.crs)
    assert "STATES" in caveat
    assert "INFERRED" not in caveat


def test_the_synthetic_drawing_really_has_no_config(declared):
    """Guards the test above from passing for the wrong reason. If this id
    somehow had a config file, `crs_for_drawing` could be reading that."""
    assert landuse.for_drawing(declared._id) is None


# ---------------------------------------------------------------------------
# Declarations this service will not act on
# ---------------------------------------------------------------------------


def test_a_projection_we_cannot_invert_is_refused_in_words(tmp_path):
    """EPSG:27700 is the British National Grid. Approximating it with a UTM
    series produces a position that is wrong by kilometres and shaped exactly
    like a right one, so the answer is no — with the reason attached."""
    path = _write_dxf(tmp_path, _csd(27700), name="osgb.dxf")
    drawing, _ = extract.extract(path)
    choice = store_landuse.crs_for_drawing(
        drawing._id, drawing=vars(drawing), config=None
    )
    assert choice.crs is None
    assert choice.layer is None
    assert "27700" in choice.note


def test_a_northing_first_declaration_is_refused_rather_than_swapped(tmp_path):
    """Swapping the axes would be a guess about what the author meant, and a
    wrong guess here still produces numbers shaped like a position."""
    path = _write_dxf(tmp_path, _csd(UTM_38N, first_axis="N"), name="northing.dxf")
    drawing, _ = extract.extract(path)
    choice = store_landuse.crs_for_drawing(
        drawing._id, drawing=vars(drawing), config=None
    )
    assert choice.crs is None
    assert "northing-first" in choice.note


def test_a_refused_declaration_falls_back_but_says_so(tmp_path):
    """The loud half of the fallback.

    A drawing placed by an inferred zone WHILE ITS OWN FILE says something
    else is the shape of a wrong map nobody questions. The config still wins
    the argument — that is what a fallback is — but the refusal travels with
    the answer."""
    path = _write_dxf(tmp_path, _csd(27700), name="osgb2.dxf")
    drawing, _ = extract.extract(path)
    inferred = landuse._build_crs(
        {
            "epsg": UTM_38N,
            "name": "WGS 84 / UTM zone 38N",
            "declared_in_file": False,
            "sources": [{"origin": "geometry", "detail": "where the extents sit"}],
        },
        where="test",
    )
    config = type("Cfg", (), {"crs": inferred, "version": None})()
    choice = store_landuse.crs_for_drawing(
        drawing._id, drawing=vars(drawing), config=config
    )
    assert choice.layer == "drawing_config"
    assert choice.crs is inferred
    assert choice.note and "27700" in choice.note
    assert "check the two against each other" in choice.note
