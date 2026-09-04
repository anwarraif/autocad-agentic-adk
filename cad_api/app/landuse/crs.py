"""Drawing coordinates -> lat/long, and why it so often has to refuse.

Why this file sits next to the land use config
------------------------------------------------
Because the `crs:` block lives in the same config file (`<drawing_id>.yaml`),
and because both answer a question of the same shape: what the numbers in this
drawing mean, and who says so. A separate `geo/` package would separate the
config from its loader without separating anything else.

Why the projection is written here instead of using `pyproj`
--------------------------------------------------------------
`pyproj` is not in the image, and `cad_api/requirements.txt` is not a file that
may be touched from here. Adding it would also mean bringing in PROJ together
with its grid database, for one family of projections whose inverse series is
closed and short.

What is used is the Krüger series for Transverse Mercator, truncated at n^4.
For the WGS 84 ellipsoid (n = 0.00167922) the n^5 term sits around 1e-13 rad,
that is about 0.6 micrometres on the surface of the earth — six orders of
magnitude below the 1e-5 degree tolerance (about 1.1 m) that `UPLIFT-14` asks
for, and far below the real uncertainty here: the file itself does not state
its UTM zone.

**What is NOT done, and not because anyone forgot:**

- **No datum transformation.** EPSG:326xx and EPSG:327xx are both WGS 84, so
  the result here is WGS 84 and nothing else. A drawing that turns out to use a
  local datum (Ain el Abd, MTRF-2000, Nahrawan) will be off by tens to hundreds
  of metres, and nothing in this file can detect it. That is why
  `declared_in_file` travels with every response.
- **No geometry reprojection.** Only points are translated. Reprojecting a ring
  changes its side lengths and its area, while every area figure in this
  project is based on drawing coordinates.
- **No default.** A drawing without a `crs:` block never returns a lat/long. A
  lat/long from the wrong zone still looks like a correct lat/long — zone 37N
  puts the reference drawing in the Red Sea, and nothing in the numbers
  shouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .. import evidence as _ev

if TYPE_CHECKING:  # pragma: no cover - for the type checker only
    from . import Source

#: WGS 84 ellipsoid. Used by EPSG:326xx and EPSG:327xx, and that is all this
#: file supports.
_A = 6378137.0
_F = 1.0 / 298.257223563
_K0 = 0.9996
_FALSE_EASTING = 500000.0
_FALSE_NORTHING_SOUTH = 10000000.0

_N = _F / (2.0 - _F)
_A_RECT = _A / (1.0 + _N) * (1.0 + _N**2 / 4.0 + _N**4 / 64.0)

#: Inverse Krüger series, truncated at n^4.
_BETA = (
    _N / 2.0 - 2.0 * _N**2 / 3.0 + 37.0 * _N**3 / 96.0 - _N**4 / 360.0,
    _N**2 / 48.0 + _N**3 / 15.0 - 437.0 * _N**4 / 1440.0,
    17.0 * _N**3 / 480.0 - 37.0 * _N**4 / 840.0,
    4397.0 * _N**4 / 161280.0,
)
_DELTA = (
    2.0 * _N - 2.0 * _N**2 / 3.0 - 2.0 * _N**3 + 116.0 * _N**4 / 45.0,
    7.0 * _N**2 / 3.0 - 8.0 * _N**3 / 5.0 - 227.0 * _N**4 / 45.0,
    56.0 * _N**3 / 15.0 - 136.0 * _N**4 / 35.0,
    4279.0 * _N**4 / 630.0,
)

#: The EPSG ranges this file knows how to invert. Outside them it REFUSES
#: rather than approximates: a wrong projection still produces numbers that
#: are shaped like a lat/long.
_UTM_NORTH = range(32601, 32661)
_UTM_SOUTH = range(32701, 32761)


class CrsUnsupported(ValueError):
    """An EPSG this module cannot invert."""

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


def zone_of(epsg: int) -> tuple[int, bool]:
    """(zone number, whether northern hemisphere) for a WGS 84 UTM EPSG."""
    if epsg in _UTM_NORTH:
        return epsg - 32600, True
    if epsg in _UTM_SOUTH:
        return epsg - 32700, False
    raise CrsUnsupported(
        "CRS_NOT_SUPPORTED",
        f"EPSG:{epsg} is not a WGS 84 UTM zone.",
        "Only EPSG:32601-32660 (north) and 32701-32760 (south) are supported. "
        "For any other projection, install pyproj and bypass this module "
        "instead of approximating it — a wrong projection still produces "
        "numbers that are shaped like a lat/long.",
    )


def central_meridian(zone: int) -> float:
    """Central meridian of a UTM zone, degrees."""
    return zone * 6.0 - 183.0


def to_lat_lon(easting: float, northing: float, *, epsg: int) -> tuple[float, float]:
    """(lat, lon) in WGS 84 degrees from a point in drawing coordinates.

    The order is (lat, lon) — not (x, y). It gets swapped in every project that
    has ever used both, so it is spelled out here and in the response field
    names (`lat`, `lon`), never as a bare pair of numbers.
    """
    zone, north = zone_of(epsg)
    xi = (float(northing) - (0.0 if north else _FALSE_NORTHING_SOUTH)) / (_K0 * _A_RECT)
    eta = (float(easting) - _FALSE_EASTING) / (_K0 * _A_RECT)

    xi_p = xi
    eta_p = eta
    for j, beta in enumerate(_BETA, start=1):
        xi_p -= beta * math.sin(2 * j * xi) * math.cosh(2 * j * eta)
        eta_p -= beta * math.cos(2 * j * xi) * math.sinh(2 * j * eta)

    chi = math.asin(max(-1.0, min(1.0, math.sin(xi_p) / math.cosh(eta_p))))
    phi = chi
    for j, delta in enumerate(_DELTA, start=1):
        phi += delta * math.sin(2 * j * chi)

    lam = math.atan2(math.sinh(eta_p), math.cos(xi_p))
    return math.degrees(phi), central_meridian(zone) + math.degrees(lam)


# --- Config ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Crs:
    """The `crs:` block of a drawing, already validated.

    `declared_in_file` is the field that prevents the easiest mistake in this
    whole spec: a lat/long that looks official while the file never stated its
    coordinate system. It travels with EVERY response that carries a lat/long,
    whatever its value.
    """

    epsg: int
    name: str
    declared_in_file: bool
    note: str = ""
    not_established: str | None = None
    how_to_verify: str | None = None
    #: `landuse.Source`, imported only while type checking so that this file
    #: does not form a cycle with `landuse/__init__.py`, which imports it.
    sources: tuple["Source", ...] = ()

    def to_lat_lon(self, easting: float, northing: float) -> tuple[float, float]:
        return to_lat_lon(easting, northing, epsg=self.epsg)

    def point(self, xy: Sequence[float] | None) -> dict[str, Any] | None:
        """One drawing point as lat/long, or `None` if the point is not there.

        `None` and not `{}`: an entity without a point has no lat/long to
        report, and that is different from a lat/long that happens to be zero
        off the coast of Ghana.
        """
        if not xy or len(xy) < 2:
            return None
        lat, lon = self.to_lat_lon(float(xy[0]), float(xy[1]))
        return {"lat": lat, "lon": lon, "from_point": [float(xy[0]), float(xy[1])]}

    def as_dict(self) -> dict[str, Any]:
        return {
            "known": True,
            "epsg": self.epsg,
            "name": self.name,
            "declared_in_file": self.declared_in_file,
            "note": self.note or None,
            "not_established": self.not_established,
            "how_to_verify": self.how_to_verify,
        }


def unknown_crs(reason: str, how_to_verify: str) -> dict[str, Any]:
    """The `crs` block for a drawing that has no CRS config.

    Not silence, not `{}`, and not a guessed zone. Its shape is the same as
    `Crs.as_dict()` so that its reader does not need two branches, and
    `known: false` is what tells them apart.
    """
    return {
        "known": False,
        "epsg": None,
        "name": None,
        "declared_in_file": False,
        "note": None,
        "not_established": reason,
        "how_to_verify": how_to_verify,
    }


#: The origins that mean "someone outside the file confirmed it". Both MUST
#: carry a `reference` under the evidence contract, so a confirmation claim
#: cannot be written without naming the document or the person.
_CONFIRMING = (_ev.Origin.EXTERNAL_KEY, _ev.Origin.HUMAN)


def confirmed_by(crs: "Crs") -> str | None:
    """The document or person that confirmed this CRS, or `None`.

    This is NOT `provenance.verified`. Since D-082, `verified` answers "did
    someone type an override for this drawing" — and for a CRS inferred from
    where the extents sit, the answer is always yes, while what the reader is
    asking is whether Roshn has confirmed the zone. That answer can only come
    from a source outside the file, whose contract demands a `reference` — so a
    confirmation cannot be typed, it has to name someone.
    """
    for source in crs.sources:
        if getattr(source, "origin", None) in _CONFIRMING:
            return getattr(source, "reference", None) or "a source outside the file"
    return None


def caveat(crs: "Crs") -> str:
    """The sentence that MUST accompany every lat/long from this CRS.

    One string, not a pattern assembled in several places: a warning that is
    assembled is a warning that can be assembled half-way.
    """
    if crs.declared_in_file:
        return (
            f"Lat/long computed from drawing coordinates using a CRS that the "
            f"file STATES ({crs.name}, EPSG:{crs.epsg})."
        )
    who = confirmed_by(crs)
    if who:
        return (
            f"Lat/long computed from drawing coordinates using a CRS that the "
            f"file does not state ({crs.name}, EPSG:{crs.epsg}) and that has "
            f"been confirmed by {who}. Say so when quoting it."
        )
    return (
        f"Lat/long computed from drawing coordinates using a CRS that was "
        f"INFERRED (EPSG:{crs.epsg}), because the file does not state its "
        f"coordinate system. Do not use it for anything binding before it is "
        f"confirmed."
    )


def bounds_lat_lon(
    crs: Crs, extents: Mapping[str, Iterable[float]] | None
) -> dict[str, Any] | None:
    """The corners of the extents box as lat/long.

    All four corners, not two: on a grid rotated against the meridian, the
    south-west corner is not the point with the smallest latitude AND the
    smallest longitude at once, and reporting two corners as "the box" would
    cut off part of the drawing.
    """
    if not extents:
        return None
    lo = list(extents.get("min") or [])
    hi = list(extents.get("max") or [])
    if len(lo) < 2 or len(hi) < 2:
        return None
    corners = [
        (lo[0], lo[1]),
        (hi[0], lo[1]),
        (hi[0], hi[1]),
        (lo[0], hi[1]),
    ]
    points = [crs.to_lat_lon(x, y) for x, y in corners]
    return {
        "corners": [{"lat": lat, "lon": lon} for lat, lon in points],
        "lat_min": min(p[0] for p in points),
        "lat_max": max(p[0] for p in points),
        "lon_min": min(p[1] for p in points),
        "lon_max": max(p[1] for p in points),
        "note": "all four corners of the extents box; on a rotated grid the "
        "south-west corner is not the point with the smallest latitude and "
        "longitude at once",
    }
