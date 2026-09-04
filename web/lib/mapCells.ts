/**
 * Client for `GET /drawings/{id}/geo` — the H3 layer the map overlays.
 *
 * A sibling of `mapGeometry`, not a replacement for it, and the difference is
 * structural rather than a matter of preference. `/geo` returns polygons on
 * layers a config classified as parcels: on the reference drawing that is
 * 2,576 of model space's 20,334 objects. It returns no roads (that layer
 * carries no land-use role at all), no LINE, no ARC, no TEXT and no
 * DIMENSION. It is the thematic layer — what the land is for, aggregated over
 * cells — and `/geometry` is the linework. A map wants both.
 *
 * Three things worth knowing before using this.
 *
 * **The totals are the whole drawing, always.** When a viewport is applied,
 * `cells` and `parcels` are filtered and `totals` is not. That is deliberate
 * on the server's part so a filtered view can never be read as the site, and
 * anything shown next to a total here has to keep that distinction.
 *
 * **A parcel is counted in exactly one cell.** The server attributes each
 * parcel to its representative cell, which is what makes `?res=9` and
 * `?res=13` agree exactly. Do not re-count client-side: a second count over
 * `parcels` would disagree near every cell boundary and there would be no way
 * to say which was right.
 *
 * **`cell` on a parcel is the join key.** It is how a set of handles — an
 * agent's answer, a selection — becomes a set of hexagons to light up.
 */

import { API_BASE, ApiError } from "./api";

export interface GeoCell {
  cell: string;
  /** Parcels in this cell, by land use. */
  counts: Record<string, number>;
  /** Their area, by land use. Present only when the drawing's units allow it
   *  to be published as m²; the response says so in `area_note`. */
  area_m2: Record<string, number>;
}

export interface GeoParcelProperties {
  handle: string;
  layer: string;
  land_use: string;
  land_use_subtype: string | null;
  land_use_verified: boolean;
  area_m2: number | null;
  /** The cell this parcel is counted in, at the STORED resolution. Coarser
   *  views reach their cell with `cell_to_parent`, never by re-indexing. */
  cell: string;
}

export interface GeoParcel {
  type: "Feature";
  geometry: { type: "Polygon"; coordinates: [number, number][][] };
  properties: GeoParcelProperties;
}

export interface GeoResponse {
  drawing_id: string;
  layout: string;
  georeferenced: boolean;
  indexed: boolean;
  epsg: number | null;
  /** The resolution this answer is aggregated at. */
  resolution: number;
  /** The resolution the drawing's cells are stored at. Asking for finer than
   *  this is refused by the server rather than invented. */
  stored_resolution: number;
  /** The ladder of levels this site occupies, derived server-side by rolling
   *  the stored cells up with `cell_to_parent`. Absent for a drawing with no
   *  cells. `res_cover` is the finest resolution at which one hexagon covers
   *  the whole site — the top of the hierarchy in Harsh's sense. */
  hierarchy?: {
    res_cover: number | null;
    res_stored: number;
    levels: { res: number; cells: number }[];
  } | null;
  /** What the backend COLOURED AND COUNTED BY, and how much of the drawing
   *  that dimension explains.
   *
   *  The client renders this; it does not decide it. A frontend that assumed
   *  "parcels" is why Sedra — 106,580 indexed cells — drew a single hexagon
   *  under a heading reading LAND USE: the index was full and the measure was
   *  wrong for the drawing. Optional so an older server still renders.
   */
  dimension?: {
    name: string;
    label: string;
    measure: string;
    measure_label: string;
    covers: number;
    of_shapes_in_scope: number;
    leaves_unexplained: number;
    is_fallback: boolean;
    chosen_because: string;
    alternatives: { name: string; label: string; available: boolean }[];
  } | null;
  cells: GeoCell[];
  parcels: { type: "FeatureCollection"; features: GeoParcel[] } | null;
  /** The ground the cells cover, dissolved into one boundary. Requested with
   *  `outline`. Its edges are hexagon edges — it is not a surveyed boundary,
   *  and nothing here may present it as one. */
  coverage_outline: {
    type: "Feature";
    geometry: { type: "MultiPolygon"; coordinates: [number, number][][][] };
    properties: Record<string, unknown>;
  } | null;
  totals: {
    parcels: number;
    counts: Record<string, number>;
    area_m2: Record<string, number>;
  };
  counts_scope: string;
  area_unit: string | null;
  area_note: string | null;
  crs_caveat: string | null;
  aggregation: string | null;
  excluded: {
    parcels_without_a_cell: number;
    parcels_without_a_usable_ring: number;
    note: string;
  } | null;
  limits: {
    max_parcel_features: number;
    parcels_omitted: number;
    max_cells: number;
    cells_omitted: number;
  } | null;
  /** Present when the drawing cannot be mapped. The server answers with the
   *  same shape rather than an empty success, so there is one branch here. */
  reason?: string | null;
  how_to_fix?: string | null;
}

export interface GeoQuery {
  layout: string;
  /** Omit to get the stored resolution. Coarser is derived; finer is refused. */
  res?: number;
  parcels?: boolean;
  outline?: boolean;
}

/** Which H3 resolution to ask for at a given camera zoom.
 *
 *  H3 is a hierarchy, and the map was showing one rung of it: a fixed res 9.
 *  That is 23 hexagons — a readable silhouette when the whole site is in
 *  frame, and six enormous plates covering everything once you zoom into a
 *  block. The ladder over the reference site, counted from the stored res-13
 *  cells via `cell_to_parent`, is
 *
 *      res  6: 1        <- the finest single hexagon that covers the site
 *      res  7: 3
 *      res  8: 7
 *      res  9: 23
 *      res 10: 115
 *      res 11: 645
 *      res 12: 1,957
 *      res 13: 2,546    <- the stored floor
 *
 *  so there is a rung to suit every zoom, and the table below picks it.
 *
 *  Bands rather than a formula. A formula would put each switch at some
 *  arbitrary fractional zoom that nobody can state or test; a band is a fact
 *  you can read off and check on screen. Each entry is the LOWEST zoom at
 *  which its resolution applies, and they are scanned in order, so the last
 *  entry is the floor for everything below it.
 *
 *  Aggregation is the server's, from the same stored cells — a coarser rung
 *  is the same evidence summed, never a different measurement. That is why
 *  the land-use totals hold at every level, and it is the property to check
 *  when a band looks wrong.
 *
 *  The numbers are tuned by eye against the reference site, and the anchor is
 *  the opening view: the camera fits Janadriyah at a zoom between 14 and 15,
 *  measured off which band fired. That view has to be the SILHOUETTE — res 9,
 *  23 hexagons — because it is the one a reader meets first and the one that
 *  has to say "this is the shape of the site". An earlier table put res 11
 *  there and drew 645 hexagons at parcel grain over the whole estate: every
 *  cell technically correct, the picture unreadable, which is the same
 *  failure as the fixed resolution it was meant to replace. So res 9 owns
 *  two whole zoom levels, and the finer rungs start where a reader has
 *  actually zoomed into a neighbourhood and can use them.
 */
const CELL_RES_BANDS: ReadonlyArray<{ minZoom: number; res: number }> = [
  { minZoom: 19, res: 13 },
  { minZoom: 18, res: 12 },
  { minZoom: 17, res: 11 },
  { minZoom: 16, res: 10 },
  { minZoom: 14, res: 9 },
  { minZoom: 12, res: 8 },
  { minZoom: 0, res: 7 },
];

/** The rung the map opens on, before a camera has reported a zoom. Res 9 is
 *  also what the fixed request used to ask for, so a drawing that never moves
 *  looks exactly as it did. */
export const DEFAULT_CELL_RES = 9;

/** How long the camera must settle before its zoom changes the request.
 *
 *  A pinch sweeps through several bands on its way to the one it lands on.
 *  Without a wait that is one request per band crossed, all but the last of
 *  them already stale by the time they arrive. */
export const CELL_RES_SETTLE_MS = 250;

export function resForZoom(zoom: number): number {
  for (const band of CELL_RES_BANDS) {
    if (zoom >= band.minZoom) return band.res;
  }
  return CELL_RES_BANDS[CELL_RES_BANDS.length - 1].res;
}

/** The rungs a drawing actually has, read off the response rather than
 *  assumed.
 *
 *  The band table above is tuned against a site stored at resolution 13, and
 *  a drawing stored coarser is not a hypothetical: the resolution is per
 *  drawing config. Asking such a drawing for res 13 is refused by the API
 *  with a 400 — correctly, since a finer cell would be invented rather than
 *  measured — and the overlay would simply stop working at high zoom.
 *
 *  So the bands propose and the drawing disposes. `hierarchy` is the
 *  authority where it is present; `stored_resolution` is the fallback, and it
 *  has always been in this payload. Nothing here guesses a bound. */
export function boundsFor(payload: GeoResponse | null): {
  min: number;
  max: number;
} | null {
  if (!payload) return null;
  const max = payload.hierarchy?.res_stored ?? payload.stored_resolution;
  if (typeof max !== "number") return null;
  const cover = payload.hierarchy?.res_cover;
  return { min: typeof cover === "number" ? cover : 0, max };
}

/** The rung to ask for at this zoom, on this drawing. */
export function clampRes(
  res: number,
  bounds: { min: number; max: number } | null,
): number {
  if (!bounds) return res;
  return Math.min(Math.max(res, bounds.min), bounds.max);
}

/** The same conditional-request cache the geometry client keeps, and for the
 *  same measured reason: returning to a drawing was refetching 1.7 MB of
 *  cells that the server can confirm unchanged in a 304 with no body.
 *
 *  Eight entries rather than four. One drawing is now worth up to six of
 *  them — one per zoom band — because the resolution is chosen by the
 *  camera, and a cache too small to hold one drawing's ladder would refetch
 *  every time the user zoomed back out. Eight holds that ladder and leaves
 *  room to step to another drawing and back. */
const MAX_CACHED = 8;
const cache = new Map<string, { etag: string | null; payload: GeoResponse }>();

function remember(key: string, etag: string | null, payload: GeoResponse) {
  cache.delete(key);
  cache.set(key, { etag, payload });
  while (cache.size > MAX_CACHED) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

export async function fetchGeo(
  drawingId: string,
  query: GeoQuery,
  signal?: AbortSignal,
): Promise<GeoResponse> {
  const params = new URLSearchParams({ layout: query.layout });
  if (query.res != null) params.set("res", String(query.res));
  if (query.parcels === false) params.set("parcels", "false");
  if (query.outline) params.set("outline", "true");

  const url = `${API_BASE}/drawings/${encodeURIComponent(drawingId)}/geo?${params}`;
  const held = cache.get(url);

  const res = await fetch(url, {
    signal,
    headers: held?.etag ? { "If-None-Match": held.etag } : undefined,
  });

  if (res.status === 304 && held) return held.payload;
  if (!res.ok) {
    let body: { error?: string; message?: string; hint?: string } = {};
    try {
      body = await res.json();
    } catch {
      /* a non-JSON body is itself the message */
    }
    throw new ApiError(
      body.error ?? `HTTP_${res.status}`,
      body.message ?? `${res.status} ${res.statusText}`,
      body.hint ?? "",
    );
  }
  const payload = (await res.json()) as GeoResponse;
  remember(url, res.headers.get("etag"), payload);
  return payload;
}

/** What a cell is worth, under the measure on screen.
 *
 *  One function so the colour, the height and the tooltip cannot disagree
 *  about what the number is — the failure that would put a cell at one height
 *  and label it another.
 */
export type CellMeasure = "count" | "area";

export function valueOf(
  cell: GeoCell,
  measure: CellMeasure,
  use: string | null,
): number {
  const source = measure === "area" ? cell.area_m2 : cell.counts;
  if (use !== null) return source[use] ?? 0;
  let total = 0;
  for (const key of Object.keys(source)) total += source[key];
  return total;
}

/** Which land use dominates a cell, for a tooltip that says what is there.
 *
 *  By count and not by area: a cell with one large open space and forty
 *  houses is a residential cell, and the reverse reading is one nobody
 *  asking about this map would mean.
 */
export function dominantUse(cell: GeoCell): string | null {
  let best: string | null = null;
  let most = 0;
  for (const [use, n] of Object.entries(cell.counts)) {
    if (n > most) {
      most = n;
      best = use;
    }
  }
  return best;
}
