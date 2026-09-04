/**
 * Client for `GET /drawings/{id}/geometry` — the vertex geometry the map
 * views draw.
 *
 * The endpoint is PROVISIONAL (see `cad_api/app/geometry_map.py` and
 * `docs/DECKGL-GEOMETRY-CONTRACT.md`). Everything that knows its shape is in
 * this file, so when the backend team settles the contract there is one place
 * to change rather than three components.
 *
 * Two things this module is deliberate about.
 *
 * **It never invents a placement.** `frame` comes back from the server and is
 * reported, not assumed. A drawing with no `crs:` block answers `world`, and
 * the map view then draws it in its own coordinates with no basemap — a real
 * view of the drawing that makes no claim about where on Earth it is. Only
 * one of the nineteen ingested drawings currently has a CRS, so this is the
 * common path, not the exotic one.
 *
 * **Areas are not measurable off these vertices.** When `frame` is `lnglat`
 * the coordinates have been reprojected for drawing; `area`, `perimeter` and
 * `length` came from the store in drawing units and are the numbers the rest
 * of the system quotes. The two must never be mixed, and the field names keep
 * them apart.
 */

import { API_BASE, ApiError } from "./api";

/** Which coordinate space `coordinates` are in. */
export type Frame = "world" | "lnglat";

/** What a feature's geometry is, as the store holds it. */
export type FeatureKind = "ring" | "path" | "segment" | "label";

export interface MapFeature {
  handle: string;
  type: string;
  layer: string;
  block_name: string | null;
  kind: FeatureKind;
  closed: boolean;
  /** `[lon, lat]` when the frame is `lnglat`, `[x, y]` when it is `world`. */
  coordinates: [number, number][];
  centroid: [number, number] | null;
  centroid_basis: "polygon_centroid" | "bbox_centre" | null;
  text: string | null;
  /** Always in DRAWING units, whatever `frame` says. */
  area: number | null;
  perimeter: number | null;
  length: number | null;
  ring_status: string | null;
  ring_orientation: string | null;
  vertex_count: number;
  /** Arc-ness per segment. Non-null means the straight lines drawn between
   *  these vertices are chords: see `approximationNotes()`. */
  bulges: number[] | null;
  has_bulge: boolean;
  /** True when this segment is an ARC's two endpoints. The straight line
   *  between them is a chord: the arc itself is not stored anywhere and
   *  cannot be reconstructed. */
  is_chord: boolean;
  ends_basis: string | null;
  shape_key: string | null;
  land_use: string | null;
  land_use_subtype: string | null;
  /** `parcel`, `network`, `overlay`, `structure`, `annotation`, or null when
   *  no config decides this layer — which is not the same as "unknown". */
  land_use_role: string | null;
  land_use_verified: boolean | null;
}

export interface CrsBlock {
  known: boolean;
  epsg: number | null;
  name: string | null;
  /** The field that stops the easiest mistake here: a lat/long that looks
   *  official while the file never stated its coordinate system. */
  declared_in_file: boolean | null;
  note: string | null;
  not_established: string | null;
  how_to_verify: string | null;
}

export interface Placement {
  can_georeference: boolean;
  why_not: string[] | null;
  units_name: string | null;
  units_code: number | null;
  units_declared_in_file: boolean;
  frame_in_use: Frame;
  fallback: string;
}

export interface CoverageRow {
  type: string;
  entities: number;
  rings: number;
  paths: number;
  label_anchors: number;
  segments: number;
  ring_refused: number;
  with_outline: number;
}

export interface GeometryResponse {
  drawing_id: string;
  layout: string;
  frame: Frame;
  frame_basis: string;
  crs: CrsBlock;
  crs_caveat: string | null;
  crs_confirmed_by: string | null;
  placement: Placement;
  units: {
    name: string | null;
    code: number | null;
    area_unit: string | null;
    area_unit_reason: string | null;
  };
  origin: {
    world: [number, number];
    lnglat: [number, number] | null;
    basis: string;
  } | null;
  bounds: {
    world: [[number, number], [number, number]] | null;
    lnglat: [[number, number], [number, number]] | null;
  };
  /** Where the drawing IS, as opposed to where its extreme corners are.
   *
   *  The smallest set of stored H3 cells holding most of the placed objects.
   *  A drawing's extreme bounding box is not where the drawing is: one
   *  correctly-placed outlier 12 km out shrinks the estate to a few pixels.
   *  Null when the drawing has no cells, and the caller then falls back to
   *  the extents and says so. */
  /** Present when a `bbox` was sent. `cells_served` is what this answer
   *  covers, and is what to accumulate and send back as `have`. */
  viewport?: {
    cells_served: string[];
    cells_already_held: string[];
    nothing_new: boolean;
  } | null;
  content_bounds: {
    lnglat: [[number, number], [number, number]];
    basis: string;
    cells_used: number;
    cells_total: number;
    entities_inside: number;
    entities_total: number;
    share: number | null;
    note: string;
  } | null;
  features: MapFeature[];
  counts: Record<FeatureKind, number>;
  total_matches: number;
  returned: number;
  offset: number;
  truncated: boolean;
  next_offset: number | null;
  /** Layers this DRAWING keeps switched off or frozen. Their features are
   *  present in `features` and should start hidden, matching AutoCAD. */
  layers_off: { name: string; off: boolean; frozen: boolean }[];
  layers_off_basis: string;
  geometry_coverage: {
    entities_in_layout: number;
    with_outline: number;
    without_outline: number;
    by_type: CoverageRow[];
    types_with_no_outline: string[];
    /** Closed polylines the extractor refused to write a ring for. Invisible
     *  to `types_with_no_outline`, because their type mostly does have one. */
    rings_refused: number;
    rings_refused_reason: string | null;
    note: string;
  };
  hint: string;
}

export interface GeometryQuery {
  layout: string;
  frame?: "auto" | Frame;
  kinds?: string;
  /** The layer selection, already in the server's own syntax: bare names
   *  include, a leading `!` excludes. See `layerParam` in page.tsx for why it
   *  is a prepared string rather than a list — the two forms have very
   *  different lengths and the shorter one is chosen per selection. */
  layers?: string;
  limit?: number;
  /** Where to resume from. Pair with `limit` to page a large layout. */
  offset?: number;
  /** The lng/lat box on screen, west,south,east,north.
   *
   *  Spends a limited feature budget on what the user is LOOKING at. The
   *  server turns it into a set of coarse H3 parent cells and matches an
   *  index; nothing here converts degrees into drawing coordinates, because
   *  `crs.py` has no forward projection and inventing one is the class of
   *  error this project fears most. */
  bbox?: [number, number, number, number];
  /** The map's Web Mercator zoom. The server drops vertices closer together
   *  than a pixel at this zoom; omitted, every stored vertex is sent. */
  zoom?: number;
  /** Coarse H3 cells the caller already holds. The server answers with the
   *  cells in `bbox` MINUS these, so a pan asks only for what is newly in
   *  view. Sent as what we HAVE rather than what we want, because deciding
   *  which cells a box covers needs H3 and that lives on the server. */
  have?: string[];
}

/** Geometry for one layout.
 *
 *  One request for the whole layout rather than a page per viewport move.
 *  Measured on the densest drawing in the corpus — Janadriyah's model space,
 *  3,169 outlines over 20,212 vertices — the call is ~1.8 s and 2.5 MB, and
 *  the alternative is a map that redraws a partial city every time it is
 *  panned, which reads as a rendering fault rather than as paging.
 */
/** The last few answers, kept so that returning to a drawing is not a
 *  re-download.
 *
 *  Measured before this existed: leaving Janadriyah and coming back refetched
 *  6.3 MB, every time, because the payload lives in React state and the state
 *  is replaced when the drawing changes. The endpoint carries a strong ETag
 *  precisely so that this can be a conditional request instead — see
 *  `docs/DECKGL-GEOMETRY-CONTRACT.md` §3a.
 *
 *  Revalidated rather than trusted. `drawing_id` is a content hash, so the
 *  geometry genuinely cannot change under a key — but the land-use config
 *  can, and it decides `land_use_role` on every feature. The server folds
 *  that into the tag, so one small round trip turns "probably still right"
 *  into "the server says so". A 304 costs no body at all.
 *
 *  Two entries. A parsed payload of ~10,000 features is tens of megabytes of
 *  JavaScript objects, and the case this is for — leave a drawing, come back
 *  — needs exactly the current one and the previous one.
 */
const MAX_CACHED = 2;
const cache = new Map<string, { etag: string | null; payload: GeometryResponse }>();

function remember(key: string, etag: string | null, payload: GeometryResponse) {
  cache.delete(key);
  cache.set(key, { etag, payload });
  while (cache.size > MAX_CACHED) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

export async function fetchGeometry(
  drawingId: string,
  query: GeometryQuery,
  signal?: AbortSignal,
): Promise<GeometryResponse> {
  const params = new URLSearchParams({ layout: query.layout });
  if (query.frame) params.set("frame", query.frame);
  if (query.kinds) params.set("kinds", query.kinds);
  if (query.layers) params.set("layers", query.layers);
  if (query.limit) params.set("limit", String(query.limit));
  if (query.offset) params.set("offset", String(query.offset));
  if (query.bbox) params.set("bbox", query.bbox.map((n) => n.toFixed(6)).join(","));
  if (query.zoom !== undefined) params.set("zoom", String(query.zoom));
  if (query.have?.length) params.set("have", query.have.join(","));

  const url = `${API_BASE}/drawings/${encodeURIComponent(drawingId)}/geometry?${params}`;
  const held = cache.get(url);

  const res = await fetch(url, {
    signal,
    // Only when there is something to validate. Sending it unconditionally
    // would make every first request a preflighted one for no reason:
    // `If-None-Match` is not a simple header, so it costs an OPTIONS.
    headers: held?.etag ? { "If-None-Match": held.etag } : undefined,
  });

  if (res.status === 304 && held) {
    // The server has confirmed what is already in hand. Nothing was
    // transferred and nothing needs parsing.
    return held.payload;
  }
  if (!res.ok) {
    // The taxonomy is the same one every other route uses, so a failure here
    // reads like a failure anywhere else — including `hint`, which is the
    // next step to take rather than a restatement of the problem.
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
  const payload = (await res.json()) as GeometryResponse;
  remember(url, res.headers.get("etag"), payload);
  return payload;
}

/** Every layer name present in the geometry, with how many features it holds.
 *
 *  Not the same list as the SVG's layers: the SVG draws every entity type,
 *  and this holds only the ones that carry vertices. Kept separate rather
 *  than reconciled, because a layer that is in one and not the other is a
 *  fact worth seeing — it is the coverage gap of §3.2 in per-layer form.
 */
export function layersInGeometry(
  features: MapFeature[],
): { name: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const f of features) counts.set(f.layer, (counts.get(f.layer) ?? 0) + 1);
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

/** Everything on screen that is an approximation, in one sentence each.
 *
 *  Two different approximations, kept apart because they have different fixes
 *  and different sizes of error.
 *
 *  A **bulged polyline** carries `path_bulges` saying a span curves; the
 *  vertices are real and only the spans between them are drawn straight.
 *
 *  A **chord** is an ARC, and there the whole shape is two points. Its centre,
 *  radius and angles were used to compute those endpoints at ingest and then
 *  not stored, so nothing downstream can recover the curve. On a road
 *  centreline that is a straight line cutting the inside of every bend.
 *
 *  Both are drawn rather than withheld, because 450 invisible arcs leave a
 *  road network full of gaps with nothing on screen to explain them. Saying so
 *  is what makes drawing them honest.
 */
export function approximationNotes(features: MapFeature[]): string[] {
  const out: string[] = [];
  const bulged = features.filter((f) => f.has_bulge).length;
  const chords = features.filter((f) => f.is_chord).length;
  if (chords > 0) {
    out.push(
      chords === 1
        ? "1 arc is drawn as a straight chord between its endpoints. The curve is not stored anywhere — only the two ends are — so this is the whole of what the drawing can give here."
        : `${chords.toLocaleString()} arcs are drawn as straight chords between their endpoints. The curves are not stored anywhere — only the two ends of each — so this is the whole of what the drawing can give here.`,
    );
  }
  if (bulged > 0) {
    out.push(
      `${bulged.toLocaleString()} outline${bulged === 1 ? " contains" : "s contain"} a curved span, drawn here as a straight line between ${bulged === 1 ? "its" : "their"} real vertices.`,
    );
  }
  return out;
}

export interface Box {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** Turn `bounds` into the four numbers a camera needs.
 *
 *  Returns null rather than a zero box when there is nothing to fit: a camera
 *  fitted to a zero box is at infinite zoom, which looks like a blank canvas
 *  and reads as a failed request.
 */
export function boundsOf(data: GeometryResponse): Box | null {
  const box = data.frame === "lnglat" ? data.bounds.lnglat : data.bounds.world;
  if (!box) return null;
  const [[minX, minY], [maxX, maxY]] = box;
  if (!(maxX > minX) || !(maxY > minY)) return null;
  return { minX, minY, maxX, maxY };
}

/** Tukey's outlier fence, in interquartile ranges either side of the quartiles.
 *
 *  1.5 is the standard constant, not a tuned one, and using it rather than a
 *  fixed percentile is the whole point: a percentile always discards the same
 *  PROPORTION of the drawing whatever its shape, while a fence discards
 *  whatever is genuinely far from the bulk and nothing when nothing is.
 *
 *  Measured on Janadriyah, which is why this changed: trimming 1% from each
 *  end framed 1,615 x 1,465 m and cut 100 plots off the edge of the estate —
 *  real plots, on the typology layers, a few hundred metres outside the box
 *  purely because the box was defined by a count. The fence frames
 *  2,713 x 2,339 m and contains all 2,545 of them, while still being four
 *  times tighter than the 11,699 m extents that made this necessary.
 */
const FENCE_IQR = 1.5;
/** Breathing room around the framed content, as a fraction of its span. */
const MARGIN = 0.06;

export interface ContentFrame {
  box: Box;
  /** Every vertex of what was handed in. What `Fit all` frames. */
  extents: Box;
  /** Features whose centroid falls outside the framed box. Not hidden — the
   *  camera simply does not start on them, and the count says so. */
  outside: number;
  /** The layer most of those outlying features are on, so the note can say
   *  WHAT is out there rather than only how many. On this corpus the answer
   *  is almost always a road centreline layer running off to join the
   *  external network, which is reassuring in a way a bare count is not. */
  outsideLayer: { name: string; count: number } | null;
  /** True when framing the content is materially tighter than framing the
   *  extents, which is the only case where the difference is worth a word. */
  tighterThanExtents: boolean;
  /** How `box` was arrived at, in words. Two answers are possible and they
   *  are not equally good: `coverage-outline` is the store's own account of
   *  where the site is, and `fence` is this file inferring it from where the
   *  centroids bunch up. A note that says which one framed the view lets a
   *  surprising camera be diagnosed instead of argued about. */
  basis: "coverage-outline" | "fence" | "extents";
}

function boundsOfFeatures(features: MapFeature[]): Box | null {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const f of features) {
    for (const [x, y] of f.coordinates) {
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (!(maxX > minX) || !(maxY > minY)) return null;
  return { minX, minY, maxX, maxY };
}

/** Where the drawing's content actually is, as opposed to where its extents are.
 *
 *  Takes the features that are ACTUALLY DRAWN, not the whole payload. That
 *  distinction arrived with a bug: the road centrelines are on a layer the
 *  file switches off, so once they were correctly hidden, a frame computed
 *  over the whole payload still reserved room for them and the note still
 *  counted 823 hidden objects as "drawn, just not framed". A camera and a
 *  caption that describe objects nobody can see are worse than none.
 *
 *  Measured on Janadriyah: the outer bounds span 11.7 x 6.9 km, while the
 *  plots sit inside 2.7 x 2.3 km. A camera fitted to the extents puts the
 *  whole estate in a corner as an unreadable smudge — which looks like a
 *  projection fault and is really a handful of outlying rings.
 *
 *  Tukey's fence rather than a percentile, and that too came from a bug: a
 *  fixed 1% trim discards the same PROPORTION whatever the drawing's shape,
 *  and on this one it cut 100 real plots off the edge of the estate. A fence
 *  discards what is genuinely far from the bulk, and nothing when nothing is.
 */
export function contentFrame(features: MapFeature[]): ContentFrame | null {
  const extents = boundsOfFeatures(features);
  if (!extents) return null;

  // Closed outlines decide the frame; open paths do not. A plot is somewhere
  // you look; a centreline leaving the site is not, and letting it set the
  // camera puts the whole masterplan in a corner. Falls back to every feature
  // when there are too few rings, so a drawing that is nothing but polylines
  // still frames sensibly.
  const rings = features.filter((f) => f.kind === "ring" && f.centroid);
  const source = rings.length >= 20 ? rings : features;
  const centroids = source
    .map((f) => f.centroid)
    .filter((c): c is [number, number] => c !== null);
  if (centroids.length < 20) {
    return {
      box: extents,
      extents,
      outside: 0,
      outsideLayer: null,
      tighterThanExtents: false,
      basis: "extents",
    };
  }

  const fence = (values: number[]): [number, number] => {
    const sorted = [...values].sort((a, b) => a - b);
    const at = (p: number) =>
      sorted[
        Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))
      ];
    const lo = at(0.25);
    const hi = at(0.75);
    const iqr = hi - lo;
    return [lo - FENCE_IQR * iqr, hi + FENCE_IQR * iqr];
  };

  let [minX, maxX] = fence(centroids.map((c) => c[0]));
  let [minY, maxY] = fence(centroids.map((c) => c[1]));

  const beyond = features.filter(
    (f) =>
      f.centroid &&
      (f.centroid[0] < minX ||
        f.centroid[0] > maxX ||
        f.centroid[1] < minY ||
        f.centroid[1] > maxY),
  );
  const byLayer = new Map<string, number>();
  for (const f of beyond) byLayer.set(f.layer, (byLayer.get(f.layer) ?? 0) + 1);
  const top = [...byLayer.entries()].sort((a, b) => b[1] - a[1])[0];

  const padX = (maxX - minX) * MARGIN;
  const padY = (maxY - minY) * MARGIN;
  minX -= padX;
  maxX += padX;
  minY -= padY;
  maxY += padY;
  if (!(maxX > minX) || !(maxY > minY)) {
    return {
      box: extents,
      extents,
      outside: 0,
      outsideLayer: null,
      tighterThanExtents: false,
      basis: "extents",
    };
  }

  const area = (b: Box) => (b.maxX - b.minX) * (b.maxY - b.minY);
  const box = { minX, minY, maxX, maxY };
  return {
    box,
    extents,
    outside: beyond.length,
    outsideLayer: top ? { name: top[0], count: top[1] } : null,
    tighterThanExtents: area(box) < area(extents) * 0.5,
    basis: "fence",
  };
}

/** The bounds of a `/geo` coverage outline, or `null` if there is nothing
 *  usable in it.
 *
 *  The outline is a MultiPolygon of hexagon boundaries in lng/lat, so it is
 *  only comparable with feature coordinates when the map is in the lng/lat
 *  frame. The caller checks that; this function only measures.
 */
export function boundsOfOutline(
  outline: { geometry: { coordinates: [number, number][][][] } } | null,
): Box | null {
  if (!outline) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const polygon of outline.geometry.coordinates) {
    for (const ring of polygon) {
      for (const [x, y] of ring) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (!(maxX > minX) || !(maxY > minY)) return null;
  return { minX, minY, maxX, maxY };
}

/** Where to point the camera, preferring the store's answer to ours.
 *
 *  `contentFrame` infers where the site is by putting a fence around the
 *  centroids. That is a good guess and it is still a guess. `/geo` can return
 *  the ground its cells cover, dissolved into one boundary — the same question
 *  answered by the system that holds the data, so when it is available it
 *  wins.
 *
 *  Two things it does NOT change. `extents` stays every vertex of what is
 *  drawn, because `Fit all` has to be able to reach an object the outline
 *  never covered — the outline is built from classified parcels, and a road
 *  running off the site is not one. And the outlying count keeps being
 *  measured against the box actually used, so the note on screen describes
 *  the camera the user is looking through.
 */
export function frameWithOutline(
  features: MapFeature[],
  outlineBounds: Box | null,
): ContentFrame | null {
  const inferred = contentFrame(features);
  if (!outlineBounds) return inferred;
  const extents = inferred?.extents ?? boundsOfFeatures(features);
  if (!extents) return null;

  const padX = (outlineBounds.maxX - outlineBounds.minX) * MARGIN;
  const padY = (outlineBounds.maxY - outlineBounds.minY) * MARGIN;
  const box = {
    minX: outlineBounds.minX - padX,
    minY: outlineBounds.minY - padY,
    maxX: outlineBounds.maxX + padX,
    maxY: outlineBounds.maxY + padY,
  };

  const beyond = features.filter(
    (f) =>
      f.centroid &&
      (f.centroid[0] < box.minX ||
        f.centroid[0] > box.maxX ||
        f.centroid[1] < box.minY ||
        f.centroid[1] > box.maxY),
  );
  const byLayer = new Map<string, number>();
  for (const f of beyond) byLayer.set(f.layer, (byLayer.get(f.layer) ?? 0) + 1);
  const top = [...byLayer.entries()].sort((a, b) => b[1] - a[1])[0];

  const area = (b: Box) => (b.maxX - b.minX) * (b.maxY - b.minY);
  return {
    box,
    extents,
    outside: beyond.length,
    outsideLayer: top ? { name: top[0], count: top[1] } : null,
    tighterThanExtents: area(box) < area(extents) * 0.5,
    basis: "coverage-outline",
  };
}

/** Roles a config uses to say "this is not a plot".
 *
 *  An overlay hatch, a block boundary, a text layer, a road centreline. Each
 *  is a decision someone made, and each is a reason to draw an outline rather
 *  than a filled area — filling a block boundary paints over every plot inside
 *  it, and extruding one puts a wall around the superblock.
 */
const NOT_AN_AREA = new Set(["overlay", "structure", "annotation", "network"]);

/** Does anything in this payload carry a land-use decision at all?
 *
 *  A per-drawing question, asked once, and the reason `drawsAsArea` needs a
 *  second argument. See its note.
 */
export function anyClassified(features: { land_use_role: string | null }[]): boolean {
  return features.some((f) => f.land_use_role !== null);
}

/** Should this feature be drawn as a filled — and extrudable — area?
 *
 *  The hard case is a ring with `land_use_role: null`, and it needs two
 *  different answers depending on the drawing it is in.
 *
 *  **Where nothing is classified**, null means the drawing has no land-use
 *  config — true of eighteen of the nineteen ingested drawings. Treating null
 *  as "not a plot" there made the 3D view empty on all eighteen: every closed
 *  shape reduced to a wireframe with no height, while the one classified
 *  drawing looked right. That is exactly how a wrong default survives review.
 *
 *  **Where layers ARE classified**, null means something quite different: the
 *  config looked at this layer and mapped nothing to it. On Janadriyah those
 *  143 rings include `1C0436A` on layer `0` — a 6,821,011 m² rectangle around
 *  the whole estate. Filling it tints the entire site, and extruding it puts a
 *  6-metre lid over the masterplan that swallows every click aimed underneath.
 *
 *  So the fallback is switched per drawing, not guessed per feature. No size
 *  threshold, no name matching: a frontend heuristic over layer geometry would
 *  be a second classifier quietly disagreeing with the config, which is the
 *  thing §3.6 of the architecture note warns against by name.
 */
export function drawsAsArea(
  feature: { kind: FeatureKind; land_use_role: string | null },
  classified: boolean,
): boolean {
  if (feature.kind !== "ring") return false;
  const role = feature.land_use_role;
  if (role === null) return !classified;
  return !NOT_AN_AREA.has(role);
}

/** Every layer present in the map's SCOPE, with its total there.
 *
 *  A separate request from the geometry itself, and separate on purpose. The
 *  layer panel needs two different numbers per layer — how many the drawing
 *  holds in this scope, and how many are currently on screen — and only the
 *  second can be counted from the features in hand. Deriving the first from
 *  them, as the panel did, made the list grow new rows as paging proceeded
 *  and made a layer's total whatever had arrived so far.
 *
 *  The answer does not change between pages, so it is fetched once per
 *  drawing and scope and is cached server-side per version.
 */
export interface ScopeLayers {
  drawing_id: string;
  layout: string;
  layers: { name: string; count: number }[];
  total: number;
  scope: string;
  scope_label: string;
  scope_note: string;
  membership_is: string;
}

export async function fetchScopeLayers(
  drawingId: string,
  layout: string,
  kinds: string,
  signal?: AbortSignal,
): Promise<ScopeLayers> {
  const params = new URLSearchParams({ layout, kinds });
  const res = await fetch(
    `${API_BASE}/drawings/${encodeURIComponent(drawingId)}/geometry/layers?${params}`,
    { signal },
  );
  if (!res.ok) {
    let body: { error?: string; message?: string; hint?: string } = {};
    try {
      body = await res.json();
    } catch {
      /* a non-JSON body is itself the message */
    }
    throw new Error(body.message ?? `HTTP ${res.status}`);
  }
  return (await res.json()) as ScopeLayers;
}
