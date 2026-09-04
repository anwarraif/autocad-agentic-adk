/**
 * Typed client for cad-api.
 *
 * The browser talks to cad-api directly rather than proxying through Next.js:
 * the SVG payloads are large and already gzip-compressed by the API, and
 * passing them through a route handler would mean decompressing and
 * recompressing several megabytes on every drawing switch for no benefit.
 * CORS is configured on cad-api for exactly the web origin.
 */

// Type-only: the tag shapes are declared next to the rules that judge them,
// in `lib/layerTags.ts`, so that a reviewer reading those rules is reading the
// fields they apply to. Nothing is imported at runtime.
import type {
  LandUseSummaryLite,
  LayerImpact,
  LayerTag,
} from "./layerTags";
// Same arrangement, same reason: the run document's shape is declared next to
// the rules that read it, in `lib/ingestRuns.ts`, and only the three routes
// live here — one client, not two.
import type {
  IngestRun,
  IngestRunList,
  UploadAccepted,
} from "./ingestRuns";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:4311";

export interface LayoutSummary {
  name: string;
  entity_count: number;
  is_modelspace: boolean;
  /** True when this "layout" is really a block definition. */
  is_block?: boolean;
}

export interface LayerSummary {
  name: string;
  color: number;
  frozen: boolean;
  off: boolean;
  entity_count: number;
}

export interface DrawingSummary {
  _id: string;
  original_filename: string;
  dxf_version: string;
  acad_release: string;
  units_code: number;
  units_name: string;
  entity_count: number;
  file_bytes: number;
  audit_errors: number;
  layouts: LayoutSummary[];
  extents: { min: number[]; max: number[] } | null;
  warnings: string[];
}

export interface RenderMeta {
  layout: string;
  rendered_entities: number;
  bytes_raw: number;
  bytes_gzip: number;
  elapsed_ms: number;
  truncated: boolean;
  truncated_reason: string | null;
  /** Foreign payloads on this layout that no DXF renderer can draw. */
  unrenderable?: { handle: string; type: string; layer: string; reason: string; located: boolean }[];
}

export interface DrawingDetail extends DrawingSummary {
  layers: LayerSummary[];
  blocks: string[];
  counts_by_type: Record<string, number>;
  counts_by_layer: Record<string, number>;
  xrefs: { block_name: string; path: string; resolved: boolean }[];
  renders: RenderMeta[];
  comment_counts: Record<string, number>;
  /** Whether this drawing has a position on the earth, and where to get it.
   *
   *  Answered by the API rather than worked out here. A drawing is mappable
   *  when it declares a coordinate system or has one configured, and
   *  seventeen of the eighteen in the store do neither — so `reason` is the
   *  ordinary case and is written to be shown, not logged. */
  geo?: {
    /** Whether the drawing has a coordinate system at all. */
    georeferenced: boolean;
    /** Whether its entities have been indexed into H3 cells. A drawing can be
     *  georeferenced and NOT indexed — re-ingesting one strips the cells —
     *  and the two need different actions by different people, so they are
     *  two booleans rather than one. */
    indexed: boolean;
    epsg: number | null;
    crs_source_layer: string | null;
    /** Why it cannot be exported. `null` when it can. */
    reason: string | null;
    /** The command that would fix it, when a command would. */
    how_to_fix?: string | null;
    /** Path, not a full URL: the caller owns the base. `null` unless the
     *  drawing is both georeferenced and indexed, so this single field is the
     *  one thing a caller has to test. */
    export_url: string | null;
  };
}

export interface EntityDetail {
  _id: string;
  drawing_id: string;
  handle: string;
  type: string;
  layer: string;
  layout: string;
  block_name: string | null;
  text: string | null;
  attribs: Record<string, string> | null;
  bbox: { min: number[]; max: number[] } | null;
  /** The middle of the bounding box. NOT the centre of the shape, and the
   *  two are far enough apart to matter: on a neighbourhood boundary polygon
   *  they sit 72.7 m apart. Use `polygon_centroid` to place a mark. */
  bbox_centre: number[] | null;
  /** Area-weighted centroid over the closed ring. Present only when
   *  `ring_status` is `complete`; `null` for everything else, with
   *  `geometry_note` saying which reason applied. */
  polygon_centroid?: number[] | null;
  /** The gate for anything point-in-polygon. `complete` is the only value
   *  whose `ring` may be trusted as a boundary — a polygon carrying arc
   *  segments or one that never closed will happily produce a ring that is
   *  not the shape it stands for. Read the value; do not test `ring` for
   *  null. */
  ring_status?: string | null;
  /** Why this entity has no usable ring, in words. */
  geometry_note?: string | null;
  /** A text entity's real insertion point, not the middle of its box. The
   *  box depends on a font, and the SHX fonts this project needs are not
   *  shipped with the drawings, so the box is itself an estimate. */
  anchor_point?: number[] | null;
  length: number | null;
  area: number | null;
  comments: Comment[];
}

export interface Comment {
  _id: string;
  drawing_id: string;
  entity_handle: string;
  body: string;
  author: string;
  created_at: string;
  status: string;
}

export interface ApiErrorBody {
  error: string;
  message: string;
  hint: string;
}

/** An error carrying cad-api's code and hint, so the UI can show the hint. */
export class ApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly hint: string,
    /** The parsed error body, whole.
     *
     *  Added for the upload route, whose `409 ALREADY_INGESTED` carries the
     *  `run_id` of the refusal record it just wrote
     *  (`docs/INGEST-RUN-CONTRACT.md`, "Routes"). Without this the client
     *  threw away the one field that lets the panel show WHY the upload was
     *  turned away, and the reader would have been left with a red line
     *  instead of the run that explains it.
     *
     *  Optional and last, so nothing that constructs an ApiError today
     *  changes. Deliberately a bag rather than a typed field: it is whatever
     *  the route sent, and the caller is expected to check before trusting a
     *  value out of it. */
    readonly details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch (cause) {
    throw new ApiError(
      "NETWORK_ERROR",
      `Could not reach cad-api at ${API_BASE}.`,
      "Is the stack running? Try `docker compose ps` and check cad-api is healthy.",
    );
  }
  if (!response.ok) {
    // Typed as the triple AND as a bag: the triple is what every route
    // promises, the bag is how a route that sends more than the triple keeps
    // it (see `ApiError.details`).
    let body: Partial<ApiErrorBody> & Record<string, unknown> = {};
    try {
      body = await response.json();
    } catch {
      /* a non-JSON error body is still an error; fall through to defaults */
    }
    throw new ApiError(
      body.error ?? `HTTP_${response.status}`,
      body.message ?? `${response.status} ${response.statusText}`,
      body.hint ?? "",
      body,
    );
  }
  return response.json() as Promise<T>;
}

export function listDrawings(): Promise<{
  total: number;
  drawings: DrawingSummary[];
}> {
  return request("/drawings");
}

export function getDrawing(id: string): Promise<DrawingDetail> {
  return request(`/drawings/${id}`);
}

export function getEntity(id: string, handle: string): Promise<EntityDetail> {
  return request(`/drawings/${id}/entities/${encodeURIComponent(handle)}`);
}

export function listComments(
  drawingId: string,
): Promise<{ total: number; comments: Comment[] }> {
  return request(`/comments?drawing_id=${encodeURIComponent(drawingId)}`);
}

export function addComment(input: {
  drawing_id: string;
  handle: string;
  body: string;
  author: string;
  client_request_id: string;
}): Promise<Comment> {
  return request("/comments", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

/** URL of the rendered SVG. Fetched as text, not as an <img> src, because the
 *  viewer needs the DOM in order to hit-test entities and toggle layers. */
export function svgUrl(drawingId: string, layout: string): string {
  return `${API_BASE}/drawings/${drawingId}/svg?layout=${encodeURIComponent(layout)}`;
}

export async function fetchSvg(
  drawingId: string,
  layout: string,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch(svgUrl(drawingId, layout), { signal });
  if (!response.ok) {
    throw new ApiError(
      "RENDER_NOT_FOUND",
      `No render for layout "${layout}".`,
      "Re-run the ingest for this drawing with --force.",
    );
  }
  return response.text();
}

// ---------------------------------------------------------------------------
// Region selection
// ---------------------------------------------------------------------------

/** AutoCAD's two selection modes. `window` takes only what is wholly inside;
 *  `crossing` takes anything the region touches. */
export type RegionMode = "window" | "crossing";
/** The shapes a user can DRAW. */
export type RegionKind = "rect" | "polygon" | "circle";
/** How a selection came about. A single click is a selection of one, and it
 *  travels the same path as a drawn region — same panel, same agent scope —
 *  so it belongs in the same type rather than in a parallel one. */
export type SelectionKind = RegionKind | "click";

/** A region **in drawing coordinates**, never in screen pixels.
 *
 *  This is the single most important property of the feature. Stored in
 *  pixels, a region stops meaning anything the moment the user zooms; stored
 *  in drawing coordinates it stays pinned to the drawing through any amount
 *  of pan and zoom, and the same handles come back every time.
 */
export interface Region {
  kind: RegionKind;
  mode: RegionMode;
  /** Two opposite corners for `rect`, three or more vertices for `polygon`. */
  points: [number, number][];
  /** The same vertices in viewBox units — the coordinate space the SVG's
   *  rendered geometry lives in. Carried alongside the drawing-space points
   *  because the two selection mechanisms need different spaces: the server
   *  compares stored bounding boxes (drawing coordinates), while a sheet
   *  layout is answered from what is actually drawn (viewBox coordinates).
   *  Both are stable under pan and zoom, so the invariance guarantee holds
   *  for either. */
  pointsView: [number, number][];
  /** The layout the region was drawn on. */
  layout: string;
}

export interface RegionGroup {
  layer: string;
  type: string;
  count: number;
}

export interface SelectionResult {
  drawing_id: string;
  layout: string | null;
  kind: SelectionKind;
  mode: RegionMode;
  region: number[][];
  region_bbox: number[];
  total: number;
  handles: string[];
  handles_truncated: boolean;
  extents: { min: number[]; max: number[] } | null;
  basis: string;
  by_layer: { name: string; count: number }[];
  by_type: { name: string; count: number }[];
  groups: RegionGroup[];
  groups_truncated: boolean;
  without_bbox?: number;
}

export function selectRegion(
  drawingId: string,
  region: Region,
  signal?: AbortSignal,
): Promise<SelectionResult> {
  return request(`/drawings/${drawingId}/selection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points: region.points.map(([x, y]) => [x, y]),
      kind: region.kind,
      mode: region.mode,
      layout: region.layout || null,
    }),
    signal,
  });
}

/** Entities in one (layer, type) group of an existing region.
 *
 *  Deliberately a second call rather than data returned with the summary: a
 *  region over a site plan matches thousands of entities, and the summary
 *  exists precisely so that none of those rows are fetched until a human asks
 *  for one group of them.
 *
 *  The region is sent again rather than the handles it produced. Filtering a
 *  cached handle list client-side would be wrong twice over: the list is
 *  capped, so a large group would come back short without saying so, and the
 *  handles carry no layer or type to filter on.
 */
export function listRegionGroup(
  drawingId: string,
  region: Region,
  group: { layer: string; type: string },
  offset = 0,
  limit = 200,
): Promise<{
  total: number;
  returned: number;
  offset: number;
  truncated: boolean;
  entities: {
    handle: string;
    type: string;
    layer: string;
    text: string | null;
    bbox_centre: number[] | null;
  }[];
}> {
  return request(`/drawings/${drawingId}/selection/rows`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points: region.points.map(([x, y]) => [x, y]),
      kind: region.kind,
      mode: region.mode,
      layout: region.layout || null,
      layer: group.layer,
      type: group.type,
      offset,
      limit,
    }),
  });
}

export interface DescribeSelectionResult {
  drawing_id: string;
  requested: number;
  found: number;
  missing_handles: string[];
  by_layout: { name: string; count: number }[];
  by_layer: { name: string; count: number }[];
  by_type: { name: string; count: number }[];
  extents: { min: number[]; max: number[] } | null;
  total_length: number | null;
  total_area: number | null;
}

/** Aggregate description of an explicit handle list.
 *
 *  Used by the UI for one thing only: sampling a rendered layout to find out
 *  which layouts its entities really belong to. See `regionBlockedReason` in
 *  page.tsx — a paper sheet draws model space through a viewport, so the
 *  handles on screen are not in the coordinate space the sheet is measured in.
 */
export function describeSelection(
  drawingId: string,
  handles: string[],
  signal?: AbortSignal,
): Promise<DescribeSelectionResult> {
  return request(`/drawings/${drawingId}/selection/describe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ handles, sample: 0 }),
    signal,
  });
}

/** Store a selection and get a reference to it.
 *
 *  The reference is what travels to the agent. Sending the handle list meant
 *  ~16 KB re-sent with every message and a 2,000-handle cap, so bigger
 *  selections came back answered only in part. Stored, the agent's tools read
 *  the whole thing server-side. Selections expire on their own.
 */
/** Store a selection and get a reference to it.
 *
 *  The exact counts travel alongside the handles, and that separation is the
 *  point: `handles` is what can be NAMED and has a ceiling, `total` /
 *  `by_layer` / `by_type` are what the selection actually HOLDS and have
 *  none. Sending both lets the agent answer "how many, on which layers"
 *  exactly for a selection far larger than any list it could be handed.
 */
export function saveSelection(
  drawingId: string,
  input: {
    handles: string[];
    layout?: string | null;
    mode?: string | null;
    kind?: string | null;
    summary?: string | null;
    total?: number;
    by_layer?: { name: string; count: number }[];
    by_type?: { name: string; count: number }[];
  },
): Promise<{
  selection_id: string;
  total: number;
  enumerated: number;
  truncated: boolean;
}> {
  return request(`/drawings/${drawingId}/selections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

// ---------------------------------------------------------------------------
// Business names for layers — UPLIFT-13
// ---------------------------------------------------------------------------
//
// The engine behind these is built, wired and tested; what was missing was a
// way to reach it that is not curl. The rules about what a valid tag is live
// in `lib/layerTags.ts`, next to the arithmetic, and nothing in this section
// decides anything — it moves JSON.

/** Every business name already given for this drawing.
 *
 *  One request for the whole drawing rather than one per layer: the panel
 *  lists up to a few hundred layers at a time and a per-row fetch would be a
 *  few hundred round trips to render a list nobody has interacted with yet.
 */
export function listLayerTags(
  drawingId: string,
  signal?: AbortSignal,
): Promise<{ drawing_id: string; tags: LayerTag[] }> {
  return request(`/drawings/${encodeURIComponent(drawingId)}/tags`, { signal });
}

/** How many objects a name would apply to, asked BEFORE anything is saved.
 *
 *  The whole point of naming by layer is that one name reaches every object
 *  on it. That is only safe if the person typing it can see the size of
 *  "every" first, which is why this is a separate read and not a field on the
 *  save response.
 */
export function getLayerImpact(
  drawingId: string,
  layer: string,
  layout: string | null,
  signal?: AbortSignal,
): Promise<LayerImpact> {
  const query = layout ? `?layout=${encodeURIComponent(layout)}` : "";
  return request(
    `/drawings/${encodeURIComponent(drawingId)}/tags/${encodeURIComponent(
      layer,
    )}/impact${query}`,
    { signal },
  );
}

/** Save one business name for one layer.
 *
 *  `source` has no default and is not optional here for the same reason it is
 *  not optional on the server: a tag without an origin is a claim without
 *  evidence, and in six months nobody can tell it from a fact. A refusal
 *  comes back as an `ApiError` carrying the server's own `hint`, which the
 *  panel shows verbatim rather than paraphrasing.
 */
export function putLayerTag(
  drawingId: string,
  layer: string,
  body: {
    business_name: string;
    source: string;
    author: string;
    use?: string | null;
  },
): Promise<LayerTag> {
  return request(
    `/drawings/${encodeURIComponent(drawingId)}/tags/${encodeURIComponent(layer)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

/** A read that is allowed to come back empty, with the reason in words.
 *
 *  Same shape and same argument as `Optional` in `lib/parcelContext.ts`, and
 *  deliberately not imported from there: that file is the seam for work being
 *  written in parallel and this one is the typed client. Making the client
 *  depend on the seam would point the arrow the wrong way.
 */
export interface Tolerant<T> {
  data: T | null;
  reason: string | null;
}

/** What the reviewed configuration already says about this drawing's layers.
 *
 *  Tolerant on purpose, and the failure modes are worth naming. A drawing
 *  with no land-use configuration is the ordinary case, not an error — most
 *  drawings here have none — and a panel that showed a red box for it would
 *  be shouting about the normal state of the world. A configuration that
 *  exists and is broken answers 400 with a message, and that message is shown
 *  where the comparison would have been.
 *
 *  Read through the land-use summary because that is the merged route that
 *  carries configured values per layer; see the note in `lib/layerTags.ts`
 *  about the effective-value route that will replace it.
 */
export async function fetchConfiguredLayerValues(
  drawingId: string,
  layout: string,
  signal?: AbortSignal,
): Promise<Tolerant<LandUseSummaryLite>> {
  if (!drawingId || !layout) {
    return { data: null, reason: "No drawing and layout to compare against." };
  }
  let response: Response;
  try {
    response = await fetch(
      `${API_BASE}/drawings/${encodeURIComponent(drawingId)}/land-use?layout=${encodeURIComponent(layout)}`,
      { signal },
    );
  } catch {
    return { data: null, reason: `Could not reach cad-api at ${API_BASE}.` };
  }
  if (!response.ok) {
    let body: Partial<ApiErrorBody> = {};
    try {
      body = await response.json();
    } catch {
      /* a non-JSON error body is still an error */
    }
    return {
      data: null,
      reason:
        body.hint && body.message
          ? `${body.message} — ${body.hint}`
          : (body.message ?? `${response.status} ${response.statusText}`),
    };
  }
  try {
    return { data: (await response.json()) as LandUseSummaryLite, reason: null };
  } catch {
    return { data: null, reason: "cad-api answered with something that is not JSON." };
  }
}

/** One document carried inside a drawing rather than drawn by it. */
export interface EmbeddedPayload {
  handle: string;
  kind: string;
  format: string;
  media_type: string;
  bytes: number;
  width: number | null;
  height: number | null;
  note: string | null;
  file: string | null;
}

export interface EmbeddedIndex {
  drawing_id: string;
  declared: number;
  payloads: EmbeddedPayload[];
  note: string | null;
  not_measured?: string | null;
}

/** The URL a browser can open one embedded document at. */
export function embeddedUrl(drawingId: string, handle: string): string {
  return `${API_BASE}/drawings/${encodeURIComponent(drawingId)}/embedded/${encodeURIComponent(handle)}`;
}

/** What this drawing carries inside it.
 *
 *  Worth the round trip, because this is where the meaning of a drawing often
 *  lives. The reference file's three "embedded OLE objects" turned out to be
 *  the approvals sheet and the land-use key — the page naming which plot
 *  numbers are commercial, which are educational, and that the rest are
 *  residential villas — while the pipeline spent weeks inferring that same
 *  fact by elimination. The banner had reported the objects honestly the
 *  whole time; there was simply no way to open one.
 *
 *  Tolerant like its neighbours: a drawing that embeds nothing is the ordinary
 *  case, and a drawing whose DXF is no longer on disk can still say how many
 *  it declares, because that count comes from the stored entities.
 */
export async function fetchEmbedded(
  drawingId: string,
  signal?: AbortSignal,
): Promise<Tolerant<EmbeddedIndex>> {
  if (!drawingId) {
    return { data: null, reason: "No drawing selected." };
  }
  let response: Response;
  try {
    response = await fetch(
      `${API_BASE}/drawings/${encodeURIComponent(drawingId)}/embedded`,
      { signal },
    );
  } catch {
    return { data: null, reason: `Could not reach cad-api at ${API_BASE}.` };
  }
  if (!response.ok) {
    return {
      data: null,
      reason: `${response.status} ${response.statusText}`,
    };
  }
  try {
    return { data: (await response.json()) as EmbeddedIndex, reason: null };
  } catch {
    return { data: null, reason: "cad-api answered with something that is not JSON." };
  }
}

/** H3 resolutions offered on the export controls.
 *
 *  Three, not fifteen. A resolution is a cell SIZE, and only a few sizes mean
 *  anything against a site plan: 9 is roughly a neighbourhood, 11 roughly a
 *  block, 13 roughly a plot. Offering 0-15 would offer continents and
 *  doorsteps beside them.
 *
 *  The API accepts any resolution and derives coarser cells from the stored
 *  ones; finer than stored is refused there, not here. These are the choices
 *  a person is given, not a claim about what the endpoint supports.
 */
export const EXPORT_RESOLUTIONS = [9, 11, 13] as const;

/** What the controls start on: block scale, the one that reads as a site. */
export const DEFAULT_EXPORT_RES = 11;

/** Whether the cell-size picker is offered on screen.
 *
 *  Off, by the owner's call on 1 September 2026, and the reason is worth
 *  keeping: the control changes only the DOWNLOADED FILE, so on screen it
 *  looks broken. Clicking through its three options does nothing visible,
 *  and for the sizes an answer usually covers it barely changes the file
 *  either — nine school parcels export as 8 hexagons at res 9 and 9 at both
 *  res 11 and 13. A control that has to be explained before it can be
 *  demonstrated is one more thing to explain in a demo.
 *
 *  Hidden, NOT removed. Every export still runs at `DEFAULT_EXPORT_RES`,
 *  which is what the picker already started on, so no file changes. The
 *  choice matters on a dense answer, and when someone wants it back this
 *  flag is the whole change — the state, the props and the markup are all
 *  still wired.
 */
export const SHOW_EXPORT_RES_PICKER: boolean = false;

/** The words beside each number, so the choice is a size and not a code. */
export const EXPORT_RESOLUTION_LABELS: Record<number, string> = {
  9: "9 · neighbourhood",
  11: "11 · block",
  13: "13 · plot",
};

// ---------------------------------------------------------------------------
// Upload and ingest runs
// ---------------------------------------------------------------------------
//
// Frozen in `docs/INGEST-RUN-CONTRACT.md`. Three routes, and they live here
// with every other read rather than in `lib/ingestRuns.ts`, because this file
// already owns `request<T>` and `ApiError` — the network error, the JSON error
// body, the code/message/hint triple. A second fetch wrapper next to the run
// types would be a second client, and the first thing it would drift on is
// exactly the error handling the panel depends on.
//
// The upload goes DIRECT to cad-api like every other call here. There is no
// server-side secret in it, CORS is already configured for the web origin, and
// a Next.js route handler in the middle would mean a 500 MB DXF crossing the
// node process for no reason at all.

/** Send one drawing. `202 Accepted`; the caller then polls the run.
 *
 *  The File object goes into the FormData as-is, never read into a string
 *  first: the browser streams a multipart body from disk, and `await
 *  file.text()` on Sedra's 500 MB DXF would try to hold half a gigabyte in a
 *  JavaScript string before a single byte left the machine.
 *
 *  No `Content-Type` header, deliberately. `fetch` sets it from the FormData
 *  along with the multipart boundary, and setting it by hand produces a body
 *  the server cannot parse — the boundary would be missing.
 *
 *  Field name `file`, exactly, from the contract: "multipart/form-data, one
 *  field: file".
 */
export function uploadDrawing(file: File): Promise<UploadAccepted> {
  const form = new FormData();
  form.append("file", file);
  return request("/drawings/upload", { method: "POST", body: form });
}

/** One run, plus the derived `stalled` / `stalled_since`. 404 when unknown —
 *  which is the ordinary answer for a run id kept in session storage across a
 *  database that has since been reset, so the caller is expected to handle it
 *  rather than treat it as broken. */
export function getIngestRun(runId: string): Promise<IngestRun> {
  return request(`/drawings/ingest-runs/${encodeURIComponent(runId)}`);
}

/** Runs, newest first. `drawing_id` filters; `limit` defaults to 20 on the
 *  server and is capped there at 100.
 *
 *  This is also the "latest run for this drawing" read: ask with `drawing_id`
 *  and take the first row. Both parameters are omitted from the query string
 *  when they are not given, so the server's defaults stay the server's. */
export function listIngestRuns(opts?: {
  drawing_id?: string;
  limit?: number;
}): Promise<IngestRunList> {
  const query = new URLSearchParams();
  if (opts?.drawing_id) query.set("drawing_id", opts.drawing_id);
  if (typeof opts?.limit === "number") query.set("limit", String(opts.limit));
  const qs = query.toString();
  return request(`/drawings/ingest-runs${qs ? `?${qs}` : ""}`);
}
