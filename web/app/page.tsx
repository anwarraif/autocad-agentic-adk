"use client";

/**
 * The viewer page: drawing picker, layout picker, SVG canvas, layer panel,
 * entity/comment panel.
 *
 * All state lives here rather than in a store: the app has one screen and one
 * selected drawing, and a store would be indirection without a second consumer.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE,
  ApiError,
  embeddedUrl,
  fetchEmbedded,
  fetchSvg,
  getDrawing,
  listComments,
  DEFAULT_EXPORT_RES,
  listDrawings,
  describeSelection,
  listRegionGroup,
  saveSelection,
  selectRegion,
  type DrawingDetail,
  type DrawingSummary,
  type EmbeddedIndex,
  type LayoutSummary,
  type Region,
  type SelectionResult,
} from "@/lib/api";
import { Viewer, type Selection } from "./components/Viewer";
import { MapViewer } from "./components/MapViewer";
import {
  fetchGeometry,
  fetchScopeLayers,
  type GeometryResponse,
  type ScopeLayers,
} from "@/lib/mapGeometry";
import {
  fetchGeo,
  resForZoom,
  boundsFor,
  clampRes,
  DEFAULT_CELL_RES,
  CELL_RES_SETTLE_MS,
  type GeoResponse,
} from "@/lib/mapCells";
import type { RegionHit, RegionMode } from "@/lib/mapRegion";
import {
  combineGroups,
  describeGroups,
  nextGroupId,
  nextRegionLabel,
  exactCounts,
  type SelectionGroup,
} from "../lib/selectionSet";
import { LayerPanel } from "./components/LayerPanel";
import { EntityPanel } from "./components/EntityPanel";
import { ChatPanel } from "./components/ChatPanel";
import { SelectionPanel } from "./components/SelectionPanel";
import { UploadPanel } from "./components/UploadPanel";
import {
  buildDomIndex,
  selectFromDom,
  type DomSelectionEntry,
  type DomSelectionResult,
} from "@/lib/domSelection";
import {
  collectHandleCandidates,
  planAnswerHighlight,
  type AnswerPlan,
} from "@/lib/answerHighlight";
import {
  describeParcelContext,
  fetchMarkPoints,
  fetchParcelContext,
  isLabelType,
  type MarkPoint,
  type Optional,
  type ParcelContext,
} from "@/lib/parcelContext";

type Tab = "layers" | "entity" | "agent" | "info" | "selection" | "upload";

/** Which of the three views is on screen.
 *
 *  Three, not two, because they answer different questions and none of them
 *  subsumes another. `svg` is the plotted sheet — border, title block,
 *  dimension text, Arabic MTEXT — rendered by ezdxf through the calibrations
 *  in `render.py`, and it is the only view that can show a paper layout at
 *  all. The two deck views draw stored vertex geometry: real geography,
 *  extrusion, and tens of thousands of features without touching the DOM.
 *  §3.1 of docs/UI-BACKEND-ARCHITECTURE.html argues the case for keeping all
 *  three rather than replacing one with another. */
type ViewMode = "deck2d" | "deck3d" | "svg";

const VIEW_LABELS: Record<ViewMode, string> = {
  deck2d: "Deck.GL 2D",
  deck3d: "Deck.GL 3D",
  svg: "Normal 2D",
};

/** Entities sampled to decide whether a layout can support region selection.
 *  Big enough that a sheet's own border cannot hide a viewport behind it,
 *  small enough to be one cheap request per layout load. */
const SAMPLE_SIZE = 40;


/** A paperspace sheet: a real layout that is not modelspace and not a block.
 *  This is "the drawing" as an engineer means it — border, title block,
 *  details at their intended scale. */
function isSheet(l: LayoutSummary): boolean {
  return !l.is_block && l.name.toLowerCase() !== "model";
}

/** Objects actually drawn for a layout, viewport expansion included.
 *  A sheet owns almost no entities of its own — the annotation sample's sheet
 *  owns 9 and draws 959 — so `entity_count` is the wrong thing to rank by. */
function drawnCount(detail: DrawingDetail | null, layout: string): number {
  return (
    detail?.renders?.find((r) => r.layout === layout)?.rendered_entities ?? 0
  );
}

/** Mirrors cad-api's `_default_layout`: a sheet first, then modelspace, then a
 *  block definition. Kept in step with the backend so a page load and a bare
 *  `GET /svg` show the same thing. */
function pickDefaultLayout(doc: DrawingDetail): string {
  const rendered = new Set(doc.renders?.map((r) => r.layout) ?? []);
  let candidates = (doc.layouts ?? []).filter((l) => rendered.has(l.name));

  // A rendered SVG is what the Normal 2D view needs. The two map views need
  // GEOMETRY, which is a different thing and is already in the store. Gating
  // the whole screen on a render meant a drawing that had been ingested
  // perfectly showed an empty layout picker and an empty map, with nothing on
  // screen to say why: Sedra stored 990,790 entities, had every one of them
  // placed and celled, and still offered no layout to look at because its
  // render stage had run out of memory.
  //
  // So when nothing is rendered, fall back to the layouts that actually hold
  // entities. The Normal 2D view already says honestly that it has no render
  // for a layout; the map views draw.
  if (candidates.length === 0) {
    candidates = (doc.layouts ?? []).filter(
      (l) => !l.is_block && (drawnCount(doc, l.name) > 0 || l.entity_count > 0),
    );
  }
  if (candidates.length === 0) return "";

  const sheets = candidates.filter(isSheet);
  if (sheets.length > 0) {
    return sheets.sort(
      (a, b) => drawnCount(doc, b.name) - drawnCount(doc, a.name),
    )[0].name;
  }
  const model = candidates.find((l) => !l.is_block);
  if (model) return model.name;

  // Nothing rendered but block definitions. Prefer model space anyway when it
  // holds anything: the map views read world geometry and do not care whether
  // a sheet was produced, and landing a reader inside one block definition of
  // a 990,790 entity drawing shows them a hedge detail instead of the estate.
  // Sedra does exactly this, because its model space is too large to render
  // and 50 of its block layouts are not.
  const modelspace = (doc.layouts ?? []).find(
    (l) => l.is_modelspace && l.entity_count > 0,
  );
  if (modelspace) return modelspace.name;
  return candidates.sort(
    (a, b) => drawnCount(doc, b.name) - drawnCount(doc, a.name),
  )[0].name;
}

/** The kinds worth drawing first, in the order that makes a map look like the
 *  drawing: closed shapes carry the outline, the area and the land use.
 *  Returns the subset of what was asked for, so a caller that never wanted
 *  rings still gets what it asked for. */
function ringsFirst(kinds: string): string {
  const wanted = kinds.split(",").map((k) => k.trim()).filter(Boolean);
  return wanted.includes("ring") ? "ring" : kinds;
}

/** Everything asked for that `ringsFirst` deferred, or null when nothing was. */
function remainingKinds(kinds: string): string | null {
  const wanted = kinds.split(",").map((k) => k.trim()).filter(Boolean);
  if (!wanted.includes("ring")) return null;
  const rest = wanted.filter((k) => k !== "ring");
  return rest.length > 0 ? rest.join(",") : null;
}

/** What the first request asks for, and the ceiling the map accumulates to.
 *
 *  The first page decides whether the map looks alive. The ceiling exists
 *  because a browser holding every vertex of a million-object drawing is a
 *  tab that stops responding; when it bites, `total_matches` still reports
 *  the whole layout, so what is on screen is never mistaken for the drawing.
 */
const FIRST_PAINT_FEATURES = 4_000;
const MAX_ACCUMULATED_FEATURES = 24_000;

/** How much url-encoded `?layers=` is worth sending.
 *
 *  Request lines are commonly capped around 8 KB and a filter that makes the
 *  request fail is worse than one applied in the browser. Half of that leaves
 *  room for the rest of the query. */
const MAX_LAYER_PARAM_CHARS = 4_000;

/** The zoom the FIRST page is fetched for, before the camera has reported one.
 *
 *  Without it the first paint asks for every stored vertex and then refetches
 *  the moment the camera settles -- which is slower than not simplifying at
 *  all. There is a chicken-and-egg here: the zoom comes from the fit, and the
 *  fit comes from the data. 14 is the zoom a site of one to five kilometres
 *  fits at, which is what these drawings are, so the first guess is usually
 *  the settled bucket and no refetch follows.
 */
const FIRST_PAINT_ZOOM = 14;

/** Zoom, rounded to a two-level bucket. See `detailZoom`. */
function detailBucket(zoom: number): number {
  return Math.max(0, Math.min(24, Math.round(zoom / 2) * 2));
}

export default function Page() {
  const [drawings, setDrawings] = useState<DrawingSummary[]>([]);
  /** Cell size for the drawing export. Lives here because the URL is built
   *  here; the viewer only draws the control. */
  const [geoExportRes, setGeoExportRes] = useState<number>(DEFAULT_EXPORT_RES);
  const [drawingId, setDrawingId] = useState<string>("");
  const [detail, setDetail] = useState<DrawingDetail | null>(null);
  const [embedded, setEmbedded] = useState<EmbeddedIndex | null>(null);
  const [layout, setLayout] = useState<string>("");
  const [svg, setSvg] = useState<string | null>(null);
  const [svgLoading, setSvgLoading] = useState(false);
  const [renderedLayers, setRenderedLayers] = useState<
    { name: string; count: number }[]
  >([]);
  const [hiddenLayers, setHiddenLayers] = useState<Set<string>>(new Set());
  // Stable identity: this runs from the Viewer's mount effect, and a fresh
  // arrow per render would re-trigger that effect in a loop.
  const onOffLayersDiscovered = useCallback(
    (names: string[]) => setHiddenLayers(new Set(names)),
    [],
  );
  const [selected, setSelected] = useState<Selection | null>(null);
  const [commentedHandles, setCommentedHandles] = useState<Set<string>>(new Set());
  const [commentsByHandle, setCommentsByHandle] = useState<
    Record<string, { author: string; body: string }[]>
  >({});
  const [tab, setTab] = useState<Tab>("entity");
  const [error, setError] = useState<string | null>(null);
  const [booting, setBooting] = useState(true);
  // The side panel can always be reached, at any window size. It used to be
  // laid out below the viewer on a narrow screen, where a tall header could
  // push it off the bottom of a 100vh page with `overflow: hidden` -- the
  // panel was still in the DOM, simply unreachable. It is now a drawer.
  const [panelOpen, setPanelOpen] = useState(true);

  // --- region selection ----------------------------------------------------
  // The region lives here, in drawing coordinates, so that it survives every
  // re-render of the viewer and can be re-queried when the mode is flipped.
  const [regionTool, setRegionTool] = useState<"none" | "polygon" | "circle">("none");
  /** The selection, as the list of parts that made it.
   *
   *  Drawing a second region used to throw the first away. It now appends,
   *  which is both what a CAD user expects and what was asked for: each part
   *  keeps its own shape and its own counts, and the panel shows them as
   *  separate groups. `region` and `regionResult` below are DERIVED from
   *  this, so every consumer written against a single selection keeps
   *  working unchanged. See lib/selectionSet.ts. */
  const [groups, setGroups] = useState<SelectionGroup[]>([]);
  const [regionLoading, setRegionLoading] = useState(false);
  const [regionError, setRegionError] = useState<string | null>(null);
  const [regionSupported, setRegionSupported] = useState(true);

  /** Every drawn shape, for the viewer to outline. A click contributes none. */
  const regions = useMemo(
    () => groups.map((g) => g.region).filter((r): r is Region => r !== null),
    [groups],
  );
  /** The part a mode flip acts on: the most recent region the user drew. */
  const region = useMemo(
    () => regions.length > 0 ? regions[regions.length - 1] : null,
    [regions],
  );
  const combined = useMemo(
    () => combineGroups(groups, { drawingId, layout }),
    [groups, drawingId, layout],
  );
  const regionResult = combined?.result ?? null;
  /** Which mechanism answers a region on THIS layout.
   *
   *  Not guessed from the layout's name — the rendered layout is sampled and
   *  the API is asked which layouts those entities really belong to. Two
   *  answers, two mechanisms:
   *
   *  - `server`: the entities live in this layout's own coordinate space
   *    (model space, block definitions), so the region is answered from
   *    stored bounding boxes — the audited, stored-truth path.
   *  - `client`: the layout shows geometry from ANOTHER space (a paper sheet
   *    projecting model space through viewports — Janadriyah's DMP Layout1
   *    spans 1199 x 817 paper millimetres while carrying entities whose
   *    stored boxes sit at 692,064 / 2,750,796). Stored boxes are the wrong
   *    numbers here, so the region is answered from what is actually DRAWN,
   *    via the rendered-geometry index. This used to be a hard block with a
   *    "switch layout" errand; selecting what you can see is the answer the
   *    block should have been.
   *
   *  `pending` (calibration still in flight) runs the client path: it is
   *  visually correct everywhere, so it is the safe default.
   */
  const [selectionMechanism, setSelectionMechanism] = useState<
    "pending" | "server" | "client"
  >("pending");
  /** The layout the visible entities are actually STORED under.
   *
   *  Usually the layout on screen — but not on a paper sheet, which draws
   *  model space through a viewport: what you see on "DMP Layout1" is stored
   *  under "Model". Filtering the tools by the sheet name therefore matches
   *  almost nothing, and the agent reports the absence as fact. Measured:
   *  asked whether a clicked LWPOLYLINE on layer VL2 was the only one on the
   *  sheet, it answered yes; there are 133, all stored under Model. This is
   *  the layout the agent must scope its queries to. */
  const [scopeLayout, setScopeLayout] = useState<string>("");
  /** Rendered-geometry index for the client mechanism, built lazily on the
   *  first selection and valid for the life of the current SVG. */
  const domIndex = useRef<DomSelectionEntry[] | null>(null);
  /** Handles handed to the agent tab as context, separate from `region` so
   *  clearing the region does not silently empty an open conversation. */
  const [agentSelection, setAgentSelection] = useState<{
    selectionId: string;
    total: number;
    summary: string;
  } | null>(null);

  // --- the agent's answer, marked on the drawing ---------------------------
  /** What the last answer named, split into what can be marked and what can
   *  only be reported. See lib/answerHighlight.ts. */
  const [answerPlan, setAnswerPlan] = useState<AnswerPlan | null>(null);
  /** handle -> the layer it was read for. Empty when nothing is marked. */
  const [answerGroups, setAnswerGroups] = useState<Map<string, string>>(
    () => new Map(),
  );
  /** Stored centroids for those objects, when cad-api can supply them. */
  const [answerPoints, setAnswerPoints] = useState<MarkPoint[] | null>(null);
  const [answerPointsNote, setAnswerPointsNote] = useState<string | null>(null);
  const answerAbort = useRef<AbortController | null>(null);

  /** The parcel a selected label sits inside. Null when the selection is not
   *  a label, or before the lookup returns. */
  const [parcel, setParcel] = useState<Optional<ParcelContext> | null>(null);
  const [parcelLoading, setParcelLoading] = useState(false);

  // --- the map views -------------------------------------------------------
  //
  //  Deck.GL 2D is the default, and it degrades rather than refusing: a
  //  layout whose objects carry no stored vertices falls back to the SVG
  //  sheet with a line saying why. Eighteen of the nineteen ingested drawings
  //  have no CRS, so "cannot be placed on a map" is the common case, and a
  //  default that showed those eighteen an error screen would be a default
  //  that is wrong most of the time. What the deck views need is *geometry*,
  //  not a CRS — without one they draw the drawing in its own coordinates.
  const [viewMode, setViewMode] = useState<ViewMode>("deck2d");
  const [geometry, setGeometry] = useState<GeometryResponse | null>(null);
  const [geometryLoading, setGeometryLoading] = useState(false);
  /** Whether the layout is bigger than one request can carry.
   *
   *  Set from the server's own `truncated`, never inferred from a row count.
   *  Where a layout arrives whole -- Janadriyah is 20,331 features under a
   *  24,000 cap -- the viewport changes nothing and is not applied, so that
   *  drawing's behaviour is exactly what it was. */
  const [refineWithViewport, setRefineWithViewport] = useState<string | null>(null);
  /** The camera, once it has stopped moving.
   *
   *  Every frame of a drag reports bounds; refetching on each would be a
   *  request per frame. Committed after the camera has been still for a
   *  moment, and keyed below rounded to four decimals -- about 11 m -- so
   *  sub-pixel jitter is not a new question. */
  const [settledBounds, setSettledBounds] = useState<{
    owner: string;
    box: [number, number, number, number];
  } | null>(null);
  const boundsSettle = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** Who the camera currently belongs to.
   *
   *  A ref rather than a dep, so `onMapBounds` keeps a stable identity -- it
   *  is read by an effect in MapViewer, and a fresh arrow per render would
   *  re-trigger that effect on every frame. */
  const boundsOwner = useRef("");
  const onMapBounds = useCallback((b: [number, number, number, number]) => {
    const owner = boundsOwner.current;
    if (boundsSettle.current) clearTimeout(boundsSettle.current);
    boundsSettle.current = setTimeout(
      () => setSettledBounds({ owner, box: b }),
      450,
    );
  }, []);
  /** The layers present in the MAP's scope, from the store.
   *
   *  The panel's membership and its denominators. Kept apart from `geometry`
   *  because it answers a question paging cannot: which layers this scope
   *  HAS, as opposed to which have turned up so far. */
  const [scopeLayers, setScopeLayers] = useState<ScopeLayers | null>(null);
  const [geometryError, setGeometryError] = useState<string | null>(null);
  const [geometryFor, setGeometryFor] = useState<string | null>(null);
  const geometryAbort = useRef<AbortController | null>(null);
  /** The last QUESTION asked, ignoring where the camera pointed. A request
   *  sharing it is a pan, and a pan keeps the features it already has. */
  const geometrySubject = useRef<string | null>(null);
  /** Coarse H3 cells already fetched for the current subject.
   *
   *  Sent back as `have` so a pan asks only for what is newly in view. Held
   *  in a ref rather than state: it must not re-trigger the effect that fills
   *  it, and nothing renders from it. Cleared whenever the subject changes,
   *  because cells belong to a drawing and a scope. */
  const heldCells = useRef<Set<string>>(new Set());
  /** The `drawing:layout:kinds` already REQUESTED, as opposed to already
   *  answered.
   *
   *  `geometryFor` records what arrived, and is written in `.finally` — far
   *  too late to stop a second request. The scope question settles in two
   *  steps (`selectionMechanism` leaving "pending", then `scopeLayout`
   *  resolving), and the effect below runs on both. Measured in the browser:
   *  every load fetched 6.3 MB twice. */
  const geometryRequested = useRef<string | null>(null);
  /** The H3 layer for the drawing on screen.
   *
   *  Fetched here rather than inside `MapViewer` so that one component owns
   *  what is loaded for which drawing. Two components fetching on their own
   *  render cycles is how a view ends up drawing one drawing's cells over
   *  another drawing's parcels for a frame or two. */
  /** Whether the map views draw text labels — and therefore whether the
   *  geometry request asks for them at all.
   *
   *  Owned here rather than inside `MapViewer` because it decides what is
   *  fetched, not just what is drawn. Measured on the reference drawing:
   *  labels are 5,522 of 9,761 features and 227 KB of a 755 KB gzipped
   *  payload — 30% of every load, for a switch that is off by default and
   *  that most sessions never touch. */
  const [showLabels, setShowLabels] = useState(false);
  const kinds = showLabels ? "ring,path,segment,label" : "ring,path,segment";

  const [cells, setCells] = useState<GeoResponse | null>(null);
  const [cellsLoading, setCellsLoading] = useState(false);
  /** Which rung of the H3 hierarchy the cells request asks for.
   *
   *  Harsh's point, and the reason this is state rather than the constant it
   *  was: H3 is a hierarchical clustering, and a view pinned to one
   *  resolution shows one hierarchy. The camera decides the rung now — see
   *  `resForZoom` for the ladder and the bands. */
  const [cellRes, setCellRes] = useState<number>(DEFAULT_CELL_RES);
  /** The zoom the geometry is fetched FOR, in coarse buckets.
   *
   *  Version 8 stores each arc's real curve, which is right, and made features
   *  fatter: measured on Atlas, 1,221 bytes per feature on Sedra against about
   *  690 before. Sending the zoom lets the server drop vertices closer
   *  together than a pixel, which took Sedra back to 705 B per feature and
   *  15.7 points down to 2.4 -- without touching what is stored.
   *
   *  BUCKETED, because this is part of the request key and a fetch per wheel
   *  notch would cost far more than the bytes it saves. Two zoom levels per
   *  bucket: the error a bucket allows is still under a pixel at its coarse
   *  end, and refetches stay rare. */
  const [detailZoom, setDetailZoom] = useState<number>(FIRST_PAINT_ZOOM);
  /** Pending band change. The camera reports every frame of a pinch; only the
   *  band it comes to rest in should cost a request. */
  const cellResSettle = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** The rungs THIS drawing has, learnt from the last payload it answered
   *  with. A ref rather than state because it only ever narrows a value the
   *  camera has already produced; putting it in the dependency list would
   *  rebuild the zoom handler on every fetch for no change in behaviour. */
  const cellResBounds = useRef<{ min: number; max: number } | null>(null);
  /** Why there are no cells, when there are none.
   *
   *  A string rather than an error, because for seventeen of the eighteen
   *  drawings "no cells" is a FACT about the drawing — it declares no
   *  coordinate system — and not a failure. The API answers with the same
   *  shape either way, so this is shown on a disabled control instead of
   *  being thrown. */
  const [cellsReason, setCellsReason] = useState<string | null>(null);
  const cellsAbort = useRef<AbortController | null>(null);
  /** What has been asked for, for the same reason as `geometryRequested`. */
  const cellsRequested = useRef<string | null>(null);
  /** The `drawing:layout` whose off-layers have been applied, so the default
   *  is seeded once and a later re-render cannot undo a user's toggle. */
  const offLayersSeeded = useRef<string | null>(null);
  /** Set once per layout when the deck views cannot draw it, so the fallback
   *  to the sheet happens exactly once and the user can still switch back. */
  const autoFellBack = useRef<string | null>(null);

  const svgAbort = useRef<AbortController | null>(null);
  const regionAbort = useRef<AbortController | null>(null);
  /** Index of the part a pending re-query belongs to, or null when the query
   *  is a new part. Set for a window<->crossing flip and for a circle
   *  reshaped by its grips: both re-ask a part that already exists, and
   *  appending the answer would silently double the selection. */
  const replacePart = useRef<number | null>(null);
  /** Rendered-geometry matches per part, for listing a part's rows. Keyed by
   *  group id because each part was measured against its own shape. */
  const groupMatches = useRef(new Map<string, DomSelectionResult["matches"]>());

  // --- load the drawing list ----------------------------------------------
  //
  // A callback rather than a one-shot effect body, because the list is no
  // longer read once. Ingesting a drawing from the Upload tab adds one to the
  // store while this page is open, and with `[]` deps the new drawing would
  // not appear in the picker until a manual reload -- the run contract names
  // that and requires the refetch.
  //
  // The selection is seeded ONLY when nothing is selected. Re-seeding it to
  // `drawings[0]` on every refetch would yank a reader off the drawing they
  // were reading the moment an unrelated ingest finished.
  const loadDrawings = useCallback(async () => {
    const { drawings: rows } = await listDrawings();
    setDrawings(rows);
    setDrawingId((current) => current || (rows[0]?._id ?? ""));
  }, []);

  useEffect(() => {
    loadDrawings()
      .catch((err: unknown) => setError(describe(err)))
      .finally(() => setBooting(false));
  }, [loadDrawings]);

  /** A drawing has just been ingested: refresh the picker so it is there.
   *
   *  The drawing is deliberately NOT selected. The upload panel offers that
   *  as a button instead, because switching the viewer out from under someone
   *  who started an ingest and went back to reading is a surprise, not a
   *  convenience. */
  const onIngested = useCallback(() => {
    loadDrawings().catch((err: unknown) => setError(describe(err)));
  }, [loadDrawings]);

  /** Whether the picker has this drawing, so the upload panel can offer to
   *  open it -- and say why it cannot, when the refetch has not landed. */
  const isDrawingListed = useCallback(
    (id: string) => drawings.some((d) => d._id === id),
    [drawings],
  );

  // --- what this drawing carries inside it ---------------------------------
  //
  //  Asked once per drawing, and asked even when the render reports no
  //  unrenderable objects: a drawing can embed a document on a layout the
  //  viewer is not currently showing, and "there is nothing inside this file"
  //  is a claim worth being right about. The route answers the count from
  //  stored entities without touching the DXF, so a drawing that embeds
  //  nothing costs one cheap query.
  useEffect(() => {
    if (!drawingId) {
      setEmbedded(null);
      return;
    }
    const control = new AbortController();
    setEmbedded(null);
    fetchEmbedded(drawingId, control.signal).then(({ data }) => {
      if (!control.signal.aborted) setEmbedded(data);
    });
    return () => control.abort();
  }, [drawingId]);

  // --- load the selected drawing's metadata --------------------------------
  useEffect(() => {
    if (!drawingId) return;
    let cancelled = false;
    setSelected(null);
    setError(null);
    // Cleared synchronously. `layout` still held the previous drawing's
    // layout name, and the SVG effect keys on [drawingId, layout] — so
    // switching drawings fired a fetch for {new drawing, old layout}, which
    // either 404'd and flashed a false "re-run the ingest" banner, or started
    // a multi-megabyte download that was thrown away a moment later.
    setLayout("");
    setDetail(null);
    getDrawing(drawingId)
      .then((doc) => {
        if (cancelled) return;
        setDetail(doc);
        setCommentedHandles(new Set(Object.keys(doc.comment_counts ?? {})));
        void listComments(drawingId)
          .then(({ comments }) => {
            if (cancelled) return;
            const map: Record<string, { author: string; body: string }[]> = {};
            comments.forEach((c) => {
              (map[c.entity_handle] ??= []).push({
                author: c.author,
                body: c.body,
              });
            });
            setCommentsByHandle(map);
          })
          .catch(() => setCommentsByHandle({}));
        setLayout(pickDefaultLayout(doc));
      })
      .catch((err: unknown) => !cancelled && setError(describe(err)));
    return () => {
      cancelled = true;
    };
  }, [drawingId]);

  // --- load the SVG --------------------------------------------------------
  useEffect(() => {
    if (!drawingId || !layout) {
      setSvg(null);
      return;
    }
    svgAbort.current?.abort();
    const controller = new AbortController();
    svgAbort.current = controller;

    setSvgLoading(true);
    setSvg(null);
    setHiddenLayers(new Set());
    setRenderedLayers([]);
    setSelected(null);

    fetchSvg(drawingId, layout, controller.signal)
      .then((text) => {
        if (controller.signal.aborted) return;
        setSvg(text);
        setError(null);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setSvg(null);
        setError(describe(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setSvgLoading(false);
      });

    return () => controller.abort();
  }, [drawingId, layout]);

  // Sample the rendered layout and check that what is on screen really lives
  // in the coordinate space the region will be measured in. Child effects run
  // before parent effects, so by the time this runs the Viewer has already
  // injected the SVG into the DOM.
  useEffect(() => {
    domIndex.current = null; // a new render invalidates the geometry index
    setSelectionMechanism("pending");
    setScopeLayout(layout);
    if (!svg || !drawingId || !layout) return;
    const nodes = Array.from(
      document.querySelectorAll(".viewer-canvas [data-handle]"),
    );
    if (nodes.length === 0) {
      setSelectionMechanism("client");
      return;
    }
    // Spread the sample across the document rather than taking the first N:
    // a sheet's own border and title block are drawn first, so the leading
    // entities are exactly the ones that would NOT reveal the mismatch.
    const step = Math.max(1, Math.floor(nodes.length / SAMPLE_SIZE));
    const handles: string[] = [];
    for (let i = 0; i < nodes.length && handles.length < SAMPLE_SIZE; i += step) {
      const handle = nodes[i].getAttribute("data-handle");
      if (handle) handles.push(handle);
    }
    if (handles.length === 0) {
      setSelectionMechanism("client");
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    describeSelection(drawingId, handles, controller.signal)
      .then((result) => {
        if (cancelled) return;
        const foreign = (result.by_layout ?? []).filter((l) => l.name !== layout);
        setSelectionMechanism(foreign.length === 0 ? "server" : "client");
        // Whichever layout most of the visible entities really belong to.
        const dominant = [...(result.by_layout ?? [])].sort(
          (a, b) => b.count - a.count,
        )[0];
        setScopeLayout(dominant?.name ?? layout);
      })
      .catch(() => {
        // The check itself failing must not cost the feature; the client
        // mechanism is safe everywhere because it tests what is drawn.
        if (!cancelled) setSelectionMechanism("client");
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [svg, drawingId, layout]);

  // The scope the MAP views ask for.
  //
  //  A map is a view of the world, and the world is model space. The layout
  //  picker chooses a SHEET, which is a different question, and letting it
  //  drive the map is what put a reader inside one block definition of a
  //  990,790 entity drawing looking at a hedge detail: Sedra's model space is
  //  too large to render, so the picker had only block layouts to offer and
  //  the map followed it there.
  //
  //  cad-api already treats model scope as "on the Model layout, OR measured
  //  into the world" (geo_h3.scope_match), so asking for model space is what
  //  reaches block-resident geometry that has been placed. Janadriyah is
  //  unchanged: its scopeLayout already resolves to Model.
  //  The model-space layout, read off the field the API states it in.
  //
  //  A previous version wrote `!l.is_block && l.entity_count > 0` here under a
  //  variable NAMED modelspace, which actually means "the first non-block
  //  layout holding anything". On Janadriyah the layout list puts the paper
  //  sheet DMP Layout1 before Model, so the map silently rescoped itself to a
  //  sheet and drew 58 paper-space outlines with no land use, while Sedra
  //  survived only because Model happens to come first in its list. The API
  //  already answers the real question with `is_modelspace`
  //  (extract._layout_summaries); restating it as a guess about position in a
  //  list is the NO SECOND OPINIONS violation this campaign keeps relearning.
  const modelSpaceLayout = useMemo(
    () =>
      (detail?.layouts ?? []).find(
        (l) => l.is_modelspace && l.entity_count > 0,
      ) ?? null,
    [detail],
  );

  const mapScope = modelSpaceLayout?.name ?? (scopeLayout || layout);

  //  Whether that scope was settled from the layout table alone. Derived from
  //  the SAME lookup as the scope itself, so the two cannot disagree: when the
  //  drawing has no populated model space, the map must wait for the
  //  `selectionMechanism` probe exactly as it always did. The probe exists to
  //  work out which layout a SHEET's visible objects really live in; when the
  //  drawing has a model space we already know, and waiting is not free -- on
  //  Sedra the probe runs against a block layout of a 990,790 entity drawing
  //  and the map sat empty behind it.
  const mapScopeIsCertain = modelSpaceLayout !== null;

  // --- geometry for the deck views -----------------------------------------
  //
  //  Scoped to `scopeLayout`, not `layout`, and that distinction is the whole
  //  reason the map views work on Janadriyah at all. The app opens on a paper
  //  sheet; the objects visible on that sheet are STORED under Model, drawn
  //  through a viewport. Asking for the sheet's geometry returns the sheet's
  //  own border and title block — a few dozen objects — and the map looks
  //  empty while the screen is full.
  //
  //  Hence the wait on `selectionMechanism`. That flag is "pending" exactly
  //  while the page is asking which layout the visible objects really live
  //  in, and `scopeLayout` is not final until it resolves. Firing before then
  //  costs a wasted 2.5 MB request against the wrong layout AND, worse,
  //  produces a real empty answer that the fallback below would believe.
  //  One question, already asked, answering two things.
  //
  //  One request per layout rather than per viewport move: 3,169 outlines
  //  over 20,212 vertices is ~1.8 s and 2.5 MB, and the alternative is a map
  //  that redraws a partial city on every pan.
  /** A drawing change must not leave ANY of the previous drawing's view state
   *  behind. Measured, because this bit immediately: Sedra is truncated, so
   *  it set `refineWithViewport` and reported its camera over north-west
   *  Riyadh at 46.71-46.78 E. Switching to Janadriyah -- whose geometry is at
   *  46.90-46.96 E and which fits in one request and needs no viewport at all
   *  -- carried both across, so it fetched a box of empty desert, drew
   *  nothing, and showed an empty legend. A correct drawing, correctly
   *  served, made to look broken by one flag and one stale rectangle.
   *
   *  `settledBounds` is only ever written from the map's own camera, so
   *  clearing it here means the next value is the new drawing's. */
  useEffect(() => {
    boundsOwner.current = `${drawingId}:${mapScope}`;
    // A DRAWING change clears the map. Keeping the previous drawing's
    // geometry on screen while the new one loads would paint Sedra under
    // Janadriyah's name, which is worse than an honest empty state -- and it
    // is the one case where an empty state IS honest. A PAN is the opposite
    // and is handled at the merge below: same drawing, so it keeps what it
    // has.
    setGeometry(null);
    setScopeLayers(null);
    setRefineWithViewport(null);
    setSettledBounds(null);
    // Back to the first-paint guess: the previous drawing's camera is not
    // this one's, and a stale bucket would fetch the wrong detail.
    setDetailZoom(FIRST_PAINT_ZOOM);
    // A pending settle from the previous drawing would otherwise land after
    // the reset and re-poison the request.
    if (boundsSettle.current) clearTimeout(boundsSettle.current);
  }, [drawingId, mapScope]);

  /** The layer selection, in the server's `?layers=` syntax, or null.
   *
   *  Whichever of the two forms is SHORTER. Isolating three layers is a short
   *  include list; hiding one layer out of 215 is a short exclude list and a
   *  ruinous include list -- Sedra's names url-encode to about 12 KB, past
   *  what a request line will carry. The server reads both from the one
   *  parameter.
   *
   *  Derived from `scopeLayers` -- the store's list -- and NOT from
   *  `panelLayers`. Not a style choice: `panelLayers` is built from the
   *  geometry in hand, so keying the geometry request on it would make the
   *  request depend on its own answer. The universe of selectable layers is
   *  the scope, which is the honest source anyway.
   *
   *  This is what makes unchecking a layer mean something. `EXT-BASE` is
   *  440,340 of the scope's 610,944 objects and sits first in `_id` order, so
   *  every capped fetch came back 100% EXT-BASE -- measured, 4,000 of 4,000 --
   *  and unchecking it left the map empty rather than revealing what was
   *  underneath. Excluded at the query, the same budget buys the layers the
   *  user actually wants to see.
   */
  const layerParam = useMemo(() => {
    if (viewMode === "svg") return null;
    const all = (scopeLayers?.layers ?? []).map((l) => l.name);
    const hidden = all.filter((n) => hiddenLayers.has(n));
    const visible = all.filter((n) => !hiddenLayers.has(n));
    // Nothing hidden is not a filter; everything hidden is a client-side
    // question about an answer already in hand, not a different request.
    if (all.length === 0 || hidden.length === 0 || visible.length === 0) return null;
    // A layer whose name STARTS with "!" cannot be excluded unambiguously in
    // this syntax. Rather than invent an escape nobody could read, such a
    // drawing keeps hiding client-side, which is what it did before.
    if (hidden.some((n) => n.startsWith("!")) || visible.some((n) => n.startsWith("!"))) {
      return null;
    }
    const exclude = hidden.map((n) => `!${n}`).join(",");
    const include = visible.join(",");
    const chosen = exclude.length <= include.length ? exclude : include;
    // Past this the request line itself is the risk, and a filter that makes
    // the request fail is worse than one applied in the browser.
    return encodeURIComponent(chosen).length > MAX_LAYER_PARAM_CHARS
      ? null
      : chosen;
  }, [viewMode, scopeLayers, hiddenLayers]);

  useEffect(() => {
    const scope = mapScope;
    if (!drawingId || !scope) return;
    if (!mapScopeIsCertain && selectionMechanism === "pending") return;

    // The viewport is applied only to a layout that does NOT fit in one
    // request. Where it fits, spending the budget "on what is on screen" and
    // "on everything" are the same thing, and Janadriyah's 2,885 outlines
    // must not start depending on where the camera happens to be.
    // Both must be stamped with the scope being drawn. Ordering alone is not
    // enough: a drawing change re-renders with the new id and the previous
    // drawing's camera still in state, and the reset effect's own update does
    // not reach the closure this effect already captured. Sedra's box over
    // north-west Riyadh went out under Janadriyah's id, which fetched empty
    // desert. Identity cannot lie in that direction.
    const owner = `${drawingId}:${scope}`;
    const box =
      settledBounds && settledBounds.owner === owner ? settledBounds.box : null;
    const viewportKey =
      refineWithViewport === owner && box
        ? box.map((n) => n.toFixed(4)).join(",")
        : "";
    // Everything about the QUESTION except where the camera is pointing. Two
    // requests sharing this prefix are asking about the same drawing, scope,
    // kinds, layers and detail -- so features already fetched under one are
    // still valid under the other, and a pan must not throw them away.
    const subject = `${drawingId}:${scope}:${kinds}:${layerParam ?? ""}:${detailZoom}`;
    const key = `${subject}:${viewportKey}`;
    // Asked for already: the deps changed but the question did not.
    if (geometryRequested.current === key) return;
    geometryRequested.current = key;

    geometryAbort.current?.abort();
    const controller = new AbortController();
    geometryAbort.current = controller;
    setGeometryLoading(true);
    setGeometryError(null);

    // Whether THIS request is still the one the page wants.
    //
    //  Keyed on the request's identity, not on the abort signal. The previous
    //  attempt tested `controller.signal.aborted` before storing a page, and
    //  an effect re-run for the SAME question aborts the controller while the
    //  guard above prevents a refetch -- so pages that had already arrived,
    //  with a 200, were thrown away. Janadriyah settled at 1,077 outlines
    //  instead of 2,885 and looked like a data fault. Identity cannot lie in
    //  that direction: if the key still matches, this answer is still wanted.
    const stillWanted = () => geometryRequested.current === key;

    /** Pages after the first, and then the kinds the first pass left out.
     *
     *  Sequential on purpose: the point is that the map is usable early, not
     *  that the whole layout arrives soonest, and twenty concurrent
     *  multi-megabyte requests are slower at the thing that matters. Every
     *  step re-checks identity, so switching drawing stops accumulation
     *  rather than pouring one drawing's geometry into another's view.
     */
    const accumulate = async (first: GeometryResponse) => {
      let held = first.features.length;
      const passes: Array<{ kinds: string; offset: number | null }> = [
        { kinds: ringsFirst(kinds), offset: first.next_offset },
        ...(remainingKinds(kinds) ? [{ kinds: remainingKinds(kinds)!, offset: 0 }] : []),
      ];

      for (const pass of passes) {
        let next = pass.offset;
        while (next !== null && held < MAX_ACCUMULATED_FEATURES && stillWanted()) {
          let more: GeometryResponse;
          try {
            more = await fetchGeometry(
              drawingId,
              {
                layout: scope,
                frame: "auto",
                kinds: pass.kinds,
                limit: FIRST_PAINT_FEATURES,
                offset: next,
                ...(viewportKey ? { bbox: box! } : {}),
                ...(layerParam ? { layers: layerParam } : {}),
                zoom: detailZoom,
                ...(viewportKey && heldCells.current.size
                  ? { have: [...heldCells.current] }
                  : {}),
              },
              controller.signal,
            );
          } catch {
            // A page that fails leaves what is already drawn. Clearing the
            // map because page nine did not arrive throws away the part that
            // is correct.
            return;
          }
          if (!stillWanted()) return;
          held += more.features.length;
          next = more.next_offset;
          setGeometry((current) =>
            current === null
              ? more
              : { ...current, features: [...current.features, ...more.features] },
          );
        }
      }
    };

    fetchGeometry(
      drawingId,
      // `frame: auto` lets the server decide: lng/lat where the drawing has
      // a CRS, its own coordinates where it does not. Asking for lng/lat
      // outright would be refused for eighteen of the nineteen drawings, and
      // a refusal is not what the user wants to see — they want the drawing.
      //
      // A PAGE, and rings before anything else. Storage order put closed
      // shapes at about 3% of the first page, so a 2,000-row request drew an
      // outline count of zero on a drawing holding 22,808 rings. Rings are
      // what carry the estate's shape, its area and its land use, so they are
      // what the first paint should be made of; lines and arcs follow.
      {
        layout: scope,
        frame: "auto",
        kinds: ringsFirst(kinds),
        limit: FIRST_PAINT_FEATURES,
        ...(viewportKey ? { bbox: box! } : {}),
        ...(layerParam ? { layers: layerParam } : {}),
        zoom: detailZoom,
        ...(viewportKey && heldCells.current.size
          ? { have: [...heldCells.current] }
          : {}),
      },
      controller.signal,
    )
      .then((data) => {
        if (!stillWanted()) return;
        // A PAN keeps what it already has. Only the viewport changed, so
        // every feature in hand is still an answer to the same question --
        // and re-fetching them costs the request's fixed price again, which
        // is 1.9 s on Sedra even after the caching above. Merged by handle so
        // a feature that appears in both pages is not drawn twice.
        //
        // A change of drawing, scope, kinds, layers or detail is a DIFFERENT
        // question and replaces wholesale, which is what `subject` decides.
        const panning = geometrySubject.current === subject;
        geometrySubject.current = subject;
        if (!panning) heldCells.current = new Set();
        for (const cell of data.viewport?.cells_served ?? []) {
          heldCells.current.add(cell);
        }
        setGeometry((current) => {
          if (!panning || current === null) return data;
          const byHandle = new Map(
            current.features.map((f) => [f.handle, f] as const),
          );
          for (const f of data.features) byHandle.set(f.handle, f);
          return { ...data, features: [...byHandle.values()] };
        });
        setGeometryError(null);
        // Said by the server, not guessed from a row count.
        if (data.truncated) setRefineWithViewport(owner);
        // Layers the FILE switches off start hidden in the map views too.
        //
        // They were not, and the bug is worth recording because it is
        // invisible from either view alone. `hiddenLayers` is seeded from the
        // rendered SVG's `data-off-layers`, which is correct — but only for
        // the layout the SVG rendered. The app opens on a paper sheet, whose
        // viewport already excludes those layers, so the sheet reports NONE
        // and the set came back empty. The map, meanwhile, draws model space,
        // where they are all present. The result was eleven layers AutoCAD
        // keeps switched off being drawn on the map and nowhere else — on
        // Janadriyah, the road centrelines and every NBHD boundary, 836
        // objects scattered up to 12 km from the estate. It read as a
        // projection fault and was a visibility one.
        //
        // Merged rather than replaced: by the time this resolves the user may
        // already have toggled something, and their choice outranks a default.
        void accumulate(data);
        if (data.layers_off.length > 0) {
          const seedKey = `${drawingId}:${scope}`;
          if (offLayersSeeded.current !== seedKey) {
            offLayersSeeded.current = seedKey;
            setHiddenLayers((current) => {
              const next = new Set(current);
              for (const layer of data.layers_off) next.add(layer.name);
              return next;
            });
          }
        }
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setGeometry(null);
        setGeometryError(describe(err));
      })
      .finally(() => {
        if (controller.signal.aborted) return;
        setGeometryLoading(false);
        // Written last and together with the result, so that the fallback
        // effect can never read a settled key against a stale payload.
        setGeometryFor(key);
      });

    //  Deliberately no `return () => controller.abort()`.
    //
    //  React runs the previous cleanup BEFORE re-running an effect, so with
    //  the guard above a benign re-run — same drawing, same layout, same
    //  kinds — would abort the good request and then return early without
    //  starting another, leaving the map empty forever. The previous request
    //  is instead aborted where a NEW one begins, a few lines up, which is
    //  the only moment it is genuinely superseded.
  }, [
    drawingId,
    mapScope,
    mapScopeIsCertain,
    selectionMechanism,
    kinds,
    // All three only ever change the key for a layout the SERVER called
    // truncated, or when the user has actually unchecked something. For a
    // layout that arrives whole with every layer on, the key is what it
    // always was.
    refineWithViewport,
    settledBounds,
    layerParam,
    detailZoom,
  ]);

  /** The scope's layer list, fetched once per drawing and scope.
   *
   *  Deliberately NOT derived from `geometry.features`. The panel used to
   *  build its membership from whatever pages had arrived, so rows appeared
   *  as the map paged and a layer's total was however many had turned up --
   *  which reads as the drawing changing while you look at it. This asks the
   *  store the membership question directly, and the answer is the same on
   *  the first page as on the last.
   *
   *  A failure leaves `scopeLayers` null and the panel falls back to what it
   *  can see, saying so. It never blanks the list.
   */
  useEffect(() => {
    const scope = mapScope;
    if (viewMode === "svg" || !drawingId || !scope) return;
    if (!mapScopeIsCertain && selectionMechanism === "pending") return;
    const controller = new AbortController();
    let live = true;
    fetchScopeLayers(drawingId, scope, kinds, controller.signal)
      .then((data) => {
        if (live) setScopeLayers(data);
      })
      .catch(() => {
        /* the panel says what it is showing instead; see `panelLayers` */
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, [drawingId, mapScope, mapScopeIsCertain, selectionMechanism, kinds, viewMode]);



  /** The camera moved. Choose the rung, once it settles.
   *
   *  `setCellRes` with the value it already holds is a no-op React bails out
   *  of, so a pan that never leaves its band costs one comparison and no
   *  render — which is what makes it safe for the viewer to report every
   *  frame rather than trying to be clever about which ones matter. */
  const onMapZoom = useCallback((zoom: number) => {
    const next = clampRes(resForZoom(zoom), cellResBounds.current);
    if (cellResSettle.current) clearTimeout(cellResSettle.current);
    cellResSettle.current = setTimeout(() => {
      cellResSettle.current = null;
      setCellRes(next);
      setDetailZoom(detailBucket(zoom));
    }, CELL_RES_SETTLE_MS);
  }, []);

  useEffect(
    () => () => {
      if (cellResSettle.current) clearTimeout(cellResSettle.current);
    },
    [],
  );

  /** The H3 cells, on the same scope question as the geometry.
   *
   *  A separate request rather than a field on the geometry payload, because
   *  the two answer different questions and fail independently: a drawing can
   *  have every outline and no cells (nothing indexed it yet) or, in
   *  principle, the reverse. Folding them together would make one failure
   *  look like the other.
   *
   *  Asked at the resolution the camera deserves, not a fixed one. Res 9 is
   *  23 cells on the reference drawing — a silhouette of the site that reads
   *  when the whole site is in frame, and far too coarse once it is not,
   *  which is the whole of Harsh's point about a single hierarchy. Coarser
   *  levels are derived by the server from the stored res-13 cells, so every
   *  rung is the same evidence aggregated, never a different measurement:
   *  that is why the land-use totals hold all the way up and down.
   *
   *  Re-asked on a band change and on nothing else. Panning does not refetch,
   *  parcels are not refetched separately, and a band already visited comes
   *  back from the conditional-request cache in `mapCells`.
   *
   *  `outline` is asked for here and not only when the overlay is switched
   *  on: it is what the camera frames the site with, and the camera runs
   *  whether or not anyone wants hexagons.
   */
  useEffect(() => {
    const scope = mapScope;
    if (!drawingId || !scope) return;
    if (!mapScopeIsCertain && selectionMechanism === "pending") return;

    const key = `${drawingId}:${scope}:${cellRes}`;
    if (cellsRequested.current === key) return;
    cellsRequested.current = key;

    cellsAbort.current?.abort();
    const controller = new AbortController();
    cellsAbort.current = controller;
    setCellsLoading(true);

    fetchGeo(
      drawingId,
      { layout: scope, res: cellRes, parcels: true, outline: true },
      controller.signal,
    )
      .then((data) => {
        if (controller.signal.aborted) return;
        setCells(data);
        // What this drawing can actually be asked for, taken from the answer
        // it just gave rather than assumed from the reference site.
        cellResBounds.current = boundsFor(data);
        // The API answers "cannot be mapped" with a 200 and a reason rather
        // than an error, so the reason is read off the payload, not a catch.
        setCellsReason(
          data.cells?.length
            ? null
            : [data.reason, data.how_to_fix].filter(Boolean).join(" — ") ||
                "This drawing has no H3 cells.",
        );
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        // Not surfaced as a page error. The drawing and its linework are
        // already on screen; losing the overlay must not take the viewer with
        // it, so this degrades to a disabled control carrying the reason.
        setCells(null);
        setCellsReason(describe(err));
      })
      .finally(() => {
        if (controller.signal.aborted) return;
        setCellsLoading(false);
      });

    // No cleanup abort, for the reason spelled out on the geometry effect
    // above: React runs it before a benign re-run, which would cancel the
    // request the guard then declines to replace.
  }, [drawingId, mapScope, mapScopeIsCertain, selectionMechanism, cellRes]);

  /** Fall back to the sheet when the deck views have nothing to draw.
   *
   *  Only on a SETTLED answer for the layout currently on screen — see
   *  `geometryFor`. Once per layout, and never sticky: the switcher still
   *  offers the deck views, and choosing one shows the explanation of why it
   *  is empty rather than bouncing the user back. An automatic switch that
   *  could not be overridden would hide the coverage gap instead of
   *  reporting it.
   */
  useEffect(() => {
    const key = `${drawingId}:${mapScope}:${kinds}`;
    if (geometryLoading || geometryFor !== key) return;
    if (autoFellBack.current === key) return;
    autoFellBack.current = key;
    const empty = geometryError !== null || (geometry?.features.length ?? 0) === 0;
    if (empty) setViewMode("svg");
  }, [
    geometry,
    geometryError,
    geometryLoading,
    geometryFor,
    drawingId,
    layout,
    scopeLayout,
    kinds,
  ]);

  /** Append a drawn region as a new part of the selection.
   *
   *  When `replacePart` names an index, the answer belongs to a part that
   *  already exists — a window/crossing flip, or a circle reshaped by its
   *  grips. It replaces that part in place; appending it would silently
   *  double the selection.
   */
  const addGroup = useCallback(
    (
      region: Region,
      result: SelectionResult,
      matches?: DomSelectionResult["matches"],
    ) => {
      setGroups((current) => {
        const target = replacePart.current;
        replacePart.current = null;
        if (target !== null && target >= 0 && target < current.length) {
          // Re-asking a part that exists: it keeps its id and its label, so
          // "Region 2" stays Region 2 after being resized and the panel does
          // not renumber under the user's hand.
          const kept = current[target];
          const part: SelectionGroup = { ...kept, region, result };
          if (matches) groupMatches.current.set(part.id, matches);
          return current.map((g, i) => (i === target ? part : g));
        }
        const part: SelectionGroup = {
          id: nextGroupId(),
          label: nextRegionLabel(current),
          region,
          result,
        };
        if (matches) groupMatches.current.set(part.id, matches);
        return [...current, part];
      });
    },
    [],
  );

  const removeGroup = useCallback((id: string) => {
    groupMatches.current.delete(id);
    setGroups((current) => current.filter((g) => g.id !== id));
  }, []);

  // --- region selection: query, clear, re-run ------------------------------
  const runRegion = useCallback(
    (next: Region) => {
      regionAbort.current?.abort();
      const controller = new AbortController();
      regionAbort.current = controller;
      setRegionLoading(true);
      setRegionError(null);
      setTab("selection");
      setPanelOpen(true);

      // Two mechanisms, one dispatch. `server` compares stored bounding boxes
      // and only holds where the layout's entities live in its own coordinate
      // space; everywhere else — and while calibration is still in flight —
      // the region is answered from the rendered geometry, which is correct
      // by construction on any layout because it tests what is on screen.
      if (selectionMechanism !== "server") {
        try {
          if (!domIndex.current) {
            const svgEl = document.querySelector<SVGSVGElement>(
              ".viewer-canvas svg",
            );
            if (!svgEl) throw new Error("no drawing is mounted");
            domIndex.current = buildDomIndex(svgEl);
          }
          const data = selectFromDom(
            domIndex.current,
            next.pointsView,
            next.kind,
            next.mode,
          );
          addGroup(next, {
            drawing_id: drawingId,
            layout: next.layout,
            kind: next.kind,
            mode: next.mode,
            region: next.pointsView,
            region_bbox: [],
            total: data.total,
            handles: data.handles,
            handles_truncated: data.handles_truncated,
            extents: null,
            basis: data.basis,
            by_layer: data.by_layer,
            by_type: data.by_type,
            groups: data.groups,
            groups_truncated: data.groups_truncated,
          }, data.matches);
        } catch (err: unknown) {
          setRegionError(describe(err));
        } finally {
          setRegionLoading(false);
        }
        return;
      }

      selectRegion(drawingId, next, controller.signal)
        .then((data) => {
          if (controller.signal.aborted) return;
          addGroup(next, data);
        })
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          setRegionError(describe(err));
        })
        .finally(() => {
          if (!controller.signal.aborted) setRegionLoading(false);
        });
    },
    [drawingId, selectionMechanism, addGroup],
  );

  const onRegionDrawn = useCallback(
    (
      points: [number, number][],
      pointsView: [number, number][],
      kind: "rect" | "polygon" | "circle",
      mode: "window" | "crossing",
    ) => {
      runRegion({ kind, mode, points, pointsView, layout });
    },
    [runRegion, layout],
  );

  /** A committed circle was reshaped by its grips.
   *
   *  `index` counts drawn regions, which is what the viewer sees; the parts
   *  list may also hold a clicked object, so the two are mapped rather than
   *  assumed equal. Getting that wrong would resize the wrong part, and it
   *  would look like the drawing had jumped.
   */
  const onRegionEdited = useCallback(
    (
      index: number,
      points: [number, number][],
      pointsView: [number, number][],
      mode: "window" | "crossing",
    ) => {
      const partIndexes = groups
        .map((g, i) => (g.region !== null ? i : -1))
        .filter((i) => i >= 0);
      const target = partIndexes[index];
      if (target === undefined) return;
      replacePart.current = target;
      runRegion({
        kind: "circle",
        mode,
        points,
        pointsView,
        layout: groups[target].region?.layout ?? layout,
      });
    },
    [groups, runRegion, layout],
  );

  const clearRegion = useCallback(() => {
    regionAbort.current?.abort();
    groupMatches.current.clear();
    setGroups([]);
    setSelected(null);
    setRegionError(null);
    setRegionLoading(false);
    setRegionTool("none");
  }, []);

  /** Flip window <-> crossing on the region already drawn. Re-queried rather
   *  than filtered client-side: the two modes ask the database a different
   *  question, and the handle list is capped. */
  const setRegionMode = useCallback(
    (mode: "window" | "crossing") => {
      if (!region) return;
      // The last region drawn is being re-asked, not joined by another. Its
      // index among the parts is what addGroup needs, and `regions` only
      // holds the drawn ones, so the click part (if any) must not shift it.
      const target = groups.reduce(
        (found, g, i) => (g.region !== null ? i : found),
        -1,
      );
      if (target < 0) return;
      replacePart.current = target;
      runRegion({ ...region, mode });
    },
    [region, runRegion, groups],
  );

  // Switching drawing or layout invalidates the region: its coordinates mean
  // something else on another layout, and silently re-running it there would
  // put a box over unrelated geometry.
  useEffect(() => {
    clearRegion();
  }, [drawingId, layout, clearRegion]);

  // The chat keeps its conversation across a drawing switch by design, but the
  // handles attached to it belong to the drawing they were selected in. Left
  // alone, the next question said "the user selected 17,434 objects" while
  // naming a different drawing_id — contradictory context handed to a model
  // that has no way to notice.
  useEffect(() => {
    setAgentSelection(null);
  }, [drawingId]);

  // Clicking empty paper deselects; if the current selection came from a
  // click, it goes with it. A drawn region is not dropped by a stray click —
  // it took deliberate work to make.
  useEffect(() => {
    if (selected === null) {
      setGroups((current) =>
        current.some((g) => g.region === null)
          ? current.filter((g) => g.region !== null)
          : current,
      );
    }
  }, [selected]);

  const regionHandles = useMemo(
    () => (regionResult ? new Set(regionResult.handles) : null),
    [regionResult],
  );

  // The concern that drove this, verbatim from the field: when a region is
  // selected, the agent must answer about THAT region; when none is, it must
  // speak for the whole file. Attachment is therefore not a button any more —
  // it is a consequence of having a selection. The active region's handles
  // follow it into the chat automatically, and clearing the region (Esc, ✕,
  // switching layout or drawing) returns the agent to whole-file scope.
  // "Detach" in the chat remains as an explicit opt-out for one selection.
  useEffect(() => {
    if (!regionResult || regionResult.total === 0) {
      setAgentSelection(null);
      return;
    }
    // The STRUCTURE of the selection travels, never its handles. How many
    // parts, what each one is, what each one holds — a bounded handful of
    // numbers that lets the agent answer "how many in the second region"
    // without being handed the second region. See lib/selectionSet.ts.
    const summary = combined
      ? describeGroups(groups, combined, layout)
      : "";

    // The selection is STORED and only its id travels. Sending the handles
    // meant ~16 KB re-sent with every message and a 2,000-handle ceiling, so
    // anything larger came back answered in part. See DECISIONS-LOG D-046.
    let cancelled = false;
    saveSelection(drawingId, {
      handles: regionResult.handles,
      layout: regionResult.layout,
      mode: regionResult.mode,
      kind: regionResult.kind,
      summary,
      // The exact figures, sent separately from the handles. `handles` may be
      // capped; these are not, so the agent can count a selection larger than
      // it could ever be handed object by object. Only what is genuinely
      // exact is sent -- overlapping parts make a summed breakdown an upper
      // bound, and shipping that as exact would plant the very error this is
      // meant to remove.
      ...(combined ? exactCounts(groups, combined) ?? {} : {}),
    })
      .then((saved) => {
        if (cancelled) return;
        setAgentSelection({
          selectionId: saved.selection_id,
          total: regionResult.total,
          summary,
        });
      })
      .catch(() => {
        // Storing failed: better no attachment than a reference that does not
        // resolve, which would make the agent describe the wrong thing.
        if (!cancelled) setAgentSelection(null);
      });
    return () => {
      cancelled = true;
    };
  }, [regionResult, combined, groups, layout, drawingId]);

  /** The button is now just a doorway: attachment already happened the moment
   *  the selection existed. */
  const askAgentAboutSelection = useCallback(() => {
    setTab("agent");
    setPanelOpen(true);
  }, []);

  /** Rows for one (layer, type) group of the current selection.
   *
   *  Server-mechanism selections are re-evaluated by the API — the returned
   *  handle list is capped, so filtering it client-side would quietly return
   *  a short group and call it complete. Rendered-geometry selections already
   *  hold every match client-side, so their rows come straight from that
   *  store; text is not in the DOM, and a handle chip without text is still a
   *  working chip. */
  /** Rows are always fetched for ONE part of the selection.
   *
   *  A combined selection has no single shape to re-evaluate, so expanding a
   *  layer x type row belongs to the part that reported it. `partId` says
   *  which; without one, the most recent part is meant. */
  const fetchGroupRows = useCallback(
    async (group: { layer: string; type: string }, partId?: string) => {
      const part = partId
        ? groups.find((g) => g.id === partId)
        : groups[groups.length - 1];
      if (!part) return [];
      // A clicked part has no region to re-evaluate; its one handle is
      // already the answer.
      if (part.result.basis === "clicked") {
        return part.result.handles.map((handle) => ({
          handle,
          type: group.type,
          layer: group.layer,
          text: null,
        }));
      }
      if (!part.region) return [];
      if (part.result.basis === "rendered-geometry") {
        return (groupMatches.current.get(part.id) ?? [])
          .filter((m) => m.layer === group.layer && m.type === group.type)
          .slice(0, 200)
          .map((m) => ({
            handle: m.handle,
            type: m.type,
            layer: m.layer,
            text: null,
          }));
      }
      const data = await listRegionGroup(drawingId, part.region, group);
      return data.entities;
    },
    [groups, drawingId],
  );

  const toggleLayer = useCallback((layer: string) => {
    setHiddenLayers((prev) => {
      const next = new Set(prev);
      if (next.has(layer)) next.delete(layer);
      else next.add(layer);
      return next;
    });
  }, []);

  const onCommentAdded = useCallback(
    (handle: string) => {
      setCommentedHandles((prev) => new Set(prev).add(handle));
      // Keep the badge count honest without a full metadata refetch.
      void listComments(drawingId)
        .then(({ comments }) => {
          setCommentedHandles(new Set(comments.map((c) => c.entity_handle)));
          const map: Record<string, { author: string; body: string }[]> = {};
          comments.forEach((c) => {
            (map[c.entity_handle] ??= []).push({
              author: c.author,
              body: c.body,
            });
          });
          setCommentsByHandle(map);
        })
        .catch(() => {
          /* the optimistic set above is already correct enough */
        });
    },
    [drawingId],
  );

  const onSelect = useCallback(
    (selection: Selection | null) => {
      setSelected(selection);
      if (!selection) return;
      setTab("entity");
      setPanelOpen(true);

      // Clicking one object IS a selection of one, and it flows through the
      // same path as a region: the Selection tab shows it and the agent
      // scopes to it. Two mechanisms for "what am I asking about" would be
      // two places for the answer to be wrong. A live region wins, because
      // clicking inside a region is how a person inspects a member of it —
      // not how they abandon it.
      // A click is a part like any other, with one difference that matters
      // in use: clicking is how a person INSPECTS, so consecutive clicks
      // replace each other instead of piling up a part per glance. Regions
      // are never touched — they took deliberate work to draw.
      setGroups((current) => {
        const kept = current.filter((g) => g.region !== null);
        const part: SelectionGroup = {
          id: nextGroupId(),
          label: `Object ${selection.handle}`,
          region: null,
          result: {
            drawing_id: drawingId,
            layout,
            kind: "click",
            mode: "window",
            region: [],
            region_bbox: [],
            total: 1,
            handles: [selection.handle],
            handles_truncated: false,
            extents: null,
            basis: "clicked",
            by_layer: [{ name: selection.layer || "(none)", count: 1 }],
            by_type: [{ name: selection.type || "(unknown)", count: 1 }],
            groups: [
              {
                layer: selection.layer || "(none)",
                type: selection.type || "(unknown)",
                count: 1,
              },
            ],
            groups_truncated: false,
          },
        };
        return [...kept, part];
      });
    },
    [drawingId, layout],
  );

  /** A rectangle dragged on the map, as a part of the selection.
   *
   *  It joins `groups` exactly like a region drawn on the sheet, so the
   *  Selection panel, the combined counts and the agent's scope all work
   *  unchanged — the map view is a free rider on the selection engine that
   *  already exists, which is the whole reason this was a small change.
   *
   *  Two honest differences travel in the payload rather than being smoothed
   *  over. `basis` says `rendered outlines`, because the test ran against the
   *  real rings rather than the bounding boxes `POST /selection` compares —
   *  so when the two disagree, the payload says why. And `region` is null:
   *  there is no client-side inverse projection, so the rectangle cannot be
   *  expressed in drawing coordinates and is not claimed to be.
   */
  const onMapRegion = useCallback(
    (hit: RegionHit, mode: RegionMode) => {
      setGroups((current) => [
        ...current,
        {
          id: nextGroupId(),
          // Just "map": the panel already renders `window rect` /
          // `crossing rect` from the result, and repeating the mode here read
          // as "Region 1 · map window · window rect".
          label: `${nextRegionLabel(current)} · map`,
          region: null,
          result: {
            drawing_id: drawingId,
            layout: scopeLayout || layout,
            kind: "rect",
            mode,
            region: [],
            region_bbox: [],
            total: hit.handles.length,
            handles: hit.handles,
            handles_truncated: false,
            extents: null,
            // Hyphenated to match `rendered-geometry`: the Selection panel
            // keys its explanation off this string, and a basis it does not
            // recognise would be described as bounding-box matching, which
            // is exactly what this is not.
            basis: "rendered-outlines",
            by_layer: hit.byLayer,
            by_type: hit.byType,
            groups: hit.groups,
            groups_truncated: false,
          },
        },
      ]);
      setTab("selection");
      setPanelOpen(true);
    },
    [drawingId, layout, scopeLayout],
  );

  /** True if a handle exists in the currently rendered layout. Used to stop
   *  the chat panel offering a clickable chip that would do nothing. */
  const isHandleInDrawing = useCallback(
    (handle: string) =>
      Boolean(
        document.querySelector(
          `.viewer-canvas [data-handle="${CSS.escape(handle)}"]`,
        ),
      ),
    [],
  );

  const selectByHandle = useCallback(
    (handle: string) => {
      const group = document.querySelector(
        `.viewer-canvas [data-handle="${CSS.escape(handle)}"]`,
      );
      if (!group) return;
      onSelect({
        handle,
        layer: group.closest("[data-layer]")?.getAttribute("data-layer") ?? "",
        type: group.getAttribute("data-type") ?? "",
      });
    },
    [onSelect],
  );

  const clearAnswer = useCallback(() => {
    answerAbort.current?.abort();
    setAnswerPlan(null);
    setAnswerPoints(null);
    setAnswerPointsNote(null);
  }, []);

  /** Turn a finished answer into marks on the drawing.
   *
   *  The whole point of the feature, and the whole difficulty, is in the
   *  validation. The agent writes handles into prose, and prose is full of
   *  tokens that look exactly like handles — years, counts, layer fragments.
   *  Three questions have to be answered about each one and only two of them
   *  can be answered here:
   *
   *    is it an entity in this drawing at all?  → the API knows
   *    is it drawn in the layout on screen?     → the document knows
   *    is it what the sentence was about?       → nobody knows, so nothing
   *                                               pretends to
   *
   *  Separating the first two is what lets the badge say "2 more not drawn in
   *  this view" without also claiming that the word "2024" is a missing
   *  object. Collapsing them would make every answer look like it referred to
   *  objects that had gone missing.
   */
  const onAgentAnswer = useCallback(
    async (
      text: string,
      answerId: string,
      readHandles: string[] = [],
      readGroups: { label: string; handles: string[] }[] = [],
    ) => {
      // Two sources, and the union of them, in this order.
      //
      // What the answer WROTE comes first because it is the most certain: a
      // handle in the sentence is an object the sentence is about. What the
      // agent READ comes second and is the larger set, because prose has a
      // readability ceiling the drawing does not share -- asked to show the
      // VL2 plots, the agent named twenty of a hundred and thirty-three, and
      // marking twenty of them told the user something untrue about the
      // drawing.
      //
      // Both still pass through the same validation below. Being read by a
      // tool is not by itself proof that a token is an entity on this layout,
      // and the badge still separates "marked" from "not drawn in this view".
      const written = collectHandleCandidates(text);
      const seen = new Set(written);
      const candidates = [...written, ...readHandles.filter((h) => !seen.has(h))];
      if (candidates.length === 0 || !drawingId) {
        clearAnswer();
        return;
      }
      answerAbort.current?.abort();
      const controller = new AbortController();
      answerAbort.current = controller;
      setAnswerPoints(null);
      setAnswerPointsNote(null);

      // Both spellings, because the model writes handles either way and the
      // candidates were normalised upward. The document is the authority on
      // which spelling it used; guessing wrong here would silently mark
      // nothing and look exactly like an answer that named nothing.
      const drawn = (handle: string) =>
        isHandleInDrawing(handle) || isHandleInDrawing(handle.toLowerCase());

      let plan: AnswerPlan;
      try {
        // One call for every candidate. `missing_handles` is the answer to
        // "which of these are not entities", which is the only way to tell a
        // handle on another layout from a number that happens to be hex.
        const described = await describeSelection(
          drawingId,
          candidates,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        const notEntities = new Set(described.missing_handles ?? []);
        plan = planAnswerHighlight({
          answerId,
          candidates,
          isEntity: (handle) => !notEntities.has(handle),
          isDrawn: drawn,
        });
      } catch {
        if (controller.signal.aborted) return;
        // Without the API the two questions collapse into one, and the honest
        // move is to answer the one that is still answerable. Everything
        // drawn here is certainly an entity and gets marked; everything else
        // is dropped in silence rather than counted, because "3 objects are
        // elsewhere" and "the word 2024 appeared" are indistinguishable from
        // here and only one of them is worth telling someone.
        plan = planAnswerHighlight({
          answerId,
          candidates,
          isEntity: drawn,
          isDrawn: drawn,
        });
        setAnswerPointsNote(
          "objects on other layouts could not be counted — cad-api did not answer",
        );
      }

      // Which kind each marked object is, so the viewer can colour by kind.
      //
      // Asked where the schools and the mosques are, painting both the same
      // colour answers half the question: the reader sees that fifteen things
      // matched and not which is which. The label is the layer the handle was
      // read for -- the viewer never interprets a name, it only needs two
      // layers to come out as two colours.
      const groups = new Map<string, string>();
      for (const group of readGroups) {
        for (const handle of group.handles) {
          if (group.label) groups.set(handle, group.label);
        }
      }
      setAnswerGroups(groups);
      setAnswerPlan(plan);
      if (!plan.highlight || plan.markable.length === 0) return;

      // Centroids, for the objects small enough to need a marker rather than
      // an outline. Best-effort by construction: the route may not be merged
      // yet, and the marks are useful without it. See lib/parcelContext.ts.
      //
      // Guarded because this runs detached from any caller that could catch
      // it — the chat panel hands the answer over and moves on. An
      // unhandled rejection here would take out the marks that had already
      // been placed, to improve marks that are a fallback anyway.
      try {
        const points = await fetchMarkPoints(
          drawingId,
          plan.markable,
          scopeLayout || layout,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setAnswerPoints(points.data);
        if (!points.data) {
          setAnswerPointsNote((current) => current ?? "approximate centres");
        }
      } catch {
        if (controller.signal.aborted) return;
        setAnswerPointsNote((current) => current ?? "approximate centres");
      }
    },
    [drawingId, layout, scopeLayout, isHandleInDrawing, clearAnswer],
  );

  // A drawing or layout switch invalidates the marks: the handles were
  // validated against a document that is no longer on screen, and re-marking
  // them elsewhere would put the previous answer over unrelated geometry.
  useEffect(() => {
    clearAnswer();
  }, [drawingId, layout, clearAnswer]);

  // --- the parcel under a selected label -----------------------------------
  // Runs only for a label, because only a label has a parcel to be inside.
  // Fetched here rather than in the panel so that the panel and the agent are
  // given the same answer; two lookups could disagree, and the disagreement
  // would surface as a model contradicting the screen.
  useEffect(() => {
    setParcel(null);
    if (!selected || !drawingId || !isLabelType(selected.type)) {
      // Cleared on the way out, not only on the way in. Left set by a
      // previous label, the flag would leave "Looking for the parcel…"
      // printed under a polyline that has no parcel to look for, for as long
      // as it stayed selected.
      setParcelLoading(false);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    setParcelLoading(true);
    fetchParcelContext(
      drawingId,
      selected.handle,
      scopeLayout || layout,
      controller.signal,
    )
      .then((result) => !cancelled && setParcel(result))
      .catch(() => {
        // A thrown error here is not the "route not built" case, which comes
        // back as data. It is a real fault, and the panel showing nothing is
        // better than the panel showing a parcel it did not read.
        if (!cancelled) setParcel(null);
      })
      .finally(() => !cancelled && setParcelLoading(false));
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [selected, drawingId, layout, scopeLayout]);

  /** The one line about the containing parcel that travels to the agent.
   *
   *  This is the fix for 23 August, 23:28: plot number `2043` was clicked,
   *  "what is the area" was asked, and the answer described the TEXT entity.
   *  Correct, and not what was asked. The model cannot know that a label
   *  stands for the thing underneath it unless something tells it, so
   *  something tells it. */
  const selectionNote = useMemo(
    () => (parcel?.data ? describeParcelContext(parcel.data) : null),
    [parcel],
  );

  const renderMeta = useMemo(
    () => detail?.renders?.find((r) => r.layout === layout),
    [detail, layout],
  );

  /** Everything this file carries that the picture cannot show.
   *
   *  Gathered in one place so the notice above the drawing can be one line
   *  that opens, rather than three paragraphs that never close. Each part is
   *  independent — a drawing can have any of them, or none, and 13 of the 18
   *  ingested files have none.
   */
  const fileNotes = useMemo(() => {
    const undrawnRows = (renderMeta?.unrenderable ?? []).filter(
      (u) => u.type !== "UNEXPECTED",
    );
    const unexpected = renderMeta?.unrenderable?.find(
      (u) => u.type === "UNEXPECTED",
    );
    const payloads = embedded?.payloads ?? [];
    return {
      undrawn: undrawnRows.length ? summariseUnrenderable(undrawnRows) : "",
      unknown: unexpected
        ? unexpected.reason.replace("not drawn, reason unknown: ", "")
        : "",
      documents: payloads.filter((p) => p.file),
      linked: payloads.filter((p) => !p.file).length,
    };
  }, [renderMeta, embedded]);

  const drawableLayoutsRaw = useMemo(() => {
    const rendered = new Set(detail?.renders?.map((r) => r.layout) ?? []);
    return (detail?.layouts ?? [])
      .filter((l) => rendered.has(l.name))
      .sort((a, b) => b.entity_count - a.entity_count);
  }, [detail]);

  /** Block definitions are offered only when there is nothing else to show.
   *
   *  They exist because `title_block-arch`, `title_block-iso` and
   *  `visualization_-_sun_and_sky_demo` keep all their content inside a block
   *  and would otherwise open blank. But a block definition is a *component*,
   *  not a view: AutoCAD does not list them beside the sheets, and Janadriyah
   *  has 77 of them while tablet.dxf has 208 — enough to bury the three real
   *  sheets in a flat list. So they appear only as a fallback. */
  const showBlockLayouts = useMemo(
    () => !drawableLayoutsRaw.some((l) => !l.is_block),
    [drawableLayoutsRaw],
  );

  /** What the Layers panel lists.
   *
   *  The SVG's layers alone are not enough once a map view exists. The panel
   *  is fed from the rendered SVG, and a sheet's SVG contains only what its
   *  viewport draws — so a layer that lives in model space and is switched
   *  off in the file appears in neither, and the checkbox that would turn it
   *  back on does not exist. That is how hiding the road centrelines by
   *  default could otherwise become hiding them permanently.
   *
   *  So while a map view is on screen the list is the union of both, and a
   *  geometry-only layer carries its geometry count. The SVG-only path is
   *  deliberately left exactly as it was.
   */
  const panelLayers = useMemo(() => {
    if (viewMode === "svg" || !geometry) return renderedLayers;

    // How many objects the DRAWING holds per layer. The drawing document
    // already carries this and it is the authoritative number; counting the
    // fetched features instead reported EXT-BASE as 20,000 on a drawing that
    // holds 459,314 of them, because that is how many happened to be in the
    // pages the map had loaded. Two different facts wearing one label.
    const inDrawing = new Map<string, number>();
    for (const l of detail?.layers ?? []) inDrawing.set(l.name, l.entity_count);

    // How many the map is currently holding, which is the other fact.
    const drawn = new Map<string, number>();
    for (const f of geometry.features) {
      drawn.set(f.layer, (drawn.get(f.layer) ?? 0) + 1);
    }

    // MEMBERSHIP comes from the scope, not from the pages in hand.
    //
    // It used to be `renderedLayers` union whatever `drawn` held, and `drawn`
    // is built from `geometry.features` -- so rows appeared as the map paged.
    // A list that grows while you watch it is indistinguishable from the
    // drawing changing, which is the same class of problem as a count that
    // disagrees with the store.
    //
    // `scopeLayers` also carries the right DENOMINATOR. `detail.layers` is
    // drawing-wide, and the map's scope is not the whole drawing: XR_BUILDING
    // FOOTPRINT holds 22,943 in the file and 22,440 in the scope, so the
    // panel was reading "of 22,943" for a total the map could never reach.
    const inScope = new Map<string, number>();
    for (const l of scopeLayers?.layers ?? []) inScope.set(l.name, l.count);

    const names = scopeLayers
      ? scopeLayers.layers.map((l) => l.name)
      : // No scope answer yet or the request failed. Falling back to what is
        // visible is worse than the store and better than an empty panel, and
        // `layersScope` below says which of the two is on screen.
        [...new Set<string>([...renderedLayers.map((l) => l.name), ...drawn.keys()])];

    return names
      .map((name) => {
        const rendered = renderedLayers.find((l) => l.name === name);
        return {
          name,
          count:
            inScope.get(name) ??
            inDrawing.get(name) ??
            rendered?.count ??
            drawn.get(name) ??
            0,
          drawn: drawn.get(name) ?? 0,
        };
      })
      .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
  }, [viewMode, geometry, renderedLayers, detail, scopeLayers]);

  /** What the layer list is a list OF, in words the panel puts on screen.
   *
   *  The fourth appearance of the sheet-versus-map conflation, and the first
   *  in front of a user: the LAYOUT picker names a sheet, the map has ignored
   *  the sheet picker since A2.1 and draws the drawing's world-placed
   *  geometry, and the layer list follows the map. So changing the picker
   *  barely moved the list and the control looked broken. It was not; nothing
   *  on screen said what the list was of.
   */
  const layersScope = useMemo(() => {
    if (viewMode === "svg") {
      return {
        label: `layers on sheet ${layout || "—"}`,
        note: "this sheet is what Normal 2D draws, so the picker governs everything here",
        fromStore: true,
      };
    }
    if (scopeLayers) {
      return {
        label: `layers in ${scopeLayers.scope_label}`,
        note: scopeLayers.scope_note,
        fromStore: true,
      };
    }
    return {
      label: "layers seen so far",
      note:
        "the scope's own layer list has not arrived, so this is built from " +
        "the geometry fetched up to now and may grow. It is not the drawing " +
        "changing",
      fromStore: false,
    };
  }, [viewMode, layout, scopeLayers]);

  /** Layouts that have a cached render, biggest first. */
  const drawableLayouts = useMemo(
    () => drawableLayoutsRaw.filter((l) => showBlockLayouts || !l.is_block),
    [drawableLayoutsRaw, showBlockLayouts],
  );

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <button
            className="brand-mark"
            onClick={() => setPanelOpen((open) => !open)}
            title={panelOpen ? "Hide the side panel" : "Show the side panel"}
            aria-expanded={panelOpen}
          >
            ◫
          </button>
          <div>
            <h1>ADK × AutoCAD</h1>
            <p>Read a DWG, click its objects, comment on them.</p>
          </div>
        </div>

        <div className="pickers">
          {/*  Three views, named for what they are rather than for how they
               are built — except that "Deck.GL" is exactly what a reviewer
               asked to see, so it stays. A segmented control rather than a
               fourth dropdown: this is the one control on the page that is
               switched constantly, and it must show all three states at once
               so that "there is also a 3D view" is discoverable without
               opening anything. */}
          <div className="view-switch" role="group" aria-label="View">
            {(["deck2d", "deck3d", "svg"] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                className={viewMode === mode ? "active" : ""}
                onClick={() => setViewMode(mode)}
                aria-pressed={viewMode === mode}
                title={
                  mode === "svg"
                    ? "The drawing as it plots: border, title block, dimension text. The only view that can show a paper sheet."
                    : mode === "deck3d"
                      ? "The same map, tilted, with plots extruded to placeholder heights."
                      : geometry?.frame === "lnglat"
                        ? "Stored outlines on a real basemap."
                        : "Stored outlines in the drawing's own coordinates."
                }
              >
                {VIEW_LABELS[mode]}
              </button>
            ))}
          </div>

          <label>
            <span>Drawing</span>
            <select
              value={drawingId}
              onChange={(e) => setDrawingId(e.target.value)}
              disabled={drawings.length === 0}
            >
              {drawings.map((d) => (
                <option key={d._id} value={d._id}>
                  {d.original_filename} · {d.entity_count.toLocaleString()} entities
                </option>
              ))}
            </select>
          </label>

          {/* In the map views this picker does NOT govern what is on screen.
              The map draws the drawing's world-placed geometry and has done
              since A2.1, so changing the sheet barely moves the layer list --
              which read as broken, because the control looked like it decided
              what was drawn and nothing said otherwise.

              Labelled rather than disabled: it still chooses the sheet that
              Normal 2D will draw, and a dead control that silently forgets
              the user's choice on the way back is worse than an honest one.
              In Normal 2D the label goes back to plain "Layout", because
              there the sheet IS what is drawn. */}
          <label
            className={`layout-picker${viewMode === "svg" ? "" : " sheet-only"}`}
            title={
              viewMode === "svg"
                ? "The sheet on screen."
                : "The map draws the drawing's world-placed geometry, not this " +
                  "sheet. Choosing here decides what Normal 2D shows."
            }
          >
            <span>
              Layout
              {viewMode !== "svg" && (
                <em className="layout-picker-note">sheet view only</em>
              )}
            </span>
            <select
              value={layout}
              onChange={(e) => setLayout(e.target.value)}
              disabled={!detail}
            >
              {/* Sheets and block definitions are separated. `tablet.dxf` has
                  208 blocks, so a flat list buries the three real layouts. */}
              {drawableLayouts.some(isSheet) && (
                <optgroup label="Sheets (as plotted)">
                  {drawableLayouts.filter(isSheet).map((l) => (
                    <option key={l.name} value={l.name}>
                      {l.name} · {drawnCount(detail, l.name).toLocaleString()} drawn
                    </option>
                  ))}
                </optgroup>
              )}
              {drawableLayouts.some((l) => !l.is_block && !isSheet(l)) && (
                <optgroup label="Model space">
                  {drawableLayouts
                    .filter((l) => !l.is_block && !isSheet(l))
                    .map((l) => (
                      <option key={l.name} value={l.name}>
                        {l.name} · {l.entity_count.toLocaleString()}
                      </option>
                    ))}
                </optgroup>
              )}
              {showBlockLayouts && drawableLayouts.some((l) => l.is_block) && (
                <optgroup label="Block definitions">
                  {drawableLayouts
                    .filter((l) => l.is_block)
                    .map((l) => (
                      <option key={l.name} value={l.name}>
                        {l.name.replace("[block] ", "")} ·{" "}
                        {l.entity_count.toLocaleString()}
                      </option>
                    ))}
                </optgroup>
              )}
            </select>
          </label>
        </div>
      </header>

      {error && (
        <div className="banner error">
          {error}
        </div>
      )}
      {booting && <div className="banner">Loading drawings…</div>}
      {!booting && drawings.length === 0 && !error && (
        <div className="banner">
          No drawings ingested yet. Run:{" "}
          <code>docker compose run --rm cad-api python -m app.ingest --all</code>
        </div>
      )}
      {/*  What this file holds that the picture cannot show — one line.
       *
       *  These were three stacked paragraphs of prose, permanently open,
       *  eating a third of the viewport above the drawing. Every fact in them
       *  is worth keeping: a blank rectangle with no explanation reads as a
       *  broken viewer rather than as a known limit of DXF, and the documents
       *  stored inside this file turned out to hold the land-use key. What
       *  was wrong was the weight, not the content. Collapsed by default,
       *  every word still one click away.
       *
       *  Nothing here is specific to any drawing. Measured across the 18
       *  ingested files: 5 have something that cannot be drawn — three of
       *  them 3D solids and lights in Autodesk's own visualization samples —
       *  and 2 carry an embedded document. The other 13 show no line at all.
       */}
      {(fileNotes.undrawn || fileNotes.documents.length > 0 || fileNotes.unknown) && (
        <details className="banner file-notes">
          {/*  English, like the rest of the chrome. The agent's replies follow
              the language of the question; the viewer's own furniture does
              not, and a line that switches language halfway reads as a bug in
              the app rather than as a translation. */}
          <summary>
            {[
              fileNotes.undrawn && `${fileNotes.undrawn} not drawn here`,
              fileNotes.documents.length > 0 &&
                `${fileNotes.documents.length} document${
                  fileNotes.documents.length === 1 ? "" : "s"
                } stored inside the file`,
              fileNotes.linked > 0 &&
                `${fileNotes.linked} linked, not stored`,
              fileNotes.unknown && "1 unexplained",
            ]
              .filter(Boolean)
              .join(" · ")}
          </summary>

          {fileNotes.undrawn && (
            <p>
              This view contains {fileNotes.undrawn} that AutoCAD draws and no
              DXF renderer can: the payload lives outside the drawing geometry.
              They appear here as blank areas.
            </p>
          )}

          {fileNotes.documents.length > 0 && (
            <p>
              Documents are stored inside this drawing. They are not part of
              the geometry, so nothing here draws them — open one to read it:{" "}
              {fileNotes.documents.map((p, i) => (
                <span key={p.handle}>
                  {i > 0 && ", "}
                  <a
                    href={embeddedUrl(drawingId, p.handle)}
                    target="_blank"
                    rel="noreferrer"
                    data-embedded-handle={p.handle}
                  >
                    {p.handle}
                  </a>{" "}
                  <span className="muted">
                    ({p.format.toUpperCase()}
                    {p.width && p.height ? `, ${p.width}×${p.height} px` : ""},{" "}
                    {(p.bytes / 1_048_576).toFixed(1)} MB)
                  </span>
                </span>
              ))}
              {fileNotes.linked > 0 && (
                <>
                  {" — "}
                  {fileNotes.linked} more declare an embedded object but carry
                  no bytes in the file: the content is linked, not stored.
                </>
              )}
            </p>
          )}

          {fileNotes.unknown && (
            <p>
              Some objects did not draw and the reason is not known —{" "}
              {fileNotes.unknown}. Reported rather than hidden; see
              docs/ADDING-A-DRAWING.md.
            </p>
          )}
        </details>
      )}
      {detail && (detail.renders?.length ?? 0) === 0 && (
        <div className="banner warn">
          Nothing in this file can be drawn by a 2D renderer.{" "}
          {explainUndrawable(detail)} It is still fully ingested and queryable —
          try the Info tab or ask the agent about it.
        </div>
      )}
      {renderMeta?.truncated && (
        <div className="banner warn">
          Partial render: {renderMeta.truncated_reason}
        </div>
      )}

      <main className={`workspace${panelOpen ? "" : " panel-closed"}`}>
        {/*  The SVG viewer stays MOUNTED when a map view is on screen, hidden
             with an attribute rather than unmounted. Three working features
             read the live SVG DOM and would break if it went away:
             `selectByHandle` and `isHandleInDrawing` resolve a handle by
             querying `.viewer-canvas [data-handle]`, and the client-side
             region engine indexes the rendered geometry. Unmounting would
             cost all three to save nothing — the document is already built,
             and a hidden one costs no frames. Same idiom as the tab panels
             below, and for the same reason. */}
        <div className="view-slot" hidden={viewMode !== "svg"}>
          <Viewer
            svg={svg}
            loading={svgLoading}
            hiddenLayers={hiddenLayers}
            commentedHandles={commentedHandles}
            selected={selected}
            onSelect={onSelect}
            onLayersDiscovered={setRenderedLayers}
            onOffLayersDiscovered={onOffLayersDiscovered}
            panelOpen={panelOpen}
            onTogglePanel={() => setPanelOpen((open) => !open)}
            commentsByHandle={commentsByHandle}
            exportUrl={
              drawingId && Object.keys(commentsByHandle).length > 0
                ? `${API_BASE}/drawings/${drawingId}/export`
                : null
            }
            /* `export_url` is the API's own answer to "is there anything to
               download", and it is null unless the drawing is BOTH
               georeferenced and indexed. Reading the two flags here and
               combining them would be a second opinion that could disagree
               with the file the button points at. */
            geoExportUrl={
              detail?.geo?.export_url
                ? `${API_BASE}${detail.geo.export_url}?res=${geoExportRes}`
                : null
            }
            geoExportRes={geoExportRes}
            onGeoExportRes={setGeoExportRes}
            geoExportReason={
              detail && !detail?.geo?.export_url
                ? [detail?.geo?.reason, detail?.geo?.how_to_fix]
                    .filter(Boolean)
                    .join(" — ") || null
                : null
            }
            regionTool={regionTool}
            onRegionToolChange={setRegionTool}
            regions={regions}
            onRegionDrawn={onRegionDrawn}
            onRegionEdit={onRegionEdited}
            onRegionClear={clearRegion}
            regionHandles={regionHandles}
            onRegionSupport={setRegionSupported}
            regionEnabled={regionSupported}
            onRegionBlockedHelp={() => {
              setTab("selection");
              setPanelOpen(true);
            }}
            answerHighlight={answerPlan?.highlight ?? null}
            answerGroups={answerGroups}
            answerPoints={answerPoints}
            answerPointsNote={answerPointsNote}
            onClearAnswerHighlight={clearAnswer}
          />
        </div>

        {/*  The map views, mounted only while one of them is on screen.
             Unlike the SVG above, nothing else in the app reads their DOM, so
             there is no reason to keep a WebGL context and 3,169 outlines
             alive behind a hidden attribute. `mode` is the single flag that
             separates 2D from 3D: same data, same layers, same picking, one
             camera and one `extruded`. */}
        {viewMode !== "svg" && (
          <div className="view-slot">
            <MapViewer
              data={geometry}
              // A view must not report absence while it is still asking.
              //
              // `geometryLoading` is only true once the REQUEST starts, and
              // the effect that starts it returns early while the page is
              // still working out which layout to ask about. In that window
              // loading was false and geometry was null, so the map said
              // "No geometry." about a drawing it had not asked for yet:
              // measured, Sedra came up with that message, an empty layout
              // picker and Layers (0) while Janadriyah loaded normally in the
              // same session. Deciding what to ask and asking are the same
              // thing from the user's seat.
              //
              // A drawing chosen, no answer and no error, is not an answer.
              loading={
                geometryLoading || (!!drawingId && !geometry && !geometryError)
              }
              error={geometryError}
              mode={viewMode === "deck3d" ? "3d" : "2d"}
              showLabels={showLabels}
              onShowLabelsChange={setShowLabels}
              cells={cells}
              cellsLoading={cellsLoading}
              cellsReason={cellsReason}
              onZoomChange={onMapZoom}
              onBoundsChange={onMapBounds}
              /* The API's own answer, for the same reason the GeoJSON export
                 button reads it rather than working it out: whether a drawing
                 has a position on the earth is a fact about the drawing, and
                 a viewer deciding it separately is a second opinion that can
                 disagree. */
              georeferencedByApi={detail?.geo?.georeferenced}
              hiddenLayers={hiddenLayers}
              commentedHandles={commentedHandles}
              regionHandles={regionHandles}
              selected={selected}
              onSelect={onSelect}
              answerHighlight={answerPlan?.highlight ?? null}
              onClearAnswerHighlight={clearAnswer}
              onRegionSelected={onMapRegion}
              panelOpen={panelOpen}
              onTogglePanel={() => setPanelOpen((open) => !open)}
            />
          </div>
        )}

        <aside className={`sidebar${panelOpen ? "" : " closed"}`}>
          <nav className="tabs">
            {(
              ["entity", "selection", "layers", "agent", "upload", "info"] as Tab[]
            ).map((name) => (
              <button
                key={name}
                className={tab === name ? "active" : ""}
                onClick={() => setTab(name)}
              >
                {name === "entity"
                  ? "Entity"
                  : name === "selection"
                    ? regionResult
                      ? `Selection (${regionResult.total.toLocaleString()})`
                      : "Selection"
                    : name === "layers"
                      ? `Layers (${panelLayers.length})`
                      : name === "agent"
                        ? "Agent"
                        : name === "upload"
                          ? "Upload"
                          : "Info"}
              </button>
            ))}
          </nav>

          {/* Panels stay mounted and are hidden with CSS rather than being
              conditionally rendered. Unmounting threw away their state: the
              chat panel lost the whole conversation *and* its session id the
              moment the user clicked a handle in the agent's reply, because
              selecting an entity switches to the Entity tab — so the one
              feature that links an answer to the geometry destroyed the
              conversation every time it was used. */}
          <div className="tab-body" hidden={tab !== "entity"}>
            <EntityPanel
              drawingId={drawingId}
              selection={selected}
              onCommentAdded={onCommentAdded}
              parcel={parcel}
              parcelLoading={parcelLoading}
              onSelectHandle={selectByHandle}
            />
          </div>
          <div className="tab-body" hidden={tab !== "selection"}>
            <SelectionPanel
              region={region}
              parts={groups}
              combined={combined}
              onRemovePart={removeGroup}
              result={regionResult}
              loading={regionLoading}
              error={regionError}
              onModeChange={setRegionMode}
              onClear={clearRegion}
              onSelectHandle={selectByHandle}
              onAskAgent={askAgentAboutSelection}
              supported={regionSupported}
              fetchGroupRows={fetchGroupRows}
            />
          </div>
          <div className="tab-body" hidden={tab !== "layers"}>
            <LayerPanel
              renderedLayers={panelLayers}
              hidden={hiddenLayers}
              onToggle={toggleLayer}
              onSetAll={setHiddenLayers}
              drawingId={drawingId}
              /* The layout the visible entities are STORED under, not the
                 sheet name. An impact count filtered by a paper sheet would
                 report almost nothing on a drawing whose sheets project model
                 space through viewports. */
              scopeLayout={scopeLayout || layout}
              scope={layersScope}
            />
          </div>
          <div className="tab-body" hidden={tab !== "agent"}>
            <ChatPanel
              drawingId={drawingId}
              drawingName={detail?.original_filename ?? ""}
              layout={layout}
              scopeLayout={scopeLayout || layout}
              isHandleInDrawing={isHandleInDrawing}
              onHandleMentioned={selectByHandle}
              selection={agentSelection}
              onClearSelection={() => setAgentSelection(null)}
              selectionNote={selectionNote}
              onAnswer={onAgentAnswer}
              onAnswerCleared={clearAnswer}
            />
          </div>
          {/* Mounted like every other panel, and here the reason is sharper
              than state: the upload panel POLLS. Unmounting it on a tab
              change would stop the poll, and a reader who looked at a layer
              mid-ingest would come back to a panel that had quietly stopped
              watching -- the exact failure the run document exists to
              prevent. */}
          <div className="tab-body" hidden={tab !== "upload"}>
            <UploadPanel
              onIngested={onIngested}
              onOpenDrawing={setDrawingId}
              isDrawingListed={isDrawingListed}
            />
          </div>
          <div className="tab-body" hidden={tab !== "info"}>
            {detail && <DrawingInfo detail={detail} meta={renderMeta} />}
          </div>
        </aside>
      </main>
    </div>
  );
}

function DrawingInfo({
  detail,
  meta,
}: {
  detail: DrawingDetail;
  meta?: { rendered_entities: number; bytes_raw: number; bytes_gzip: number; elapsed_ms: number };
}) {
  const topTypes = Object.entries(detail.counts_by_type ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  return (
    <div className="info-panel">
      <dl className="entity-facts">
        <dt>File</dt>
        <dd>{detail.original_filename}</dd>
        <dt>drawing_id</dt>
        <dd><code>{detail._id}</code></dd>
        <dt>Format</dt>
        <dd>{detail.dxf_version} · {detail.acad_release}</dd>
        <dt>Units</dt>
        <dd>
          {detail.units_name}
          {detail.units_code === 0 && (
            <span className="muted"> (not declared in file)</span>
          )}
        </dd>
        <dt>Entities</dt>
        <dd>{detail.entity_count.toLocaleString()}</dd>
        <dt>Layers</dt>
        <dd>{detail.layers?.length ?? 0}</dd>
        <dt>Blocks</dt>
        <dd>{detail.blocks?.length ?? 0}</dd>
        <dt>Source size</dt>
        <dd>{(detail.file_bytes / 1e6).toFixed(1)} MB</dd>
        {detail.audit_errors > 0 && (
          <>
            <dt>Audit</dt>
            <dd>{detail.audit_errors} errors recovered on open</dd>
          </>
        )}
      </dl>

      {detail.extents && (
        <>
          <h4>Extents (computed, not from header)</h4>
          <p className="mono small">
            min [{detail.extents.min.map((n) => n.toFixed(2)).join(", ")}]<br />
            max [{detail.extents.max.map((n) => n.toFixed(2)).join(", ")}]<br />
            <span className="muted">in {detail.units_name}</span>
          </p>
        </>
      )}

      {meta && (
        <>
          <h4>This render</h4>
          <p className="mono small">
            {meta.rendered_entities.toLocaleString()} entities ·{" "}
            {(meta.bytes_raw / 1e6).toFixed(1)} MB →{" "}
            {(meta.bytes_gzip / 1e6).toFixed(2)} MB gzip ·{" "}
            {(meta.elapsed_ms / 1000).toFixed(1)} s
          </p>
        </>
      )}

      <h4>Entity types</h4>
      <ul className="bar-list">
        {topTypes.map(([type, n]) => (
          <li key={type}>
            <span>{type}</span>
            <span className="muted">{n.toLocaleString()}</span>
          </li>
        ))}
      </ul>

      {detail.xrefs?.length > 0 && (
        <>
          <h4>External references ({detail.xrefs.length})</h4>
          <ul className="bar-list">
            {detail.xrefs.map((x) => (
              <li key={x.block_name}>
                <span>{x.block_name}</span>
                <span className="muted">{x.resolved ? "resolved" : "unresolved"}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      {detail.warnings?.length > 0 && (
        <>
          <h4>Warnings</h4>
          <ul className="bar-list">
            {detail.warnings.map((w, i) => (
              <li key={i}><span className="muted small">{w}</span></li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

/** Say what is in a drawing that produced no render at all, so a blank screen
 *  is an explanation rather than a dead end. */
function explainUndrawable(detail: DrawingDetail): string {
  const counts = detail.counts_by_type ?? {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (total === 0) {
    return "The file contains no entities at all — its content did not survive the DWG to DXF conversion.";
  }
  const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const top = ranked
    .slice(0, 3)
    .map(([type, n]) => `${n} × ${type}`)
    .join(", ");

  // The old sentence blamed 3D solids whatever the drawing held, and on Sedra
  // it listed 774,095 × LINE, 85,903 × LWPOLYLINE and 49,167 × ARC before
  // saying they had no 2D projection. Those are exactly the things a 2D
  // renderer draws, so the message sent readers after a cause that was not
  // there. The reason is now read off the types instead of assumed.
  const drawable2d = new Set([
    "LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE",
    "TEXT", "MTEXT", "HATCH", "SOLID", "POINT", "DIMENSION", "LEADER", "INSERT",
  ]);
  const drawn = ranked
    .filter(([type]) => drawable2d.has(type))
    .reduce((sum, [, n]) => sum + n, 0);

  if (drawn > total / 2) {
    return (
      `It holds ${top}, which a 2D renderer can draw. Nothing was rendered ` +
      `because no sheet of this drawing was produced — its content sits ` +
      `inside block definitions rather than on a layout. The map views read ` +
      `the geometry directly and are not affected.`
    );
  }
  return `It holds ${top} — 3D solids and meshes have no 2D projection to draw.`;
}

/** "2 PDF underlays and 3 embedded OLE objects" */
function summariseUnrenderable(
  items: { type: string }[],
): string {
  const names: Record<string, [string, string]> = {
    PDFUNDERLAY: ["PDF underlay", "PDF underlays"],
    DWFUNDERLAY: ["DWF underlay", "DWF underlays"],
    DGNUNDERLAY: ["DGN underlay", "DGN underlays"],
    OLE2FRAME: ["embedded OLE object", "embedded OLE objects"],
    OLEFRAME: ["embedded OLE object", "embedded OLE objects"],
    ACAD_PROXY_ENTITY: ["proxy entity", "proxy entities"],
    "3DSOLID": ["3D solid", "3D solids"],
    BODY: ["3D body", "3D bodies"],
    REGION: ["region", "regions"],
  };
  const counts = new Map<string, number>();
  items.forEach((i) => counts.set(i.type, (counts.get(i.type) ?? 0) + 1));
  const parts = [...counts.entries()].map(([type, n]) => {
    const [one, many] = names[type] ?? [type, `${type} objects`];
    return `${n} ${n === 1 ? one : many}`;
  });
  return parts.length > 1
    ? `${parts.slice(0, -1).join(", ")} and ${parts.at(-1)}`
    : parts[0];
}

function describe(err: unknown): string {
  if (err instanceof ApiError) {
    return err.hint ? `${err.message} — ${err.hint}` : err.message;
  }
  return err instanceof Error ? err.message : String(err);
}
