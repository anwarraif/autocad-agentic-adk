"use client";

/**
 * The deck.gl views — one instance, 2D and 3D a flag apart.
 *
 * ## What this is not
 *
 * It is not a replacement for the SVG viewer, and §3.1 of the architecture
 * note gives the three reasons: the plot fidelity lives in `render.py` and is
 * not portable (six calibrations, each with a measured failure behind it);
 * paper sheets are in millimetres and cannot go on a map at all; and identity
 * and hit-testing are already solved in the DOM. What deck.gl is strictly
 * better at is what the SVG viewer cannot do at all — real geography,
 * extrusion, and tens of thousands of features at 60 fps.
 *
 * ## Two placements, and why the second one exists
 *
 * **Georeferenced.** The drawing has a `crs:` block, the geometry arrives as
 * lng/lat, and it sits on a basemap. deck.gl carries 64-bit emulation for
 * `LNGLAT`, which is what makes this precise enough: Janadriyah's easting is
 * 691,308, and at that magnitude a float32 step is about 5 cm — feed raw UTM
 * into a shader and lines visibly wobble as you zoom.
 *
 * **Plain.** Only one of the nineteen ingested drawings has a CRS. The other
 * eighteen still have geometry worth looking at, so they are drawn in their
 * own coordinates with an orbit camera and no basemap. Coordinates are
 * translated by the site origin first, for the same float32 reason: the
 * subtraction happens here in float64 and what reaches the GPU is small.
 *
 * The switch between them is `data.frame`, which the server decides. Nothing
 * in this file guesses a placement — a drawing that cannot be placed says so
 * rather than appearing somewhere plausible and wrong.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { MapView, OrbitView, COORDINATE_SYSTEM, WebMercatorViewport } from "@deck.gl/core";
import { PathLayer, PolygonLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
// Version-matched with the backend's h3 4.5.0. The only call used is
// `cellToParent`, which is the same direction the server derives in — coarser
// cells are always reached from finer ones, never re-indexed. Deriving them
// the other way makes the two disagree for anything near a cell boundary.
import { cellToParent, getResolution } from "h3-js";
// Aliased: react-map-gl's default export is named `Map`, which shadows the
// global `Map` constructor for this whole module — `new Map<string, number>()`
// below then fails to compile with a message that names neither.
import { Map as BaseMap, type MapRef } from "react-map-gl/maplibre";
import { outlineForDrawing } from "@/lib/outlineForDrawing.mjs";
import { nextSize } from "@/lib/appSize.mjs";
import "maplibre-gl/dist/maplibre-gl.css";

import type { AnswerHighlight } from "@/lib/answerHighlight";
import {
  approximationNotes,
  anyClassified,
  boundsOfOutline,
  drawsAsArea,
  frameWithOutline,
  type Box,
  type GeometryResponse,
  type MapFeature,
} from "@/lib/mapGeometry";
import {
  dominantUse,
  valueOf,
  type CellMeasure,
  type GeoCell,
  type GeoResponse,
} from "@/lib/mapCells";
import { CELL_INK, classify, colorForValue } from "@/lib/cellColors";
import {
  catchmentRings,
  fetchVicinity,
  totalCounts,
  type CatchmentRing,
  type VicinityResponse,
} from "@/lib/vicinity";
import {
  colorForUse,
  colorForUseRgba,
  legendFor,
  labelForUse,
  OUTLINE_INK,
} from "@/lib/landUseColors";
import { cellHeights, placeholderHeights, type HeightSource } from "@/lib/heights";
import {
  rectFrom,
  selectInRect,
  type RegionHit,
  type RegionMode,
} from "@/lib/mapRegion";
import type { Selection } from "./Viewer";

/** Which of the two deck views is on screen. The data, the layers and every
 *  interaction are identical; the camera and one flag are not. */
export type MapMode = "2d" | "3d";

export type Basemap = "satellite" | "streets" | "none";

/** Raster styles rather than vector ones, and no API key anywhere.
 *
 *  A vector basemap would look better and would need a token, a token needs
 *  an account, and an account is a thing that expires halfway through a demo.
 *  Both sources below are open with attribution, which is rendered on the
 *  canvas because attribution that is not shown is not attribution. */
const BASEMAP_STYLES: Record<
  Exclude<Basemap, "none">,
  { tiles: string[]; attribution: string }
> = {
  satellite: {
    tiles: [
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    ],
    attribution: "Imagery: Esri, Maxar, Earthstar Geographics",
  },
  streets: {
    tiles: ["https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"],
    attribution: "© OpenStreetMap contributors, © CARTO",
  },
};

function styleFor(basemap: Exclude<Basemap, "none">) {
  const { tiles, attribution } = BASEMAP_STYLES[basemap];
  return {
    version: 8 as const,
    sources: {
      base: { type: "raster" as const, tiles, tileSize: 256, attribution },
    },
    layers: [{ id: "base", type: "raster" as const, source: "base" }],
  };
}

/** Fill opacity. Low enough that satellite imagery reads through — the whole
 *  reason to put a drawing on imagery is to compare the two — and high enough
 *  that a land use is still identifiable at a glance. */
const FILL_ALPHA_2D = 150;
const FILL_ALPHA_3D = 225;

/** How far from the cursor a pick may reach, in pixels. A plot at site zoom
 *  is a few pixels across, and requiring a hit on its exact centre is the
 *  difference between "click a plot" and "hunt for a plot". */
/** The catchment boundary's colour.
 *
 *  Lime, and chosen by elimination rather than taste: it has to be
 *  distinguishable from every encoding already on the canvas — the eight
 *  land-use hues, the blue cell ramp, sky blue for the selection and amber
 *  for an agent's answer. Lime is in none of those and reads on both
 *  satellite imagery and the dark chrome. */
const CATCHMENT_INK: [number, number, number, number] = [190, 242, 100, 255];

/** The answer's own colours, and the casing that makes them survive a
 *  basemap.
 *
 *  Amber alone does not. Janadriyah sits in desert, and on satellite imagery
 *  the ground is a warm tan of very nearly amber's hue and lightness: an
 *  amber hexagon over it reads as a slightly different patch of sand, and an
 *  amber parcel outlined in the pale cream this used to use disappears
 *  outright. Asked in review on 1 September 2026 with the two pictures side
 *  by side, and the complaint was exactly right.
 *
 *  The fix is not another hue. Any colour picked to beat tan loses somewhere
 *  else -- the dark chrome with no basemap, the near-white of the streets
 *  basemap, the eight land-use hues underneath -- and a highlight that is
 *  legible on one background and invisible on another is the same bug with a
 *  different screenshot. What works on ALL of them is a light mark with a
 *  DARK CASING under it, the way a map label stays readable over aerial
 *  photography: the casing carries the shape, the amber carries the meaning.
 *
 *  Not a new convention either. The comment markers on this same canvas are
 *  already amber filled and dark outlined, and have been legible over the
 *  desert the whole time; this extends that idiom to the answer, which
 *  should have had it from the start.
 */
const ANSWER_INK: [number, number, number, number] = [251, 191, 36, 255];
/** Near-black, and slightly transparent so the drawing underneath still
 *  shows through the casing rather than being cut out by it. */
const ANSWER_CASING: [number, number, number, number] = [9, 13, 20, 225];

const PICK_RADIUS_PX = 4;
/** How far a press may travel and still count as a click rather than a pan. */
const CLICK_SLOP_PX = 4;

interface ViewStateLngLat {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

interface ViewStateOrbit {
  target: [number, number, number];
  zoom: number;
  rotationX: number;
  rotationOrbit: number;
}

interface Props {
  data: GeometryResponse | null;
  loading: boolean;
  error: string | null;
  mode: MapMode;
  /** The H3 layer, or `null` when this drawing has none. Passed in rather
   *  than fetched here for the same reason `data` is: the page owns what is
   *  loaded for which drawing, and two components fetching on their own
   *  render cycles is how a view ends up showing one drawing's cells over
   *  another drawing's parcels. */
  /** Whether to draw text labels. Lifted to the page: it decides whether the
   *  geometry request asks for labels at all, so a component that kept its
   *  own copy would be toggling a switch the fetch could not see. */
  showLabels: boolean;
  onShowLabelsChange: (on: boolean) => void;
  /** Where the camera is looking, west,south,east,north in degrees.
   *
   *  Reported from the same effect as `onZoomChange` and for the same reason:
   *  the OPENING camera never passes through `onViewStateChange`, because
   *  `fitTo` writes the view state directly. Every frame of a drag lands
   *  here; the PAGE decides what is worth refetching, because the page owns
   *  the request. */
  onBoundsChange?: (bounds: [number, number, number, number]) => void;
  cells: GeoResponse | null;
  cellsLoading?: boolean;
  /** Reports the camera's zoom, so the page can ask for the H3 resolution
   *  that suits what is on screen.
   *
   *  The camera lives here and the request lives there, so one of them has to
   *  cross. The zoom crosses rather than the resolution because which rung of
   *  the hierarchy a zoom deserves is a property of the DATA — how many cells
   *  the site has at each resolution — and the page is what holds the data.
   *  A viewer deciding it would be guessing about a ladder it cannot see.
   *
   *  Only meaningful under a lng/lat camera: the orbit view's zoom is a
   *  log-scale of a drawing's own units, which shares a name with web
   *  mercator zoom and nothing else. Ungeoreferenced drawings have no cells
   *  to re-ask for anyway, so this stays silent for them. */
  onZoomChange?: (zoom: number) => void;
  /** Why there are no cells, when there are none and the reason is a fact
   *  about the drawing rather than a failure. Shown, not swallowed. */
  cellsReason?: string | null;
  /** Whether the API says this drawing has a position on the earth.
   *
   *  Read from the drawing rather than inferred from whether the geometry
   *  payload happened to carry a CRS. Working it out here would be a second
   *  opinion that could disagree with the one the rest of the app uses. */
  georeferencedByApi?: boolean;
  /** Shared with the SVG viewer, so hiding a layer hides it in every view.
   *  This is the point of Fig 12: the map views are free riders on state the
   *  page already keeps. */
  hiddenLayers: Set<string>;
  commentedHandles: Set<string>;
  /** Every handle in the current selection, from whichever view produced it.
   *  Shared with the SVG viewer, so a region drawn on the sheet is visible on
   *  the map and vice versa. */
  regionHandles: Set<string> | null;
  selected: Selection | null;
  onSelect: (selection: Selection | null) => void;
  /** Layers present in the geometry, reported up so the layer panel can show
   *  which of them the map can actually draw. */
  onLayersDiscovered?: (layers: { name: string; count: number }[]) => void;
  answerHighlight: AnswerHighlight | null;
  onClearAnswerHighlight: () => void;
  /** A rectangle selection, already resolved to handles. Not a `Region` in
   *  drawing coordinates: in the georeferenced placement the rectangle is
   *  drawn in degrees, and there is no client-side inverse projection to turn
   *  it back into eastings. Sending the answer rather than the shape also
   *  keeps `POST /selection` out of the loop entirely — the test ran against
   *  the real outlines, which is a better answer than the server's bounding
   *  boxes, and pretending otherwise would report the wrong `basis`. */
  onRegionSelected?: (hit: RegionHit, mode: RegionMode) => void;
  panelOpen: boolean;
  onTogglePanel: () => void;
}

export function MapViewer({
  data,
  loading,
  error,
  mode,
  showLabels,
  onShowLabelsChange,
  cells,
  cellsLoading = false,
  cellsReason = null,
  onZoomChange,
  onBoundsChange,
  georeferencedByApi,
  hiddenLayers,
  commentedHandles,
  regionHandles,
  selected,
  onSelect,
  onLayersDiscovered,
  answerHighlight,
  onClearAnswerHighlight,
  onRegionSelected,
  panelOpen,
  onTogglePanel,
}: Props) {
  const [basemap, setBasemap] = useState<Basemap>("satellite");
  /** Which land use the legend is isolating, if any.
   *
   *  Not a nicety. A land-use map is a choropleth, and no eight-hue palette
   *  clears the all-pairs colour-separation floors — see the note in
   *  `landUseColors.ts`. Isolating one use is the facet escape that makes the
   *  palette honest, so this control is part of the encoding, not a filter
   *  bolted on afterwards. */
  const [isolated, setIsolated] = useState<string | null>(null);
  /** Whether the hexagon overlay is drawn.
   *
   *  Off by default. The drawing is what this app is for; the cells are an
   *  analysis laid over it, and someone who has not asked for an analysis
   *  should be looking at the drawing. */
  const [showCells, setShowCells] = useState(false);
  /** What the hexagons encode. Count and area answer different questions —
   *  "where are the plots" against "where is the land" — and on a masterplan
   *  with 300 m2 houses beside a 6 ha park they disagree sharply. */
  const [measure, setMeasure] = useState<CellMeasure>("count");
  const [hoverCell, setHoverCell] = useState<{
    x: number;
    y: number;
    /** The whole bucket, not its id — `bucket.cell` is the hexagon. */
    bucket: GeoCell;
  } | null>(null);

  // --- catchment ------------------------------------------------------------
  //
  //  Owned here rather than by the page, unlike the geometry and the cells.
  //  Those are shared — the layers panel counts them, the fallback reads them
  //  — and a catchment is not: it is drawn on this canvas, cleared from this
  //  canvas, and nothing else in the app has a use for it. Lifting it to the
  //  page would add a prop chain that only ever has one consumer.
  const [catchmentOpen, setCatchmentOpen] = useState(false);
  const [catchment, setCatchment] = useState<VicinityResponse | null>(null);
  const [catchmentLoading, setCatchmentLoading] = useState(false);
  const [catchmentError, setCatchmentError] = useState<string | null>(null);
  /** Rings, not metres. The reach in metres is what the RESPONSE reports,
   *  because it depends on the resolution the recipe actually ran at; asking
   *  for "100 m" here and printing "100 m" there would be this component
   *  agreeing with itself rather than with the server. */
  const [catchmentK, setCatchmentK] = useState(2);
  const [catchmentSubject, setCatchmentSubject] = useState<string>("");
  const catchmentAbort = useRef<AbortController | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; feature: MapFeature } | null>(
    null,
  );
  const hostRef = useRef<HTMLDivElement | null>(null);
  const deckRef = useRef<{
    deck?: {
      canvas: HTMLCanvasElement;
      getViewports: () => { unproject: (xy: number[]) => number[] }[];
      pickObject: (opts: {
        x: number;
        y: number;
        radius?: number;
      }) => { object?: unknown } | null;
    };
  } | null>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);

  // --- rectangle selection -------------------------------------------------
  //
  //  An explicit mode rather than shift-drag. The SVG viewer can use
  //  shift-drag because nothing else wants it; deck's controller already
  //  binds shift-drag to rotate, and quietly stealing it would break the one
  //  gesture the 3D view most needs.
  const [selecting, setSelecting] = useState(false);
  /** The live rectangle, in CSS pixels relative to the canvas. Pixels, not
   *  coordinates: the drag is a screen gesture, and the mode it chooses is a
   *  screen-direction question. Converted once, on release. */
  const [drag, setDrag] = useState<{
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  } | null>(null);

  /** Whether this view is drawing on the earth.
   *
   *  The API's answer wins when there is one: `georeferenced` is a fact about
   *  the drawing and the drawing detail states it. The frame of the geometry
   *  payload is the fallback for the case where the page has not resolved the
   *  detail yet, and it agrees whenever both are present — they are two
   *  readings of the same CRS config. */
  const georeferenced =
    georeferencedByApi !== undefined
      ? georeferencedByApi && data?.frame === "lnglat"
      : data?.frame === "lnglat";

  /** Whenever the container's size is known or changes, tell the basemap.
   *
   *  Idempotent: `resize()` on a map already at that size does nothing. */
  useEffect(() => {
    if (!size) return;
    baseMapRef.current?.getMap()?.resize();
  }, [size]);

  /** Measure whatever element is CURRENTLY the map's host.
   *
   *  A callback ref, not an effect, because `hostRef` is attached to four
   *  different `<section>`s -- the loading, error, empty and normal branches
   *  below all render one. An effect with `[]` deps observes whichever of
   *  them existed at mount and then keeps observing it after React has
   *  swapped it out, so it watches a detached node for the rest of the
   *  session.
   *
   *  That is what made switching drawings blank the map. Changing drawing
   *  turns `loading` on, the loading branch mounts a NEW section, the
   *  observer is still on the old one, `size` never updates, and
   *  `{viewState && size && <DeckGL/>}` renders nothing at all. Measured in
   *  Chrome: the container reported 1327x597 with zero canvases in it, the
   *  toolbar said "2,885 outlines", every panel was correct, and the only
   *  cure a user could find was resizing the window. Present at 26d9f9d,
   *  before any of this session's work.
   *
   *  A callback ref runs on every attach and detach, so the observer always
   *  follows the live element.
   */
  /** The basemap instance, so it can be TOLD to re-measure.
   *
   *  `reuseMaps` hands back a recycled MapLibre map, and a recycled map keeps
   *  the canvas size it was put away with. Passing `style={{width, height}}`
   *  sets the element's CSS and MapLibre does not act on it: measured against
   *  Atlas, deck's canvas was a correct 1327x703 while MapLibre's sat at its
   *  300x150 default, so the drawing painted and the satellite behind it did
   *  not. Deck follows `size` through its props; MapLibre has to be asked.
   */
  const baseMapRef = useRef<MapRef | null>(null);
  const sizeObserver = useRef<ResizeObserver | null>(null);
  const resizeListener = useRef<(() => void) | null>(null);
  const attachHost = useCallback((el: HTMLDivElement | null) => {
    hostRef.current = el;
    sizeObserver.current?.disconnect();
    sizeObserver.current = null;
    if (resizeListener.current) {
      window.removeEventListener("resize", resizeListener.current);
      resizeListener.current = null;
    }
    if (!el) return;
    // Measure NOW as well as on change: the observer's first callback is
    // asynchronous, and an element that already has its size would otherwise
    // spend a frame with none -- which is a frame of blank map.
    // ONE rule for adopting a measurement, in `lib/appSize.mjs`, rather than a
    // fourth place that decides for itself. Every consumer of "how big is
    // this" reads the same function, and its guard asserts the thing that has
    // failed three times now: a stale size must not survive a change.
    const measure = () => {
      const box = el.getBoundingClientRect();
      setSize((prev) =>
        nextSize(prev, { width: box.width, height: box.height }),
      );
    };
    measure();
    const observer = new ResizeObserver(() => measure());
    observer.observe(el);
    sizeObserver.current = observer;

    // A window resize as well as an element resize. A ResizeObserver fires on
    // the BOX, and a layout that reflows without changing this element's box
    // -- a breakpoint stacking the panels around it, say -- can leave the
    // holder believing a width the window no longer has. This is the
    // "does not recover" half of the report, and it costs one listener.
    window.addEventListener("resize", measure);
    resizeListener.current = measure;
  }, []);

  // --- geometry, split by what it is ---------------------------------------

  /** For the plain placement: every coordinate moved to a local origin.
   *
   *  Done once, in float64, before anything reaches the GPU. Janadriyah's
   *  coordinates run to seven figures, and float32 cannot hold them to better
   *  than a few centimetres — which is visible as vertices that snap and
   *  edges that will not close.
   */
  const translated = useMemo(() => {
    if (!data || data.frame !== "world" || !data.origin) return data?.features ?? [];
    const [ox, oy] = data.origin.world;
    return data.features.map((f) => ({
      ...f,
      coordinates: f.coordinates.map(
        ([x, y]) => [x - ox, y - oy] as [number, number],
      ),
      centroid: f.centroid
        ? ([f.centroid[0] - ox, f.centroid[1] - oy] as [number, number])
        : null,
    }));
  }, [data]);

  const visible = useMemo(
    () =>
      translated.filter(
        (f) =>
          !hiddenLayers.has(f.layer) &&
          (isolated === null || f.land_use === isolated),
      ),
    [translated, hiddenLayers, isolated],
  );

  /** Whether this drawing's layers carry any land-use decision. Asked once
   *  per payload, over EVERY feature rather than the visible ones — hiding
   *  the only classified layer must not flip the whole drawing into its
   *  unclassified fallback and repaint everything else. */
  const classified = useMemo(
    () => anyClassified(data?.features ?? []),
    [data],
  );

  /** Placeholder until the backend can answer "how tall is this".
   *
   *  Built here rather than inside the layer so that swapping in
   *  `measuredHeights()` is one line, and so the caveat printed on the canvas
   *  comes from the source itself rather than from a separate constant that
   *  could disagree with it. */
  const heights: HeightSource = useMemo(
    () => placeholderHeights(classified),
    [classified],
  );

  const parcels = useMemo(
    () => visible.filter((f) => drawsAsArea(f, classified)),
    [visible, classified],
  );
  /** Rings that are not plots: block boundaries, hatched overlays, anything a
   *  config marked `structure` or `overlay`. Drawn as outlines only — filling
   *  them would paint over the parcels they contain and, in 3D, would put a
   *  wall around every superblock. */
  const boundaries = useMemo(
    () => visible.filter((f) => f.kind === "ring" && !drawsAsArea(f, classified)),
    [visible, classified],
  );
  /** Open runs of points: polylines and single segments alike.
   *
   *  A LINE's two endpoints and an open polyline's vertex list are the same
   *  thing to a PathLayer, and splitting them into two layers would buy
   *  nothing but a second set of accessors to keep in step. What distinguishes
   *  them travels per feature, in `is_chord`. */
  const paths = useMemo(
    () => visible.filter((f) => f.kind === "path" || f.kind === "segment"),
    [visible],
  );
  const labels = useMemo(
    () => (showLabels ? visible.filter((f) => f.kind === "label" && f.text) : []),
    [visible, showLabels],
  );

  /** What a rectangle may select: exactly what is on screen.
   *
   *  `visible` is not that. Label anchors are fetched with the geometry and
   *  are in `visible` whether or not the Labels switch is on — measured, a
   *  window drag over the estate returned 800 TEXT and 381 outlines, and the
   *  800 were invisible. Selecting something the user cannot see is worse
   *  than missing it: the count is wrong and there is nothing on screen that
   *  explains why.
   */
  const selectable = useMemo(
    () => [...parcels, ...boundaries, ...paths, ...labels],
    [parcels, boundaries, paths, labels],
  );

  const legend = useMemo(
    () =>
      legendFor(
        translated.filter((f) => f.kind === "ring"),
        cells?.dimension?.name === "layer" ? "layer" : "land_use",
      ),
    // The dimension belongs here: without it the memo keeps the previous
    // dimension's rows when only the dimension changed.
    [translated, cells?.dimension?.name],
  );

  const answered = useMemo(
    () => new Set(answerHighlight?.handles ?? []),
    [answerHighlight],
  );

  /** Objects the agent named, as points. Only those the map can place — an
   *  object with no centroid is counted on the badge rather than dropped
   *  silently, which is the same rule the SVG viewer's marker overlay uses. */
  const answerPoints = useMemo(
    () => translated.filter((f) => answered.has(f.handle) && f.centroid),
    [translated, answered],
  );
  const commentPoints = useMemo(
    () => translated.filter((f) => commentedHandles.has(f.handle) && f.centroid),
    [translated, commentedHandles],
  );

  useEffect(() => {
    if (!onLayersDiscovered || !data) return;
    const counts = new Map<string, number>();
    for (const f of data.features) counts.set(f.layer, (counts.get(f.layer) ?? 0) + 1);
    onLayersDiscovered(
      [...counts.entries()]
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count),
    );
  }, [data, onLayersDiscovered]);

  // --- the hexagon overlay --------------------------------------------------

  /** The cells actually drawn: those with something in them under the current
   *  measure and the current isolation.
   *
   *  Isolation is honoured here as well as on the parcels, so clicking
   *  "Education" in the legend leaves the hexagons showing where the schools
   *  are rather than where everything is. A cell that holds none of the
   *  isolated use drops out entirely instead of being drawn empty — an empty
   *  hexagon over a place with no schools is a statement about the grid, not
   *  about the site. */
  const cellData = useMemo(() => {
    if (!cells?.cells?.length) return [];
    return cells.cells.filter((c) => valueOf(c, measure, isolated) > 0);
  }, [cells, measure, isolated]);

  /** The classing behind the colours, and the maximum behind the heights.  /** The classing behind the colours, and the maximum behind the heights.
   *
   *  Computed over the cells ON SCREEN, so isolating a land use rescales the
   *  ramp to that use. A ramp kept on the all-uses scale would paint every
   *  education cell the palest step and say nothing. */
  const classing = useMemo(
    () => classify(cellData.map((c) => valueOf(c, measure, isolated))),
    [cellData, measure, isolated],
  );

  /** Heights for the hexagons. Measured, unlike the parcels' — see the note
   *  in `heights.ts` about why both can be on screen at once. */
  const cellHeightSource = useMemo(
    () =>
      cellHeights(
        measure,
        classing?.max ?? 0,
        cells?.area_unit ?? data?.units.area_unit ?? null,
      ),
    [measure, classing, cells, data],
  );

  /** The cells an agent's answer lands in.
   *
   *  The join the H3 plan called "one more hop": an answer already names
   *  handles, `/geo` gives every parcel the cell it is counted in, so the
   *  handles become cells. Mapped to the resolution ON SCREEN with
   *  `cellToParent` — the same direction the server derives in, so a lit
   *  hexagon is the one the server would have counted the parcel in.
   *
   *  Only parcels can be reached this way. An answer naming a road or a text
   *  object lights no hexagon, because `/geo` never held one for it, and the
   *  badge below says so rather than letting a partial highlight read as a
   *  complete one. */
  const answerCells = useMemo(() => {
    if (!cells?.parcels || answered.size === 0) return null;
    const target = cells.resolution;
    const out = new Set<string>();
    let matched = 0;
    for (const feature of cells.parcels.features) {
      if (!answered.has(feature.properties.handle)) continue;
      matched++;
      const cell = feature.properties.cell;
      if (!cell) continue;
      try {
        out.add(getResolution(cell) === target ? cell : cellToParent(cell, target));
      } catch {
        /* a cell coarser than the target has no parent at it; skip rather
           than guess at one */
      }
    }
    return { cells: out, matched, asked: answered.size };
  }, [cells, answered]);

  /** The answer's cells, as the layer that lights them wants them.
   *
   *  Taken from `cells.cells` rather than from `cellData` on purpose: the
   *  drawn set is filtered by whichever land use the legend has isolated, and
   *  an answer must not half-disappear because the reader is looking at one
   *  use. If a cell somehow is not in the payload it is still drawn, from its
   *  id alone — the hexagon's position comes from the id, and an answer that
   *  silently dropped a cell would be the one failure this layer exists to
   *  prevent. */
  const answerCellData = useMemo<GeoCell[]>(() => {
    if (!answerCells || answerCells.cells.size === 0) return [];
    const known = new Map((cells?.cells ?? []).map((c) => [c.cell, c]));
    return [...answerCells.cells].map(
      (id) => known.get(id) ?? { cell: id, counts: {}, area_m2: {} },
    );
  }, [answerCells, cells]);

  const runCatchment = useCallback(
    (subject: string, k: number) => {
      if (!data || !subject) return;
      catchmentAbort.current?.abort();
      const controller = new AbortController();
      catchmentAbort.current = controller;
      setCatchmentLoading(true);
      setCatchmentError(null);
      fetchVicinity(
        data.drawing_id,
        { layout: data.layout, subject, k, outline: true, bands: true },
        controller.signal,
      )
        .then((result) => {
          if (controller.signal.aborted) return;
          setCatchment(result);
          setCatchmentError(null);
        })
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          setCatchment(null);
          setCatchmentError(
            err instanceof Error ? err.message : "The catchment could not be computed.",
          );
        })
        .finally(() => {
          if (controller.signal.aborted) return;
          setCatchmentLoading(false);
        });
    },
    [data],
  );

  const clearCatchment = useCallback(() => {
    catchmentAbort.current?.abort();
    setCatchment(null);
    setCatchmentError(null);
    setCatchmentLoading(false);
  }, []);

  /** A catchment belongs to the drawing and layout it was computed for.
   *  Switching either must drop it rather than leave a disk from one site
   *  drawn over another. */
  useEffect(() => {
    clearCatchment();
    setCatchmentOpen(false);
  }, [data?.drawing_id, data?.layout, clearCatchment]);

  /** The disks, as polygons deck can draw. */
  const catchmentShapes: CatchmentRing[] = useMemo(
    () => catchmentRings(catchment),
    [catchment],
  );
  /** The subjects the disks are drawn around, so they can be picked out from
   *  the parcels inside them. */
  const catchmentSubjects = useMemo(
    () => new Set((catchment?.subjects ?? []).map((s) => s.handle)),
    [catchment],
  );

  /** The site boundary as the store describes it, when it gave us one and the
   *  view is in the frame it is expressed in. */
  const outlineBounds = useMemo(
    () =>
      georeferenced && data?.frame === "lnglat"
        ? // `outlineForDrawing`, not `cells.coverage_outline` directly. The
          // geometry and the cells are two separate requests, so on a drawing
          // change `data` is the new drawing while `cells` is still the old
          // one -- and the camera framed the PREVIOUS site. Worse, the fit
          // effect keys on `framing.basis`, which reads "outline" either way,
          // so once fitted to the stale box the key stopped changing when the
          // right outline arrived and it never refitted. See the guard in
          // lib/outlineForDrawing.test.mjs.
          boundsOfOutline(outlineForDrawing(data, cells))
        : null,
    [georeferenced, data, cells],
  );

  // --- camera ---------------------------------------------------------------

  const [lngLatView, setLngLatView] = useState<ViewStateLngLat | null>(null);
  const [orbitView, setOrbitView] = useState<ViewStateOrbit | null>(null);
  /** Identifies the data the camera was fitted to, so a new drawing or layout
   *  refits and a re-render of the same data does not yank the view back. */
  const fittedTo = useRef<string | null>(null);

  /** Where the content is, as opposed to where the extents are. Computed once
   *  per payload: it sorts 3,169 centroids, which is nothing, but doing it on
   *  every camera move would be. */
  /** The DEFAULT camera's box, and what it is a box of.
   *
   *  The server's `content_bounds` when it has one: the smallest set of stored
   *  H3 cells holding most of the placed objects. A drawing's extreme
   *  bounding box is not where the drawing is -- Janadriyah's `NBHD *`
   *  boundaries are real, correctly placed, up to 12 km out, and were
   *  shrinking the estate to a few pixels. See section 13.6 of
   *  docs/INTAKE-TO-AGENT-PLAN.md.
   *
   *  Falling back to the old content frame is deliberate rather than a
   *  cushion: a drawing with no cells has nothing better, and a view that
   *  went blank because an optional summary was absent would be the rule in
   *  section 2 broken again.
   */
  const contentBox = useMemo<{ box: Box; source: "cells" | "features" } | null>(() => {
    const c = data?.content_bounds;
    if (c && data?.frame === "lnglat") {
      const [[minX, minY], [maxX, maxY]] = c.lnglat;
      return { box: { minX, minY, maxX, maxY }, source: "cells" };
    }
    const inferred = frameWithOutline(visible, outlineBounds);
    return inferred?.box ? { box: inferred.box, source: "features" } : null;
  }, [data, visible, outlineBounds]);

  const framing = useMemo(
    () => frameWithOutline(visible, outlineBounds),
    [visible, outlineBounds],
  );

  const fitTo = useCallback(
    (box: Box | null) => {
      if (!data || !size || !box) return;
      if (data.frame === "lnglat") {
        const viewport = new WebMercatorViewport({
          width: size.width,
          height: size.height,
        });
        const { longitude, latitude, zoom } = viewport.fitBounds(
          [
            [box.minX, box.minY],
            [box.maxX, box.maxY],
          ],
          { padding: 48 },
        );
        setLngLatView({
          longitude,
          latitude,
          zoom,
          pitch: mode === "3d" ? 50 : 0,
          bearing: 0,
        });
      } else {
        const spanX = box.maxX - box.minX;
        const spanY = box.maxY - box.minY;
        const origin = data.origin?.world ?? [0, 0];
        setOrbitView({
          // The bounds are in untranslated coordinates; the geometry is not.
          target: [
            (box.minX + box.maxX) / 2 - origin[0],
            (box.minY + box.maxY) / 2 - origin[1],
            0,
          ],
          zoom: Math.log2(
            Math.min((size.width - 96) / spanX, (size.height - 96) / spanY),
          ),
          rotationX: mode === "3d" ? 50 : 90,
          rotationOrbit: 0,
        });
      }
    },
    [data, size, mode],
  );

  /** The default camera: where the drawing is, not where its corners are. */
  const fit = useCallback(
    () => fitTo(contentBox?.box ?? framing?.box ?? null),
    [fitTo, contentBox, framing],
  );
  /** Everything, outliers included. Offered because "the camera did not start
   *  on it" and "it is not there" must never look the same. */
  const fitAll = useCallback(
    () => fitTo(framing?.extents ?? null),
    [fitTo, framing],
  );

  useEffect(() => {
    if (!data || !size) return;
    // The framing basis is part of the key. The coverage outline arrives on
    // its own request, after the geometry, and it frames the site better than
    // the fence does — so the camera must be allowed to settle once when it
    // lands. Without the basis in the key that better answer would be
    // computed and then ignored.
    const key = `${data.drawing_id}:${data.layout}:${data.frame}:${framing?.basis ?? "none"}:${contentBox?.source ?? "none"}:${data.content_bounds?.cells_used ?? ""}`;
    if (fittedTo.current === key) return;
    // Nothing to aim at YET. Recording the key here anyway is what left the
    // camera over the PREVIOUS drawing: `fitTo(null)` moves nothing, the key
    // was marked as fitted regardless, and if `framing` then arrived without
    // changing `basis` the retry never came. Janadriyah's geometry is at
    // 46.90 E and the camera stayed at Sedra's 46.7 E, so a correct drawing,
    // correctly fetched, showed an empty map. An attempt that could not run
    // is not an attempt.
    const target = contentBox?.box ?? framing?.box ?? framing?.extents ?? null;
    if (!target) return;
    fittedTo.current = key;
    fit();
  }, [data, size, fit, framing?.basis, framing?.box, framing?.extents, contentBox]);

  /** Tell the page what the camera is looking at, so the H3 request can
   *  follow it up and down the hierarchy.
   *
   *  Reported from an effect on the view state rather than from
   *  `onViewStateChange`, because the OPENING camera never passes through
   *  that handler — `fitTo` writes the view state directly. A map that only
   *  learned its zoom once the user touched it would open on whatever rung
   *  the default happened to name, which on a site that fits the screen at
   *  zoom 15 is the wrong one. Watching the state itself covers the fit, the
   *  Fit button, the 2D/3D switch and every wheel turn with one rule.
   *
   *  Every frame of a drag lands here. That is deliberate and cheap: the page
   *  turns a zoom into a resolution and only re-renders when the RESOLUTION
   *  changes, after the camera has settled. */
  useEffect(() => {
    if (!georeferenced || !lngLatView) return;
    onZoomChange?.(lngLatView.zoom);
    if (onBoundsChange && size) {
      // Flat [west, south, east, north] in this deck.gl version, not the
      // nested corner pairs the older API returned.
      const [west, south, east, north] = new WebMercatorViewport({
        ...lngLatView,
        width: size.width,
        height: size.height,
      }).getBounds();
      onBoundsChange([west, south, east, north]);
    }
  }, [georeferenced, lngLatView, onZoomChange, onBoundsChange, size]);

  /** Switching 2D↔3D tilts the camera; it does not move it.
   *
   *  Keeping the centre and the zoom is what makes the two views feel like
   *  one view seen two ways. Refitting on every switch would lose the plot
   *  the user was looking at, which is the only reason they switched. */
  useEffect(() => {
    setLngLatView((v) => (v ? { ...v, pitch: mode === "3d" ? 50 : 0 } : v));
    setOrbitView((v) => (v ? { ...v, rotationX: mode === "3d" ? 50 : 90 } : v));
  }, [mode]);

  // --- layers ---------------------------------------------------------------

  const coordinateSystem = georeferenced
    ? COORDINATE_SYSTEM.LNGLAT
    : COORDINATE_SYSTEM.CARTESIAN;

  const extruded = mode === "3d";

  const fillFor = useCallback(
    (f: MapFeature): [number, number, number, number] => {
      if (selected?.handle === f.handle) return [56, 189, 248, 255];
      if (answered.has(f.handle)) return [251, 191, 36, 235];
      // A member of the selection keeps its land-use hue and gains opacity.
      // Painting the whole selection one colour would answer "what did I
      // select" while destroying "what is in it", and on a mixed-use block
      // the second question is the one being asked.
      const inRegion = regionHandles?.has(f.handle) ?? false;
      return colorForUseRgba(
        f.land_use,
        inRegion ? 250 : extruded ? FILL_ALPHA_3D : FILL_ALPHA_2D,
      );
    },
    [selected, answered, extruded, regionHandles],
  );

  const lineFor = useCallback(
    (f: MapFeature): [number, number, number, number] => {
      if (selected?.handle === f.handle) return [255, 255, 255, 255];
      // Dark, not the pale cream this used to be. The fill is already amber
      // and nearly opaque; what the parcel needed was an EDGE that separates
      // it from whatever it is sitting on, and on desert imagery a cream
      // edge on an amber fill is one colour.
      if (answered.has(f.handle)) return ANSWER_CASING;
      if (regionHandles?.has(f.handle)) return [56, 189, 248, 255];
      return OUTLINE_INK;
    },
    [selected, answered, regionHandles],
  );

  const elevationFor = useCallback(
    (f: MapFeature) => heights.heightFor(f)?.metres ?? 0,
    [heights],
  );

  const pickSelection = useCallback(
    (feature: MapFeature | null) => {
      if (!feature) {
        onSelect(null);
        return;
      }
      // The same `Selection` the SVG viewer emits, deliberately. Everything
      // downstream — the entity panel, comments, the agent's scope, the
      // selection set — is reached through this one shape, and a second
      // selection type would be a second place for the answer to be wrong.
      onSelect({
        handle: feature.handle,
        layer: feature.layer,
        type: feature.type,
      });
    },
    [onSelect],
  );

  /** Screen pixels -> the frame the geometry is in.
   *
   *  Through deck's own viewport rather than a hand-rolled inverse: the
   *  viewport already knows the projection, the pitch and the zoom, and a
   *  second implementation of that maths would be wrong the first time
   *  someone tilted the camera.
   */
  const unproject = useCallback((x: number, y: number): [number, number] | null => {
    const viewport = deckRef.current?.deck?.getViewports?.()[0];
    if (!viewport) return null;
    const point = viewport.unproject([x, y]);
    return [point[0], point[1]];
  }, []);

  const finishDrag = useCallback(() => {
    const box = drag;
    setDrag(null);
    if (!box || !onRegionSelected) return;
    // A press that never travelled is a click, not a region. Without this a
    // stray click while the mode is on wipes the selection with an empty
    // rectangle.
    if (Math.abs(box.x1 - box.x0) < 4 && Math.abs(box.y1 - box.y0) < 4) return;

    const a = unproject(box.x0, box.y0);
    const b = unproject(box.x1, box.y1);
    if (!a || !b) return;
    // Left-to-right is window, right-to-left is crossing. AutoCAD's rule, and
    // the same one the SVG viewer uses -- decided on the SCREEN direction of
    // the drag, which survives any camera the map is under.
    const mode: RegionMode = box.x1 >= box.x0 ? "window" : "crossing";
    onRegionSelected(selectInRect(selectable, rectFrom(a, b), mode), mode);
  }, [drag, onRegionSelected, unproject, selectable]);

  /** Where a press began, so a release can tell a click from a drag. */
  const pressAt = useRef<{ x: number; y: number } | null>(null);

  /** Select whatever is under a screen point.
   *
   *  Deliberately NOT deck's own `onClick`. Two reasons, one fatal and one
   *  merely bad:
   *
   *  * It never fires here. deck derives `click` from a Hammer tap gesture
   *    whose recognizer chain did not resolve in this environment —
   *    `deck.props.onClick` is a function and `deck.pickObject()` at the same
   *    pixel returns the parcel, but the gesture never reaches the handler.
   *    Verified against the live instance rather than guessed at.
   *  * Even where it fires, that tap is configured `requireFailure:
   *    ['dblclick']`, so every selection waits out the double-click interval
   *    before anything appears. On a viewer whose whole job is "click an
   *    object, read about it", a third of a second of nothing is a real cost.
   *
   *  Picking directly is immediate, deterministic, and lets us ask for a
   *  radius — worth having when the target is a 300 m² plot a few pixels
   *  across at site zoom.
   */
  const pickAt = useCallback(
    (clientX: number, clientY: number) => {
      const deck = deckRef.current?.deck;
      if (!deck) return;
      const rect = deck.canvas.getBoundingClientRect();
      const info = deck.pickObject({
        x: clientX - rect.left,
        y: clientY - rect.top,
        radius: PICK_RADIUS_PX,
      });
      const object = info?.object as MapFeature | GeoCell | undefined;
      // A hexagon is an aggregate, not an object: there is no entity behind
      // it for the panel to open. Clicking one is treated as clicking the
      // ground it covers, which clears the selection — the same thing that
      // happens without the overlay, so turning it on does not change what a
      // click means.
      const feature =
        object && "handle" in object ? (object as MapFeature) : null;
      pickSelection(feature);
    },
    [pickSelection],
  );

  const layers = useMemo(() => {
    if (!data) return [];
    const triggers = {
      getFillColor: [selected?.handle, answered.size, extruded, regionHandles],
      getLineColor: [selected?.handle, answered.size, regionHandles],
      getElevation: [extruded, heights.id],
      getLineWidth: [selected?.handle, regionHandles],
    };
    return [
      /*  The hexagons go first, so they are drawn UNDER the drawing.
          The overlay is context for the linework, not a replacement for it:
          putting it on top would hide the parcels it is aggregating, which
          is the one thing a reader needs in order to check it.

          Its picking is off. Two pickable layers stacked on the same ground
          would make "click a plot" resolve to whichever the picker reached
          first, and the plot is always the intended target. The cells report
          themselves on hover instead, which costs nothing and takes nothing
          away. */
      ...(showCells && cellData.length > 0 && georeferenced
        ? [
            new H3HexagonLayer<GeoCell>({
              id: "cells",
              data: cellData,
              getHexagon: (c) => c.cell,
              // High precision keeps a cell's drawn boundary on the boundary
              // the server counted against. The cheap mode approximates the
              // shape, which is fine for a heatmap and not fine for a layer
              // whose whole claim is "this parcel is counted in this cell".
              highPrecision: true,
              extruded: extruded,
              filled: true,
              stroked: true,
              getFillColor: (c) =>
                colorForValue(
                  valueOf(c, measure, isolated),
                  classing,
                  // Transparent in both, and MORE so in 3D rather than
                  // less. A column has depth: in 3D the fill is crossed
                  // twice and stacks against its neighbours, so the value
                  // that reads as a light wash flat reads as a solid wall
                  // extruded. The drawing has to stay visible through it.
                  extruded ? 150 : 120,
                ),
              // Every cell is stroked. A sequential ramp must pass through
              // the luminance of any mid-tone basemap somewhere, so at some
              // point in the ramp the fill cannot carry the shape; the
              // boundary always can. See `cellColors.ts`.
              getLineColor: (c) =>
                answerCells?.cells.has(c.cell)
                  ? ([251, 191, 36, 255] as [number, number, number, number])
                  : CELL_INK,
              getLineWidth: (c) => (answerCells?.cells.has(c.cell) ? 2.5 : 1),
              lineWidthUnits: "pixels",
              lineWidthMinPixels: 1,
              getElevation: (c) =>
                cellHeightSource.heightFor(valueOf(c, measure, isolated)).metres,
              elevationScale: 1,
              /*  Pickable, and safe because this layer is FIRST. deck picks
                  the topmost pickable object, so a parcel under the cursor
                  always wins and a cell only answers where there is no parcel
                  — which is exactly where a reader would be asking what the
                  cell holds. Clicking one selects nothing: `pickAt` takes
                  only objects that carry a handle, because a cell is an
                  aggregate and there is no entity for the panel to show. */
              pickable: true,
              updateTriggers: {
                getFillColor: [measure, isolated, classing, extruded],
                getLineColor: [answerCells],
                getLineWidth: [answerCells],
                getElevation: [measure, isolated, cellHeightSource.id],
              },
            }),
          ]
        : []),
      /*  The cells an answer landed in, lit ON TOP of the ordinary grid.
          Still under the drawing, for the same reason the grid is: the
          parcels the answer actually names have to stay readable over the
          hexagons that hold them.

          A second layer rather than a branch inside the first. The grid's
          fill is a choropleth carrying a measured value, and an answer is
          not a value on that scale — folding the two into one
          `getFillColor` would put a categorical amber into a sequential
          blue ramp and make both unreadable. Two layers keep the grid
          saying what it says and the answer saying what it says.

          Amber, matching the parcels the answer marks in the linework above
          and the marker overlay in the SVG view, so one answer is one
          colour wherever the reader is looking.

          Not pickable: hovering a lit cell should report what the grid
          under it holds, which is the layer below still answering. */
      ...(showCells && answerCellData.length > 0 && georeferenced
        ? [
            //  The casing. A wide dark edge drawn UNDER the amber one, so
            //  that what separates the answer's hexagons from the ground is
            //  a contrast in lightness rather than in hue — the one
            //  difference that survives tan desert, a white street map and
            //  the dark chrome alike. Unfilled: it is an edge, not a tint.
            new H3HexagonLayer<GeoCell>({
              id: "cells-answer-casing",
              data: answerCellData,
              getHexagon: (c) => c.cell,
              highPrecision: true,
              extruded: false,
              filled: false,
              stroked: true,
              getLineColor: ANSWER_CASING,
              getLineWidth: 6,
              lineWidthUnits: "pixels",
              lineWidthMinPixels: 5,
              pickable: false,
              //  Never buried. In 3D the ordinary grid becomes columns, and a
              //  flat mark at ground level disappears behind the nearest one
              //  — the answer would be present in the data and invisible on
              //  the screen, which is the same failure as the wrong colour.
              //  Ignoring depth is right for this layer specifically: it is
              //  not a thing in the scene, it is a statement about the scene.
              parameters: { depthCompare: "always" as const },
            }),
            new H3HexagonLayer<GeoCell>({
              id: "cells-answer",
              data: answerCellData,
              getHexagon: (c) => c.cell,
              highPrecision: true,
              // Flat even in 3D. This layer says WHERE, and an extruded
              // column would both hide the grid's own height (which does
              // carry a value) and read as a measurement of the answer.
              extruded: false,
              filled: true,
              stroked: true,
              // Denser than the 110 it opened at. That value was chosen
              // against the dark chrome, where a light wash is plenty, and
              // it is not enough over imagery — the tan reads straight
              // through it and the hexagon becomes a patch of sand.
              getFillColor: [251, 191, 36, extruded ? 140 : 165],
              getLineColor: ANSWER_INK,
              getLineWidth: 2.5,
              lineWidthUnits: "pixels",
              lineWidthMinPixels: 2,
              pickable: false,
              parameters: { depthCompare: "always" as const },
            }),
          ]
        : []),
      new PolygonLayer<MapFeature>({
        id: "boundaries",
        data: boundaries,
        coordinateSystem,
        getPolygon: (f) => f.coordinates,
        filled: false,
        stroked: true,
        getLineColor: (f) =>
          regionHandles?.has(f.handle)
            ? ([56, 189, 248, 255] as [number, number, number, number])
            : ([139, 152, 165, 190] as [number, number, number, number]),
        getLineWidth: 1,
        lineWidthUnits: "pixels",
        lineWidthMinPixels: 1,
        pickable: true,
        updateTriggers: { getLineColor: [regionHandles] },
      }),
      new PathLayer<MapFeature>({
        id: "paths",
        data: paths,
        coordinateSystem,
        getPath: (f) => f.coordinates,
        getColor: (f) =>
          selected?.handle === f.handle || regionHandles?.has(f.handle)
            ? [56, 189, 248, 255]
            : // A chord is drawn dimmer than a real segment. It is the only
              // signal available without a dash pattern, and it means a road
              // that visibly fades through a bend is telling the truth about
              // itself rather than asserting a straight line.
              f.is_chord
              ? [226, 232, 240, 150]
              : [226, 232, 240, 235],
        getWidth: 1.6,
        widthUnits: "pixels",
        // Roads were legible only on a pale basemap at 1 px. A centreline is
        // the thing a planner traces with a finger; it has to survive being
        // drawn over satellite imagery of a sand-coloured site.
        widthMinPixels: 1.5,
        pickable: true,
        updateTriggers: { getColor: [selected?.handle, regionHandles] },
        parameters: { depthWriteEnabled: false },
      }),
      new PolygonLayer<MapFeature>({
        id: "parcels",
        data: parcels,
        coordinateSystem,
        getPolygon: (f) => f.coordinates,
        extruded,
        wireframe: false,
        filled: true,
        stroked: true,
        getFillColor: fillFor,
        getLineColor: lineFor,
        getElevation: elevationFor,
        // Metres, and only meaningful because the drawing is in metres. A
        // drawing in inches never reaches this layer with a georeferenced
        // frame — the endpoint refuses it — and in the plain placement the
        // number is in the drawing's own units, which is what the caveat on
        // the canvas says.
        elevationScale: 1,
        // The selected plot gets a heavier ring, not just a different fill.
        // On a residential estate the selection colour lands next to 2,380
        // parcels already painted blue, and "slightly brighter blue" is not
        // something anyone can find on a screen.
        getLineWidth: (f) =>
          selected?.handle === f.handle ? 3 : regionHandles?.has(f.handle) ? 2 : 1,
        lineWidthUnits: "pixels",
        lineWidthMinPixels: 1,
        pickable: true,
        autoHighlight: true,
        highlightColor: [56, 189, 248, 200],
        material: { ambient: 0.5, diffuse: 0.65, shininess: 24, specularColor: [40, 50, 60] },
        updateTriggers: triggers,
      }),
      new ScatterplotLayer<MapFeature>({
        id: "comments",
        data: commentPoints,
        coordinateSystem,
        getPosition: (f) => f.centroid!,
        getRadius: 6,
        radiusUnits: "pixels",
        radiusMinPixels: 5,
        getFillColor: [251, 191, 36, 235],
        getLineColor: [17, 24, 33, 255],
        stroked: true,
        lineWidthMinPixels: 1.5,
        pickable: true,
      }),
      new ScatterplotLayer<MapFeature>({
        id: "answer-marks",
        data: answerPoints,
        coordinateSystem,
        getPosition: (f) => f.centroid!,
        getRadius: 9,
        radiusUnits: "pixels",
        radiusMinPixels: 7,
        // Filled as well as stroked now. An unfilled amber ring is two thin
        // amber lines against the ground; a filled disc with a dark edge is
        // the same mark the comments use, and it is findable on imagery.
        filled: true,
        stroked: true,
        getFillColor: ANSWER_INK,
        getLineColor: ANSWER_CASING,
        lineWidthMinPixels: 2,
        pickable: false,
      }),
      /*  The catchment goes LAST, over everything it encloses.
          Opposite to the hexagons, and for the opposite reason: the cells are
          context to look through, the catchment is a boundary to look at. A
          disk edge hidden behind the parcels it contains would be useless,
          because its whole job is to say where the counting stopped.

          Unfilled. A fill over 33 cells would tint every parcel inside it and
          break the land-use reading exactly where the reader is trying to
          count by land use. */
      ...(catchmentShapes.length > 0
        ? [
            new PolygonLayer<CatchmentRing>({
              id: "catchment",
              data: catchmentShapes,
              coordinateSystem,
              getPolygon: (d) => d.rings,
              filled: false,
              stroked: true,
              extruded: false,
              getLineColor: CATCHMENT_INK,
              getLineWidth: 2.5,
              lineWidthUnits: "pixels",
              lineWidthMinPixels: 2,
              pickable: false,
            }),
            /*  The subjects, ringed in the same colour. Without them the disks
                are shapes with nothing at their centre, and "around what?" is
                the first thing anyone asks of a catchment. */
            new PolygonLayer<MapFeature>({
              id: "catchment-subjects",
              data: visible.filter((f) => catchmentSubjects.has(f.handle)),
              coordinateSystem,
              getPolygon: (f) => f.coordinates,
              filled: true,
              stroked: true,
              extruded: false,
              getFillColor: [190, 242, 100, 90],
              getLineColor: CATCHMENT_INK,
              getLineWidth: 2.5,
              lineWidthUnits: "pixels",
              lineWidthMinPixels: 2,
              pickable: false,
            }),
          ]
        : []),
      new TextLayer<MapFeature>({
        id: "labels",
        data: labels,
        coordinateSystem,
        getPosition: (f) => f.coordinates[0],
        getText: (f) => f.text ?? "",
        getSize: 11,
        sizeUnits: "pixels",
        getColor: [235, 241, 247, 235],
        outlineWidth: 3,
        outlineColor: [10, 14, 20, 220],
        fontSettings: { sdf: true },
        getTextAnchor: "middle",
        getAlignmentBaseline: "center",
        pickable: false,
      }),
    ];
  }, [
    data,
    showCells,
    cellData,
    answerCellData,
    classing,
    measure,
    isolated,
    answerCells,
    cellHeightSource,
    georeferenced,
    catchmentShapes,
    catchmentSubjects,
    visible,
    boundaries,
    paths,
    parcels,
    commentPoints,
    answerPoints,
    labels,
    coordinateSystem,
    extruded,
    fillFor,
    lineFor,
    elevationFor,
    selected?.handle,
    answered.size,
    heights.id,
    regionHandles,
  ]);

  /** The view, and the basemap style, built once per configuration.
   *
   *  Both used to be constructed inline in the JSX, which meant a new instance
   *  on every render — and `onHover` sets state, so that was every pointer
   *  move across the canvas. deck.gl treats a new view object as a new view:
   *  it tears the viewport and its interaction state down and builds them
   *  again. The visible symptom was that clicking a plot stopped selecting it,
   *  because the press and the release landed on two different deck states.
   *  Rebuilding the MapLibre style object had the same shape of cost, with the
   *  basemap re-reading its sources each time.
   */
  const views = useMemo(
    () =>
      georeferenced
        ? new MapView({ id: "map", controller: true })
        : new OrbitView({ id: "orbit", controller: true, orbitAxis: "Z" }),
    [georeferenced],
  );
  const mapStyle = useMemo(
    () => (basemap === "none" ? null : styleFor(basemap)),
    [basemap],
  );

  // --- what the canvas has to say about itself ------------------------------

  const notes = useMemo(() => {
    if (!data) return [];
    const out: string[] = [];
    const coverage = data.geometry_coverage;
    if (coverage.without_outline > 0) {
      out.push(
        `${coverage.without_outline.toLocaleString()} of ${coverage.entities_in_layout.toLocaleString()} objects in this layout have no stored outline and are not drawn here — ${coverage.types_with_no_outline.join(", ")}. They are in the drawing; the map cannot draw them yet.`,
      );
    }
    if (framing?.tighterThanExtents) {
      const times =
        framing.box.maxX > framing.box.minX
          ? Math.round(
              (framing.extents.maxX - framing.extents.minX) /
                (framing.box.maxX - framing.box.minX),
            )
          : null;
      // Naming the layer matters more than the count. "1,191 objects are
      // somewhere off-screen" reads as a fault; "836 of them are road
      // centrelines" reads as a drawing that extends past its estate, which
      // is what it is.
      const mostly = framing.outsideLayer
        ? ` Most of them — ${framing.outsideLayer.count.toLocaleString()} — are on “${framing.outsideLayer.name}”.`
        : "";
      out.push(
        `The camera frames where the drawing's content is, not its full extents${
          times && times > 1 ? `, which are ${times}× wider` : ""
        }. ${framing.outside.toLocaleString()} object${
          framing.outside === 1 ? " sits" : "s sit"
        } outside the opening view — they are drawn, just not framed.${mostly} “Fit all” shows them.`,
      );
    }
    if (contentBox?.source === "cells" && data.content_bounds) {
      const c = data.content_bounds;
      out.push(
        `The opening view frames where the drawing IS, measured from the ` +
          `${c.cells_total.toLocaleString()} map cells its objects occupy: the ` +
          `${c.cells_used.toLocaleString()} busiest of them hold ` +
          `${c.entities_inside.toLocaleString()} of ${c.entities_total.toLocaleString()} ` +
          `placed objects. The rest are real and correctly placed — a site ` +
          `boundary or key plan parked far from the work — and are still drawn. ` +
          `“Fit all” frames the true extent instead.`,
      );
    }
    out.push(...approximationNotes(data.features));
    if (extruded && !heights.measured) out.push(heights.note);
    if (showCells && cells) {
      if (extruded) out.push(cellHeightSource.note);
      if (classing) out.push(classing.basis);
      // What the overlay is NOT. `/geo` holds classified parcels only, so
      // saying "2,576 of 20,334" here is the difference between a reader
      // treating the hexagons as the drawing and treating them as one view
      // of part of it.
      out.push(
        `The hexagons aggregate ${cells.totals.parcels.toLocaleString()} classified parcels — not the whole drawing. Roads, text, dimensions and anything on an unclassified layer are drawn as linework and counted in no cell. ${cells.counts_scope ?? ""}`,
      );
      if (answerCells && answerCells.matched < answerCells.asked) {
        out.push(
          `${answerCells.matched.toLocaleString()} of the ${answerCells.asked.toLocaleString()} objects in the current answer are parcels with a cell; the rest light no hexagon because the cell layer never held one for them.`,
        );
      }
      if (cells.limits && cells.limits.parcels_omitted > 0) {
        out.push(
          `${cells.limits.parcels_omitted.toLocaleString()} parcels were left out of the cell payload's outlines by its size cap. The cell counts above still include them.`,
        );
      }
      if (cells.excluded && cells.excluded.parcels_without_a_cell > 0) {
        out.push(
          `${cells.excluded.parcels_without_a_cell.toLocaleString()} parcels have no cell and are in no hexagon. ${cells.excluded.note}`,
        );
      }
      // Counted but not drawn, which is a different gap and the one that is
      // actually non-zero here. It matters beyond a footnote: these parcels
      // are INSIDE the hexagon totals but carry no outline in the cell
      // payload, so an answer naming one of them lights no hexagon even
      // though the count behind that hexagon includes it. Verified against
      // the reference drawing: 31 parcels, 28 of them open space.
      if (cells.excluded && cells.excluded.parcels_without_a_usable_ring > 0) {
        out.push(
          `${cells.excluded.parcels_without_a_usable_ring.toLocaleString()} parcels are counted in the hexagons but have no outline the store will vouch for. They are in the totals; an answer that names one will not light its hexagon. ${cells.excluded.note}`,
        );
      }
    }
    if (catchment) {
      // The boundary is the disk's own edge. Saying so on the canvas is what
      // stops a lime ring being read as a survey line or as a 100 m circle.
      out.push(catchment.outline_note);
      out.push(catchment.counted_from);
      out.push(`Not measured here: ${catchment.not_measured}`);
      if (catchment.network_note) out.push(catchment.network_note);
      // The sample cap, stated where it can mislead. A reader who opens a
      // subject expecting its neighbour list to be the count would be wrong
      // by 92 on this drawing.
      const capped = catchment.subjects.filter(
        (subject) => subject.neighbours_total > subject.neighbours_sample.length,
      );
      if (capped.length > 0) {
        out.push(
          `The counts here are complete, but the neighbour LISTS are a sample: the recipe names at most ${catchment.limits.neighbours_listed_per_subject} objects per subject, and ${capped.length} of ${catchment.subjects.length} subjects have more than that.`,
        );
      }
    }
    if (framing?.basis === "coverage-outline") {
      out.push(
        "The camera is framed on the ground this drawing's cells cover, which is the store's own account of where the site is. Its edges are hexagon edges — it is not a surveyed boundary.",
      );
    }
    if (data.crs.known && data.crs.declared_in_file === false) {
      out.push(
        `The coordinate system was inferred, not declared: ${data.crs.name} (EPSG:${data.crs.epsg}). ${data.crs.how_to_verify ?? ""}`,
      );
    }
    return out;
  }, [
    data,
    extruded,
    heights,
    framing,
    showCells,
    cells,
    classing,
    cellHeightSource,
    answerCells,
    catchment,
  ]);

  // --- render ---------------------------------------------------------------

  if (error) {
    return (
      <section className="viewer map-viewer" ref={attachHost}>
        <div className="map-empty">
          <h4>This drawing cannot be shown on the map</h4>
          <p>{error}</p>
          <p className="muted">
            The Normal 2D view draws it as a plotted sheet, which does not need
            a coordinate system.
          </p>
        </div>
      </section>
    );
  }

  // Only when there is NOTHING to show. A refetch is an ADDITION, never a
  // replacement: panning does not change the drawing, so what is already
  // painted stays painted while more arrives. Blanking a correct view in
  // order to fetch more of it is the same family as reporting absence while a
  // request is in flight, and the owner watched the map blink on and off as
  // he panned. The small indicator below carries the "still fetching" fact
  // instead, and a DRAWING change clears `data` at the page, so that case
  // still gets the full state it deserves.
  if (!data) {
    return (
      <section className="viewer map-viewer" ref={attachHost}>
        <div className="map-empty">
          {/* "No geometry." is only ever true with no drawing chosen. With
              one chosen and no payload, the honest answer is that the answer
              has not arrived -- the page's `loading` now covers the window
              before the request even starts, so this cannot claim absence
              about a drawing it has not finished asking about. */}
          <p className="muted">
            {loading ? "Loading geometry…" : "No drawing selected."}
          </p>
        </div>
      </section>
    );
  }

  if (data.features.length === 0) {
    return (
      <section className="viewer map-viewer" ref={attachHost}>
        <div className="map-empty">
          <h4>Nothing in this layout has a stored outline</h4>
          <p>
            {data.geometry_coverage.entities_in_layout.toLocaleString()} objects
            are here, and none of them carries vertices:{" "}
            {data.geometry_coverage.types_with_no_outline.join(", ")}.
          </p>
          <p className="muted">{data.geometry_coverage.note}</p>
        </div>
      </section>
    );
  }

  const viewState = georeferenced ? lngLatView : orbitView;

  return (
    <section className="viewer map-viewer" ref={attachHost}>
      <div className="viewer-toolbar map-toolbar">
        <button
          className="toolbar-btn"
          onClick={onTogglePanel}
          title={panelOpen ? "Hide the side panel" : "Show the side panel"}
        >
          {panelOpen ? "▶" : "◀"}
        </button>
        <span className="toolbar-sep" />
        <span className="toolbar-label">
          {mode === "3d" ? "Deck.GL 3D" : "Deck.GL 2D"} ·{" "}
          {data.counts.ring.toLocaleString()} outlines
        </span>
        <span className="toolbar-sep" />
        {georeferenced ? (
          <label className="toolbar-select">
            <span>Basemap</span>
            <select
              value={basemap}
              onChange={(e) => setBasemap(e.target.value as Basemap)}
            >
              <option value="satellite">Satellite</option>
              <option value="streets">Streets</option>
              <option value="none">None</option>
            </select>
          </label>
        ) : (
          <span className="toolbar-label muted" title={data.placement.fallback}>
            Not georeferenced — drawing coordinates
          </span>
        )}
        <button
          className="toolbar-btn"
          onClick={fit}
          title={
            contentBox?.source === "cells" && data?.content_bounds
              ? `Frame where the drawing is: ${data.content_bounds.note}`
              : "Frame where the content is — the default camera"
          }
        >
          Fit
        </button>
        {(contentBox?.source === "cells" || framing?.tighterThanExtents) && (
          <button
            className="toolbar-btn"
            onClick={fitAll}
            title="Frame the whole extents, outlying objects included"
          >
            Fit all
          </button>
        )}
        <label className="toolbar-check">
          <input
            type="checkbox"
            checked={showLabels}
            onChange={(e) => onShowLabelsChange(e.target.checked)}
          />
          <span>Labels</span>
        </label>
        {/*  The overlay control appears only where there is an overlay to
             control. A drawing with no cells gets the reason on a disabled
             checkbox rather than a missing one: a control that is simply not
             there reads as a feature that is broken. */}
        {georeferenced && (cells || cellsReason || cellsLoading) && (
          <label
            className={`toolbar-check${cells?.cells?.length ? "" : " muted"}`}
            title={
              cells?.cells?.length
                ? "Draw the H3 cells this drawing is indexed into, under the drawing"
                : (cellsReason ?? "This drawing has no cells to draw")
            }
          >
            <input
              type="checkbox"
              checked={showCells && !!cells?.cells?.length}
              disabled={!cells?.cells?.length}
              onChange={(e) => setShowCells(e.target.checked)}
            />
            <span>
              Hexagons
              {cellsLoading
                ? " …"
                : cells?.cells?.length
                  ? ` (${(showCells ? cellData.length : cells.cells.length).toLocaleString()})`
                  : ""}
            </span>
          </label>
        )}
        {showCells && cells?.cells?.length ? (
          <label className="toolbar-select">
            <span>Cells show</span>
            <select
              value={measure}
              onChange={(e) => setMeasure(e.target.value as CellMeasure)}
            >
              <option value="count">Parcel count</option>
              <option
                value="area"
                disabled={!cells.area_unit}
                title={cells.area_note ?? undefined}
              >
                Parcel area
              </option>
            </select>
          </label>
        ) : null}
        {/*  The catchment needs cells, so it is offered exactly where cells
             exist. Its own control rather than a mode on the hexagon switch:
             a catchment answers "what is near this", which is a different
             question from "where is the density", and the two are useful at
             the same time. */}
        {georeferenced && cells?.cells?.length ? (
          <button
            className={`toolbar-btn${catchmentOpen || catchment ? " accent" : ""}`}
            onClick={() => setCatchmentOpen((open) => !open)}
            aria-pressed={catchmentOpen}
            title="Draw the cell neighbourhood around an object or around every parcel of one land use, and count what is inside it"
          >
            {catchment
              ? `Catchment · ${catchment.subjects.length}`
              : "Catchment"}
          </button>
        ) : null}
        {onRegionSelected && (
          <button
            className={`toolbar-btn${selecting ? " accent" : ""}`}
            onClick={() => {
              setSelecting((on) => !on);
              setDrag(null);
            }}
            aria-pressed={selecting}
            title="Drag a box over the drawing. Left to right selects what is wholly inside; right to left selects anything it touches."
          >
            {selecting ? "Selecting ✓" : "Select area"}
          </button>
        )}
        {answerHighlight && (
          <button className="toolbar-btn accent" onClick={onClearAnswerHighlight}>
            {answerHighlight.label} ✕
          </button>
        )}
      </div>

      <div
        className="map-canvas"
        onPointerDown={(e) => {
          pressAt.current = { x: e.clientX, y: e.clientY };
        }}
        onPointerUp={(e) => {
          const start = pressAt.current;
          pressAt.current = null;
          // A press that travelled is a pan, not a selection. Same threshold
          // and same reasoning as the SVG viewer, which learned it the hard
          // way: without it every drag ended by selecting whatever happened
          // to be under the cursor when the user let go.
          if (
            !start ||
            selecting ||
            Math.abs(e.clientX - start.x) > CLICK_SLOP_PX ||
            Math.abs(e.clientY - start.y) > CLICK_SLOP_PX
          ) {
            return;
          }
          // Only presses that landed on the drawing itself. The legend, the
          // notes and the attribution are children of this element, and a
          // click on the legend must isolate a land use rather than clear the
          // selection behind it.
          if (!(e.target as HTMLElement).closest(".deck-events-root")) return;
          pickAt(e.clientX, e.clientY);
        }}
      >
        {viewState && size && (
          <DeckGL
            ref={deckRef as never}
            // Told, not left to observe.
            //
            // Both deck and MapLibre size their canvas from the container
            // they mount into. On a DRAWING SWITCH they mount while the pane
            // is still being laid out and neither follows it afterwards:
            // measured in Chrome, the container was 1327x597 with the deck
            // canvas at 300x150 and MapLibre's at 337x168, so the drawing
            // filled a small rectangle and the satellite tiles never
            // appeared. `size` comes from the callback ref above, which does
            // follow the live element, so passing it makes the canvas track
            // the container by construction.
            width={size.width}
            height={size.height}
            views={views}
            viewState={viewState as never}
            onViewStateChange={({ viewState: next }) => {
              if (georeferenced) setLngLatView(next as unknown as ViewStateLngLat);
              else setOrbitView(next as unknown as ViewStateOrbit);
            }}
            // Panning is disabled while a box is being dragged, or the map
            // would slide out from under the rectangle. Everything else --
            // wheel zoom, rotate -- stays live.
            controller={selecting ? { dragPan: false, dragRotate: false } : true}
            layers={layers}
            onHover={(info) => {
              const object = info.object as MapFeature | GeoCell | undefined;
              const isFeature = !!object && "handle" in object;
              setHover(
                isFeature
                  ? { x: info.x, y: info.y, feature: object as MapFeature }
                  : null,
              );
              setHoverCell(
                object && !isFeature
                  ? { x: info.x, y: info.y, bucket: object as GeoCell }
                  : null,
              );
            }}
            getCursor={({ isDragging }) =>
              isDragging ? "grabbing" : hover ? "pointer" : "grab"
            }
          >
            {georeferenced && mapStyle && (
              <BaseMap
                ref={baseMapRef}
                onLoad={() => baseMapRef.current?.getMap()?.resize()}
                reuseMaps
                style={{ width: size.width, height: size.height }}
                mapStyle={mapStyle}
                attributionControl={false}
              />
            )}
          </DeckGL>
        )}

        {loading && (
          <div className="map-fetching" role="status" aria-live="polite">
            <span className="map-fetching-dot" />
            Fetching more
          </div>
        )}

        {selecting && (
          <div
            className="map-drag-catcher"
            onPointerDown={(e) => {
              const box = e.currentTarget.getBoundingClientRect();
              const x = e.clientX - box.left;
              const y = e.clientY - box.top;
              e.currentTarget.setPointerCapture(e.pointerId);
              setDrag({ x0: x, y0: y, x1: x, y1: y });
            }}
            onPointerMove={(e) => {
              if (!drag) return;
              const box = e.currentTarget.getBoundingClientRect();
              setDrag({
                ...drag,
                x1: e.clientX - box.left,
                y1: e.clientY - box.top,
              });
            }}
            onPointerUp={finishDrag}
            onPointerCancel={() => setDrag(null)}
          >
            {drag && (
              <div
                /* Blue for window, green for crossing -- coloured DURING the
                   drag, not after, so the user sees which mode they are in
                   while they can still change it. Same feedback AutoCAD
                   gives, and the same two colours the SVG viewer uses. */
                className={`map-drag-rect ${drag.x1 >= drag.x0 ? "window" : "crossing"}`}
                style={{
                  left: Math.min(drag.x0, drag.x1),
                  top: Math.min(drag.y0, drag.y1),
                  width: Math.abs(drag.x1 - drag.x0),
                  height: Math.abs(drag.y1 - drag.y0),
                }}
              />
            )}
          </div>
        )}

        {hover && (
          <div
            className="map-tip"
            style={{ left: hover.x + 14, top: hover.y + 14 }}
          >
            <strong>{hover.feature.handle}</strong>
            <span className="muted"> · {hover.feature.type}</span>
            <div>{hover.feature.layer}</div>
            {/* Identity is never colour alone: the use is named here, in the
                entity panel, and in the legend. */}
            <div className="map-tip-use">
              {labelForUse(hover.feature.land_use)}
              {hover.feature.land_use_subtype
                ? ` · ${hover.feature.land_use_subtype}`
                : ""}
            </div>
            {hover.feature.area != null && (
              <div className="muted">
                {hover.feature.area.toLocaleString(undefined, {
                  maximumFractionDigits: 1,
                })}{" "}
                {data.units.area_unit ?? "sq. units"}
              </div>
            )}
            {extruded && (
              <div className="muted">
                {heights.heightFor(hover.feature)?.basis ?? "no height"}
              </div>
            )}
          </div>
        )}

        {/*  A cell's tooltip prints the number it is drawn from, so nothing
             on this layer has to be read off a colour. That is what the
             contrast note in `cellColors.ts` obligates, and it is also just
             the useful thing to show. */}
        {hoverCell && (
          <div
            className="map-tip"
            style={{ left: hoverCell.x + 14, top: hoverCell.y + 14 }}
          >
            <strong>
              {valueOf(hoverCell.bucket, measure, isolated).toLocaleString(
                undefined,
                { maximumFractionDigits: 1 },
              )}
            </strong>
            <span className="muted">
              {" "}
              {measure === "area"
                ? (cells?.area_unit ?? "sq. units")
                : valueOf(hoverCell.bucket, measure, isolated) === 1
                  ? "parcel"
                  : "parcels"}
              {isolated ? ` · ${labelForUse(isolated)}` : ""}
            </span>
            <div className="map-tip-use">
              {isolated
                ? labelForUse(isolated)
                : `Mostly ${labelForUse(dominantUse(hoverCell.bucket))}`}
            </div>
            {/* The cell id, because it is the thing that can be pasted into
                h3geo.org and checked against this view by someone who does
                not trust it. */}
            <div className="muted">
              {hoverCell.bucket.cell} · res {cells?.resolution}
            </div>
            {extruded && (
              <div className="muted">
                {
                  cellHeightSource.heightFor(
                    valueOf(hoverCell.bucket, measure, isolated),
                  ).basis
                }
              </div>
            )}
          </div>
        )}

        {showCells && classing && cells?.cells?.length ? (
          <div className="map-legend map-cell-legend">
            <div className="map-legend-head">
              <span>
                {measure === "area" ? "Parcel area" : "Parcels"} per cell
              </span>
            </div>
            {/*  Bounds, not just swatches. The classes are quantiles over this
                 drawing at this resolution, so the same colour on another map
                 is not the same number — printing the range is what stops the
                 ramp being read as an absolute scale. */}
            {classing.colors.map((color, i) => (
              <div className="map-legend-row static" key={`${color}-${i}`}>
                <i style={{ background: color }} />
                <span>
                  {classing.lower[i] === classing.breaks[i]
                    ? classing.breaks[i].toLocaleString(undefined, {
                        maximumFractionDigits: 1,
                      })
                    : `${classing.lower[i].toLocaleString(undefined, { maximumFractionDigits: 1 })}–${classing.breaks[i].toLocaleString(undefined, { maximumFractionDigits: 1 })}`}
                </span>
              </div>
            ))}
          </div>
        ) : null}

        <div className="map-legend">
          <div className="map-legend-head">
            {/* What the SERVER said it measured, never what this component
                assumed. The heading used to read LAND USE whatever the
                payload was counting. */}
            <span title={cells?.dimension?.chosen_because ?? undefined}>
              {cells?.dimension?.label ?? "Land use"}
              {cells?.dimension?.is_fallback ? (
                <em className="legend-fallback"> · fallback</em>
              ) : null}
            </span>
            {isolated !== null && (
              <button className="linkish" onClick={() => setIsolated(null)}>
                show all
              </button>
            )}
          </div>
          {legend.map((entry) => (
            <button
              key={entry.use ?? "_none"}
              className={`map-legend-row${isolated === entry.use ? " on" : ""}`}
              onClick={() =>
                setIsolated(isolated === entry.use ? null : entry.use)
              }
              title={
                isolated === entry.use
                  ? "Show every land use again"
                  : `Show only ${entry.label}`
              }
            >
              <i style={{ background: entry.color }} />
              <span>{entry.label}</span>
              <b>{entry.count.toLocaleString()}</b>
            </button>
          ))}
          {cells?.dimension && cells.dimension.leaves_unexplained > 0 ? (
            <div className="map-legend-coverage">
              {cells.dimension.covers.toLocaleString()} of{" "}
              {cells.dimension.of_shapes_in_scope.toLocaleString()} shapes
              carry a {cells.dimension.label.toLowerCase()}
              {cells.dimension.is_fallback
                ? ""
                : " — the rest are on layers nobody has classified yet"}
            </div>
          ) : null}
        </div>

        {catchmentOpen && (
          <div className="map-catchment">
            <div className="map-legend-head">
              <span>Catchment</span>
              <button className="linkish" onClick={() => setCatchmentOpen(false)}>
                close
              </button>
            </div>

            <label className="map-catchment-row">
              <span>Around</span>
              <select
                value={catchmentSubject}
                onChange={(e) => setCatchmentSubject(e.target.value)}
              >
                <option value="">Choose…</option>
                {/* The selected object first, because if something is
                    selected it is almost always what the question is about. */}
                {selected && (
                  <option value={selected.handle}>
                    This object · {selected.handle}
                  </option>
                )}
                {/* Then every land use present, which is the form the
                    acceptance question takes: "…of EACH school". */}
                {legend
                  .filter((entry) => entry.use !== null)
                  .map((entry) => (
                    <option key={entry.use} value={entry.use!}>
                      Every {entry.label.toLowerCase()} parcel ({entry.count})
                    </option>
                  ))}
              </select>
            </label>

            <label className="map-catchment-row">
              <span>Rings</span>
              <select
                value={catchmentK}
                onChange={(e) => setCatchmentK(Number(e.target.value))}
              >
                {[1, 2, 3, 4, 5].map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </label>

            <div className="map-catchment-actions">
              <button
                className="toolbar-btn accent"
                disabled={!catchmentSubject || catchmentLoading}
                onClick={() => runCatchment(catchmentSubject, catchmentK)}
              >
                {catchmentLoading ? "Counting…" : "Draw it"}
              </button>
              {catchment && (
                <button className="toolbar-btn" onClick={clearCatchment}>
                  Clear
                </button>
              )}
            </div>

            {catchmentError && <p className="map-catchment-err">{catchmentError}</p>}

            {catchment && (
              <div className="map-catchment-result">
                {/*  The reach the recipe ACHIEVED, both numbers, always. One
                     of them is what a reader will quote and the other is why
                     the first one is approximate; printing only the first
                     turns a hexagon into a circle. */}
                <p className="map-catchment-reach">
                  {catchment.subjects.length} subject
                  {catchment.subjects.length === 1 ? "" : "s"} ·{" "}
                  {catchment.method.rings} ring
                  {catchment.method.rings === 1 ? "" : "s"} at resolution{" "}
                  {catchment.method.resolution} reaches{" "}
                  <b>{catchment.method.reach_centre_to_centre_m} m</b> centre to
                  centre and {catchment.method.reach_to_disk_corner_m} m to the
                  disk&rsquo;s corners.
                </p>

                <table className="map-catchment-table">
                  <tbody>
                    {Object.entries(totalCounts(catchment))
                      .sort((a, b) => b[1] - a[1])
                      .map(([use, n]) => (
                        <tr key={use}>
                          <td>
                            <i style={{ background: colorForUse(use) }} />
                            {labelForUse(use)}
                          </td>
                          <td>
                            <b>{n.toLocaleString()}</b>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>

                {/*  Said plainly rather than left to be discovered: with more
                     than one subject these are per-subject counts added up, so
                     a parcel near two schools is in the total twice. It is the
                     right total for the question asked per school and the
                     wrong one for "near any school". */}
                {catchment.subjects.length > 1 && (
                  <p className="map-catchment-note">
                    Counted per subject and added up. Two subjects close
                    together share neighbours, and a parcel in both disks is in
                    this total twice.
                  </p>
                )}
                <p className="map-catchment-note">{catchment.caveat}</p>
              </div>
            )}
          </div>
        )}

        {notes.length > 0 && (
          <details
            className={`map-notes${catchmentOpen ? " beside-catchment" : ""}`}
          >
            <summary>{notes.length} note{notes.length === 1 ? "" : "s"} about this view</summary>
            {notes.map((note, i) => (
              <p key={i}>{note}</p>
            ))}
          </details>
        )}

        {georeferenced && basemap !== "none" && (
          <div className="map-attrib">{BASEMAP_STYLES[basemap].attribution}</div>
        )}
      </div>
    </section>
  );
}
