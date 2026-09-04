"use client";

/**
 * The CAD viewer.
 *
 * The SVG comes from cad-api with `data-handle`, `data-layer` and `data-type`
 * on a `<g>` around every entity. This component never parses DXF; it only
 * manipulates that markup. Three consequences shape the code:
 *
 * - **One click listener on the container, not one per entity.** Janadriyah
 *   renders to ~20,000 groups; attaching a listener to each would stall the
 *   tab. `Element.closest("[data-handle]")` walks up from the clicked `<path>`
 *   to the group that carries the identity.
 *
 * - **Pan and zoom move the `viewBox`, not a CSS transform.** A CSS transform
 *   scales stroke width with the drawing, so lines fatten as you zoom in. The
 *   SVG ships `vector-effect: non-scaling-stroke`, which only behaves as
 *   intended under viewBox scaling.
 *
 * - **Layer visibility and selection are attribute writes, not re-renders.**
 *   The stylesheet inside the SVG turns `data-layer-hidden` into
 *   `display: none`. Hiding a layer with 8,000 entities costs one attribute
 *   set; doing it in React would cost 8,000 reconciliations.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  EXPORT_RESOLUTIONS,
  EXPORT_RESOLUTION_LABELS,
  SHOW_EXPORT_RES_PICKER,
} from "@/lib/api";
import {
  MARKER_DIAMETER_PX,
  MAX_ANSWER_MARKS,
  MAX_ZOOM_TO,
  markShapeFor,
  markerCentre,
  type AnswerHighlight,
} from "@/lib/answerHighlight";
import { isLabelType, type MarkPoint } from "@/lib/parcelContext";

export interface Selection {
  handle: string;
  layer: string;
  type: string;
}

interface ViewBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** The drawing -> viewBox affine the renderer wrote onto the `<svg>` root,
 *  in SVG's own `matrix(a b c d e f)` order. See `ATTR_WORLD_TO_SVG` in
 *  `cad_api/app/render.py`. */
interface WorldTransform {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
}

/** A region while the user is still drawing it. Vertices are already in
 *  **drawing coordinates** — the conversion happens at the moment of the
 *  gesture, never later, so nothing downstream ever holds a pixel. */
interface RegionDraft {
  kind: "rect" | "polygon" | "circle";
  /** rect: [start, current]. polygon: every placed vertex.
   *  circle: [centre] — the rim is wherever `cursor` is. */
  points: [number, number][];
  /** The live cursor, so the shape follows the mouse. */
  cursor: [number, number] | null;
  /** Screen point where the gesture began — the sign of the drag decides
   *  window vs crossing, and that is a screen-direction question. */
  startScreenX: number;
  startScreenY: number;
  /** True once the pointer has travelled far enough to mean a drag. Without
   *  it a Shift+click that never moved was committed as a zero-area region. */
  movedScreen: boolean;
  /** Updated live while dragging, so the box is already blue or green before
   *  the user lets go — the same feedback AutoCAD gives. */
  mode: "window" | "crossing";
}

/** A committed circle mid-reshape. */
interface CircleEdit {
  /** Position in `regions`. */
  index: number;
  /** Dragging the centre moves the circle; dragging the rim resizes it. */
  grip: "move" | "resize";
  centre: [number, number];
  rim: [number, number];
  mode: "window" | "crossing";
  /** True once the pointer has travelled far enough to mean a drag, so a bare
   *  click on a grip does not re-ask the region for nothing. */
  moved: boolean;
}

interface Props {
  svg: string | null;
  loading: boolean;
  hiddenLayers: Set<string>;
  commentedHandles: Set<string>;
  selected: Selection | null;
  onSelect: (selection: Selection | null) => void;
  onLayersDiscovered: (layers: { name: string; count: number }[]) => void;
  /** Layers the FILE keeps switched off/frozen; the drawing starts with them
   *  hidden, matching AutoCAD, but they stay toggleable and hoverable. */
  onOffLayersDiscovered: (names: string[]) => void;
  panelOpen: boolean;
  onTogglePanel: () => void;
  /** handle -> comments, for the hover popup. */
  commentsByHandle: Record<string, { author: string; body: string }[]>;
  /** Download URL for the derived DXF that carries the comments in-file. */
  exportUrl: string | null;
  /** Download URL for this drawing's GeoJSON, or `null` when the drawing has
   *  no coordinate system.
   *
   *  A separate prop from `exportUrl` because the two answer different
   *  questions. The DWG/DXF exports appear once there is a comment to carry;
   *  this one appears once the drawing has a position on the earth, and
   *  seventeen of the eighteen drawings here never will. */
  geoExportUrl: string | null;
  /** Why there is no GeoJSON to download, when there is none. Shown on a
   *  disabled control rather than left as an absence: a button that is simply
   *  missing reads as a feature that is broken. */
  geoExportReason: string | null;
  /** Cell size for the drawing export, and the way to change it. Owned by
   *  the page rather than here: the viewer draws the toolbar, the page owns
   *  what the URL says. */
  geoExportRes: number;
  onGeoExportRes: (res: number) => void;

  // --- region selection ----------------------------------------------------
  /** "polygon" turns clicks into vertex placement. A rectangle needs no mode:
   *  it is Shift+drag, so plain drag keeps meaning pan. */
  regionTool: "none" | "polygon" | "circle";
  onRegionToolChange: (tool: "none" | "polygon" | "circle") => void;
  /** Every committed region, in drawing coordinates, drawn over the SVG.
   *  A selection can be made of several, so this is a list — an empty one
   *  when the selection is a click, or nothing at all. */
  regions: { kind: "rect" | "polygon" | "circle"; mode: "window" | "crossing"; points: [number, number][] }[];
  onRegionDrawn: (
    points: [number, number][],
    pointsView: [number, number][],
    kind: "rect" | "polygon" | "circle",
    mode: "window" | "crossing",
  ) => void;
  /** A committed circle was reshaped by its grips. `index` is its position
   *  in `regions`, so the caller re-asks that part in place rather than
   *  adding another one. */
  onRegionEdit: (
    index: number,
    points: [number, number][],
    pointsView: [number, number][],
    mode: "window" | "crossing",
  ) => void;
  onRegionClear: () => void;
  /** Handles the region matched, for highlighting. Null when none is active. */
  regionHandles: Set<string> | null;
  /** Whether this render carries the drawing->viewBox transform. Without it
   *  region selection is impossible, and saying so beats a tool that silently
   *  selects nothing. */
  onRegionSupport: (supported: boolean) => void;
  /** False when this layout cannot support a region at all — no transform, or
   *  a sheet whose geometry belongs to another coordinate space. The gestures
   *  are then inert, because a region that quietly matches the wrong objects
   *  is worse than one that cannot be drawn. */
  regionEnabled: boolean;
  /** Called when the user clicks the region button WHILE it is blocked.
   *  Opens the explanation instead of doing nothing: a `disabled` button
   *  cannot be clicked, and in the field that read as "the feature is broken"
   *  — the user stared at a dead control while the reason, and the one-click
   *  way out, sat unread in a tab they had no cue to open. */
  onRegionBlockedHelp: () => void;

  // --- answer highlight ----------------------------------------------------
  /** Objects the agent's latest answer named, already validated against this
   *  layout. Null when there is no answer to show or the user cleared it.
   *
   *  This is the feature asked for three times on 23 August 2026 and missing
   *  all three times. It is a separate channel from `regionHandles` on
   *  purpose: the user must always be able to tell what they selected from
   *  what the agent answered, and one set of marks in one colour cannot say
   *  both. */
  answerHighlight: AnswerHighlight | null;
  /** handle -> the layer it was read for, so marks can be coloured by kind.
   *
   *  One colour for the whole answer says "these fifteen things matched" and
   *  stops there. Asked where the schools and the mosques are, that is half
   *  the answer withheld: the reader can see the count and not which of them
   *  is which. Empty map means one colour, which is the right picture when
   *  everything marked is the same kind of thing. */
  answerGroups: Map<string, string>;
  /** Stored centroids for the highlighted handles, when cad-api can supply
   *  them. Null while they are loading or when the route does not exist yet;
   *  the marks degrade rather than disappear. See lib/parcelContext.ts. */
  answerPoints: MarkPoint[] | null;
  /** Why the centroids are missing, if they are. Shown on the badge rather
   *  than swallowed, so a coarser mark says so. */
  answerPointsNote: string | null;
  onClearAnswerHighlight: () => void;
}

/** One circle as path data, so many of them can share a single node.
 *
 *  Two arcs rather than four: an SVG arc cannot span a full turn in one
 *  command, and two half-turns is the shortest form that closes.
 */
function circlePath(cx: number, cy: number, r: number): string {
  return (
    `M ${cx - r} ${cy}` +
    `A ${r} ${r} 0 1 0 ${cx + r} ${cy}` +
    `A ${r} ${r} 0 1 0 ${cx - r} ${cy}Z`
  );
}

/** Zoom step per wheel notch. 1.15 is fast enough to cross a site plan in a
 *  few flicks without overshooting a door handle. */
const ZOOM_STEP = 1.15;
const MIN_SPAN = 1e-4;

/** Objects individually highlighted before the viewer stops marking them and
 *  lets the region outline speak for the selection. See the highlight effect. */
const MAX_HIGHLIGHTED = 750;

/** How many distinct mark colours the stylesheet defines.
 *
 *  Six, and the seventh kind reuses the first. A palette that grows without
 *  limit stops being readable long before it stops being possible: past half
 *  a dozen hues on a drawing that is already coloured, "which one is this"
 *  costs more than the grouping saves. Answers that name more than six layers
 *  are answers about a whole site rather than about a comparison.
 */
const MARK_COLOURS = 6;

/** Radius, in screen pixels, within which a press counts as grabbing a grip.
 *  Larger than the dot it draws: a target you can only hit dead-centre reads
 *  as a broken control rather than a small one. */
const GRIP_GRAB_PX = 11;
/** Drawn size of a grip dot, in screen pixels. */
const GRIP_DRAW_PX = 5;

/** How far from a line a click may land and still select it, in SCREEN
 *  pixels.
 *
 *  Pixels, not drawing units, and that is the correction: a tolerance in
 *  drawing units is a different tolerance at every magnification, so it is
 *  either useless when zoomed in or a blunt instrument when zoomed out. Six
 *  pixels is six pixels at 100% and at 4000%.
 */
const HIT_TOLERANCE_PX = 6;

/** Budget for one hover hit-test, in milliseconds.
 *
 *  The hover preview walks the full element stack so that it agrees with what
 *  a click would select, and that walk is the expensive call in this file:
 *  `elementsFromPoint` measured **6.95 ms** on the reference sheet against
 *  0.16 ms for the singular `elementFromPoint`. One of those per frame is
 *  affordable; two are not, and on a bigger document tomorrow one may not be
 *  either.
 *
 *  So it is measured rather than assumed. The viewer times its own hover
 *  probes and switches the preview off if the median exceeds this budget,
 *  saying so in the toolbar. Selection itself is untouched — a click can
 *  afford one 7 ms walk, because there is one of them per click.
 */
const HOVER_BUDGET_MS = 9;
/** Probes timed before the budget is judged. Enough to see past the first
 *  call, which pays for style resolution the rest of them reuse. */
const HOVER_PROBE_SAMPLE = 8;

export function Viewer({
  svg,
  loading,
  hiddenLayers,
  commentedHandles,
  selected,
  onSelect,
  onLayersDiscovered,
  onOffLayersDiscovered,
  panelOpen,
  onTogglePanel,
  commentsByHandle,
  exportUrl,
  geoExportUrl,
  geoExportReason,
  geoExportRes,
  onGeoExportRes,
  regionTool,
  onRegionToolChange,
  regions,
  onRegionDrawn,
  onRegionEdit,
  onRegionClear,
  regionHandles,
  onRegionSupport,
  regionEnabled,
  onRegionBlockedHelp,
  answerHighlight,
  answerGroups,
  answerPoints,
  answerPointsNote,
  onClearAnswerHighlight,
}: Props) {
  const [hover, setHover] = useState<{
    x: number;
    y: number;
    handle: string;
    comments: { author: string; body: string }[];
  } | null>(null);
  /** The entity a click would select right now.
   *
   *  Shown before the click, because the alternative is a user who finds out
   *  what they hit by hitting it. On dense linework where several objects
   *  overlap that is not a nicety — the whole complaint at 23:28 on 23 August
   *  was that a click selected something other than what was meant, and a
   *  preview is how a person notices that before it costs them anything.
   */
  const [hoverTarget, setHoverTarget] = useState<string | null>(null);
  /** Set once the measured cost of the hover walk exceeds its budget. The
   *  preview then stops and the toolbar says so, rather than the whole viewer
   *  quietly becoming sluggish. */
  const [hoverDegraded, setHoverDegraded] = useState(false);
  const hoverTimings = useRef<number[]>([]);
  const hoverFrame = useRef<number | null>(null);
  /** How many objects the answer overlay could not place a marker for,
   *  because nothing could say where their centre is. Reported on the badge:
   *  a mark that is not drawn has to be a number somewhere. */
  const [unplaced, setUnplaced] = useState(0);
  const hostRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const baseViewBox = useRef<ViewBox | null>(null);
  const [viewBox, setViewBox] = useState<ViewBox | null>(null);
  const panState = useRef<{
    x: number;
    y: number;
    vb: ViewBox;
    moved: boolean;
    /** Live drag offset in CSS pixels, applied as a transform until release. */
    dx: number;
    dy: number;
  } | null>(null);
  /** Pending rAF for the pan transform, so a burst of pointer events costs
   *  one style write per frame rather than one per event. */
  const panFrame = useRef<number | null>(null);
  // The element actually under the pointer when the gesture started.
  //
  // `click` cannot be trusted for hit-testing here: `onPointerDown` calls
  // setPointerCapture so that a drag continues when the cursor leaves the
  // canvas, and while capture is held the browser retargets the compatibility
  // click to the capture element. `event.target` in onClick is therefore the
  // container div, never the <path> the user aimed at, and closest() finds no
  // handle. Recorded here instead.
  const downTarget = useRef<Element | null>(null);
  /** Where the press landed, in client pixels.
   *
   *  The hit test is run against this rather than against the click event,
   *  for the same reason `downTarget` was recorded before it: by the time the
   *  click arrives the pointer may have moved a pixel or two, and on dense
   *  linework a pixel or two is a different object. */
  const pressX = useRef(0);
  const pressY = useRef(0);
  const [isPanning, setIsPanning] = useState(false);

  // --- region selection state ---------------------------------------------
  // The affine the renderer put on the <svg> root, and the <g> the overlay is
  // drawn into. Both are refs rather than state: they belong to the injected
  // DOM, which this component owns imperatively (see the file header).
  const worldToSvg = useRef<WorldTransform | null>(null);
  const overlay = useRef<SVGGElement | null>(null);
  /** A SECOND overlay, for the marks that answer a question.
   *
   *  Separate from the region overlay because the two are rebuilt by
   *  different effects on different triggers: the region redraws when a shape
   *  changes, the answer marks redraw on every zoom because they are sized in
   *  screen pixels. Sharing one group would mean each effect wiping the
   *  other's work, which is the kind of bug that only shows up when a user
   *  does both things at once.
   */
  const answerOverlay = useRef<SVGGElement | null>(null);
  const [draft, setDraft] = useState<RegionDraft | null>(null);
  /** A committed circle being reshaped by its grips. Held separately from
   *  `draft`, which is a shape being CREATED: the two look similar on screen
   *  and behave differently on release, and merging them would mean every
   *  branch below asking which one it really was. */
  const [edit, setEdit] = useState<CircleEdit | null>(null);
  /** The ref is the SOURCE OF TRUTH for the pointer handlers; the state above
   *  exists so the overlay re-renders.
   *
   *  Assigning the ref during render instead — `editRef.current = edit` — is
   *  the obvious version and it has a race: React batches, so a pointerup
   *  arriving before the re-render reads the value from before the gesture
   *  started, and the reshape is silently dropped. `panState` next to it is a
   *  plain ref for exactly this reason. Written in the handlers, both are
   *  correct however the events happen to be scheduled. */
  const editRef = useRef<CircleEdit | null>(null);
  /** Set both at once so they cannot disagree. */
  const applyEdit = useCallback((next: CircleEdit | null) => {
    editRef.current = next;
    setEdit(next);
  }, []);
  /** Which grip the pointer is hovering, so the cursor can say so before the
   *  user commits to a drag. A mode you cannot see is a mode you cannot
   *  trust — the same reason Space shows a hand (D-049). */
  const [overGrip, setOverGrip] = useState<"move" | "resize" | null>(null);
  // Read inside pointer handlers that must not be re-created on every draft
  // change; a stale closure here would freeze the rubber band mid-drag.
  /** Same arrangement as `editRef` below, and for the same reason: the ref is
   *  what the pointer handlers read, the state is what makes the overlay
   *  re-render. Assigning the ref during render left a window where a
   *  pointerup that arrived before React re-rendered saw the value from
   *  before the gesture began, and threw the shape away without a word. */
  const draftRef = useRef<RegionDraft | null>(null);
  const applyDraft = useCallback(
    (next: RegionDraft | null | ((current: RegionDraft | null) => RegionDraft | null)) => {
      const value =
        typeof next === "function"
          ? (next as (c: RegionDraft | null) => RegionDraft | null)(draftRef.current)
          : next;
      draftRef.current = value;
      setDraft(value);
    },
    [],
  );
  /** Space held = pan, while the polygon tool is armed.
   *
   *  Field report, verbatim problem: with the tool armed, click placed a
   *  vertex but drag STILL panned, so drawing and navigating shared the same
   *  hand movements and users could not tell which one they were about to
   *  do. Drawing is now modal: a plain drag does nothing, and panning is an
   *  explicit chord — hold Space and drag — the way image editors do it.
   *  Wheel zoom stays, because a scroll is never ambiguous with a click. */
  const spaceHeld = useRef(false);
  /** Mirrors `spaceHeld` for rendering. The ref is what the pointer handlers
   *  read (no stale closures); this is what the cursor reads. A mode the user
   *  cannot see is a mode they will not trust — holding Space must LOOK like
   *  grabbing before they drag. */
  const [spacePanReady, setSpacePanReady] = useState(false);

  /** Set when a gesture has already been consumed by the region tools, so the
   *  browser's trailing compatibility click does not also hit-test entities.
   *
   *  Cleared on a timer rather than by the next click, and that difference is
   *  the whole point: the compatibility click arrives immediately after
   *  pointerup, so a zero-delay timeout is late enough to catch it and early
   *  enough that nothing else can. Latching until "the next click on the
   *  canvas" instead meant that drawing a region and then touching anything
   *  else first — a panel button, a tab — left the flag armed, and the user's
   *  next attempt to click an entity was swallowed with no sign. This mirrors
   *  how `panState` is torn down a few lines below. */
  const suppressNextClick = useRef(false);

  // --- inject the SVG ------------------------------------------------------
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    if (!svg) {
      host.innerHTML = "";
      svgRef.current = null;
      baseViewBox.current = null;
      setViewBox(null);
      return;
    }

    host.innerHTML = svg;
    const element = host.querySelector("svg");
    if (!element) return;

    // The API emits a page-sized SVG (width="480mm"). Strip the physical
    // dimensions so it fills the pane and scales with the container.
    element.removeAttribute("width");
    element.removeAttribute("height");
    element.setAttribute("width", "100%");
    element.setAttribute("height", "100%");
    element.style.display = "block";
    element.style.touchAction = "none";

    const raw = element.getAttribute("viewBox");
    const parts = raw ? raw.split(/[\s,]+/).map(Number) : null;
    const box: ViewBox =
      parts && parts.length === 4 && parts.every((n) => Number.isFinite(n))
        ? { x: parts[0], y: parts[1], w: parts[2], h: parts[3] }
        : { x: 0, y: 0, w: 1000, h: 1000 };

    svgRef.current = element as SVGSVGElement;
    baseViewBox.current = box;
    setViewBox(box);

    // Counted from the rendered markup, not from the drawing document. The
    // document's per-layer totals are for the whole file; on a sheet they
    // describe a different population than the checkboxes control, so a layer
    // could read "3" while hiding hundreds of objects.
    const layers = Array.from(element.querySelectorAll("[data-layer]")).map(
      (node) => ({
        name: node.getAttribute("data-layer") ?? "",
        count: node.querySelectorAll("[data-handle]").length,
      }),
    );
    onLayersDiscovered(
      layers.filter((l) => l.name).sort((a, b) => b.count - a.count),
    );

    // Layers the source file keeps OFF/frozen (see render.py): start hidden.
    const offAttr = element.getAttribute("data-off-layers") ?? "";
    onOffLayersDiscovered(offAttr ? offAttr.split(",") : []);

    // --- make closed shapes clickable through their middle ----------------
    //
    // The complaint, verbatim, at 19:44 on 23 August: *"why you are able to
    // click schools?"* — because a school parcel is 2,749 m² and a house plot
    // is 12 x 25 m. That was never the real difference. The real difference is
    // that a browser hit-tests an SVG path on its PAINT, and these paths are
    // painted as a 1-2 px stroke with no fill, so every entity in this drawing
    // is a target the width of its own outline. A big parcel simply has more
    // outline to hit. Zoomed out, a small plot's outline is a few pixels of
    // line and the click lands on paper.
    //
    // AutoCAD does not behave that way and neither should this: clicking
    // inside a closed shape selects it. `pointer-events: visible` asks the
    // browser to hit-test the fill REGION whether or not the fill is painted,
    // which turns the interior into the target without changing a single
    // pixel of the drawing.
    //
    // It is applied only to paths that actually close. That restriction is
    // the whole safety of the change: give it to an open path and the browser
    // still has a fill region for it — the implied closure — so a road
    // centreline running the width of the site would become a click target
    // the size of the site, swallowing everything under it. Measured on this
    // store, open polygons are not an edge case: 3,090 of 7,877 rings are
    // open and 1,298 more carry arcs.
    //
    // Which paths close is read from the path data rather than assumed from
    // the entity type, because the type does not know: an LWPOLYLINE may or
    // may not carry the closed flag, and the renderer is the only thing that
    // has already decided. One pass over the paths, once per layout, next to
    // a render that takes seconds.
    let closedPaths = 0;
    element.querySelectorAll("[data-handle] path").forEach((node) => {
      const d = node.getAttribute("d");
      if (!d) return;
      // Scanned backwards rather than trimmed, to avoid allocating a second
      // copy of every path string in a document that holds tens of thousands
      // of them, some with over a thousand vertices.
      let i = d.length - 1;
      while (i >= 0 && (d[i] === " " || d[i] === "\n" || d[i] === "\t")) i -= 1;
      if (i >= 0 && (d[i] === "Z" || d[i] === "z")) {
        node.setAttribute("data-closed", "true");
        closedPaths += 1;
      }
    });
    // Nothing closed means nothing gained, and it also means the assumption
    // behind the pass is wrong for this renderer. Recorded on the root so the
    // condition is visible in devtools instead of being invisible in a
    // feature that merely does not work.
    element.setAttribute("data-closed-paths", String(closedPaths));

    // --- region selection scaffolding -------------------------------------
    worldToSvg.current = parseWorldTransform(
      element.getAttribute("data-world-to-svg"),
    );
    onRegionSupport(worldToSvg.current !== null);

    // Region styling is injected here rather than baked into the SVG by the
    // renderer, and that is a deliberate split: the drawing's own stylesheet
    // describes the drawing, while highlight colours are a decision of this
    // viewer. Baking them in would mean re-rendering 368 cached layouts to
    // change a shade of purple.
    const style = document.createElementNS(SVG_NS, "style");
    style.textContent = REGION_CSS;
    element.appendChild(style);

    // The overlay lives *inside* the SVG, in viewBox coordinates. That is the
    // whole trick: pan and zoom move the viewBox, so the region tracks the
    // drawing with no JavaScript at all and cannot drift out of step with it.
    const group = document.createElementNS(SVG_NS, "g");
    group.setAttribute("data-region-overlay", "true");
    // Never let the region steal a click meant for an entity underneath.
    group.setAttribute("pointer-events", "none");
    element.appendChild(group);
    overlay.current = group;

    // The answer marks go last, so they sit above the region outline. When
    // both are on screen the agent's answer is the newer information and the
    // one the user is looking for.
    const marks = document.createElementNS(SVG_NS, "g");
    marks.setAttribute("data-answer-overlay", "true");
    marks.setAttribute("pointer-events", "none");
    element.appendChild(marks);
    answerOverlay.current = marks;

    // A new document is a new performance question. The hover budget was
    // measured against the PREVIOUS layout, and carrying that verdict over
    // would either punish a small drawing for a big one or clear a big one on
    // a small one's evidence.
    hoverTimings.current = [];
    setHoverDegraded(false);
    setHoverTarget(null);
  }, [svg, onLayersDiscovered, onOffLayersDiscovered, onRegionSupport]);

  // A drag interrupted by an unmount (drawing switch, navigation) would
  // otherwise leave a frame callback holding a reference to a dead element.
  useEffect(
    () => () => {
      if (panFrame.current !== null) cancelAnimationFrame(panFrame.current);
    },
    [],
  );

  // --- apply viewBox -------------------------------------------------------
  useEffect(() => {
    if (svgRef.current && viewBox) {
      svgRef.current.setAttribute(
        "viewBox",
        `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`,
      );
    }
  }, [viewBox]);

  // --- layer visibility ----------------------------------------------------
  useEffect(() => {
    const element = svgRef.current;
    if (!element) return;
    element.querySelectorAll("[data-layer]").forEach((node) => {
      const layer = node.getAttribute("data-layer") ?? "";
      if (hiddenLayers.has(layer)) {
        node.setAttribute("data-layer-hidden", "true");
      } else {
        node.removeAttribute("data-layer-hidden");
      }
    });
  }, [hiddenLayers, svg]);

  // --- selection highlight -------------------------------------------------
  useEffect(() => {
    const element = svgRef.current;
    if (!element) return;
    element
      .querySelectorAll("[data-selected]")
      .forEach((node) => node.removeAttribute("data-selected"));
    if (selected) {
      const node = element.querySelector(
        `[data-handle="${cssEscape(selected.handle)}"]`,
      );
      node?.setAttribute("data-selected", "true");
    }
  }, [selected, svg]);

  // --- comment markers -----------------------------------------------------
  useEffect(() => {
    const element = svgRef.current;
    if (!element) return;
    element
      .querySelectorAll("[data-commented]")
      .forEach((node) => node.removeAttribute("data-commented"));
    commentedHandles.forEach((handle) => {
      element
        .querySelector(`[data-handle="${cssEscape(handle)}"]`)
        ?.setAttribute("data-commented", "true");
    });
  }, [commentedHandles, svg]);

  // --- commented-entity hover zones ---------------------------------------
  // The commented entity is often a small label sitting on top of (or under)
  // a big hatch, so elementFromPoint hands hover to the hatch and the popup
  // never appears — measured on Janadriyah: comments on TEXT 86DD3 were
  // unreachable through HATCH 2CA45AD covering the same block. Two fixes,
  // applied in the hover handler below: the full elementsFromPoint STACK is
  // searched for a commented entity (not just the topmost hit), and each
  // commented entity's screen rect is kept here so hovering NEAR one (like
  // TrueView's redline neighbourhood) still pops the comment.
  // The commented entities' NODES are cached; their positions are not.
  //
  // That split is the whole lesson from the bug this replaced. Which elements
  // carry comments changes only when the drawing or the comment set changes,
  // so looking them up is worth caching -- `querySelector` over a 20,330-node
  // SVG is the expensive half. Where those elements are on screen changes on
  // every pan and zoom, so it must never be cached; it is read live below.
  const commentedNodes = useRef<{ handle: string; node: Element }[]>([]);
  useEffect(() => {
    const element = svgRef.current;
    if (!element) {
      commentedNodes.current = [];
      return;
    }
    const found: { handle: string; node: Element }[] = [];
    for (const handle of Object.keys(commentsByHandle)) {
      if ((commentsByHandle[handle]?.length ?? 0) === 0) continue;
      const node = element.querySelector(`[data-handle="${cssEscape(handle)}"]`);
      if (node) found.push({ handle, node });
    }
    commentedNodes.current = found;
  }, [commentsByHandle, svg]);

  // No cached rectangle list any more, deliberately.
  //
  // There used to be one, refreshed by an effect keyed on the viewBox. It
  // worked only because it was a *fallback*: the expensive stack walk ran
  // first and did the real work, so a stale cache was invisible. Turning that
  // cache into the primary test for performance made its staleness fatal --
  // every hover went dead at once. Screen positions are derived state; the
  // DOM already holds them, and there are only ever a handful of commented
  // entities, so they are read fresh below instead of mirrored here.

  /** The commented entity at (or within `pad` px of) a screen point, looking
   *  through overlapping entities instead of stopping at the topmost one. */
  const commentedHandleNear = useCallback(
    (x: number, y: number, pad: number): string | null => {
      // Cheap gate, measured against a real drawing.
      //
      // `elementsFromPoint` returns the entire stack under the cursor, and on
      // Janadriyah's modelspace -- 20,330 entity groups -- that costs **6.95
      // ms per call**, against 0.16 ms for the singular `elementFromPoint`.
      // It used to run on every pointermove, so moving the mouse spent ~40%
      // of a 60 Hz frame before anything was drawn. Tolerable while nothing
      // tracked the cursor; not once a rubber band does.
      //
      // The gate is exact rather than approximate: an element's bounding rect
      // always contains any point that hit-tests to it, so if no commented
      // entity's padded rect contains the point, the stack cannot contain a
      // commented entity either.
      //
      // Rects are read live from the cached nodes. A drawing has a handful of
      // commented entities (this project's whole database has 11), so this is
      // a few getBoundingClientRect calls on a clean layout -- and unlike a
      // cached rectangle list it cannot go stale behind a pan or a zoom.
      let nearest: string | null = null;
      for (const { handle, node } of commentedNodes.current) {
        const rect = node.getBoundingClientRect();
        // A hidden layer collapses to an empty rect; nothing to hover.
        if (rect.width === 0 && rect.height === 0) continue;
        if (
          x >= rect.left - pad && x <= rect.right + pad &&
          y >= rect.top - pad && y <= rect.bottom + pad
        ) {
          nearest = handle;
          break;
        }
      }
      if (nearest === null) return null;

      // Only now is the stack worth walking: commented entities can overlap,
      // and the one actually under the cursor beats the one whose padded rect
      // happened to match first.
      for (const el of document.elementsFromPoint(x, y)) {
        const handle = el
          .closest?.("[data-handle]")
          ?.getAttribute("data-handle");
        if (handle && (commentsByHandle[handle]?.length ?? 0) > 0) {
          return handle;
        }
      }
      return nearest;
    },
    [commentsByHandle],
  );

  /** Find the entity group at or near a screen point.
   *
   * CAD linework draws 1-2 px wide, and the browser's hit target is exactly
   * the stroke -- so a click had to land on the line to the pixel, which
   * feels broken ("the cursor is always the pan hand"). If nothing is hit
   * dead-on, a small ring of offsets is sampled with elementFromPoint, so a
   * click within ~8 px of a line still selects it. Empty paper stays
   * clickable-to-deselect because the samples find no [data-handle] there.
   */
  /** The entity a click at this point should select.
   *
   *  Making closed shapes clickable through their middle solved one problem
   *  and created another: parcels nest. A plot sits inside a block boundary
   *  which sits inside a site boundary, so a point in the middle of a plot is
   *  now inside three shapes at once, and the browser's answer — the topmost
   *  painted element — is whichever the renderer happened to draw last. That
   *  is not a decision, it is an accident of document order.
   *
   *  Two rules resolve it, in this order.
   *
   *  **A label on its glyph wins.** If the point is literally on the ink of a
   *  piece of text, the text is what was clicked. This costs nothing to
   *  detect: a glyph's hit region IS its letter shape, so a text entity turns
   *  up in the stack only when the pointer is on the ink and not merely
   *  inside the word's box.
   *
   *  **Otherwise the smallest shape wins.** The smallest shape containing a
   *  point is the most specific thing there — the plot rather than the block,
   *  the block rather than the site. It is measured, not assumed: no
   *  vertex count, no axis, no plot module, nothing about what the shape
   *  looks like (G6). A boundary that happens to be smaller than the parcel
   *  inside it would win, and would be the right answer if it did.
   *
   *  The size used is the on-screen bounding box, which is a proxy for area
   *  and not area itself. It is the cheap measurement the DOM already holds,
   *  and it only ever ORDERS candidates — it never rejects one, so a shape
   *  whose box flatters it still competes rather than disappearing.
   */
  const resolveEntityAt = useCallback(
    (x: number, y: number): Element | null => {
      const stack = document.elementsFromPoint(x, y);
      let best: Element | null = null;
      let bestArea = Infinity;
      const seen = new Set<string>();

      for (const element of stack) {
        // The overlays are pointer-events:none and never appear here, but the
        // page chrome around the canvas does; stop at the first thing that is
        // not part of the drawing.
        const group = element.closest?.("[data-handle]");
        if (!group) continue;
        const handle = group.getAttribute("data-handle") ?? "";
        if (seen.has(handle)) continue;
        seen.add(handle);

        if (isLabelType(group.getAttribute("data-type"))) return group;

        let area = Infinity;
        try {
          const box = (group as SVGGraphicsElement).getBBox();
          area = box.width * box.height;
        } catch {
          /* not rendered; it keeps an infinite area and loses to anything */
        }
        // `<=` rather than `<`: among equals the one drawn later wins, which
        // is what the browser would have said anyway and what a user reading
        // the picture would expect from two shapes on top of each other.
        if (area <= bestArea) {
          best = group;
          bestArea = area;
        }
      }
      return best;
    },
    [],
  );

  /** The same answer, with the near-miss fallback for thin linework.
   *
   *  An open polyline is still only as clickable as its stroke, so the ring of
   *  samples stays. It is walked with the CHEAP hit-test, though, and the full
   *  resolution runs only at a sample that found something: the ring is
   *  seventeen probes, and seventeen stack walks at 6.95 ms each would put a
   *  visible stutter on the most common gesture of all — clicking empty paper
   *  to deselect, which is exactly the case that reaches the end of the ring.
   */
  const pickEntityAt = useCallback(
    (x: number, y: number): Element | null => {
      const direct = resolveEntityAt(x, y);
      if (direct) return direct;
      for (const radius of [3, HIT_TOLERANCE_PX]) {
        for (const [dx, dy] of [
          [radius, 0], [-radius, 0], [0, radius], [0, -radius],
          [radius, radius], [-radius, -radius], [radius, -radius], [-radius, radius],
        ]) {
          const cheap = document
            .elementFromPoint(x + dx, y + dy)
            ?.closest("[data-handle]");
          if (cheap) return resolveEntityAt(x + dx, y + dy) ?? cheap;
        }
      }
      return null;
    },
    [resolveEntityAt],
  );

  /** Show what a click here would select, before the click.
   *
   *  Throttled to one probe per animation frame: pointermove fires faster
   *  than the screen refreshes, and a hit-test per event would pay several
   *  times over for a picture that can only be drawn once.
   *
   *  And then measured, because a throttle is not a guarantee. The stack walk
   *  cost 6.95 ms on the reference sheet, which fits inside a 16 ms frame with
   *  room for the pan transform and nothing else; a denser drawing arriving
   *  tomorrow would not fit, and the symptom would be a viewer that feels
   *  broken with no indication why. So the probes are timed, and if their
   *  median passes `HOVER_BUDGET_MS` the preview switches itself off and the
   *  toolbar says it has. Clicking is unaffected, because a click pays this
   *  cost once rather than sixty times a second.
   */
  const previewHover = useCallback((x: number, y: number) => {
    if (hoverDegraded) return;
    if (hoverFrame.current !== null) return;
    hoverFrame.current = requestAnimationFrame(() => {
      hoverFrame.current = null;
      const started = performance.now();
      const group = resolveEntityAt(x, y);
      const elapsed = performance.now() - started;

      setHoverTarget(group?.getAttribute("data-handle") ?? null);

      const timings = hoverTimings.current;
      if (timings.length < HOVER_PROBE_SAMPLE) {
        timings.push(elapsed);
        if (timings.length === HOVER_PROBE_SAMPLE) {
          // Median, not mean: the first probe of a fresh document also pays
          // for style resolution the rest reuse, and one outlier should not
          // condemn a drawing that is comfortably fast.
          const sorted = [...timings].sort((a, b) => a - b);
          const median = sorted[Math.floor(sorted.length / 2)];
          if (median > HOVER_BUDGET_MS) {
            setHoverDegraded(true);
            setHoverTarget(null);
          }
        }
      }
    });
  }, [hoverDegraded, resolveEntityAt]);

  // A pending probe holding a reference into a document that has been
  // replaced would mark a handle from the previous drawing.
  useEffect(
    () => () => {
      if (hoverFrame.current !== null) cancelAnimationFrame(hoverFrame.current);
    },
    [],
  );

  // --- hover preview: paint it --------------------------------------------
  // One attribute on one node, the same mechanism as selection. Deliberately
  // not a React render: the marked node is inside the injected SVG, which
  // this component owns imperatively.
  useEffect(() => {
    const element = svgRef.current;
    if (!element) return;
    element
      .querySelectorAll("[data-hover-target]")
      .forEach((node) => node.removeAttribute("data-hover-target"));
    if (hoverTarget) {
      element
        .querySelector(`[data-handle="${cssEscape(hoverTarget)}"]`)
        ?.setAttribute("data-hover-target", "true");
    }
  }, [hoverTarget, svg]);

  // --- region: screen -> drawing coordinates -------------------------------
  /** A screen point in drawing coordinates, or null if this render has no
   *  transform.
   *
   *  Two steps, and the first is not the viewBox arithmetic used by pan and
   *  zoom above. That arithmetic assumes the viewBox fills the element, and it
   *  does not: the SVG has no `preserveAspectRatio`, so the browser applies
   *  the default `xMidYMid meet` and letterboxes. Measured on Janadriyah's
   *  modelspace in a 1200x900 pane, the drawing occupies 722 px of the 900 and
   *  sits centred — so the naive mapping is out by up to 11% in y, which on
   *  this drawing is several hundred metres. For panning that error is
   *  self-cancelling and invisible; for selection it would silently return the
   *  wrong objects. `getScreenCTM()` is the browser's own answer and is exact
   *  whatever the fitting. The existing pan/zoom maths is left untouched.
   */
  const screenToWorld = useCallback(
    (clientX: number, clientY: number): [number, number] | null => {
      const element = svgRef.current;
      const matrix = worldToSvg.current;
      if (!element || !matrix) return null;
      const ctm = element.getScreenCTM();
      if (!ctm) return null;
      const inViewBox = new DOMPoint(clientX, clientY).matrixTransform(
        ctm.inverse(),
      );
      return svgToWorld(matrix, inViewBox.x, inViewBox.y);
    },
    [],
  );

  /** How many drawing units one screen pixel currently spans.
   *
   *  Measured by pushing two points one pixel apart back through
   *  `screenToWorld` rather than read off the transform, so it stays right
   *  under pan, zoom, and the letterboxing `preserveAspectRatio` introduces.
   *  It is what lets a grab tolerance be stated once, in pixels, and mean the
   *  same thing at every magnification.
   */
  const worldPerPixel = useCallback(
    (clientX: number, clientY: number): number | null => {
      const here = screenToWorld(clientX, clientY);
      const overOne = screenToWorld(clientX + 1, clientY);
      if (!here || !overOne) return null;
      const d = Math.hypot(overOne[0] - here[0], overOne[1] - here[1]);
      return d > 0 ? d : null;
    },
    [screenToWorld],
  );

  /** Which grip of which circle the pointer is on, if any.
   *
   *  Distance maths rather than a DOM hit-test, because the overlay is
   *  deliberately `pointer-events: none` so it can never steal a click meant
   *  for an entity underneath. Grips must not cost us that.
   */
  const gripAt = useCallback(
    (
      clientX: number,
      clientY: number,
    ): { index: number; grip: "move" | "resize" } | null => {
      const point = screenToWorld(clientX, clientY);
      const unit = worldPerPixel(clientX, clientY);
      if (!point || unit === null) return null;
      const tolerance = GRIP_GRAB_PX * unit;
      // Last drawn first: circles overlap, and the one on top is the one the
      // user just made and most likely means.
      for (let i = regions.length - 1; i >= 0; i -= 1) {
        const shape = regions[i];
        if (shape.kind !== "circle" || shape.points.length < 2) continue;
        const [cx, cy] = shape.points[0];
        const [rx, ry] = shape.points[1];
        const radius = Math.hypot(rx - cx, ry - cy);
        const distance = Math.hypot(point[0] - cx, point[1] - cy);
        if (distance <= tolerance) return { index: i, grip: "move" };
        if (Math.abs(distance - radius) <= tolerance) {
          return { index: i, grip: "resize" };
        }
      }
      return null;
    },
    [regions, screenToWorld, worldPerPixel],
  );

  // --- region: draw the overlay -------------------------------------------
  // Runs on every change of the committed region or the in-progress draft.
  // Not on viewBox change: the overlay is expressed in viewBox units, so pan
  // and zoom move it without this code running at all.
  useEffect(() => {
    const group = overlay.current;
    const matrix = worldToSvg.current;
    if (!group) return;
    group.replaceChildren();
    if (!matrix) return;

    /** A circle in the overlay, from its centre and a point on the rim.
     *
     *  The radius is measured AFTER projection — the distance between the
     *  two projected points — rather than by scaling the world radius. The
     *  drawing-to-viewBox transform is a fit, so the two agree; taking the
     *  projected distance is simply the one that stays right if it ever
     *  stops being a plain uniform scale.
     */
    const drawCircle = (
      pair: [number, number][],
      mode: "window" | "crossing",
    ) => {
      if (pair.length < 2) return;
      const [cx, cy] = worldToSvgPoint(matrix, ...pair[0]);
      const [rx, ry] = worldToSvgPoint(matrix, ...pair[1]);
      const radius = Math.hypot(rx - cx, ry - cy);
      if (!(radius > 0)) return;
      const shape = document.createElementNS(SVG_NS, "circle");
      shape.setAttribute("cx", String(cx));
      shape.setAttribute("cy", String(cy));
      shape.setAttribute("r", String(radius));
      // The same classes the other shapes use, so window/crossing keep one
      // colour language across all three tools.
      shape.setAttribute("class", `region-shape region-${mode}`);
      group.appendChild(shape);
    };

    /** The handles that make a committed circle editable.
     *
     *  A dot at the centre to move it, and four on the rim to resize it. The
     *  rim is grabbable anywhere along its length; the four dots are
     *  affordance, not the hit area. Without them the circle looks finished
     *  rather than adjustable, and nobody discovers they can drag it.
     */
    const drawGrips = (pair: [number, number][]) => {
      if (pair.length < 2) return;
      const element = svgRef.current;
      const rect = element?.getBoundingClientRect();
      const current = element?.getAttribute("viewBox")?.split(/[\s,]+/).map(Number);
      // Sized in viewBox units so they land at a constant size on screen, the
      // same correction the polygon vertices make. Deriving this from the
      // world->viewBox scale instead made dots 414px wide on one drawing.
      const unitsPerPixel =
        rect && rect.width > 0 && current && current.length === 4
          ? current[2] / rect.width
          : 1;
      const r = Math.max(unitsPerPixel * GRIP_DRAW_PX, Number.MIN_VALUE);
      const [cx, cy] = worldToSvgPoint(matrix, ...pair[0]);
      const [rx, ry] = worldToSvgPoint(matrix, ...pair[1]);
      const radius = Math.hypot(rx - cx, ry - cy);

      const dot = (x: number, y: number, cls: string) => {
        const node = document.createElementNS(SVG_NS, "circle");
        node.setAttribute("cx", String(x));
        node.setAttribute("cy", String(y));
        node.setAttribute("r", String(r));
        node.setAttribute("class", cls);
        group.appendChild(node);
      };

      dot(cx, cy, "region-grip region-grip-centre");
      if (radius > 0) {
        for (const [dx, dy] of [[1, 0], [0, 1], [-1, 0], [0, -1]]) {
          dot(cx + dx * radius, cy + dy * radius, "region-grip");
        }
      }
    };

    const draw = (
      points: [number, number][],
      mode: "window" | "crossing",
      open: boolean,
    ) => {
      if (points.length < 2) return;
      const projected = points.map(([x, y]) => worldToSvgPoint(matrix, x, y));
      const d =
        `M ${projected.map((p) => `${p[0]} ${p[1]}`).join(" L ")}` +
        (open ? "" : " Z");
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("d", d);
      path.setAttribute("class", `region-shape region-${mode}`);
      group.appendChild(path);

      // Vertex dots, sized so they land at a constant ~4 px on screen. The
      // radius has to be expressed in viewBox units, so it is derived from
      // how many viewBox units one pixel currently spans — NOT from the
      // world->viewBox scale, which is a property of the drawing and not of
      // the screen at all. Using the latter made the dots 414 px wide on one
      // drawing and 0.15 px on another.
      if (open) {
        const element = svgRef.current;
        const rect = element?.getBoundingClientRect();
        const current = element?.getAttribute("viewBox")?.split(/[\s,]+/).map(Number);
        const unitsPerPixel =
          rect && rect.width > 0 && current && current.length === 4
            ? current[2] / rect.width
            : 1;
        const radius = Math.max(unitsPerPixel * 4, Number.MIN_VALUE);
        projected.forEach(([px, py], index) => {
          const dot = document.createElementNS(SVG_NS, "circle");
          dot.setAttribute("cx", String(px));
          dot.setAttribute("cy", String(py));
          // The FIRST vertex is the door the polygon closes through, so it
          // reads differently: larger and amber. Clicking it closes the
          // shape — the gesture every map tool has taught people.
          dot.setAttribute("r", String(index === 0 ? radius * 1.9 : radius));
          dot.setAttribute(
            "class",
            index === 0 ? "region-vertex region-vertex-first" : "region-vertex",
          );
          group.appendChild(dot);
        });
      }
    };

    if (draft) {
      if (draft.kind === "circle" && draft.cursor) {
        drawCircle([draft.points[0], draft.cursor], draft.mode);
      } else if (draft.kind === "rect" && draft.cursor) {
        const [x0, y0] = draft.points[0];
        const [x1, y1] = draft.cursor;
        draw(
          [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
          ],
          draft.mode,
          false,
        );
      } else if (draft.kind === "polygon") {
        const live = draft.cursor
          ? [...draft.points, draft.cursor]
          : draft.points;
        draw(live as [number, number][], draft.mode, true);
        // The would-be closing edge, dashed, from the live end back to the
        // FIRST vertex. This answers "where does my region actually end?"
        // while it is still being drawn: the enclosed area is always the
        // filled shape bounded by this edge, never a guess.
        if (live.length >= 2) {
          const [fx, fy] = worldToSvgPoint(matrix, ...draft.points[0]);
          const [lx, ly] = worldToSvgPoint(matrix, ...live[live.length - 1]);
          const closing = document.createElementNS(SVG_NS, "path");
          closing.setAttribute("d", `M ${lx} ${ly} L ${fx} ${fy}`);
          closing.setAttribute("class", "region-closing-edge");
          group.appendChild(closing);
        }
      }
    } else {
      // Every part of the selection is outlined, so two regions read as two
      // regions rather than as one that moved.
      for (const [index, shape] of regions.entries()) {
        if (shape.kind === "circle") {
          // A circle mid-reshape is drawn from the live drag, so the outline
          // follows the pointer while the counts stay put until release.
          const live =
            edit && edit.index === index
              ? ([edit.centre, edit.rim] as [number, number][])
              : shape.points;
          drawCircle(live, shape.mode);
          drawGrips(live);
          continue;
        }
        const points =
          shape.kind === "rect" ? rectCorners(shape.points) : shape.points;
        draw(points, shape.mode, false);
      }
    }
  }, [draft, edit, regions, svg]);

  // --- region: highlight what it matched -----------------------------------
  // One pass over the entity groups rather than one querySelector per handle:
  // a selection can carry 5,000 handles, and 5,000 selector lookups against a
  // 20,000-node document is the difference between instant and a visible
  // stall.
  useEffect(() => {
    const element = svgRef.current;
    if (!element) return;
    const nodes = element.querySelectorAll("[data-handle]");
    // Above the cap the highlight is switched off, and that is a considered
    // trade, not a shortcut. Repainting a recoloured stroke across thousands
    // of paths on a 19,445-group sheet is what made this page heavy — with
    // 2,504 objects lit, Chrome could not even capture the tab; with the
    // highlight off and the same region outlined, it was instant. And the
    // information loss is nil: a window selection now provably contains
    // nothing drawn outside the outline, so the outline already shows the
    // selection. Below the cap, per-object marking still earns its cost.
    if (regionHandles && regionHandles.size > MAX_HIGHLIGHTED) {
      nodes.forEach((node) => node.removeAttribute("data-in-region"));
      return;
    }
    if (!regionHandles || regionHandles.size === 0) {
      nodes.forEach((node) => node.removeAttribute("data-in-region"));
      return;
    }
    nodes.forEach((node) => {
      const handle = node.getAttribute("data-handle") ?? "";
      if (regionHandles.has(handle)) {
        node.setAttribute("data-in-region", "true");
      } else {
        node.removeAttribute("data-in-region");
      }
    });
  }, [regionHandles, svg]);

  // --- answer highlight: measure once, redraw on zoom ----------------------
  //
  // Split into two effects because the two halves change on different things,
  // and running the expensive half on the cheap trigger is the mistake this
  // file has already made once (see the cached-rectangle note above the
  // comment hover).
  //
  // WHERE each object is, in viewBox units, does not change when the user
  // zooms: viewBox coordinates are the drawing, and zooming moves the window
  // over it. So the measuring is done once per answer.
  //
  // HOW each object is marked does change, because "too small to outline" is a
  // fact about the current magnification. That is arithmetic over numbers
  // already in hand, and it is the only thing a zoom re-runs.
  const answerGeometry = useRef<{
    boxes: Map<string, { x: number; y: number; w: number; h: number }>;
    extents: { x: number; y: number; w: number; h: number } | null;
  }>({ boxes: new Map(), extents: null });
  /** Bumped when the measurement above is replaced, so the drawing effect
   *  re-runs. A ref alone would change without telling anyone. */
  const [answerMeasured, setAnswerMeasured] = useState(0);

  useEffect(() => {
    const element = svgRef.current;
    const boxes = new Map<string, { x: number; y: number; w: number; h: number }>();
    let extents: { x: number; y: number; w: number; h: number } | null = null;

    if (element && answerHighlight && answerHighlight.handles.length > 0) {
      // Extents are unioned over EVERY named object, including the ones past
      // the marking cap. That is the point of the aggregate mark: it stands
      // for all of them, so it has to reach all of them.
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      answerHighlight.handles.forEach((handle, index) => {
        const node = element.querySelector<SVGGraphicsElement>(
          `[data-handle="${cssEscape(handle)}"]`,
        );
        if (!node) return;
        let box: DOMRect;
        try {
          box = node.getBBox();
        } catch {
          return; // not rendered; it has no place on screen to mark
        }
        if (index < MAX_ANSWER_MARKS) {
          boxes.set(handle, { x: box.x, y: box.y, w: box.width, h: box.height });
        }
        minX = Math.min(minX, box.x);
        minY = Math.min(minY, box.y);
        maxX = Math.max(maxX, box.x + box.width);
        maxY = Math.max(maxY, box.y + box.height);
      });
      if (Number.isFinite(minX)) {
        extents = { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
      }
    }

    answerGeometry.current = { boxes, extents };
    setAnswerMeasured((n) => n + 1);
  }, [answerHighlight, svg]);

  useEffect(() => {
    const element = svgRef.current;
    const group = answerOverlay.current;
    if (!element || !group) return;

    group.replaceChildren();
    element
      .querySelectorAll("[data-answer]")
      .forEach((node) => {
        node.removeAttribute("data-answer");
        node.removeAttribute("data-answer-colour");
        node.removeAttribute("data-answer-fill");
      });

    if (!answerHighlight || answerHighlight.handles.length === 0) {
      setUnplaced(0);
      return;
    }

    // How many viewBox units one screen pixel spans right now. This is what
    // makes a marker the same size at every magnification, and it is read
    // from the live viewBox and the element's own width rather than from the
    // drawing->viewBox scale. Deriving it from the latter is a mistake this
    // file has made before and recorded: it produced dots 414 px wide on one
    // drawing and 0.15 px on another, because that scale is a property of the
    // drawing and this question is about the screen.
    const rect = element.getBoundingClientRect();
    const current = element.getAttribute("viewBox")?.split(/[\s,]+/).map(Number);
    if (!(rect.width > 0) || !current || current.length !== 4) return;
    const unitsPerPixel = current[2] / rect.width;

    const { boxes, extents } = answerGeometry.current;
    const total = answerHighlight.handles.length;

    // --- over the cap: one mark for all of them ---------------------------
    //
    // A last resort, and it reads like one. A single dashed rectangle around
    // five hundred plots says "somewhere in this neighbourhood" about an area
    // the reader is already looking at; what carries insight is the plots
    // themselves being coloured. So the cap that sends an answer down this
    // path was raised to 750 — the number this page already survives for
    // region highlighting — and this branch now only runs for sets larger
    // than any drawing here produces from one question.
    //
    // Above it the answer is still ANSWERED: the outline says where, and the
    // badge says how many were not drawn individually. What is refused is the
    // freeze, not the information.
    if (total > MAX_ANSWER_MARKS) {
      if (extents) {
        const pad = 6 * unitsPerPixel;
        const box = document.createElementNS(SVG_NS, "rect");
        box.setAttribute("x", String(extents.x - pad));
        box.setAttribute("y", String(extents.y - pad));
        box.setAttribute("width", String(extents.w + pad * 2));
        box.setAttribute("height", String(extents.h + pad * 2));
        box.setAttribute("class", "answer-extents");
        group.appendChild(box);
      }
      setUnplaced(0);
      return;
    }

    // --- under the cap: one mark per object -------------------------------
    const points = new Map<string, MarkPoint>();
    (answerPoints ?? []).forEach((p) => points.set(p.handle, p));
    const matrix = worldToSvg.current;
    let missing = 0;
    // One list per colour, so a marker keeps the colour of its kind.
    const exactDots: ([number, number][] | undefined)[] = [];
    const approxDots: ([number, number][] | undefined)[] = [];

    /** Colour index for a handle, from its group. Stable across a turn
     *  because the group order is the order the answer read them in. */
    const seen = new Map<string, number>();
    const colourOf = (handle: string): number => {
      const label = answerGroups.get(handle);
      if (!label) return 0;
      let index = seen.get(label);
      if (index === undefined) {
        index = seen.size % MARK_COLOURS;
        seen.set(label, index);
      }
      return index;
    };

    for (const handle of answerHighlight.handles) {
      const box = boxes.get(handle);
      if (!box) {
        // Named, validated, and not drawn here after all — counted, because a
        // mark that never appeared has to be a number somewhere.
        missing += 1;
        continue;
      }
      const sizePx = Math.max(box.w, box.h) / unitsPerPixel;
      const shape = markShapeFor(sizePx);

      if (shape === "outline") {
        // Its own outline, in the agent's colour, via one attribute. The
        // entity is already drawn; re-tracing its geometry into the overlay
        // would double the path count for no gain.
        //
        // Two attributes now: the colour index, and whether the shape may be
        // FILLED. Filling is allowed only for a ring the store calls
        // `complete`, because a fill on an open polyline paints a shape the
        // drawing does not contain -- the renderer closes the path to fill
        // it, inventing an edge between the two loose ends.
        const node = element.querySelector<SVGGraphicsElement>(
          `[data-handle="${cssEscape(handle)}"]`,
        );
        if (node) {
          node.setAttribute("data-answer", "true");
          node.setAttribute("data-answer-colour", String(colourOf(handle)));
          if (points.get(handle)?.ring_status === "complete") {
            node.setAttribute("data-answer-fill", "true");
          }
        }
        continue;
      }

      // Too small to outline: this is the circle Prasang asked for, and it
      // has to be a fixed size on SCREEN. A circle in drawing units sized to
      // suit a plot is a third of a pixel across when the whole 15.6 km site
      // is in view — invisible at exactly the magnification where a marker is
      // the only thing that could help.
      const stored = points.get(handle);
      const centre = markerCentre({
        polygonCentroid:
          stored?.polygon_centroid && matrix
            ? (worldToSvgPoint(
                matrix,
                stored.polygon_centroid[0],
                stored.polygon_centroid[1],
              ) as [number, number])
            : null,
        // The rendered box's middle, already in viewBox units. Allowed only
        // on this branch, where the object measures under eight pixels, so
        // the two candidate centres cannot be further apart than that on the
        // screen the user is looking at. On a large polygon the same
        // substitution is 72.7 m out and lands in the next plot — which is
        // why the large ones take the outline branch and never come here.
        bboxCentre: [box.x + box.w / 2, box.y + box.h / 2],
        shape,
      });
      if (!centre) {
        missing += 1;
        continue;
      }

      const bucket = centre.exact ? exactDots : approxDots;
      const index = colourOf(handle);
      (bucket[index] ??= []).push(centre.point);
    }

    // One node per KIND of marker, not one per object.
    //
    // This is what lets the cap be high enough to stop mattering. A circle
    // element per plot is 2,380 DOM nodes for one question about a whole
    // neighbourhood, and the cost of that -- not of the marking itself -- is
    // what used to send large answers down the aggregate-rectangle path,
    // where a dashed box around the entire site replaced the only thing worth
    // looking at. Merged into a path, the same 2,380 marks are two nodes.
    const radius = (MARKER_DIAMETER_PX / 2) * unitsPerPixel;
    for (const [buckets, className] of [
      [exactDots, "answer-marker"] as const,
      [approxDots, "answer-marker answer-marker-approx"] as const,
    ]) {
      buckets.forEach((dots, colour) => {
        if (!dots || !dots.length) return;
        const path = document.createElementNS(SVG_NS, "path");
        path.setAttribute(
          "d",
          dots.map(([x, y]) => circlePath(x, y, radius)).join(""),
        );
        path.setAttribute("class", className);
        path.setAttribute("data-answer-colour", String(colour));
        group.appendChild(path);
      });
    }

    setUnplaced(missing);
  }, [answerHighlight, answerGroups, answerPoints, answerMeasured, viewBox, svg]);

  /** Fit the view to everything the answer named.
   *
   *  Offered, never automatic. The user is looking at something; moving their
   *  screen without being asked is how a tool loses someone's place, and the
   *  marks are visible from wherever they are standing precisely because the
   *  small ones are drawn at a fixed size on screen.
   */
  const zoomToAnswer = useCallback(() => {
    const extents = answerGeometry.current.extents;
    if (!extents || !(extents.w >= 0) || !(extents.h >= 0)) return;
    // A single small object has near-zero extents; padding by a fraction of
    // the base view keeps the result a view rather than a microscope.
    const base = baseViewBox.current;
    const floor = base ? Math.max(base.w, base.h) * 0.002 : MIN_SPAN;
    const w = Math.max(extents.w, floor) * 1.25;
    const h = Math.max(extents.h, floor) * 1.25;
    setViewBox({
      x: extents.x + extents.w / 2 - w / 2,
      y: extents.y + extents.h / 2 - h / 2,
      w,
      h,
    });
  }, []);

  /** The committed vertices in viewBox units, for the rendered-geometry
   *  selection path. Same affine the overlay uses, so what gets tested is
   *  exactly what gets drawn. */
  const toViewPoints = useCallback(
    (points: [number, number][]): [number, number][] => {
      const matrix = worldToSvg.current;
      if (!matrix) return points;
      return points.map(([x, y]) => worldToSvgPoint(matrix, x, y));
    },
    [],
  );

  /** A drawing-space point in screen pixels — the forward of `screenToWorld`,
   *  used to hit-test the first-vertex close target against a real click. */
  const worldToScreen = useCallback(
    (point: [number, number]): [number, number] | null => {
      const element = svgRef.current;
      const matrix = worldToSvg.current;
      if (!element || !matrix) return null;
      const [vx, vy] = worldToSvgPoint(matrix, point[0], point[1]);
      const ctm = element.getScreenCTM();
      if (!ctm) return null;
      const projected = new DOMPoint(vx, vy).matrixTransform(ctm);
      return [projected.x, projected.y];
    },
    [],
  );

  /** Finish a polygon: at least three vertices, otherwise discard it. */
  const closePolygon = useCallback(() => {
    const current = draftRef.current;
    if (!current || current.kind !== "polygon") return;
    // Closing with a double-click on the last vertex is the natural gesture,
    // and it places that vertex twice: the first click of a double-click is a
    // normal click and adds a point. A zero-length edge is harmless to the
    // area test but it is not the shape the user drew, so it is removed here
    // rather than reasoned about everywhere downstream.
    const points = current.points.filter(
      (point, index, all) =>
        index === 0 ||
        point[0] !== all[index - 1][0] ||
        point[1] !== all[index - 1][1],
    );
    if (points.length >= 3) {
      onRegionDrawn(points, toViewPoints(points), "polygon", current.mode);
    }
    applyDraft(null);
    onRegionToolChange("none");
    // The double-click that closed the polygon still has a trailing click to
    // deliver; it must not land on an entity.
    suppressNextClick.current = true;
    setTimeout(() => {
      suppressNextClick.current = false;
    }, 0);
  }, [onRegionDrawn, onRegionToolChange, toViewPoints, applyDraft]);

  // Space is tracked window-wide with the same input guard as Escape/Enter:
  // held down it turns a drag into a pan while the polygon tool is armed.
  // preventDefault stops the page from scrolling on the spacebar mid-draw.
  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (event.code !== "Space") return;
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      spaceHeld.current = true;
      setSpacePanReady(true);
      if (regionTool === "polygon" || draftRef.current) event.preventDefault();
    };
    const up = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        spaceHeld.current = false;
        setSpacePanReady(false);
      }
    };
    // A keyup never arrives if the window loses focus mid-hold, which would
    // leave the viewer permanently in "about to pan".
    const blur = () => {
      spaceHeld.current = false;
      setSpacePanReady(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, [regionTool]);

  // Enter closes a polygon, Escape abandons whatever is being drawn. Bound to
  // the window because the canvas is a div that never holds focus.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // The panels stay mounted and hidden with CSS, so this listener is live
      // while the user is typing a comment or a question. Without this guard,
      // Escape in the chat box destroyed the region they had just drawn, and
      // Enter — which sends a chat message — also committed a half-drawn
      // polygon and yanked them off the Agent tab.
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (event.key === "Escape") {
        if (draftRef.current) {
          applyDraft(null);
          event.preventDefault();
        } else if (regionTool === "polygon" || regionTool === "circle") {
          onRegionToolChange("none");
        } else if (regions.length > 0) {
          onRegionClear();
        }
      } else if (event.key === "Enter" && draftRef.current?.kind === "polygon") {
        closePolygon();
        event.preventDefault();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [regionTool, regions, onRegionToolChange, onRegionClear, closePolygon, applyDraft]);

  // --- interaction ---------------------------------------------------------
  const onClick = useCallback(
    (event: React.MouseEvent) => {
      // A drag that ends over the canvas also fires click; ignore it.
      if (panState.current?.moved) return;
      // Same for a region drag, which never touches `panState`: the trailing
      // compatibility click was reaching the hit-test with `downTarget` still
      // pointing at whatever sat under the START of the drag, so drawing a
      // region immediately replaced the Selection panel with the Entity panel
      // — or, over empty paper, silently deselected.
      if (suppressNextClick.current) {
        suppressNextClick.current = false;
        return;
      }

      // While the polygon tool is armed, a click places a vertex instead of
      // selecting an entity. Nothing else about clicking changes, so plain
      // clicking still selects the moment the tool is put away.
      if (regionTool === "polygon" && regionEnabled) {
        // `detail` counts the clicks in the current sequence. The second click
        // of a double-click arrives here BEFORE dblclick fires, so without
        // this the closing gesture planted an extra vertex — and a
        // double-click in empty space planted two, committing a shape the
        // user had not drawn.
        if (event.detail > 1) return;
        // A click landing on the FIRST vertex closes the polygon instead of
        // stacking another point there — the visible amber dot is a target,
        // and hitting a target should do what the target promises.
        const current = draftRef.current;
        if (current?.kind === "polygon" && current.points.length >= 3) {
          const firstOnScreen = worldToScreen(current.points[0]);
          if (
            firstOnScreen &&
            Math.hypot(
              event.clientX - firstOnScreen[0],
              event.clientY - firstOnScreen[1],
            ) <= 12
          ) {
            closePolygon();
            return;
          }
        }
        const point = screenToWorld(event.clientX, event.clientY);
        if (!point) return;
        applyDraft((current) =>
          current && current.kind === "polygon"
            ? { ...current, points: [...current.points, point], cursor: point }
            : {
                kind: "polygon",
                points: [point],
                cursor: point,
                startScreenX: event.clientX,
                startScreenY: event.clientY,
                movedScreen: true,
                // A rectangle reads intent from drag direction (AutoCAD's
                // left-to-right / right-to-left). A polygon has no direction,
                // so it must choose: window, because drawing a shape AROUND
                // things means those things — and crossing additionally drags
                // in every long entity that merely clips the boundary, which
                // reads on screen as "it selected the whole drawing".
                mode: "window",
              },
        );
        return;
      }

      // Resolved from the click point rather than from the pointerdown
      // target, and that is a change with a reason.
      //
      // `downTarget` was here because a captured pointer retargets the
      // trailing click to the container, so `event.target` is useless. It is
      // still the truth about which ELEMENT the browser hit — but "which
      // element" is now the wrong question. With closed shapes clickable
      // through the middle, several entities cover the point, and picking one
      // needs all of them (see `resolveEntityAt`). `downTarget` can only ever
      // offer the topmost, which is precisely the answer that put a TEXT
      // entity on screen when a parcel was meant.
      //
      // The press position is used, not the release position: they differ by
      // a pixel or two on a real mouse, and the object under the press is the
      // one the user aimed at.
      //
      // Unless the two disagree by more than a press and a release can. A
      // click can reach here without a pointerdown of its own — a synthesised
      // click, a click delivered after a press that began somewhere else —
      // and the recorded press would then be stale or still at the origin,
      // which hit-tests the top-left corner of the window. The click's own
      // coordinates are the safe answer whenever the pair is not credible.
      const drift = Math.hypot(
        event.clientX - pressX.current,
        event.clientY - pressY.current,
      );
      const [hitX, hitY] =
        drift <= HIT_TOLERANCE_PX
          ? [pressX.current, pressY.current]
          : [event.clientX, event.clientY];
      const group = pickEntityAt(hitX, hitY);
      if (!group) {
        onSelect(null);
        return;
      }
      onSelect({
        handle: group.getAttribute("data-handle") ?? "",
        layer:
          group.closest("[data-layer]")?.getAttribute("data-layer") ?? "",
        type: group.getAttribute("data-type") ?? "",
      });
    },
    [onSelect, pickEntityAt, regionTool, regionEnabled, screenToWorld, closePolygon, worldToScreen],
  );

  /** A double click closes the polygon — the AutoCAD gesture. */
  const onDoubleClick = useCallback(() => {
    if (draftRef.current?.kind === "polygon") closePolygon();
  }, [closePolygon]);

  // Attached manually rather than via onWheel: React registers wheel at the
  // root as a *passive* listener, so preventDefault() inside a React handler
  // does nothing and the browser still runs ctrl+wheel page zoom and trackpad
  // overscroll while the user is zooming the drawing.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const handler = (event: WheelEvent) => onWheelNative(event);
    host.addEventListener("wheel", handler, { passive: false });
    return () => host.removeEventListener("wheel", handler);
  });

  const onWheelNative = useCallback((event: WheelEvent) => {
    const element = svgRef.current;
    if (!element) return;
    event.preventDefault();

    setViewBox((current) => {
      if (!current) return current;
      const rect = element.getBoundingClientRect();
      // Cursor position in viewBox units, so the point under the pointer
      // stays put — zooming to the centre instead feels like the drawing
      // is running away.
      const fx = (event.clientX - rect.left) / rect.width;
      const fy = (event.clientY - rect.top) / rect.height;
      const anchorX = current.x + fx * current.w;
      const anchorY = current.y + fy * current.h;

      const k = event.deltaY > 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
      const w = Math.max(current.w * k, MIN_SPAN);
      const h = Math.max(current.h * k, MIN_SPAN);
      return { x: anchorX - fx * w, y: anchorY - fy * h, w, h };
    });
  }, []);

  const onPointerDown = useCallback(
    (event: React.PointerEvent) => {
      downTarget.current = event.target as Element;
      pressX.current = event.clientX;
      pressY.current = event.clientY;
      if (!viewBox) return;

      // Grips are checked FIRST, before the shift-rectangle and before pan.
      // A press on a grip is unambiguous — the user is holding a control that
      // is drawn on screen — and letting pan take it would mean the drawing
      // slides away under the handle they were reaching for.
      if (!spaceHeld.current && regions.length > 0) {
        const grip = gripAt(event.clientX, event.clientY);
        if (grip) {
          const shape = regions[grip.index];
          try {
            (event.currentTarget as Element).setPointerCapture(event.pointerId);
          } catch {
            /* not capturable; the grip still tracks inside the canvas */
          }
          applyEdit({
            index: grip.index,
            grip: grip.grip,
            centre: shape.points[0],
            rim: shape.points[1],
            mode: shape.mode,
            moved: false,
          });
          return;
        }
      }

      // Shift+drag draws a rectangle instead of panning. Shift is what keeps
      // plain drag meaning pan: taking the bare drag for selection would have
      // been the smaller change and the wrong one, because panning is the
      // gesture people use constantly and selection is occasional.
      if (event.shiftKey && regionTool === "none" && regionEnabled) {
        const point = screenToWorld(event.clientX, event.clientY);
        if (point) {
          // Capture keeps the drag alive when the cursor leaves the canvas.
          // It is an improvement, not a requirement, so a browser that
          // refuses the pointer id must not cost the user the gesture — the
          // band simply stops following once the cursor is outside.
          try {
            (event.currentTarget as Element).setPointerCapture(event.pointerId);
          } catch {
            /* not capturable; the region still draws inside the canvas */
          }
          applyDraft({
            kind: "rect",
            points: [point],
            cursor: point,
            startScreenX: event.clientX,
            startScreenY: event.clientY,
            movedScreen: false,
            mode: "window",
          });
          return;
        }
      }

      // A circle is drawn radially: press at the centre, drag out, release
      // on the rim. One gesture rather than the polygon's click-per-vertex,
      // because a circle has no vertices to place and every CAD tool draws
      // one this way. Modal like the polygon, so a plain drag here does not
      // pan — Space+drag still does.
      if (regionTool === "circle" && regionEnabled && !spaceHeld.current) {
        const point = screenToWorld(event.clientX, event.clientY);
        if (point) {
          try {
            (event.currentTarget as Element).setPointerCapture(event.pointerId);
          } catch {
            /* not capturable; the circle still draws inside the canvas */
          }
          applyDraft({
            kind: "circle",
            points: [point],
            cursor: point,
            startScreenX: event.clientX,
            startScreenY: event.clientY,
            movedScreen: false,
            // Radial, so there is no drag DIRECTION to read window vs
            // crossing from. Crossing is the honest default for "what is
            // within this radius"; the panel's toggle re-asks in window mode.
            mode: "crossing",
          });
          return;
        }
      }

      // Drawing a polygon is modal: a plain drag does NOTHING, so the next
      // click is unambiguously a vertex. Panning while drawing is the
      // explicit Space+drag chord handled just below.
      const polygonMode =
        (regionTool === "polygon" && regionEnabled) ||
        draftRef.current?.kind === "polygon";
      if (polygonMode && !spaceHeld.current) return;

      (event.currentTarget as Element).setPointerCapture(event.pointerId);
      panState.current = {
        x: event.clientX,
        y: event.clientY,
        vb: viewBox,
        moved: false,
        dx: 0,
        dy: 0,
      };
      // Promised up front so the compositor has the layer ready before the
      // first move, rather than promoting it mid-gesture and stuttering on
      // the frame that does it.
      if (svgRef.current) svgRef.current.style.willChange = "transform";
      setIsPanning(true);
    },
    // `regions` and `gripAt` belong here. Without them this callback kept a
    // closure from before the first circle existed, so `regions.length > 0`
    // was false for ever and the grip check never ran — dragging a grip drew
    // a second circle instead of resizing the first.
    [viewBox, regionTool, regionEnabled, screenToWorld, regions, gripAt, applyEdit, applyDraft],
  );

  /** Move the drawing during a drag WITHOUT touching the viewBox.
   *
   *  Panning used to call `setViewBox` on every pointermove, and paid twice
   *  for it at pointer rate: React re-rendered this component, and the
   *  browser re-rasterised the SVG, because changing a viewBox invalidates
   *  every vector inside it. On a 19,445-group sheet at 700% zoom that is a
   *  slideshow — reported from the field, accurately, as "ngelag banget".
   *
   *  A CSS transform on the `<svg>` is composited instead: the layer already
   *  exists, so a frame costs a matrix multiply and nothing is redrawn. The
   *  viewBox is brought up to date once, on release, where a single
   *  re-raster is invisible. Correctness is unaffected — the transform and
   *  the viewBox express the same translation, and `getScreenCTM` (which is
   *  what region drawing and hit-testing use) accounts for both.
   */
  const dragPan = useCallback((event: React.PointerEvent) => {
    const state = panState.current;
    if (!state || !svgRef.current) return;
    state.dx = event.clientX - state.x;
    state.dy = event.clientY - state.y;
    if (Math.abs(state.dx) > 3 || Math.abs(state.dy) > 3) state.moved = true;
    if (panFrame.current !== null) return;
    panFrame.current = requestAnimationFrame(() => {
      panFrame.current = null;
      const live = panState.current;
      const element = svgRef.current;
      if (!live || !element) return;
      element.style.transform = `translate3d(${live.dx}px, ${live.dy}px, 0)`;
    });
  }, []);

  /** Fold the drag offset into the viewBox and drop the transform.
   *
   *  Order matters: the viewBox is written through React, the transform is
   *  cleared imperatively, and doing the second before the first would show
   *  one un-panned frame. Clearing it in the same tick as the state update
   *  keeps them in one paint.
   */
  const commitPan = useCallback(() => {
    if (panFrame.current !== null) {
      cancelAnimationFrame(panFrame.current);
      panFrame.current = null;
    }
    const state = panState.current;
    const element = svgRef.current;
    if (!element) return;
    element.style.willChange = "";
    if (!state || (state.dx === 0 && state.dy === 0)) {
      element.style.transform = "";
      return;
    }
    const rect = element.getBoundingClientRect();
    const next = {
      x: state.vb.x - (state.dx / rect.width) * state.vb.w,
      y: state.vb.y - (state.dy / rect.height) * state.vb.h,
      w: state.vb.w,
      h: state.vb.h,
    };
    element.style.transform = "";
    setViewBox(next);
  }, []);

  const onPointerMove = useCallback((event: React.PointerEvent) => {
    // Reshaping a committed circle. Only the OUTLINE follows the pointer:
    // re-asking the database on every move would repeat the mistake that made
    // panning a slideshow, and a count that flickers through a hundred wrong
    // values on the way to the right one is not information.
    const editing = editRef.current;
    if (editing) {
      const point = screenToWorld(event.clientX, event.clientY);
      if (point) {
        setHover(null);
        if (editing.grip === "move") {
          // The whole circle travels: the rim keeps its offset, so the radius
          // is unchanged by a move.
          const dx = point[0] - editing.centre[0];
          const dy = point[1] - editing.centre[1];
          applyEdit({
            ...editing,
            centre: point,
            rim: [editing.rim[0] + dx, editing.rim[1] + dy],
            moved: true,
          });
        } else {
          applyEdit({ ...editing, rim: point, moved: true });
        }
      }
      return;
    }

    // A region is being drawn: track the cursor, and for a rectangle decide
    // window vs crossing from the direction of the drag, live.
    const drawing = draftRef.current;
    if (drawing) {
      const point = screenToWorld(event.clientX, event.clientY);
      if (point) {
        setHover(null);
        const moved =
          drawing.movedScreen ||
          Math.abs(event.clientX - drawing.startScreenX) > 3 ||
          Math.abs(event.clientY - drawing.startScreenY) > 3;
        if (drawing.kind === "circle") {
          applyDraft({ ...drawing, cursor: point, movedScreen: moved });
          // Radial drag is exclusively a circle gesture; nothing below runs.
          return;
        }
        if (drawing.kind === "rect") {
          applyDraft({
            ...drawing,
            cursor: point,
            movedScreen: moved,
            // AutoCAD's rule, applied live so the box is already the right
            // colour before the button comes up.
            mode: event.clientX >= drawing.startScreenX ? "window" : "crossing",
          });
          // A rectangle drag is exclusively a region gesture; nothing below
          // it should run.
          return;
        }
        applyDraft({ ...drawing, cursor: point });
        // A polygon is placed by clicks, so dragging between vertices must
        // still pan — the gesture is not otherwise reachable while the tool
        // is armed. Fall through to the PAN handling below, but skip the
        // hover lookup on the way: a comment popup while the user is drawing
        // a region is noise, and paying for the hit-test on every move is
        // what made the rubber band stutter.
        dragPan(event);
        return;
      }
    }

    // Hover popup: only while NOT dragging. The comment data is already in
    // memory, so this costs one closest() walk per move — no network.
    // Cheap: a few distance computations over the circle regions, no DOM
    // query. Kept before the comment hover so a grip wins the cursor.
    if (!panState.current && regions.length > 0) {
      const grip = gripAt(event.clientX, event.clientY);
      setOverGrip(grip ? grip.grip : null);
    } else if (overGrip !== null) {
      setOverGrip(null);
    }

    if (!panState.current) {
      const handle = commentedHandleNear(event.clientX, event.clientY, 18);
      const comments = handle ? commentsByHandle[handle] : undefined;
      setHover(
        handle && comments && comments.length > 0
          ? { x: event.clientX, y: event.clientY, handle, comments }
          : null,
      );
      previewHover(event.clientX, event.clientY);
    } else {
      setHover(null);
      setHoverTarget(null);
    }

    dragPan(event);
  }, [commentsByHandle, previewHover, screenToWorld, dragPan, gripAt, regions, overGrip, applyEdit, applyDraft]);

  const endPan = useCallback((cancelled = false) => {
    // Pan teardown happens FIRST and unconditionally. It used to sit after an
    // early return on the rectangle branch, so a pointerdown that started a
    // pan and was finished by a region gesture left `isPanning` true and
    // `panState.moved` set — which swallowed the next entity click and
    // suppressed the comment hover popup until the user dragged again.
    setIsPanning(false);

    // A reshaped circle is re-asked exactly once, here. The outline has been
    // following the pointer; this is where the counts catch up.
    const reshaping = editRef.current;
    if (reshaping) {
      applyEdit(null);
      suppressNextClick.current = true;
      setTimeout(() => {
        suppressNextClick.current = false;
      }, 0);
      if (!cancelled && reshaping.moved) {
        const pair: [number, number][] = [reshaping.centre, reshaping.rim];
        // A grip dragged onto its own centre would be a zero-radius circle,
        // which the API rejects. Dropping it here keeps the region the user
        // already had rather than replacing it with an error.
        const radius = Math.hypot(
          pair[1][0] - pair[0][0],
          pair[1][1] - pair[0][1],
        );
        if (radius > 0) {
          onRegionEdit(reshaping.index, pair, toViewPoints(pair), reshaping.mode);
        }
      }
      return;
    }

    // Before the state is torn down: the offset lives on it.
    commitPan();
    const state = panState.current;
    // Cleared on the next tick so the click handler can still see `moved`.
    setTimeout(() => {
      if (panState.current === state) panState.current = null;
    }, 0);

    // Finish a rectangle. A polygon is not finished here: its vertices are
    // placed by clicks, so pointerup happens between vertices.
    const drawing = draftRef.current;
    if (drawing?.kind === "circle") {
      applyDraft(null);
      suppressNextClick.current = true;
      setTimeout(() => {
        suppressNextClick.current = false;
      }, 0);
      if (cancelled) return;
      const [centre] = drawing.points;
      const rim = drawing.cursor;
      // A press that never moved is a mis-click, not a circle. Judged on
      // SCREEN travel for the same reason the rectangle is: it is the
      // gesture the user made, not the world distance it happened to cover.
      if (drawing.movedScreen && rim) {
        const pair: [number, number][] = [centre, rim];
        onRegionDrawn(pair, toViewPoints(pair), "circle", drawing.mode);
      }
      return;
    }
    if (drawing?.kind === "rect") {
      applyDraft(null);
      suppressNextClick.current = true;
      setTimeout(() => {
        suppressNextClick.current = false;
      }, 0);
      // A gesture the browser took away is not a gesture the user completed.
      if (cancelled) return;
      const [start] = drawing.points;
      const end = drawing.cursor;
      // A Shift+click that never moved is a mis-click, not a region, and
      // turning it into a zero-area selection would wipe the previous region
      // for nothing. Judged on SCREEN travel rather than on the resulting
      // world box: a perfectly level drag has zero height in both, and it is
      // a real gesture the user made — it goes through and comes back as a
      // readable INVALID_REGION rather than vanishing without a word.
      if (drawing.movedScreen && end) {
        const corners: [number, number][] = [
          [Math.min(start[0], end[0]), Math.min(start[1], end[1])],
          [Math.max(start[0], end[0]), Math.max(start[1], end[1])],
        ];
        onRegionDrawn(corners, toViewPoints(corners), "rect", drawing.mode);
      }
    }
  }, [onRegionDrawn, onRegionEdit, toViewPoints, commitPan, applyEdit, applyDraft]);

  const resetView = useCallback(() => {
    if (baseViewBox.current) setViewBox({ ...baseViewBox.current });
  }, []);

  const zoomBy = useCallback((factor: number) => {
    setViewBox((current) => {
      if (!current) return current;
      const w = Math.max(current.w * factor, MIN_SPAN);
      const h = Math.max(current.h * factor, MIN_SPAN);
      return {
        x: current.x + (current.w - w) / 2,
        y: current.y + (current.h - h) / 2,
        w,
        h,
      };
    });
  }, []);

  const zoomPercent = baseViewBox.current && viewBox
    ? Math.round((baseViewBox.current.w / viewBox.w) * 100)
    : 100;

  return (
    <div className="viewer">
      <div className="viewer-toolbar">
        <button onClick={() => zoomBy(1 / ZOOM_STEP)} title="Zoom in">+</button>
        <button onClick={() => zoomBy(ZOOM_STEP)} title="Zoom out">−</button>
        <button onClick={resetView} title="Fit to view">Fit</button>
        <span className="zoom-readout">{zoomPercent}%</span>
        {/* Blocked is a STATE, not `disabled`: a disabled button swallows the
            click, and the click is exactly the moment the user is asking
            "why not?". Clicking while blocked opens the Selection tab, which
            explains the viewport situation and offers the one-click switch to
            the layout that works. The drawing gestures stay gated separately
            on `regionEnabled`, so this cannot arm the tool on a sheet. */}
        <button
          className={
            !regionEnabled
              ? "region-button blocked"
              : regionTool === "polygon"
                ? "region-button active"
                : "region-button"
          }
          aria-disabled={!regionEnabled}
          onClick={() => {
            if (!regionEnabled) {
              onRegionBlockedHelp();
              return;
            }
            applyDraft(null);
            onRegionToolChange(regionTool === "polygon" ? "none" : "polygon");
          }}
          title={
            regionEnabled
              ? "Draw a selection polygon: click each corner, double-click or Enter to close, Esc to cancel"
              : "Area selection is unavailable for this render — click for details"
          }
        >
          ⬠ Polygon
        </button>
        {/* Same guard as the polygon button: armed only when the render can
            actually answer a region, and a blocked click opens the reason
            rather than doing nothing. */}
        <button
          className={
            !regionEnabled
              ? "region-button blocked"
              : regionTool === "circle"
                ? "region-button active"
                : "region-button"
          }
          aria-disabled={!regionEnabled}
          onClick={() => {
            if (!regionEnabled) {
              onRegionBlockedHelp();
              return;
            }
            applyDraft(null);
            onRegionToolChange(regionTool === "circle" ? "none" : "circle");
          }}
          title={
            regionEnabled
              ? "Draw a selection circle: press at the centre, drag outwards, release on the rim. Esc cancels"
              : "Area selection is unavailable for this render — click for details"
          }
        >
          ◯ Circle
        </button>
        {(regions.length > 0 || draft) && (
          <button
            className="region-button"
            onClick={() => {
              applyDraft(null);
              onRegionClear();
            }}
            title="Remove the selection region (Esc)"
          >
            ✕ Region
          </button>
        )}

        {/* The answer badge. It is a COUNT first and a control second: the
            one thing a person needs from a set of marks they did not make is
            how many there are and whether any are missing. */}
        {answerHighlight && (
          <span className="answer-badge" title={answerPointsNote ?? undefined}>
            <span className="answer-swatch" aria-hidden="true" />
            <span>{answerHighlight.label}</span>
            {unplaced > 0 && (
              <span className="answer-badge-note">
                · {unplaced.toLocaleString()} could not be placed
              </span>
            )}
            {/* The caveat comes from the caller as words, not as a flag.
                A badge that renders its own wording for someone else's
                condition says the wrong thing the first time the condition
                changes. */}
            {answerPointsNote && (
              <span className="answer-badge-note">· {answerPointsNote}</span>
            )}
            {answerHighlight.handles.length > 0 &&
              answerHighlight.handles.length <= MAX_ZOOM_TO && (
                <button
                  className="link-button"
                  onClick={zoomToAnswer}
                  title="Fit the view to everything this answer names"
                >
                  show all
                </button>
              )}
            <button
              className="link-button"
              onClick={onClearAnswerHighlight}
              title="Remove the agent's marks from the drawing"
            >
              clear
            </button>
          </span>
        )}
        <span className="viewer-hint">
          {!regionEnabled
            ? "drag to pan · wheel to zoom · click an entity"
            : regionTool === "polygon"
            ? spacePanReady
              ? "pan mode — drag to move the drawing, release Space to keep drawing"
              : "click each corner · click the first point (or Enter / double-click) to close · hold Space to pan · Esc cancels"
            : regionTool === "circle"
            ? spacePanReady
              ? "pan mode — drag to move the drawing, release Space to keep drawing"
              : "press at the centre and drag outwards · hold Space to pan · Esc cancels"
            : draft
              ? draft.mode === "window"
                ? "window — only what is wholly inside"
                : "crossing — anything it touches"
              : hoverDegraded
                ? "drag to pan · wheel to zoom · click an entity · shift+drag to select an area · hover preview off: this drawing is too dense to hit-test every frame"
                : "drag to pan · wheel to zoom · click an entity · shift+drag to select an area"}
        </span>
        {exportUrl && (
          <>
            <a
              className="export-button"
              href={`${exportUrl}?format=dwg`}
              title="Download a DWG that carries these comments inside the file — the format AutoCAD and Autodesk Viewer accept most reliably, ~10x smaller than DXF"
            >
              ⭳ Export DWG
            </a>
            <a
              className="export-button"
              href={exportUrl}
              title="Download a DXF that carries these comments inside the file — XDATA on each entity, a hover tooltip in AutoCAD, and visible notes on the AI_MARKUP layer"
            >
              ⭳ Export DXF
            </a>
          </>
        )}
        {/* GeoJSON, on its own condition. It is beside the other two because
            it is the same act — take this drawing somewhere else — and it is
            NOT inside their block because it appears for a different reason:
            they need a comment to carry, this needs the drawing to have a
            position on the earth. */}
        {geoExportUrl ? (
          <a
            className="export-button"
            href={geoExportUrl}
            title="Download this drawing as GeoJSON: parcels coloured by land use and the H3 cells over them, ready to drop into geojson.io or any map. Coordinates come from a coordinate system that was inferred — the file says so on every feature."
          >
            ⭳ Export GeoJSON
          </a>
        ) : geoExportReason ? (
          <span
            className="export-button export-button-off"
            title={geoExportReason}
            aria-disabled="true"
          >
            ⭳ Export GeoJSON
          </span>
        ) : null}
        {geoExportUrl && SHOW_EXPORT_RES_PICKER ? (
          <label className="export-res">
            <span className="export-res-label">Cell size</span>
            <select
              value={geoExportRes}
              onChange={(e) => onGeoExportRes(Number(e.target.value))}
              title="How big the hexagon cells are in the exported file. The parcels are the same either way."
            >
              {EXPORT_RESOLUTIONS.map((r) => (
                <option key={r} value={r}>
                  {EXPORT_RESOLUTION_LABELS[r] ?? r}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <button
          className="panel-toggle"
          onClick={onTogglePanel}
          title="Show or hide the side panel"
        >
          {panelOpen ? "Hide panel ›" : "‹ Show panel"}
        </button>
      </div>
      <div
        ref={hostRef}
        className={
          `viewer-canvas${isPanning ? " panning" : ""}` +
          (regionTool !== "none" || draft ? " region-drawing" : "") +
          (edit
            ? edit.grip === "move"
              ? " grip-moving"
              : " grip-resizing"
            : overGrip === "move"
              ? " grip-move"
              : overGrip === "resize"
                ? " grip-resize"
                : "") +
          (spacePanReady ? " space-pan" : "")
        }
        onClick={onClick}
        onDoubleClick={onDoubleClick}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={() => endPan()}
        onPointerCancel={() => endPan(true)}
        onPointerLeave={() => {
          setHover(null);
          // The preview promises what the NEXT click will select. With the
          // pointer off the canvas there is no next click, and a mark left
          // behind would go on promising something after the question stopped
          // being asked.
          setHoverTarget(null);
        }}
      />
      {hover && (
        <div
          className="hover-popup"
          style={{ left: hover.x + 14, top: hover.y + 14 }}
        >
          <div className="hover-popup-handle">{hover.handle}</div>
          {hover.comments.slice(0, 3).map((comment, index) => (
            <p key={index}>
              <strong>{comment.author}</strong> {comment.body}
            </p>
          ))}
          {hover.comments.length > 3 && (
            <p className="muted small">+{hover.comments.length - 3} more — click to open</p>
          )}
        </div>
      )}
      {loading && (
        <div className="viewer-overlay">
          <div className="spinner" />
          <p>Rendering drawing…</p>
        </div>
      )}
      {!loading && !svg && (
        <div className="viewer-overlay">
          <p>Select a drawing to begin.</p>
        </div>
      )}
    </div>
  );
}

/** CSS.escape with a fallback: handles are hex, but a malformed one must not
 *  be able to break out of the attribute selector. */
function cssEscape(value: string): string {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(value);
  }
  return value.replace(/["\\]/g, "\\$&");
}

const SVG_NS = "http://www.w3.org/2000/svg";

/** Styling for the region overlay and the entities it matched.
 *
 *  Injected into the SVG by this component rather than written by the
 *  renderer: the drawing's own stylesheet describes the drawing, and a
 *  highlight colour is a decision of the viewer. Baking these rules into the
 *  cached SVGs would mean re-rendering every layout to change one of them.
 *
 *  The `:not()` guards are load-bearing. Without them the region highlight
 *  would out-specify the selected-entity and commented-entity rules that ship
 *  in the SVG, and clicking an object inside a region would appear to do
 *  nothing.
 */
const REGION_CSS = `
[data-region-overlay] .region-shape{fill:none;vector-effect:non-scaling-stroke;stroke-width:3}
[data-region-overlay] .region-window{stroke:#1d4ed8}
[data-region-overlay] .region-crossing{stroke:#15803d;stroke-dasharray:10 7}
[data-region-overlay] .region-vertex{fill:#2563eb;stroke:#ffffff;stroke-width:.5;vector-effect:non-scaling-stroke}
[data-region-overlay] .region-vertex-first{fill:#f59e0b;stroke:#ffffff;stroke-width:1;vector-effect:non-scaling-stroke}
[data-region-overlay] .region-closing-edge{fill:none;stroke:#16a34a;stroke-width:1.5;opacity:.6;stroke-dasharray:4 7;vector-effect:non-scaling-stroke}
[data-in-region='true']:not([data-selected]):not([data-commented]) path{stroke:#7c4dff!important}
/* The region is OUTLINED, never filled — and that is a performance fact, not
   a taste one. A translucent fill over this SVG forces the compositor to
   blend across the whole enclosed area on top of ~27,000 paths; measured on
   Janadriyah's sheet it froze the renderer hard enough that Chrome could
   neither screenshot the tab nor run a script in it for 30+ seconds, while
   the same page with the region cleared responded instantly. The outline
   alone still says which area and which mode (solid blue = window, dashed
   green = crossing), and it costs nothing.

   The highlight must not out-shout the drawing.
   A region over dense site-plan linework legitimately matches thousands of
   objects; recolouring each of them AND thickening its stroke to 2 turned
   the whole sheet into a purple flood that read as "everything got clicked".
   The colour still says which objects matched, the stroke width is left
   alone so the drawing stays itself, and the region boundary is now the
   loudest thing on screen because it is the actual answer to "what did I
   select". */
/* Closed shapes are clickable through their middle.

   'visible' rather than 'all': 'all' would hit-test regardless of visibility,
   so switching a layer off would leave its objects still selectable through
   a drawing they are no longer part of. 'visible' respects the visibility
   the layer panel controls and ignores only whether the fill is PAINTED,
   which is the single thing standing between a small plot and a person
   trying to click it.

   Scoped to closed paths, tagged during injection. An open path has a fill
   region too — the implied closure — and granting it to a road centreline
   would put a click target the size of the site over everything beneath it. */
[data-handle] path[data-closed='true']{pointer-events:visible}

/* What a click would select, shown before the click.

   A lighter shade of the SELECTION colour, because that is what it predicts.
   Using the agent's colour here would have the preview claim, for as long as
   the pointer rests on something, that the agent had named it. */
[data-hover-target='true']:not([data-selected]):not([data-answer]) path{stroke:#38bdf8!important}

/* The agent's answer.

   A different colour from selection (blue) and from a region match (purple),
   because "what I picked" and "what the agent answered" are different claims
   and a viewer that cannot tell them apart is a viewer that misattributes
   the machine's opinion to the user. The halo is what makes it survive dense
   linework: on a site plan a recoloured 1 px stroke disappears into the
   drawing it is drawn on top of. */
/* The :not() guards carry the same weight here as they do on the region rule
   above, and for a sharper reason. Written bare, this rule scores lower than
   the region rule that precedes it -- :not() counts towards specificity -- so
   an object that was BOTH inside a drawn region and named by the answer would
   have kept the region's purple, and the mark the user was waiting for would
   have been invisible exactly where they had been looking hardest. */
/* Six mark colours, one per kind of thing the answer named. A palette that
   grows without limit stops being readable long before it stops being
   possible; past half a dozen hues on a drawing that is already coloured,
   "which one is this" costs more than the grouping saves. */
[data-answer-overlay],[data-answer='true']{--mark-0:#db2777;--mark-1:#0ea5e9;--mark-2:#f59e0b;
  --mark-3:#22c55e;--mark-4:#a855f7;--mark-5:#ef4444}
[data-answer='true']{--mark:var(--mark-0)}
[data-answer='true'][data-answer-colour='1']{--mark:var(--mark-1)}
[data-answer='true'][data-answer-colour='2']{--mark:var(--mark-2)}
[data-answer='true'][data-answer-colour='3']{--mark:var(--mark-3)}
[data-answer='true'][data-answer-colour='4']{--mark:var(--mark-4)}
[data-answer='true'][data-answer-colour='5']{--mark:var(--mark-5)}
[data-answer='true']:not([data-selected]):not([data-commented]) path{stroke:var(--mark)!important;stroke-width:3!important}
/* Filled, not merely outlined — but only where the store says the ring is
   closed. A fill on an open polyline paints a shape the drawing does not
   contain: the renderer closes the path to fill it, inventing an edge
   between the two loose ends. */
[data-answer='true'][data-answer-fill='true']:not([data-selected]) path{fill:var(--mark)!important;fill-opacity:.38!important}
[data-answer='true']:not([data-selected]){filter:drop-shadow(0 0 3px var(--mark)) drop-shadow(0 0 10px var(--mark))}
[data-answer-overlay] .answer-marker{fill:none;stroke:var(--mark-0);stroke-width:2.5;
  vector-effect:non-scaling-stroke;filter:drop-shadow(0 0 4px rgba(219,39,119,.9))}
[data-answer-overlay] .answer-marker[data-answer-colour='1']{stroke:var(--mark-1);filter:drop-shadow(0 0 4px var(--mark-1))}
[data-answer-overlay] .answer-marker[data-answer-colour='2']{stroke:var(--mark-2);filter:drop-shadow(0 0 4px var(--mark-2))}
[data-answer-overlay] .answer-marker[data-answer-colour='3']{stroke:var(--mark-3);filter:drop-shadow(0 0 4px var(--mark-3))}
[data-answer-overlay] .answer-marker[data-answer-colour='4']{stroke:var(--mark-4);filter:drop-shadow(0 0 4px var(--mark-4))}
[data-answer-overlay] .answer-marker[data-answer-colour='5']{stroke:var(--mark-5);filter:drop-shadow(0 0 4px var(--mark-5))}
/* Placed from the bounding box rather than the stored centroid. Dashed so the
   difference is visible rather than merely documented -- the object is under
   eight pixels across, so the two centres are within that of each other, and
   saying which one was used costs nothing. */
[data-answer-overlay] .answer-marker-approx{stroke-dasharray:3 3}
[data-answer-overlay] .answer-extents{fill:none;stroke:#db2777;stroke-width:2;
  stroke-dasharray:12 8;vector-effect:non-scaling-stroke}

/* Re-asserted last so that clicking an entity always shows the click.
   In the renderer's own stylesheet the commented rule is written after the
   selected rule at equal specificity and both are !important, so a commented
   entity stayed orange when selected and the click looked ignored. That is a
   defect this file can fix without re-rendering 368 cached layouts, which is
   what changing render.py would cost. */
[data-selected='true'] path{stroke:#0284c7!important;stroke-width:3!important}
`;

/** Read the six numbers the renderer wrote on the `<svg>` root.
 *
 *  Returns null for anything malformed or singular rather than a default
 *  transform: a wrong transform selects the wrong objects silently, while a
 *  missing one lets the UI say region selection is unavailable for this
 *  render — which is the honest and recoverable failure.
 */
function parseWorldTransform(raw: string | null): WorldTransform | null {
  if (!raw) return null;
  const parts = raw.trim().split(/[\s,]+/).map(Number);
  if (parts.length !== 6 || !parts.every((n) => Number.isFinite(n))) return null;
  const [a, b, c, d, e, f] = parts;
  if (Math.abs(a * d - b * c) < 1e-12) return null;
  return { a, b, c, d, e, f };
}

/** Drawing coordinates -> viewBox coordinates. */
function worldToSvgPoint(
  t: WorldTransform,
  x: number,
  y: number,
): [number, number] {
  return [t.a * x + t.c * y + t.e, t.b * x + t.d * y + t.f];
}

/** viewBox coordinates -> drawing coordinates: the inverse of the above.
 *
 *  This is the direction that matters. Everything the user draws arrives in
 *  screen pixels and has to become drawing coordinates *immediately*, because
 *  a region kept in pixels stops meaning anything the moment the view moves.
 */
function svgToWorld(
  t: WorldTransform,
  x: number,
  y: number,
): [number, number] {
  const det = t.a * t.d - t.b * t.c;
  const dx = x - t.e;
  const dy = y - t.f;
  return [(t.d * dx - t.c * dy) / det, (t.a * dy - t.b * dx) / det];
}

/** The four corners of a rectangle stored as two opposite corners. */
function rectCorners(points: [number, number][]): [number, number][] {
  const [[x0, y0], [x1, y1]] = points;
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
}
