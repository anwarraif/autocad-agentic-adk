/**
 * Region selection over RENDERED geometry — what a sheet layout actually shows.
 *
 * Why this exists. A paper-space sheet draws model space through viewports:
 * the entities on screen carry MODEL coordinates in the database while the
 * sheet itself is measured in millimetres of paper. A region drawn on the
 * sheet therefore cannot be answered from stored bounding boxes — the numbers
 * live in a different space — and the first release simply disabled selection
 * there. Users experienced that as "the feature is broken on the layout the
 * app opens with", and they were right to.
 *
 * The correct question on a sheet is not "which stored boxes does this region
 * cover" but "which of the marks I can SEE does it cover". And the browser
 * already knows where every mark is: the renderer wrote each entity's paths
 * in viewBox units, which never change after render — pan and zoom only move
 * the root viewBox attribute. So this module indexes the rendered geometry
 * once per layout (~380 ms measured on a 19,445-group sheet) and answers any
 * number of selections from that index, instantly and entirely client-side.
 *
 * Layouts whose entities live in their own coordinate space (model space,
 * block definitions) keep using the server path in `selectRegion` — that is
 * the audited, stored-truth mechanism. This module is the sheet's mechanism,
 * and the two are dispatched in page.tsx by the same calibration that used to
 * merely block.
 *
 * Semantics, stated plainly because they differ from the server path:
 * matching here is against DRAWN geometry (per-path bounding rects), so an
 * entity projected through two viewports is selectable at either projection,
 * and "window" means at least one of its drawn appearances lies wholly inside
 * the region. The panel says so; nothing pretends to be a stored-extent test.
 */

export interface DomSelectionEntry {
  handle: string;
  layer: string;
  type: string;
  /** Bounding rects of each rendered path, in viewBox units. Several per
   *  entity when the renderer split it (or drew it through two viewports). */
  rects: { x: number; y: number; w: number; h: number }[];
}

export interface DomSelectionResult {
  total: number;
  handles: string[];
  handles_truncated: boolean;
  by_layer: { name: string; count: number }[];
  by_type: { name: string; count: number }[];
  groups: { layer: string; type: string; count: number }[];
  groups_truncated: boolean;
  /** Every match with its metadata, for client-side group row listing. */
  matches: { handle: string; layer: string; type: string }[];
  basis: "rendered-geometry";
}

/** Mirrors MAX_SELECTION_HANDLES on the server, so the two mechanisms never
 *  disagree about how many handles a selection may carry. */
// Matches the API ceiling. The browser already holds every match; capping
// lower threw away information it had for free.
const MAX_HANDLES = 25000;
const MAX_GROUPS = 300;

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/** Build the rendered-geometry index for the currently mounted SVG.
 *
 *  Costs one `getBBox` per drawn path (measured: ~26,970 paths → well under a
 *  second, once per layout). The result is immutable for the life of the
 *  render: viewBox coordinates are the drawing, not the view.
 */
export function buildDomIndex(svgRoot: SVGSVGElement): DomSelectionEntry[] {
  const entries: DomSelectionEntry[] = [];
  const groups = svgRoot.querySelectorAll<SVGGraphicsElement>("[data-handle]");
  groups.forEach((group) => {
    const handle = group.getAttribute("data-handle");
    if (!handle) return;
    const layer =
      group.closest("[data-layer]")?.getAttribute("data-layer") ?? "(none)";
    const type = group.getAttribute("data-type") ?? "(unknown)";

    const rects: DomSelectionEntry["rects"] = [];
    const paths = group.querySelectorAll<SVGGraphicsElement>("path");
    const shapes: ArrayLike<SVGGraphicsElement> =
      paths.length > 0 ? paths : [group];
    for (let i = 0; i < shapes.length; i++) {
      try {
        const b = shapes[i].getBBox();
        // Zero-extent shapes (a lone point) still occupy a spot on screen;
        // keep them so a crossing region can catch them.
        rects.push({ x: b.x, y: b.y, w: b.width, h: b.height });
      } catch {
        /* not rendered — nothing to select */
      }
    }
    if (rects.length > 0) entries.push({ handle, layer, type, rects });
  });
  return entries;
}

// ---------------------------------------------------------------------------
// Geometry — a deliberate port of cad_api/app/region.py
//
// The server cannot run these tests for a sheet (its numbers are in another
// space), so the rules are duplicated here. Kept rule-for-rule in the same
// order as the Python so a divergence is findable by reading the two files
// side by side; region.py's docstrings carry the reasoning.
// ---------------------------------------------------------------------------

type Pt = [number, number];
type Box = [number, number, number, number]; // minx, miny, maxx, maxy

function boxesOverlap(a: Box, b: Box): boolean {
  return !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3]);
}

function boxInsideBox(inner: Box, outer: Box): boolean {
  return (
    inner[0] >= outer[0] &&
    inner[1] >= outer[1] &&
    inner[2] <= outer[2] &&
    inner[3] <= outer[3]
  );
}

function pointInPolygon(x: number, y: number, poly: Pt[]): boolean {
  let inside = false;
  let j = poly.length - 1;
  for (let i = 0; i < poly.length; i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y) {
      const crossingX = ((xj - xi) * (y - yi)) / (yj - yi) + xi;
      if (x < crossingX) inside = !inside;
    }
    j = i;
  }
  return inside;
}

function orientation(a: Pt, b: Pt, c: Pt): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function onSegment(a: Pt, b: Pt, c: Pt): boolean {
  return (
    Math.min(a[0], b[0]) <= c[0] &&
    c[0] <= Math.max(a[0], b[0]) &&
    Math.min(a[1], b[1]) <= c[1] &&
    c[1] <= Math.max(a[1], b[1])
  );
}

function segmentsCross(p1: Pt, p2: Pt, p3: Pt, p4: Pt): boolean {
  const d1 = orientation(p3, p4, p1);
  const d2 = orientation(p3, p4, p2);
  const d3 = orientation(p1, p2, p3);
  const d4 = orientation(p1, p2, p4);
  if (d1 > 0 !== d2 > 0 && d3 > 0 !== d4 > 0) return true;
  if (d1 === 0 && onSegment(p3, p4, p1)) return true;
  if (d2 === 0 && onSegment(p3, p4, p2)) return true;
  if (d3 === 0 && onSegment(p1, p2, p3)) return true;
  if (d4 === 0 && onSegment(p1, p2, p4)) return true;
  return false;
}

function boxEdges(box: Box): [Pt, Pt][] {
  const corners: Pt[] = [
    [box[0], box[1]],
    [box[2], box[1]],
    [box[2], box[3]],
    [box[0], box[3]],
  ];
  return corners.map((c, i) => [c, corners[(i + 1) % 4]]);
}

function polygonEdges(poly: Pt[]): [Pt, Pt][] {
  return poly.map((p, i) => [p, poly[(i + 1) % poly.length]]);
}

function polygonBBox(poly: Pt[]): Box {
  const xs = poly.map((p) => p[0]);
  const ys = poly.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function boxCrossesPolygon(box: Box, poly: Pt[], polyBox: Box): boolean {
  if (!boxesOverlap(polyBox, box)) return false;
  for (const [px, py] of poly) {
    if (px >= box[0] && px <= box[2] && py >= box[1] && py <= box[3]) return true;
  }
  const corners: Pt[] = [
    [box[0], box[1]],
    [box[2], box[1]],
    [box[2], box[3]],
    [box[0], box[3]],
  ];
  for (const c of corners) if (pointInPolygon(c[0], c[1], poly)) return true;
  for (const [a, b] of polygonEdges(poly)) {
    for (const [c, d] of boxEdges(box)) if (segmentsCross(a, b, c, d)) return true;
  }
  return false;
}

function boxInsidePolygon(box: Box, poly: Pt[]): boolean {
  const corners: Pt[] = [
    [box[0], box[1]],
    [box[2], box[1]],
    [box[2], box[3]],
    [box[0], box[3]],
  ];
  for (const c of corners) if (!pointInPolygon(c[0], c[1], poly)) return false;
  for (const [a, b] of polygonEdges(poly)) {
    for (const [c, d] of boxEdges(box)) if (segmentsCross(a, b, c, d)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/** Run a region over the rendered-geometry index.
 *
 *  `points` are in viewBox units — two opposite corners for a rect, the
 *  vertices for a polygon. Modes follow AutoCAD, applied to drawn appearances:
 *  crossing = any of the entity's drawn rects touches the region; window = at
 *  least one drawn rect lies wholly inside it.
 */
/** Centre and radius from the two points a circle region carries.
 *  Mirrors `circle_of` in cad_api/app/region.py: same wire format, same
 *  meaning, so a sheet layout and a model layout answer the same question. */
function circleOf(points: Pt[]): { cx: number; cy: number; r: number } {
  const [[cx, cy], [rx, ry]] = points;
  return { cx, cy, r: Math.hypot(rx - cx, ry - cy) };
}

/** Every corner within the radius. Sufficient on its own for a circle, unlike
 *  a polygon: a disc is convex, so it contains the convex hull of anything it
 *  contains, and a box is the hull of its corners. */
function boxInsideCircle(box: Box, cx: number, cy: number, r: number): boolean {
  if (r <= 0) return false;
  const r2 = r * r;
  const corners: Pt[] = [
    [box[0], box[1]],
    [box[2], box[1]],
    [box[2], box[3]],
    [box[0], box[3]],
  ];
  return corners.every(([x, y]) => (x - cx) ** 2 + (y - cy) ** 2 <= r2);
}

/** The centre clamped into the box is the box's closest point to the centre;
 *  they overlap exactly when it is within the radius. Covers the centre being
 *  inside the box, the circle sitting wholly inside it, and a corner touch,
 *  without a case for any of them. */
function boxCrossesCircle(box: Box, cx: number, cy: number, r: number): boolean {
  if (r < 0) return false;
  const nx = Math.min(Math.max(cx, box[0]), box[2]);
  const ny = Math.min(Math.max(cy, box[1]), box[3]);
  return (cx - nx) ** 2 + (cy - ny) ** 2 <= r * r;
}

export function selectFromDom(
  index: DomSelectionEntry[],
  points: Pt[],
  kind: "rect" | "polygon" | "circle",
  mode: "window" | "crossing",
): DomSelectionResult {
  const matches: DomSelectionResult["matches"] = [];

  if (kind === "circle") {
    const { cx, cy, r } = circleOf(points);
    for (const entry of index) {
      // The same asymmetry the other two branches carry, and for the same
      // reason: "wholly inside" has to hold for EVERY piece the entity
      // draws, or one clipped fragment takes the whole object with it.
      const hit =
        mode === "window"
          ? entry.rects.every((rect) =>
              boxInsideCircle(
                [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
                cx,
                cy,
                r,
              ),
            )
          : entry.rects.some((rect) =>
              boxCrossesCircle(
                [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
                cx,
                cy,
                r,
              ),
            );
      if (hit) matches.push({ handle: entry.handle, layer: entry.layer, type: entry.type });
    }
  } else if (kind === "rect") {
    const [a, b] = points;
    const region: Box = [
      Math.min(a[0], b[0]),
      Math.min(a[1], b[1]),
      Math.max(a[0], b[0]),
      Math.max(a[1], b[1]),
    ];
    for (const entry of index) {
      // Window means the entity is WHOLLY inside, so every piece it draws
      // must be inside — `every`, not `some`. Written as `some` it took an
      // entity the moment one fragment fell in the region, and then
      // highlighted the whole thing: a border polyline clipped by the region
      // painted the entire sheet purple, which is what "everything got
      // selected" looked like on screen. Crossing stays `some`: touching
      // anywhere is the whole point of crossing.
      const hit =
        mode === "window"
          ? entry.rects.every((r) =>
              boxInsideBox([r.x, r.y, r.x + r.w, r.y + r.h], region),
            )
          : entry.rects.some((r) =>
              boxesOverlap([r.x, r.y, r.x + r.w, r.y + r.h], region),
            );
      if (hit) matches.push({ handle: entry.handle, layer: entry.layer, type: entry.type });
    }
  } else {
    const polyBox = polygonBBox(points);
    for (const entry of index) {
      // Same asymmetry as the rectangle branch above, and for the same
      // reason: wholly-inside has to hold for every drawn piece.
      const hit =
        mode === "window"
          ? entry.rects.every((r) =>
              boxInsidePolygon([r.x, r.y, r.x + r.w, r.y + r.h], points),
            )
          : entry.rects.some((r) => {
              const box: Box = [r.x, r.y, r.x + r.w, r.y + r.h];
              // Cheap prefilter first, same as the server's indexed stage.
              if (!boxesOverlap(box, polyBox)) return false;
              return boxCrossesPolygon(box, points, polyBox);
            });
      if (hit) matches.push({ handle: entry.handle, layer: entry.layer, type: entry.type });
    }
  }

  // Summaries, shaped exactly like the server's so the panel cannot tell the
  // mechanisms apart except by the `basis` field.
  const byLayer = new Map<string, number>();
  const byType = new Map<string, number>();
  const byGroup = new Map<string, number>();
  for (const m of matches) {
    byLayer.set(m.layer, (byLayer.get(m.layer) ?? 0) + 1);
    byType.set(m.type, (byType.get(m.type) ?? 0) + 1);
    // NUL separator, not a space: layer names contain spaces
    // ("00_External Road"), and a space split would shear them apart.
    const key = `${m.layer}\u0000${m.type}`;
    byGroup.set(key, (byGroup.get(key) ?? 0) + 1);
  }
  const sortDesc = (entries: [string, number][]) =>
    entries.sort((x, y) => y[1] - x[1]);

  const groupsAll = sortDesc([...byGroup.entries()]).map(([key, count]) => {
    const [layer, type] = key.split("\u0000");
    return { layer, type, count };
  });

  return {
    total: matches.length,
    handles: matches.slice(0, MAX_HANDLES).map((m) => m.handle),
    handles_truncated: matches.length > MAX_HANDLES,
    by_layer: sortDesc([...byLayer.entries()]).map(([name, count]) => ({ name, count })),
    by_type: sortDesc([...byType.entries()]).map(([name, count]) => ({ name, count })),
    groups: groupsAll.slice(0, MAX_GROUPS),
    groups_truncated: groupsAll.length > MAX_GROUPS,
    matches,
    basis: "rendered-geometry",
  };
}
