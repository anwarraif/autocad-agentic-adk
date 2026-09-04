/**
 * Where the height of an extruded shape comes from.
 *
 * An interface with two implementations from day one, per §3.6 of the
 * architecture note. The placeholder is not throwaway scaffolding — it is the
 * same call signature the real data will arrive through, so wiring measured
 * heights in later is a swap rather than a rewrite of the 3D view.
 *
 * ## Why a hash of the handle and never `Math.random()`
 *
 * Random heights reshuffle the city on every re-render and every reload.
 * Three costs, all of them real:
 *
 *   * it reads as a bug to anyone watching a demo — the skyline moves when
 *     nothing was touched;
 *   * two screenshots of the same view disagree, so nothing can be reviewed;
 *   * "that tower looked wrong" becomes impossible to investigate, because
 *     the tower is not there any more.
 *
 * A hash of the handle gives a stable skyline for free, and bucketing by land
 * use lets a villa and a school differ believably instead of every building
 * being the same box.
 *
 * ## Why every height carries a basis
 *
 * The only thing worse than a placeholder height is a placeholder height that
 * nobody can tell from a measured one. `basis` travels with every number and
 * the UI prints it, so a figure on screen always says whether it was
 * measured, estimated, or invented for the picture. When the backend supplies
 * real figures they arrive through `measuredHeights()` with their own basis,
 * and the label changes without the view changing.
 */

import { drawsAsArea } from "./mapGeometry";

export interface HeightInput {
  handle: string;
  kind: "ring" | "path" | "segment" | "label";
  land_use: string | null;
  land_use_role: string | null;
  /** Drawing units, from the store. Used only to keep a plot's box in
   *  proportion to the plot — never to derive a height from an area. */
  area: number | null;
}

export interface Height {
  /** Metres. The extrusion unit deck.gl is given for a lng/lat layer. */
  metres: number;
  basis: string;
}

export interface HeightSource {
  readonly id: string;
  /** True only when the figures came from the file or a survey. The UI reads
   *  this to decide whether to caveat the view, and nothing else decides it. */
  readonly measured: boolean;
  /** A sentence for the UI, naming what these numbers are. */
  readonly note: string;
  heightFor(feature: HeightInput): Height | null;
}

/** FNV-1a over the handle.
 *
 *  Any stable hash would do; this one is four lines, has no dependency, and
 *  spreads short hexadecimal strings — which is all the input ever is —
 *  evenly enough that neighbouring handles do not get neighbouring heights.
 */
function hash(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** A number in [0, 1) from a handle, stable across reloads and machines. */
function unit(handle: string): number {
  return hash(handle) / 0x100000000;
}

/** Storeys by land use, as a plausible low-rise masterplan.
 *
 *  These are a *shape*, not a claim. The ranges say "a villa is shorter than
 *  a school is shorter than a tower" so the 3D view reads as a place rather
 *  than as noise; they do not say how tall anything on this site actually is,
 *  and `measured: false` is what keeps that distinction visible.
 */
const STOREYS: Record<string, [number, number]> = {
  residential: [1, 3],
  commercial: [2, 6],
  education: [2, 4],
  religious: [1, 3],
  community: [1, 3],
  utility: [1, 2],
  open_space: [0, 0],
  road: [0, 0],
};

const DEFAULT_STOREYS: [number, number] = [1, 3];
const METRES_PER_STOREY = 3.2;

/** @param classified whether this drawing's layers carry any land-use
 *  decision. Passed in rather than derived, so that this and the layer that
 *  paints the fills read one value — a fill with no height, or a height under
 *  no fill, is a floating or invisible box. */
export function placeholderHeights(classified: boolean): HeightSource {
  return {
    id: `placeholder-hash:${classified ? "classified" : "unclassified"}`,
    measured: false,
    note:
      "Heights are placeholders: a stable hash of each object's handle, " +
      "bucketed by land use. They are the same on every reload and on every " +
      "machine, and they are not this site's real heights.",
    heightFor(feature) {
      // Only a filled area gets a box, and `drawsAsArea` is the single
      // definition of what that is — shared with the layer that paints them,
      // so a fill can never appear without a height or vice versa. A road
      // centreline, an annotation and a block boundary are not buildings.
      if (!drawsAsArea(feature, classified)) return null;
      const [low, high] = STOREYS[feature.land_use ?? ""] ?? DEFAULT_STOREYS;
      if (high === 0) return null;
      const storeys = low + Math.floor(unit(feature.handle) * (high - low + 1));
      return {
        metres: storeys * METRES_PER_STOREY,
        basis: `placeholder — ${storeys} storey${storeys === 1 ? "" : "s"} at ${METRES_PER_STOREY} m, from a hash of handle ${feature.handle}`,
      };
    },
  };
}

/** Heights from real data, keyed by handle.
 *
 *  Nothing calls this yet — it exists so that the 3D view is already written
 *  against the interface the real figures will arrive through. When the
 *  backend can answer "how tall is this", the change is one line where the
 *  source is chosen, and `measured: true` removes the caveat from the UI by
 *  itself.
 *
 *  `basis` is per-handle and required, because the ask in §3.7 is not just a
 *  number: a figure read from a block attribute, a figure estimated from a
 *  typology, and a figure surveyed on site are three different things, and a
 *  view that shows them identically is one that cannot be checked.
 */
export function measuredHeights(
  metresByHandle: Map<string, { metres: number; basis: string }>,
  note: string,
): HeightSource {
  return {
    id: "measured",
    measured: true,
    note,
    heightFor(feature) {
      const hit = metresByHandle.get(feature.handle);
      // Absent rather than zero. A building with no recorded height is not a
      // building of height zero, and flattening it would quietly delete it
      // from the view.
      return hit ? { metres: hit.metres, basis: hit.basis } : null;
    },
  };
}

/**
 * Height for an H3 cell, from what the cell actually holds.
 *
 * This is the first extrusion in this app that is measured rather than
 * invented, and the distinction is the reason `HeightSource` exists at all. A
 * cell's parcel count and its parcel area are read off the drawing: they are
 * the same numbers `/geo` publishes as `totals` and the same ones the agent
 * quotes. Extruding by them states a fact.
 *
 * It does NOT make building heights measured. A parcel is still extruded by a
 * hash of its handle, because nothing in the store knows how tall a building
 * is. Two extrusions can be on screen at once — hexagons by count, parcels by
 * placeholder — and they carry different bases for exactly that reason. A
 * view that showed them identically would be one where a reader could not
 * tell a measurement from a guess.
 *
 * The scale is a display choice and says so. The metre figure a cell is drawn
 * at is not a claim that anything there is that tall; it is a bar chart with
 * hexagonal bars, and `basis` prints the underlying value and its unit.
 */
export interface CellHeight {
  metres: number;
  basis: string;
}

export interface CellHeightSource {
  readonly id: string;
  readonly measured: boolean;
  readonly note: string;
  /** Metres per unit of the measure, so the tallest cell reaches a sensible
   *  height whatever the measure is counting. */
  readonly scale: number;
  heightFor(value: number): CellHeight;
}

/** How tall the largest cell should stand, in metres.
 *
 *  Chosen against the site rather than against the numbers, and lowered once
 *  after looking at it: Janadriyah's parcels sit inside about 1.9 km, and at
 *  600 m the tallest columns were a third of the site across. They read as
 *  walls — they hid the cells behind them and, worse, hid the drawing the
 *  overlay exists to be compared against, which is the one thing this view
 *  is for.
 *
 *  At 260 m the tallest column is about a seventh of the site's width: still
 *  clearly a bar chart, still ordered by eye, and low enough that the parcels
 *  underneath stay legible at a normal camera pitch.
 */
const TALLEST_CELL_M = 260;

export function cellHeights(
  measure: "count" | "area",
  max: number,
  unit: string | null,
): CellHeightSource {
  // A zero maximum means an empty overlay, not a division to attempt.
  const scale = max > 0 ? TALLEST_CELL_M / max : 0;
  const noun = measure === "area" ? `${unit ?? "sq. units"} of parcel` : "parcels";
  return {
    id: `cell-${measure}:${max}`,
    measured: true,
    scale,
    note:
      `Hexagon height is ${measure === "area" ? "parcel area" : "parcel count"} ` +
      `per cell, measured off the drawing — the tallest cell holds ` +
      `${max.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${noun}. ` +
      `The height is scaled for the picture; the number it stands for is the ` +
      `measurement. Parcel heights in the same view are NOT measured.`,
    heightFor(value) {
      return {
        metres: Math.max(0, value) * scale,
        basis:
          `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${noun} ` +
          `in this cell — measured`,
      };
    },
  };
}
