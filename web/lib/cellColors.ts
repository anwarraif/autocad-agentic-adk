/**
 * Colour for the H3 overlay, by magnitude.
 *
 * ## Why this is not `landUseColors`
 *
 * The parcel fills answer *what is this*, which is identity, and identity gets
 * a categorical palette with a fixed slot per use. A hexagon answers *how much
 * is here*, which is magnitude, and magnitude gets a sequential ramp: one hue,
 * light to dark. Reusing the categorical palette for a count is the standard
 * way to make a choropleth unreadable — the reader sees eight hues and looks
 * for eight categories that are not there.
 *
 * The two encodings share a screen, so they are deliberately different in
 * kind: the parcels vary in hue at roughly constant lightness, the cells vary
 * in lightness at constant hue. That is what lets a reader tell the analysis
 * layer from the drawing underneath it without being told.
 *
 * ## The ramp, and what it was checked against
 *
 * Six steps of OKLCH hue 235 with monotonically decreasing lightness — the
 * one property a sequential ramp is actually judged on. Hue 235 is cool
 * because the imagery under it is not: the site reads as sand at roughly hue
 * 70–90, so a cool ramp separates from its own background.
 *
 * Contrast was measured against both surfaces this sits on. Against the dark
 * chrome (`--panel`, #131a22) the worst step is 3.15:1, which clears the 3:1
 * floor, so the overlay survives with the basemap switched off. Against
 * mid-tone desert imagery the middle of the ramp measures ~1.1:1, and that is
 * not a fixable defect — a ramp that spans light to dark must pass through
 * the luminance of any mid-tone background somewhere. It is why every cell is
 * stroked: the boundary carries the shape when the fill cannot, which is the
 * mark spec's answer to overlapping marks and does the job here. The legend
 * prints the numeric bound of every class and the tooltip prints the exact
 * value, so no reading of this layer depends on separating two fills by eye.
 */

/** Light → dark, one hue. Order is the encoding; do not sort it. */
export const CELL_RAMP = [
  "#c2e4f8",
  "#90cdf1",
  "#5eb6e6",
  "#209ed7",
  "#0085c3",
  "#006dab",
] as const;

/** The stroke every cell gets, for the reason in the header. */
export const CELL_INK: [number, number, number, number] = [10, 16, 24, 170];

export interface CellClassing {
  /** Upper bound of each class, ascending and strictly increasing. */
  breaks: number[];
  /** The value each class starts at, for a legend that prints ranges. */
  lower: number[];
  /** The colour of each class. NOT always the whole ramp: when the data has
   *  fewer distinct values than the ramp has steps, the ramp is sampled down
   *  so that every swatch on the legend means something different. */
  colors: string[];
  /** Largest value seen. Printed so an empty-looking map can be told from a
   *  flat one. */
  max: number;
  /** How the breaks were chosen, in words, for the notes panel. */
  basis: string;
}

/** Evenly spaced steps from the ramp, always ending on the darkest.
 *
 *  The darkest step has to stay pinned to the highest class or the encoding
 *  stops being "darker means more" the moment the ramp is sampled down.
 */
function rampOf(count: number): string[] {
  const n = CELL_RAMP.length;
  if (count >= n) return [...CELL_RAMP];
  if (count === 1) return [CELL_RAMP[n - 1]];
  return Array.from(
    { length: count },
    (_, i) => CELL_RAMP[Math.round((i / (count - 1)) * (n - 1))],
  );
}

/**
 * Quantile breaks over the values present.
 *
 * Quantile rather than equal-interval because these counts are skewed: a
 * masterplan puts most of its parcels in a few dense cells and scatters the
 * rest, and equal intervals would then paint 90% of the site the palest step
 * and call it a map. Quantile spends the ramp where the data is.
 *
 * **The classes are deduplicated, and the ramp shrinks to fit.** Quantiles
 * over a short or repetitive set produce the same bound several times over:
 * isolating a land use with nine parcels spread one-per-cell produced six
 * classes that all read "1", six swatches apart. Six colours for one value is
 * a legend that invents distinctions the data does not contain. So the breaks
 * are made strictly increasing and the ramp is sampled down to however many
 * survive — one class, one meaning.
 *
 * The remaining cost is that classes are data-dependent, so two drawings — or
 * one drawing at two resolutions, or with a use isolated — do not share a
 * scale. That is why the legend prints the bounds rather than only the
 * colours, and why `basis` says so.
 */
export function classify(values: number[]): CellClassing | null {
  const sorted = values.filter((v) => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
  if (sorted.length === 0) return null;

  const distinct = [...new Set(sorted)];
  const wanted = Math.min(CELL_RAMP.length, distinct.length);

  // One class per value when there are few enough that quantiles would only
  // manufacture duplicates.
  let breaks: number[];
  if (distinct.length <= CELL_RAMP.length) {
    breaks = distinct;
  } else {
    const raw: number[] = [];
    for (let i = 0; i < wanted; i++) {
      const at = Math.min(
        sorted.length - 1,
        Math.max(0, Math.ceil(((i + 1) / wanted) * sorted.length) - 1),
      );
      raw.push(sorted[at]);
    }
    breaks = [...new Set(raw)].sort((a, b) => a - b);
    // The top class must always reach the maximum, or the densest cell falls
    // through every test and is drawn as if it were the smallest.
    if (breaks[breaks.length - 1] < sorted[sorted.length - 1]) {
      breaks.push(sorted[sorted.length - 1]);
    }
  }

  const lower: number[] = [];
  let previous: number | null = null;
  for (const bound of breaks) {
    // Integers step by one; anything else starts where the last class ended,
    // because a continuous measure has no "next" value to name.
    lower.push(
      previous === null
        ? sorted[0]
        : Number.isInteger(bound) && Number.isInteger(previous)
          ? previous + 1
          : previous,
    );
    previous = bound;
  }

  return {
    breaks,
    lower,
    colors: rampOf(breaks.length),
    max: sorted[sorted.length - 1],
    basis:
      `${breaks.length === 1 ? "One class" : `${breaks.length} classes`} over ` +
      `the ${sorted.length.toLocaleString()} cells that hold anything` +
      `${distinct.length <= CELL_RAMP.length ? ", one per distinct value" : ", by quantile"}. ` +
      `Classes follow this drawing at this resolution, so the same colour on a ` +
      `different drawing is not the same number.`,
  };
}

export function stepFor(value: number, classing: CellClassing): number {
  for (let i = 0; i < classing.breaks.length; i++) {
    if (value <= classing.breaks[i]) return i;
  }
  return classing.breaks.length - 1;
}

/** `#rrggbb` → `[r, g, b, a]`. */
function rgba(hex: string, alpha: number): [number, number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
    alpha,
  ];
}

export function colorForValue(
  value: number,
  classing: CellClassing | null,
  alpha: number,
): [number, number, number, number] {
  if (!classing || value <= 0) return [125, 135, 148, Math.min(alpha, 90)];
  return rgba(classing.colors[stepFor(value, classing)], alpha);
}
