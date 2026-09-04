/**
 * Colour for the map views, by land use.
 *
 * ## The honest note about this palette
 *
 * A land-use map is a choropleth, which means every category is on screen at
 * once and any two of them can end up adjacent — the *all-pairs* case, not
 * the *adjacent-in-a-legend* case. Held to the all-pairs gate, no eight-hue
 * palette passes; a sweep of evenly spaced OKLCH hues over the accepted
 * lightness band found that **four** is the most that clears both colour-
 * vision-deficiency and normal-vision separation floors against this app's
 * surfaces. This drawing has nine land uses.
 *
 * So the palette below is validated to the *adjacent* gate — worst adjacent
 * CVD ΔE 8.4, worst adjacent normal-vision ΔE 19.3 against `--panel`
 * (#131a22) — and the gap is closed by secondary encoding rather than
 * pretended away. Identity is never carried by colour alone:
 *
 *   1. a legend is always present, naming every use with its count;
 *   2. hovering a feature names its use, its layer and its handle;
 *   3. clicking one opens the Entity panel, which names the use in words;
 *   4. the legend doubles as a filter — clicking a use isolates it, which is
 *      the "facet" escape the all-pairs cap actually asks for.
 *
 * Anyone who cannot separate two fills by eye has three other ways to read
 * them. That is the trade, stated rather than buried.
 *
 * ## Two rules that are not stylistic
 *
 * **Colour follows the use, never its rank.** The slot for `residential` is
 * fixed whether it is the largest use or absent entirely. Assigning by count
 * would repaint every surviving category the moment a layer was hidden, and a
 * map whose colours move when you filter it cannot be compared with itself.
 *
 * **A use with no decision is grey, not a colour.** `null` means no config
 * classifies that layer. That is different from a layer classified as
 * `unknown`, and different again from an empty one. Giving it a hue would
 * state a land use the drawing never states.
 */

/** The categorical steps, chosen for a dark surface and validated as a set.
 *  Order is the CVD-safety mechanism, not decoration — see the note above. */
const SLOTS = [
  "#3987e5", // blue
  "#d95926", // orange
  "#199e70", // aqua
  "#c98500", // yellow
  "#d55181", // magenta
  "#008300", // green
  "#9085e9", // violet
  "#e66767", // red
] as const;

/** Land use → slot. Fixed, and deliberately longer than any one drawing's
 *  vocabulary so that a drawing that introduces `industrial` does not shift
 *  the colour of everything before it. */
const USE_SLOT: Record<string, number> = {
  residential: 0,
  commercial: 1,
  open_space: 2,
  education: 3,
  community: 4,
  religious: 5,
  utility: 6,
  road: 7,
};

/** Everything that is not a decision: no config, an explicit non-answer, or
 *  machinery rather than a land use. */
const NEUTRAL = "#7d8794";

/** Ink for outlines drawn over satellite imagery.
 *
 *  A basemap is photography, so "contrast against the surface" has no fixed
 *  answer — the surface is whatever is under that parcel. A dark ring around
 *  every fill is the mark spec's answer to overlapping marks, and here it
 *  doubles as the only thing that keeps a fill legible over both a pale
 *  desert tile and a dark road. */
export const OUTLINE_INK: [number, number, number, number] = [12, 16, 22, 220];

export function colorForUse(use: string | null): string {
  if (!use) return NEUTRAL;
  const slot = USE_SLOT[use];
  return slot === undefined ? NEUTRAL : SLOTS[slot];
}

/** `#rrggbb` → deck.gl's `[r, g, b, a]`. */
export function rgba(
  hex: string,
  alpha = 255,
): [number, number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
    alpha,
  ];
}

export function colorForUseRgba(
  use: string | null,
  alpha: number,
): [number, number, number, number] {
  return rgba(colorForUse(use), alpha);
}

/** How a use should be written in a legend.
 *
 *  The config's vocabulary is snake_case because it is a key. A key is not a
 *  label, and shipping one to a reader is how `open_space` ends up in a
 *  screenshot that goes to a client.
 */
export function labelForUse(use: string | null): string {
  if (!use) return "No classification";
  return use
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export interface LegendEntry {
  use: string | null;
  label: string;
  color: string;
  count: number;
}

/** The legend, in the palette's fixed slot order.
 *
 *  Sorted by slot rather than by count for the same reason the colours are
 *  assigned that way: a legend that reorders itself when a layer is hidden
 *  makes two screenshots of the same drawing disagree.
 */
export function legendFor(
  features: { land_use: string | null; layer?: string | null }[],
  /** The dimension the SERVER said it counted by. `land_use` keeps the
   *  original behaviour exactly; `layer` buckets by the layer each shape is
   *  drawn on.
   *
   *  Passed in rather than guessed, and defaulted so every existing caller is
   *  unchanged. The heading and the rows have to agree: a legend headed LAYER
   *  whose every row said "No classification" was reporting the land-use
   *  buckets of a drawing that has none, under the name of a dimension it was
   *  not using. */
  dimension: "land_use" | "layer" = "land_use",
): LegendEntry[] {
  const counts = new Map<string | null, number>();
  for (const f of features) {
    const key = dimension === "layer" ? (f.layer ?? null) : f.land_use;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  // By land use the order is the poster's; by layer there is no such order,
  // so the biggest bucket leads and the reader sees what the drawing is
  // mostly made of.
  const rank = (use: string | null) =>
    use === null ? 999 : (USE_SLOT[use] ?? 998);
  const entries = [...counts.entries()].map(([use, count]) => ({
    use,
    label: dimension === "layer" ? (use ?? "No layer") : labelForUse(use),
    color: dimension === "layer" ? colorForLayer(use) : colorForUse(use),
    count,
  }));
  return dimension === "layer"
    ? entries.sort((a, b) => b.count - a.count).slice(0, LEGEND_MAX_ROWS)
    : entries.sort((a, b) => rank(a.use) - rank(b.use));
}

/** A drawing can hold 705 layers. A legend that lists them all is a wall, not
 *  a key, so the busiest are shown and the rest are reachable in the layer
 *  panel, which is the view built for exactly that question. */
export const LEGEND_MAX_ROWS = 12;


/** A stable colour for a layer NAME.
 *
 *  Deterministic, so the same layer is the same colour on every reload and
 *  between the legend and the map. Nothing is CLAIMED by the choice -- these
 *  are not land-use colours and the legend heading says so; they exist only
 *  so that twelve rows are twelve distinguishable things rather than twelve
 *  identical grey squares. Returns a CSS colour string, matching
 *  `colorForUse`, because the legend swatch is a background.
 */
export function colorForLayer(layer: string | null): string {
  if (!layer) return colorForUse(null);
  let h = 0;
  for (let i = 0; i < layer.length; i += 1) {
    h = (h * 31 + layer.charCodeAt(i)) >>> 0;
  }
  // Golden-angle hue spacing: adjacent hashes land far apart on the wheel, so
  // near-identical layer names do not come out near-identical colours.
  return `hsl(${((h * 137.508) % 360).toFixed(1)} 52% 62%)`;
}
