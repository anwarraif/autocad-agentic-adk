/**
 * A selection made of several parts.
 *
 * Until now a selection was one region: drawing a second replaced the first,
 * and everything the user had built up vanished without being asked. In a CAD
 * tool that is the wrong default — AutoCAD accumulates too — and the field
 * report was exact: *"saat region kedua dipilih maka selectionnya yang
 * terambil hanya region kedua saja, sedangkan info dari region pertama
 * sebelumnya tidaklah masuk."* — kept in the reporter's own words because it
 * is a quotation and translating a quotation makes it something else. In
 * English: "when a second region is selected, only that second region is
 * taken, while the information from the first region does not come along."
 *
 * So a selection is now a LIST of groups, each keeping the shape that made it
 * and its own counts. The scope model from D-050 is unchanged and this is why
 * it survives: the drawing and layout are still the world, and the selection
 * is still the focus. The focus simply stopped being required to be one
 * contiguous piece of paper.
 *
 * The combining below is arithmetic, not estimation, with one honest edge:
 * a single region returns at most 5,000 handles, so a union built from
 * truncated lists is incomplete. That case is reported rather than smoothed
 * over — see `combineGroups`.
 */

import type { Region, RegionMode, SelectionKind, SelectionResult } from "./api";

/** One part of a selection: a region the user drew, or a single click. */
export interface SelectionGroup {
  /** Stable across re-renders, so React keys and removal survive reordering. */
  id: string;
  /** What the user sees: "Region 1", "Object 2CA4595". */
  label: string;
  /** The shape, for redrawing the outline. Null for a clicked object. */
  region: Region | null;
  /** This part's own result — exact, and independent of the others. */
  result: SelectionResult;
}

/** A selection combined from its parts, plus the facts about the combining
 *  that the parts alone cannot tell you. */
export interface CombinedSelection {
  result: SelectionResult;
  /** Objects that fall in more than one part. Null when it cannot be known,
   *  which happens only when some part hit the per-region handle cap. */
  overlap: number | null;
  /** True when at least one part returned a capped handle list, so the union
   *  is a floor rather than the answer. */
  incomplete: boolean;
  /** True when the layer/type breakdown had to be summed across parts that
   *  overlap, and is therefore an upper bound until the server re-counts it. */
  facetsApproximate: boolean;
}

let counter = 0;
/** Ids are generated here rather than from the content, because two identical
 *  regions drawn twice are two groups and must not collide. */
export function nextGroupId(): string {
  counter += 1;
  return `g${counter}`;
}

/** How many region groups exist, for naming the next one. */
export function nextRegionLabel(groups: SelectionGroup[]): string {
  // Counted by what the part IS, not by whether a shape was stored for it.
  // The two used to be the same thing, because every region came from a shape
  // drawn on the sheet. A rectangle dragged on the map is a region with no
  // shape in drawing coordinates -- there is no client-side inverse
  // projection -- and counting by `region` numbered every one of those
  // "Region 1".
  const n = groups.filter((g) => g.result.kind !== "click").length + 1;
  return `Region ${n}`;
}

function sumCounts(
  groups: SelectionGroup[],
  pick: (r: SelectionResult) => { name: string; count: number }[],
): { name: string; count: number }[] {
  const total = new Map<string, number>();
  for (const group of groups) {
    for (const row of pick(group.result)) {
      total.set(row.name, (total.get(row.name) ?? 0) + row.count);
    }
  }
  return [...total.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

/**
 * Fold the parts into one selection.
 *
 * The union of handles is the honest total: two regions that overlap share
 * objects, and counting those twice would inflate every figure downstream —
 * including the ones the agent quotes. Where the parts cannot overlap the
 * layer and type breakdowns are exact sums; where they can, they are marked
 * approximate so the caller can decide whether to ask the server to re-count.
 */
export function combineGroups(
  groups: SelectionGroup[],
  fallback: { drawingId: string; layout: string },
): CombinedSelection | null {
  if (groups.length === 0) return null;
  if (groups.length === 1) {
    return {
      result: groups[0].result,
      overlap: 0,
      incomplete: groups[0].result.handles_truncated,
      facetsApproximate: false,
    };
  }

  const seen = new Set<string>();
  const handles: string[] = [];
  for (const group of groups) {
    for (const handle of group.result.handles) {
      if (!seen.has(handle)) {
        seen.add(handle);
        handles.push(handle);
      }
    }
  }

  const incomplete = groups.some((g) => g.result.handles_truncated);
  const summed = groups.reduce((n, g) => n + g.result.total, 0);
  // With every handle list complete, the difference between what the parts
  // claim and what the union holds IS the overlap. With any list capped the
  // subtraction is meaningless, so it is refused rather than guessed.
  const overlap = incomplete ? null : summed - handles.length;

  const first = groups[0].result;
  const result: SelectionResult = {
    drawing_id: first.drawing_id || fallback.drawingId,
    layout: first.layout ?? fallback.layout,
    // The combined shape is not one shape. `kind` is kept for the fields that
    // still need a value; the panel reads the group list, not this.
    kind: first.kind,
    mode: first.mode as RegionMode,
    region: [],
    region_bbox: [],
    total: incomplete ? summed : handles.length,
    handles,
    handles_truncated: incomplete,
    extents: combineExtents(groups),
    basis: "combined",
    by_layer: sumCounts(groups, (r) => r.by_layer),
    by_type: sumCounts(groups, (r) => r.by_type),
    // Layer x type rows are not offered for a combined selection: expanding
    // one re-queries a single region, and there is no single region to
    // re-query. Each part keeps its own rows and stays expandable.
    groups: [],
    groups_truncated: false,
  };

  return {
    result,
    overlap,
    incomplete,
    facetsApproximate: overlap === null || overlap > 0,
  };
}

function combineExtents(groups: SelectionGroup[]): SelectionResult["extents"] {
  const boxes = groups
    .map((g) => g.result.extents)
    .filter((e): e is NonNullable<SelectionResult["extents"]> => e !== null);
  if (boxes.length === 0) return null;
  const min = [0, 1].map((i) => Math.min(...boxes.map((b) => b.min[i] ?? 0)));
  const max = [0, 1].map((i) => Math.max(...boxes.map((b) => b.max[i] ?? 0)));
  return { min, max };
}

/**
 * The figures that may honestly be called exact, or null when none can be.
 *
 * These travel to the server and become what the agent quotes, so calling a
 * number exact here is a promise. Three cases, and only the first two keep it:
 *
 * - one part: `total` and the breakdowns come straight from the region query,
 *   which counts by aggregation. Exact however many handles were named.
 * - several parts that do not overlap and none truncated: the union size is
 *   exact and the breakdowns are exact sums.
 * - several parts that DO overlap: summing the breakdowns counts the shared
 *   objects twice. The union total is still exact, so it is sent alone and
 *   the breakdowns are left for the server to recompute from the handles.
 *
 * A part that hit the enumeration ceiling makes the union itself unknowable,
 * and then nothing is claimed.
 */
export function exactCounts(
  groups: SelectionGroup[],
  combined: CombinedSelection,
): { total?: number; by_layer?: { name: string; count: number }[]; by_type?: { name: string; count: number }[] } | null {
  const single = groups.length === 1;
  const totalIsExact = single || !combined.incomplete;
  if (!totalIsExact) return null;

  const out: {
    total?: number;
    by_layer?: { name: string; count: number }[];
    by_type?: { name: string; count: number }[];
  } = { total: combined.result.total };

  if (!combined.facetsApproximate) {
    out.by_layer = combined.result.by_layer.slice(0, 500);
    out.by_type = combined.result.by_type.slice(0, 500);
  }
  return out;
}

/**
 * What the agent is told about the shape of the selection.
 *
 * The handles themselves never appear here — they travel by reference, as
 * D-046 requires. What travels is the STRUCTURE: how many parts, what each
 * one is, and what each one holds. That is a bounded handful of numbers, and
 * it is what lets the agent answer "how many did I select in the second
 * region" without being handed the second region.
 */
export function describeGroups(
  groups: SelectionGroup[],
  combined: CombinedSelection,
  layout: string,
): string {
  const where = `on layout "${combined.result.layout ?? layout}"`;
  if (groups.length === 1) {
    const only = groups[0];
    const layers = only.result.by_layer
      .slice(0, 5)
      .map((l) => `${l.name} (${l.count})`)
      .join(", ");
    const types = only.result.by_type
      .slice(0, 5)
      .map((t) => `${t.name} (${t.count})`)
      .join(", ");
    return (
      `${only.result.total} objects selected ${where}` +
      (only.result.kind === "click" ? "" : ` (${only.result.mode} region)`) +
      `. Top layers: ${layers}. Top types: ${types}.`
    );
  }

  // Layers AND types, both labelled. Giving layers alone was measured to
  // cause a real error: asked how many DIMENSIONs were selected, the agent
  // had only layer figures, saw a layer called "DIM", and added those —
  // reporting 812 where the truth was 818, because 6 DIMENSIONs live on a
  // layer called DimMinor. A name that looks like an answer will be used as
  // one, so the honest fix is to supply the figure it actually needed.
  const parts = groups
    .map((g) => {
      const layers = g.result.by_layer
        .slice(0, 3)
        .map((l) => `${l.name} (${l.count})`)
        .join(", ");
      const types = g.result.by_type
        .slice(0, 3)
        .map((t) => `${t.name} (${t.count})`)
        .join(", ");
      // Same reasoning as `nextRegionLabel`: `kind` says how the part was
      // made, `region` only says whether its outline can be redrawn. For
      // every part that carries a shape this is the identical string.
      const how =
        g.result.kind === "click"
          ? "clicked"
          : `${g.result.mode} ${g.result.kind}`;
      return (
        `${g.label}: ${g.result.total} objects, ${how}` +
        (layers ? `, top layers ${layers}` : "") +
        (types ? `, top types ${types}` : "")
      );
    })
    .join("; ");

  // Whether the parts may be added is stated either way. Left unsaid, the
  // agent has to guess, and guessing wrong is silent: with overlapping
  // regions a sum is too big, and nothing in the answer would show it.
  const overlapNote =
    combined.overlap === null
      ? " Some parts hit the 5,000-object cap, so the total is a floor rather than the count, and the parts must NOT be added."
      : combined.overlap > 0
        ? ` ${combined.overlap} objects fall in more than one part, so the parts must NOT be added — the total above already counts them once.`
        : " No object falls in more than one part.";

  return (
    `${combined.result.total} objects selected ${where}, in ` +
    `${groups.length} separate parts — ${parts}.${overlapNote} ` +
    `Treat all ${groups.length} parts together as the selection unless the ` +
    `question names one of them. The per-part figures above are the top few ` +
    `only; for an exact count of any one layer or type across the whole ` +
    `selection, call describe_selection rather than adding these.`
  );
}
