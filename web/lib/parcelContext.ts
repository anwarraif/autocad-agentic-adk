/**
 * The seam between this viewer and work that is still being written.
 *
 * Three things UPLIFT-12 needs are being built at the same time as it, by
 * other people, in files this agent must not touch:
 *
 *   - `polygon_centroid` for a list of handles      (UPLIFT-01, stored; needs
 *                                                    an endpoint to read it in
 *                                                    bulk — see PROGRESS.md)
 *   - `contained_by` for a text entity              (UPLIFT-03, `join_labels`
 *                                                    run in reverse)
 *   - land-use on a parcel                          (UPLIFT-02)
 *
 * Every one of them is reached through this file and nowhere else. If the
 * shapes below turn out to differ from what lands, one file changes and the
 * viewer does not — which is the entire reason this file exists rather than a
 * fetch call sitting in a component.
 *
 * The second reason is absence. None of these endpoints may exist when this
 * code first runs, and the honest behaviour then is not an error dialog and
 * certainly not a guess: it is `null` plus a sentence saying what is missing
 * and why (G3, G8). Every function here returns exactly that. Nothing in the
 * viewer is allowed to break because a route is not deployed yet, and nothing
 * in the viewer is allowed to invent a number in its place.
 *
 * The expected request and response shapes are written out in PROGRESS.md so
 * the coordinator can diff them against what U01/U02/U03 actually shipped.
 */

import { API_BASE, ApiError } from "./api";

/** A read that is allowed to come back empty because the feature behind it
 *  may not exist yet.
 *
 *  `reason` is never shown as an error. It is shown where the data would have
 *  been, so a missing panel explains itself instead of looking broken.
 */
export interface Optional<T> {
  data: T | null;
  reason: string | null;
}

const ABSENT = (reason: string): Optional<never> => ({ data: null, reason });

/** GET/POST that treats 404 and 501 as "not built yet" rather than as failure.
 *
 *  The distinction matters more than it looks. A 404 here means the campaign
 *  has not merged that route; a 500 means it merged and is broken. Collapsing
 *  the two would hide a real fault behind a reassuring "coming soon".
 */
async function optionalRequest<T>(
  path: string,
  init: RequestInit | undefined,
  missing: string,
): Promise<Optional<T>> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    return ABSENT(`Could not reach cad-api at ${API_BASE}.`);
  }
  if (response.status === 404 || response.status === 501) {
    return ABSENT(missing);
  }
  if (!response.ok) {
    let body: { message?: string; hint?: string } = {};
    try {
      body = await response.json();
    } catch {
      /* a non-JSON error body is still an error */
    }
    return ABSENT(
      body.hint
        ? `${body.message ?? response.statusText} — ${body.hint}`
        : (body.message ?? `${response.status} ${response.statusText}`),
    );
  }
  try {
    return { data: (await response.json()) as T, reason: null };
  } catch (cause) {
    throw new ApiError(
      "BAD_RESPONSE",
      "cad-api returned something that is not JSON.",
      String(cause),
    );
  }
}

// ---------------------------------------------------------------------------
// Centroids — UPLIFT-01 storage, read in bulk
// ---------------------------------------------------------------------------

/** One entity's geometry, as far as marking it is concerned.
 *
 *  `polygon_centroid` is area-weighted and `bbox_centre` is the middle of the
 *  box; they are different points and the difference is not academic. On a
 *  neighbourhood boundary polygon they sit 72.7 m apart, which is far enough
 *  to land a mark inside the wrong plot. Both are carried so the viewer can
 *  choose knowingly instead of taking whichever one happens to be present.
 *
 *  `ring_status` rides along because it says WHY a centroid is missing. A
 *  polygon with arc segments has no linearised ring and therefore no
 *  centroid, and "this one has arcs" is a better thing to show a user than a
 *  blank.
 */
export interface MarkPoint {
  handle: string;
  polygon_centroid: [number, number] | null;
  bbox_centre: [number, number] | null;
  ring_status: string | null;
}

/** Centroids for a handful of handles, in one call.
 *
 *  Deliberately one request for the whole list. The alternative — `get_entity`
 *  per handle — is up to 200 round trips for one answer, on a page whose
 *  measured problem is that it already stalls.
 *
 *  EXPECTED CONTRACT (see PROGRESS.md; not yet merged at the time of writing):
 *    POST /drawings/{id}/geometry/points  {handles: string[], layout: string}
 *    -> {points: MarkPoint[], missing: string[]}
 */
export async function fetchMarkPoints(
  drawingId: string,
  handles: string[],
  layout: string,
  signal?: AbortSignal,
): Promise<Optional<MarkPoint[]>> {
  if (handles.length === 0) return { data: [], reason: null };
  const result = await optionalRequest<{ points: MarkPoint[] }>(
    `/drawings/${encodeURIComponent(drawingId)}/geometry/points`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ handles, layout }),
      signal,
    },
    "Stored centroids are not exposed by this build of cad-api yet.",
  );
  // The endpoint answers with an envelope; the viewer wants the list. An
  // envelope that arrives without its list is treated as absence rather than
  // as an empty answer — "the route replied with nothing recognisable" and
  // "there are no centroids" are different facts and only one of them is
  // worth acting on.
  if (!result.data) return { data: null, reason: result.reason };
  if (!Array.isArray(result.data.points)) {
    return ABSENT("cad-api answered without a `points` list.");
  }
  return { data: result.data.points, reason: null };
}

// ---------------------------------------------------------------------------
// Text -> parcel — UPLIFT-03 run in reverse
// ---------------------------------------------------------------------------

/** A parcel that contains a text entity.
 *
 *  This is the answer to the failure of 23 August at 23:28: the plot number
 *  `2043` was clicked, "what is the area" was asked, and the agent described
 *  the TEXT entity. Literally correct, and useless. The area a person means
 *  is the parcel the label sits inside, and until something can name that
 *  parcel there is no way to answer the question they asked.
 *
 *  `land_use_verified` is carried and never defaulted to true. A land use
 *  derived from a layer-name pattern is an inference (G4), and a panel that
 *  prints it without that qualifier turns a guess into a caption.
 */
export interface ContainingParcel {
  handle: string;
  layer: string;
  area: number | null;
  /** The unit the area is in, or null when the file declares none. Never
   *  assumed to be metres: 11 of the 16 drawings here are in inches and 3
   *  state nothing at all (G2). */
  area_unit: string | null;
  land_use: string | null;
  land_use_verified: boolean | null;
}

export interface ParcelContext {
  selected: { handle: string; type: string; layer: string; text: string | null };
  contained_by: ContainingParcel[];
  /** The sentence the agent and the panel both show. Comes from the API so
   *  that the wording cannot drift between what the user reads and what the
   *  model is told. */
  note: string | null;
}

/** The parcels containing one entity's anchor point.
 *
 *  Point-in-polygon, not bounding box, and that is the whole value. Measured
 *  on the reference drawing, a bbox test against one school parcel pulls in
 *  42 texts where the polygon test finds 1 — the plots sit on a grid rotated
 *  by about 49 degrees, so an axis-aligned box is far larger than the shape
 *  it stands for. A bbox answer here is not a rough answer, it is a wrong one.
 *
 *  EXPECTED CONTRACT (see PROGRESS.md):
 *    GET /drawings/{id}/entities/{handle}/containing?layout=<layout>
 *    -> ParcelContext
 */
export function fetchParcelContext(
  drawingId: string,
  handle: string,
  layout: string,
  signal?: AbortSignal,
): Promise<Optional<ParcelContext>> {
  return optionalRequest<ParcelContext>(
    `/drawings/${encodeURIComponent(drawingId)}/entities/${encodeURIComponent(
      handle,
    )}/containing?layout=${encodeURIComponent(layout)}`,
    { signal },
    "Point-in-polygon lookup is not available in this build of cad-api yet.",
  );
}

/** Types whose selection should trigger a parcel lookup.
 *
 *  DXF entity types, not layer names — this is the file format's vocabulary,
 *  which is the same in every drawing, and not the drawing's own, which is
 *  not (G1, G10). A viewer that knew which layers meant parcels would be a
 *  viewer that had learned one site plan by heart.
 */
const LABEL_TYPES = new Set(["TEXT", "MTEXT", "ATTRIB", "ATTDEF"]);

export function isLabelType(type: string | null | undefined): boolean {
  return LABEL_TYPES.has((type ?? "").toUpperCase());
}

/** One line of context handed to the agent alongside a selected label.
 *
 *  Short on purpose. The model is being told which object the question is
 *  probably about, not being fed the parcel's record — it can call the tools
 *  for that, and everything it reads that way carries a handle a reader can
 *  check.
 */
export function describeParcelContext(context: ParcelContext): string | null {
  if (context.contained_by.length === 0) return null;
  const parcels = context.contained_by
    .map((p) => `${p.handle} (layer ${p.layer})`)
    .join(", ");
  return (
    `[the selected object is a ${context.selected.type} label` +
    (context.selected.text ? ` reading "${context.selected.text}"` : "") +
    `, and it sits inside ${parcels}. A question about area, size or type is ` +
    `almost certainly about the PARCEL, not about the text. Answer about the ` +
    `parcel and say that what was clicked is its label.]`
  );
}
