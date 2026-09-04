/**
 * Client for the `h3_vicinity` recipe — what is near something, as a shape.
 *
 * The question this exists to draw was relayed from Dr. Adel: *how many
 * houses or villas are within 100 metres of the school?* The recipe already
 * answers it in numbers. What a map adds is the catchment itself, and the one
 * rule that makes drawing it honest:
 *
 * **Draw the disk that was counted, never a circle at the nominal radius.**
 * A hexagonal neighbourhood is not a circle. At resolution 11 two rings reach
 * 99.3 m from cell centre to cell centre and 128.0 m to the disk's corners,
 * so a parcel 120 m away is counted if it lies towards a corner and not if it
 * lies towards a flat. A circle drawn at "100 m" would be a different shape
 * from the one the count came from, and every parcel in the gap between them
 * would be visible evidence that the number is wrong — when the number is
 * right and the circle is the lie. So the boundary drawn here is the server's
 * own dissolved outline, and the reach it actually achieved is printed beside
 * it.
 *
 * It rides `run_analysis` like every other recipe. There is no vicinity
 * endpoint and this file does not want one.
 */

import { API_BASE, ApiError } from "./api";

export interface VicinityBand {
  ring: number;
  from_m: number;
  to_m: number;
  counts: Record<string, number>;
  total: number;
}

export interface VicinityOutline {
  type: "Feature";
  geometry: { type: "MultiPolygon"; coordinates: [number, number][][][] };
  properties: { cells: number; note: string };
}

export interface VicinitySubject {
  handle: string;
  layer: string;
  land_use: string | null;
  /** Cells the subject itself occupies. The rings start from all of them, so
   *  the reach is measured from the footprint rather than from a centre. */
  own_cells: number;
  disk_cells: number;
  counts: Record<string, number>;
  area: Record<string, number>;
  area_unit: string | null;
  network: Record<string, unknown>;
  /** A SAMPLE of the neighbours, capped by the server. Never the whole set —
   *  see `limits.neighbours_listed_per_subject`, and never treat its length
   *  as the count: `neighbours_total` is the count. */
  neighbours_sample: string[];
  neighbours_total: number;
  bands: VicinityBand[] | null;
  outline: VicinityOutline | null;
}

export interface VicinityResponse {
  recipe: string;
  drawing_id: string;
  layout: string;
  params_used: Record<string, unknown>;
  subject: string;
  chosen_by: string;
  subjects: VicinitySubject[];
  subjects_without_cells: unknown[];
  method: {
    resolution: number;
    stored_resolution: number;
    rings: number;
    cell_edge_m: number;
    step_between_cells_m: number;
    /** What k actually reached, centre to centre. The number to quote. */
    reach_centre_to_centre_m: number;
    /** And to the disk's corners, which is further. Both, always. */
    reach_to_disk_corner_m: number;
    how: string;
  };
  limits: {
    rings: number;
    subjects: number;
    neighbours_listed_per_subject: number;
  };
  caveat: string;
  not_measured: string;
  counted_from: string;
  outline_note: string;
  scope_note: string;
  network_note: string | null;
  exact_check: Record<string, unknown> | null;
}

export interface VicinityQuery {
  layout: string;
  /** A handle, or a land use such as `education` to take every parcel of it. */
  subject: string;
  k: number;
  outline?: boolean;
  bands?: boolean;
}

export async function fetchVicinity(
  drawingId: string,
  query: VicinityQuery,
  signal?: AbortSignal,
): Promise<VicinityResponse> {
  const url =
    `${API_BASE}/drawings/${encodeURIComponent(drawingId)}/analyses/h3_vicinity` +
    `?layout=${encodeURIComponent(query.layout)}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      subject: query.subject,
      k: query.k,
      outline: query.outline ?? true,
      bands: query.bands ?? true,
    }),
    signal,
  });
  if (!res.ok) {
    let body: { error?: string; message?: string; hint?: string; detail?: unknown } = {};
    try {
      body = await res.json();
    } catch {
      /* a non-JSON body is itself the message */
    }
    throw new ApiError(
      body.error ?? `HTTP_${res.status}`,
      body.message ??
        (typeof body.detail === "string"
          ? body.detail
          : `${res.status} ${res.statusText}`),
      body.hint ?? "",
    );
  }
  return (await res.json()) as VicinityResponse;
}

/** Every ring of every subject's outline, flattened for a polygon layer.
 *
 *  A MultiPolygon's first ring is its exterior and the rest are holes, which
 *  is exactly what deck's `getPolygon` takes, so the nesting is preserved
 *  rather than flattened away — a disk with a hole in it is a real shape here
 *  (a subject whose own cells are excluded from its own ring band would leave
 *  one), and drawing the hole as a second filled blob would be wrong.
 */
export interface CatchmentRing {
  handle: string;
  rings: [number, number][][];
}

export function catchmentRings(result: VicinityResponse | null): CatchmentRing[] {
  if (!result) return [];
  const out: CatchmentRing[] = [];
  for (const subject of result.subjects) {
    for (const polygon of subject.outline?.geometry.coordinates ?? []) {
      out.push({ handle: subject.handle, rings: polygon });
    }
  }
  return out;
}

/** The counts across every subject, for a one-line summary.
 *
 *  Summed, and deliberately NOT deduplicated: two schools 80 m apart share
 *  neighbours, and a parcel in both disks is counted twice here. That is the
 *  right total for "how many houses are near a school" asked per school, and
 *  the wrong one for "how many houses are near any school". The UI says which
 *  it is showing rather than leaving the reader to assume.
 */
export function totalCounts(result: VicinityResponse | null): Record<string, number> {
  const out: Record<string, number> = {};
  for (const subject of result?.subjects ?? []) {
    for (const [use, n] of Object.entries(subject.counts)) {
      out[use] = (out[use] ?? 0) + n;
    }
  }
  return out;
}
