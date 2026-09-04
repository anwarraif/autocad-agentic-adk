/**
 * Which coverage outline the camera is allowed to frame with.
 *
 * The map's geometry and its H3 cells are two SEPARATE requests. On a drawing
 * change the geometry lands first, so for a while `data` is the new drawing
 * and `cells` is still the old one -- and the outline the camera framed was
 * the previous drawing's site boundary.
 *
 * That is not merely a wrong camera for one frame. `frameWithOutline` reports
 * `basis: "outline"` either way, and the fit effect keys on that basis, so once
 * the camera had been fitted to the stale outline the key stopped changing
 * when the RIGHT outline arrived -- and it never refitted. Measured: selecting
 * Janadriyah loaded 3,156 outlines and 60 layers correctly while the view
 * stayed parked over Sedra, about 19 km west.
 *
 * A box has to carry the drawing it belongs to, or it cannot be checked. This
 * is the same identity rule `stillWanted` uses for in-flight pages, applied to
 * a payload rather than a request.
 *
 * Plain JavaScript on purpose: it is imported by the viewer AND by a
 * `node --test` guard, and this project has no JS test runner to add one for.
 *
 * @param {{drawing_id?: string} | null | undefined} data   the geometry payload
 * @param {{drawing_id?: string, coverage_outline?: unknown} | null | undefined} cells the /geo payload
 * @returns {unknown | null} the outline to frame with, or null when it belongs elsewhere
 */
export function outlineForDrawing(data, cells) {
  if (!data || !cells) return null;
  if (!data.drawing_id || !cells.drawing_id) return null;
  if (data.drawing_id !== cells.drawing_id) return null;
  return cells.coverage_outline ?? null;
}
