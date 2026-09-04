/** Types for the plain-JS identity rule shared with the `node --test` guard.
 *
 *  Generic over the outline so the caller keeps its own precise type rather
 *  than being handed `unknown` and having to assert it back. The rule cares
 *  only about the two `drawing_id`s; the outline it passes through untouched.
 */
export declare function outlineForDrawing<T>(
  data: { drawing_id?: string } | null | undefined,
  cells: { drawing_id?: string; coverage_outline?: T | null } | null | undefined,
): T | null;
