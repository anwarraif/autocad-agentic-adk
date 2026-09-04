/** Types for the one size rule, shared with the `node --test` guard. */
export interface Size {
  width: number;
  height: number;
}
export declare function isMeasured(size: Size | null | undefined): boolean;
export declare function nextSize(
  prev: Size | null,
  measured: Size | null | undefined,
): Size | null;
export declare function isStale(
  prev: Size | null,
  measured: Size | null | undefined,
): boolean;
