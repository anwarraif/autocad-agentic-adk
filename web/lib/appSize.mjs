/**
 * The one rule for adopting a measured size.
 *
 * This app has now been given three different fixes for "how wide is it", in
 * three different places: MapLibre's canvas stuck at 300x150, deck's child
 * wrapper collapsed to the same, and the pane reporting a size that never
 * updated. Each was patched where it showed, which is how a fourth site
 * appears. The rule belongs in one place and every consumer reads it.
 *
 * The failure mode is always the same shape: a STALE size survives a change.
 * So the rule is stated as its opposite -- a measurement that differs is
 * always adopted -- and the guard asserts exactly that.
 *
 * Plain JavaScript because the app and a `node --test` guard both import it.
 *
 * @typedef {{width: number, height: number}} Size
 */

/** Whether a measurement is usable at all. Zero means "not laid out yet". */
export function isMeasured(size) {
  return (
    !!size &&
    Number.isFinite(size.width) &&
    Number.isFinite(size.height) &&
    size.width > 0 &&
    size.height > 0
  );
}

/**
 * The size to hold after a measurement.
 *
 * @param {Size|null} prev      what is currently held
 * @param {Size|null} measured  what was just observed
 * @returns {Size|null}
 */
export function nextSize(prev, measured) {
  // An unusable measurement is not evidence of anything, so it cannot
  // overwrite a good one -- a pane hidden behind a tab reports 0x0, and
  // discarding the real size there is how a map comes back blank.
  if (!isMeasured(measured)) return prev;
  if (!isMeasured(prev)) return measured;
  // ANY difference wins. Rounding to a tolerance was tempting and is the trap:
  // a stale width is exactly the case where the difference is real and the
  // holder believes it is not.
  if (prev.width !== measured.width || prev.height !== measured.height) {
    return measured;
  }
  return prev;
}

/** True when what is held disagrees with what was measured. */
export function isStale(prev, measured) {
  return isMeasured(measured) && nextSize(prev, measured) !== prev;
}
