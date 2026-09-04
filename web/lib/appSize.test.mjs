import { test } from "node:test";
import assert from "node:assert/strict";
import { isMeasured, isStale, nextSize } from "./appSize.mjs";

const WIDE = { width: 1707, height: 811 };
const NARROW = { width: 872, height: 414 };

test("a stale size never survives a resize", () => {
  // The reported symptom: the window is full width, the app is laid out at
  // about 870 px, and it does not recover. Whatever produced the narrow
  // measurement, holding on to it after a wider one arrives is the defect.
  assert.deepEqual(nextSize(NARROW, WIDE), WIDE);
  assert.equal(isStale(NARROW, WIDE), true);
  // ...and in the other direction, which is a real resize too.
  assert.deepEqual(nextSize(WIDE, NARROW), NARROW);
});

test("an unusable measurement cannot overwrite a good one", () => {
  // A pane hidden behind a tab reports 0x0. Believing that is how the map
  // comes back blank when the tab is shown again.
  for (const bad of [{ width: 0, height: 0 }, { width: 100, height: 0 }, null,
                     { width: NaN, height: 10 }]) {
    assert.deepEqual(nextSize(WIDE, bad), WIDE, JSON.stringify(bad));
    assert.equal(isStale(WIDE, bad), false);
  }
});

test("the first usable measurement is adopted", () => {
  assert.deepEqual(nextSize(null, WIDE), WIDE);
  assert.equal(nextSize(null, { width: 0, height: 0 }), null);
});

test("an identical measurement is not a change", () => {
  const held = { ...WIDE };
  assert.equal(nextSize(held, { ...WIDE }), held, "same size keeps the same object");
  assert.equal(isStale(held, { ...WIDE }), false);
});

test("isMeasured refuses what cannot be laid out", () => {
  assert.equal(isMeasured(WIDE), true);
  assert.equal(isMeasured({ width: 0, height: 5 }), false);
  assert.equal(isMeasured(undefined), false);
});
