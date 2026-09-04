import { test } from "node:test";
import assert from "node:assert/strict";
import { outlineForDrawing } from "./outlineForDrawing.mjs";

const SEDRA = "ab83d2caac1f899d";
const JANADRIYAH = "596212db022a3397";
const OUTLINE = { type: "Polygon", coordinates: [[[46.72, 24.83]]] };

test("the camera never frames an outline belonging to another drawing", () => {
  // The measured failure: geometry has switched to Janadriyah, the /geo
  // payload is still Sedra's, and the two sites are about 19 km apart.
  assert.equal(
    outlineForDrawing(
      { drawing_id: JANADRIYAH },
      { drawing_id: SEDRA, coverage_outline: OUTLINE },
    ),
    null,
    "a box belonging to another drawing must not reach the camera",
  );
});

test("the matching outline is used", () => {
  assert.equal(
    outlineForDrawing(
      { drawing_id: JANADRIYAH },
      { drawing_id: JANADRIYAH, coverage_outline: OUTLINE },
    ),
    OUTLINE,
  );
});

test("an unidentifiable payload is refused rather than trusted", () => {
  // Absence is not a match. A payload that cannot say which drawing it
  // describes is exactly the one that must not steer the camera.
  assert.equal(outlineForDrawing({ drawing_id: JANADRIYAH }, { coverage_outline: OUTLINE }), null);
  assert.equal(outlineForDrawing({}, { drawing_id: JANADRIYAH, coverage_outline: OUTLINE }), null);
  assert.equal(outlineForDrawing(null, { drawing_id: JANADRIYAH }), null);
  assert.equal(outlineForDrawing({ drawing_id: JANADRIYAH }, null), null);
});

test("a matching payload with no outline yet yields null, not undefined", () => {
  assert.equal(outlineForDrawing({ drawing_id: SEDRA }, { drawing_id: SEDRA }), null);
});
