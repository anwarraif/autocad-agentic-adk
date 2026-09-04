/**
 * Turning an agent's answer into marks on the drawing.
 *
 * Asked three times in the meeting of 23 August 2026 and absent from the
 * product all three times: *"it should circle the schools"*. What existed was
 * a linkified handle per mention, so an answer naming nine parcels cost nine
 * clicks and nine losses of context. What is wanted is that the objects an
 * answer names are marked the moment the answer lands.
 *
 * This module holds the part of that with no DOM and no React in it: which
 * tokens are worth asking about, how a validated set is split into what can be
 * marked and what cannot, and which of the three mark shapes a given object
 * gets. It is separate from the viewer for one reason — every rule here is a
 * decision someone will want to argue with, and an argument is easier against
 * forty lines of arithmetic than against a component.
 *
 * The counting rules are the load-bearing part, not the drawing. A highlight
 * that quietly drops what it could not mark is a highlight that makes the
 * answer look more complete than it is.
 */

/** Handle-shaped tokens: hex, 2-7 characters.
 *
 *  Nowhere near sufficient on its own — "2024", "DEAD", "292" and "ACAD" all
 *  match, and the agent is explicitly instructed to quote layer names and
 *  counts. Every candidate is therefore checked against the drawing before it
 *  is treated as an object; see `planAnswerHighlight`. Kept here rather than
 *  in the chat panel because the panel and the viewer must agree on what a
 *  candidate is, and two copies of a regex is two chances to disagree.
 */
export const HANDLE_CANDIDATE = /\b([0-9A-Fa-f]{2,7})\b/g;

/** Objects marked individually before the viewer switches to one aggregate
 *  mark for the whole answer.
 *
 *  4,000, raised twice on 24 August 2026 as each number turned out to be a
 *  number rather than a limit. 200 first, after watching what it produced. Asked to show 512 plots, the viewer drew a single dashed
 *  rectangle around the whole neighbourhood — technically honest, and useless:
 *  it says "somewhere in here" about an area the reader can already see. The
 *  same question over 133 plots coloured the plots themselves, which is the
 *  thing worth looking at.
 *
 *  The old ceiling inherited a measurement that does not describe this code
 *  path. What froze the page was RECOLOURING thousands of existing strokes
 *  across a 19,445-group sheet. An answer mark does not recolour anything; it
 *  appends one small node to an overlay group. Region highlighting already
 *  carries 750 on the same page, so this is the number the drawing has been
 *  shown to survive.
 *
 *  Above it nothing is silently dropped: `AnswerPlan.overflow` says how many
 *  went unmarked and the badge prints it (G7).
 */
export const MAX_ANSWER_MARKS = 12000;

/** Objects an answer may name before "show all" stops being offered.
 *
 *  From UPLIFT-12. Fitting the viewport to 300 scattered objects zooms out to
 *  most of the drawing, which is not a useful place to be sent.
 */
export const MAX_ZOOM_TO = 20;

/** Below this on-screen size an object gets a marker circle instead of an
 *  outline of itself, in CSS pixels.
 *
 *  The threshold is a legibility fact, not a property of any drawing: an
 *  outline traced around something six pixels across is a smudge, and the
 *  answer to "which one" has to be visible from where the user is standing.
 */
export const SMALL_OBJECT_PX = 8;

/** Diameter of the marker circle, in CSS pixels — fixed on SCREEN, never in
 *  drawing units.
 *
 *  This is the whole reason the marker works. Janadriyah spans 15.6 km; a
 *  circle 30 m across is a third of a pixel at full extent and invisible
 *  exactly when it is most needed. A 24 px circle is the same 24 px at every
 *  magnification, which is what "circle the schools" has to mean on a drawing
 *  this size.
 */
export const MARKER_DIAMETER_PX = 24;

/** A set of objects an agent's answer named, ready to be marked.
 *
 *  `origin` exists so the viewer can colour these differently from what the
 *  user selected by hand. Being unable to tell "what I picked" from "what the
 *  agent answered" would make the mark actively misleading, so the field is
 *  not decoration.
 */
export interface AnswerHighlight {
  /** Validated: every one of these is an entity drawn in the layout on
   *  screen. Handles that exist elsewhere are counted, not included. */
  handles: string[];
  /** Shown on the badge. Derived from the counts, never from the model's
   *  prose — a label the model wrote could disagree with what is drawn. */
  label: string;
  origin: "agent";
  /** Identifies the answer, so a new one clears the previous marks rather
   *  than accumulating a drawing that gets steadily more marked up. */
  answerId: string;
}

/** What became of every token the answer contained.
 *
 *  The three-way split is the point. A token that is not an entity at all was
 *  never an object reference and must not be counted as a missing one — most
 *  of them are years, counts and layer fragments that merely look hex. A
 *  token that IS an entity but is not drawn here is a real object the user
 *  cannot see, and that is worth saying out loud.
 */
export interface AnswerPlan {
  highlight: AnswerHighlight | null;
  /** Entities named by the answer and drawn in the layout on screen. */
  markable: string[];
  /** Real entities in this drawing that this view does not draw. Reported,
   *  never dropped. */
  elsewhere: string[];
  /** Tokens that resolve to no entity at all. Not reported: they were never
   *  object references, and calling them "missing objects" would invent a
   *  problem out of the word "2024". */
  notEntities: string[];
  /** How many markable objects exceed `MAX_ANSWER_MARKS` and will therefore
   *  be covered by the aggregate mark instead of individually. */
  overflow: number;
  /** True when the answer named few enough objects to offer "show all". */
  canZoomTo: boolean;
}

/** Candidate handles in an answer, in order of appearance, deduplicated.
 *
 *  Case is normalised upward because handles are stored uppercase and the
 *  model writes them either way; the caller checks both forms against the
 *  drawing anyway, but returning one canonical spelling stops the same object
 *  being asked about twice.
 */
export function collectHandleCandidates(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const match of text.matchAll(HANDLE_CANDIDATE)) {
    const token = match[1].toUpperCase();
    if (seen.has(token)) continue;
    seen.add(token);
    out.push(token);
    if (out.length >= MAX_CANDIDATES) break;
  }
  return out;
}

/** Distinct hex-shaped tokens taken from one answer before the rest are
 *  ignored.
 *
 *  Not a real constraint on any answer anyone would read — a reply naming two
 *  thousand distinct objects is a data dump, not a sentence — but every list
 *  sent to an API deserves a ceiling that is written down rather than
 *  discovered. The endpoint behind it accepts 25,000, so this is the
 *  narrower and therefore the operative limit.
 */
export const MAX_CANDIDATES = 2000;

/** Split validated candidates into what can be marked and what must be said.
 *
 *  `isEntity` answers "is this token an object in this drawing at all" and is
 *  expected to come from the API, which is the only thing that knows. `isDrawn`
 *  answers "is it on screen right now" and comes from the rendered document.
 *  Keeping them as two questions is deliberate: collapsing them would make an
 *  object on another layout indistinguishable from the number 300, and the
 *  first deserves a sentence while the second deserves silence.
 */
export function planAnswerHighlight(input: {
  answerId: string;
  candidates: string[];
  isEntity: (handle: string) => boolean;
  isDrawn: (handle: string) => boolean;
}): AnswerPlan {
  const markable: string[] = [];
  const elsewhere: string[] = [];
  const notEntities: string[] = [];

  for (const handle of input.candidates) {
    if (!input.isEntity(handle)) {
      notEntities.push(handle);
    } else if (input.isDrawn(handle)) {
      markable.push(handle);
    } else {
      elsewhere.push(handle);
    }
  }

  const named = markable.length + elsewhere.length;
  if (named === 0) {
    return {
      highlight: null,
      markable,
      elsewhere,
      notEntities,
      overflow: 0,
      canZoomTo: false,
    };
  }

  const overflow = Math.max(0, markable.length - MAX_ANSWER_MARKS);
  return {
    // Above the cap the handles still travel in full. The viewer decides how
    // to DRAW them; throwing them away here would also throw away "show all"
    // and the extents the aggregate mark is computed from.
    highlight: {
      handles: markable,
      label: describeAnswerHighlight(
        markable.length,
        elsewhere.length,
        overflow,
        notEntities.length,
      ),
      origin: "agent",
      answerId: input.answerId,
    },
    markable,
    elsewhere,
    notEntities,
    overflow,
    canZoomTo: markable.length > 0 && markable.length <= MAX_ZOOM_TO,
  };
}

/** The badge text.
 *
 *  Written from the counts rather than from the answer's own words. The model
 *  may say "nine schools"; what is drawn is whatever survived validation, and
 *  a badge that says nine over eight marks is a badge that lies about the
 *  picture it is attached to.
 *
 *  Deliberately says "objects", not what kind of objects. Naming the kind
 *  would mean reading meaning off a layer name in the viewer, which is the
 *  one thing G1 and G10 both forbid — the drawing's vocabulary belongs in
 *  config, and the viewer has none.
 */
export function describeAnswerHighlight(
  marked: number,
  elsewhere: number,
  overflow: number,
  invented: number = 0,
): string {
  const drawn = marked - overflow;
  const parts: string[] = [];
  parts.push(
    overflow > 0
      ? `${marked.toLocaleString()} object${marked === 1 ? "" : "s"} in this answer`
      : `${drawn.toLocaleString()} object${drawn === 1 ? "" : "s"} marked`,
  );
  if (overflow > 0) {
    parts.push(
      `too many to mark one by one — ${overflow.toLocaleString()} not drawn individually`,
    );
  }
  if (elsewhere > 0) {
    parts.push(
      `${elsewhere.toLocaleString()} more not drawn in this view`,
    );
  }
  //  Handles the answer WROTE that are not objects in this drawing.
  //
  //  This module's own header says a highlight that quietly drops what it
  //  could not mark is a highlight that makes the answer look more complete
  //  than it is — and until now `notEntities` was computed here and dropped
  //  on the floor, which is that failure in its own house.
  //
  //  It is not the same thing as `elsewhere`, and merging them would be the
  //  real mistake: an object on another layout EXISTS and is simply off
  //  screen, while these do not exist at all. One is a limit of the view, the
  //  other is the answer being wrong, and a reader deciding whether to trust
  //  a figure needs to be able to tell them apart.
  //
  //  Said as a count and not as a list. The reader cannot do anything with
  //  the invented strings themselves, and printing them would put five more
  //  official-looking addresses on the screen — the opposite of the point.
  if (invented > 0) {
    parts.push(
      `${invented.toLocaleString()} handle${invented === 1 ? "" : "s"} ` +
        `named in the text ${invented === 1 ? "is not an object" : "are not objects"} ` +
        `in this drawing`,
    );
  }
  return parts.join(" · ");
}

/** How one object should be marked at the current magnification.
 *
 *  `sizePx` is the object's largest on-screen dimension. The branch is taken
 *  per object and re-taken on every zoom, because "too small to outline" is a
 *  statement about the view and not about the object.
 */
export type MarkShape = "outline" | "marker";

export function markShapeFor(sizePx: number): MarkShape {
  return sizePx >= SMALL_OBJECT_PX ? "outline" : "marker";
}

/** Where a marker circle goes.
 *
 *  `polygon_centroid` is the right answer and `bbox_centre` is not: on a
 *  neighbourhood boundary polygon the two are 72.7 m apart, which puts the
 *  mark in a neighbouring plot. So the stored centroid is used whenever the
 *  API can supply it.
 *
 *  The fallback is allowed only here, and only because of which branch this
 *  is. A marker circle is drawn precisely when the object measures under
 *  `SMALL_OBJECT_PX` on screen, so its whole bounding box is under eight
 *  pixels across and the two candidate points cannot be more than that far
 *  apart *in the picture the user is looking at*. The 72.7 m case is a large
 *  polygon, and a large polygon takes the outline branch, where no centre
 *  point is used at all.
 *
 *  A large object with no stored centroid therefore gets no marker rather
 *  than a guessed one — `null`, with the caller free to say why.
 */
export function markerCentre(input: {
  polygonCentroid: [number, number] | null;
  bboxCentre: [number, number] | null;
  shape: MarkShape;
}): { point: [number, number]; exact: boolean } | null {
  if (input.polygonCentroid) {
    return { point: input.polygonCentroid, exact: true };
  }
  if (input.shape === "marker" && input.bboxCentre) {
    return { point: input.bboxCentre, exact: false };
  }
  return null;
}
