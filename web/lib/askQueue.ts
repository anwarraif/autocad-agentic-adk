/**
 * What happens when a second question is asked while the first is running.
 *
 * A15 asked for the behaviour to be DECIDED — queue, cancel, or parallel —
 * and `docs/A15-CONCURRENT-QUESTIONS.md` decided it: queue, one deep, and say
 * so out loud. This file is the saying-so, expressed as arithmetic rather
 * than as a component, for the same reason `answerHighlight.ts` is: every
 * rule below is one somebody will want to argue with, and an argument is
 * easier against a pure function than against a panel.
 *
 * What was measured on 24 August 2026, and what this replaces: the second
 * question was **dropped in silence**. The button was pressed, nothing was
 * sent, the text stayed in the box, and no message of any kind appeared. The
 * lock existed and was correct; it simply had no surface. From where the user
 * stands, that is indistinguishable from the app being broken — which is the
 * exact confusion the meeting of 23 August ended in.
 *
 * Three rules, and none of them is about being polite:
 *
 *   1. An accepted question is ACKNOWLEDGED. The user already pressed send;
 *      the screen owes them a record of it.
 *   2. A refused question is REFUSED IN WORDS, and its text is left where the
 *      user typed it. Silence plus an emptied box is the worst of both.
 *   3. The queue is ONE. An unbounded queue on a tool whose answers take tens
 *      of seconds does not remove the confusion, it relocates it.
 */

/** How many questions may wait behind the running one.
 *
 *  One, from A15. The number is small on purpose: each answer here costs
 *  between two seconds and the route's three-minute ceiling, so a queue of
 *  five is a promise the user cannot supervise and will not remember making.
 */
export const QUEUE_DEPTH = 1;

/** A tool call as the stream reports it. Mirrors the chat panel's shape so
 *  the two cannot drift; kept here because the sentences below are built from
 *  it and they are what this module exists to test. */
export interface AskStep {
  name: string;
  args: string;
  done: boolean;
}

/** Everything the submit decision depends on. Deliberately not the React
 *  state — two booleans and a string are enough to decide, and a function
 *  that took the panel's state could not be exercised without the panel. */
export interface AskQueueState {
  /** True while a turn is in flight. */
  running: boolean;
  /** The question already waiting, or `null`. At most one, see `QUEUE_DEPTH`. */
  queued: string | null;
}

/** What the panel should do with a submission.
 *
 *  `refuse` carries the sentence rather than an error code because the
 *  sentence is the feature: the whole defect being fixed is that the refusal
 *  had no words. `keepDraft` says the text must stay in the box — a refused
 *  question that also loses its text is the silent drop again, with extra
 *  steps.
 */
export type SubmitDecision =
  | { action: "start"; text: string }
  | { action: "queue"; text: string; note: string }
  | { action: "refuse"; reason: string; keepDraft: true }
  | { action: "ignore" };

/** The one sentence a third question gets.
 *
 *  It names the cap, what is occupying it, and the two ways out. A refusal
 *  that says only "please wait" leaves the reader to guess whether the app
 *  heard them at all.
 */
export function refusalSentence(): string {
  return (
    "Not sent — one question is running and one is already waiting, and this " +
    "queue holds one. Wait for the running answer, or cancel either of them. " +
    "Your text is still in the box."
  );
}

/** The acknowledgement a queued question gets, shown where the answer will
 *  land rather than as a toast that scrolls away. */
export function queuedSentence(): string {
  return "Waiting — it will be sent as soon as the running question finishes.";
}

export function decideSubmit(
  state: AskQueueState,
  raw: string,
): SubmitDecision {
  const text = raw.trim();
  // An empty submission is not a refusal: nothing was asked, so there is
  // nothing to explain. Saying "not sent" for a blank box would train the
  // reader to ignore the sentence that matters.
  if (!text) return { action: "ignore" };
  if (!state.running) return { action: "start", text };
  if (state.queued === null) {
    return { action: "queue", text, note: queuedSentence() };
  }
  return { action: "refuse", reason: refusalSentence(), keepDraft: true };
}

// ---------------------------------------------------------------------------
// How long is too long
// ---------------------------------------------------------------------------
//
// These come from A10, which measured this stack rather than guessing at it,
// and they are thresholds for what the SCREEN SAYS — never for cutting
// anything off. Nothing here cancels a turn; only the user does that.

/** Past this, a wait stops being ordinary and the screen says so.
 *
 *  20 s, from A10's client-side measurement: loading the reference drawing
 *  froze the viewer for 20-40 s at a stretch. Below that number a wait is
 *  indistinguishable from the page's own known cost, so calling it slow would
 *  be crying wolf.
 */
export const SLOW_AFTER_MS = 20_000;

/** Past this, no measurement on this stack explains the wait any more.
 *
 *  60 s, chosen against A10's slowest single render — 47,825 ms for one
 *  layout, the worst number in the whole stack. A turn still running a full
 *  minute later has exceeded the slowest thing anyone here has measured, and
 *  that is worth telling the reader rather than leaving them to wonder.
 */
export const UNEXPLAINED_AFTER_MS = 60_000;

/** The route's own ceiling: `AGENT_TIMEOUT_MS` in
 *  `web/app/api/agent/stream/route.ts`, whose default is 180 s.
 *
 *  Duplicated as a number rather than imported because that value lives in a
 *  server module and this one runs in the browser. If the deployment raises
 *  the env var, this line becomes a lower bound and the wording below still
 *  holds — it says the route "should have" given up, not that it has.
 */
export const ROUTE_TIMEOUT_MS = 180_000;

/** `1:04`, `12s`. Seconds below a minute, because at that scale the reader is
 *  judging "is it moving" and not reading a clock. */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

/** The tool being called right now, or `null` when none is open.
 *
 *  The last step that has not come back — not the last step overall. A
 *  finished call is a fact about the past and naming it as "running" would be
 *  the same category of lie as the missing surface this file replaces.
 */
export function currentStep(steps: AskStep[]): AskStep | null {
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    if (!steps[i].done) return steps[i];
  }
  return null;
}

/** The headline over a running turn.
 *
 *  A15's second requirement, stated exactly: *"the running one names the tool
 *  it is calling, not '1 step'."* The data is already there — the SOURCES row
 *  under a finished answer is built from the same stream — so this needs no
 *  new endpoint, only that it be shown while it is still true.
 *
 *  Three states, and they are genuinely different facts:
 *   - nothing read yet, which is where a hung agent sits;
 *   - a named tool in flight, which is the ordinary case;
 *   - every call returned and text being written, which is the last stretch.
 */
export function runningHeadline(steps: AskStep[], elapsedMs: number): string {
  const clock = formatElapsed(elapsedMs);
  const done = steps.filter((s) => s.done).length;
  const open = currentStep(steps);
  if (open) {
    return `Reading ${open.name} · ${clock}`;
  }
  if (steps.length === 0) {
    // Named as an absence rather than as progress. "Starting…" would imply
    // something is underway that nothing has yet confirmed.
    return `No tool called yet · ${clock}`;
  }
  return `Writing the answer · ${done} tool${done === 1 ? "" : "s"} read · ${clock}`;
}

/** The extra sentence a long wait earns, or `null` while the wait is ordinary.
 *
 *  Every one of these says what is known and what is not. None of them claims
 *  the agent is stuck: this module cannot tell, and pretending otherwise
 *  would be the mirror image of the silence it is here to remove.
 */
export function progressNote(elapsedMs: number): string | null {
  if (elapsedMs >= ROUTE_TIMEOUT_MS) {
    return (
      "Past the route's own timeout. It should have given up by now, so if " +
      "this line stays the connection is stuck — cancelling is the way out."
    );
  }
  if (elapsedMs >= UNEXPLAINED_AFTER_MS) {
    return (
      "Longer than the slowest read ever measured on this stack (47.8s). " +
      "Still connected; cancel if a narrower question would do."
    );
  }
  if (elapsedMs >= SLOW_AFTER_MS) {
    return "Longer than usual, but within what has been measured here before.";
  }
  return null;
}

/** What the log says after the user cancels.
 *
 *  It names who cancelled — the user — because A15's argument against silent
 *  cancellation is precisely that people conclude their answer "went
 *  missing". It also states the one thing the browser cannot promise: the
 *  agent may keep working server-side for a while yet.
 */
export function cancelSentence(
  kind: "running" | "queued",
  elapsedMs: number,
): string {
  if (kind === "queued") {
    return "You cancelled the waiting question. It was never sent.";
  }
  return (
    `You cancelled this question after ${formatElapsed(elapsedMs)}. ` +
    "No answer was produced; the agent may take a moment to stop on its side."
  );
}

/** A queued question, shortened for a one-line chip without losing its sense.
 *
 *  Cut on a word boundary and marked with an ellipsis, so a truncated
 *  question reads as truncated rather than as a question someone typed badly.
 */
export function shortenQuestion(text: string, max = 90): string {
  const clean = text.trim().replace(/\s+/g, " ");
  if (clean.length <= max) return clean;
  const cut = clean.slice(0, max);
  const space = cut.lastIndexOf(" ");
  return `${(space > max * 0.6 ? cut.slice(0, space) : cut).trimEnd()}…`;
}
