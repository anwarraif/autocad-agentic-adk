"use client";

/**
 * Chat with the ADK agent about the open drawing.
 *
 * Deliberately degradable: if the agent container is not running, this panel
 * says so and the rest of the viewer keeps working. The agent is the most
 * fragile part of the stack (external model API, MCP process, network) and
 * the drawing viewer must not depend on it.
 *
 * Handles that the agent mentions are turned into buttons that select the
 * entity in the viewer — that link between the answer and the geometry is
 * what separates this from a chatbot that could be making things up.
 *
 * ONE QUESTION AT A TIME, AND IT SAYS SO (A15). Asking again while an answer
 * is still being produced used to do nothing at all: the button was pressed,
 * nothing was sent, the text stayed in the box, and no message appeared. The
 * lock was right and invisible, which is worse than any of the three
 * behaviours the question was asking between — a hung agent and a working one
 * looked identical, and on this stack the ambiguous window is tens of
 * seconds, repeatedly. It now queues, one deep, and every part of that is on
 * screen: the waiting question, the tool the running one is calling, an
 * explicit cancel, and a refusal in words when a third arrives. The rules are
 * in `lib/askQueue.ts` so they can be read without reading this file.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { foldedHandlesNote, languageOf } from "@/lib/language";
import { API_BASE } from "@/lib/api";
import {
  HANDLE_CANDIDATE,
  collectHandleCandidates,
} from "@/lib/answerHighlight";
import {
  DEFAULT_EXPORT_RES,
  EXPORT_RESOLUTIONS,
  EXPORT_RESOLUTION_LABELS,
  SHOW_EXPORT_RES_PICKER,
} from "@/lib/api";
import {
  QUEUE_DEPTH,
  cancelSentence,
  currentStep,
  decideSubmit,
  progressNote,
  queuedSentence,
  runningHeadline,
} from "@/lib/askQueue";

interface Message {
  role: "user" | "agent" | "system";
  text: string;
  /** Stable identity, so a question that is waiting can be found again when
   *  it starts or when the user cancels it. Index positions cannot do this:
   *  the log grows underneath a queued question while it waits. */
  id?: string;
  /** Accepted, shown, and not yet sent. The screen owes an acknowledgement to
   *  anyone who pressed send; leaving the text in the box was the defect. */
  pending?: boolean;
  /** Withdrawn before it was ever sent. Kept in the log rather than deleted —
   *  a question that vanishes is the silence this panel is fixing. */
  cancelled?: boolean;
  /** What the agent did, in order, while producing this answer. Kept with
   *  the message rather than in transient state so the trace survives the
   *  next question — the reader can still audit an old answer. */
  steps?: { name: string; args: string; done: boolean }[];
  /** Derived from the steps that actually executed. Not the model's word for
   *  where its facts came from: its record of what it read. */
  citations?: { tool: string; detail: string }[];
  ungrounded?: boolean;
  /** The marks this answer put on the drawing, kept so they can be put back.
   *
   *  Only present on an answer that actually marked something: a failed,
   *  ungrounded or scope-breaching turn never reached the marking path, and
   *  offering to replay marks it never made would be an invitation to a
   *  claim the trace does not support.
   *
   *  Scoped to the document it was validated against. Handles are small hex
   *  numbers and are per-drawing, so the same token means different objects
   *  in two files; replaying an answer over another drawing would mark
   *  unrelated geometry, which is exactly the leak G10 forbids. */
  answer?: AnswerMarks;
}

/** One finished answer's marks, in the form the marking path takes them.
 *
 *  Deliberately the same four arguments as `onAnswer`, unchanged: replaying
 *  goes back through the one path that validates handles against the API and
 *  the rendered document, rather than a second, cheaper path that would
 *  eventually disagree with the first. */
interface AnswerMarks {
  id: string;
  text: string;
  readHandles: string[];
  groups: { label: string; handles: string[] }[];
  /** What the handles were validated against. Both, because a paper sheet
   *  draws a different set of objects than the model space behind it. */
  drawingId: string;
  layout: string;
}

interface Props {
  drawingId: string;
  drawingName: string;
  /** The layout on screen. Sent with every turn because it is the default
   *  scope for any count the agent reports — without it, "how many" is
   *  ambiguous by a factor of eight on this drawing. */
  layout: string;
  /** The layout the visible entities are STORED under — what the agent must
   *  filter by. Differs from `layout` on a paper sheet. See page.tsx. */
  scopeLayout: string;
  /** True if the handle exists in the currently rendered layout. */
  isHandleInDrawing: (handle: string) => boolean;
  onHandleMentioned: (handle: string) => void;
  /** A region selection handed over from the viewer.
   *
   *  Only handles and counts travel — never the entities. The model is given
   *  *access* to the selection, not the selection itself: it calls
   *  `describe_selection` for the shape of it and `get_entity` for any object
   *  it wants in detail. That is what keeps every claim traceable to a handle
   *  and what stops a 1,284-object selection from filling the context window
   *  with rows nobody asked for. */
  selection: { selectionId: string; total: number; summary: string } | null;
  onClearSelection: () => void;
  /** One line about what the selected object is a label FOR, when it is a
   *  label at all.
   *
   *  It travels with every turn rather than being asked for, because the
   *  model has no way to discover it: the connection between a plot number
   *  and the plot is point-in-polygon over stored geometry, and nothing in
   *  the text of a plot number suggests a parcel exists. Without it the
   *  answer to "what is the area" is the area of a piece of text — which is
   *  what happened, on 23 August 2026, and is why this prop is here. */
  selectionNote: string | null;
  /** A finished answer, handed over so the objects it names can be marked in
   *  the drawing without the user clicking anything.
   *
   *  Asked for three times in the meeting of 23 August 2026 — *"it should
   *  circle the schools"* — and absent all three times. What existed was the
   *  chip below: one button per mention, so an answer naming nine parcels was
   *  nine clicks and nine losses of place. The chips stay, because clicking
   *  one still means "take me to this one"; what changes is that they are no
   *  longer the only way to see where the answer is pointing.
   *
   *  The raw text travels rather than a handle list. Deciding which tokens
   *  are objects needs the drawing and the API, and neither belongs in a chat
   *  panel — see `planAnswerHighlight`. */
  onAnswer: (
    text: string,
    answerId: string,
    readHandles: string[],
    groups: { label: string; handles: string[] }[],
  ) => void;
  /** Called when a new question is asked, so the previous answer's marks come
   *  off before the next ones go on. Two answers' worth of marks in one
   *  colour would be one answer that appears to name twice as much as it
   *  does. */
  onAnswerCleared: () => void;
}

// The handle pattern lives in lib/answerHighlight.ts, next to the rest of the
// rules for turning an answer into marks. Two copies of it would be two
// chances for the chips and the marks to disagree about which words in an
// answer are objects, and they must agree: they are two views of one claim.

//: The resolution an exported answer's cell overlay is aggregated at, and the
//  rings drawn around a vicinity answer. Both are the defaults the recipe and
//  the map already use — 2 rings is about 100 m at resolution 11 — so a file
//  exported from an answer shows the same neighbourhood the answer counted.

const EXPORT_RINGS = 2;

/** The handles an answer's file should contain.
 *
 *  The same two sources the marks were built from, in the same order: what
 *  the answer WROTE and what the agent READ. Reconstructed here rather than
 *  handed back from the viewer because the viewer holds a validated PLAN —
 *  by then the handles that could not be marked have been dropped — and an
 *  export built from that would silently lose every object that is real but
 *  not drawn in the layout on screen.
 *
 *  Nothing is validated here either. The export route filters to objects that
 *  exist and have a boundary it will vouch for, which is the same test the
 *  marking path applies, so the file and the marks agree without this file
 *  knowing how either works. */
function exportHandles(marks: AnswerMarks): string[] {
  const written = collectHandleCandidates(marks.text);
  const seen = new Set(written);
  return [...written, ...marks.readHandles.filter((h) => !seen.has(h))];
}

/** Whether this answer came from the vicinity recipe.
 *
 *  Read off the TOOL TRACE, never off the prose. The trace is what the turn
 *  actually did; the sentence is what the model wrote about it, and the two
 *  can differ. `run_analysis` records the recipe name in its arguments, which
 *  is enough to know a catchment is worth drawing — the ring count is not in
 *  there, so the file uses the recipe's own default and says so. */
function isVicinityAnswer(message: Message): boolean {
  return (message.steps ?? []).some(
    (step) =>
      step.name === "run_analysis" && (step.args || "").includes("h3_vicinity"),
  );
}

/** UUID that also works outside a secure context.
 *
 *  `crypto.randomUUID` is defined only on secure origins. `http://localhost`
 *  qualifies, so the documented compose flow is fine — but reaching the app
 *  over a LAN IP left it undefined, and calling it during render crashed the
 *  whole panel. */
function newRequestId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `req-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function ChatPanel({
  drawingId,
  drawingName,
  layout,
  scopeLayout,
  isHandleInDrawing,
  onHandleMentioned,
  selection,
  onClearSelection,
  selectionNote,
  onAnswer,
  onAnswerCleared,
}: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState<{
    steps: { name: string; args: string; done: boolean }[];
    text: string;
  } | null>(null);
  /** The one question waiting behind the running one. `QUEUE_DEPTH` is 1, and
   *  a third submission is refused in words rather than dropped. */
  const [queued, setQueued] = useState<{ id: string; text: string } | null>(null);
  /** Milliseconds the running turn has been going. Drives the clock, the
   *  "longer than usual" note, and the sentence printed on cancel. */
  const [elapsed, setElapsed] = useState(0);
  const sessionId = useRef<string>(newRequestId());
  /** Aborts the running turn. The route's own `cancel()` stops pulling from
   *  the agent when the browser lets go, so this is a real cancellation and
   *  not just a hidden tab. */
  const abortRef = useRef<AbortController | null>(null);
  const startedAtRef = useRef<number>(0);
  /** Which answer's marks are on the drawing right now, or null.
   *
   *  Kept here rather than read back from the viewer because the viewer holds
   *  a validated plan, not an answer: by the time the marks exist, the handles
   *  that could not be marked have already been dropped out of them, and the
   *  log needs to point at the answer as it was asked, not at what survived. */
  const [markedAnswerId, setMarkedAnswerId] = useState<string | null>(null);
  /** Cell size for the answer export. One choice for the panel rather than
   *  one per message: the reader is picking how they want to LOOK at answers,
   *  and making them re-pick on every line would be a setting pretending to
   *  be a per-item control. */
  const [exportRes, setExportRes] = useState<number>(DEFAULT_EXPORT_RES);

  // A clock that only exists while something is running. Restarted from the
  // turn's own start time rather than accumulated, so a re-render cannot make
  // the wait look shorter than it was.
  useEffect(() => {
    if (!busy) {
      setElapsed(0);
      return;
    }
    const started = startedAtRef.current || Date.now();
    setElapsed(Date.now() - started);
    const timer = setInterval(() => setElapsed(Date.now() - started), 1000);
    return () => clearInterval(timer);
  }, [busy]);

  const runTurn = useCallback(
    async (text: string) => {
      startedAtRef.current = Date.now();
      setBusy(true);
      // The previous answer's marks come off as the next question goes out,
      // not when the next answer arrives. Otherwise the drawing keeps claiming
      // to show the answer to a question that has already been replaced, for
      // however long the turn takes — and a turn here can take a minute.
      onAnswerCleared();
      setMarkedAnswerId(null);
      // The live turn is its own state so the steps can be shown as they
      // happen; it is folded into `messages` when the turn ends.
      setLive({ steps: [], text: "" });

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await fetch("/api/agent/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            message: text,
            drawing_id: drawingId,
            drawing_name: drawingName,
            layout,
            scope_layout: scopeLayout,
            session_id: sessionId.current,
            selection: selection
              ? {
                  selection_id: selection.selectionId,
                  total: selection.total,
                  summary: selection.summary,
                }
              : undefined,
            selection_note: selectionNote ?? undefined,
          }),
        });
        if (!response.body) throw new Error("no stream");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        const steps: { name: string; args: string; done: boolean }[] = [];
        let answer = "";
        let citations: { tool: string; detail: string }[] = [];
        let ungrounded: string | null = null;
        let readHandles: string[] = [];
        let readGroups: { label: string; handles: string[] }[] = [];
        let breach: string | null = null;
        /** Context, not a doubt: why the answer names a layout other than
         *  the one on screen. It must NOT join `ungrounded` or `breach` —
         *  those two suppress marking, and an answer about the space a
         *  sheet draws has objects worth showing. */
        let scopeNote: string | null = null;
        let failure: { message: string; hint: string } | null = null;

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";
          for (const frame of frames) {
            const line = frame.split("\n").find((l) => l.startsWith("data:"));
            if (!line) continue;
            let event: Record<string, unknown>;
            try {
              event = JSON.parse(line.slice(5).trim());
            } catch {
              continue;
            }
            if (event.type === "tool_call") {
              steps.push({
                name: String(event.name),
                args: String(event.args ?? ""),
                done: false,
              });
              setLive({ steps: [...steps], text: answer });
            } else if (event.type === "tool_result") {
              const open = [...steps]
                .reverse()
                .find((x) => x.name === event.name && !x.done);
              if (open) open.done = true;
              setLive({ steps: [...steps], text: answer });
            } else if (event.type === "text") {
              answer += (answer ? "\n\n" : "") + String(event.delta ?? "");
              setLive({ steps: [...steps], text: answer });
            } else if (event.type === "reread") {
              // The first attempt asserted figures without reading anything,
              // so the route threw it away and is running the turn again.
              // Shown as a step rather than as prose: it is something that
              // happened while the answer was being made, which is exactly
              // what the step trace is for, and it keeps the finished answer
              // free of apparatus the reader did not ask for.
              steps.push({
                name: "re-reading the drawing",
                args: String(event.message ?? ""),
                done: false,
              });
              setLive({ steps: [...steps], text: answer });
            } else if (event.type === "ungrounded") {
              ungrounded = String(event.message ?? "");
            } else if (event.type === "scope_note") {
              scopeNote = String(event.message ?? "");
            } else if (event.type === "scope_violation") {
              // Kept separate from `ungrounded`: that one means "nothing was
              // read", this one means "the wrong thing was read". Both can fire.
              breach = String(event.message ?? "");
            } else if (event.type === "done") {
              citations = (event.citations as typeof citations) ?? [];
              if (typeof event.text === "string" && event.text.trim()) {
                answer = event.text.trim();
              }
            } else if (event.type === "read_handles") {
              // What the agent READ this turn. A better source for marks than
              // what it wrote: prose has a readability ceiling that the
              // drawing does not share. See the route.
              readHandles = Array.isArray(event.handles)
                ? (event.handles as string[])
                : [];
              readGroups = Array.isArray(event.groups)
                ? (event.groups as { label: string; handles: string[] }[])
                : [];
            } else if (event.type === "error") {
              failure = {
                message: String(event.message ?? "The agent failed."),
                hint: String(event.hint ?? ""),
              };
            }
          }
        }

        // Marks are made only for an answer that actually said something — a
        // failed turn has no objects to point at, and an ungrounded one read
        // nothing, so marking its handles would be putting the viewer's
        // authority behind a claim the trace does not support. The same test
        // decides whether the finished message can ever be replayed: an answer
        // that was never allowed to mark must not offer to mark later.
        //
        // Settled before the message is written rather than after it, so the
        // answer in the log and the marks on the drawing carry one identity.
        // Two ids for one answer would make "which answer is currently marked"
        // a question with no answer.
        // Named something, as well as said something.
        //
        // The four tests below decide whether an answer was ALLOWED to mark.
        // This fifth one asks whether it had anything to mark, and without it
        // an answer that names no object still carries marks — so the chat
        // offered to re-mark nothing and to export an empty file, while the
        // drawing showed no marks at all. Seen on an answer that was a
        // clarifying question: "please tell me which land uses you want".
        const named =
          collectHandleCandidates(answer).length > 0 || readHandles.length > 0;
        const marks: AnswerMarks | null =
          !failure && answer && !ungrounded && !breach && named
            ? {
                id: `${sessionId.current}:${Date.now()}`,
                text: answer,
                readHandles,
                groups: readGroups,
                drawingId,
                layout,
              }
            : null;

        setMessages((prev) => [
          ...prev,
          failure
            ? {
                // Message AND hint, and the trace stays attached. This used to
                // show `hint || message`, which dropped the sentence saying
                // WHAT went wrong and kept only the one saying what to do
                // about it -- and it threw away the steps, which are the
                // evidence that the drawing was read before the answer was
                // lost.
                role: "system",
                text: [failure.message, failure.hint].filter(Boolean).join(" "),
                steps: steps.map((x) => ({ ...x, done: true })),
                citations,
              }
            : {
                role: "agent",
                text: answer || "(the agent returned no text)",
                steps: steps.map((x) => ({ ...x, done: true })),
                citations,
                ungrounded: Boolean(ungrounded) || Boolean(breach),
                answer: marks ?? undefined,
              },
        ]);
        for (const warning of [ungrounded, breach, scopeNote]) {
          if (warning) {
            setMessages((prev) => [...prev, { role: "system", text: warning }]);
          }
        }

        // Mark the drawing.
        if (marks) {
          onAnswer(marks.text, marks.id, marks.readHandles, marks.groups);
          setMarkedAnswerId(marks.id);
        }
      } catch (cause) {
        // A cancellation is not a failure, and telling someone to check the
        // container logs because they pressed Cancel is how a real error
        // message gets ignored. The two are separated here by name.
        const aborted =
          typeof cause === "object" &&
          cause !== null &&
          (cause as { name?: string }).name === "AbortError";
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            text: aborted
              ? cancelSentence("running", Date.now() - startedAtRef.current)
              : "Could not reach the agent service. The viewer still works — " +
                "check `docker compose logs cad-agent`.",
          },
        ]);
      } finally {
        abortRef.current = null;
        setLive(null);
        setBusy(false);
      }
    },
    [
      drawingId,
      drawingName,
      layout,
      scopeLayout,
      selection,
      selectionNote,
      onAnswer,
      onAnswerCleared,
    ],
  );

  // The queue, drained. Written as an effect rather than as a call at the end
  // of `runTurn` so that a waiting question starts after ANY ending — a clean
  // answer, an agent error, or a cancellation — instead of only after the one
  // path somebody remembered to wire.
  useEffect(() => {
    if (busy || !queued || !drawingId) return;
    const next = queued;
    setQueued(null);
    setMessages((prev) =>
      prev.map((m) => (m.id === next.id ? { ...m, pending: false } : m)),
    );
    void runTurn(next.text);
  }, [busy, queued, drawingId, runTurn]);

  /** One submission, decided by `lib/askQueue.ts` rather than here.
   *
   *  Every branch produces something visible. That is the entire point of the
   *  change: there is no path through this function that ends in nothing
   *  happening while text sits in the box unexplained.
   */
  const submit = useCallback(() => {
    if (!drawingId) return;
    const decision = decideSubmit({ running: busy, queued: queued?.text ?? null }, draft);
    if (decision.action === "ignore") return;

    if (decision.action === "refuse") {
      // The text stays in the box on purpose: it was refused, not sent, and
      // emptying the box would leave the user retyping a question the app
      // never explained losing.
      setMessages((prev) => [...prev, { role: "system", text: decision.reason }]);
      return;
    }

    const id = newRequestId();
    if (decision.action === "queue") {
      setMessages((prev) => [
        ...prev,
        { role: "user", text: decision.text, id, pending: true },
      ]);
      setQueued({ id, text: decision.text });
      setDraft("");
      return;
    }

    setMessages((prev) => [...prev, { role: "user", text: decision.text, id }]);
    setDraft("");
    void runTurn(decision.text);
  }, [drawingId, busy, queued, draft, runTurn]);

  const cancelRunning = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const cancelQueued = useCallback(() => {
    if (!queued) return;
    const withdrawn = queued;
    setQueued(null);
    setMessages((prev) => [
      ...prev.map((m) =>
        m.id === withdrawn.id ? { ...m, pending: false, cancelled: true } : m,
      ),
      { role: "system", text: cancelSentence("queued", 0) },
    ]);
  }, [queued]);

  /** Put a finished answer's marks back on the drawing.
   *
   *  Asked for as the smallest version of a question the map will ask again:
   *  an answer scrolls away, the marks it made come off with the next
   *  question, and the only way back to "where were those nine schools" was
   *  to ask it again and wait through another turn.
   *
   *  Goes through `onAnswer`, the same path the live turn uses, and hands it
   *  the same four values that turn produced. Nothing is cached about the
   *  RESULT: the handles are re-validated against the API and the rendered
   *  document on every replay, so an answer replayed after the drawing has
   *  changed underneath it degrades the way a fresh answer would — fewer
   *  marks and a badge that says so — instead of asserting a picture that was
   *  true a minute ago. */
  const replayAnswer = useCallback(
    (marks: AnswerMarks) => {
      onAnswer(marks.text, marks.id, marks.readHandles, marks.groups);
      setMarkedAnswerId(marks.id);
    },
    [onAnswer],
  );

  // The viewer drops the marks when the drawing or the layout changes — the
  // handles were validated against a document that is no longer on screen.
  // This mirrors that, and only that: the log would otherwise go on saying an
  // answer is marked over geometry it was never checked against.
  useEffect(() => {
    setMarkedAnswerId(null);
  }, [drawingId, layout]);

  const openStep = live ? currentStep(live.steps) : null;
  const waitNote = busy ? progressNote(elapsed) : null;

  return (
    <div className="chat-panel">
      <p className="muted small">
        Ask about <strong>{drawingName || "this drawing"}</strong>. The agent
        reads it through the same API the viewer uses.
      </p>

      {selection && (
        <div className="chat-selection">
          <span>
            <strong>{selection.total.toLocaleString()}</strong> selected object
            {selection.total === 1 ? "" : "s"} attached
            <span className="muted"> · answered in full</span>
          </span>
          <button
            className="link-button"
            onClick={onClearSelection}
            title="Stop sending this selection with your questions"
          >
            detach
          </button>
        </div>
      )}

      <div className="chat-log">
        {messages.length === 0 && !selection && !live && (
          <ul className="suggestions">
            {[
              "What layers are in this drawing?",
              "How many block references are there, and of what types?",
              "Find any text mentioning a room or plot number.",
            ].map((s) => (
              <li key={s}>
                <button onClick={() => setDraft(s)}>{s}</button>
              </li>
            ))}
          </ul>
        )}
        {messages.map((message, i) => {
          // Replayable only where replaying would be honest: the answer marked
          // something when it landed, and the document it was validated
          // against is the one on screen. A past answer about another drawing
          // stays readable and stops being clickable — see `AnswerMarks`.
          const marks =
            message.answer &&
            message.answer.drawingId === drawingId &&
            message.answer.layout === layout
              ? message.answer
              : null;
          const isMarked = Boolean(marks && markedAnswerId === marks.id);
          return (
            <div
              key={message.id ?? i}
              className={
                `chat-message ${message.role}` +
                (message.pending ? " pending" : "") +
                (message.cancelled ? " cancelled" : "") +
                (marks ? " replayable" : "") +
                (isMarked ? " marked" : "")
              }
              onClick={
                marks
                  ? (event) => {
                      // Two things inside an answer are not a request to replay
                      // it: a handle chip, which means "take me to this one",
                      // and a drag across the text, which means someone is
                      // reading. Both used to be the only ways to interact with
                      // an answer and neither may be taken over by this.
                      if ((event.target as HTMLElement).closest("button")) return;
                      if (window.getSelection()?.toString()) return;
                      replayAnswer(marks);
                    }
                  : undefined
              }
            >
              {message.role === "agent" && (message.steps?.length ?? 0) > 0 && (
                <StepTrace steps={message.steps!} collapsed />
              )}
              {message.role === "agent"
                ? renderAnswer(
                    message.text,
                    isHandleInDrawing,
                    onHandleMentioned,
                  )
                : message.text}
              {message.pending && (
                <div className="queued-note">
                  <span>{queuedSentence()}</span>
                  <button className="link-button" onClick={cancelQueued}>
                    cancel
                  </button>
                </div>
              )}
              {message.cancelled && (
                <div className="queued-note">Never sent — you cancelled it.</div>
              )}
              {message.role === "agent" && (message.citations?.length ?? 0) > 0 && (
                <div className="citations">
                  <span className="citations-label">Sources</span>
                  {message.citations!.map((c, k) => (
                    <span key={k} className="citation" title={c.detail}>
                      {c.tool}
                      {c.detail ? ` · ${c.detail}` : ""}
                    </span>
                  ))}
                </div>
              )}
              {/* The affordance said out loud. Clicking the answer is the
                  gesture, but a region that does something on click and looks
                  like a paragraph is a feature nobody finds; this line is what
                  makes it discoverable, and it is also the keyboard route to
                  it. When these marks are the ones on the drawing it stops
                  being a button and becomes a statement of fact — pressing it
                  again would redraw what is already drawn. */}
              {marks && (
                <div className="answer-export">
                  {SHOW_EXPORT_RES_PICKER && (
                  <label className="export-res">
                    <span className="export-res-label">Cell size</span>
                    <select
                      value={exportRes}
                      onChange={(e) => setExportRes(Number(e.target.value))}
                      title="How big the hexagon cells are in the exported file. The parcels are the same either way."
                    >
                      {EXPORT_RESOLUTIONS.map((r) => (
                        <option key={r} value={r}>
                          {EXPORT_RESOLUTION_LABELS[r] ?? r}
                        </option>
                      ))}
                    </select>
                  </label>
                  )}
                  {/* The objects an answer named, as a file. It carries the
                      same colours, properties and caveat as the drawing's own
                      export because it IS that export, filtered — one code
                      path, so the two can never disagree about what a parcel
                      is.

                      Offered only where `marks` is, which means only on an
                      answer that actually marked something: an answer that
                      named no object has no file to give, and a button that
                      downloads an empty collection is worse than no button. */}
                  <a
                    className="answer-export-link"
                    href={
                      `${API_BASE}/drawings/${marks.drawingId}` +
                      `/geo/export.geojson?res=${exportRes}` +
                      `&handles=${encodeURIComponent(exportHandles(marks).join(","))}` +
                      (isVicinityAnswer(message) ? `&rings=${EXPORT_RINGS}` : "")
                    }
                    title={
                      isVicinityAnswer(message)
                        ? `Download the objects this answer named, plus the catchment it counted inside — ${EXPORT_RINGS} rings, about 100 m. Drop it into geojson.io or any map.`
                        : "Download the objects this answer named as GeoJSON, coloured by land use. Drop it into geojson.io or any map."
                    }
                  >
                    ⭳{" "}
                    {isVicinityAnswer(message)
                      ? "Export answer + catchment"
                      : "Export this answer as GeoJSON"}
                  </a>
                </div>
              )}
              {marks && (
                <div className="answer-remark">
                  {isMarked ? (
                    <span className="answer-remark-on">
                      marked on the drawing
                    </span>
                  ) : (
                    <button
                      className="link-button"
                      onClick={() => replayAnswer(marks)}
                      title="Mark the objects this answer named in the drawing again"
                    >
                      mark these on the drawing again
                    </button>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {/* The live turn.
            The headline names the tool being called right now, not "1 step" —
            the same data the SOURCES row is built from, shown while it is
            still true. A slow answer and a stuck agent look completely
            different once the running tool and the clock are both visible,
            and telling them apart is what A15 is for. */}
        {live && (
          <div className="chat-message agent running">
            <div className="turn-status" aria-live="polite">
              <span className="turn-headline">
                {runningHeadline(live.steps, elapsed)}
              </span>
              <button
                className="link-button"
                onClick={cancelRunning}
                title="Stop this question. Nothing is saved and no answer is produced."
              >
                Cancel
              </button>
            </div>
            {openStep?.args && (
              <div className="turn-args mono">{openStep.args}</div>
            )}
            {waitNote && <div className="turn-note">{waitNote}</div>}
            <StepTrace steps={live.steps} />
            {live.text
              ? renderAnswer(live.text, isHandleInDrawing, onHandleMentioned)
              : null}
          </div>
        )}
      </div>

      {/* The cap, stated before it is reached rather than only when it bites.
          A limit a user discovers by being refused is a limit that reads as a
          bug the first time. */}
      {busy && (
        <div className="ask-queue-state small">
          1 running{queued ? " · 1 waiting" : ""} · this queue holds{" "}
          {QUEUE_DEPTH}
        </div>
      )}

      <textarea
        rows={2}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        placeholder={
          busy
            ? "Ask the next one — it will be sent when this answer lands…"
            : selection
              ? "Ask about the selected objects…"
              : "Ask about this drawing…"
        }
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <div className="comment-actions">
        {/* Never disabled while there is text to send. A dead button is the
            defect this replaces: the refusal has to be a sentence, and a
            sentence needs a click to arrive. */}
        <button
          onClick={submit}
          disabled={!draft.trim()}
          title={
            busy && queued
              ? "One is running and one is waiting; pressing this explains why it cannot be sent yet."
              : busy
                ? "Queued behind the running question."
                : "Send this question"
          }
        >
          {busy ? (queued ? "Queue full" : "Queue it") : "Ask"}
        </button>
        <span className="muted small">Enter to send</span>
      </div>
    </div>
  );
}

/** Turn handle-shaped tokens in the agent's reply into buttons that select the
 *  entity in the viewer.
 *
 *  A token only becomes a chip if it resolves to a real entity in the drawing
 *  currently on screen. Offering a chip that silently does nothing is worse
 *  than plain text: it implies the answer is verifiable when it is not. */
/** Handle chips rendered as buttons before the rest fall back to plain text.
 *
 *  A cap here rather than in the prompt, because the prompt was tried twice
 *  and lost twice. Told to quote about ten handles and let the marks do the
 *  rest, the model answered "show me where the VL2 plots are" by listing all
 *  one hundred and thirty-three.
 *
 *  Nothing is ever dropped. An earlier version discarded everything after the
 *  twelfth handle, prose included, and a distance answer lost the sentence at
 *  the end saying which parcels could not be measured at all. Past the cap a
 *  handle is still shown; it just stops being a button.
 */
const MAX_INLINE_HANDLES = 12;

function handleRuns(
  text: string,
  exists: (handle: string) => boolean,
  onHandle: (handle: string) => void,
  state: { shown: number; folded: number },
  seed: number,
): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  for (const match of text.matchAll(HANDLE_CANDIDATE)) {
    const index = match.index ?? 0;
    const token = match[1];
    // Handles are stored uppercase; accept either case from the model.
    const handle = exists(token)
      ? token
      : exists(token.toUpperCase())
        ? token.toUpperCase()
        : null;
    if (!handle) continue;

    if (index > cursor) parts.push(text.slice(cursor, index));
    if (state.shown >= MAX_INLINE_HANDLES) {
      parts.push(token);
      state.folded += 1;
    } else {
      parts.push(
        <button
          key={`${seed}-${handle}-${index}`}
          className="handle-chip"
          onClick={() => onHandle(handle)}
          title={`Select entity ${handle} in the viewer`}
        >
          {token}
        </button>,
      );
      state.shown += 1;
    }
    cursor = index + match[0].length;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts;
}

/** An agent answer, rendered as the document it is written as.
 *
 *  The model writes Markdown — `**bold**`, `*` bullets, and `|` tables — and
 *  this panel used to print it as one flat run of characters. So a distance
 *  answer arrived as a wall with literal asterisks in it, and the table the
 *  model had gone to the trouble of laying out read as a row of pipes.
 *
 *  Deliberately small. This renders the four things the answers actually
 *  contain and nothing else: bold, bullet lists, tables, and paragraphs. A
 *  full Markdown library would also bring links and images and raw HTML into
 *  a panel that displays text written by a model, which is a larger surface
 *  than the formatting is worth.
 *
 *  Handles stay clickable inside all of it: the inline pass runs on every text
 *  run, so a handle in a table cell is the same chip as a handle in a
 *  sentence.
 */
function renderAnswer(
  text: string,
  exists: (handle: string) => boolean,
  onHandle: (handle: string) => void,
): React.ReactNode {
  const state = { shown: 0, folded: 0 };
  const blocks: React.ReactNode[] = [];
  const lines = text.split("\n");
  let i = 0;
  let key = 0;

  const isTableRow = (line: string) => /^\s*\|.*\|\s*$/.test(line);
  const isSeparator = (line: string) => /^\s*\|[\s:|-]+\|\s*$/.test(line);
  const bullet = /^(\s*)[*+-]\s+(.*)$/;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i += 1;
      continue;
    }

    // --- table ------------------------------------------------------------
    if (isTableRow(line) && i + 1 < lines.length && isSeparator(lines[i + 1])) {
      const cells = (row: string) =>
        row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = cells(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && isTableRow(lines[i])) {
        rows.push(cells(lines[i]));
        i += 1;
      }
      blocks.push(
        <div className="answer-table-scroll" key={`t${key++}`}>
          <table className="answer-table">
            <thead>
              <tr>
                {head.map((c, n) => (
                  <th key={n}>{renderInline(c, exists, onHandle, state)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, r) => (
                <tr key={r}>
                  {row.map((c, n) => (
                    <td key={n}>{renderInline(c, exists, onHandle, state)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    // --- bullet list ------------------------------------------------------
    if (bullet.test(line)) {
      const items: { depth: number; text: string }[] = [];
      while (i < lines.length) {
        const m = lines[i].match(bullet);
        if (m) {
          // Indentation is the model's, and it uses it to mean nesting. Four
          // spaces per level is what it writes; anything deeper is clamped so
          // one stray tab cannot indent an item off the panel.
          items.push({
            depth: Math.min(2, Math.floor(m[1].replace(/\t/g, "    ").length / 4)),
            text: m[2],
          });
          i += 1;
          continue;
        }
        // A wrapped continuation line belongs to the item above it.
        if (lines[i].trim() && !isTableRow(lines[i]) && items.length) {
          items[items.length - 1].text += " " + lines[i].trim();
          i += 1;
          continue;
        }
        break;
      }
      blocks.push(
        <ul className="answer-list" key={`l${key++}`}>
          {items.map((item, n) => (
            <li key={n} data-depth={item.depth}>
              {renderInline(item.text, exists, onHandle, state)}
            </li>
          ))}
        </ul>,
      );
      continue;
    }

    // --- paragraph --------------------------------------------------------
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !bullet.test(lines[i]) &&
      !isTableRow(lines[i])
    ) {
      para.push(lines[i].trim());
      i += 1;
    }
    blocks.push(
      <p className="answer-para" key={`p${key++}`}>
        {renderInline(para.join(" "), exists, onHandle, state)}
      </p>,
    );
  }

  if (state.folded) {
    // In the answer's own language. The panel writes this sentence, not the
    // model, and a note in the wrong language reads as the assistant losing
    // the thread mid-reply — which is what it looked like under an
    // Indonesian answer about 60 mismatched plots.
    blocks.push(
      <p className="handle-fold" key="fold">
        {foldedHandlesNote(state.folded, languageOf(text))}
      </p>,
    );
  }
  return <>{blocks}</>;
}

/** Bold and handle chips inside one run of text.
 *
 *  `state` is shared across the whole answer so the chip cap counts the
 *  message rather than the paragraph — twelve chips per bullet in a list of
 *  nine bullets is the wall this cap exists to prevent.
 */
function renderInline(
  text: string,
  exists: (handle: string) => boolean,
  onHandle: (handle: string) => void,
  state: { shown: number; folded: number },
): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  let key = 0;
  // `**bold**` and `` `code` ``. Backticks are how the model writes layer
  // names, and a layer name is exactly the thing a reader needs to see set
  // apart from the sentence around it.
  const marks = /\*\*([^*]+)\*\*|`([^`]+)`/g;
  for (const m of text.matchAll(marks)) {
    const at = m.index ?? 0;
    if (at > cursor) {
      parts.push(...handleRuns(text.slice(cursor, at), exists, onHandle, state, key++));
    }
    const inner = m[1] ?? m[2] ?? "";
    const runs = handleRuns(inner, exists, onHandle, state, key++);
    parts.push(
      m[1] !== undefined ? (
        <strong key={`b${key++}`}>{runs}</strong>
      ) : (
        <code key={`c${key++}`} className="answer-code">
          {runs}
        </code>
      ),
    );
    cursor = at + m[0].length;
  }
  if (cursor < text.length) {
    parts.push(...handleRuns(text.slice(cursor), exists, onHandle, state, key++));
  }
  return <>{parts}</>;
}

/** The agent's steps, as they happen or as a record afterwards.
 *
 *  Each line is one tool call with the arguments that scoped it. This is the
 *  same data the citations are built from — shown live it explains the wait,
 *  shown afterwards it lets a reader audit the answer without trusting it.
 */
function StepTrace({
  steps,
  collapsed = false,
}: {
  steps: { name: string; args: string; done: boolean }[];
  collapsed?: boolean;
}) {
  const [open, setOpen] = useState(!collapsed);
  if (steps.length === 0) return null;
  return (
    <div className="step-trace">
      {collapsed && (
        <button className="step-toggle" onClick={() => setOpen((v) => !v)}>
          {open ? "▾" : "▸"} {steps.length} step{steps.length === 1 ? "" : "s"}
        </button>
      )}
      {open &&
        steps.map((step, i) => (
          <div key={i} className={`step${step.done ? " done" : ""}`}>
            <span className="step-mark">{step.done ? "✓" : "…"}</span>
            <span className="step-name">{step.name}</span>
            {step.args && <span className="step-args">{step.args}</span>}
          </div>
        ))}
    </div>
  );
}
