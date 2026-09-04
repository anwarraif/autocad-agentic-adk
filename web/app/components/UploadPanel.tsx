"use client";

/**
 * Upload a drawing, and watch it being ingested.
 *
 * The panel owns exactly one piece of state that matters: the run id. Every
 * other thing on screen is derived from the run document that
 * `GET /drawings/ingest-runs/{run_id}` returns, and the derivation lives in
 * `lib/ingestRuns.ts` rather than here.
 *
 * That split is the feature, not tidiness. `docs/INGEST-RUN-CONTRACT.md` set
 * the rule — the user must never stare at a spinner, and a mid-run refresh
 * must return to live progress — and the only way to keep it is for the
 * screen to hold no progress of its own. Nothing here accumulates: the stage
 * list is not appended to as events arrive, it is re-read. So a reload, a
 * second tab, and a run started from the CLI all show the same thing, because
 * all three are reading the same document.
 *
 * Polling, not a stream, for the same reason. A socket would have made this
 * component the owner of the progress, which is the arrangement that lost
 * everything on refresh in the first place.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getIngestRun, uploadDrawing } from "@/lib/api";
import {
  POLL_INTERVAL_MS,
  UPLOAD_EXTENSIONS,
  fileRefusalReason,
  fileSummary,
  formatBytes,
  formatElapsed,
  isTerminal,
  outcomeSentence,
  overallHeadline,
  refusalSentence,
  revisionSentence,
  stageLadder,
  stalledSentence,
  uploadBlockReason,
  type IngestRun,
} from "@/lib/ingestRuns";

/** Where the run id is kept across a refresh.
 *
 *  Session storage rather than local: the run being watched belongs to this
 *  tab and this sitting. Surviving a browser restart three days later would
 *  mean opening the app to a panel reporting on an ingest the reader has
 *  forgotten, and the run list is the right way to find that one.
 */
const RUN_KEY = "adk.ingest-run-id";

/** Both accessors are wrapped: storage throws outright in a locked-down
 *  browser, and losing the ability to upload because a preference is off
 *  would be a poor trade for a convenience. */
function rememberRun(runId: string | null): void {
  try {
    if (runId) window.sessionStorage.setItem(RUN_KEY, runId);
    else window.sessionStorage.removeItem(RUN_KEY);
  } catch {
    /* private mode, or storage disabled: the panel still works, it just
       forgets on refresh */
  }
}

function recallRun(): string | null {
  try {
    return window.sessionStorage.getItem(RUN_KEY);
  } catch {
    return null;
  }
}

/** The same join the chat panel settled on (`ChatPanel.tsx:456`): message AND
 *  hint. Showing only the hint drops the sentence saying what went wrong and
 *  keeps only the one saying what to do about it. */
function describe(err: unknown): string {
  if (err instanceof ApiError) {
    return [err.message, err.hint].filter(Boolean).join(" ");
  }
  return err instanceof Error ? err.message : String(err);
}

interface Props {
  /** Called once, when a run reaches `done`.
   *
   *  The drawing picker loads once with `[]` deps (`app/page.tsx`), so a
   *  drawing ingested while the page is open would not appear in it until a
   *  manual reload — the contract names this and requires the refetch. */
  onIngested: (drawingId: string) => void;
  /** Show a drawing in the viewer. */
  onOpenDrawing: (drawingId: string) => void;
  /** Whether the picker already has this drawing. Asked rather than assumed:
   *  the refetch is a round trip, and offering "Open it" a moment before the
   *  option exists would select nothing. */
  isDrawingListed: (drawingId: string) => boolean;
}

export function UploadPanel({
  onIngested,
  onOpenDrawing,
  isDrawingListed,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [sending, setSending] = useState(false);
  /** The run id being watched. The one durable piece of state, and the only
   *  thing put in session storage. */
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<IngestRun | null>(null);
  /** An upload that never produced a run: a network failure, or a refusal the
   *  route answered without writing a record. Shown inline. */
  const [error, setError] = useState<string | null>(null);
  /** A poll that failed while a run was already on screen. Kept separate from
   *  `error` because it must not replace the run: the last known progress is
   *  still the best information available, and blanking it to show a network
   *  message would throw away the answer to keep the complaint. */
  const [pollError, setPollError] = useState<string | null>(null);
  /** Ticks once a second so the clocks move between polls. */
  const [now, setNow] = useState<number>(() => Date.now());

  const fileInput = useRef<HTMLInputElement | null>(null);
  /** Whether any read of this run has succeeded, for the poll's error branch.
   *  A ref because the poll loop closes over its own start. */
  const haveRun = useRef(false);
  /** The run already announced as done, so the drawing list is refetched once
   *  per run and not once per poll. */
  const announced = useRef<string | null>(null);

  // --- come back to a run that was already going ---------------------------
  //
  // The whole point of the run id being the only state. Reading storage in an
  // effect rather than in `useState`'s initialiser because this component is
  // rendered on the server first, where `window` does not exist.
  useEffect(() => {
    const saved = recallRun();
    if (saved) setRunId(saved);
  }, []);

  // --- poll ----------------------------------------------------------------
  //
  // Re-read every 1500 ms while the run is running; stop dead on any terminal
  // state. A chained timeout rather than an interval so a slow answer cannot
  // stack requests on top of each other.
  useEffect(() => {
    if (!runId) {
      setRun(null);
      haveRun.current = false;
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const read = async () => {
      try {
        const next = await getIngestRun(runId);
        if (cancelled) return;
        haveRun.current = true;
        setRun(next);
        setPollError(null);
        if (!isTerminal(next.overall)) {
          timer = setTimeout(read, POLL_INTERVAL_MS);
        }
      } catch (cause) {
        if (cancelled) return;
        if (!haveRun.current) {
          // Nothing was ever read, so there is nothing to keep. The ordinary
          // cause is a remembered id whose record is gone — a reset database,
          // a run from another stack — and leaving the panel pointing at it
          // would make uploading impossible. Forget it and say why.
          rememberRun(null);
          setRunId(null);
          setError(
            `The ingest run this tab was watching could not be read. ${describe(cause)}`,
          );
          return;
        }
        // A run WAS read, so the last progress stands and the failure is
        // annotated beside it. Keep asking: as far as anything here knows the
        // ingest is still going, and giving up on one blip would strand the
        // panel on a stale line with no way back.
        setPollError(describe(cause));
        timer = setTimeout(read, POLL_INTERVAL_MS);
      }
    };

    read();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [runId]);

  // --- the clock -----------------------------------------------------------
  //
  // Only while something is running, and read from `Date.now()` rather than
  // accumulated, so a re-render cannot make a wait look shorter than it was.
  const running = run?.overall === "running";
  useEffect(() => {
    if (!running) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [running]);

  // --- tell the page a drawing arrived -------------------------------------
  useEffect(() => {
    if (!run || run.overall !== "done") return;
    if (announced.current === run._id) return;
    announced.current = run._id;
    onIngested(run.drawing_id);
  }, [run, onIngested]);

  const startUpload = useCallback(async () => {
    if (!file) return;
    setSending(true);
    setError(null);
    setPollError(null);
    // The previous run is let go BEFORE the new one is asked for, storage
    // included. Otherwise an upload that fails without producing a record
    // would leave the old run id remembered, and the next refresh would come
    // back watching an ingest that finished two uploads ago.
    rememberRun(null);
    setRunId(null);
    setRun(null);
    haveRun.current = false;
    announced.current = null;
    try {
      const accepted = await uploadDrawing(file);
      rememberRun(accepted.run_id);
      setRunId(accepted.run_id);
    } catch (cause) {
      // Every refusal is ALSO written as a run document, and the 409 carries
      // that record's id. Following it is what turns "409 ALREADY_INGESTED"
      // into a panel that explains itself — the duplicate upload is the
      // commonest outcome here, and it is the store working, not a fault.
      const refusedRun =
        cause instanceof ApiError && typeof cause.details.run_id === "string"
          ? cause.details.run_id
          : null;
      if (refusedRun) {
        rememberRun(refusedRun);
        setRunId(refusedRun);
      } else {
        setError(describe(cause));
      }
    } finally {
      setSending(false);
    }
  }, [file]);

  /** Clear the finished run and the chosen file, ready for another.
   *
   *  The input's value is reset as well: without it, choosing the same file
   *  again fires no change event and the button stays dead for reasons the
   *  reader cannot see. */
  const startAnother = useCallback(() => {
    rememberRun(null);
    setRunId(null);
    setRun(null);
    setError(null);
    setPollError(null);
    setFile(null);
    announced.current = null;
    if (fileInput.current) fileInput.current.value = "";
  }, []);

  const blockReason = uploadBlockReason({ file, sending, run });
  const fileReason = file ? fileRefusalReason(file) : null;
  const ladder = stageLadder(run, now);
  const stalled = run ? stalledSentence(run, now) : null;
  const refusal = run ? refusalSentence(run) : null;
  const revision = run ? revisionSentence(run) : null;
  const outcome = run ? outcomeSentence(run, now) : null;
  const listed = run ? isDrawingListed(run.drawing_id) : false;

  return (
    <div className="upload-panel">
      <p className="upload-intro small muted">
        Ingest a {UPLOAD_EXTENSIONS.join(" or ")} from here instead of the CLI.
        Every stage is written down as it happens, so this panel survives a
        reload: refresh mid-run and it comes back to the same place.
      </p>

      <div className="upload-picker">
        <input
          ref={fileInput}
          type="file"
          accept={UPLOAD_EXTENSIONS.join(",")}
          onChange={(e) => {
            setFile(e.target.files?.[0] ?? null);
            setError(null);
          }}
        />
        {file && <p className="upload-file mono small">{fileSummary(file)}</p>}
        {/* Said the moment the file is chosen, not on hover over a dead
            button. Being told the file is too large before pressing anything
            is the difference between a rule and a trap. */}
        {fileReason && <p className="upload-file-reason small">{fileReason}</p>}

        <div className="upload-actions">
          <button
            onClick={startUpload}
            disabled={Boolean(blockReason)}
            /* Off with the reason on hover, never hidden: a control that is
               simply absent reads as a feature that is broken. Same rule as
               the export control, globals.css:737-741. */
            title={
              blockReason ??
              "Send this file to cad-api and watch the stages as they run."
            }
          >
            {sending ? "Sending…" : "Upload"}
          </button>
          {run && isTerminal(run.overall) && (
            <button
              className="link-button"
              onClick={startAnother}
              title="Clear this run and choose another file. The record stays in the run history."
            >
              Upload another
            </button>
          )}
        </div>
      </div>

      {error && <p className="banner error">{error}</p>}

      {run && (
        <div className="ingest-run">
          {/* One headline, and the clock is part of it rather than beside it:
              "Ingesting · render · 4:12" is one fact, and splitting the stage
              from its own elapsed time invites reading the number as the
              whole run's. `overallHeadline` decides which it is. */}
          <p className={`ingest-headline ${run.overall}`}>
            {overallHeadline(run, now)}
          </p>
          <p className="ingest-meta small muted mono">
            {run.filename} · {formatBytes(run.bytes)} · {run.source} ·{" "}
            {run.drawing_id}
          </p>

          {/* A stalled run says what it knows and never claims to be dead:
              nothing in the browser can tell, and the flag it is reading is
              derived on the server precisely because a stalled run cannot be
              trusted to report itself. */}
          {stalled && <p className="ingest-note">{stalled}</p>}

          {pollError && (
            <p className="ingest-note">
              The last refresh failed, so the stages below may be behind.
              Still asking every {POLL_INTERVAL_MS / 1000}s. {pollError}
            </p>
          )}

          {/* A refusal is an explanation, not an error spinner. The duplicate
              upload is the commonest one by far and it means the store already
              has this drawing — which is the system working. */}
          {refusal && <p className="ingest-refusal">{refusal}</p>}

          {revision && <p className="ingest-revision">{revision}</p>}

          {run.stages.length > 0 && (
            <div className="ingest-stages">
              {ladder.map((row) => (
                <div key={row.name} className="ingest-stage-row">
                  <div className={`ingest-stage ${row.status}`}>
                    <span className="ingest-stage-mark">{row.mark}</span>
                    <span className="ingest-stage-name">{row.label}</span>
                    {row.detail && (
                      <span className="ingest-stage-detail">{row.detail}</span>
                    )}
                    {row.elapsedMs > 0 && (
                      <span className="ingest-stage-clock">
                        {formatElapsed(row.elapsedMs)}
                      </span>
                    )}
                  </div>
                  {/* Warnings do not fail a stage, so they sit under a tick
                      without contradicting it. */}
                  {row.warnings.map((warning, i) => (
                    <p key={i} className="ingest-stage-warning">
                      {warning}
                    </p>
                  ))}
                </div>
              ))}
            </div>
          )}

          {outcome && (
            <p className={`ingest-outcome ${run.overall}`}>{outcome}</p>
          )}

          {/* Offered whenever the drawing is actually in the picker — which
              includes a refused duplicate and a skipped re-ingest, where the
              drawing the reader wanted has been in the store the whole time
              and telling them so is the most useful thing on the screen. */}
          <div className="ingest-actions">
            <button
              className="link-button"
              onClick={() => onOpenDrawing(run.drawing_id)}
              disabled={!listed}
              title={
                listed
                  ? "Show this drawing in the viewer."
                  : "The drawing list does not have this drawing; it is not in the store."
              }
            >
              Open it in the viewer
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
