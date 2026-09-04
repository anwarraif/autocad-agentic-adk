/**
 * What an ingest run SAYS, expressed as arithmetic rather than as a panel.
 *
 * `docs/INGEST-RUN-CONTRACT.md` froze the run document on 3 September 2026 and
 * gave the reason it exists: ingestion takes minutes on a large drawing, and
 * before this the only way to watch one was the CLI's stdout — a browser
 * refresh lost everything. The rule the owner set is that the user must never
 * stare at a spinner. Every sentence below is that rule turned into words.
 *
 * This file is separate from the panel for the same reason `askQueue.ts` is:
 * every line here is a judgement somebody will want to argue with — when a
 * wait stops being ordinary, what "skipped" is allowed to imply, whether a
 * refusal is an error — and an argument is easier against a pure function than
 * against a component. Nothing here fetches, holds state, or renders; the
 * panel calls it and the same calls can be made from a test.
 *
 * Three things it is careful about:
 *
 *   1. `stalled` and `stalled_since` are READ from the run, never computed
 *      here. The contract derives them on the server and says why: a stalled
 *      run is by definition one that stopped writing, so a flag it would have
 *      had to write itself cannot be trusted. Recomputing them in the browser
 *      would put a second, disagreeing opinion on the screen.
 *   2. `skipped` and `refused` are NOT failures. `skipped` is the idempotency
 *      path — the content is already ingested and current — and `refused` is
 *      most often a duplicate upload. Both get an explanation, neither gets a
 *      red box or a spinner.
 *   3. A stage the backend reports and this file does not recognise is shown,
 *      not dropped. `detail` is free-form by contract, and "never silently
 *      drop" applies to fields as much as to files.
 */

// ---------------------------------------------------------------------------
// The document, exactly as the contract defines it
// ---------------------------------------------------------------------------

/** Where a run ended up. `running` is the only non-terminal value.
 *
 *  Five, not three, because `skipped` and `refused` are genuinely different
 *  facts from `failed` and the screen has to be able to say so: nothing went
 *  wrong when a drawing was already ingested, and nothing went wrong when an
 *  upload was correctly turned away. Contract, "Run document". */
export type IngestOverall = "running" | "done" | "failed" | "refused" | "skipped";

/** Where one stage got to. `pending` never appears in a stored document — the
 *  backend appends a stage when it starts it — and is the status this module
 *  fills the unreported tail of the pipeline with, so the reader can see the
 *  whole ladder rather than only the part that has happened. */
export type IngestStageStatus =
  | "pending"
  | "running"
  | "done"
  | "failed"
  | "skipped";

/** The pipeline in the order it runs. Contract, "Stages": this is exactly the
 *  pipeline that exists today, not an aspiration.
 *
 *  `done` is in the list because the contract puts it there — it is the
 *  terminal marker carrying `elapsed_ms`, and leaving it out would mean the
 *  ladder quietly disagreed with the document it is rendering. */
export const STAGE_ORDER = [
  "upload",
  "convert",
  "extract",
  "store",
  "render",
  "dossier",
  "landuse_draft",
  "h3",
  "done",
] as const;

export type IngestStageName = (typeof STAGE_ORDER)[number];

/** What each stage is called on screen.
 *
 *  Named for what the stage DOES to the drawing, not for the function that
 *  does it: `landuse_draft` is a machine name and "land-use draft" is the
 *  thing a reviewer is waiting for. Unknown names fall through to the raw
 *  string in `stageLabel`, which is the honest rendering of a stage this
 *  build has not been told about. */
export const STAGE_LABELS: Record<IngestStageName, string> = {
  upload: "upload",
  convert: "convert to DXF",
  extract: "extract",
  store: "store",
  render: "render layouts",
  dossier: "dossier",
  landuse_draft: "land-use draft",
  h3: "h3 cells",
  done: "finish",
};

/** One stage as the run document carries it.
 *
 *  Every field but `name` and `status` is optional because the contract's own
 *  example omits them — a skipped `convert` carries neither timestamp — and a
 *  type that demanded them would make the panel crash on the document the
 *  contract prints. */
export interface IngestStage {
  name: string;
  status: IngestStageStatus;
  started_at?: string | null;
  finished_at?: string | null;
  /** Free-form by contract, "never a string, so the UI can render what it
   *  understands and ignore the rest". Read through `stageDetail`. */
  detail?: Record<string, unknown> | null;
}

/** The same triple every other cad-api error uses (`main.ApiError`), so a
 *  refusal reads like every other refusal in this app rather than like a new
 *  kind of thing. */
export interface IngestRefusal {
  code: string;
  message: string;
  hint: string;
}

/** The same-name-different-hash predecessor, when there is one.
 *
 *  A flag, not a claim. `recipes/revision.py:116` reports the ambiguity rather
 *  than resolving it, and this field must be shown the same way: it says a
 *  drawing with this name is already here, not that this one supersedes it. */
export interface IngestRevisionOf {
  drawing_id: string;
  filename: string;
  ingested_at: string;
}

/** The run document plus the two fields the READ routes derive. */
export interface IngestRun {
  /** The RUN id, `uuid4().hex[:16]`. Not the drawing id: a drawing can be
   *  ingested more than once and each attempt is its own record. */
  _id: string;
  drawing_id: string;
  filename: string;
  /** `"upload"` or `"cli"`. Both doors write the same documents, so a run
   *  started from a terminal is watchable here too. Typed wide because the
   *  panel must not break if a third door is added. */
  source: string;
  bytes: number;
  started_at: string;
  /** Rewritten on every stage transition. This is what the watchdog reads. */
  updated_at: string;
  finished_at: string | null;
  overall: IngestOverall;
  refusal: IngestRefusal | null;
  revision_of: IngestRevisionOf | null;
  stages: IngestStage[];
  /** Derived by the read route, never stored. True when the run is running
   *  and `updated_at` is older than `STALL_AFTER_S`. */
  stalled?: boolean;
  /** The `updated_at` the stall is measured from, so the screen can name a
   *  time instead of spinning forever. */
  stalled_since?: string | null;
}

/** `202 Accepted` from `POST /drawings/upload`. The UI then polls the run. */
export interface UploadAccepted {
  run_id: string;
  drawing_id: string;
  filename: string;
  overall: IngestOverall;
  revision_of: IngestRevisionOf | null;
}

/** `GET /drawings/ingest-runs`, newest first. */
export interface IngestRunList {
  total_returned: number;
  runs: IngestRun[];
}

// ---------------------------------------------------------------------------
// Numbers the screen depends on
// ---------------------------------------------------------------------------

/** How often the run is re-read while it is running.
 *
 *  1500 ms, from the contract's UI section. Polling rather than a stream is
 *  the whole reason a mid-run refresh works: the browser holds one string —
 *  the run id — and the document is the only truth. A socket would have made
 *  the panel the owner of the progress, which is the arrangement that lost
 *  everything on refresh in the first place. */
export const POLL_INTERVAL_MS = 1500;

/** After this much silence the read routes call a running run stalled.
 *
 *  180 s, and the contract says where the number came from: Janadriyah's
 *  render stage alone runs ~178 s. Duplicated here as a number rather than
 *  imported because it lives in the Python read route; it is used only to
 *  WORD the sentence below, never to decide it — the decision is the server's
 *  `stalled` flag. */
export const STALL_AFTER_S = 180;

/** The upload ceiling, `UPLOAD_MAX_BYTES` in the contract.
 *
 *  600 MB, which clears Sedra's 500 MB DXF with room and refuses a runaway
 *  before it fills the disk. Checked in the browser only to avoid pushing
 *  600 MB up a wire that is going to answer 413 — the server is the
 *  authority, and this constant is allowed to be out of date without
 *  anything being wrong. */
export const UPLOAD_MAX_BYTES = 600 * 1024 * 1024;

/** The two extensions the route accepts (`415 UNSUPPORTED_FILE` otherwise).
 *  Same argument as the ceiling: a local check saves a pointless round trip,
 *  it does not replace the server's. */
export const UPLOAD_EXTENSIONS = [".dxf", ".dwg"] as const;

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/** `42s`, `4:12`, `1:04:09`.
 *
 *  Deliberately NOT `askQueue.formatElapsed`, which is right for what it
 *  measures and wrong here. That clock counts an agent turn against a
 *  three-minute route timeout, so it never has to render an hour and stops at
 *  minutes. An ingest is measured in minutes by design — one render stage
 *  alone is ~178 s — and a run that has been going for seventy minutes must
 *  not read as `70:14`, which a reader takes for seventy seconds at a glance.
 */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  if (total < 60) return `${total}s`;
  const seconds = total % 60;
  const minutes = Math.floor(total / 60) % 60;
  const hours = Math.floor(total / 3600);
  const mm = String(minutes).padStart(2, "0");
  const ss = String(seconds).padStart(2, "0");
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${minutes}:${ss}`;
}

/** `21:18` in the reader's own timezone, or `null` for anything unparseable.
 *
 *  Null rather than a fallback string because the callers below build
 *  sentences: a sentence that says "stalled since Invalid Date" is worse than
 *  one that omits the clause. */
export function formatClock(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return null;
  return `${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
}

/** `524.8 MB`. Decimal MB, because that is what a file manager shows the user
 *  for the same file and disagreeing with it invites a bug report. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "unknown size";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["kB", "MB", "GB"];
  let value = bytes / 1000;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

/** Milliseconds between two ISO stamps, or 0 when either is missing or bad. */
function spanMs(from: string | null | undefined, to: number): number {
  if (!from) return 0;
  const started = new Date(from).getTime();
  if (Number.isNaN(started)) return 0;
  return Math.max(0, to - started);
}

// ---------------------------------------------------------------------------
// Reading a run
// ---------------------------------------------------------------------------

/** True when the run has stopped moving and polling must stop with it.
 *
 *  Everything that is not `running` is terminal, including the two states
 *  that are not failures. A panel that kept polling a refused upload would be
 *  the spinner the contract exists to abolish, just a politer one. */
export function isTerminal(overall: IngestOverall | null | undefined): boolean {
  return Boolean(overall) && overall !== "running";
}

/** Whether the panel should ask again. Written as its own function so the
 *  polling rule is one testable line rather than a condition buried in an
 *  effect. A run that has not been read yet (`null`) is worth one read. */
export function shouldPoll(run: IngestRun | null): boolean {
  return run === null || run.overall === "running";
}

/** The stage in flight, or `null`.
 *
 *  The first RUNNING stage, not the last stage in the array. A finished stage
 *  is a fact about the past, and naming one as current would be the same
 *  category of lie the whole feature is here to remove. */
export function currentStage(run: IngestRun | null): IngestStage | null {
  if (!run) return null;
  return run.stages.find((s) => s.status === "running") ?? null;
}

/** The stage that failed, or `null`. Used to name the failure rather than
 *  making the reader scan the ladder for the one red mark. */
export function failedStage(run: IngestRun | null): IngestStage | null {
  if (!run) return null;
  return run.stages.find((s) => s.status === "failed") ?? null;
}

/** How long the run has been going, or how long it took.
 *
 *  Measured to `finished_at` once there is one, so a finished run's clock
 *  stops instead of counting up forever behind a "done" headline. */
export function runElapsedMs(run: IngestRun, now: number = Date.now()): number {
  const end = run.finished_at ? new Date(run.finished_at).getTime() : now;
  return spanMs(run.started_at, Number.isNaN(end) ? now : end);
}

/** The same for one stage. `0` for a stage that never started, which reads as
 *  no clock at all rather than as an instant one. */
export function stageElapsedMs(
  stage: IngestStage,
  now: number = Date.now(),
): number {
  if (!stage.started_at) return 0;
  const end = stage.finished_at ? new Date(stage.finished_at).getTime() : now;
  return spanMs(stage.started_at, Number.isNaN(end) ? now : end);
}

/** The name of a stage, in words. Falls through to the raw name for anything
 *  this build does not know, which is how an added stage shows up rather than
 *  disappearing. */
export function stageLabel(name: string): string {
  return STAGE_LABELS[name as IngestStageName] ?? name;
}

/** The one-character mark beside a stage.
 *
 *  Deliberately the same vocabulary as the agent's `StepTrace`
 *  (`web/app/components/ChatPanel.tsx:1156-1159`): ✓ for done, … for running.
 *  Two panels that both show a list of steps should not teach the reader two
 *  sets of symbols. `–` for skipped rather than ✓, because "we did not need
 *  to do this" and "we did this" are different claims. */
export function stageMark(status: IngestStageStatus): string {
  switch (status) {
    case "done":
      return "✓";
    case "running":
      return "…";
    case "failed":
      return "✕";
    case "skipped":
      return "–";
    default:
      return "·";
  }
}

/** One row of the ladder the panel draws. */
export interface StageRow {
  name: string;
  label: string;
  status: IngestStageStatus;
  mark: string;
  /** Milliseconds this stage has taken, or 0 when it has not started. */
  elapsedMs: number;
  /** What the stage's `detail` says, in words, or `null`. */
  detail: string | null;
  /** Warnings from `detail.warnings`. They do not fail a stage, so they are
   *  carried separately rather than folded into `detail`. */
  warnings: string[];
}

/** The whole pipeline, reported and unreported alike.
 *
 *  The contract's stage list is fixed and known in advance, and that is worth
 *  showing: a reader watching `extract` wants to know that four more stages
 *  are coming, not to discover them one at a time. Stages the document does
 *  not carry yet are `pending`.
 *
 *  A stage the backend reports that is NOT in `STAGE_ORDER` is appended
 *  rather than dropped. It should not happen — the contract is frozen — but
 *  "never silently drop" is not a rule about files only, and a panel that
 *  hides an unrecognised stage is a panel that lies about what ran.
 */
export function stageLadder(
  run: IngestRun | null,
  now: number = Date.now(),
): StageRow[] {
  const reported = new Map<string, IngestStage>();
  for (const stage of run?.stages ?? []) reported.set(stage.name, stage);

  const known: string[] = [...STAGE_ORDER];
  const extra = [...reported.keys()].filter((n) => !known.includes(n));

  return [...known, ...extra].map((name) => {
    const stage = reported.get(name);
    const status: IngestStageStatus = stage?.status ?? "pending";
    return {
      name,
      label: stageLabel(name),
      status,
      mark: stageMark(status),
      elapsedMs: stage ? stageElapsedMs(stage, now) : 0,
      detail: stage ? stageDetail(stage) : null,
      warnings: stage ? stageWarnings(stage) : [],
    };
  });
}

/** Keys that carry no meaning for a reader and are left out on purpose.
 *  `sha16` is the drawing id, which the panel already prints once where it
 *  means something; repeating it beside the upload stage is noise. */
const DETAIL_SKIP = new Set(["sha16", "warnings"]);

/** Numbers whose name alone does not say what they count. */
const DETAIL_WORDS: Record<string, string> = {
  entities: "entities",
  layers: "layers",
  layouts: "layouts",
  audit_errors: "audit errors",
};

/** What a stage's `detail` says, in one line, or `null` when it says nothing.
 *
 *  Written to the contract's instruction — "render what it understands and
 *  ignore the rest" — with one deliberate softening: after the keys this
 *  build knows by name, any remaining scalar is printed as `key value`. A
 *  backend that starts reporting `rendered_layouts` then shows it as a
 *  slightly ugly line rather than as nothing at all, and nobody has to ship
 *  a frontend change to see a number the backend already computed. Capped at
 *  three so an enthusiastic `detail` cannot push the ladder off the panel.
 */
export function stageDetail(stage: IngestStage): string | null {
  const detail = stage.detail;
  if (!detail || typeof detail !== "object") return null;

  const parts: string[] = [];

  // `why` first: on a skipped or failed stage it is the whole answer, and
  // burying it after three counts would be answering a different question.
  if (typeof detail.why === "string" && detail.why) parts.push(detail.why);

  if (typeof detail.bytes === "number") parts.push(formatBytes(detail.bytes));
  for (const [key, word] of Object.entries(DETAIL_WORDS)) {
    const value = detail[key];
    if (typeof value === "number") parts.push(`${value.toLocaleString()} ${word}`);
  }
  if (typeof detail.elapsed_ms === "number") {
    parts.push(formatElapsed(detail.elapsed_ms));
  }

  const named = new Set([
    "why",
    "bytes",
    "elapsed_ms",
    ...Object.keys(DETAIL_WORDS),
  ]);
  let spare = 3;
  for (const [key, value] of Object.entries(detail)) {
    if (spare === 0) break;
    if (named.has(key) || DETAIL_SKIP.has(key)) continue;
    if (typeof value === "number") {
      parts.push(`${key} ${value.toLocaleString()}`);
      spare -= 1;
    } else if (typeof value === "string" && value) {
      parts.push(`${key} ${value}`);
      spare -= 1;
    } else if (typeof value === "boolean") {
      parts.push(`${key} ${value ? "yes" : "no"}`);
      spare -= 1;
    }
  }

  return parts.length > 0 ? parts.join(" · ") : null;
}

/** `detail.warnings`, filtered to the strings. Warnings do not fail a stage
 *  (contract, "Stages"), so they are shown beside a ✓ without contradicting
 *  it. */
export function stageWarnings(stage: IngestStage): string[] {
  const raw = stage.detail?.warnings;
  if (!Array.isArray(raw)) return [];
  return raw.filter((w): w is string => typeof w === "string" && w.length > 0);
}

// ---------------------------------------------------------------------------
// Sentences
// ---------------------------------------------------------------------------

/** The line at the top of the panel: what is happening, and for how long.
 *
 *  Five states because there are five, and flattening them is how a duplicate
 *  upload ends up looking like a crash. The running case names the stage —
 *  the same argument `askQueue.runningHeadline` makes about naming the tool —
 *  because "ingesting · 4:12" and "ingesting · render · 4:12" are the
 *  difference between a wait and an explained wait.
 */
export function overallHeadline(
  run: IngestRun,
  now: number = Date.now(),
): string {
  const clock = formatElapsed(runElapsedMs(run, now));
  switch (run.overall) {
    case "running": {
      const open = currentStage(run);
      return open
        ? `Ingesting · ${stageLabel(open.name)} · ${clock}`
        : `Ingesting · ${clock}`;
    }
    case "done":
      return `Ingested in ${clock}`;
    case "skipped":
      return "Already ingested";
    case "refused":
      return "Not ingested";
    case "failed": {
      const bad = failedStage(run);
      return bad ? `Failed at ${stageLabel(bad.name)}` : "Failed";
    }
    default:
      return `${run.overall} · ${clock}`;
  }
}

/** The sentence a finished run earns, or `null` while it is still running.
 *
 *  `skipped` gets words rather than a tick because it is the one outcome a
 *  reader is most likely to misread: nothing was ingested, nothing went
 *  wrong, and the drawing they wanted is already in the list.
 */
export function outcomeSentence(
  run: IngestRun,
  now: number = Date.now(),
): string | null {
  switch (run.overall) {
    case "done":
      return (
        `${run.filename} is in the store — ${formatElapsed(runElapsedMs(run, now))}, ` +
        `${formatBytes(run.bytes)}. The drawing list has been refreshed.`
      );
    case "skipped":
      return (
        `Nothing to do: this file's contents are already stored and current, ` +
        `so it was not ingested again. The drawing is already in the list.`
      );
    case "failed":
      return (
        "The ingest stopped. The stage marked ✕ above says where and why; " +
        "the stages before it did finish, so their work is already stored."
      );
    default:
      return null;
  }
}

/** A refusal, in the shape every other error in this app uses.
 *
 *  `[message, hint].filter(Boolean).join(" ")` — the same join
 *  `ChatPanel.tsx:456` settled on, and for the same reason: showing only the
 *  hint drops the sentence saying WHAT happened and keeps only the one saying
 *  what to do about it.
 *
 *  A refusal is not an error state on screen. The commonest one by far is a
 *  duplicate upload, which is the store working correctly, and it must read
 *  as an explanation.
 */
export function refusalSentence(run: IngestRun): string | null {
  if (!run.refusal) return null;
  const { message, hint } = run.refusal;
  return [message, hint].filter(Boolean).join(" ") || null;
}

/** Why a run that stopped writing is worth looking at, or `null`.
 *
 *  Reads the server's flag; never decides it. The sentence names the time
 *  rather than only the duration, because "stalled since 21:18" is something
 *  a person can check against their own memory of what they were doing, and
 *  "stalled for 6:12" is not.
 *
 *  It does not claim the run is dead. Nothing in the browser can tell — the
 *  render stage alone has been measured at ~178 s — so it says what is known
 *  and stops there.
 */
export function stalledSentence(
  run: IngestRun,
  now: number = Date.now(),
): string | null {
  if (!run.stalled) return null;
  const since = run.stalled_since ?? run.updated_at;
  const clock = formatClock(since);
  const quiet = formatElapsed(spanMs(since, now));
  const when = clock ? `since ${clock}` : "for a while";
  return (
    `Nothing has been written ${when} — ${quiet} of silence. ` +
    `The longest stage measured on this stack takes about ${Math.round(STALL_AFTER_S / 60)} minutes, ` +
    "so a little past that is ordinary; well past it means the ingest stopped without saying so. " +
    "Check `docker compose logs cad-api`."
  );
}

/** The predecessor warning, or `null`.
 *
 *  Worded as ambiguity, not as a decision, because that is what
 *  `PREDECESSOR_RULE` reports. Saying "this replaces X" would be the panel
 *  resolving something the rule deliberately refused to resolve.
 */
export function revisionSentence(run: IngestRun): string | null {
  const prior = run.revision_of;
  if (!prior) return null;
  const when = prior.ingested_at ? new Date(prior.ingested_at) : null;
  const date =
    when && !Number.isNaN(when.getTime()) ? when.toLocaleDateString() : null;
  return (
    `A drawing already here is called "${prior.filename}"` +
    (date ? `, ingested ${date}` : "") +
    ". Its contents differ from this one's, so both are kept and neither is " +
    "assumed to supersede the other."
  );
}

// ---------------------------------------------------------------------------
// Whether an upload can even be attempted
// ---------------------------------------------------------------------------

/** Everything the Upload button's state depends on. Three fields rather than
 *  the panel's state, so the rule can be exercised without the panel. */
export interface UploadGateState {
  file: File | null;
  /** True while the multipart POST is in flight. */
  sending: boolean;
  /** The run being watched, if any. */
  run: IngestRun | null;
}

/** Why the Upload button is off, or `null` when it is on.
 *
 *  A reason, never a hidden button. `globals.css:737-741` argues the case on
 *  the export control and it holds here: a control that is simply absent
 *  reads as a feature that is broken, while one that is visibly off with the
 *  reason on hover reads as a state the app is in.
 *
 *  The extension and size checks duplicate the server's 415 and 413 on
 *  purpose. They cost nothing, and the alternative is pushing 600 MB up a
 *  wire in order to be told the extension was wrong.
 */
export function uploadBlockReason(state: UploadGateState): string | null {
  if (state.sending) return "The file is still being sent.";
  if (state.run && state.run.overall === "running") {
    return "An ingest is already running. Wait for it to finish, or reload to start over.";
  }
  if (!state.file) {
    return `Choose a ${UPLOAD_EXTENSIONS.join(" or ")} file first.`;
  }
  return fileRefusalReason(state.file);
}

/** Why this file cannot be sent, or `null`.
 *
 *  Split out from the gate so the panel can say it beside the chosen file the
 *  moment it is chosen, rather than only on hover over a dead button. Being
 *  told the file is too big before pressing anything is the difference
 *  between a rule and a trap.
 */
export function fileRefusalReason(file: File): string | null {
  const name = file.name.toLowerCase();
  if (!UPLOAD_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return (
      `Only ${UPLOAD_EXTENSIONS.join(" and ")} files are accepted, and this one is ` +
      `"${file.name}".`
    );
  }
  if (file.size > UPLOAD_MAX_BYTES) {
    return (
      `${formatBytes(file.size)} is over the ${formatBytes(UPLOAD_MAX_BYTES)} upload ceiling. ` +
      "Ingest a file this large from the CLI instead."
    );
  }
  return null;
}

/** The chosen file, described in one line before anything has been sent. */
export function fileSummary(file: File): string {
  return `${file.name} · ${formatBytes(file.size)}`;
}
