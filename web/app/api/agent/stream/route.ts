/**
 * Streaming BFF for the ADK agent.
 *
 * Why streaming, and why Server-Sent Events rather than a WebSocket.
 *
 * A tool-using turn on this drawing takes anywhere from two seconds to a
 * minute, and the non-streaming route could only show the word "thinking…"
 * for the whole of it. That is not a cosmetic problem: with nothing to look
 * at, a slow answer and a hung agent are indistinguishable, so people either
 * wait on something broken or give up on something that was working.
 *
 * The traffic here is one-directional — the browser asks once, the server
 * narrates until it is done — which is precisely the shape SSE has. A
 * WebSocket would add a second protocol, its own reconnection semantics and
 * its own proxy caveats to buy a return channel nothing uses. ADK already
 * exposes `/run_sse`, so the whole path is: browser → this route → ADK, with
 * events forwarded as they arrive.
 *
 * What the browser receives (each a `data:` line of JSON):
 *
 *   {type:"tool_call",  name, args}   the agent is about to read something
 *   {type:"tool_result",name, ok}     that read came back
 *   {type:"text",       delta}        answer text as it is produced
 *   {type:"done",       toolCalls, citations}
 *   {type:"error",      error, message, hint}
 *
 * `citations` are built here from the tool calls that actually executed, not
 * from anything the model says about its sources. A model can write a
 * plausible citation for a call it never made; an execution trace cannot.
 */

// The scope contract lives in one file that this route and the headless
// evaluation harness both read. Two hand-maintained copies drifted once, and
// the drift was invisible: every headless result claiming D-073 was fixed had
// been measured against a weaker prompt than this route ships.
import CONTRACT from "@/lib/scope-contract.json";

const AGENT_URL = process.env.AGENT_URL ?? "http://cad-agent:8000";
const API_INTERNAL = process.env.CAD_API_INTERNAL_URL ?? "http://cad-api:8000";
const APP_NAME = process.env.CAD_AGENT_APP ?? "cad_agent";
const USER_ID = "local";
const AGENT_TIMEOUT_MS = Number(process.env.AGENT_TIMEOUT_MS ?? 180_000);

interface TurnRequest {
  message?: string;
  drawing_id?: string;
  drawing_name?: string;
  /** The layout on screen. Without it every count is ambiguous: on this
   *  drawing "how many entities" has four correct answers depending on
   *  scope, and they differ by a factor of eight. */
  layout?: string;
  /** The layout the visible entities are STORED under. Equals `layout`
   *  except on a paper sheet, which draws model space through a viewport. */
  scope_layout?: string;
  session_id?: string;
  selection?: { selection_id?: string; total?: number; summary?: string };
  /** What the selected object is a label FOR, when it is a label.
   *
   *  Built in the browser from a point-in-polygon lookup, because that is
   *  where the selection is. It is context, not an instruction: it says which
   *  parcel contains the clicked text and lets the model decide what the
   *  question was about. */
  selection_note?: string;
}

/** One tool call, as it will be shown to the user under the answer. */
interface Citation {
  tool: string;
  detail: string;
}

export async function POST(request: Request): Promise<Response> {
  let payload: TurnRequest;
  try {
    payload = await request.json();
  } catch {
    return sseError("BAD_REQUEST", "Body must be JSON.", "Send {message, drawing_id, session_id}.");
  }
  const message = (payload.message ?? "").trim();
  if (!message) {
    return sseError("BAD_REQUEST", "Empty message.", "Type a question first.");
  }
  const sessionId = encodeURIComponent(payload.session_id ?? "default");

  // Idempotent: ADK answers 400 when the session already exists, which is
  // fine — it only has to exist.
  await fetch(
    `${AGENT_URL}/apps/${APP_NAME}/users/${USER_ID}/sessions/${sessionId}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        state: payload.drawing_id ? { active_drawing: payload.drawing_id } : {},
      }),
    },
  ).catch(() => undefined);

  // Fetched before the turn opens, so the menu travels with the question.
  const catalogue = await analysisCatalogue();

  let upstream: Response;
  try {
    upstream = await fetch(`${AGENT_URL}/run_sse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        app_name: APP_NAME,
        user_id: USER_ID,
        session_id: sessionId,
        new_message: { role: "user", parts: [{ text: buildPrompt(payload, message, catalogue) }] },
        streaming: false,
      }),
      signal: AbortSignal.timeout(AGENT_TIMEOUT_MS),
    });
  } catch (cause) {
    const timedOut = cause instanceof Error && cause.name === "TimeoutError";
    return sseError(
      timedOut ? "AGENT_TIMEOUT" : "AGENT_UNREACHABLE",
      timedOut
        ? `The agent did not answer within ${AGENT_TIMEOUT_MS / 1000}s.`
        : `Could not reach the agent at ${AGENT_URL}.`,
      timedOut
        ? "Try a narrower question, or check `docker compose logs cad-agent`."
        : "The agent container may not be running. Everything else in the viewer works without it.",
    );
  }

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    return sseError(
      "AGENT_ERROR",
      `Agent returned ${upstream.status}.`,
      detail.slice(0, 400) || "Check `docker compose logs cad-agent`.",
    );
  }

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  // Tools whose result said, in a field rather than in prose, that it
  // established nothing. Kept for the check after the answer is written.
  const unsettled = new Map<string, string>();

  /** Every analysis this turn actually ran, by name.
   *
   *  Read from the tool calls rather than from the answer, because the answer
   *  is where the confusion becomes invisible: a recipe that measures the
   *  wrong thing still returns a real number, in the right units, with real
   *  sources under it. */
  const recipesRun = new Set<string>();

  //: Whether this turn ran an analysis that intersects two answer sets.
  let composed = false;

  /** Readings that answered only because the server re-ran them once.
   *
   *  A recipe that settles nothing can publish the exact call that would
   *  settle it, and the server executes that plan rather than describing it
   *  (see `recipes/selfheal.py`). The figure that comes back is real; the
   *  SCOPE it was measured over was chosen by machine and confirmed by no one.
   *  So the result is quotable — refusing would publish "there are none" over
   *  a question that was never asked — and the sentence that makes it true has
   *  to travel with it.
   *
   *  Recorded here because this is the only place that sees both the tool
   *  results and the sentence written from them. It reports; it never edits
   *  the answer. */
  const retried = new Map<string, string>();

  const reader = upstream.body.getReader();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (obj: unknown) =>
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));

      const citations: Citation[] = [];
      /** drawing_ids read by tools that are not the drawing on screen. */
      const foreign = new Set<string>();
      let toolCalls = 0;
      let text = "";
      let buffer = "";
      /** Handles the agent actually READ this turn, from tool responses.
       *
       *  The viewer used to mark only the handles an answer managed to write
       *  in prose, and prose has a readability ceiling: asked to show the VL2
       *  plots, the agent named twenty of a hundred and thirty-three because
       *  a list of a hundred and thirty-three handles is not an answer
       *  anyone reads. Twenty got marked and the other hundred and thirteen
       *  did not, which is the drawing telling the user something untrue
       *  about its own contents.
       *
       *  What the agent read is a better source than what it wrote. It is
       *  recorded here rather than parsed out of the text, so it cannot
       *  disagree with the sources already on screen. */
      const readHandles = new Set<string>();
      /** Which layer each handle was read for.
       *
       *  Asked where the schools and the mosques are, an answer that paints
       *  both the same colour has answered half the question: the reader can
       *  see that fifteen things matched and not which of them is which. The
       *  label is the layer, because the layer is what the query was scoped
       *  to -- the viewer never has to interpret a name, it just needs two
       *  handles from different layers to come out different colours. */
      const groupOf = new Map<string, string>();
      /** Did the person ask to be SHOWN something, in their own words.
       *
       *  Read from the user's message and never from the model's. Asked
       *  "show me where all the house types are", the agent answered with a
       *  correct summary of thirteen layers and called nothing that returns a
       *  handle, so the drawing stayed blank -- and which tools it reaches
       *  for varies between identical questions, so this cannot be fixed by
       *  asking it more firmly. It has been asked three times.
       *
       *  The cost of being wrong is asymmetric, which is what makes this
       *  worth doing: a mark nobody wanted is one click on `clear`, and a
       *  mark that never appeared is a person concluding the drawing has no
       *  houses in it. */
      const wantsLocating = LOCATING_WORDS.test(payload.message ?? "");
      /** Layers a classifying tool named, waiting to be narrowed by the
       *  answer and then resolved. */
      const locatable: { layout: string | null; layers: Set<string> } = {
        layout: null,
        layers: new Set(),
      };
      /** Arguments of the last call to each tool, so a response can be paired
       *  with the filter that produced it. */
      const lastArgs = new Map<string, Record<string, unknown>>();
      /** Set-completion reads, awaited before the marks are sent. */
      const completing: Promise<void>[] = [];

      const handleEvent = (event: unknown) => {
        const parts =
          (event as { content?: { parts?: Record<string, unknown>[] } })?.content?.parts;
        if (!Array.isArray(parts)) return;
        for (const part of parts) {
          const call = part.functionCall as { name?: string; args?: Record<string, unknown> } | undefined;
          if (call?.name) {
            toolCalls += 1;
            const target = (call.args ?? {}).drawing_id;
            if (
              typeof target === "string" &&
              payload.drawing_id &&
              target !== payload.drawing_id
            ) {
              foreign.add(target);
            }
            lastArgs.set(call.name, call.args ?? {});
            const detail = describeArgs(call.args ?? {});
            citations.push({ tool: call.name, detail });
            send({ type: "tool_call", name: call.name, args: detail });
          }
          const result = part.functionResponse as { name?: string; response?: unknown } | undefined;
          if (result?.name) {
            const body = result.response as Record<string, unknown> | undefined;
            const before = new Set(readHandles);
            collectHandles(body, readHandles);
            const callLayer = lastArgs.get(result.name)?.layer;
            if (typeof callLayer === "string" && callLayer) {
              for (const handle of readHandles) {
                if (!before.has(handle)) groupOf.set(handle, callLayer);
              }
            }
            completing.push(
              completeTruncatedSet(
                result.name,
                lastArgs.get(result.name),
                body,
                payload.drawing_id,
                readHandles,
                groupOf,
              ),
            );
            if (wantsLocating) {
              const named = collectLandUseLayers(body);
              const layout = lastArgs.get(result.name)?.layout;
              if (named.length && typeof layout === "string") {
                locatable.layout = layout;
                for (const layer of named) locatable.layers.add(layer);
              }
            }
            // Did any reading come back saying it settled nothing? A recipe
            // that cannot establish its answer publishes
            // `conclusion_safety.usable_as_evidence: false`, and an answer
            // built on one inherits an emptiness it did not measure. Recorded
            // here because this is the only place that sees both the tool
            // results and the sentence written from them.
            if (result.name === "run_analysis") {
              const recipe = String(lastArgs.get(result.name)?.recipe ?? "");
              if (recipe) recipesRun.add(recipe);
              if (recipe.includes("combine")) composed = true;
            }
            if (result.name === "find_duplicates") recipesRun.add("find_duplicates");
            // Named at the JSON boundary, like `functionResponse` above it.
            // Without the shape, `conclusion_safety` arrives as `unknown`,
            // narrows to `{}` under a truthiness check, and reading a field
            // off it fails the production build — which is how this route
            // stopped being buildable while the running container, started
            // before the break, went on serving the old bundle and looking
            // healthy.
            const safety = body?.conclusion_safety as
              | { usable_as_evidence?: unknown; why?: unknown; verdict?: unknown }
              | undefined;
            if (safety && safety.usable_as_evidence === false) {
              unsettled.set(result.name, String(safety.why ?? safety.verdict ?? ""));
            }
            const scopes = retriedScopes(body);
            if (scopes.length) {
              retried.set(result.name, scopes.join("; "));
            }
            send({ type: "tool_result", name: result.name, ok: body?.ok !== false });

          }
          if (typeof part.text === "string" && part.text.trim()) {
            text += (text ? "\n\n" : "") + part.text.trim();
            // Held, not sent yet.
            //
            // Whether this answer is publishable depends on something that is
            // not known until the stream ends — whether any tool ran — and an
            // answer already on screen cannot be withdrawn. Nothing is lost by
            // waiting: the upstream runs with `streaming: false`, so a turn's
            // prose arrives as one part near the end regardless.
          }
        }
      };

      /** Consume one upstream turn's SSE frames. */
      const drain = async (from: ReadableStreamDefaultReader<Uint8Array>) => {
        buffer = "";
        for (;;) {
          const { done, value } = await from.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          // SSE frames are separated by a blank line; anything after the last
          // one is a partial frame and waits for the next chunk.
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";
          for (const frame of frames) {
            for (const line of frame.split("\n")) {
              if (!line.startsWith("data:")) continue;
              const raw = line.slice(5).trim();
              if (!raw || raw === "[DONE]") continue;
              try {
                handleEvent(JSON.parse(raw));
              } catch {
                /* a frame we cannot parse is not a reason to kill the turn */
              }
            }
          }
        }
      };

      try {
        await drain(reader);

        // Ask again, once, when a turn asserted figures without reading.
        //
        // The session keeps its history, so the previous question's tool
        // results are in the context and cheaper to reuse than to re-fetch.
        // Measured twice on 25 August 2026, on the second and third questions
        // of a run: the model reproduced `land_use_summary` figures from an
        // earlier turn, ran nothing, marked nothing, and was published under
        // the warning below. Both times the numbers were right, which is the
        // hard part — the answer looks perfect and is untraceable.
        //
        // Two instructions were tried first, one in the system prompt and one
        // in the per-turn prompt naming the visible consequence. Neither
        // held. So this stops asking and re-runs the turn with the tool
        // requirement stated as the whole task. It costs a second round trip
        // ONLY in the case that would otherwise ship a warning, and the
        // warning still fires if the retry also reads nothing.
        if (toolCalls === 0 && payload.drawing_id && assertsFacts(text, payload)) {
          const firstTry = text;
          text = "";
          try {
            const again = await fetch(`${AGENT_URL}/run_sse`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                app_name: APP_NAME,
                user_id: USER_ID,
                session_id: sessionId,
                new_message: {
                  role: "user",
                  parts: [{ text: retryPrompt(payload, message, catalogue) }],
                },
                streaming: false,
              }),
              signal: AbortSignal.timeout(AGENT_TIMEOUT_MS),
            });
            if (again.ok && again.body) {
              send({ type: "reread", message: rereadNote() });
              await drain(again.body.getReader());
            }
          } catch {
            /* a failed retry is not worse than the answer we already have */
          }
          // Nothing came back the second time: keep the first answer rather
          // than publishing silence, and let the warning below speak.
          if (!text.trim()) text = firstTry;
        }

        if (text.trim()) send({ type: "text", delta: text });

        // Zero tool calls used to raise one blanket warning. Measured on 20
        // Aug, that conflated two opposite situations and served the user
        // badly in both:
        //
        //   "What is the capital of France?"  -> the agent correctly refused,
        //   which necessarily costs zero tool calls, and was told it had
        //   crashed. An alarm that cries wolf on correct behaviour is one the
        //   user learns to click past.
        //
        //   "How many DIMENSIONs did I select vs the layout?" -> the agent
        //   answered 1393 (right, remembered from an earlier turn) and 1928
        //   (invented; the truth is 9867), perfectly formatted and perfectly
        //   scope-labelled. The old warning told the user to restart a
        //   container. It never told them the number was fiction.
        //
        // So classify. What matters is not that no tool ran, but whether an
        // unread answer asserted something checkable.
        if (toolCalls === 0 && payload.drawing_id) {
          if (assertsFacts(text, payload)) {
            send({
              type: "ungrounded",
              severity: "unverified",
              message:
                "Do not trust the figures in this answer. It was written " +
                "without reading the drawing — no tool ran, so nothing above " +
                "is sourced, and remembered numbers from earlier in this " +
                "conversation may belong to a different scope. Ask again to " +
                "force a fresh read; if it keeps happening, check " +
                "`docker compose logs cad-agent`.",
            });
          }
          // No figures and no tools is a refusal or a clarifying question.
          // That is the agent working, and it gets no warning.
        }

        // An answer that reports NOTHING FOUND while a reading behind it
        // reported NOTHING SETTLED.
        //
        // Measured on one question over five runs: "which parcels are both
        // size outliers and without road frontage" was answered "there are
        // none" while the frontage step had established nothing at all. The
        // true answer is one parcel. The two sentences are indistinguishable
        // to a reader -- an empty set looks the same whether it was measured
        // or merely inherited -- and only the tool results tell them apart,
        // which is why the check lives here and not in the instruction.
        //
        // It reports; it does not rewrite. The answer stays the agent's own.
        if (unsettled.size > 0 && claimsNothingFound(text)) {
          const [name, why] = [...unsettled.entries()][0];
          send({
            type: "ungrounded",
            severity: "unverified",
            message:
              "This answer reports that nothing was found, but `" + name +
              "` came back having established nothing" +
              (why ? ": " + why : "") +
              ". An empty result inherited from a step that settled nothing " +
              "is not the same as a measured absence. Ask again naming the " +
              "parameters that step suggested, or treat this as unanswered.",
          });
        }

        // A question that asked for two conditions AT ONCE, answered with an
        // empty finding, by a turn that never composed anything.
        //
        // This is the failure the check above could not see, and measuring is
        // what showed the difference. Over five runs of one question -- "which
        // parcels are BOTH size outliers AND without road frontage" -- the
        // wrong answers did not inherit an empty step: they never ran the
        // analyses at all. One turn went to the schedule check and the land
        // use summary and answered "there are none" from neither. There is no
        // unsettled result to notice, because nothing relevant was read.
        //
        // So the trigger is the QUESTION's shape, which the user typed, not
        // the model's judgement: two conditions joined by "and", an answer
        // that reports nothing found, and no composition in the trace. The
        // route reports it; the answer is left as written.
        // An empty answer after work was done. Measured: one turn spent nine
        // calls pulling entities and measuring them handle by handle across
        // three layers -- trying to compute by hand what one analysis answers
        // -- and then wrote nothing at all. Silence after ten seconds of
        // reading is the worst outcome of the three: a wrong answer can be
        // argued with, and a refusal tells you where you stand.
        if (!text.trim() && toolCalls > 0) {
          send({
            type: "ungrounded",
            severity: "unverified",
            message:
              "No answer was written, although " + toolCalls +
              (toolCalls === 1 ? " read was " : " reads were ") +
              "made. Nothing here is a finding — treat the question as " +
              "unanswered rather than as answered with nothing, and ask " +
              "again, naming the analysis you want if you know it.",
          });
        }

        if (
          asksForTwoConditions(message) &&
          claimsNothingFound(text) &&
          !composed
        ) {
          send({
            type: "ungrounded",
            severity: "unverified",
            message:
              "This question asked for two conditions at once, and the answer " +
              "reports that nothing satisfies both — but no combined analysis " +
              "was run, so the two sets were never actually intersected. Ask " +
              "again for `combine_findings`, which performs the intersection " +
              "server-side and refuses when either side has established " +
              "nothing.",
          });
        }

        // An answer whose scope was chosen by an automatic re-run, written
        // without saying so.
        //
        // The opposite failure to the two above, and it arrives looking like
        // success: a figure, a handle, a confident sentence. What is missing
        // is that one side of it was measured over a layer the server picked
        // by itself. Asked which parcels are both size outliers and without
        // road frontage, the composition now answers `205D12B` with no
        // parameters given at all -- because the frontage side re-ran itself
        // against the layer its own corridor test ranked first. That is a good
        // answer and an unconfirmed scope, and a reader who is told only the
        // first half has been misled by an improvement.
        //
        // Matched on the retried VALUES appearing in the prose, not on any
        // phrasing, so an answer that already names them is left alone.
        if (retried.size > 0 && text.trim()) {
          const missing = [...retried.entries()].filter(
            ([, scope]) => !mentionsScope(text, scope),
          );
          if (missing.length) {
            send({
              type: "ungrounded",
              severity: "unverified",
              message:
                "One reading behind this answer established nothing on its " +
                "first run and was re-run once by the server, over a scope it " +
                "chose itself: " +
                missing.map(([name, scope]) => name + " → " + scope).join("; ") +
                ". The figures are really measured, but over that scope, and " +
                "nothing has confirmed it is the right one. Confirm it before " +
                "quoting these numbers, or ask again naming the scope yourself.",
            });
          }
        }

        // Ground claimed twice, answered by counting COPIES.
        //
        // `find_duplicates` finds the same shape stored more than once.
        // `overlap_scan` finds two different parcels covering the same
        // ground. Both are "twice", both return an area in square metres, and
        // only one of them answers a question about double-counted land.
        //
        // Measured on this drawing: asked for the total parcel area minus
        // every square metre claimed twice, the turn called `find_duplicates`
        // eight times and subtracted 30,397.747 m2 of copies. The ground
        // really claimed by two parcels is 29,241.258 m2. The answer was well
        // formed, cited its sources, and was wrong by a whole concept -- and
        // nothing downstream could tell, because there was no empty finding,
        // no unsettled reading and no retry to notice.
        //
        // So the right figure is fetched rather than described. Same API the
        // viewer uses, one call, and the correction says plainly which recipe
        // produced which number so the reader can choose.
        if (
          payload.drawing_id &&
          asksAboutSharedGround(message) &&
          recipesRun.has("find_duplicates") &&
          !recipesRun.has("overlap_scan")
        ) {
          const shared = await sharedGroundTotal(
            payload.drawing_id,
            payload.scope_layout || payload.layout,
          );
          if (shared) {
            send({
              type: "ungrounded",
              severity: "unverified",
              message:
                "This question is about ground claimed twice, and the answer " +
                "was built from `find_duplicates`, which counts COPIES of the " +
                "same shape — not two different parcels covering the same " +
                "ground. `overlap_scan` answers that one: " +
                shared +
                ". Prefer that figure for anything about double-counted area, " +
                "and read the number above as the area of repeated geometry.",
            });
          }
        }

        // A drawing other than the one on screen was read. D-050 calls the
        // drawing an absolute boundary; the instruction that states it is
        // advisory, so this reports when it did not hold.
        if (foreign.size > 0) {
          send({
            type: "scope_violation",
            drawings: [...foreign],
            message:
              `This answer read ${foreign.size} drawing` +
              `${foreign.size === 1 ? "" : "s"} other than the one on screen. ` +
              "Figures in it may not describe the drawing you are looking at.",
          });
        }
        // An empty answer after the drawing was actually read is not an
        // answer, and it must not be delivered as one.
        //
        // Measured on 24 August 2026: asked for plot counts and average areas
        // per typology, the agent made thirteen `stats` calls, twelve of them
        // fine, and emitted a garbled tool name on the seventh. ADK raises on
        // an unknown tool, the stream ended, and the user was shown
        // "(the agent returned no text)" -- no reason, no hint, and twelve
        // successful reads thrown away. Asked again, the same question
        // answered perfectly. The capability was never missing; only the
        // report of its failure was.
        //
        // So say what happened. The tools that DID run are already on screen
        // as sources, which is the evidence that the drawing was read and the
        // answer alone was lost.
        if (toolCalls > 0 && !text.trim()) {
          send({
            type: "error",
            error: "ANSWER_LOST",
            message:
              `The drawing was read — ${toolCalls} tool call` +
              `${toolCalls === 1 ? " ran and is" : "s ran and are"} listed as sources — ` +
              "but the agent produced no text. Measured on this stack, the " +
              "tool calls themselves all succeeded: the turn ends after a " +
              "long run of large reads with an empty final message. It is " +
              "not a question the drawing cannot answer.",
            hint:
              "Ask the same question again, narrower — one condition at a " +
              "time, or naming the analysis you want. A turn that reads less " +
              "finishes; the reads themselves were not the problem.",
          });
        }
        await Promise.allSettled(completing);
        // Narrowed by the answer, then resolved.
        //
        // `land_use_summary` classifies every use in the drawing, so locating
        // everything it named marks the schools and the mosques too when the
        // question was about houses -- measured: 2,591 marks for a question
        // whose own answer says 2,380. The model's words are a poor source of
        // HANDLES and a good source of SCOPE, so they are used for the second
        // and never for the first.
        // Precise beats broad, and only one of the two may run.
        //
        // Resolving a layer marks everything on it. That is the right answer
        // when the tools named a subject and no objects -- "show me the
        // residential plots" -- and the wrong one the moment a tool DID name
        // objects, because the specific finding then disappears into its own
        // layer. Measured: "show me which plots have no label inside them"
        // marked 2,569 objects, more than the drawing holds parcels, because
        // the handles `join_labels` had found were widened to every layer the
        // summary mentioned. The question asks for a subset and was answered
        // with the whole.
        //
        // So layer resolution is a FALLBACK. It runs only when the turn read
        // no handles of its own.
        if (
          wantsLocating &&
          !readHandles.size &&
          locatable.layout &&
          locatable.layers.size
        ) {
          const spoken = [...locatable.layers].filter((layer) =>
            text.toLowerCase().includes(layer.toLowerCase()),
          );
          // Nothing named means the answer spoke about the drawing without
          // naming layers; marking every classified parcel is then the honest
          // reading of "show me where".
          const chosen = spoken.length ? spoken : [...locatable.layers];
          for (const layer of chosen) {
            if (readHandles.size >= MAX_READ_HANDLES) break;
            await pageHandles(
              payload.drawing_id!,
              { layer, layout: locatable.layout },
              MAX_READ_HANDLES,
              readHandles,
              groupOf,
            );
          }
        }
        if (readHandles.size) {
          send({
            type: "read_handles",
            handles: [...readHandles].slice(0, MAX_READ_HANDLES),
            total: readHandles.size,
            groups: [...readHandles]
              .slice(0, MAX_READ_HANDLES)
              .reduce<{ label: string; handles: string[] }[]>((acc, handle) => {
                const label = groupOf.get(handle) ?? "";
                const row = acc.find((g) => g.label === label);
                if (row) row.handles.push(handle);
                else acc.push({ label, handles: [handle] });
                return acc;
              }, []),
          });
        }
        // Asked to be SHOWN something, and nothing appeared on the drawing.
        //
        // Measured across the ten tools a question most often lands on, seven
        // return no handle at all: `land_use_summary`, `measure`,
        // `parcel_inventory`, `junction_census`, `drafting_hygiene`,
        // `drawing_tables` and `stats` all answer in counts and totals. So
        // whether anything gets marked depends on which tool the turn happened
        // to choose, and from the outside that looks like the viewer working
        // sometimes and not others.
        //
        // The honest fix at this layer is not to invent marks. It is to say
        // that nothing could be located and what to ask instead, so a blank
        // drawing stops reading as "there is nothing there".
        if (wantsLocating && !readHandles.size && toolCalls > 0 && text.trim()) {
          send({
            type: "ungrounded",
            severity: "unverified",
            message:
              "Nothing was marked on the drawing. The readings behind this " +
              "answer report counts and totals rather than individual " +
              "objects, so there were no handles to show — it does not mean " +
              "the objects are not there. Ask for the objects themselves " +
              "(\"list the parcels on that layer\", or name the layer you " +
              "want shown) and they will be marked.",
          });
        }

        // A sheet on screen, an answer about the space it draws.
        //
        // Both are correct and the pairing reads as a mistake. A paper layout
        // holds its own furniture — border, title block, viewport frames — and
        // draws model space through a window, so a question about roads or
        // plots is answered under the model layout even while the sheet is the
        // thing being looked at. The contract already tells the model to name
        // the scope it used, and it does. What the READER is missing is why
        // the two names differ: on this drawing the sheet carries no parcels
        // of its own at all, so "0" would have been the alternative.
        //
        // Only when the answer actually names the other space, so a reply
        // genuinely about the sheet's own furniture is left alone.
        if (
          payload.layout &&
          payload.scope_layout &&
          payload.scope_layout !== payload.layout &&
          text.toLowerCase().includes(payload.scope_layout.toLowerCase())
        ) {
          send({
            type: "scope_note",
            message:
              `You are looking at "${payload.layout}", a sheet that draws ` +
              `"${payload.scope_layout}" through a viewport. The objects on ` +
              `it are stored under "${payload.scope_layout}", which is why ` +
              `the answer names that one — it describes what is on the sheet ` +
              `in front of you, not a different drawing. Ask about the ` +
              `border, the title block or the viewport frames to get the ` +
              `sheet's own contents instead.`,
          });
        }

        send({ type: "done", toolCalls, citations, text });
      } catch (cause) {
        send({
          type: "error",
          error: "STREAM_INTERRUPTED",
          message: `The answer stopped part-way: ${cause instanceof Error ? cause.name : "unknown"}.`,
          hint: "Ask again; if it repeats, check `docker compose logs cad-agent`.",
        });
      } finally {
        controller.close();
      }
    },
    cancel() {
      // The user navigated away or asked something else: stop pulling from
      // the agent rather than letting the turn run on unread.
      void reader.cancel();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      // Without this, a proxy that buffers would defeat the whole point.
      "X-Accel-Buffering": "no",
    },
  });
}

/** A tool call's arguments, condensed to what a reader needs to judge it.
 *
 *  This is the citation's substance: "which drawing, which layout, which
 *  filter". Long values are trimmed rather than dropped, because a citation
 *  that hides its scope is not a citation.
 */
function describeArgs(args: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(args)) {
    if (value === null || value === undefined || value === "") continue;
    if (key === "drawing_id") continue; // constant for the whole conversation
    if (Array.isArray(value)) {
      parts.push(`${key}=${value.length} items`);
      continue;
    }
    const text = String(value);
    parts.push(`${key}=${text.length > 40 ? text.slice(0, 40) + "…" : text}`);
  }
  return parts.join(", ");
}

/** Does this answer assert something we never gave it and never read?
 *
 *  Used to decide whether an answer written without any tool call is a
 *  harmless one or an unsourced claim. The test is per-number, and the
 *  distinction it draws is the whole point:
 *
 *  - Figures we SUPPLIED this turn are sourced. When a user selects two
 *    regions, the per-part counts travel in the prompt on purpose, so an
 *    answer that quotes them back is repeating the selection the user just
 *    made, not inventing anything. Flagging that taught people to ignore the
 *    warning, which is how a real one gets missed.
 *  - Figures we did NOT supply, in an answer where nothing was read, came
 *    from somewhere unverifiable. That is the failure this exists for: an
 *    answer once reported 1,928 DIMENSIONs where the truth was 9,867, in a
 *    sentence that was otherwise perfectly formed.
 *
 *  Commas are stripped from both sides so 1,335 and 1335 are one number.
 */
/** Did the question ask for two conditions at once?
 *
 *  Read from what the USER typed, never from the model's interpretation of
 *  it. "both … and", "… AND …" between two clauses, "which … also …" — the
 *  shapes a combination question actually takes. Narrow on purpose: it only
 *  ever fires alongside an empty finding and a trace with no composition, so
 *  the cost of a false positive is one warning on an answer that reported
 *  nothing and did not intersect anything.
 */
function asksForTwoConditions(question: string): boolean {
  const q = (question ?? "").toLowerCase();
  return (
    /both[\s\S]{0,80}and/.test(q) ||
    /and[\s\S]{0,40}also/.test(q) ||
    /which[\s\S]{0,60}also/.test(q)
  );
}

/** Does this answer assert that nothing matched?
 *
 *  Deliberately narrow. It looks for the shapes an empty finding actually
 *  takes in these answers, and it is only ever consulted when a tool has
 *  ALREADY reported that it settled nothing -- so a false positive costs a
 *  warning on an answer that was built on an unsettled reading anyway.
 */
function claimsNothingFound(text: string): boolean {
  const t = text.toLowerCase();
  return (
    /there are no/.test(t) ||
    /no parcels/.test(t) ||
    /none of the/.test(t) ||
    /not? (?:any )?(?:parcels?|entities|layers) (?:were|was) found/.test(t) ||
    /0 parcels?/.test(t) ||
    /no .{0,24}(?:were|are) found/.test(t)
  );
}

/** Is the question about ground covered by two different things?
 *
 *  Read from what the USER typed. Narrow on purpose: it only ever fires
 *  alongside a trace that ran `find_duplicates` and never ran `overlap_scan`,
 *  so a false positive costs one extra reading and a sentence naming it.
 */
function asksAboutSharedGround(question: string): boolean {
  const q = (question ?? "").toLowerCase();
  return (
    /claimed\s+(?:by\s+)?(?:more than one|twice|two)/.test(q) ||
    /double[-\s]?count/.test(q) ||
    /counted\s+twice/.test(q) ||
    /overlap/.test(q) ||
    /same\s+(?:ground|land|area)\s+(?:twice|by two)/.test(q) ||
    /inflated\s+by/.test(q)
  );
}

/** The overlap scan's total shared area, as a sentence, or null.
 *
 *  Returns null rather than a partial phrase on anything unexpected: a
 *  correction that cannot state its own figure is worse than no correction,
 *  because it casts doubt without replacing it.
 */
async function sharedGroundTotal(
  drawingId: string,
  layout: string | null | undefined,
): Promise<string | null> {
  try {
    const query = layout ? `?layout=${encodeURIComponent(layout)}` : "";
    const response = await fetch(
      `${API_INTERNAL}/drawings/${encodeURIComponent(drawingId)}/analyses/overlap_scan${query}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
        signal: AbortSignal.timeout(120_000),
      },
    );
    if (!response.ok) return null;
    const body = (await response.json()) as {
      overlaps?: {
        count?: unknown;
        shared_area_total?: { value?: unknown; unit?: unknown };
      };
    };
    const total = body.overlaps?.shared_area_total;
    const value = total?.value;
    const count = body.overlaps?.count;
    if (typeof value !== "number" || typeof count !== "number") return null;
    const unit = typeof total?.unit === "string" ? total.unit : "";
    return (
      `${value.toLocaleString("en-US", { maximumFractionDigits: 3 })}` +
      `${unit ? " " + unit : ""} over ${count} overlapping pair` +
      `${count === 1 ? "" : "s"}`
    );
  } catch {
    return null;
  }
}

/** Every scope an automatic re-run chose in this response, as text.
 *
 *  Reads the retry's own record rather than guessing from shape: a healed body
 *  carries `auto_retry.plan.params`, and a composition carries `retried_with`
 *  on the side that healed. Both are the parameters as executed, so what is
 *  reported is what actually ran.
 */
function retriedScopes(body: unknown): string[] {
  const out: string[] = [];
  const seen = (value: unknown) => {
    if (!value || typeof value !== "object") return;
    const params = value as Record<string, unknown>;
    const parts = Object.entries(params)
      .filter(([, v]) => v !== null && v !== undefined)
      .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(", ") : String(v)}`);
    if (parts.length) out.push(parts.join(" "));
  };
  const root = body as Record<string, any> | null;
  if (!root || typeof root !== "object") return out;
  if (root.auto_retry?.succeeded === true) seen(root.auto_retry?.plan?.params);
  for (const side of ["left", "right"]) {
    if (root[side]?.retried === true) seen(root[side]?.retried_with);
  }
  return out;
}

/** Does the answer already name the values the retry chose?
 *
 *  Matched on the VALUES, not on the parameter names and not on any wording:
 *  an answer that says "on the ROW layer" has told the reader what they need,
 *  whether or not it used the word `road_layers`. Values shorter than three
 *  characters are skipped -- they match by accident.
 */
function mentionsScope(text: string, scope: string): boolean {
  const haystack = text.toLowerCase();
  const values = scope
    .split(/\s(?=[a-z_]+=)/)
    .map((pair) => pair.slice(pair.indexOf("=") + 1))
    .flatMap((value) => value.split(", "))
    .map((value) => value.trim().toLowerCase())
    .filter((value) => value.length >= 3);
  if (!values.length) return true;
  return values.every((value) => haystack.includes(value));
}

function assertsFacts(text: string, payload: TurnRequest): boolean {
  // The drawing's own name, id and layout are identity, not claims — this
  // file is called "JANADRIYAH DMP - 20240506.dxf", and a refusal that names
  // its own scope should not be accused of inventing figures.
  let rest = text;
  for (const own of [
    payload.drawing_name,
    payload.drawing_id,
    payload.layout,
    payload.scope_layout,
  ]) {
    if (own) rest = rest.split(own).join(" ");
  }
  const digits = (value: string) => value.replace(/,/g, "").match(/\d+/g) ?? [];
  const supplied = new Set(digits(payload.selection?.summary ?? ""));
  if (payload.selection?.total !== undefined) {
    supplied.add(String(payload.selection.total));
  }
  return digits(rest).some((n) => !supplied.has(n));
}


/** Words that only appear in Indonesian, common enough to catch a short
 *  question. Function words rather than nouns: a question can be about a
 *  mosque in either language. */
const INDONESIAN_MARKERS =
  /\b(yang|dan|dari|untuk|dengan|adalah|tidak|ada|ini|itu|saya|anda|apa|apakah|berapa|dimana|bagaimana|tolong|mohon|kenapa|mengapa|atau|juga|bisa|sudah|belum|akan|pada|nya)\b/i;

function languageDirective(message: string): string {
  const name = INDONESIAN_MARKERS.test(message) ? "Indonesian" : "English";
  return (
    `[Write your entire reply in ${name}, because that is the language of ` +
    `the question below. Tool responses contain prose in another language; ` +
    `read it and write what it means in ${name} rather than quoting it ` +
    `across. Numbers, units, handles, layer names and text quoted from the ` +
    `drawing are never translated.]`
  );
}

/** `Date.now`, behind a name, so the cache below can be reasoned about. */
const TimeSource = { now: () => Date.now() };

/** How long the analysis catalogue is reused before it is fetched again. */
const CATALOGUE_TTL_MS = 5 * 60 * 1000;
let catalogueText: string | null = null;
let catalogueFetchedAt = 0;

/** The analysis catalogue, as a menu that travels with the question.
 *
 *  Measured over twenty-four turns on twelve questions, each of which has a
 *  reviewed recipe written to answer it: five reached that recipe. Of those
 *  five, FOUR had called `list_analyses` first. Of the nineteen that missed,
 *  not one had. The recipes were never chosen badly — they were never seen,
 *  because the catalogue sits behind a discovery call the model rarely makes,
 *  while improvising from the generic read tools always looks available.
 *
 *  So the menu moves in front of the question instead of behind a call. This
 *  is data rather than another rule, which matters: the instruction is
 *  already long enough that each added rule made behaviour less predictable,
 *  and what was missing here was never a rule but a list of what exists.
 *
 *  Read from the registry, never written here, so a recipe added or renamed
 *  appears without anyone editing this file — and nothing in it is specific
 *  to one drawing. Cached because it changes at deploy time, not per question.
 */
async function analysisCatalogue(): Promise<string | null> {
  const now = TimeSource.now();
  if (catalogueText !== null && now - catalogueFetchedAt < CATALOGUE_TTL_MS) {
    return catalogueText;
  }
  try {
    const response = await fetch(`${API_INTERNAL}/analyses`, {
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) return catalogueText;
    const body = (await response.json()) as {
      recipes?: { recipe?: string; name?: string; answers?: string }[];
    };
    const rows = (body.recipes ?? [])
      .map((row) => {
        const name = row.recipe ?? row.name;
        if (!name) return null;
        const answers = (row.answers ?? "").replace(/\s+/g, " ").trim();
        return `  ${name} — ${answers}`;
      })
      .filter((row): row is string => row !== null);
    if (!rows.length) return catalogueText;
    catalogueText = rows.join("\n");
    catalogueFetchedAt = now;
  } catch {
    // Fail open. A turn without the catalogue is the behaviour that shipped
    // before it existed, not a broken one.
  }
  return catalogueText;
}

function buildPrompt(
  payload: TurnRequest,
  message: string,
  catalogue: string | null,
): string {
  const lines: string[] = [];

  // --- the world -----------------------------------------------------------
  // The drawing and the layout on screen are not hints, they are the frame
  // the question was asked inside. The user is looking at one sheet of one
  // file; an answer drawn from anywhere else is wrong even when its numbers
  // are right.
  if (payload.drawing_id) {
    lines.push(
      CONTRACT.drawing
        .replace("{drawing_id}", payload.drawing_id)
        .replace(
          "{drawing_name}",
          payload.drawing_name
            ? CONTRACT.drawing_name_suffix.replace(
                "{drawing_name}",
                payload.drawing_name,
              )
            : "",
        ),
    );
  }
  if (payload.layout) {
    const scope = payload.scope_layout || payload.layout;
    const projected = scope !== payload.layout;
    const fill = (t: string) =>
      t.split("{layout}").join(payload.layout!).split("{scope}").join(scope);
    lines.push(
      fill(projected ? CONTRACT.layout_projected : CONTRACT.layout_plain),
    );
    lines.push(fill(CONTRACT.default_scope));
  }

  // --- the focus -----------------------------------------------------------
  const selectionId = payload.selection?.selection_id;
  if (selectionId) {
    const total = payload.selection?.total ?? 0;
    lines.push(
      `[the user has selected ${total} object${total === 1 ? "" : "s"}: ` +
        `${payload.selection?.summary ?? ""}]`,
    );
    lines.push(
      `[selection_id: ${selectionId}. Call ` +
        `describe_selection(drawing_id, selection_id="${selectionId}") — it ` +
        `reads the selection WHOLE, so answer for all ${total} of them and ` +
        `never ask the user to narrow it down.]`,
    );
    // The distinction that makes this useful rather than merely narrow.
    lines.push(
      `[the selection is the FOCUS of the question, not a wall around what ` +
        `you may look at. Answer about it first. When the question needs ` +
        `context — "is this the only one?", "how many more like it?", "is ` +
        `this unusual?" — you SHOULD widen to the layout or the drawing using ` +
        `query_entities, distinct_values or spatial_query, and then label ` +
        `every figure with the scope it came from ("in your selection: 12; ` +
        `on this layout: 340"). An unlabelled number is the one thing that ` +
        `cannot be checked.]`,
    );
  } else {
    lines.push(CONTRACT.no_selection);
  }

  // What the clicked object is a label FOR.
  //
  // Placed after the selection and before the evidence rules, because it
  // qualifies the selection: it does not widen the scope, it says which thing
  // inside that scope the question is most likely about. The rule it carries
  // is in the contract file rather than written here, so that the sentence
  // the model reads and the sentence a reviewer reads are the same sentence.
  if (payload.selection_note) {
    lines.push(payload.selection_note);
    lines.push(CONTRACT.label_selection);
  }

  lines.push(CONTRACT.evidence);

  // The language to answer in, said on the turn rather than only in the
  // standing instruction.
  //
  // The standing rule was tried twice and lost twice, and the wording was
  // not the problem. Several tool responses are mostly Indonesian prose --
  // the evidence `statement`, `basis`, `method`, `scope_note` -- and for a
  // question whose answer IS that response, the model follows the language
  // it is reading rather than the one it was told. Measured: three
  // consecutive English questions answered entirely in Indonesian with the
  // rule at the top of the instruction the whole time. Making the wording
  // firmer made it worse, because the firmer version quoted the Indonesian
  // it was warning about.
  //
  // Said here it arrives in the same turn as the question, immediately
  // before it. Detection is deliberately crude: these are the only two
  // languages this stack is used in, and being wrong costs a reply in the
  // other one -- which is the failure it is fixing, so it cannot do worse
  // than not trying.
  // Read again, this turn, and here is what happens if you do not.
  //
  // The session keeps its history, so the tool results of earlier questions
  // are sitting in the context and are cheaper to reuse than to re-fetch.
  // Twice on 25 August 2026 the model did exactly that: asked where the
  // villas and houses are, and later how many house types there are, it
  // answered from the previous turn's `land_use_summary`, ran no tool, and
  // was published under the "do not trust the figures" warning — with
  // nothing marked on the drawing, because marks come from what the tools
  // read this turn. The figures happened to be right both times.
  //
  // The system instruction already said to read again and did not hold. What
  // is different here is placement and consequence: it arrives in the same
  // turn as the question, and it names what the reader will see.
  lines.push(
    "[Answer this turn from fresh tool reads. Tool results earlier in this " +
      "conversation belong to earlier questions and may be scoped " +
      "differently. If no tool runs this turn, the reply is shown to the " +
      "user under a warning that its figures cannot be trusted, and no " +
      "object is marked in the drawing — both of which happen even when the " +
      "numbers are correct. Refusals and clarifying questions are exempt: " +
      "they assert nothing.]",
  );
  // The menu, in front of the question rather than behind a call.
  if (catalogue) {
    lines.push(
      "[Reviewed analyses, available through run_analysis. Each is server-side " +
        "code with published limits and its own evidence. When the question " +
        "matches what one of them answers, run it rather than assembling an " +
        "answer from the generic read tools — call list_analyses for its " +
        "parameters. When none of them matches, say so and answer from the " +
        "read tools instead:\n" +
        catalogue +
        "]",
    );
  }
  lines.push(languageDirective(message));
  lines.push(message);
  return lines.join("\n");
}

/** Words that mean "put it on the drawing", in either language used here. */
const LOCATING_WORDS =
  /\b(where|show|locate|highlight|mark|point out|find)\b|\b(mana|dimana|tunjukkan|tampilkan|tandai)\b/i;

/** Layer names under any `by_layer` array in a response. */
/** How many distinct layers one turn may resolve to handles.
 *
 *  A bound, not a preference. Paging a layer costs one request, and a response
 *  that mentions every layer in a large drawing would otherwise turn one
 *  question into hundreds of round trips. Whatever is dropped is dropped
 *  silently here only because nothing is claimed about it: the marks are an
 *  aid, and the answer's own text is the finding.
 */
const MAX_LOCATABLE_LAYERS = 40;

/** Every layer name a tool response mentions, wherever it sits.
 *
 *  Shape-agnostic, for the same reason `collectHandles` is. It began by
 *  reading `by_layer` rows alone, which is the shape `land_use_summary`
 *  happens to use — so `parcel_inventory`, `size_module` and
 *  `junction_census` all named layers the viewer then could not act on, and
 *  asking to be shown their subject marked nothing. `layer` is a DXF field
 *  name, true of every drawing, so nothing here is tied to one file.
 *
 *  Three shapes, because these are the three the registry actually produces:
 *  a `by_layer` array of rows, a row carrying its own `layer`, and a `layers`
 *  array of names.
 */
function collectLandUseLayers(value: unknown, depth = 0, out: string[] = []): string[] {
  if (depth > 6 || value == null || typeof value !== "object") return out;
  if (out.length >= MAX_LOCATABLE_LAYERS) return out;
  const push = (name: unknown) => {
    if (
      typeof name === "string" &&
      name.trim() &&
      !out.includes(name) &&
      out.length < MAX_LOCATABLE_LAYERS
    ) {
      out.push(name);
    }
  };
  if (Array.isArray(value)) {
    for (const item of value) collectLandUseLayers(item, depth + 1, out);
    return out;
  }
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (key === "by_layer" && Array.isArray(child)) {
      for (const row of child) push((row as Record<string, unknown>)?.layer);
      continue;
    }
    if (key === "layer") {
      push(child);
      continue;
    }
    if (key === "layers" && Array.isArray(child)) {
      for (const item of child) {
        // Either a bare name, or a row that carries one. Both occur.
        if (typeof item === "string") push(item);
        else push((item as Record<string, unknown>)?.layer);
      }
      continue;
    }
    collectLandUseLayers(child, depth + 1, out);
  }
  return out;
}

/** Fill in the objects a truncated query matched but did not return.
 *
 *  This exists because prompting could not be made to work, in either
 *  direction, and the drawing paid for it. Told to raise `limit` so the marks
 *  would cover the set, the model asked for 200 and then printed all 133
 *  handles into a sentence nobody reads. Told to stop printing handles, it
 *  asked for `limit=1`, marked one plot, and wrote "these are now highlighted
 *  in the viewer" — an answer that is confident, short, and false.
 *
 *  So the number of objects marked stops depending on what the model chose.
 *  When a query reports more matches than it returned, the same filter is
 *  re-run here, server-side, for handles only. The model never sees the extra
 *  rows, so its context does not grow; the viewer sees the whole set, so what
 *  it draws matches what the answer claims.
 *
 *  Bounded on purpose. Above `MAX_READ_HANDLES` nothing is fetched: the
 *  viewer stops marking individually at 200 and says so, and pulling three
 *  thousand handles to throw most of them away is work that buys a worse
 *  answer.
 */
async function completeTruncatedSet(
  tool: string,
  args: Record<string, unknown> | undefined,
  body: Record<string, unknown> | undefined,
  drawingId: string | undefined,
  into: Set<string>,
  groupOf?: Map<string, string>,
): Promise<void> {
  if (tool !== "query_entities" || !args || !body || !drawingId) return;
  if (args.drawing_id && args.drawing_id !== drawingId) return;
  // The counts are looked for anywhere in the response, not at its top level.
  // ADK wraps a tool result before handing it on, and MCP wraps it again; the
  // first version of this read `body.total_matches` directly, found undefined
  // every time, and silently did nothing -- a set-completion step that never
  // completed a set, and never said so.
  const counts = findCounts(body);
  if (!counts) return;
  const { total, returned } = counts;
  // No cliff at the ceiling. An earlier version bailed out entirely when a
  // set was larger than the cap, so 800 matches produced a full mark-up and
  // 801 produced whatever the model had happened to type — a difference of
  // one object changing the picture completely, with nothing on screen
  // saying why. Now the cap only limits how much is fetched; the viewer draws
  // what it can and the badge says how many it could not.
  if (returned >= total) return;

  const filters: Record<string, string> = {};
  for (const key of ["layer", "type", "block_name", "layout", "text_contains"]) {
    const value = args[key];
    if (typeof value === "string" && value) filters[key] = value;
  }
  if (!Object.keys(filters).length) return; // an unfiltered sweep is not a set
  await pageHandles(
    drawingId,
    filters,
    Math.min(total, MAX_READ_HANDLES),
    into,
    groupOf,
  );
}

/** Read handles for one filter, a page at a time, into `into`.
 *
 *  Paged because cad-api caps a listing at 100 rows, and that cap exists to
 *  protect the MODEL's context. Nothing fetched here reaches the model, so
 *  the right answer to the cap is to turn the pages rather than raise it for
 *  everyone: 512 plots is six requests and no extra tokens.
 */
async function pageHandles(
  drawingId: string,
  filters: Record<string, string>,
  wanted: number,
  into: Set<string>,
  groupOf?: Map<string, string>,
): Promise<void> {
  // One request, not one per hundred.
  //
  // This used to page `/entities`, whose 100-row cap protects the MODEL's
  // context — every row there is a field competing with the answer. Nothing
  // fetched here reaches the model, so paying that cap cost 99 round trips
  // and 133 seconds to mark the 9,867 dimensions of the reference drawing.
  // `/entities/handles` returns seven characters per object and no geometry.
  const query = new URLSearchParams(filters);
  query.set("limit", String(wanted));
  try {
    const response = await fetch(
      `${API_INTERNAL}/drawings/${encodeURIComponent(drawingId)}/entities/handles?${query}`,
      { signal: AbortSignal.timeout(30_000) },
    );
    if (!response.ok) return;
    const body = (await response.json()) as { handles?: unknown };
    if (!Array.isArray(body.handles)) return;
    for (const handle of body.handles) {
      if (typeof handle !== "string") continue;
      if (into.size >= MAX_READ_HANDLES) return;
      into.add(handle);
      if (groupOf && filters.layer && !groupOf.has(handle)) {
        groupOf.set(handle, filters.layer);
      }
    }
  } catch {
    // The marks are an extra. Failing to widen them is not a reason to lose
    // the answer they belong to.
  }
}

/** Find `{total_matches, returned}` wherever a wrapper put them.
 *
 *  Shape-agnostic for the same reason `collectHandles` is: this proxy sits
 *  between two frameworks that each add an envelope, and a hard-coded path
 *  through them is a path that breaks on an upgrade without a single error.
 */
function findCounts(
  value: unknown,
  depth = 0,
): { total: number; returned: number } | null {
  if (depth > 6 || value == null || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (
    typeof record.total_matches === "number" &&
    typeof record.returned === "number"
  ) {
    return { total: record.total_matches, returned: record.returned };
  }
  for (const child of Object.values(record)) {
    const hit = findCounts(child, depth + 1);
    if (hit) return hit;
  }
  return null;
}

/** Ceiling on handles forwarded from tool responses.
 *
 *  Matched to the viewer's own ceiling, and both are set high enough that a
 *  real question does not reach them: "where are all the house types" is
 *  2,380 parcels across thirteen layers, and an earlier 800 here quietly
 *  turned that into a dashed rectangle. A handle is seven characters; four
 *  thousand of them is about thirty kilobytes on the one turn that asks for
 *  it. The viewer -- not this proxy -- decides where marking stops, and says
 *  so on the badge.
 */
const MAX_READ_HANDLES = 12000;

/** Walk a tool response and collect every `handle` it carries.
 *
 *  Deliberately shape-agnostic. Twenty-two tools return handles under a dozen
 *  different keys -- `entities`, `rows`, `labels`, `contained_by`, `members`,
 *  `sample_by_group` -- and a list of known paths would go stale the first
 *  time a tool grew a field. A `handle` is a `handle` wherever it sits.
 */
/** A DXF handle: hex, and short. Named once so the two tests cannot drift. */
const HANDLE = /^[0-9A-Fa-f]{2,7}$/;

function collectHandles(value: unknown, into: Set<string>, depth = 0): void {
  if (into.size >= MAX_READ_HANDLES || depth > 8 || value == null) return;
  if (Array.isArray(value)) {
    for (const item of value) collectHandles(item, into, depth + 1);
    return;
  }
  if (typeof value !== "object") return;
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (
      typeof child === "string" &&
      (key === "handle" || key === "target_handle") &&
      HANDLE.test(child)
    ) {
      into.add(child);
      if (into.size >= MAX_READ_HANDLES) return;
      continue;
    }
    // A LIST of bare handles, which is how several tools publish a set.
    //
    // Measured: `overlap_scan` writes each pair as
    // `handles: ["205CD54", "220B253"]` -- strings, not objects with a
    // `handle` field -- so 23 pairs and 46 parcels were invisible to the
    // viewer while the answer described them in prose. The reader was told
    // which parcels overlap and shown none of them.
    //
    // Only a plural key holding hex strings qualifies, so a list of layer
    // names or numbers cannot be mistaken for one.
    if (
      Array.isArray(child) &&
      (key === "handles" || key === "target_handles") &&
      child.every((item) => typeof item === "string" && HANDLE.test(item))
    ) {
      for (const item of child as string[]) {
        into.add(item);
        if (into.size >= MAX_READ_HANDLES) return;
      }
      continue;
    }
    collectHandles(child, into, depth + 1);
  }
}

function sseError(error: string, message: string, hint: string): Response {
  const body = `data: ${JSON.stringify({ type: "error", error, message, hint })}\n\n`;
  return new Response(body, {
    headers: { "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache" },
  });
}

/** The prompt for the second attempt, when the first asserted without reading.
 *
 *  Deliberately not the original question with a scolding attached. The task
 *  IS the read: naming the tools it may reach for and forbidding the shortcut
 *  it just took leaves less room to take it again than any amount of firmer
 *  wording did.
 */
function retryPrompt(
  payload: TurnRequest,
  message: string,
  catalogue: string | null,
): string {
  return [
    "[Your previous reply to this question was written without calling any " +
      "tool, so nothing in it can be traced to the drawing and nothing was " +
      "marked on screen. It has not been shown to the user. Answer again, " +
      "and this time CALL THE TOOLS FIRST — `land_use_summary`, `measure`, " +
      "`stats`, `query_entities`, `proximity_count`, `scheduled_area` or " +
      "whichever fit — then write the answer from what they return. Do not " +
      "reuse figures from earlier in this conversation, even if you are " +
      "confident they are correct: they were measured for a different " +
      "question and the reader has no way to check them.]",
    buildPrompt(payload, message, catalogue),
  ].join("\n");
}

/** What the panel says while the second attempt runs. */
function rereadNote(): string {
  return (
    "That answer was written without reading the drawing, so it was not " +
    "shown. Reading it now."
  );
}
