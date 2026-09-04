/**
 * BFF for the ADK agent.
 *
 * The browser never talks to the agent directly: the agent URL stays inside
 * the compose network, and this is where auth would attach when this leaves
 * localhost.
 *
 * ADK's api_server exposes a session-per-conversation model, so a turn is two
 * calls: ensure the session exists, then post the message. A missing agent is
 * reported as a normal JSON error with a hint, never as a thrown 500 — the
 * viewer stays usable when the agent is down.
 */

const AGENT_URL = process.env.AGENT_URL ?? "http://cad-agent:8000";
const APP_NAME = process.env.CAD_AGENT_APP ?? "cad_agent";
const USER_ID = "local";

/** A tool-using turn can legitimately run long; this is the ceiling before
 *  the caller is told the agent is stuck rather than left waiting. */
const AGENT_TIMEOUT_MS = Number(process.env.AGENT_TIMEOUT_MS ?? 180_000);

interface TurnRequest {
  message?: string;
  drawing_id?: string;
  session_id?: string;
  selection?: { handles?: string[]; total?: number; summary?: string };
}

/** Handles carried into a turn as context.
 *
 *  Below `describe_selection`'s own 5,000 cap on purpose. This list is
 *  re-sent verbatim with EVERY message in the conversation, so it is a
 *  recurring cost, not a one-off: 5,000 handles is roughly 40 KB of literal
 *  hex per turn. 2,000 keeps that near 16 KB while still covering the great
 *  majority of real selections whole.
 *
 *  When a selection is larger, the counts stay exact and the prompt says how
 *  many handles the model is actually holding — a model told "15,377 objects"
 *  while given 2,000 handles would otherwise reason from a number that does
 *  not describe what it can reach. */
const MAX_CONTEXT_HANDLES = 2000;

export async function POST(request: Request): Promise<Response> {
  let payload: TurnRequest;
  try {
    payload = await request.json();
  } catch {
    return json(400, {
      error: "BAD_REQUEST",
      message: "Body must be JSON.",
      hint: "Send {message, drawing_id, session_id}.",
    });
  }

  const message = (payload.message ?? "").trim();
  if (!message) {
    return json(400, {
      error: "BAD_REQUEST",
      message: "Empty message.",
      hint: "Type a question first.",
    });
  }

  // Path segment, so it must be escaped: it comes straight from the client
  // body, and a value containing "/" or ".." would reshape the ADK API path.
  const sessionId = encodeURIComponent(payload.session_id ?? "default");

  try {
    // Idempotent: ADK returns 400 if the session already exists, which is
    // fine — we only need it to exist, not to be new.
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

    const prompt = buildPrompt(payload, message);

    // A multi-tool turn can legitimately take tens of seconds, but without a
    // ceiling a stalled model call left the request hanging for ever and the
    // chat panel's "Asking…" button disabled with no way back.
    let response: Response;
    try {
      response = await fetch(`${AGENT_URL}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          app_name: APP_NAME,
          user_id: USER_ID,
          session_id: sessionId,
          new_message: { role: "user", parts: [{ text: prompt }] },
        }),
        signal: AbortSignal.timeout(AGENT_TIMEOUT_MS),
      });
    } catch (cause) {
      // Separated from the parse failures below: "cannot connect" and "the
      // agent replied with nonsense" need different advice, and reporting
      // both as UNREACHABLE sends people to debug a container that is fine.
      const timedOut = cause instanceof Error && cause.name === "TimeoutError";
      return json(504, {
        error: timedOut ? "AGENT_TIMEOUT" : "AGENT_UNREACHABLE",
        message: timedOut
          ? `The agent did not answer within ${AGENT_TIMEOUT_MS / 1000}s.`
          : `Could not reach the agent at ${AGENT_URL}.`,
        hint: timedOut
          ? "The model call may be stuck. Try a narrower question, or check `docker compose logs cad-agent`."
          : "The agent container may not be running: `docker compose ps cad-agent`. Everything else in the viewer works without it.",
      });
    }

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      return json(response.status, {
        error: "AGENT_ERROR",
        message: `Agent returned ${response.status}.`,
        hint:
          detail.slice(0, 400) ||
          "Check `docker compose logs cad-agent`. The viewer works without it.",
      });
    }

    let events: unknown;
    try {
      events = await response.json();
    } catch {
      return json(502, {
        error: "AGENT_BAD_RESPONSE",
        message: "The agent replied with a body that is not JSON.",
        hint: "The agent is reachable but returned something unexpected. Check `docker compose logs cad-agent`.",
      });
    }

    const toolCalls = countToolCalls(events);
    return Response.json({
      reply: guardUngroundedAnswer(extractReply(events), toolCalls, payload),
      tool_calls: toolCalls,
      raw_events: Array.isArray(events) ? events.length : 0,
    });
  } catch (cause) {
    return json(500, {
      error: "BFF_ERROR",
      message: `Unexpected failure talking to the agent: ${
        cause instanceof Error ? cause.name : "unknown"
      }.`,
      hint: "This is a bug in the web layer, not in the agent. The viewer works without the agent.",
    });
  }
}

/** ADK /run returns the whole event list for the turn; the answer is the text
 *  of the final model events. Tool-call events carry no text and are skipped. */
function extractReply(events: unknown): string {
  if (!Array.isArray(events)) return "(no reply)";
  const chunks: string[] = [];
  for (const event of events) {
    const parts = (event as { content?: { parts?: { text?: string }[] } })
      ?.content?.parts;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      if (typeof part.text === "string" && part.text.trim()) {
        chunks.push(part.text.trim());
      }
    }
  }
  return chunks.length > 0 ? chunks.join("\n\n") : "(the agent returned no text)";
}

function json(status: number, body: Record<string, unknown>): Response {
  return Response.json(body, { status });
}


/** Assemble the turn the agent actually sees.
 *
 *  The selection is described, never inlined as data: the model gets the
 *  handles and is pointed at the tools that turn them into facts. Building
 *  this here rather than in the browser keeps the wording of the agent
 *  contract on the server, next to the rest of the ADK plumbing.
 */
function buildPrompt(payload: TurnRequest, message: string): string {
  const lines: string[] = [];
  if (payload.drawing_id) {
    lines.push(`[active drawing_id: ${payload.drawing_id}]`);
  }

  const handles = (payload.selection?.handles ?? []).slice(0, MAX_CONTEXT_HANDLES);
  if (handles.length > 0) {
    const total = payload.selection?.total ?? handles.length;
    lines.push(
      `[the user has selected an area of the drawing: ${payload.selection?.summary ?? ""}]`,
    );
    lines.push(
      `[you hold ${handles.length} of those ${total} handles. Call ` +
        `describe_selection(drawing_id, handles) for the shape of the ` +
        `selection — counts per layer and type, total length and area, ` +
        `extents — and get_entity(drawing_id, handle) for any single object ` +
        `worth a closer look. Answer only from what the tools return, and ` +
        `quote handles so every claim can be checked in the viewer.]`,
    );
    if (handles.length < total) {
      lines.push(
        `[note: the selection is larger than the handle list you were given, ` +
          `so per-handle answers cover ${handles.length} of ${total}. Say so ` +
          `if it matters to the answer.]`,
      );
    }
    lines.push(`[selection handles: ${handles.join(",")}]`);
  }

  lines.push(message);
  return lines.join("\n");
}


/** How many tool calls the agent made in this turn. */
function countToolCalls(events: unknown): number {
  if (!Array.isArray(events)) return 0;
  let count = 0;
  for (const event of events) {
    const parts = (event as { content?: { parts?: Record<string, unknown>[] } })
      ?.content?.parts;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      if (part.functionCall) count += 1;
    }
  }
  return count;
}

/** Refuse to pass off an answer the agent produced without touching the data.
 *
 *  This exists because of a measured failure, not a hypothetical one. When
 *  cad-mcp is restarted, the agent's MCP session dies; ADK reports that as a
 *  *warning* -- "Failed to get tools from toolset McpToolset: Session
 *  terminated" -- and then runs the turn with no tools at all. The model does
 *  not say it has no tools. It answers anyway, fluently and specifically:
 *  asked for the total length of a selection five times in a row it returned
 *  29,484.58, 738.74, 514,642.50, 2,238.93 and 20,402.68. The true figure is
 *  5,965.078. It also reported "13 layers" for a drawing with 292.
 *
 *  Every question about a drawing requires at least one tool call, because
 *  the agent has no other way to see the drawing. Zero calls therefore means
 *  the answer is invented, and the only safe thing to do with it is say so.
 *  The text is still shown, because hiding it would make the failure harder
 *  to diagnose, not easier.
 */
function guardUngroundedAnswer(
  reply: string,
  toolCalls: number,
  payload: TurnRequest,
): string {
  if (toolCalls > 0 || !payload.drawing_id) return reply;
  return (
    "⚠ This answer did not read the drawing. The agent made no tool " +
    "calls, which means it could not reach cad-mcp — restarting cad-mcp " +
    "ends the session the agent holds, and it does not reconnect. Do not " +
    "trust anything below; run `docker compose restart cad-agent` and ask " +
    "again.\n\n---\n\n" +
    reply
  );
}
