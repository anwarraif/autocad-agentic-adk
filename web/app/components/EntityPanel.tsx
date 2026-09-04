"use client";

/**
 * Detail and comments for the selected entity.
 *
 * Comment submission is optimistic: the comment appears immediately marked
 * `pending`, and is replaced by the server's copy on success or rolled back
 * with an error on failure. A comment is a small, low-risk write; making the
 * user watch a spinner for a round-trip makes the app feel broken when it is
 * merely polite.
 *
 * `client_request_id` is generated per submission so that a retry after a
 * flaky network cannot leave two identical comments on the same door.
 */

import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  addComment,
  getEntity,
  type Comment,
  type EntityDetail,
} from "@/lib/api";
import type { Selection } from "./Viewer";
import type { Optional, ParcelContext } from "@/lib/parcelContext";

interface Props {
  drawingId: string;
  selection: Selection | null;
  onCommentAdded: (handle: string) => void;
  /** The parcel a selected label sits inside, when the selection is a label
   *  and cad-api can answer.
   *
   *  This panel is where the failure of 23 August at 23:28 was visible: the
   *  plot number `2043` was clicked and everything on screen described a
   *  TEXT entity on a layer. Every fact was right and none of them was the
   *  answer, because the thing the number stands for is the parcel under it.
   *
   *  Fetched by the page rather than here, because the same lookup is sent to
   *  the agent as context and one fetch has to serve both — otherwise the
   *  panel and the model can be looking at different parcels.
   */
  parcel: Optional<ParcelContext> | null;
  parcelLoading: boolean;
  /** Select the containing parcel, so the panel is a way through to it and
   *  not merely a mention of it. */
  onSelectHandle: (handle: string) => void;
}

interface PendingComment extends Comment {
  pending?: boolean;
}

export function EntityPanel({
  drawingId,
  selection,
  onCommentAdded,
  parcel,
  parcelLoading,
  onSelectHandle,
}: Props) {
  const [entity, setEntity] = useState<EntityDetail | null>(null);
  const [comments, setComments] = useState<PendingComment[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!selection) {
      setEntity(null);
      setComments([]);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    getEntity(drawingId, selection.handle)
      .then((detail) => {
        if (cancelled) return;
        setEntity(detail);
        setComments(detail.comments ?? []);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setEntity(null);
        setError(
          err instanceof ApiError
            ? `${err.message}${err.hint ? ` — ${err.hint}` : ""}`
            : String(err),
        );
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [drawingId, selection]);

  const submit = useCallback(async () => {
    if (!selection || !draft.trim() || submitting) return;
    const body = draft.trim();
    const requestId = crypto.randomUUID();
    const optimistic: PendingComment = {
      _id: requestId,
      drawing_id: drawingId,
      entity_handle: selection.handle,
      body,
      author: "user:local",
      created_at: new Date().toISOString(),
      status: "open",
      pending: true,
    };

    setSubmitting(true);
    setComments((prev) => [...prev, optimistic]);
    setDraft("");
    setError(null);

    try {
      const saved = await addComment({
        drawing_id: drawingId,
        handle: selection.handle,
        body,
        author: "user:local",
        client_request_id: requestId,
      });
      setComments((prev) =>
        prev.map((c) => (c._id === requestId ? { ...saved, pending: false } : c)),
      );
      onCommentAdded(selection.handle);
    } catch (err: unknown) {
      // Roll the optimistic comment back and put the text back in the box, so
      // nothing the user typed is lost.
      setComments((prev) => prev.filter((c) => c._id !== requestId));
      setDraft(body);
      setError(
        err instanceof ApiError
          ? `${err.message}${err.hint ? ` — ${err.hint}` : ""}`
          : String(err),
      );
    } finally {
      setSubmitting(false);
    }
  }, [drawingId, draft, selection, submitting, onCommentAdded]);

  if (!selection) {
    return (
      <p className="muted">
        Click any entity in the drawing to inspect it and attach a comment.
      </p>
    );
  }

  return (
    <div className="entity-panel">
      <dl className="entity-facts">
        <dt>Handle</dt>
        <dd>
          <code>{selection.handle}</code>
        </dd>
        <dt>Type</dt>
        <dd>{entity?.type ?? selection.type ?? "—"}</dd>
        <dt>Layer</dt>
        <dd title={entity?.layer ?? selection.layer}>
          {entity?.layer ?? selection.layer ?? "—"}
        </dd>
        {entity?.block_name && (
          <>
            <dt>Block</dt>
            <dd>{entity.block_name}</dd>
          </>
        )}
        {entity?.layout && (
          <>
            <dt>Layout</dt>
            <dd>{entity.layout}</dd>
          </>
        )}
        {entity?.length != null && (
          <>
            <dt>Length</dt>
            <dd>{entity.length.toFixed(3)}</dd>
          </>
        )}
        {entity?.area != null && (
          <>
            <dt>Area</dt>
            <dd>{entity.area.toFixed(3)}</dd>
          </>
        )}
      </dl>

      {loading && <p className="muted small">Loading entity…</p>}

      {entity?.text && (
        <div className="entity-text">
          <h4>Text</h4>
          <p>{entity.text}</p>
        </div>
      )}

      {/* What this label is a label FOR.
          Shown above the comment box and below the text, because that is the
          reading order of the question people actually ask: what does this
          say, and what does it say it about. */}
      {parcelLoading && (
        <p className="muted small">Looking for the parcel this sits inside…</p>
      )}
      {parcel?.data && parcel.data.contained_by.length > 0 && (
        <div className="entity-parcel">
          <h4>Inside</h4>
          {parcel.data.contained_by.map((p) => (
            <dl className="entity-facts" key={p.handle}>
              <dt>Parcel</dt>
              <dd>
                <button
                  className="link-button"
                  onClick={() => onSelectHandle(p.handle)}
                  title="Select this parcel in the drawing"
                >
                  <code>{p.handle}</code>
                </button>
              </dd>
              <dt>Layer</dt>
              <dd title={p.layer}>{p.layer}</dd>
              {p.area != null && (
                <>
                  <dt>Area</dt>
                  {/* The unit comes from the drawing, and a drawing is
                      allowed not to declare one. 11 of the 16 files here are
                      in inches and 3 state nothing; writing "m" because the
                      last file happened to be metric is how a viewer starts
                      lying quietly (G2). */}
                  <dd>
                    {p.area.toLocaleString(undefined, {
                      maximumFractionDigits: 3,
                    })}{" "}
                    <span className="muted">
                      {p.area_unit ?? "drawing units"}
                    </span>
                  </dd>
                </>
              )}
              {p.land_use && (
                <>
                  <dt>Land use</dt>
                  <dd>
                    {p.land_use}
                    {/* An unverified classification came from a pattern over
                        a layer name, which is an inference. Printing it bare
                        would promote a guess to a caption (G4). */}
                    {p.land_use_verified === false && (
                      <span className="muted"> (inferred, not verified)</span>
                    )}
                  </dd>
                </>
              )}
            </dl>
          ))}
          {parcel.data.note && (
            <p className="muted small">{parcel.data.note}</p>
          )}
        </div>
      )}
      {parcel?.data && parcel.data.contained_by.length === 0 && (
        <p className="muted small">
          This label does not fall inside any closed shape on this layout.
          That is an absence of a parcel around it, not a failure to look.
        </p>
      )}
      {/* Absence explains itself. A panel that simply omits the section when
          the route is missing would look identical to one where the label
          genuinely sits inside nothing, and those are different facts. */}
      {parcel && !parcel.data && parcel.reason && (
        <p className="muted small">{parcel.reason}</p>
      )}

      {entity?.attribs && Object.keys(entity.attribs).length > 0 && (
        <div className="entity-attribs">
          <h4>Block attributes</h4>
          <dl className="entity-facts">
            {Object.entries(entity.attribs).map(([tag, value]) => (
              <div key={tag} className="attrib-row">
                <dt>{tag}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      <div className="comments">
        <h4>Comments ({comments.length})</h4>
        {comments.length === 0 && (
          <p className="muted small">No comments on this entity yet.</p>
        )}
        <ul className="comment-list">
          {comments.map((comment) => (
            <li key={comment._id} className={comment.pending ? "pending" : ""}>
              <p>{comment.body}</p>
              <span className="comment-meta">
                {comment.author}
                {comment.pending
                  ? " · saving…"
                  : ` · ${new Date(comment.created_at).toLocaleString()}`}
              </span>
            </li>
          ))}
        </ul>

        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={`Comment on ${selection.handle}…`}
          rows={3}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") void submit();
          }}
        />
        <div className="comment-actions">
          <button
            onClick={() => void submit()}
            disabled={!draft.trim() || submitting}
          >
            {submitting ? "Saving…" : "Add comment"}
          </button>
          <span className="muted small">⌘/Ctrl + Enter</span>
        </div>
      </div>

      {error && <p className="error">{error}</p>}
    </div>
  );
}
