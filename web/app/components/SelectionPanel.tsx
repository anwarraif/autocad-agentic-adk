"use client";

/**
 * What a region selected — summary first, rows only on request.
 *
 * The design constraint that shapes everything here: a region drawn over a
 * site plan routinely matches thousands of objects. Janadriyah's modelspace
 * holds 20,330. A flat list of "the selection" is therefore not a smaller
 * version of this panel, it is an unusable one — the reviewer scrolls past
 * 1,284 rows of DIMENSION looking for the two plot numbers they wanted.
 *
 * So the panel answers three questions in order, and never fetches data for a
 * question that has not been asked:
 *
 *   1. How much did I select?      — one line, always
 *   2. What kind of thing is it?   — counts per layer and per type
 *   3. Show me those ones          — rows, one (layer, type) group at a time
 *
 * Step 3 is the only call that returns entities, and it is paged.
 */

import { useCallback, useEffect, useState } from "react";
import {
  type Region,
  type RegionGroup,
  type SelectionResult,
} from "@/lib/api";
import type { CombinedSelection, SelectionGroup } from "@/lib/selectionSet";

/** Above this many matches the group rows are not offered until asked for a
 *  second time, and the summary stands on its own. Chosen to match the
 *  reviewer's own threshold in the brief: below it a person may reasonably
 *  want to see everything, above it they want to narrow first. */
const BROWSE_THRESHOLD = 500;

interface Props {
  region: Region | null;
  /** The parts the selection is made of. One is the ordinary case; more than
   *  one is what a user builds by drawing a second region without clearing
   *  the first. */
  parts: SelectionGroup[];
  /** The parts folded together, with the facts about the folding — how much
   *  overlapped, and whether any part hit its cap. Null when nothing is
   *  selected. */
  combined: CombinedSelection | null;
  onRemovePart: (id: string) => void;
  result: SelectionResult | null;
  loading: boolean;
  error: string | null;
  /** Flip window <-> crossing and re-run the same region. */
  onModeChange: (mode: "window" | "crossing") => void;
  onClear: () => void;
  onSelectHandle: (handle: string) => void;
  /** Hand the selection to the agent tab as context. */
  onAskAgent: () => void;
  /** False when this render carries no drawing->viewBox transform. */
  supported: boolean;
  /** Rows for one (layer, type) group of the current selection. Injected
   *  because the answer lives in different places per mechanism: the server
   *  re-evaluates the region for stored-truth selections, while a
   *  rendered-geometry selection already holds its matches client-side. The
   *  panel neither knows nor cares which — it just asks. */
  fetchGroupRows: (group: RegionGroup, partId?: string) => Promise<
    { handle: string; type: string; layer: string; text: string | null }[]
  >;
}

type Row = { handle: string; type: string; layer: string; text: string | null };

/** One part of a multi-part selection: its own counts, its own rows, its own
 *  removal. Kept as a component so each part owns its expansion state — the
 *  alternative was keying three maps in the parent by part AND group, which
 *  is the same thing spelled worse. */
function PartBlock({
  part,
  onRemove,
  onSelectHandle,
  fetchGroupRows,
}: {
  part: SelectionGroup;
  onRemove: () => void;
  onSelectHandle: (handle: string) => void;
  fetchGroupRows: (group: RegionGroup, partId?: string) => Promise<Row[]>;
}) {
  const [expanded, setExpanded] = useState(false);
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [rows, setRows] = useState<Record<string, Row[]>>({});
  const [busy, setBusy] = useState<string | null>(null);

  // `kind` says how this part was made; `region` only says whether its
  // outline can be redrawn on the sheet. A rectangle dragged on the map has
  // no drawing-coordinate shape and used to describe itself as "clicked".
  const shape =
    part.result.kind === "click"
      ? "clicked"
      : `${part.result.mode} ${part.result.kind}`;

  const openGroup = useCallback(
    async (group: RegionGroup) => {
      const key = keyOf(group);
      if (openKey === key) {
        setOpenKey(null);
        return;
      }
      setOpenKey(key);
      if (rows[key]) return;
      setBusy(key);
      try {
        const entities = await fetchGroupRows(group, part.id);
        setRows((r) => ({ ...r, [key]: entities }));
      } finally {
        setBusy(null);
      }
    },
    [openKey, rows, fetchGroupRows, part.id],
  );

  return (
    <li className="selection-part">
      <div className="selection-part-head">
        <button
          className="selection-part-toggle"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
        >
          <span className="selection-part-caret">{expanded ? "▾" : "▸"}</span>
          <strong>{part.label}</strong>
          <span className="muted"> · {shape} · </span>
          {part.result.total.toLocaleString()} object
          {part.result.total === 1 ? "" : "s"}
        </button>
        <button
          className="selection-part-remove"
          onClick={onRemove}
          title={`Remove ${part.label} from the selection`}
          aria-label={`Remove ${part.label}`}
        >
          ×
        </button>
      </div>

      {expanded && (
        <ul className="selection-part-groups">
          {part.result.groups.slice(0, 40).map((group) => {
            const key = keyOf(group);
            return (
              <li key={key}>
                <button
                  className="selection-group-row"
                  onClick={() => openGroup(group)}
                  aria-expanded={openKey === key}
                >
                  <span className="selection-group-name">
                    {group.layer} · {group.type}
                  </span>
                  <span className="selection-group-count">
                    {group.count.toLocaleString()}
                  </span>
                </button>
                {openKey === key && (
                  <ul className="selection-rows">
                    {busy === key && <li className="muted small">reading…</li>}
                    {(rows[key] ?? []).map((row) => (
                      <li key={row.handle}>
                        <button
                          className="handle-chip"
                          onClick={() => onSelectHandle(row.handle)}
                        >
                          {row.handle}
                        </button>
                        {row.text ? <span className="muted"> {row.text}</span> : null}
                      </li>
                    ))}
                    {busy !== key && (rows[key] ?? []).length === 0 && (
                      <li className="muted small">no rows returned</li>
                    )}
                  </ul>
                )}
              </li>
            );
          })}
          {part.result.groups.length > 40 && (
            <li className="muted small">
              {part.result.groups.length - 40} more groups in this part
            </li>
          )}
        </ul>
      )}
    </li>
  );
}

type GroupKey = string;

const keyOf = (group: { layer: string; type: string }): GroupKey =>
  `${group.layer}\u0000${group.type}`;

export function SelectionPanel({
  region,
  parts,
  combined,
  onRemovePart,
  result,
  loading,
  error,
  onModeChange,
  onClear,
  onSelectHandle,
  onAskAgent,
  supported,
  fetchGroupRows,
}: Props) {
  const [open, setOpen] = useState<Set<GroupKey>>(new Set());
  const [rows, setRows] = useState<
    Record<GroupKey, { handle: string; type: string; layer: string; text: string | null }[]>
  >({});
  const [rowState, setRowState] = useState<Record<GroupKey, "loading" | "error" | "done">>(
    {},
  );
  const [showAllGroups, setShowAllGroups] = useState(false);

  /** How many of the returned handles the viewer could actually mark.
   *
   *  The headline count and the purple highlight can legitimately differ, and
   *  saying nothing about it invites the reader to trust whichever they
   *  happened to look at. */
  const [highlighted, setHighlighted] = useState<number | null>(null);

  // A new region invalidates every expanded group: the same (layer, type)
  // pair now describes a different set of objects.
  useEffect(() => {
    setOpen(new Set());
    setRows({});
    setRowState({});
    setShowAllGroups(false);
    if (!result) {
      setHighlighted(null);
      return;
    }
    // One frame later: the Viewer writes the attributes in its own effect.
    const timer = window.setTimeout(
      () =>
        setHighlighted(
          document.querySelectorAll(".viewer-canvas [data-in-region='true']")
            .length,
        ),
      0,
    );
    return () => window.clearTimeout(timer);
  }, [result]);

  const toggleGroup = useCallback(
    async (group: RegionGroup) => {
      if (!region) return;
      const key = keyOf(group);
      setOpen((prev) => {
        const next = new Set(prev);
        if (next.has(key)) next.delete(key);
        else next.add(key);
        return next;
      });
      if (rows[key] || rowState[key] === "loading") return;

      setRowState((s) => ({ ...s, [key]: "loading" }));
      try {
        const entities = await fetchGroupRows(group);
        setRows((r) => ({ ...r, [key]: entities }));
        setRowState((s) => ({ ...s, [key]: "done" }));
      } catch {
        setRowState((s) => ({ ...s, [key]: "error" }));
      }
    },
    [region, rows, rowState, fetchGroupRows],
  );

  if (!supported) {
    return (
      <div className="selection-panel">
        <p className="muted">
          Area selection is not available for this render: it was produced
          before the viewer started recording the drawing-to-screen transform.
          Re-ingest the drawing to enable it.
        </p>
      </div>
    );
  }

  // Keyed on the RESULT, not on a drawn shape. A click selects one object and
  // produces a result with no region attached, and the panel used to answer
  // "Nothing selected" while its own tab counted (1).
  if (!result && !loading) {
    return (
      <div className="selection-panel">
        <p className="muted">
          Nothing selected. Hold <kbd>Shift</kbd> and drag a box over the
          drawing, or use <strong>⬠ Polygon</strong> in the toolbar for an
          irregular area.
        </p>
        <ul className="bar-list selection-legend">
          <li>
            <span><span className="swatch window" /> left → right</span>
            <span className="muted">window — only what is wholly inside</span>
          </li>
          <li>
            <span><span className="swatch crossing" /> right → left</span>
            <span className="muted">crossing — anything it touches</span>
          </li>
        </ul>
        <p className="muted small">
          The region is stored in drawing coordinates, so it stays on the same
          part of the drawing however far you zoom or pan.
        </p>
      </div>
    );
  }

  if (loading && !result) {
    return <p className="muted">Selecting…</p>;
  }

  if (error) {
    return (
      <div className="selection-panel">
        <p className="banner error">{error}</p>
        <button onClick={onClear}>Clear region</button>
      </div>
    );
  }

  if (!result) return null;

  const groups = showAllGroups ? result.groups : result.groups.slice(0, 12);
  const browsable = result.total <= BROWSE_THRESHOLD || showAllGroups;

  return (
    <div className="selection-panel">
      <div className="selection-headline">
        {result.kind === "click" && (
          <span className="selection-source">clicked · </span>
        )}
        <strong>{result.total.toLocaleString()}</strong> object
        {result.total === 1 ? "" : "s"}
        {parts.length > 1 ? ` in ${parts.length} parts` : ""} ·{" "}
        {result.by_layer.length.toLocaleString()} layer
        {result.by_layer.length === 1 ? "" : "s"} ·{" "}
        {result.by_type.length.toLocaleString()} type
        {result.by_type.length === 1 ? "" : "s"}
      </div>

      <div className="selection-actions">
        {result.kind !== "click" && (
        <div className="mode-switch" role="group" aria-label="Selection mode">
          <button
            className={result.mode === "window" ? "active" : ""}
            onClick={() => onModeChange("window")}
            title={
              parts.length > 1
                ? "Only objects wholly inside the region (AutoCAD: window) — applies to the last region drawn"
                : "Only objects wholly inside the region (AutoCAD: window)"
            }
          >
            Window
          </button>
          <button
            className={result.mode === "crossing" ? "active" : ""}
            onClick={() => onModeChange("crossing")}
            title={
              parts.length > 1
                ? "Every object the region touches (AutoCAD: crossing) — applies to the last region drawn"
                : "Every object the region touches (AutoCAD: crossing)"
            }
          >
            Crossing
          </button>
        </div>
        )}
        <button
          onClick={onClear}
          title={
            parts.length > 1
              ? "Remove every part of the selection (Esc)"
              : "Remove the region (Esc)"
          }
        >
          {parts.length > 1 ? "Clear all" : "Clear"}
        </button>
      </div>

      {parts.length > 1 && (
        <>
          <div className="selection-parts-head">
            {parts.length} parts
            {combined && combined.overlap !== null && combined.overlap > 0 && (
              <span className="muted">
                {" "}
                · {combined.overlap.toLocaleString()} object
                {combined.overlap === 1 ? "" : "s"} in more than one, counted
                once
              </span>
            )}
          </div>
          <ul className="selection-parts">
            {parts.map((part) => (
              <PartBlock
                key={part.id}
                part={part}
                onRemove={() => onRemovePart(part.id)}
                onSelectHandle={onSelectHandle}
                fetchGroupRows={fetchGroupRows}
              />
            ))}
          </ul>
          {combined?.incomplete && (
            <p className="muted small">
              At least one part matched more than 5,000 objects and returned a
              capped list, so the total above is a floor rather than the count.
              Draw smaller regions to get an exact figure.
            </p>
          )}
        </>
      )}

      <button
        className="ask-agent"
        onClick={onAskAgent}
        disabled={result.total === 0}
        title="Send these handles to the agent as context — the entities themselves are never sent"
      >
        Ask agent about selection
      </button>

      {result.total === 0 && (
        <p className="muted small">
          Nothing matched. In <strong>window</strong> mode an object has to sit
          entirely inside the region — try <strong>crossing</strong>.
        </p>
      )}

      {result.handles_truncated && (
        <p className="muted small">
          Highlighting the first {result.handles.length.toLocaleString()} of{" "}
          {result.total.toLocaleString()}. The counts below are exact.
        </p>
      )}
      {result.total > 750 && (
        <p className="muted small">
          Too many to mark individually — the outline shows the region, and a
          window selection contains nothing drawn outside it.
        </p>
      )}
      {highlighted !== null && highlighted > 0 && highlighted < result.handles.length && (
        <p className="muted small">
          {(result.handles.length - highlighted).toLocaleString()} of the
          matched objects are not drawn on this layout — they exist in the
          drawing but produce no geometry here, so they are counted and not
          highlighted.
        </p>
      )}

      {result.total > 0 && (
        <>
          <h4>By layer</h4>
          <ul className="bar-list">
            {result.by_layer.slice(0, 12).map((row) => (
              <li key={row.name}>
                <span>{row.name}</span>
                <span className="muted">{row.count.toLocaleString()}</span>
              </li>
            ))}
          </ul>
          {result.by_layer.length > 12 && (
            <p className="muted small">
              + {result.by_layer.length - 12} more layers
            </p>
          )}

          <h4>By type</h4>
          <ul className="bar-list">
            {result.by_type.slice(0, 12).map((row) => (
              <li key={row.name}>
                <span>{row.name}</span>
                <span className="muted">{row.count.toLocaleString()}</span>
              </li>
            ))}
          </ul>

          <h4>Groups</h4>
          {!browsable && (
            <p className="muted small">
              {result.total.toLocaleString()} objects is a lot to list. Open a
              group below to load just that one, or narrow the region.
            </p>
          )}
          <ul className="group-list">
            {groups.map((group) => {
              const key = keyOf(group);
              const isOpen = open.has(key);
              return (
                <li key={key}>
                  <button
                    className="group-header"
                    onClick={() => void toggleGroup(group)}
                    aria-expanded={isOpen}
                  >
                    <span className="group-caret">{isOpen ? "▾" : "▸"}</span>
                    <span className="group-name">
                      {group.layer} · {group.type}
                    </span>
                    <span className="muted">{group.count.toLocaleString()}</span>
                  </button>
                  {isOpen && (
                    <div className="group-body">
                      {rowState[key] === "loading" && (
                        <p className="muted small">Loading…</p>
                      )}
                      {rowState[key] === "error" && (
                        <p className="muted small">Could not load this group.</p>
                      )}
                      {(rows[key] ?? []).map((row) => (
                        <button
                          key={row.handle}
                          className="handle-chip"
                          onClick={() => onSelectHandle(row.handle)}
                          title="Select this object in the viewer"
                        >
                          {row.handle}
                          {row.text ? ` · ${row.text}` : ""}
                        </button>
                      ))}
                      {rowState[key] === "done" &&
                        (rows[key] ?? []).length < group.count && (
                          <p className="muted small">
                            showing {(rows[key] ?? []).length} of{" "}
                            {group.count.toLocaleString()}
                          </p>
                        )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
          {result.groups.length > groups.length && (
            <button
              className="link-button"
              onClick={() => setShowAllGroups(true)}
            >
              Show all {result.groups.length} groups
            </button>
          )}
        </>
      )}

      <p className="muted small selection-basis">
        {result.basis === "clicked"
          ? "One object, picked directly in the drawing. The agent is scoped to it until you clear the selection."
          : result.basis === "rendered-geometry"
          ? "This layout shows another space through viewports, so matching " +
            "compares what is actually drawn here — the region selects " +
            "exactly what you see, and an entity visible in two viewports " +
            "can be picked at either."
          : result.basis === "rendered-outlines"
          ? "Matching compares the outlines the map draws, not stored " +
            "bounding boxes, so this is stricter than a region drawn on the " +
            "sheet: a box that clips the empty corner of a diagonal plot " +
            "does not select it. What it cannot reach is anything with no " +
            "stored outline — block placements, arcs, circles, hatches — " +
            "which the map does not draw and this cannot select."
          : "Matching compares stored bounding boxes, not exact geometry — " +
            "a long diagonal line can be caught by a region that only clips " +
            "the empty corner of its box."}
        {result.without_bbox
          ? ` ${result.without_bbox} object(s) in range have no bounding box and cannot be selected.`
          : ""}
      </p>
    </div>
  );
}
