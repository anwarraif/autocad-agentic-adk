"use client";

/**
 * Layer visibility control, and the place where a typology gets a name.
 *
 * Lists the layers that actually appear in the rendered SVG, not every layer
 * declared in the file. One drawing here declares 292 layers but only 68
 * produce geometry in modelspace; offering the other 224 as toggles would be
 * a list of switches that do nothing.
 *
 * NAMING (UPLIFT-13). The request was *"can we click and give an appropriate
 * business name to each type, so that we do the tagging?"*, and the follow-up
 * pinned the unit: *"if we do it for one, it should reflect for all the
 * plots."* So the target is the LAYER, not the entity — which is also the
 * unit the drawing itself uses, since the typology is carried by layer names.
 * Per-entity tagging is deliberately absent: tens of thousands of entities
 * cannot be named by hand, and naming a fraction of them produces data that
 * is half true.
 *
 * Three things are on screen before anything is stored, and each is here
 * because of a specific way this could go wrong:
 *
 *   - HOW MANY OBJECTS the name will cover, split between real layouts and
 *     block definitions. "Name it once, it applies everywhere" is only safe
 *     if the person can see how big everywhere is.
 *   - WHERE THE CURRENT VALUE COMES FROM, and when a tag stands in front of a
 *     configured value, both values. An override that hides what it overrode
 *     makes "why did this answer change" unanswerable.
 *   - SOURCE, required, with no default and no checkbox. A tick box can be
 *     ticked without thinking; a sentence naming who decided and on what
 *     basis has to be typed. UPLIFT-08 removed `verified` for exactly this
 *     reason and the screen followed it.
 *
 * The rules live in `lib/layerTags.ts`. This file renders them.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  API_BASE,
  ApiError,
  fetchConfiguredLayerValues,
  getLayerImpact,
  listLayerTags,
  putLayerTag,
} from "@/lib/api";
import {
  EMPTY_DRAFT,
  MAX_BUSINESS_NAME,
  configLayerLabel,
  describeConfigValue,
  describeImpact,
  describeScopedShare,
  effectiveValue,
  formatStamp,
  indexConfigByLayer,
  indexTagsByLayer,
  problemFor,
  validateDraft,
  type LayerConfigValue,
  type LayerImpact,
  type LayerTag,
  type TagDraft,
} from "@/lib/layerTags";

/** Where the author's name is remembered between tags.
 *
 *  Remembered, never defaulted to a placeholder. `author` answers "whose hand
 *  was on the button", and filling it in with a generic account name would
 *  make every tag in the database look like it was saved by the same nobody.
 */
const AUTHOR_KEY = "cad.tag.author";

interface Props {
  /** Layers present in the SVG, with the number of objects each one holds in
   *  *this* render — counted from the markup, so the number always matches
   *  what the checkbox actually hides. */
  /** `count` is how many objects the DRAWING holds on this layer.
   *  `drawn` is how many the map currently holds, which is smaller
   *  whenever the view has fetched part of a large layout. They are
   *  different facts and are shown as different facts. */
  renderedLayers: { name: string; count: number; drawn?: number }[];
  hidden: Set<string>;
  onToggle: (layer: string) => void;
  onSetAll: (hidden: Set<string>) => void;
  /** The open drawing. Empty while the page is still booting. */
  drawingId: string;
  /** The layout the visible entities are STORED under — what "on screen"
   *  means for an impact count. Differs from the layout on screen when a
   *  paper sheet draws model space through a viewport. */
  scopeLayout: string;
  /** What this list is a list OF, said on screen.
   *
   *  In the map views the list is the drawing's world-placed geometry and NOT
   *  the sheet the layout picker names; in Normal 2D the two are the same
   *  thing. Without this the panel looked like it was ignoring the picker.
   *  `fromStore` is false while the scope's own list has not arrived, when
   *  the list is built from fetched geometry and may still grow. */
  scope?: { label: string; note: string; fromStore: boolean };
}

export function LayerPanel({
  renderedLayers,
  hidden,
  onToggle,
  onSetAll,
  drawingId,
  scopeLayout,
  scope,
}: Props) {
  const [filter, setFilter] = useState("");
  const [tags, setTags] = useState<Record<string, LayerTag>>({});
  const [config, setConfig] = useState<Record<string, LayerConfigValue>>({});
  /** Why there is no configured value to compare against, when there is none.
   *  Shown where the comparison would have been, never as an error: most
   *  drawings here carry no land-use configuration at all, and that is the
   *  ordinary state of the world rather than a fault. */
  const [configReason, setConfigReason] = useState<string | null>(null);
  const [tagsError, setTagsError] = useState<string | null>(null);
  /** The one layer whose editor is open. One at a time on purpose: two open
   *  forms make it easy to type a source into the wrong one. */
  const [editing, setEditing] = useState<string | null>(null);
  /** Set the first time an editor is opened, and never unset.
   *
   *  The configured values come from the land-use summary, which aggregates
   *  over every entity in the layout — cheap on a small drawing and not cheap
   *  on the big one. The panels on this page stay mounted whether or not
   *  anyone looks at them, so an eager fetch here would add that cost to
   *  every drawing load for a comparison nobody has asked to see. A10's
   *  measured defect on this page is stalling; this is one of the ways a page
   *  gets there. */
  const [configWanted, setConfigWanted] = useState(false);
  const [author, setAuthor] = useState("");

  useEffect(() => {
    try {
      setAuthor(window.localStorage.getItem(AUTHOR_KEY) ?? "");
    } catch {
      /* a browser with storage disabled simply asks for the name each time */
    }
  }, []);

  useEffect(() => {
    if (!drawingId) return;
    const controller = new AbortController();
    setTagsError(null);
    listLayerTags(drawingId, controller.signal)
      .then((body) => setTags(indexTagsByLayer(body.tags)))
      .catch((cause) => {
        if (controller.signal.aborted) return;
        setTags({});
        setTagsError(
          cause instanceof ApiError
            ? `${cause.message}${cause.hint ? ` ${cause.hint}` : ""}`
            : "Could not read the names already given for this drawing.",
        );
      });
    return () => controller.abort();
  }, [drawingId]);

  useEffect(() => {
    if (!configWanted || !drawingId || !scopeLayout) return;
    const controller = new AbortController();
    fetchConfiguredLayerValues(drawingId, scopeLayout, controller.signal).then(
      (result) => {
        if (controller.signal.aborted) return;
        setConfig(indexConfigByLayer(result.data));
        setConfigReason(result.reason);
      },
    );
    return () => controller.abort();
  }, [configWanted, drawingId, scopeLayout]);

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return renderedLayers;
    // The filter searches the business name as well as the layer name. Once a
    // layer has a name people use, that name is what they will type — being
    // forced back to the drawing's own code to find it would undo the point
    // of naming it.
    return renderedLayers.filter(
      (l) =>
        l.name.toLowerCase().includes(needle) ||
        (tags[l.name]?.business_name ?? "").toLowerCase().includes(needle),
    );
  }, [renderedLayers, filter, tags]);

  const onSaved = useCallback((tag: LayerTag) => {
    setTags((prev) => ({ ...prev, [tag.target]: tag }));
    try {
      window.localStorage.setItem(AUTHOR_KEY, tag.author);
    } catch {
      /* nothing to do; the field is simply typed again next time */
    }
    setAuthor(tag.author);
  }, []);

  if (renderedLayers.length === 0) {
    // An empty list still has a scope, and saying which one is the whole
    // point of this panel's header. "No layers in this render" left the
    // reader to guess whether the map found nothing or the sheet did.
    return (
      <div className="layer-panel">
        {scope && (
          <div
            className={`layer-scope${scope.fromStore ? "" : " provisional"}`}
            title={scope.note}
          >
            Nothing yet in {scope.label}
          </div>
        )}
        <p className="muted">
          {scope
            ? "No layers here yet. A sheet that has not finished rendering " +
              "reports none, and so does one that really holds none."
            : "No layers in this render."}
        </p>
      </div>
    );
  }

  const namedHere = renderedLayers.filter((l) => tags[l.name]).length;

  return (
    <div className="layer-panel">
      {scope && (
        <div
          className={`layer-scope${scope.fromStore ? "" : " provisional"}`}
          title={scope.note}
        >
          Showing {renderedLayers.length.toLocaleString()} {scope.label}
          {scope.fromStore ? null : <em> · still arriving</em>}
        </div>
      )}
      <div className="layer-actions">
        <button onClick={() => onSetAll(new Set())}>Show all</button>
        <button onClick={() => onSetAll(new Set(renderedLayers.map((l) => l.name)))}>
          Hide all
        </button>
      </div>
      {renderedLayers.length > 12 && (
        <input
          className="layer-filter"
          placeholder={`Filter ${renderedLayers.length} layers…`}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      )}
      {tagsError && <p className="tag-warning">{tagsError}</p>}
      <ul className="layer-list">
        {visible.map((layer) => {
          const value = effectiveValue(
            layer.name,
            tags[layer.name],
            config[layer.name],
          );
          const open = editing === layer.name;
          return (
            <li key={layer.name} className={open ? "layer-row open" : "layer-row"}>
              <div className="layer-line">
                <label
                  title={
                    layer.drawn !== undefined && layer.drawn < layer.count
                      ? `${layer.name} — ${layer.count.toLocaleString()} objects on this layout, ` +
                        `${layer.drawn.toLocaleString()} of them currently fetched into the map`
                      : `${layer.name} — ${layer.count.toLocaleString()} objects on this layout`
                  }
                >
                  <input
                    type="checkbox"
                    checked={!hidden.has(layer.name)}
                    onChange={() => onToggle(layer.name)}
                  />
                  <span className="layer-name">{layer.name}</span>
                  {/* The layout's count, from the drawing. When the map holds
                      fewer than that, the shortfall is shown beside it rather
                      than replacing it: a count of fetched features and a
                      count of the layer are different facts, and showing one
                      while looking like the other is how this project has
                      been misled before. */}
                  <span className="layer-count">
                    {layer.drawn !== undefined && layer.drawn < layer.count ? (
                      <>
                        <span className="layer-drawn">
                          {layer.drawn.toLocaleString()}
                        </span>
                        <span className="layer-of"> of </span>
                        {layer.count.toLocaleString()}
                      </>
                    ) : (
                      layer.count.toLocaleString()
                    )}
                  </span>
                </label>
                <button
                  className="layer-tag-button"
                  onClick={() => {
                    setConfigWanted(true);
                    setEditing(open ? null : layer.name);
                  }}
                  title={
                    value.businessName
                      ? "Change the business name for every object on this layer"
                      : "Give every object on this layer a business name"
                  }
                >
                  {open ? "Close" : value.origin === "user_tag" ? "Edit" : "Name"}
                </button>
              </div>

              {/* The name, and where it came from, on the row itself. A value
                  whose origin is only visible after a click is a value people
                  quote without knowing whether anyone signed it. */}
              {value.businessName && (
                <div className="layer-tagline">
                  <span className="layer-business-name">{value.businessName}</span>
                  <span className="muted">
                    {value.origin === "user_tag"
                      ? ` · named by ${value.author ?? "someone"}${
                          value.updatedAt ? `, ${formatStamp(value.updatedAt)}` : ""
                        }`
                      : ` · from ${configLayerLabel(value.configValue?.config_layer)}`}
                  </span>
                </div>
              )}

              {open && (
                <TagEditor
                  key={layer.name}
                  drawingId={drawingId}
                  layer={layer.name}
                  scopeLayout={scopeLayout}
                  tag={tags[layer.name] ?? null}
                  config={config[layer.name] ?? null}
                  configReason={configReason}
                  author={author}
                  onSaved={onSaved}
                  onClose={() => setEditing(null)}
                />
              )}
            </li>
          );
        })}
      </ul>
      <p className="muted small">
        {renderedLayers.length - hidden.size} of {renderedLayers.length} layers
        visible · {namedHere} named
      </p>
      {/* The last step of a flow that was built four fifths of the way: name
          layers on screen, store them, and then -- here -- get them out as a
          file a person can read, diff and commit. Without this the business
          decisions live only in a shared database with no review history, and
          the next drawing starts from nothing.

          A plain link rather than fetch-and-blob. The file is text the browser
          can already display, the export is deterministic so the same tags
          give the same bytes every time, and a download this page starts
          itself is the one thing a sandboxed viewer cannot deliver. */}
      {namedHere > 0 && (
        <p className="muted small">
          <a
            className="tag-export-link"
            href={`${API_BASE}/drawings/${encodeURIComponent(drawingId)}/tags/export.yaml`}
            target="_blank"
            rel="noreferrer"
            title="Open these tags as YAML shaped like a land-use config: read it, then commit it to the repo"
          >
            Export {namedHere} tag{namedHere === 1 ? "" : "s"} as config YAML
          </a>
        </p>
      )}
    </div>
  );
}

/** The form for one layer.
 *
 *  Mounted per layer with `key={layer}` so that opening a second one starts
 *  from an empty draft rather than inheriting a half-typed source from the
 *  first. Attributing one layer's justification to another is precisely the
 *  failure `source` exists to prevent.
 */
function TagEditor({
  drawingId,
  layer,
  scopeLayout,
  tag,
  config,
  configReason,
  author,
  onSaved,
  onClose,
}: {
  drawingId: string;
  layer: string;
  scopeLayout: string;
  tag: LayerTag | null;
  config: LayerConfigValue | null;
  configReason: string | null;
  author: string;
  onSaved: (tag: LayerTag) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<TagDraft>({
    ...EMPTY_DRAFT,
    businessName: tag?.business_name ?? "",
    use: tag?.use ?? "",
    // `source` is NOT carried over from the existing tag. Re-signing a name
    // means saying why again; pre-filling the justification would let a
    // changed name inherit the reasoning for the old one.
    author,
  });
  /** The whole drawing, because that is the scope a tag actually has. */
  const [whole, setWhole] = useState<LayerImpact | null>(null);
  /** The layout on screen, which answers "how many of those can I see". */
  const [scoped, setScoped] = useState<LayerImpact | null>(null);
  const [impactError, setImpactError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<{ message: string; hint: string } | null>(
    null,
  );
  const [saved, setSaved] = useState<string | null>(null);
  const [showProblems, setShowProblems] = useState(false);

  useEffect(() => {
    if (!drawingId) return;
    const controller = new AbortController();
    setImpactError(null);
    Promise.all([
      getLayerImpact(drawingId, layer, null, controller.signal),
      scopeLayout
        ? getLayerImpact(drawingId, layer, scopeLayout, controller.signal)
        : Promise.resolve(null),
    ])
      .then(([all, here]) => {
        if (controller.signal.aborted) return;
        setWhole(all);
        setScoped(here);
      })
      .catch((cause) => {
        if (controller.signal.aborted) return;
        setImpactError(
          cause instanceof ApiError
            ? `${cause.message}${cause.hint ? ` ${cause.hint}` : ""}`
            : "Could not count the objects this name would cover.",
        );
      });
    return () => controller.abort();
  }, [drawingId, layer, scopeLayout]);

  const problems = validateDraft(draft);
  const value = effectiveValue(layer, tag, config);
  const impactSentence = describeImpact(whole);
  const shareSentence = describeScopedShare(whole, scoped);
  const configSentence = describeConfigValue(config);

  const save = useCallback(async () => {
    if (validateDraft(draft).length > 0) {
      setShowProblems(true);
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      const stored = await putLayerTag(drawingId, layer, {
        business_name: draft.businessName.trim(),
        source: draft.source.trim(),
        author: draft.author.trim(),
        use: draft.use.trim() || null,
      });
      onSaved(stored);
      setSaved(
        whole && whole.entity_count > 0
          ? `Saved. ${whole.entity_count.toLocaleString()} object` +
              `${whole.entity_count === 1 ? "" : "s"} on this layer now carry ` +
              "this name, with no re-ingest."
          : "Saved.",
      );
      // The justification is cleared, and the complaint about it being empty
      // is not raised again until the next attempt. Re-signing a name means
      // saying why again — but a red field the instant a save succeeds reads
      // as a failure, and this one succeeded.
      setDraft((prev) => ({ ...prev, source: "" }));
      setShowProblems(false);
    } catch (cause) {
      setSaveError(
        cause instanceof ApiError
          ? { message: cause.message, hint: cause.hint }
          : {
              message: "The name was not saved.",
              hint: "Nothing was written. Check that cad-api is reachable and try again.",
            },
      );
    } finally {
      setSaving(false);
    }
  }, [draft, drawingId, layer, onSaved, whole]);

  const nameProblem = showProblems ? problemFor(problems, "business_name") : null;
  const sourceProblem = showProblems ? problemFor(problems, "source") : null;
  const authorProblem = showProblems ? problemFor(problems, "author") : null;

  return (
    <div className="tag-editor">
      {/* 1. What it will touch, before anything is stored. */}
      <div className="tag-impact">
        {impactError ? (
          <span className="tag-warning">{impactError}</span>
        ) : impactSentence ? (
          <>
            <div>{impactSentence}</div>
            {shareSentence && <div className="muted">{shareSentence}</div>}
          </>
        ) : (
          <span className="muted">Counting the objects this would cover…</span>
        )}
      </div>

      {/* 2. What the value is now, and what it would stand in front of. */}
      <div className="tag-current">
        {value.origin === "user_tag" && (
          <div>
            <span className="tag-label">Now</span> {value.businessName}
            <span className="muted">
              {" "}
              · named by {value.author ?? "someone"}
              {value.updatedAt ? `, ${formatStamp(value.updatedAt)}` : ""}
              {value.revision ? ` · revision ${value.revision}` : ""}
            </span>
            {value.sourceNote && (
              <div className="tag-source-note">{value.sourceNote}</div>
            )}
          </div>
        )}
        {value.replacedTag && (
          <div className="tag-previous">
            <span className="tag-label">Was</span>{" "}
            {value.replacedTag.business_name ?? "(no name)"}
            <span className="muted">
              {" "}
              · {value.replacedTag.author ?? "someone"}
              {value.replacedTag.updated_at
                ? `, ${formatStamp(value.replacedTag.updated_at)}`
                : ""}
            </span>
          </div>
        )}
        {configSentence ? (
          <div className={value.overridden ? "tag-overridden" : ""}>
            <span className="tag-label">
              {value.overridden ? "Overrides" : "Config"}
            </span>{" "}
            {configSentence}
          </div>
        ) : configReason ? (
          <div className="muted">
            No configured value to compare against — {configReason}
          </div>
        ) : (
          <div className="muted">
            Nothing configured for this layer, so this name is the only value
            it will have.
          </div>
        )}
      </div>

      {/* 3. The form. `source` is required and has no default. */}
      <label className="tag-field">
        <span>Business name</span>
        <input
          value={draft.businessName}
          maxLength={MAX_BUSINESS_NAME + 40}
          placeholder="What the people who use this drawing call this type"
          onChange={(e) =>
            setDraft((prev) => ({ ...prev, businessName: e.target.value }))
          }
        />
        {nameProblem && <span className="tag-problem">{nameProblem.message}</span>}
      </label>

      <label className="tag-field">
        <span>
          Land use <span className="muted">optional</span>
        </span>
        <input
          value={draft.use}
          placeholder="Only if this name is also a land use"
          onChange={(e) => setDraft((prev) => ({ ...prev, use: e.target.value }))}
        />
      </label>

      <label className="tag-field">
        <span>Source</span>
        <textarea
          rows={3}
          value={draft.source}
          placeholder="Who decided this, when, and on what basis — a meeting, a document, a typology key"
          onChange={(e) =>
            setDraft((prev) => ({ ...prev, source: e.target.value }))
          }
        />
        {sourceProblem ? (
          <span className="tag-problem">{sourceProblem.message}</span>
        ) : (
          <span className="muted small">
            Required. Whoever reads this tag was not in the room when you typed
            it.
          </span>
        )}
      </label>

      <label className="tag-field">
        <span>Your name</span>
        <input
          value={draft.author}
          placeholder="Who is saving this"
          onChange={(e) =>
            setDraft((prev) => ({ ...prev, author: e.target.value }))
          }
        />
        {authorProblem && (
          <span className="tag-problem">{authorProblem.message}</span>
        )}
      </label>

      {saveError && (
        <p className="tag-warning">
          {saveError.message}
          {saveError.hint ? ` ${saveError.hint}` : ""}
        </p>
      )}
      {saved && <p className="tag-saved">{saved}</p>}

      <div className="comment-actions">
        <button onClick={() => void save()} disabled={saving}>
          {saving ? "Saving…" : "Save name"}
        </button>
        <button className="link-button" onClick={onClose}>
          close
        </button>
        {/* The reasons it cannot be saved, listed rather than expressed as a
            dead button. A disabled control with no explanation is the same
            silence A15 is about, in a different panel. */}
        {problems.length > 0 && (
          <span className="muted small">
            {problems.length} field{problems.length === 1 ? "" : "s"} still
            needed
          </span>
        )}
      </div>
    </div>
  );
}
