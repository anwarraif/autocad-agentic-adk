/**
 * Naming a typology from the screen: the rules, without the panel.
 *
 * UPLIFT-13 exists because the drawing does not carry business names for its
 * own typologies — the codes in the layer table are the draughtsman's, and
 * 43,109 text entities were searched without one of them explaining what any
 * code means. The engine for fixing that is built and wired; what was missing
 * was a way to type a name in without editing a file.
 *
 * Everything here is arithmetic and sentences: no React, no DOM, no fetch.
 * That is not tidiness. There is no frontend test harness in this repo, so a
 * rule that lives inside a component is a rule nobody can check, and three of
 * the four rules below are ones a reviewer should be able to disagree with
 * line by line:
 *
 *   - a tag is refused before the round trip, in the same words the server
 *     would have used (`validateDraft`);
 *   - the impact of a name is stated as a count with its scope attached,
 *     never as a bare number (`describeImpact`);
 *   - a tag that overrides config shows BOTH values, and a tag that replaces
 *     an earlier tag shows both of those too (`effectiveValue`).
 *
 * On `verified`: there is no checkbox. UPLIFT-08 removed the field, and the
 * note at the foot of `docs/UPLIFT-13-BUSINESS-TAGGING.md` records why the
 * screen followed. A box can be ticked without thinking; a sentence saying
 * who decided and on what basis has to be typed. `source` is that sentence,
 * and it is required.
 */

// ---------------------------------------------------------------------------
// Limits, mirrored from the server on purpose
// ---------------------------------------------------------------------------
//
// `cad_api/app/store_recipes.py` holds the authority (MAX_BUSINESS_NAME,
// MAX_SOURCE) and refuses independently of anything typed here. These copies
// exist so the refusal arrives while the user is still looking at the field
// that caused it, rather than after a round trip. If the two ever disagree
// the server wins, and its message is displayed verbatim.

export const MAX_BUSINESS_NAME = 120;
export const MAX_SOURCE = 2000;

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/** What a previous version of a tag held. One snapshot, not a history: enough
 *  to answer "who changed it from what" without growing an audit log inside a
 *  shared database. */
export interface TagSnapshot {
  business_name: string | null;
  use: string | null;
  source: string | null;
  author: string | null;
  updated_at: string | null;
  revision: number | null;
}

/** A stored tag, as `GET /drawings/{id}/tags` returns it. */
export interface LayerTag {
  _id: string;
  drawing_id: string;
  target_kind: string;
  /** The layer this tag names. */
  target: string;
  business_name: string;
  use: string | null;
  /** Where the name came from, in words, typed by a person. Required. */
  source: string;
  author: string;
  created_at?: string | null;
  updated_at?: string | null;
  revision?: number | null;
  previous?: TagSnapshot | null;
  tag_contract_version?: number;
}

/** How many objects a name is about to apply to, before anything is saved.
 *
 *  The block split is not decoration. Geometry inside a block definition is
 *  rescaled by every insertion of that block, so folding it into one total
 *  would let a reader believe the whole number is drawn where they are
 *  looking.
 */
export interface LayerImpact {
  drawing_id: string;
  layer: string;
  layout: string | null;
  entity_count: number;
  by_layout: Record<string, number>;
  in_block_definitions: number;
  on_real_layouts: number;
  layouts_truncated: number;
  /** The server's own sentence. Shown alongside ours, never instead of the
   *  numbers — see `describeImpact`. */
  impact_note?: string | null;
  why_empty?: string | null;
}

/** The value a layer already carries from reviewed configuration, which a
 *  screen tag overrides.
 *
 *  `business_name` is nullable and usually null: the land-use configuration
 *  maps a layer to a USE, not to a business name. Saying so is the honest
 *  answer — the config had no name to be overridden, only a category.
 */
export interface LayerConfigValue {
  layer: string;
  business_name: string | null;
  use: string | null;
  /** Which configuration layer decided it: a per-drawing override, a global
   *  pattern, a default, or none. Rendered through `configLayerLabel`. */
  config_layer: string | null;
  note: string | null;
}

/** What the user is typing. Strings only — the panel owns no other state. */
export interface TagDraft {
  businessName: string;
  /** Optional: a business name is not always a land use. */
  use: string;
  source: string;
  author: string;
}

export const EMPTY_DRAFT: TagDraft = {
  businessName: "",
  use: "",
  source: "",
  author: "",
};

/** One reason a draft cannot be saved yet.
 *
 *  `code` matches the server's refusal codes so that a message shown before
 *  the request and a message returned by it are recognisably the same
 *  complaint rather than two different-sounding ones.
 */
export interface DraftProblem {
  field: "business_name" | "use" | "source" | "author";
  code: string;
  message: string;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/** Why a draft cannot be saved, in the order the fields appear.
 *
 *  An empty list means the server will accept it, as far as anything the
 *  browser can know. The list is never collapsed to a boolean in the UI:
 *  "Save is disabled" without a reason is the same silence A15 is about, in a
 *  different panel.
 */
export function validateDraft(draft: TagDraft): DraftProblem[] {
  const problems: DraftProblem[] = [];
  const name = draft.businessName.trim();
  const source = draft.source.trim();
  const author = draft.author.trim();

  if (!name) {
    problems.push({
      field: "business_name",
      code: "TAG_WITHOUT_NAME",
      message:
        "Write the business name people already use for this typology. It is " +
        "the whole content of the tag; without it nothing is stored.",
    });
  } else if (name.length > MAX_BUSINESS_NAME) {
    problems.push({
      field: "business_name",
      code: "TAG_NAME_TOO_LONG",
      message:
        `${name.length} characters, and the limit is ${MAX_BUSINESS_NAME}. A ` +
        "name the length of a paragraph is a note, and notes belong in Source.",
    });
  }

  if (!source) {
    problems.push({
      field: "source",
      code: "TAG_WITHOUT_SOURCE",
      message:
        "Say who decided this, when, and on what basis — a meeting, a " +
        "document, a typology key. Whoever reads this tag was not in the room " +
        "when you typed it, and without an origin it cannot be told apart " +
        "from a guess.",
    });
  } else if (source.length > MAX_SOURCE) {
    problems.push({
      field: "source",
      code: "TAG_SOURCE_TOO_LONG",
      message:
        `${source.length} characters, and the limit is ${MAX_SOURCE}. Condense ` +
        "it to a reference someone else can open.",
    });
  }

  if (!author) {
    problems.push({
      field: "author",
      code: "TAG_WITHOUT_AUTHOR",
      message:
        "Say who is saving this. Source explains the basis; author says whose " +
        "hand was on the button, and the two are not always the same person.",
    });
  }

  return problems;
}

/** The problem for one field, so a message can sit under the input that
 *  caused it instead of in a list at the bottom. */
export function problemFor(
  problems: DraftProblem[],
  field: DraftProblem["field"],
): DraftProblem | null {
  return problems.find((p) => p.field === field) ?? null;
}

// ---------------------------------------------------------------------------
// Impact
// ---------------------------------------------------------------------------

/** How many objects this name is about to cover, said in full.
 *
 *  This is the first thing UPLIFT-13 asks to be visible, and the reason is in
 *  the request itself: *"if we do it for one, it should reflect for all the
 *  plots."* Naming once and having it apply everywhere is only safe if the
 *  person doing it can see how big "everywhere" is before they commit.
 *
 *  Built from the numbers rather than passed through from the server's
 *  sentence, so that every clause here can be checked against a field. The
 *  server's own note is shown next to it, not in place of it.
 */
export function describeImpact(impact: LayerImpact | null): string | null {
  if (!impact) return null;
  const scope = impact.layout
    ? ` on layout ${impact.layout}`
    : " across this whole drawing, block definitions included";

  if (impact.entity_count === 0) {
    return (
      `No objects carry this layer${scope}, so naming it changes nothing you ` +
      "can see. That is an answer, not a failure: a layer can be declared in " +
      "the layer table and hold nothing."
    );
  }

  const parts: string[] = [
    `${impact.entity_count.toLocaleString()} object` +
      `${impact.entity_count === 1 ? "" : "s"} will carry this name${scope}.`,
  ];

  if (impact.in_block_definitions > 0) {
    parts.push(
      `${impact.in_block_definitions.toLocaleString()} of them sit inside ` +
        "block definitions, so how many are actually drawn depends on how " +
        `often those blocks are inserted — ` +
        `${impact.on_real_layouts.toLocaleString()} are on real layouts.`,
    );
  }
  if (impact.layouts_truncated > 0) {
    parts.push(
      `${impact.layouts_truncated} further layout` +
        `${impact.layouts_truncated === 1 ? " is" : "s are"} not broken out ` +
        "here; the total above still counts them.",
    );
  }
  return parts.join(" ");
}

/** How much of a drawing-wide impact is visible on the layout in front of you.
 *
 *  A tag is a fact about the whole drawing, so the headline count has to be
 *  the whole drawing's. But a person looking at one sheet will read that
 *  number as "what I can see", and on a layout that draws none of them the
 *  gap between the two is total. Both numbers, and the relationship between
 *  them, or neither.
 */
export function describeScopedShare(
  whole: LayerImpact | null,
  scoped: LayerImpact | null,
): string | null {
  if (!whole || !scoped || !scoped.layout) return null;
  if (whole.entity_count === 0) return null;
  if (scoped.entity_count >= whole.entity_count) {
    return `All of them are on ${scoped.layout}, the layout you are looking at.`;
  }
  if (scoped.entity_count === 0) {
    return (
      `None of them are on ${scoped.layout}, the layout you are looking at — ` +
      "the name is still stored for the whole drawing, it simply changes " +
      "nothing on this screen."
    );
  }
  return (
    `${scoped.entity_count.toLocaleString()} of them are on ${scoped.layout}, ` +
    "the layout you are looking at; the rest are elsewhere in the drawing."
  );
}

// ---------------------------------------------------------------------------
// Effective value: config overridden by tag, with both still visible
// ---------------------------------------------------------------------------

export type ValueOrigin = "user_tag" | "config" | "none";

/** What a layer is called right now, and everything that was true before.
 *
 *  The third thing UPLIFT-13 asks to be visible. An override that hides what
 *  it overrode makes "why did this answer change" unanswerable by anybody,
 *  so both survive: `configValue` is what the reviewed configuration says,
 *  `replacedTag` is the tag revision this one displaced.
 */
export interface EffectiveValue {
  layer: string;
  businessName: string | null;
  use: string | null;
  origin: ValueOrigin;
  /** True when a screen tag is standing in front of a configured value. */
  overridden: boolean;
  configValue: LayerConfigValue | null;
  replacedTag: TagSnapshot | null;
  author: string | null;
  updatedAt: string | null;
  /** The person's own sentence about where the name came from. */
  sourceNote: string | null;
  revision: number | null;
}

export function effectiveValue(
  layer: string,
  tag: LayerTag | null | undefined,
  config: LayerConfigValue | null | undefined,
): EffectiveValue {
  const hasConfigValue = Boolean(config && (config.business_name || config.use));
  return {
    layer,
    businessName: tag?.business_name ?? config?.business_name ?? null,
    use: (tag?.use || null) ?? config?.use ?? null,
    origin: tag ? "user_tag" : hasConfigValue ? "config" : "none",
    overridden: Boolean(tag) && hasConfigValue,
    configValue: config ?? null,
    // A snapshot with nothing in it is not a predecessor. The first revision
    // of a tag carries `previous: null`, and an object full of nulls would
    // render as "replaced: (nothing)" — a change that never happened.
    replacedTag:
      tag?.previous && (tag.previous.business_name || tag.previous.use)
        ? tag.previous
        : null,
    author: tag?.author ?? null,
    updatedAt: tag?.updated_at ?? null,
    sourceNote: tag?.source ?? null,
    revision: tag?.revision ?? null,
  };
}

/** `drawing_override` -> "a per-drawing override". Unknown values are shown
 *  as they arrived rather than swallowed: a config layer this build has not
 *  heard of is information, and hiding it would be the wrong kind of tidy. */
export function configLayerLabel(value: string | null | undefined): string {
  switch (value) {
    case "drawing_override":
      return "a per-drawing override";
    case "global_pattern":
      return "a global naming pattern";
    case "default_use":
      return "a configured default";
    case "no_config":
      return "no configuration";
    default:
      return value ? value : "an unnamed configuration layer";
  }
}

/** One line describing what a tag would override, or `null` when there is
 *  nothing configured to override. */
export function describeConfigValue(
  config: LayerConfigValue | null,
): string | null {
  if (!config) return null;
  const value = config.business_name ?? config.use;
  if (!value) return null;
  const kind = config.business_name ? "name" : "land use";
  return `Configured ${kind}: ${value} — from ${configLayerLabel(config.config_layer)}.`;
}

// ---------------------------------------------------------------------------
// Reading configured values out of the land-use summary
// ---------------------------------------------------------------------------
//
// UPLIFT-13's own plan (PROGRESS.md, PERMINTAAN 2) asks for a route that
// returns the effective value for one layer with `config_value` already
// folded in. That route is not merged, and the honest thing while it is not
// is to derive the configured side from a route that IS merged rather than to
// show a blank where a comparison belongs. `GET /drawings/{id}/land-use`
// carries exactly the configured classification per layer.
//
// When the effective-value route lands, one function changes — the index
// below is replaced by the route's `config_value` — and no component moves.

/** The slice of the land-use summary this file reads. Deliberately partial:
 *  the summary carries evidence, areas and open questions that a naming panel
 *  has no business rendering. */
export interface LandUseSummaryLite {
  uses?: {
    use?: string;
    by_layer?: {
      layer?: string;
      land_use?: string | null;
      land_use_subtype?: string | null;
      config_layer?: string | null;
      note?: string | null;
    }[];
  }[];
  roles_not_counted?: {
    layer?: string;
    land_use?: string | null;
    config_layer?: string | null;
    note?: string | null;
  }[];
  config_file?: string | null;
}

/** `layer -> configured value`, from a land-use summary.
 *
 *  Both `uses[].by_layer` and `roles_not_counted` are read. A layer whose
 *  role is not "parcel" is still a layer somebody may want to name, and
 *  leaving it out would make the panel claim it has no configured value when
 *  it plainly does.
 */
export function indexConfigByLayer(
  summary: LandUseSummaryLite | null,
): Record<string, LayerConfigValue> {
  const index: Record<string, LayerConfigValue> = {};
  if (!summary) return index;

  const add = (row: {
    layer?: string;
    land_use?: string | null;
    land_use_subtype?: string | null;
    config_layer?: string | null;
    note?: string | null;
  }) => {
    const layer = (row.layer ?? "").trim();
    if (!layer || index[layer]) return;
    const use = row.land_use ?? null;
    index[layer] = {
      layer,
      // The land-use configuration maps layers to uses, never to business
      // names — which is the gap UPLIFT-13 exists to fill. Reporting null
      // here says exactly that: there is a category, and there is no name.
      business_name: null,
      use: row.land_use_subtype ? `${use} · ${row.land_use_subtype}` : use,
      config_layer: row.config_layer ?? null,
      note: row.note ?? null,
    };
  };

  for (const useRow of summary.uses ?? []) {
    for (const layerRow of useRow.by_layer ?? []) add(layerRow);
  }
  for (const layerRow of summary.roles_not_counted ?? []) add(layerRow);
  return index;
}

/** `layer -> tag`, from the tag list. */
export function indexTagsByLayer(
  tags: LayerTag[] | null,
): Record<string, LayerTag> {
  const index: Record<string, LayerTag> = {};
  for (const tag of tags ?? []) {
    if (tag.target_kind === "layer" && tag.target) index[tag.target] = tag;
  }
  return index;
}

/** A stored timestamp as a short local date, or the raw string when it is not
 *  a date this browser can parse. Never a blank: a tag whose date cannot be
 *  read still has one, and showing nothing would imply otherwise. */
export function formatStamp(value: string | null | undefined): string {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
