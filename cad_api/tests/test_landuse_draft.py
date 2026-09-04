"""`app/landuse_draft.py` — the drafted land-use config, DOSSIER Phase 3.

Two parts, following the habit of `test_dossier_read.py`.

The first part is **fixtures only**. The drafting core is pure over its inputs
— a layer table, a Dossier profile per layer, and the global patterns — so
every rule below is alive on a laptop with no database anywhere near it.

The second part reads the **real store** and skips while naming its reason if
`MONGODB_URI` is not set. It runs the same invariant over EVERY drawing in the
store rather than over the comfortable one (G5): a draft that is loadable for
the reference drawing and explodes on the eighteenth is not a draft, it is a
demo. `pytest -rs` names what was skipped; read that before trusting a green
run.

The tests that matter most here are the ones written to catch a promotion
rather than to confirm a success:

* `test_a_pattern_hit_is_never_live_in_the_drafted_file`. `verified` is
  DERIVED from the config layer: writing a pattern hit into
  `landuse/<drawing_id>.yaml` does not record it, it PROMOTES it to
  `verified: true` with no human involved. The draft's `layers:` is therefore
  empty as written, and this test fails the moment anything appears there.
* `test_accepting_every_ready_proposal_loads_through_the_real_loader`. A draft
  the loader rejects is worse than no draft.
* `test_accept_never_sweeps_up_a_template`. Two markers exist precisely so
  that an entry missing a human decision cannot be accepted in one mechanical
  pass. If `accept()` ever uncommented a `#? ` line, the role of an ambiguous
  layer would silently become `parcel`.
* `test_a_layer_no_pattern_matches_gets_no_use`. Silence and invention are
  both wrong; the third answer — the profile plus a stated gap — is what this
  phase exists to produce.

Drawing-specific strings appear below only as fixtures, which rule G1 exempts.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import evidence as ev  # noqa: E402
from app import landuse  # noqa: E402
from app import landuse_draft as ld  # noqa: E402


# --- fixture builders ---------------------------------------------------------


def profile(
    role: str | None,
    *,
    layer: str = "L",
    layout: str = "Model",
    entities: int = 10,
    kind: str = "count",
    value=None,
    unit: str | None = None,
    buckets: int = 1,
    anomaly_kinds=(),
) -> dict:
    """A `dossier_read.profiles_for` row, verbose form.

    `unit=None` is a first-class state and the default, because 11 of the
    drawings in this store are in inches and 3 declare nothing at all (G2).
    """
    return {
        "layer": layer,
        "layout": layout,
        "entities": entities,
        "role": role,
        "profiled": True,
        "role_leader": {
            "family": role,
            "share": 0.9,
            "runner_up": "other",
            "runner_up_share": 0.1,
        },
        "measure": {
            "kind": kind,
            "value": entities if value is None else value,
            "unit": unit,
            "unit_reason": None if unit else "this bucket declares no unit",
            "measured_entities": entities,
            "unmeasured_entities": 0,
        },
        "anomalies": {"kinds": list(anomaly_kinds), "flags": []},
        "scope": {
            "buckets_for_this_layer": buckets,
            "entities_all_buckets": entities,
        },
    }


def drawing(layers, *, drawing_id: str = "aaaabbbbccccdddd", name="demo.dxf") -> dict:
    return {
        "_id": drawing_id,
        "original_filename": name,
        "layers": [
            {"name": n, "entity_count": c} for n, c in layers
        ],
    }


#: One drawing exercising every branch at once: a name a pattern reads, a
#: geometry with no name to go with it, an ambiguous geometry, a name whose
#: geometry contradicts it, an empty declared layer, and a layer nothing at all
#: is known about.
SPREAD = drawing(
    [
        ("Primary School", 3),
        ("00_Prop - Centreline", 120),
        ("TREE-SYMBOLS", 40),
        ("PARK-LABELS", 9),
        ("Car Parking Area", 0),
        ("XX-99", 0),
    ]
)

SPREAD_PROFILES = {
    "Primary School": profile("region", entities=3, kind="area", value=1234.5, unit="m2"),
    "00_Prop - Centreline": profile(
        "network", entities=120, kind="length", value=9876.5, unit="m"
    ),
    "TREE-SYMBOLS": profile("points", entities=40),
    "PARK-LABELS": profile("annotation", entities=9),
}


def spread_draft(**kwargs) -> dict:
    return ld.build_draft(SPREAD, profiles=SPREAD_PROFILES, **kwargs)


def entry_of(draft: dict, layer: str) -> dict:
    for row in draft["entries"]:
        if row["layer"] == layer:
            return row
    raise AssertionError(f"{layer!r} is not in the draft at all")


def load_as_config(text: str, tmp_path, drawing_id: str, monkeypatch):
    """Load `text` through the REAL loader, from a directory that is not any
    config directory of the running service.

    Isolation is done at the OWNER: `_config_search_dirs` is the one function
    that lists where configs live, so overriding it to return only the
    temporary directory routes both `config_path` and `configured_drawing_ids`
    there. Patching `CONFIG_DIR` alone was not enough once the loader gained a
    writable overlay ahead of it -- a real accepted config in that overlay
    then shadowed the drawing under test, and a drawing with proposals looked
    like one with none. Overriding the list itself cannot desynchronise the
    same way a subset of it can.

    The override is put back before returning, not merely at the end of the
    test: a loop over every drawing in the store would otherwise have the
    second iteration reading the first one's temporary directory.
    """
    original = landuse._config_search_dirs
    config_dir = tmp_path / "landuse_under_test"
    config_dir.mkdir(exist_ok=True)
    (config_dir / f"{drawing_id}.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setattr(landuse, "_config_search_dirs", lambda: (config_dir,))
    landuse.clear_cache()
    try:
        return landuse.for_drawing(drawing_id)
    finally:
        # Restore this one attribute explicitly, not monkeypatch.undo(), which
        # would revert every patch the calling test has applied.
        monkeypatch.setattr(landuse, "_config_search_dirs", original)
        landuse.clear_cache()


# --- 1. Nothing in a draft is ever a fact ------------------------------------


def test_a_pattern_hit_is_never_live_in_the_drafted_file():
    """`layers:` is empty as written, and that is the whole design.

    `evidence.Provenance.verified` is true for -- and only for -- the
    `drawing_override` layer. So a pattern hit written into
    `landuse/<drawing_id>.yaml` is not recorded, it is PROMOTED: "a pattern
    proposed this" becomes "a human confirmed this" with nobody involved. The
    proposals therefore stay commented until a person uncomments them.
    """
    text = ld.render_yaml(spread_draft())
    raw = yaml.safe_load(text)

    assert not raw.get("layers"), (
        "the drafted file classifies a layer as written; a proposal must not "
        "arrive as a drawing_override, which is what `verified: true` is "
        "derived from"
    )
    assert raw["default_use"] == landuse.UNKNOWN_USE
    assert raw["drawing_id"] == SPREAD["_id"]


def test_the_drafted_file_never_writes_a_retired_key():
    """`verified:` and `source:` are refused by the loader by name (D-082)."""
    text = ld.render_yaml(spread_draft())
    accepted = yaml.safe_load(ld.accept(text)) or {}
    for name, entry in (accepted.get("layers") or {}).items():
        for key in landuse.RETIRED_KEYS:
            assert key not in entry, f"{name} carries the retired key {key!r}"


def test_a_pattern_hit_is_served_verified_false_while_it_stays_commented(
    tmp_path, monkeypatch
):
    """The draft in place: the layer IS classified, by the pattern layer.

    This is the payoff of leaving `layers:` empty. The drawing gains partial
    coverage the day it arrives (G4) and not one entry claims to be verified.
    """
    pattern_set = landuse.patterns()
    text = ld.render_yaml(spread_draft(pattern_set=pattern_set))
    config = load_as_config(text, tmp_path, SPREAD["_id"], monkeypatch)

    assert config is not None and not config.layers

    hit = landuse.classify(
        SPREAD["_id"], "Primary School", config=config, pattern_set=pattern_set
    )
    assert hit is not None
    assert hit.use == "education"
    assert hit.config_layer is ev.ConfigLayer.GLOBAL_PATTERN
    assert hit.provenance().verified is False


# --- 2. The draft loads, as written and as accepted --------------------------


def test_accepting_every_ready_proposal_loads_through_the_real_loader(
    tmp_path, monkeypatch
):
    """A draft the loader rejects is worse than no draft (hard rule 6).

    `accept()` is mechanical: it deletes the accept marker and changes nothing
    else. What comes out has to be a config the running loader reads, with
    every role in `ROLES`, every use in `USES`, and sources on every entry.
    """
    draft = spread_draft()
    accepted = ld.accept(ld.render_yaml(draft))
    config = load_as_config(accepted, tmp_path, draft["drawing_id"], monkeypatch)

    assert config is not None
    ready = {e["layer"] for e in draft["entries"] if e["ready"]}
    assert set(config.layers) == ready

    for name, land_use in config.layers.items():
        assert land_use.use in landuse.USES
        assert land_use.role in landuse.ROLES
        assert land_use.sources, f"{name} was written without a source"
        assert land_use.config_layer is ev.ConfigLayer.DRAWING_OVERRIDE
        assert land_use.provenance().verified is True, (
            "an accepted entry IS verified -- that is what the human's "
            "uncommenting signed for"
        )


def test_accept_never_sweeps_up_a_template():
    """Two markers, and this is why there are two.

    `TREE-SYMBOLS` profiles as `points`, which names no config role. Accepting
    it mechanically would let `landuse.DEFAULT_ROLE` -- `parcel` -- decide, and
    the layer would silently become plots nobody counted.
    """
    draft = spread_draft()
    assert entry_of(draft, "TREE-SYMBOLS")["ready"] is False

    accepted = yaml.safe_load(ld.accept(ld.render_yaml(draft))) or {}
    assert "TREE-SYMBOLS" not in (accepted.get("layers") or {})
    assert landuse.DEFAULT_ROLE == landuse.PARCEL_ROLE, (
        "this test exists because a missing role means `parcel`; if that ever "
        "changes, the reason for the second marker changes with it"
    )


def test_every_accepted_entry_carries_a_role_or_has_no_geometry():
    """Nothing is accepted while its role is the loader's default by accident."""
    draft = spread_draft()
    for row in draft["entries"]:
        if not row["ready"]:
            continue
        assert row["role"] or row["declared_but_empty"], (
            f"{row['layer']} is offered as ready with no role and real "
            "geometry; accepting it would sign for `parcel` unread"
        )


def test_layer_names_that_would_break_a_hand_written_yaml(tmp_path, monkeypatch):
    """Layer names are not ours to sanitise; the renderer must survive them."""
    awkward = [
        ('A "quoted" name', 4),
        ("colon: inside", 4),
        ("hash # inside", 4),
        ("0", 4),
        ("yes", 4),
        ("مرفق تعليمي", 4),
        ("back\\slash", 4),
    ]
    doc = drawing(awkward, drawing_id="0123456789abcdef")
    profiles = {name: profile("region", entities=n) for name, n in awkward}
    draft = ld.build_draft(doc, profiles=profiles)

    config = load_as_config(
        ld.accept(ld.render_yaml(draft)), tmp_path, doc["_id"], monkeypatch
    )
    assert set(config.layers) == {name for name, _ in awkward}


# --- 3. No use without attribution -------------------------------------------


def test_a_layer_no_pattern_matches_gets_no_use():
    """No guess, and no silence either: the profile plus a stated gap."""
    row = entry_of(spread_draft(), "00_Prop - Centreline")

    assert row["use"] is None
    assert row["matched_pattern"] is None
    assert row["status"] == ld.STATUS_GEOMETRY_ONLY
    assert row["open_questions"], "a missing use must be named, not left blank"

    written = row["entry"]
    assert written["use"] == landuse.UNKNOWN_USE
    assert written["unknown"] and written["how_to_verify"]
    origins = [s["origin"] for s in written["sources"]]
    assert "absence" in origins, "the missing pattern hit is itself a source"
    assert "geometry" in origins


def test_a_use_is_only_ever_the_pattern_that_proposed_it():
    """`use` comes from the pattern set and from nowhere else."""
    pattern_set = landuse.patterns()
    row = entry_of(spread_draft(pattern_set=pattern_set), "Primary School")
    rule = pattern_set.first_match("Primary School")

    assert rule is not None
    assert row["use"] == rule.use
    assert row["subtype"] == rule.subtype
    assert row["matched_pattern"] == rule.match
    assert rule.match in row["entry"]["sources"][0]["detail"]
    assert row["entry"]["sources"][0]["observed"] == "Primary School"


def test_no_pattern_no_dossier_is_a_name_and_nothing_else():
    row = entry_of(spread_draft(), "XX-99")
    assert row["status"] == ld.STATUS_NAME_ONLY
    assert row["use"] is None and row["role"] is None
    assert row["ready"] is False, (
        "an entry that claims nothing must not be offered for acceptance; it "
        "would add a line saying only that the layer exists"
    )


# --- 4. Role comes from geometry, and ambiguity is reported not resolved ------


@pytest.mark.parametrize(
    "geometric, expected",
    [
        ("region", "parcel"),
        ("network", "network"),
        ("annotation", "annotation"),
        ("points", None),
        ("mixed", None),
        (None, None),
        ("something nobody has met yet", None),
    ],
)
def test_the_geometry_to_role_mapping_is_the_documented_one(geometric, expected):
    role, basis = ld.role_from_geometry(geometric)
    assert role == expected
    assert basis, "a role that maps to nothing must still say why"
    if role is not None:
        assert role in landuse.ROLES


def test_every_mapped_role_is_a_role_the_loader_accepts():
    assert set(ld.ROLE_FOR_GEOMETRY.values()) <= landuse.ROLES


def test_ambiguous_geometry_is_reported_and_never_picked():
    """`points` and `mixed` propose nothing, and say so in words."""
    for layer, kind in (("TREE-SYMBOLS", "points"),):
        row = entry_of(spread_draft(), layer)
        assert row["geometric_role"] == kind
        assert row["role"] is None
        assert row["blocking"], "an unresolved role must block acceptance"
        assert kind in row["role_basis"]

    mixed = ld.draft_entry(
        "ANY", profile=profile("mixed", entities=7), rule=None, declared_entities=7
    )
    assert mixed["role"] is None
    assert "mixed" in mixed["role_basis"]
    assert mixed["ready"] is False


def test_the_name_never_decides_the_role():
    """The C-ROAD trap, in miniature: same name, two geometries, two roles.

    Swapping the profile changes the drafted role; swapping the name does not.
    """
    named = "ROAD - CENTRELINE"
    as_network = ld.draft_entry(named, profile=profile("network", entities=99))
    as_annotation = ld.draft_entry(named, profile=profile("annotation", entities=99))

    assert as_network["role"] == "network"
    assert as_annotation["role"] == "annotation"

    other_name = ld.draft_entry("Z", profile=profile("network", entities=99))
    assert other_name["role"] == as_network["role"]


def test_a_name_that_disagrees_with_its_geometry_is_flagged():
    """A land-use word over geometry that is not closed rings is a warning.

    Stated without any table from a use to a role (G1): what is claimed is
    about the ROLE alone -- this layer holds no ring, therefore no plot.
    """
    row = entry_of(spread_draft(), "PARK-LABELS")
    assert row["use"] == "open_space"
    assert row["role"] == "annotation"
    assert row["tension"], "a name/geometry disagreement must be surfaced"
    assert row["tension"] in row["entry"]["note"]
    assert "DISAGREEMENT" in ld.render_yaml(spread_draft())


# --- 5. Units, absence, and the things that must not be invented -------------


def test_a_unit_is_never_invented(tmp_path, monkeypatch):
    """No metre appears because a metre was convenient (G2)."""
    doc = drawing([("NO-UNIT-LAYER", 5)], drawing_id="1111222233334444")
    profiles = {
        "NO-UNIT-LAYER": profile(
            "network", entities=5, kind="length", value=12.5, unit=None
        )
    }
    row = entry_of(ld.build_draft(doc, profiles=profiles), "NO-UNIT-LAYER")
    assert "drawing units" in row["measure"]
    assert " m" not in row["measure"] and "inch" not in row["measure"]


def test_a_count_is_never_given_a_unit():
    facts = {
        "looked": True,
        "measure_kind": "count",
        "measure_value": 40,
        "unit": None,
        "unit_reason": "none declared",
    }
    assert ld.measure_phrase(facts) == "count 40 entities"


def test_an_unmeasured_value_is_absent_and_not_zero():
    facts = {
        "looked": True,
        "measure_kind": "area",
        "measure_value": None,
        "unit": "m2",
    }
    phrase = ld.measure_phrase(facts)
    assert "absent" in phrase and "not zero" in phrase
    assert "0" not in phrase.replace("G8", "")


def test_a_multi_layout_layer_never_prints_one_figure_labelled_as_another():
    """Two entity counts, two questions, and both are named.

    `entities` is the layer over every bucket; `bucket_entities` is the one
    bucket the measure was computed over. Printing the first beside the second
    would put a number under the wrong label -- and a figure whose label is
    wrong is worse than a figure that is missing.
    """
    doc = drawing([("SPLIT", 192)], drawing_id="3333444455556666")
    split = profile(
        "network", entities=169, kind="length", value=4781.37, unit="inch", buckets=4
    )
    split["scope"]["entities_all_buckets"] = 192
    draft = ld.build_draft(doc, profiles={"SPLIT": split})
    row = entry_of(draft, "SPLIT")

    assert row["bucket_entities"] == 169
    assert row["entities"] == 192

    text = ld.render_yaml(draft)
    assert "169 entities, length 4781.37 inch" in text
    assert "192 entities over all of them" in text
    assert "never summed across layouts (G2)" in text


def test_a_count_is_not_stated_twice():
    """"18 entities, count 18 entities" reads as two figures. It is one."""
    row = ld.draft_entry("ANY", profile=profile("points", entities=18))
    assert row["profile_phrase"] == "count 18 entities"
    assert "18 entities, count" not in row["entry"]["sources"][-1]["detail"]


def test_accepting_a_use_less_entry_says_what_it_costs():
    """The residual stops seeing a layer that a config names.

    `store_landuse` builds the residual over layers that are NOT classified, so
    an accepted `use: unknown` entry takes its layer out of that search. That
    is a real trade and the reviewer is told about it rather than discovering
    it later in an answer that quietly got worse.
    """
    row = entry_of(spread_draft(), "00_Prop - Centreline")
    note = row["entry"]["note"]
    assert "no longer `unclassified`" in note
    assert "residual" in note
    assert "ONE CONSEQUENCE" in ld.render_yaml(spread_draft())


def test_a_declared_but_empty_layer_says_so_rather_than_guessing_a_role():
    row = entry_of(spread_draft(), "Car Parking Area")
    assert row["declared_but_empty"] is True
    assert row["use"] == "parking"
    assert row["role"] is None
    assert row["ready"] is True, (
        "an empty layer has no geometry to have a role, and the loader's "
        "default role has nothing to count on it"
    )
    assert "ZERO entities" in row["entry"]["note"]
    assert "role" not in row["entry"]


# --- 6. No Dossier: a gap, stated, never an error ----------------------------


def test_without_a_dossier_every_layer_is_name_only_and_the_gap_is_printed():
    """G3: no data is answered with a reason, never with a guess or a 500."""
    note = "no Dossier has been built for this drawing yet"
    draft = ld.build_draft(SPREAD, profiles={}, profiles_note=note)

    assert draft["counts"]["geometry_only"] == 0
    assert draft["counts"]["profiled"] == 0
    for row in draft["entries"]:
        assert row["role"] is None

    text = ld.render_yaml(draft)
    assert "ONE GAP, STATED" in text
    assert note.split()[0] in text
    assert note[:30] in " ".join(ld.report_lines(draft, path="x"))


def test_profiles_from_store_reports_a_missing_dossier_without_calling_it_empty(
    monkeypatch,
):
    stub = _stub_dossier_read(monkeypatch, document=None)
    profiles, note = ld.profiles_from_store("deadbeefdeadbeef", ["A", "B"])

    assert profiles == {}
    assert note and "NOT a statement that the drawing is empty" in note
    assert "dossier_backfill" in note
    assert stub["profiles_calls"] == []


def test_profiles_from_store_chunks_around_the_stated_cap(monkeypatch):
    """The per-call cap is a response budget; its own message says to ask in
    smaller calls. A 95-layer drawing is 3 calls over ONE fetched document,
    not 95 trips to the store, and not 40 layers with the rest dropped."""
    names = [f"L{i}" for i in range(95)]
    stub = _stub_dossier_read(
        monkeypatch,
        document={"_id": "x"},
        profiles={n: profile("region", layer=n) for n in names},
        cap=40,
    )
    profiles, note = ld.profiles_from_store("x", names)

    assert note is None
    assert set(profiles) == set(names)
    assert stub["fetches"] == 1
    assert [len(call) for call in stub["profiles_calls"]] == [40, 40, 15]


def _stub_dossier_read(monkeypatch, *, document, profiles=None, cap=40):
    """Replace `app.dossier_read` with a recorder. No database, no Mongo."""
    import types

    import app as app_pkg

    state = {"fetches": 0, "profiles_calls": []}
    module = types.ModuleType("app.dossier_read")
    module.MAX_PROFILE_LAYERS = cap

    def dossier_for(drawing_id):
        state["fetches"] += 1
        return document

    def profiles_for(drawing_id, layer_names, *, layout=None, dossier=None):
        names = list(layer_names)
        state["profiles_calls"].append(names)
        assert dossier is document, "the store must be read once, not per chunk"
        return {n: (profiles or {})[n] for n in names if n in (profiles or {})}

    module.dossier_for = dossier_for
    module.profiles_for = profiles_for
    monkeypatch.setitem(sys.modules, "app.dossier_read", module)
    monkeypatch.setattr(app_pkg, "dossier_read", module, raising=False)
    return state


# --- 7. Where the file goes --------------------------------------------------


def test_write_draft_refuses_the_config_directory(tmp_path):
    """Hard rule 4. A file in `landuse/` is a file the loader believes."""
    draft = spread_draft()

    with pytest.raises(ld.DraftRefused) as excinfo:
        ld.write_draft(draft, out_dir=landuse.CONFIG_DIR)
    assert excinfo.value.code == "DRAFT_IN_CONFIG_DIR"

    with pytest.raises(ld.DraftRefused):
        ld.write_draft(
            draft, path=landuse.CONFIG_DIR / f"{draft['drawing_id']}.yaml"
        )
    with pytest.raises(ld.DraftRefused):
        ld.write_draft(draft, out_dir=landuse.CONFIG_DIR / "_standards")

    assert not list(landuse.CONFIG_DIR.glob("*.draft.yaml"))


def test_write_draft_writes_where_it_says_and_returns_the_path(tmp_path):
    draft = spread_draft()
    written = ld.write_draft(draft, out_dir=tmp_path / "review")

    assert written.exists()
    assert written.name == f"{draft['drawing_id']}{ld.DRAFT_SUFFIX}"
    assert written.name != f"{draft['drawing_id']}.yaml", (
        "the draft must not carry the file name the loader looks for, even by "
        "accident"
    )
    assert written.read_text(encoding="utf-8") == ld.render_yaml(draft)


def test_the_default_directory_is_never_the_config_directory(monkeypatch, tmp_path):
    monkeypatch.setenv(ld.DRAFT_DIR_ENV, str(tmp_path / "elsewhere"))
    assert ld.default_draft_dir() == tmp_path / "elsewhere"

    monkeypatch.delenv(ld.DRAFT_DIR_ENV, raising=False)
    assert landuse.CONFIG_DIR.resolve() != ld.default_draft_dir().resolve()


# --- 8. The report, and the counts it prints ---------------------------------


def test_the_counts_are_a_partition_and_each_one_is_labelled():
    c = spread_draft()["counts"]
    assert c["use_proposed"] + c["geometry_only"] + c["name_only"] == c["drafted"]
    assert c["ready_to_accept"] + c["needs_a_decision"] == c["drafted"]
    assert c["profiled"] + c["unprofiled"] == c["drafted"]
    assert c["already_in_config"] + c["drafted"] == c["layers"]


def test_the_report_names_the_path_and_refuses_to_imply_anything_is_live():
    draft = spread_draft()
    lines = ld.report_lines(draft, path="/somewhere/review/x.draft.yaml")
    text = "\n".join(lines)

    assert "/somewhere/review/x.draft.yaml" in text
    assert "verified: false" in text
    assert "NOT in" in text and landuse.CONFIG_DIR.name in text
    assert str(draft["counts"]["use_proposed"]) in text


def test_the_report_is_ascii_so_a_console_cannot_refuse_it():
    text = "\n".join(ld.report_lines(spread_draft(), path="x"))
    assert text.isascii(), "the readiness report goes to a terminal we do not own"


# --- 9. Nothing travels between drawings (G10) -------------------------------


def test_the_same_layer_name_in_two_drawings_is_read_twice():
    """Meaning is never carried from one file into another.

    Same name, different geometry, different draft -- and neither draft can
    see the other, because the only inputs are that drawing's own layer table,
    its own Dossier and the global patterns.
    """
    name = "PLOT-EDGE"
    first = ld.build_draft(
        drawing([(name, 10)], drawing_id="1111111111111111"),
        profiles={name: profile("region", entities=10)},
    )
    second = ld.build_draft(
        drawing([(name, 10)], drawing_id="2222222222222222"),
        profiles={name: profile("annotation", entities=10)},
    )

    assert entry_of(first, name)["role"] == "parcel"
    assert entry_of(second, name)["role"] == "annotation"
    assert first["drawing_id"] != second["drawing_id"]


def test_a_draft_only_ever_contains_this_drawing_s_own_layers():
    draft = spread_draft()
    assert {row["layer"] for row in draft["entries"]} == {
        row["name"] for row in ld.declared_layers(SPREAD)
    }


def test_layers_a_human_already_decided_are_left_alone(tmp_path, monkeypatch):
    """A draft that argues with its own reviewer wastes both of them."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / f"{SPREAD['_id']}.yaml").write_text(
        "version: 1\n"
        f"drawing_id: \"{SPREAD['_id']}\"\n"
        "default_use: unknown\n"
        "layers:\n"
        '  "Primary School":\n'
        "    use: commercial\n"
        "    sources:\n"
        "      - {origin: human, reference: 'the drawing author, 25 Aug 2026',\n"
        "         detail: 'confirmed by the author'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(landuse, "CONFIG_DIR", config_dir)
    landuse.clear_cache()
    config = landuse.for_drawing(SPREAD["_id"])
    landuse.clear_cache()

    draft = ld.build_draft(SPREAD, profiles=SPREAD_PROFILES, config=config)
    row = entry_of(draft, "Primary School")

    assert row["status"] == ld.STATUS_ALREADY
    assert row["entry"] is None
    assert draft["counts"]["already_in_config"] == 1

    text = ld.render_yaml(draft)
    assert "ALREADY DECIDED BY A HUMAN" in text
    accepted = yaml.safe_load(ld.accept(text)) or {}
    assert "Primary School" not in (accepted.get("layers") or {})


# --- 10. The real store, over every drawing in it (G5) -----------------------

_STORE_REASON = (
    "MONGODB_URI is not set, so the drawing store was not read. These are the "
    "checks that run over EVERY drawing rather than over a fixture."
)


def _store_drawings():
    from app.mongo import COLL_DRAWINGS, coll

    return list(coll(COLL_DRAWINGS).find({}))


@pytest.mark.skipif(not os.environ.get("MONGODB_URI"), reason=_STORE_REASON)
def test_every_drawing_in_the_store_drafts_and_the_result_loads(
    tmp_path, monkeypatch
):
    """The invariant that catches the nineteenth drawing (G5).

    For every drawing the store holds -- read from the store, never
    hard-coded -- the draft must render, load as written with nothing
    classified, and load again with every ready proposal accepted.
    """
    drawings = _store_drawings()
    assert drawings, "the store answered with no drawings at all"

    for document in drawings:
        draft = ld.draft_for_drawing(document)
        text = ld.render_yaml(draft)

        as_written = yaml.safe_load(text)
        assert not as_written.get("layers"), (
            f"{document['_id']}: the draft classifies a layer as written"
        )

        config = load_as_config(
            ld.accept(text), tmp_path, draft["drawing_id"], monkeypatch
        )
        ready = {e["layer"] for e in draft["entries"] if e["ready"]}
        assert set(config.layers) == ready
        for land_use in config.layers.values():
            assert land_use.role in landuse.ROLES
            assert land_use.use in landuse.USES
            assert land_use.sources
