"""The ingest run history, and the upload route that opens one.

The owner's rule for this feature is that the user must never stare at a
spinner: every stage is written to MongoDB as it happens, so a browser
refresh re-reads the same truth instead of restarting a guess. These tests
hold that rule to three specific promises.

1. The history is a TRANSCRIPT of the pipeline, not a second copy of it. The
   stage names are exactly the steps `ingest_file` already had, in the order
   it already ran them.
2. A refused upload is a RECORDED event. "Never silently drop" applies to the
   files intake turns away, not only to the ones it keeps.
3. A stalled run says so. A run that stopped writing cannot be trusted to
   have written a flag saying it stopped, so the verdict is derived on read.

Route functions are called directly rather than over HTTP, which is how the
rest of this suite tests routes (see `tests/test_combine.py`).
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import UploadFile

from app import ingest_runs, main
from app.mongo import COLL_INGEST_RUNS, coll


# ---------------------------------------------------------------------------
# The contract's shape
# ---------------------------------------------------------------------------


def test_stage_names_are_the_pipeline_that_exists():
    """The stages are the steps ingest already had, in the order it ran them.

    Written down because the temptation with a progress feature is to invent
    a nicer-looking sequence than the code performs, and then the history is
    a story rather than a record.
    """
    assert ingest_runs.STAGES == (
        "upload",
        "convert",
        "extract",
        "store",
        "render",
        "dossier",
        "landuse_draft",
        "h3",
        "done",
    )


def test_the_run_collection_is_registered_as_ours():
    """An unnamed collection is exactly the surprise the fence prevents."""
    from app import mongo

    assert mongo.COLL_INGEST_RUNS == "autocad_ingest_runs"
    assert mongo.COLL_INGEST_RUNS.startswith(mongo.PREFIX)
    assert mongo.COLL_INGEST_RUNS in mongo.OWNED_COLLECTIONS


def test_a_run_id_is_not_a_drawing_id():
    """One drawing can be ingested many times; each attempt is its own record."""
    first, second = ingest_runs.new_run_id(), ingest_runs.new_run_id()
    assert first != second
    assert len(first) == 16


# ---------------------------------------------------------------------------
# The watchdog, which is derived and never stored
# ---------------------------------------------------------------------------


def _doc(overall: str, updated_age_s: int) -> dict:
    return {
        "_id": "r1",
        "overall": overall,
        "updated_at": datetime.now(timezone.utc) - timedelta(seconds=updated_age_s),
        "stages": [],
    }


def test_a_running_run_that_stopped_writing_is_called_stalled():
    decorated = ingest_runs._decorate(
        _doc(ingest_runs.OVERALL_RUNNING, ingest_runs.STALL_AFTER_S + 30)
    )
    assert decorated["stalled"] is True
    assert decorated["stalled_since"] is not None


def test_a_running_run_that_wrote_recently_is_not_stalled():
    decorated = ingest_runs._decorate(_doc(ingest_runs.OVERALL_RUNNING, 5))
    assert decorated["stalled"] is False
    assert decorated["stalled_since"] is None


def test_a_finished_run_is_never_stalled_however_old():
    """A run that ended last week is finished, not hanging."""
    decorated = ingest_runs._decorate(_doc(ingest_runs.OVERALL_DONE, 7 * 86_400))
    assert decorated["stalled"] is False


def test_the_stall_threshold_clears_the_longest_measured_stage():
    """Sedra measured this, after Janadriyah had set it too low.

    The threshold was 180 s, taken from Janadriyah's ~178 s render. Sedra then
    ran a single atomic extract over 500 MB for 24 minutes, writing nothing
    while it worked, and the run was reported as stalled for twenty of them.
    A watchdog that cries wolf on the largest healthy drawing in the corpus is
    worse than none, because the next reader learns to ignore it.
    """
    SEDRA_EXTRACT_S = 24 * 60
    assert ingest_runs.STALL_AFTER_S > SEDRA_EXTRACT_S


def test_decorate_renames_the_id_so_the_api_shape_is_stable():
    decorated = ingest_runs._decorate(_doc(ingest_runs.OVERALL_DONE, 1))
    assert decorated["run_id"] == "r1"
    assert "_id" not in decorated


# ---------------------------------------------------------------------------
# The recorder never breaks an ingest
# ---------------------------------------------------------------------------


def test_the_null_recorder_writes_nothing_and_raises_nothing():
    """The default for every existing caller, so their behaviour is unchanged."""
    run = ingest_runs.NullRecorder()
    run.start("extract")
    run.finish("extract", entities=5)
    run.fail("store", "boom")
    run.succeeded()
    run.refused("X", "y", "z")
    assert run.run_id == ""


def test_a_storage_failure_in_the_recorder_is_swallowed(monkeypatch):
    """Losing a progress line beats losing the ingest.

    A recorder that raised would turn a history feature into a brand new way
    to fail an ingestion that would otherwise have succeeded.
    """
    from pymongo.errors import PyMongoError

    class Boom:
        def update_one(self, *_args, **_kwargs):
            raise PyMongoError("no database today")

    monkeypatch.setattr(ingest_runs, "coll", lambda _name: Boom())
    run = ingest_runs.Recorder("whatever")
    run.finish("extract", entities=1)  # must not raise


def test_an_unknown_stage_name_is_refused_rather_than_written(monkeypatch):
    """A typo would otherwise write a stage nothing renders."""
    calls: list = []

    class Spy:
        def update_one(self, *args, **kwargs):
            calls.append(args)

    monkeypatch.setattr(ingest_runs, "coll", lambda _name: Spy())
    ingest_runs.Recorder("r").start("extarct")
    assert calls == []


# ---------------------------------------------------------------------------
# Against the real local database
# ---------------------------------------------------------------------------


@pytest.fixture
def cleanup_runs():
    created: list[str] = []
    yield created
    if created:
        coll(COLL_INGEST_RUNS).delete_many({"_id": {"$in": created}})


def test_a_run_is_written_stage_by_stage_not_at_the_end(cleanup_runs):
    """The whole point: the record exists WHILE the work is happening."""
    run_id = ingest_runs.create(filename="probe.dxf", source="test")
    cleanup_runs.append(run_id)

    opened = ingest_runs.get(run_id)
    assert opened is not None
    assert opened["overall"] == ingest_runs.OVERALL_RUNNING
    assert [s["status"] for s in opened["stages"]] == [ingest_runs.PENDING] * 9

    recorder = ingest_runs.Recorder(run_id)
    recorder.finish("upload", bytes=10)
    recorder.start("extract")

    midway = ingest_runs.get(run_id)
    stages = {s["name"]: s for s in midway["stages"]}
    assert stages["upload"]["status"] == ingest_runs.DONE
    assert stages["upload"]["detail"]["bytes"] == 10
    assert stages["extract"]["status"] == ingest_runs.RUNNING
    assert stages["store"]["status"] == ingest_runs.PENDING
    assert midway["overall"] == ingest_runs.OVERALL_RUNNING


def test_a_skipped_run_is_not_reported_as_done(cleanup_runs):
    """Already-stored content is usable, but nothing after extract ran.

    Calling that `done` would be a lie the next reader has to un-learn.
    """
    run_id = ingest_runs.create(filename="again.dxf", source="test")
    cleanup_runs.append(run_id)
    ingest_runs.Recorder(run_id).already_current(drawing_id="abc")

    doc = ingest_runs.get(run_id)
    assert doc["overall"] == ingest_runs.OVERALL_SKIPPED
    stages = {s["name"]: s for s in doc["stages"]}
    assert stages["store"]["status"] == ingest_runs.SKIPPED
    assert stages["done"]["status"] == ingest_runs.DONE


def test_a_refusal_is_recorded_with_its_reason(cleanup_runs):
    run_id = ingest_runs.create(filename="dupe.dxf", source="test")
    cleanup_runs.append(run_id)
    ingest_runs.Recorder(run_id).refused(
        "ALREADY_INGESTED", "identical content", "open it from the picker"
    )

    doc = ingest_runs.get(run_id)
    assert doc["overall"] == ingest_runs.OVERALL_REFUSED
    assert doc["refusal"]["code"] == "ALREADY_INGESTED"
    assert doc["refusal"]["message"] == "identical content"
    assert doc["finished_at"] is not None


def test_latest_filters_by_drawing_and_orders_newest_first(cleanup_runs):
    older = ingest_runs.create(filename="a.dxf", source="test", drawing_id="dr-x")
    newer = ingest_runs.create(filename="b.dxf", source="test", drawing_id="dr-x")
    other = ingest_runs.create(filename="c.dxf", source="test", drawing_id="dr-y")
    cleanup_runs.extend([older, newer, other])

    rows = ingest_runs.latest(drawing_id="dr-x", limit=10)
    ids = [r["run_id"] for r in rows]
    assert other not in ids
    assert set(ids) >= {older, newer}
    assert ids.index(newer) <= ids.index(older)


# ---------------------------------------------------------------------------
# The upload route
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def stage_uploads_in_tmp(tmp_path, monkeypatch):
    """Never let a test write into the real, shared SVG volume.

    Learned the hard way: two of the refusal tests below did not redirect the
    staging directory, so running the suite in a root container created a
    root-owned `/data/svg/uploads` in the volume the SERVICE writes to as a
    non-root user, and every later upload failed with PermissionError. A test
    that breaks the running system is worse than no test.
    """
    monkeypatch.setattr(main, "_upload_dir", lambda _settings: tmp_path)


class _NoPool:
    """Accepts the submission without running the ingest.

    The route's job under test is what it decides, not what the pipeline
    afterwards does with it.
    """

    def __init__(self) -> None:
        self.submitted: list = []

    def submit(self, *args, **kwargs):
        self.submitted.append(args)


def _upload(name: str, payload: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(payload), filename=name)


def test_a_file_that_is_not_a_drawing_is_refused_by_name(monkeypatch, cleanup_runs):
    monkeypatch.setattr(main, "_INGEST_POOL", _NoPool())

    with pytest.raises(main.ApiError) as caught:
        main.upload_drawing(file=_upload("notes.pdf", b"%PDF-1.7"))

    assert caught.value.status == 415
    assert caught.value.code == "UNSUPPORTED_FILE"
    assert ".pdf" in caught.value.message
    _collect_runs_for(cleanup_runs, "notes.pdf")


def test_an_empty_upload_is_refused(monkeypatch, cleanup_runs):
    monkeypatch.setattr(main, "_INGEST_POOL", _NoPool())

    with pytest.raises(main.ApiError) as caught:
        main.upload_drawing(file=_upload("empty.dxf", b""))

    assert caught.value.status == 415
    _collect_runs_for(cleanup_runs, "empty.dxf")


def test_an_oversized_upload_is_stopped_and_names_the_cli(monkeypatch, cleanup_runs):
    """The check is inside the copy loop, so the file never fully lands."""
    monkeypatch.setattr(main, "_INGEST_POOL", _NoPool())
    monkeypatch.setattr(main, "UPLOAD_MAX_BYTES", 1024)

    with pytest.raises(main.ApiError) as caught:
        main.upload_drawing(file=_upload("huge.dxf", b"x" * 4096))

    assert caught.value.status == 413
    assert "app.ingest" in (caught.value.hint or "")
    _collect_runs_for(cleanup_runs, "huge.dxf")


def test_identical_bytes_are_refused_and_the_existing_drawing_is_named(
    monkeypatch, cleanup_runs
):
    """The duplicate rule, stated in terms the user can act on."""
    monkeypatch.setattr(main, "_INGEST_POOL", _NoPool())
    monkeypatch.setattr(
        main.store,
        "get_drawing",
        lambda _id: {
            "_id": _id,
            "original_filename": "already-here.dxf",
            "ingested_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
        },
    )

    with pytest.raises(main.ApiError) as caught:
        main.upload_drawing(file=_upload("copy.dxf", b"  0\nSECTION\n  0\nEOF\n"))

    assert caught.value.status == 409
    assert caught.value.code == "ALREADY_INGESTED"
    assert "already-here.dxf" in caught.value.message
    assert "2026-08-01" in caught.value.message
    _collect_runs_for(cleanup_runs, "copy.dxf")


def test_a_refused_upload_still_leaves_a_run_document(monkeypatch, cleanup_runs):
    """Never silently drop, applied to the files intake turns away."""
    monkeypatch.setattr(main, "_INGEST_POOL", _NoPool())

    with pytest.raises(main.ApiError) as caught:
        main.upload_drawing(file=_upload("nope.txt", b"hello"))

    run_id = _run_id_from_hint(caught.value.hint or "")
    cleanup_runs.append(run_id)
    doc = ingest_runs.get(run_id)
    assert doc is not None
    assert doc["overall"] == ingest_runs.OVERALL_REFUSED
    assert doc["refusal"]["code"] == "UNSUPPORTED_FILE"


def test_a_new_drawing_is_accepted_and_queued(monkeypatch, cleanup_runs, tmp_path):
    monkeypatch.setattr(main, "_INGEST_POOL", _NoPool())
    monkeypatch.setattr(main.store, "get_drawing", lambda _id: None)
    monkeypatch.setattr(main.store, "list_drawings", lambda **_k: [])

    body = b"  0\nSECTION\n  2\nHEADER\n  0\nENDSEC\n  0\nEOF\n"
    answer = main.upload_drawing(file=_upload("fresh.dxf", body))
    cleanup_runs.append(answer["run_id"])

    assert answer["overall"] == ingest_runs.OVERALL_RUNNING
    assert answer["bytes"] == len(body)
    assert len(answer["drawing_id"]) == 16
    # The file landed, under its own name, ready for the pipeline to read.
    assert (tmp_path / "fresh.dxf").read_bytes() == body

    doc = ingest_runs.get(answer["run_id"])
    stages = {s["name"]: s for s in doc["stages"]}
    assert stages["upload"]["status"] == ingest_runs.DONE
    assert stages["upload"]["detail"]["sha16"] == answer["drawing_id"]


def test_the_upload_hash_is_the_drawing_id_and_not_a_second_hash(
    monkeypatch, cleanup_runs, tmp_path
):
    """There is one identity for a drawing and it is the content hash."""
    from app.extract import compute_drawing_id

    monkeypatch.setattr(main, "_INGEST_POOL", _NoPool())
    monkeypatch.setattr(main.store, "get_drawing", lambda _id: None)
    monkeypatch.setattr(main.store, "list_drawings", lambda **_k: [])

    body = b"  0\nSECTION\n  0\nEOF\n"
    answer = main.upload_drawing(file=_upload("hashme.dxf", body))
    cleanup_runs.append(answer["run_id"])

    source = tmp_path / "hashme.dxf"
    assert answer["drawing_id"] == compute_drawing_id(source)


def _run_id_from_hint(hint: str) -> str:
    """The refusal hint carries the run id so the UI can show the record."""
    marker = "(run "
    start = hint.index(marker) + len(marker)
    return hint[start:].rstrip(")")


def _collect_runs_for(cleanup: list[str], filename: str) -> None:
    """Register whatever runs this filename opened, so the test cleans up."""
    # Through the INDEXED listing and filtered in Python, rather than a
    # `find({"filename": ...})`. There is no index on `filename` and there is
    # no production query that wants one, so adding an index to please a test
    # would be inventing a shape the system does not have. Under
    # `notablescan` -- which is how Atlas runs -- the direct find is refused.
    for row in ingest_runs.latest(limit=100):
        if row.get("filename") != filename:
            continue
        cleanup.append(row["run_id"])


def test_a_bare_predecessor_id_does_not_crash_the_run(cleanup_runs):
    """The regression this feature actually shipped, then caught end to end.

    `recipes.revision.find_predecessors` answers with an ID string, not a
    mapping. The recorder called `dict()` on it, which raised ValueError deep
    inside the arrival path and failed a run whose drawing had already been
    stored, rendered and profiled successfully. The bug was invisible until a
    same-name upload actually found a predecessor.

    Two things had to be true for that to be survivable, and both are asserted
    here and in the test below: the id must be accepted, and a crash anywhere
    in ingest must still close the run rather than leave it saying `running`.
    """
    run_id = ingest_runs.create(filename="rev.dxf", source="test")
    cleanup_runs.append(run_id)

    ingest_runs.Recorder(run_id).set_revision_of("e822c0493c09cb58")

    doc = ingest_runs.get(run_id)
    assert doc["revision_of"] == {"drawing_id": "e822c0493c09cb58"}


def test_a_mapping_predecessor_is_kept_whole(cleanup_runs):
    run_id = ingest_runs.create(filename="rev2.dxf", source="test")
    cleanup_runs.append(run_id)
    block = {"drawing_id": "abc", "filename": "old.dxf", "ingested_at": None}

    ingest_runs.Recorder(run_id).set_revision_of(block)

    assert ingest_runs.get(run_id)["revision_of"] == block


def test_predecessor_block_names_the_drawing_for_a_human(monkeypatch):
    """An id alone tells a reader nothing; the panel needs a name and a date."""
    from app import ingest as ingest_mod

    monkeypatch.setattr(
        ingest_mod.store,
        "get_drawing",
        lambda _id: {"original_filename": "site-plan.dxf", "ingested_at": "2026-08-01"},
    )
    block = ingest_mod.predecessor_block("abc123")
    assert block == {
        "drawing_id": "abc123",
        "filename": "site-plan.dxf",
        "ingested_at": "2026-08-01",
    }
    assert ingest_mod.predecessor_block(None) is None


def test_a_crash_inside_ingest_closes_the_run_instead_of_leaving_it_running(
    cleanup_runs,
):
    """A thread that dies silently leaves the exact spinner this feature abolishes."""
    run_id = ingest_runs.create(filename="boom.dxf", source="test")
    cleanup_runs.append(run_id)

    def explode(_path, recorder=None):
        raise ValueError("something deep went wrong")

    main._run_ingest(explode, main.Path("/tmp/boom.dxf"), run_id)

    doc = ingest_runs.get(run_id)
    assert doc["overall"] == ingest_runs.OVERALL_FAILED
    assert "something deep went wrong" in doc["refusal"]["message"]
    assert doc["finished_at"] is not None


# ---------------------------------------------------------------------------
# The stale sweep, which Sedra broke
# ---------------------------------------------------------------------------


def test_the_stale_sweep_is_bounded_and_does_not_grow_with_the_drawing():
    """Sedra made this a measured defect rather than a theoretical one.

    The sweep used to be one `delete_many` carrying `{"_id": {"$nin": ...}}`
    with every fresh id in it. That is a single BSON command document, and
    MongoDB caps a command at 16 MB. Sedra extracts 990,790 entities and the
    sweep failed with `DocumentTooLarge: 'delete' command document too large`
    AFTER a 24 minute extract had succeeded, so the drawing was unstorable for
    a reason that had nothing to do with the drawing.

    What is asserted here is the property that prevents it: whatever the size
    of the drawing, no single delete carries more than one batch of ids, and
    a drawing with nothing stale sends no delete at all.
    """
    from app import store

    seen: list[dict] = []

    class FakeCollection:
        def find(self, query, projection):
            # Two ids stored; one of them is not in this ingest.
            return iter([{"_id": "d:AAA"}, {"_id": "d:OLD"}])

        def delete_many(self, query):
            seen.append(query)

            class Result:
                deleted_count = len(query["_id"]["$in"])

            return Result()

    original = store.coll
    store.coll = lambda _name: FakeCollection()
    try:
        removed = store._sweep_stale("d", ["d:AAA"])
    finally:
        store.coll = original

    assert removed == 1
    assert len(seen) == 1
    # The delete names what is stale, never what is fresh: an $nin of every
    # fresh id is the shape that could not scale.
    assert seen[0] == {"_id": {"$in": ["d:OLD"]}}


def test_a_sweep_with_nothing_stale_sends_no_delete_at_all():
    from app import store

    calls: list = []

    class FakeCollection:
        def find(self, query, projection):
            return iter([{"_id": "d:AAA"}])

        def delete_many(self, query):
            calls.append(query)
            raise AssertionError("no delete should be issued")

    original = store.coll
    store.coll = lambda _name: FakeCollection()
    try:
        assert store._sweep_stale("d", ["d:AAA"]) == 0
    finally:
        store.coll = original
    assert calls == []


def test_the_sweep_batches_so_one_command_never_carries_a_whole_drawing():
    """The property that makes a one million entity drawing storable."""
    from app import store

    stored = [f"d:{i:07d}" for i in range(store._BULK_BATCH * 2 + 7)]
    sizes: list[int] = []

    class FakeCollection:
        def find(self, query, projection):
            return iter({"_id": _id} for _id in stored)

        def delete_many(self, query):
            ids = query["_id"]["$in"]
            sizes.append(len(ids))

            class Result:
                deleted_count = len(ids)

            return Result()

    original = store.coll
    store.coll = lambda _name: FakeCollection()
    try:
        removed = store._sweep_stale("d", [])  # nothing fresh: all are stale
    finally:
        store.coll = original

    assert removed == len(stored)
    assert max(sizes) <= store._BULK_BATCH
    assert sum(sizes) == len(stored)


# ---------------------------------------------------------------------------
# Which layouts an ingest renders, and why Sedra forced a cap
# ---------------------------------------------------------------------------


def _layouts(*pairs):
    return {"layouts": [{"name": n, "entity_count": c} for n, c in pairs]}


def test_empty_layouts_are_not_offered_for_render():
    """A blank SVG in the picker is indistinguishable from a broken render."""
    from app.ingest import _renderable_layouts

    names, skipped, too_big = _renderable_layouts(_layouts(("Model", 5), ("Sheet-1", 0)))
    assert names == ["Model"]
    assert skipped == 0
    assert too_big == []


def test_real_layouts_come_first_and_are_never_capped():
    """They are the sheets somebody drew; they are what a reviewer opens."""
    from app.ingest import _renderable_layouts, MAX_BLOCK_LAYOUT_RENDERS

    many_real = [(f"Sheet-{i}", 3) for i in range(MAX_BLOCK_LAYOUT_RENDERS + 20)]
    names, skipped, _too_big = _renderable_layouts(_layouts(*many_real))
    assert len(names) == len(many_real)
    assert skipped == 0


def test_block_layouts_are_capped_and_the_remainder_is_counted():
    """Sedra brings 487 block layouts against 21 real ones.

    Rendering every one of them on a 500 MB document is what turned a large
    ingest into an unbounded one. The cap is the fix; the count is what stops
    it being a silent drop.
    """
    from app.ingest import _renderable_layouts, MAX_BLOCK_LAYOUT_RENDERS

    real = [("Model", 19)] + [(f"Sheet-{i}", 20) for i in range(20)]
    blocks = [(f"[block] B{i}", 40) for i in range(487)]
    names, skipped, _too_big = _renderable_layouts(_layouts(*(real + blocks)))

    assert names[: len(real)] == [n for n, _ in real], "real layouts must lead"
    assert len(names) == len(real) + MAX_BLOCK_LAYOUT_RENDERS
    assert skipped == 487 - MAX_BLOCK_LAYOUT_RENDERS
    assert all(n.startswith("[block] ") for n in names[len(real):])


def test_a_drawing_with_only_block_layouts_still_renders_some():
    """The Sedra shape: nothing in Model worth drawing, everything in blocks."""
    from app.ingest import _renderable_layouts, MAX_BLOCK_LAYOUT_RENDERS

    names, skipped, _too_big = _renderable_layouts(
        _layouts(*[(f"[block] B{i}", 10) for i in range(120)])
    )
    assert len(names) == MAX_BLOCK_LAYOUT_RENDERS
    assert skipped == 120 - MAX_BLOCK_LAYOUT_RENDERS


# ---------------------------------------------------------------------------
# A run whose process was killed, and a drawing whose entities never landed
# ---------------------------------------------------------------------------


def test_a_run_abandoned_long_enough_is_closed(cleanup_runs):
    """A SIGKILL is not an exception, so nothing in the dying process runs.

    When the OOM killer took Sedra's ingest, the run document kept saying
    `running` and would have said it for ever. The stall verdict tells a
    reader it has gone quiet; this is what turns that into a verdict.
    """
    run_id = ingest_runs.create(filename="killed.dxf", source="test")
    cleanup_runs.append(run_id)
    stale = datetime.now(timezone.utc) - timedelta(
        seconds=ingest_runs.ABANDON_AFTER_S + 60
    )
    coll(COLL_INGEST_RUNS).update_one({"_id": run_id}, {"$set": {"updated_at": stale}})

    ingest_runs.reap_abandoned()

    doc = ingest_runs.get(run_id)
    assert doc["overall"] == ingest_runs.OVERALL_FAILED
    assert doc["refusal"]["code"] == "INGEST_ABANDONED"
    assert doc["finished_at"] is not None


def test_a_run_that_is_merely_slow_is_left_alone(cleanup_runs):
    """Sedra's extract writes nothing for 24 minutes and is perfectly healthy."""
    run_id = ingest_runs.create(filename="slow.dxf", source="test")
    cleanup_runs.append(run_id)
    quiet = datetime.now(timezone.utc) - timedelta(seconds=ingest_runs.STALL_AFTER_S + 60)
    coll(COLL_INGEST_RUNS).update_one({"_id": run_id}, {"$set": {"updated_at": quiet}})

    ingest_runs.reap_abandoned()

    assert ingest_runs.get(run_id)["overall"] == ingest_runs.OVERALL_RUNNING


def test_the_abandon_threshold_is_looser_than_the_stall_one():
    """Stalled says it has gone quiet; abandoned says it is not coming back.

    The second is a much stronger claim and must need much more silence.
    """
    assert ingest_runs.ABANDON_AFTER_S > ingest_runs.STALL_AFTER_S


def test_a_finished_run_is_never_reopened_by_the_reaper(cleanup_runs):
    run_id = ingest_runs.create(filename="done.dxf", source="test")
    cleanup_runs.append(run_id)
    ingest_runs.Recorder(run_id).succeeded()
    stale = datetime.now(timezone.utc) - timedelta(days=30)
    coll(COLL_INGEST_RUNS).update_one({"_id": run_id}, {"$set": {"updated_at": stale}})

    ingest_runs.reap_abandoned()

    assert ingest_runs.get(run_id)["overall"] == ingest_runs.OVERALL_DONE


def test_a_drawing_whose_entities_never_landed_is_not_offered(monkeypatch):
    """The picker must not advertise a drawing with nothing behind it.

    The drawing document is written before `replace_entities`, so a failure in
    between leaves an entry announcing hundreds of thousands of entities and
    drawing nothing, which to a user is a broken viewer.
    """
    from app import store

    seen: dict = {}

    class FakeCursor:
        def sort(self, *_a, **_k):
            return self

        def __iter__(self):
            return iter(())

    class FakeColl:
        def find(self, query, *_args, **_kwargs):
            seen["query"] = query
            return FakeCursor()

    monkeypatch.setattr(store, "coll", lambda _name: FakeColl())
    store.list_drawings()

    assert seen["query"] == {"ingest_incomplete": {"$ne": True}}, (
        "the flag is negative so a drawing ingested before it existed stays visible"
    )


def test_a_layout_that_would_expand_past_the_budget_is_refused_not_attempted():
    """Sedra's model space: 19 entities, roughly 990,000 drawn.

    Attempting it took the process past a 15.47 GiB ceiling and the OOM
    killer ended the whole ingest. A refusal costs one sheet and says which
    one and why; a kill costs everything and says nothing.
    """
    from app.ingest import MAX_RENDER_EXPANSION, _renderable_layouts

    doc = {
        "layouts": [
            {"name": "Model", "entity_count": 19,
             "expanded_entity_count": 990_000},
            {"name": "Sheet-1", "entity_count": 21,
             "expanded_entity_count": 21},
        ]
    }
    names, _skipped, too_big = _renderable_layouts(doc)

    assert names == ["Sheet-1"], "the affordable sheet is still drawn"
    assert too_big == [("Model", 990_000)]
    assert MAX_RENDER_EXPANSION < 990_000


def test_a_drawing_ingested_before_the_cost_was_measured_keeps_its_behaviour():
    """No `expanded_entity_count` means the layout's own count is the guide."""
    from app.ingest import _renderable_layouts

    names, _skipped, too_big = _renderable_layouts(
        {"layouts": [{"name": "Model", "entity_count": 46_754}]}
    )
    assert names == ["Model"]
    assert too_big == []


def test_a_drawing_whose_ingest_failed_can_be_uploaded_again(
    monkeypatch, cleanup_runs, tmp_path
):
    """Otherwise a failed ingest is a dead end: the entry is unusable and the
    only door to replacing it refuses on the fingerprint it left behind."""
    monkeypatch.setattr(main, "_INGEST_POOL", _NoPool())
    monkeypatch.setattr(
        main.store, "get_drawing",
        lambda _id: {"_id": _id, "original_filename": "half.dxf",
                     "ingest_incomplete": True, "ingested_at": None},
    )
    monkeypatch.setattr(main.store, "list_drawings", lambda **_k: [])

    answer = main.upload_drawing(file=_upload("half.dxf", b"  0\nSECTION\n  0\nEOF\n"))
    cleanup_runs.append(answer["run_id"])

    assert answer["overall"] == ingest_runs.OVERALL_RUNNING


def test_a_complete_drawing_is_still_refused_as_a_duplicate(
    monkeypatch, cleanup_runs, tmp_path
):
    """The idempotency rule is not loosened for healthy drawings."""
    monkeypatch.setattr(main, "_INGEST_POOL", _NoPool())
    monkeypatch.setattr(
        main.store, "get_drawing",
        lambda _id: {"_id": _id, "original_filename": "whole.dxf",
                     "ingested_at": datetime(2026, 8, 1, tzinfo=timezone.utc)},
    )

    with pytest.raises(main.ApiError) as caught:
        main.upload_drawing(file=_upload("whole.dxf", b"  0\nSECTION\n  0\nEOF\n"))

    assert caught.value.code == "ALREADY_INGESTED"
    _collect_runs_for(cleanup_runs, "whole.dxf")


def test_a_crashed_ingest_names_the_stage_that_raised_and_carries_its_traceback():
    """A dead ingest must always name its cause.

    Proven by making a stage fail deliberately and reading the reason back off
    the run, because the failure mode this guards is precisely the one nobody
    sees: a background task raises, the exception reaches only the container
    log, the reaper eventually closes the run, and no one can say why.

    Two things are asserted that the previous version got wrong. The failure
    is recorded against the stage that was RUNNING -- blaming `done` leaves
    the real stage at `pending`, which on this project is already the shape of
    a silent death -- and the traceback travels with it, because
    "unexpected OperationFailure: ..." names a type and a message but not a
    line, and on a cluster that refuses un-indexed queries the message alone
    does not say which query.
    """
    run_id = ingest_runs.create(filename="probe.dxf", source="test")
    recorder = ingest_runs.Recorder(run_id)
    recorder.start("dossier")

    def boom() -> None:
        raise RuntimeError("the dossier aggregation was refused")

    try:
        boom()
    except RuntimeError as exc:
        named = recorder.crashed(exc)

    assert named == "dossier", "the stage that was running, not `done`"

    doc = ingest_runs.get(run_id)
    assert doc["overall"] == ingest_runs.OVERALL_FAILED
    stages = {s["name"]: s for s in doc["stages"]}
    assert stages["dossier"]["status"] == ingest_runs.FAILED
    detail = stages["dossier"]["detail"]
    assert "the dossier aggregation was refused" in detail["why"]
    assert "RuntimeError" in detail["why"]
    # The frames nearest the raise, which is what says WHERE.
    assert "boom" in detail["traceback_tail"]
    assert "test_ingest_runs.py" in detail["traceback_tail"]
    # `done` must NOT be the one blamed.
    assert stages["done"]["status"] != ingest_runs.FAILED
    assert doc["refusal"]["code"] == "INGEST_CRASHED"
    assert "dossier" in doc["refusal"]["hint"]
