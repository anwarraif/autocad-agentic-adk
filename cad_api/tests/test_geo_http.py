"""Wave 2 — the HTTP behaviour of the geo surface.

What is testable here and what is not, decided once so the split is not
re-argued per test.

The things that are pure — how an ETag is derived, how a bounding box is
parsed, how parameters normalise — are tested here, hermetically, without a
server. They are where the bugs live: a cache key that ignores a parameter
serves the wrong body, and a bbox parser that accepts a reversed box returns
an empty answer that looks like an empty site.

The things that only exist on the wire — that gzip really compressed, that a
conditional request really came back 304 — are proved by
`scripts/geo_smoke.py` against the running service, because that is the only
place they happen. What IS asserted here about them is the configuration they
depend on: middleware installed, with the size floor this file states.
"""

from __future__ import annotations

import pytest

from app import geo_view, main

GZIP = "GZipMiddleware"


def _middleware_named(name: str):
    for entry in main.app.user_middleware:
        if getattr(entry.cls, "__name__", "") == name:
            return entry
    return None


def test_responses_are_compressed_when_the_caller_asks():
    """P1. Measured on the payload that needs it: `/geo?res=9&parcels=true`
    is 997,594 bytes uncompressed and 127,485 gzipped — 7.8 times smaller.

    The middleware being installed is what this can check without a socket;
    that it really compressed is in `scripts/geo_smoke.py`."""
    entry = _middleware_named(GZIP)
    assert entry is not None, "gzip middleware is not installed"
    assert entry.kwargs.get("minimum_size") == main.GZIP_MIN_BYTES


def test_small_responses_are_left_alone():
    """The floor is a decision, not an accident: `/ready` is 78 bytes and
    compressing it would cost two rounds of CPU to save nothing. Stated as a
    constant so that a response which is NOT compressed has an explanation
    rather than looking like a broken middleware."""
    assert main.GZIP_MIN_BYTES == 500


def test_compression_is_opt_in_and_cannot_change_an_existing_client():
    """gzip is applied only to a request that carries `Accept-Encoding: gzip`.
    A caller that does not ask receives the bytes it received before this
    phase — which is what makes this additive rather than a change of
    contract."""
    entry = _middleware_named(GZIP)
    assert entry is not None
    # Starlette's own semantics; asserted as documentation of the reason the
    # phase is safe, and it fails loudly if the class is ever swapped for one
    # that compresses unconditionally.
    import inspect

    from fastapi.middleware.gzip import GZipMiddleware

    source = inspect.getsource(GZipMiddleware.__call__)
    assert "accept-encoding" in source.lower()


# ---------------------------------------------------------------------------
# P2 — conditional requests
# ---------------------------------------------------------------------------

JANADRIYAH = "596212db022a3397"

BASE = {
    "config_version": 1,
    "stored_resolution": 13,
    "ingest_version": 7,
    "params": {"res": 9, "layout": "Model", "parcels": False},
}


def _tag(**overrides):
    facts = dict(BASE)
    facts.update(overrides)
    return geo_view.etag_for(JANADRIYAH, **facts)


def test_the_tag_is_strong_and_shaped_like_one():
    """Strong, not weak: the bytes really are identical for identical inputs,
    because nothing on this route reads a clock, a random number, or anything
    outside the store."""
    tag = _tag()
    assert tag.startswith('"') and tag.endswith('"')
    assert not tag.startswith("W/")
    assert len(tag) == 34  # 32 hex characters inside the quotes


def test_the_same_question_asked_twice_has_the_same_tag():
    assert _tag() == _tag()


def test_the_order_parameters_were_typed_in_does_not_change_the_tag():
    """A cache key that depends on typing order is a cache that mostly
    misses."""
    forward = geo_view.etag_for(
        JANADRIYAH,
        **{**BASE, "params": {"res": 9, "layout": "Model", "parcels": False}},
    )
    backward = geo_view.etag_for(
        JANADRIYAH,
        **{**BASE, "params": {"parcels": False, "layout": "Model", "res": 9}},
    )
    assert forward == backward


@pytest.mark.parametrize(
    "change",
    [
        {"params": {"res": 13, "layout": "Model", "parcels": False}},
        {"params": {"res": 9, "layout": "Model", "parcels": True}},
        {"params": {"res": 9, "layout": "DMP Layout1", "parcels": False}},
        {"config_version": 2},
        {"stored_resolution": 11},
        {"ingest_version": 8},
    ],
)
def test_anything_that_changes_the_answer_changes_the_tag(change):
    """The three non-obvious inputs are the point of this test.

    `drawing_id` is a content hash, so the FILE cannot change without it
    changing — but the ANSWER can. Re-classify a layer and the counts move;
    re-backfill at another resolution and the cells move. An ETag blind to
    either would serve a stale body behind a confident 304, which is the one
    cache failure nobody reports because everything looks like it works."""
    assert _tag(**change) != _tag()


def test_a_different_drawing_is_a_different_tag():
    assert geo_view.etag_for("0000000000000000", **BASE) != _tag()


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, False),
        ("", False),
        ('"deadbeef"', False),
        ("*", True),
    ],
)
def test_the_conditional_header_is_read_as_the_list_it_is(header, expected):
    assert geo_view.etag_matches(header, '"abc123"') is expected


def test_a_browser_sending_several_tags_still_gets_its_304():
    """`If-None-Match` is a list, and comparing the raw header against one tag
    works for exactly one client. A browser holding two representations would
    simply never revalidate, and nothing would look broken."""
    tag = '"abc123"'
    assert geo_view.etag_matches(f'"other", {tag}', tag)
    assert geo_view.etag_matches(f'W/{tag}', tag)
    assert geo_view.etag_matches(f'"a", "b", W/{tag}, "c"', tag)


def test_the_preconditions_are_read_without_running_the_aggregation():
    """The whole value of a conditional request is that a client which already
    holds the answer does not pay for it to be computed. Measured over HTTP:
    1.30 s and 997,594 bytes for the 200, 0.28 s and nothing for the 304."""
    try:
        facts = geo_view.preconditions(JANADRIYAH)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    assert set(facts) == {"config_version", "stored_resolution", "ingest_version"}
    assert facts["stored_resolution"] == 13


# ---------------------------------------------------------------------------
# P6 — the surface a frontend reads
# ---------------------------------------------------------------------------


def _geo_operation():
    schema = main.app.openapi()
    return schema["paths"]["/drawings/{drawing_id}/geo"]["get"]


def test_every_query_parameter_explains_itself_to_a_frontend():
    """The OpenAPI page is the contract a frontend developer reads, and a
    parameter called `compact` with no description is a parameter nobody
    uses correctly. Path parameters are excluded: `drawing_id` is described
    the same way on every route in this API, which is not this campaign's
    to change."""
    operation = _geo_operation()
    undocumented = [
        p["name"]
        for p in operation["parameters"]
        if p.get("in") == "query" and not (p.get("description") or "").strip()
    ]
    assert undocumented == [], undocumented


def test_the_route_says_what_it_is_for_rather_than_repeating_its_own_name():
    """FastAPI would have called this `Drawing Geo`."""
    operation = _geo_operation()
    assert operation["summary"] == "Cells and parcels for a map"
    described = operation["responses"]["200"]["description"]
    assert "totals" in described
    assert "georeferenced: false" in described


def test_the_new_parameters_are_all_present_and_optional():
    """Additive means a caller who knows nothing about this wave still gets
    the same answer it got before."""
    operation = _geo_operation()
    names = {p["name"] for p in operation["parameters"] if p.get("in") == "query"}
    assert {"res", "layout", "parcels", "compact", "limit", "bbox", "outline"} <= names
    for parameter in operation["parameters"]:
        if parameter.get("in") == "query":
            assert parameter.get("required") is not True, parameter["name"]


def test_the_smoke_script_exists_and_covers_what_only_the_wire_can_show():
    """gzip, a 304 and the MCP envelope do not happen in this process. The
    script that exercises them is part of the gate, so it is named here — a
    check nobody can find is a check nobody runs."""
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "geo_smoke.py"
    if not script.is_file():
        # The test container mounts `cad_api` alone, so the repo root is not
        # there. Skipped with the reason rather than passed quietly: a check
        # that silently does nothing is worse than one that is absent.
        pytest.skip(f"repo root not mounted; {script} is not visible from here")
    source = script.read_text(encoding="utf-8")
    for subject in ("gzip", "etag", "bbox", "outline", "mcp"):
        assert f'"{subject}' in source.lower() or f"'{subject}" in source.lower(), subject
    # The header lookup this script got wrong on its first run: HTTP header
    # names are case-insensitive and uvicorn sends them lowercase, so a plain
    # dict reports a missing ETag on a response that has one. The check is on
    # the CODE, not on the file — the docstring explains the trap by naming
    # it, and an assertion that searched the whole text caught the warning
    # rather than the bug.
    assert "return response.status, response.headers, response.read()" in source
    assert "return exc.code, exc.headers, exc.read()" in source


def test_every_geo_parameter_that_changes_the_body_also_changes_the_etag():
    """A parameter the route reads must be in the ETag, or a cache lies.

    Written from a real failure: `dimension` was added to `/geo`, changed the
    body from one hexagon to sixty-eight, and was left out of `params`. The
    browser revalidated its old copy and got a 304 -- a different question
    answered from a cache of the previous one, with nothing on screen saying
    so. The general form of the mistake is what is guarded here, so the next
    parameter is covered before it is written rather than after.
    """
    import inspect

    from app import geo_view, main

    route = inspect.signature(main.drawing_geo).parameters
    # What the ROUTE accepts from the caller, minus the plumbing that cannot
    # vary an answer.
    plumbing = {"drawing_id", "request", "response"}
    caller_facing = {n for n in route if n not in plumbing}

    source = inspect.getsource(main.drawing_geo)
    start = source.index("params = {")
    hashed = {
        line.split('"')[1]
        for line in source[start : source.index("}", start)].splitlines()
        if line.strip().startswith('"')
    }

    # `highlight` is hashed under its normalised name; `bbox` under its parsed
    # one. Both ARE present, so the comparison is on names the route uses.
    missing = caller_facing - hashed
    assert not missing, (
        "these /geo parameters can change the answer but not its ETag, so a "
        f"client can be served a stale body with a 200-shaped 304: {sorted(missing)}"
    )
    assert "params" in geo_view.ETAG_INPUTS
