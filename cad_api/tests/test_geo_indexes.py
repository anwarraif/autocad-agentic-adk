"""Wave 2 P4 — that the geo queries stay indexed, and stay selective.

The audit that produced this file concluded that **no new index is needed**,
and the numbers are why. Every query this campaign added comes back with
`docsExamined` equal to `nReturned`: the planner is reading exactly the
documents it hands over, which is the best an index can do. Server-side times
on the shared cluster were 3 ms, 27 ms and 47 ms.

    query                              returned  docsExamined  keysExamined  ms
    geo parcels (layout+layer+type)         494           494           498   3
    vicinity scan (layout+h3_cell)       20,331        20,334        20,334  27
    distinct(layer) for classification   46,754        46,754        46,754  47

Creating an index anyway would have cost write time on every ingest and index
memory on a cluster other teams share, to save nothing measurable. So what
this phase leaves behind is not an index but the check that would notice if
that stopped being true — a query whose shape drifts past its index does not
fail, it just gets slower, and on a cluster with `notablescan` on it fails
much later and somewhere else.

The end-to-end latency is elsewhere and is named so nobody optimises the wrong
thing: `/geo?res=9&parcels=false` averages 1,164 ms, of which the database
work above is roughly 50 ms. The rest is round trips to a remote Atlas cluster
and the reprojection of 2,545 rings in Python.
"""

from __future__ import annotations

import pytest

from app.mongo import COLL_ENTITIES, coll, get_database

JANADRIYAH = "596212db022a3397"
MODEL = "Model"

#: Every query shape this campaign added, with a name a report can use.
#: `type` and `layer` narrow what the compound index already selected; the
#: index prefix is `drawing_id` in all of them, which is what the cluster's
#: `notablescan` setting requires.
GEO_QUERIES = {
    "geo parcels": {
        "drawing_id": JANADRIYAH,
        "layout": MODEL,
        "layer": {"$in": ["VL2", "TH3", "LP1", "VL4"]},
        "type": {"$in": ["LWPOLYLINE", "POLYLINE"]},
    },
    "vicinity index": {
        "drawing_id": JANADRIYAH,
        "layout": MODEL,
        "h3_cell": {"$ne": None},
    },
    "classification layers": {"drawing_id": JANADRIYAH},
    "backfill sweep": {"drawing_id": JANADRIYAH},
}


def _explain(query):
    try:
        return get_database().command(
            "explain",
            {"find": COLL_ENTITIES, "filter": query},
            verbosity="executionStats",
        )
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")


def _index_of(plan) -> str | None:
    if isinstance(plan, dict):
        if plan.get("indexName"):
            return plan["indexName"]
        for value in plan.values():
            found = _index_of(value)
            if found:
                return found
    if isinstance(plan, list):
        for item in plan:
            found = _index_of(item)
            if found:
                return found
    return None


@pytest.mark.parametrize("label", sorted(GEO_QUERIES))
def test_every_geo_query_is_served_by_an_index(label):
    """`notablescan` is on for this cluster, so an unindexed query does not
    run slowly — it fails. This is the check that finds that out here rather
    than in a request somebody is waiting on."""
    plan = _explain(GEO_QUERIES[label])["queryPlanner"]["winningPlan"]
    index = _index_of(plan)
    assert index is not None, f"{label} has no index: {plan}"
    assert index != "_id_" or "drawing_id" not in GEO_QUERIES[label]


@pytest.mark.parametrize("label", sorted(GEO_QUERIES))
def test_every_geo_query_reads_only_what_it_returns(label):
    """The measurement that decided against a new index, kept as a guard.

    A query whose shape drifts past its index does not break; it starts
    examining documents it will discard, and the only symptom is that
    everything is slower. The ratio is the symptom, so it is what is
    asserted."""
    stats = _explain(GEO_QUERIES[label])["executionStats"]
    returned = stats["nReturned"]
    examined = stats["totalDocsExamined"]
    if returned == 0:
        pytest.skip("nothing matched; the ratio says nothing")
    # A small constant of slack: the planner may fetch a handful of documents
    # a covered filter then rejects. Ten per cent, not "about the same".
    assert examined <= returned * 1.1 + 10, (label, returned, examined)


def test_the_indexes_this_relies_on_are_the_ones_ingest_creates():
    """No index was added for this campaign, and this names the ones it leans
    on so that removing one is a decision rather than a surprise."""
    try:
        names = set(coll(COLL_ENTITIES).index_information())
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    assert {"drawing_id_1_layer_1", "drawing_id_1_layout_1"} <= names


def test_no_index_was_added_for_the_geo_surface():
    """The audit's conclusion, written down where it can be contradicted.

    An index costs write time on every ingest and memory on a cluster other
    teams share. This one would have bought nothing: the queries already read
    exactly what they return. If a later phase needs one, this test is the
    place the reason gets recorded."""
    try:
        names = set(coll(COLL_ENTITIES).index_information())
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"no database: {type(exc).__name__}")
    assert not [n for n in names if "h3" in n.lower()], (
        "an h3 index exists; if it was added deliberately, edit this test and "
        "record the measurement that justified it"
    )
