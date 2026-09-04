"""Build a Drawing Dossier from what the store already holds.

The ENGINE lives here, inside `app`, and the command line over it lives in
`scripts/dossier_backfill.py`. That split is not tidiness: `cad_api/Dockerfile`
copies `app` and nothing else, and the ingest hook runs INSIDE the container.
An engine kept in `scripts/` is an engine the running service cannot reach, so
a drawing arriving in production would have got a Dossier missing the parts
this module computes -- which is precisely the "not computed" answer the whole
campaign exists to stop giving.

Idempotent by construction: `drawing_id` is a content hash, so the Dossier is
keyed by the FILE. Re-running overwrites in place, and a new version of a
drawing arrives as a new id with a Dossier of its own.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

from . import dossier as dossier_mod
from .mongo import COLL_DOSSIERS, COLL_DRAWINGS, COLL_ENTITIES, coll

#: Fields the lanes read. Fetched explicitly rather than whole documents: an
#: entity carries ring vertex arrays that no profiler looks at, and 46,754 of
#: them is the difference between a query that fits in memory and one that
#: does not.
PROJECTION: dict[str, int] = {
    "handle": 1,
    "type": 1,
    "layer": 1,
    "layout": 1,
    "block_name": 1,
    "text": 1,
    "length": 1,
    "area": 1,
    "bbox": 1,
    "bbox_centre": 1,
    "ring_status": 1,
    "shape_key": 1,
    "anchor_point": 1,
    "polygon_centroid": 1,
    "attribs": 1,
}

#: How many labels per annotation layer are resolved to their containing
#: parcel. The Dossier needs the ROUTING fact -- "numbers on this layer name
#: parcels on those layers" -- not a containment answer for every label:
#: `get_entity` already answers that on demand, and doing it here would be
#: point-in-polygon over thousands of candidates per label on every backfill.
#: The sample size travels with the finding, because a sampled fact that does
#: not admit it was sampled is exactly the kind of confident half-truth this
#: campaign exists to end.
CONTAINMENT_SAMPLE = 5


def _drawings(drawing_id: str | None) -> list[dict[str, Any]]:
    query = {"_id": drawing_id} if drawing_id else {}
    return list(coll(COLL_DRAWINGS).find(query))


def _rows_by_bucket(drawing_id: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Every entity of one drawing, grouped by (layer, layout).

    `drawing_id` is the filter prefix on purpose and not by habit: this
    cluster runs with `notablescan`, so a query without an indexed plan FAILS
    rather than merely running slowly.
    """
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    cursor = coll(COLL_ENTITIES).find({"drawing_id": drawing_id}, PROJECTION)
    for row in cursor:
        key = (str(row.get("layer") or ""), str(row.get("layout") or ""))
        buckets.setdefault(key, []).append(row)
    return buckets


def _units_by_layout(drawing: dict[str, Any]) -> dict[str, Any]:
    """Units per layout, taken from the drawing and never invented.

    `$INSUNITS` describes model space. Stamping it on a paper layout produced
    this project's worst published figure, so a layout that does not declare
    its own units gets `None` and a reason instead of the header's answer
    (rule G2).
    """
    from app import store  # noqa: PLC0415 -- deferred; store imports ezdxf

    out: dict[str, Any] = {}
    for entry in drawing.get("layouts") or []:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if not name:
            continue
        try:
            out[str(name)] = store._unit_names(dict(drawing), str(name))
        except Exception as exc:  # noqa: BLE001 -- reported, never guessed
            out[str(name)] = {"name": None, "why_no_unit": f"could not be read: {exc}"}
    return out


def _containment_sample(
    drawing_id: str, buckets: dict[tuple[str, str], list[dict[str, Any]]]
) -> dict[str, Any]:
    """Which region layers the labels of each annotation layer sit inside.

    One layer-level fact per annotation layer, from a sample. It answers the
    routing question -- "a question naming one of these numbers is a question
    about a parcel on THOSE layers" -- which is what an agent needs in order
    to reach for the right tool. It is not, and does not pretend to be, a
    containment answer for every label.
    """
    from app import store_spatial  # noqa: PLC0415 -- deferred, Mongo-backed

    out: dict[str, Any] = {}
    for (layer, layout), rows in buckets.items():
        texts = [r for r in rows if r.get("text")]
        if not texts:
            continue
        # Spread the sample across the layer instead of taking the first N.
        # The first N rows sit wherever the drafter happened to start, so five
        # of them describe one corner of the site and call it the layer.
        step = max(1, len(texts) // CONTAINMENT_SAMPLE)
        sample = texts[::step][:CONTAINMENT_SAMPLE]

        # Two different questions, kept apart. `nearest` is the SMALLEST
        # container by area, which is the parcel the label actually names --
        # `parcels_containing` returns nested containers smallest-first for
        # exactly this reason. `all_containers` is every ring the label falls
        # inside, which on a master plan means the district and the site
        # boundary too. Pooling them produces the true and useless sentence
        # "plot numbers sit inside the site boundary", and an agent routed by
        # that would answer a question about a 300 m2 plot with the area of
        # the estate.
        nearest: dict[str, int] = {}
        containers: dict[str, int] = {}
        resolved = 0
        for row in sample:
            try:
                found = store_spatial.parcels_containing(
                    drawing_id, str(row.get("handle")), layout=layout, limit=5
                )
            except Exception:  # noqa: BLE001 -- a label that cannot be placed is a fact
                continue
            rows_in = found.get("contained_by") or []
            if rows_in:
                resolved += 1
                first = str(rows_in[0].get("layer") or "")
                if first:
                    nearest[first] = nearest.get(first, 0) + 1
            for container in rows_in:
                name = str(container.get("layer") or "")
                if name:
                    containers[name] = containers.get(name, 0) + 1
        if not containers:
            continue
        out[f"{layer}::{layout}"] = {
            "layer": layer,
            "layout": layout,
            "sampled": len(sample),
            "of_labels": len(texts),
            "resolved": resolved,
            "names_features_on": dict(sorted(nearest.items(), key=lambda kv: -kv[1])),
            "also_inside": dict(sorted(containers.items(), key=lambda kv: -kv[1])),
            "basis": (
                f"{len(sample)} of {len(texts)} labels on this layer, taken at "
                f"every {step}th row so the sample spans the layer, were tested "
                "point-in-polygon against the closed rings of this layout. "
                "`names_features_on` is the SMALLEST container of each label -- "
                "the feature it labels. `also_inside` adds every larger ring it "
                "falls within, which on a site plan includes the district and "
                "the site boundary; those are not what the label names. This is "
                "a SAMPLE: it says which layers these labels name, not where "
                "every label sits. Ask get_entity for one label's own container."
            ),
        }
    return out


def build_for(drawing: dict[str, Any]) -> dict[str, Any]:
    drawing_id = str(drawing.get("_id"))
    buckets = _rows_by_bucket(drawing_id)
    doc = dossier_mod.build_dossier(
        drawing,
        buckets,
        entity_total=drawing.get("entity_count"),
        units_by_layout=_units_by_layout(drawing),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
    sample = _containment_sample(drawing_id, buckets)
    if sample:
        doc["label_containment_sample"] = list(sample.values())
    return doc


