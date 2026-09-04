"""Snapshot the shared database's collection names, before and after.

`roshn-gpt-dev-main` belongs to several teams. This script exists so that
"we only added `autocad_*` collections and touched nothing else" is a claim
that can be checked against a file, rather than trusted.

It reads collection *names* only. No document from a collection outside our
prefix is ever fetched.

    python -m app.baseline before  > docs/MONGO-BASELINE.md
    python -m app.baseline after  >> docs/MONGO-BASELINE.md
    python -m app.baseline diff        # exit 1 if a foreign collection moved
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import mongo

#: Must live on a persisted volume. It was originally written to `/data`,
#: which the compose service does not persist (only `/data/dxf` and
#: `/data/svg` are mounted), so the snapshot vanished with the container that
#: wrote it. The durable record is `docs/MONGO-BASELINE.md` in git; this file
#: is the machine-readable copy used by `diff`.
_STATE = Path(
    os.environ.get("CAD_BASELINE_FILE", "/data/svg/.mongo-baseline.json")
)


def _snapshot() -> list[str]:
    return mongo.list_all_collection_names()


def _write_state(names: list[str]) -> None:
    _STATE.parent.mkdir(parents=True, exist_ok=True)
    _STATE.write_text(
        json.dumps(
            {"captured_at": datetime.now(timezone.utc).isoformat(), "collections": names},
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_state() -> list[str]:
    if not _STATE.exists():
        raise SystemExit(
            "no baseline recorded yet; run `python -m app.baseline before` first"
        )
    return json.loads(_STATE.read_text(encoding="utf-8"))["collections"]


def _render_markdown(title: str, names: list[str]) -> str:
    ours = [n for n in names if n.startswith(mongo.PREFIX)]
    theirs = [n for n in names if not n.startswith(mongo.PREFIX)]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"## {title}",
        "",
        f"- Captured: {stamp}",
        f"- Database: `{mongo.DB_NAME}`",
        f"- Total collections: **{len(names)}**",
        f"- Ours (`{mongo.PREFIX}*`): **{len(ours)}**",
        f"- Other teams': **{len(theirs)}**",
        "",
        f"### Ours (`{mongo.PREFIX}*`)",
        "",
    ]
    lines += [f"- `{n}`" for n in ours] or ["- _(none)_"]
    lines += ["", "### Other teams' — must be byte-identical before and after", ""]
    lines += [f"- `{n}`" for n in theirs] or ["- _(none)_"]
    lines.append("")
    return "\n".join(lines)


#: Printed when the gate fails. It lives here, not in a document, because the
#: person reading it is staring at a failure and will not go looking for docs.
_FOREIGN_GUIDANCE = """This database is shared: 285 of its collections belong to other teams, and
they deploy while we work. A collection that APPEARED is therefore not by
itself proof that this project created it. On 24 Aug 2026 four `rec_*`
collections appeared mid-session and not one of them was ours.

Before treating this as our fault, check the two things that would both have
to be true for it to be:

  1. This repo names it at all:
       grep -rn '<name>' --include='*.py' .
  2. Something reaches the database without going through `mongo.coll()`,
     which raises on any name outside the prefix:
       grep -rn 'get_database()' --include='*.py' cad_api/
     That should return only the lines inside `mongo.py` itself.

If both come back clean, the change belongs to another team, and the honest
action is to re-run `before` to re-baseline -- NOT to loosen this check.

A collection that DISAPPEARED is a different matter and deserves alarm
either way: nothing here deletes collections, ours or anyone's."""


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "before"

    if command == "before":
        names = _snapshot()
        _write_state(names)
        print(_render_markdown("Baseline — BEFORE this project wrote anything", names))
        return 0

    if command == "after":
        names = _snapshot()
        print(_render_markdown("Baseline — AFTER this session", names))
        return 0

    if command == "diff":
        before = set(_read_state())
        after = set(_snapshot())
        added = sorted(after - before)
        removed = sorted(before - after)

        foreign_added = [n for n in added if not n.startswith(mongo.PREFIX)]
        foreign_removed = [n for n in removed if not n.startswith(mongo.PREFIX)]

        print(_render_markdown("Baseline — AFTER this session", sorted(after)))
        print("### Difference\n")
        print(f"- Added: {', '.join(f'`{n}`' for n in added) or '_(none)_'}")
        print(f"- Removed: {', '.join(f'`{n}`' for n in removed) or '_(none)_'}")
        print()

        if foreign_added or foreign_removed:
            print("**FAILED — a collection outside our prefix changed:**\n")
            for n in foreign_added:
                print(f"- appeared: `{n}`")
            for n in foreign_removed:
                print(f"- disappeared: `{n}`")
            print()
            print(_FOREIGN_GUIDANCE)
            return 1

        print(
            "**PASS — every change is inside the `autocad_` prefix. "
            "No other team's collection was created, renamed or removed.**"
        )
        return 0

    print(f"unknown command {command!r}; use before | after | diff", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
