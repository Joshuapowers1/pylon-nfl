#!/usr/bin/env python3
"""Rebuild config/sleeper_ids.json from Sleeper's player dictionary.

Sleeper asks that the full dictionary be pulled at most once a day, because it
is roughly 15MB of every player the service has ever recorded. The weekly
rebuild is well inside that. Only the trimmed result is written to disk: a
roster import has to resolve someone who is on an NFL team, plus the handful of
recent cuts still sitting on a bench, which is about 1,100 names rather than
4,300, and it ships inside a single-file web page.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pylon.config import TEAM_ALIASES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "config" / "sleeper_ids.json"
PROJECTIONS = ROOT / "out" / "player_projections.csv"
URL = "https://api.sleeper.app/v1/players/nfl"
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}
FA_SEARCH_RANK_CUTOFF = 250
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm(name: str) -> str:
    """The same normaliser the browser uses, so both ends agree on a match."""
    cleaned = re.sub(r"[-]", " ", re.sub(r"[.\'\u2019]", "", str(name).lower().strip()))
    return " ".join(w for w in cleaned.split() if w and w not in SUFFIXES)


def projected_positions() -> dict[str, str]:
    """Every player the model has an opinion about.

    Without this the table is filtered purely on Sleeper's own popularity rank,
    which is a proxy for the wrong thing. Run in late August it keeps almost
    everyone, because rosters are still full. Run again after final cuts it
    quietly drops every projected player who is now a free agent, and drops
    anyone Sleeper files at a defensive position regardless of team. The import
    then reports those players as unmatched and the page looks broken for a
    reason nobody can see. If the model projects him, he stays in the table.
    """
    if not PROJECTIONS.exists():
        return {}
    with PROJECTIONS.open() as fh:
        return {norm(r["player"]): (r.get("position") or "").upper()
                for r in csv.DictReader(fh) if r.get("player")}


def coverage(table: dict, wanted: dict[str, str]) -> int:
    have = {norm(v[0]) for v in table.values() if v}
    return len(set(wanted) & have)


def trim(raw: dict, wanted: dict[str, str]) -> dict:
    out: dict[str, list] = {}
    for pid, p in raw.items():
        name = p.get("full_name") or p.get("last_name") or ""
        if not name:
            continue
        pos = p.get("position") or ""
        code = str(p.get("team") or "").upper()
        team = TEAM_ALIASES.get(code, code) if code else "FA"
        key = norm(name)
        keep_for_model = key in wanted

        # Sleeper files a two-way player at his defensive position. Travis
        # Hunter comes back as a defensive back, and storing that would put him
        # in no fantasy slot at all on import. Where the model has an opinion
        # about a player, the model's position is the one that governs, because
        # that is the position the projection was made for.
        if keep_for_model and wanted[key] in FANTASY_POSITIONS:
            pos = wanted[key]

        if pos not in FANTASY_POSITIONS:
            continue
        if team == "FA" and not keep_for_model:
            rank = p.get("search_rank")
            if rank is None or rank > FA_SEARCH_RANK_CUTOFF:
                continue
        out[str(pid)] = [name, pos, team]

    # Names are not unique in Sleeper's dictionary. There is a linebacker
    # called Josh Allen, and promoting him to quarterback because he shares a
    # name with a projected one is how an import silently resolves the wrong
    # human. Where a name is shared and at least one of them genuinely plays a
    # fantasy position, the others are dropped rather than rewritten.
    genuine: dict[str, bool] = {}
    for pid, (name, pos, _team) in out.items():
        key = norm(name)
        genuine[key] = genuine.get(key, False) or (
            (raw.get(pid, {}).get("position") or "") in FANTASY_POSITIONS)
    return {
        pid: row for pid, row in out.items()
        if (raw.get(pid, {}).get("position") or "") in FANTASY_POSITIONS
        or not genuine.get(norm(row[0]), False)
    }


def main() -> int:
    with urllib.request.urlopen(URL, timeout=180) as r:
        raw = json.load(r)
    if not isinstance(raw, dict) or len(raw) < 1000:
        print("refused: response does not look like the player dictionary", file=sys.stderr)
        return 1
    wanted = projected_positions()
    table = trim(raw, wanted)
    if len(table) < 600:
        print(f"refused: trimmed to {len(table)} players, which is too few", file=sys.stderr)
        return 1

    # A weekly job that can quietly make the roster import worse is worse than
    # a weekly job that stops. Compare against what is already on disk and
    # refuse to regress, rather than discovering it in October.
    if wanted and DEST.exists():
        before = coverage(json.loads(DEST.read_text()), wanted)
        after = coverage(table, wanted)
        if after < before:
            print(f"refused: would cover {after} projected players, down from "
                  f"{before}. Keeping the existing table.", file=sys.stderr)
            return 1
        print(f"projected players resolvable: {before} -> {after} of {len(wanted)}")
    DEST.write_text(json.dumps(table, separators=(",", ":"), sort_keys=True))
    print(f"wrote {DEST.name}: {len(table)} of {len(raw)} players, "
          f"{DEST.stat().st_size / 1024:.0f}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
