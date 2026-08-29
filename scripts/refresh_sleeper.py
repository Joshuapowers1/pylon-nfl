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

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pylon.config import TEAM_ALIASES  # noqa: E402

DEST = Path(__file__).resolve().parents[1] / "config" / "sleeper_ids.json"
URL = "https://api.sleeper.app/v1/players/nfl"
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}
FA_SEARCH_RANK_CUTOFF = 250


def trim(raw: dict) -> dict:
    out: dict[str, list] = {}
    for pid, p in raw.items():
        pos = p.get("position") or ""
        if pos not in FANTASY_POSITIONS:
            continue
        name = p.get("full_name") or p.get("last_name") or ""
        if not name:
            continue
        code = str(p.get("team") or "").upper()
        team = TEAM_ALIASES.get(code, code) if code else "FA"
        if team == "FA":
            rank = p.get("search_rank")
            if rank is None or rank > FA_SEARCH_RANK_CUTOFF:
                continue
        out[str(pid)] = [name, pos, team]
    return out


def main() -> int:
    with urllib.request.urlopen(URL, timeout=180) as r:
        raw = json.load(r)
    if not isinstance(raw, dict) or len(raw) < 1000:
        print("refused: response does not look like the player dictionary", file=sys.stderr)
        return 1
    table = trim(raw)
    if len(table) < 600:
        print(f"refused: trimmed to {len(table)} players, which is too few", file=sys.stderr)
        return 1
    DEST.write_text(json.dumps(table, separators=(",", ":"), sort_keys=True))
    print(f"wrote {DEST.name}: {len(table)} of {len(raw)} players, "
          f"{DEST.stat().st_size / 1024:.0f}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
