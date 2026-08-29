"""Roster import support for the fantasy platforms people actually use.

Sleeper hands out numeric player ids and nothing else on a roster call, so a
lookup table has to travel with the site. ESPN returns full names inline, and a
pasted roster is text, so both of those resolve through the same name
normaliser the ADP join already uses. The table here is therefore Sleeper-only.

config/sleeper_ids.json is the trimmed table, refreshed by
scripts/refresh_sleeper.py and committed. The source it comes from is a 15MB
dictionary of every player Sleeper has ever seen, which does not belong in a
repository and does not belong in a browser either.
"""
from __future__ import annotations

import json

from .config import CONF
from .draft import _norm_name

TABLE = CONF / "sleeper_ids.json"


def sleeper_table() -> dict:
    """sleeper_id -> [name, position, team], for rosterable players only."""
    return json.loads(TABLE.read_text())


def build_index(players_df) -> dict:
    """Everything a browser needs to turn a foreign roster into our players.

    ``sleeper`` maps Sleeper's ids onto plain names. ``keys`` maps our
    normalised name key onto the exact ``player`` string used everywhere else
    in the bundle, so the front end resolves a pasted line or an ESPN name with
    one lookup instead of a scan.
    """
    keys = {_norm_name(r["player"]): r["player"] for _, r in players_df.iterrows()}
    return {"sleeper": sleeper_table(), "keys": keys}
