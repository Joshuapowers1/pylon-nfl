"""Roster import: the lookup table and the name keys the browser resolves on.

The front end matches a foreign roster by normalising a name and looking it up
in a dictionary built here. If the two normalisers ever disagree, every import
silently returns an empty roster, so the shape of that table is worth pinning.
"""
import json

import pandas as pd
import pytest

from pylon.draft import _norm_name
from pylon.platforms import build_index, sleeper_table
from pylon.config import OUT, TEAMS

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}


@pytest.fixture(scope="module")
def table():
    return sleeper_table()


@pytest.fixture(scope="module")
def players_df():
    return pd.read_csv(OUT / "player_projections.csv")


def test_table_is_a_sane_size(table):
    # Roughly one rosterable player per NFL roster spot that matters, plus the
    # recently cut. Far more than this means the free-agent filter broke and
    # the browser is being shipped Sleeper's entire history.
    assert 600 < len(table) < 2200


def test_every_entry_is_name_position_team(table):
    for pid, row in table.items():
        assert pid.isdigit() or pid.isalnum()
        name, pos, team = row
        assert name and isinstance(name, str)
        assert pos in FANTASY_POSITIONS
        assert team == "FA" or team in TEAMS


def test_teams_are_normalised(table):
    # nflverse and Sleeper disagree on a handful of codes. If an alias slipped
    # through, a player would show a team the rest of the site has no logo for.
    codes = {row[2] for row in table.values()} - {"FA"}
    assert codes <= set(TEAMS)


def test_well_known_players_are_present(table):
    names = {row[0] for row in table.values()}
    for who in ["Josh Allen", "Bijan Robinson", "Brock Bowers"]:
        assert who in names


def test_index_keys_resolve_every_projected_player(players_df):
    idx = build_index(players_df)
    assert len(idx["keys"]) == len(players_df)
    for _, r in players_df.iterrows():
        assert idx["keys"][_norm_name(r["player"])] == r["player"]


def test_suffixes_do_not_break_the_join(players_df):
    # Sleeper writes "Tyrone Tracy Jr." and ESPN writes "Tyrone Tracy". Both
    # have to land on the same row or an import loses the player.
    idx = build_index(players_df)
    assert _norm_name("Tyrone Tracy Jr.") == _norm_name("Tyrone Tracy")
    assert _norm_name("Ja'Marr Chase") == "jamarr chase"
    assert _norm_name("Amon-Ra St. Brown") == "amon ra st brown"


def test_index_is_small_enough_to_ship(players_df):
    # It is embedded in a single-file page, so it has a budget.
    payload = json.dumps(build_index(players_df), separators=(",", ":"))
    assert len(payload) < 90_000
