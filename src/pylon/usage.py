"""
PYLON :: usage
Extracts every player's share of their own team's opportunity from play-by-play.

Shares, not counts. A running back who got 250 carries on a team that ran 500
times is a very different player from one who got 250 on a team that ran 350,
and only the share survives a change of team, coordinator or game script. The
whole player model is built on shares and then multiplied back up by the team
volume projections from adjustments.project_context().

One detail here does more work than everything else combined: the denominator
is the team's opportunity IN THE GAMES THE PLAYER ACTUALLY PLAYED, not over the
full season. Dividing by the season total silently punishes every player who
missed time -- a receiver who misses six games looks like he lost a third of his
role when in fact his role never changed. Get this wrong and the model fades
every player coming back from injury, which is exactly the population where
draft-day value lives.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowerss · @joshuampowers
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import RAW, PROC, TEAM_ALIASES

RZ = 20          # yards from the end zone that count as red zone
LATE_WEEK = 13   # weeks 13-18 of the most recent season


def _norm(s):
    return s.replace(TEAM_ALIASES)


def _prep(pbp_all, seasons, week_min):
    d = pbp_all[(pbp_all["season"].isin(seasons)) & (pbp_all["season_type"] == "REG")
                & (pbp_all["special"] != 1) & (pbp_all["week"] >= week_min)].copy()
    d["posteam"] = _norm(d["posteam"])
    d["is_pass"] = (d["qb_dropback"] == 1).astype(int)
    d["pass_att"] = ((d["is_pass"] == 1) & (d["sack"] != 1) & (d["qb_scramble"] != 1)).astype(int)
    d["is_rush"] = ((d["rush"] == 1) & (d["qb_dropback"] != 1)).astype(int)
    d["rz"] = (d["yardline_100"] <= RZ).astype(int)
    d["rz_pass"] = ((d["rz"] == 1) & (d["pass_att"] == 1)).astype(int)
    d["rz_rush"] = ((d["rz"] == 1) & (d["is_rush"] == 1)).astype(int)
    return d


def _team_game_totals(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["season", "posteam", "game_id"], observed=True).agg(
        g_pass_att=("pass_att", "sum"), g_rush_att=("is_rush", "sum"),
        g_rz_pass=("rz_pass", "sum"), g_rz_rush=("rz_rush", "sum"),
        g_dropbacks=("is_pass", "sum"),
    ).reset_index()
    td = d[d["touchdown"] == 1].groupby(["season", "posteam", "game_id"], observed=True).agg(
        g_pass_td=("pass_touchdown", "sum"), g_rush_td=("rush_touchdown", "sum")).reset_index()
    return g.merge(td, on=["season", "posteam", "game_id"], how="left").fillna(0)


def _played_denominators(d, tg, id_cols) -> pd.DataFrame:
    """For each player, sum the team's opportunity over only the games in which
    that player recorded at least one snap of measurable involvement."""
    frames = []
    for col in id_cols:
        sub = d[d[col].notna()][["season", "posteam", "game_id", col]].rename(columns={col: "player_id"})
        frames.append(sub)
    appear = pd.concat(frames, ignore_index=True).drop_duplicates()
    j = appear.merge(tg, on=["season", "posteam", "game_id"], how="left")
    den = j.groupby(["season", "posteam", "player_id"], observed=True).agg(
        team_pass_att=("g_pass_att", "sum"), team_rush_att=("g_rush_att", "sum"),
        team_rz_pass_att=("g_rz_pass", "sum"), team_rz_rush_att=("g_rz_rush", "sum"),
        team_dropbacks=("g_dropbacks", "sum"),
        team_pass_td=("g_pass_td", "sum"), team_rush_td=("g_rush_td", "sum"),
        games_played=("game_id", "nunique"),
    ).reset_index()
    return den


def player_usage(pbp_all: pd.DataFrame, seasons=(2024, 2025), week_min: int = 0) -> pd.DataFrame:
    d = _prep(pbp_all, seasons, week_min)
    tg = _team_game_totals(d)

    rec = d[(d["pass_att"] == 1) & (d["receiver_player_id"].notna())]
    r = rec.groupby(["season", "posteam", "receiver_player_id", "receiver_player_name"], observed=True).agg(
        targets=("pass_att", "sum"), receptions=("complete_pass", "sum"),
        rec_yards=("receiving_yards", "sum"), rec_td=("pass_touchdown", "sum"),
        air_yards=("air_yards", "sum"), rz_targets=("rz", "sum"),
    ).reset_index().rename(columns={"receiver_player_id": "player_id", "receiver_player_name": "player_name"})

    rush = d[(d["is_rush"] == 1) & (d["rusher_player_id"].notna())]
    ru = rush.groupby(["season", "posteam", "rusher_player_id", "rusher_player_name"], observed=True).agg(
        rush_att=("is_rush", "sum"), rush_yards=("rushing_yards", "sum"),
        rush_td=("rush_touchdown", "sum"), rz_rush=("rz", "sum"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id", "rusher_player_name": "player_name"})

    pas = d[(d["is_pass"] == 1) & (d["passer_player_id"].notna())]
    pa = pas.groupby(["season", "posteam", "passer_player_id", "passer_player_name"], observed=True).agg(
        dropbacks=("is_pass", "sum"), pass_att_p=("pass_att", "sum"),
        completions=("complete_pass", "sum"), pass_yards=("passing_yards", "sum"),
        pass_td=("pass_touchdown", "sum"), ints=("interception", "sum"),
        sacks_taken=("sack", "sum"),
    ).reset_index().rename(columns={"passer_player_id": "player_id", "passer_player_name": "player_name"})

    # A scramble is a dropback, so it is excluded from `is_rush`; and nflverse
    # credits it to rusher_player_id, not passer_player_id, so it is also missing
    # from the passing group. It therefore falls through BOTH buckets and
    # vanishes -- roughly 40% of a mobile quarterback's carries and 70% of his
    # rushing yards. Pick it up explicitly.
    scr = d[(d["qb_scramble"] == 1) & (d["rusher_player_id"].notna())]
    sc = scr.groupby(["season", "posteam", "rusher_player_id"], observed=True).agg(
        scrambles=("qb_scramble", "sum"), scramble_yards=("rushing_yards", "sum"),
        scramble_td=("rush_touchdown", "sum"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id"})
    pa = pa.merge(sc, on=["season", "posteam", "player_id"], how="left").fillna(
        {"scrambles": 0, "scramble_yards": 0, "scramble_td": 0})

    u = r.merge(ru, on=["season", "posteam", "player_id", "player_name"], how="outer") \
         .merge(pa, on=["season", "posteam", "player_id", "player_name"], how="outer")

    den = _played_denominators(d, tg, ["receiver_player_id", "rusher_player_id", "passer_player_id"])
    u = u.merge(den, on=["season", "posteam", "player_id"], how="left").fillna(0)

    u["target_share"] = u["targets"] / u["team_pass_att"].replace(0, np.nan)
    u["rush_share"] = u["rush_att"] / u["team_rush_att"].replace(0, np.nan)
    u["rz_target_share"] = u["rz_targets"] / u["team_rz_pass_att"].replace(0, np.nan)
    u["rz_rush_share"] = u["rz_rush"] / u["team_rz_rush_att"].replace(0, np.nan)
    u["dropback_share"] = u["dropbacks"] / u["team_dropbacks"].replace(0, np.nan)
    u["catch_rate"] = u["receptions"] / u["targets"].replace(0, np.nan)
    u["yards_per_target"] = u["rec_yards"] / u["targets"].replace(0, np.nan)
    u["ypc"] = u["rush_yards"] / u["rush_att"].replace(0, np.nan)
    u["ypa"] = u["pass_yards"] / u["pass_att_p"].replace(0, np.nan)
    u["td_rate_pass"] = u["pass_td"] / u["pass_att_p"].replace(0, np.nan)
    u["int_rate"] = u["ints"] / u["pass_att_p"].replace(0, np.nan)
    return u.fillna(0)


def build():
    """Two usage tables, not one.

    The season-long table answers "what was this player's role in 2025". The
    late-season table answers "what was his role by December", which is a
    different and usually more relevant question. A rookie who took over a
    backfield in week 11 has a full-season carry share that describes a job he
    no longer has. Blending the two is how the model learns role trajectory
    instead of averaging it away."""
    from . import ingest
    raw = ingest.load_raw([2024, 2025])
    u = player_usage(raw)
    u.to_csv(PROC / "player_usage.csv", index=False)
    late = player_usage(raw[raw["season"] == 2025], seasons=(2025,), week_min=LATE_WEEK)
    late.to_csv(PROC / "player_usage_late.csv", index=False)
    return u
