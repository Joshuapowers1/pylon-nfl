"""
PYLON :: ingest
Loads nflverse play-by-play and turns it into the two tables the rest of the
engine runs on:

  plays_clean   one row per meaningful scrimmage play, 2023-2025
  team_context  one row per team-season: pace, pass rate, drive efficiency

Design note: everything downstream is opponent-adjusted, so this layer stays
deliberately dumb. Its only job is filtering out plays that do not carry
predictive signal (garbage time, special teams, penalties that wiped the play).

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowers · @joshuampowers
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .config import RAW, PROC, TRAIN_SEASONS, WP_FLOOR, WP_CEIL, TEAM_ALIASES

USE_COLS = [
    "season","week","game_id","posteam","defteam","home_team","away_team","posteam_type",
    "epa","wp","play_type","pass","rush","qb_dropback","success","yards_gained",
    "season_type","xpass","pass_oe","air_yards","yardline_100","score_differential",
    "receiver_player_id","receiver_player_name","rusher_player_id","rusher_player_name",
    "passer_player_id","passer_player_name","touchdown","pass_touchdown","rush_touchdown",
    "complete_pass","interception","fumble_lost","sack","special","penalty","div_game",
    "drive","fixed_drive_result","game_seconds_remaining","qb_scramble",
    "passing_yards","receiving_yards","rushing_yards",
]


def _norm_team(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_ALIASES)


def load_raw(seasons=TRAIN_SEASONS) -> pd.DataFrame:
    frames = []
    for yr in seasons:
        df = pd.read_parquet(RAW / f"pbp_{yr}.parquet", columns=USE_COLS)
        frames.append(df)
    pbp = pd.concat(frames, ignore_index=True)
    for c in ("posteam", "defteam", "home_team", "away_team"):
        pbp[c] = _norm_team(pbp[c])
    return pbp


def clean_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-season, non-special-teams scrimmage plays with a valid EPA,
    inside a competitive win-probability band."""
    m = (
        (pbp["season_type"] == "REG")
        & (pbp["special"] != 1)
        & (pbp["epa"].notna())
        & (pbp["posteam"].notna())
        & (pbp["defteam"].notna())
        & ((pbp["pass"] == 1) | (pbp["rush"] == 1))
        & (pbp["wp"].between(WP_FLOOR, WP_CEIL))
    )
    out = pbp.loc[m].copy()
    # A dropback that ends in a scramble is still a passing-game play call.
    out["is_pass"] = (out["qb_dropback"] == 1).astype(int)
    out["is_rush"] = ((out["rush"] == 1) & (out["qb_dropback"] != 1)).astype(int)
    out = out[(out["is_pass"] == 1) | (out["is_rush"] == 1)]
    out["home_off"] = (out["posteam_type"] == "home").astype(int)
    return out.reset_index(drop=True)


def team_context(pbp_all: pd.DataFrame) -> pd.DataFrame:
    """Pace and play-mix, computed on ALL regular-season plays (not the
    competitive-band subset) because volume projections need the full sample."""
    m = (
        (pbp_all["season_type"] == "REG")
        & (pbp_all["special"] != 1)
        & (pbp_all["posteam"].notna())
        & ((pbp_all["pass"] == 1) | (pbp_all["rush"] == 1))
    )
    d = pbp_all.loc[m].copy()
    d["is_pass"] = (d["qb_dropback"] == 1).astype(int)
    d["is_rush"] = ((d["rush"] == 1) & (d["qb_dropback"] != 1)).astype(int)
    # A sack is a lost dropback, not a pass attempt. Split them for volume work.
    d["pass_att"] = ((d["is_pass"] == 1) & (d["sack"] != 1) & (d["qb_scramble"] != 1)).astype(int)
    d["scramble"] = ((d["is_pass"] == 1) & (d["qb_scramble"] == 1)).astype(int)
    d["sack_n"] = (d["sack"] == 1).astype(int)

    g = d.groupby(["season", "posteam"], observed=True)
    ctx = g.agg(
        games=("game_id", "nunique"),
        plays=("epa", "size"),
        dropbacks=("is_pass", "sum"),
        pass_att=("pass_att", "sum"),
        scrambles=("scramble", "sum"),
        sacks=("sack_n", "sum"),
        rush_att=("is_rush", "sum"),
        pass_oe=("pass_oe", "mean"),
        epa_play=("epa", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})

    ctx["plays_per_game"] = ctx["plays"] / ctx["games"]
    ctx["pass_att_per_game"] = ctx["pass_att"] / ctx["games"]
    ctx["rush_att_per_game"] = ctx["rush_att"] / ctx["games"]
    ctx["dropback_rate"] = ctx["dropbacks"] / ctx["plays"]

    # Offensive touchdowns per game -- the denominator for the TD-share model.
    td = d[d["touchdown"] == 1].groupby(["season", "posteam"], observed=True).agg(
        pass_td=("pass_touchdown", "sum"), rush_td=("rush_touchdown", "sum")
    ).reset_index().rename(columns={"posteam": "team"})
    ctx = ctx.merge(td, on=["season", "team"], how="left").fillna({"pass_td": 0, "rush_td": 0})
    ctx["off_td_per_game"] = (ctx["pass_td"] + ctx["rush_td"]) / ctx["games"]
    return ctx


def build(seasons=TRAIN_SEASONS):
    raw = load_raw(seasons)
    plays = clean_plays(raw)
    ctx = team_context(raw)
    plays.to_parquet(PROC / "plays_clean.parquet", index=False)
    ctx.to_csv(PROC / "team_context.csv", index=False)
    return plays, ctx, raw


if __name__ == "__main__":
    p, c, _ = build()
    print(f"plays_clean: {len(p):,} rows | team_context: {len(c)} team-seasons")
    print(c.sort_values("plays_per_game", ascending=False).head(5)[["season","team","plays_per_game","pass_att_per_game"]])
