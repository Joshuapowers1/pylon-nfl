"""
PYLON :: streaming
Measures the streaming premium instead of asserting it.

The player model prices replacement level for quarterbacks and tight ends above
the season-long Nth-best player, because a manager who never drafts one can
start whoever has the best matchup each week. That premium was the only input in
the whole engine that was a guess. This module replays it.

The simulation is deliberately pessimistic about the streamer's information. He
picks on matchup, not on outcome: each week he takes whichever unrostered
quarterback faces the softest pass defense, using only ratings built from games
already played. He does not get to see the score first. Anything he gains is
therefore attainable, not hindsight.

--------------------------------------------------------------------------
PYLON . NFL projection and market-edge engine
Joshua Powers . joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl . linkedin.com/in/joshuampowerss . @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import RAW, PROC, OUT, CONF, SCORING, TEAM_ALIASES


def weekly_points(seasons=(2023, 2024, 2025)) -> pd.DataFrame:
    """Fantasy points per player per week, in full PPR, straight from the plays."""
    from . import ingest
    raw = ingest.load_raw(list(seasons))
    d = raw[(raw["season_type"] == "REG") & (raw["special"] != 1)].copy()
    d["posteam"] = d["posteam"].replace(TEAM_ALIASES)
    d["pass_att"] = ((d["qb_dropback"] == 1) & (d["sack"] != 1) & (d["qb_scramble"] != 1)).astype(int)

    key = ["season", "week", "posteam"]
    pas = d[d["passer_player_id"].notna()].groupby(key + ["passer_player_id", "passer_player_name"],
                                                   observed=True).agg(
        pass_yards=("passing_yards", "sum"), pass_td=("pass_touchdown", "sum"),
        ints=("interception", "sum")).reset_index().rename(
        columns={"passer_player_id": "pid", "passer_player_name": "name"})
    rus = d[d["rusher_player_id"].notna()].groupby(key + ["rusher_player_id"], observed=True).agg(
        rush_yards=("rushing_yards", "sum"), rush_td=("rush_touchdown", "sum")).reset_index().rename(
        columns={"rusher_player_id": "pid"})
    m = pas.merge(rus, on=key + ["pid"], how="outer").fillna(0)
    s = SCORING
    m["points"] = (m["pass_yards"] * s["pass_yd"] + m["pass_td"] * s["pass_td"]
                   + m["ints"] * s["interception"] + m["rush_yards"] * s["rush_yd"]
                   + m["rush_td"] * s["rush_td"])
    # Only real starters: a quarterback with fewer than fifteen pass attempts in a
    # week was mopping up, and nobody streams him.
    att = d[d["passer_player_id"].notna()].groupby(key + ["passer_player_id"],
                                                   observed=True)["pass_att"].sum().reset_index()
    att = att.rename(columns={"passer_player_id": "pid", "pass_att": "att"})
    m = m.merge(att, on=key + ["pid"], how="left").fillna({"att": 0})
    return m[m["att"] >= 15]


def measure(seasons=(2023, 2024, 2025), teams: int = 12, start_week: int = 4) -> dict:
    """Compare a matchup-streaming manager against the season-long QB{rep}."""
    from . import ratings as R
    wp = weekly_points(seasons)
    plays = pd.read_parquet(PROC / "plays_clean.parquet")
    rep_rank = round(14 * teams / 12)

    per_season = []
    for season in seasons:
        w = wp[wp["season"] == season]
        weeks = sorted(w["week"].unique())
        # Season-long finish, which is what naive replacement level uses.
        season_ppg = (w.groupby("pid")["points"].sum() / w.groupby("pid")["week"].nunique()) \
            .sort_values(ascending=False)
        if len(season_ppg) < rep_rank:
            continue
        naive = float(season_ppg.iloc[rep_rank - 1])

        got = []
        for wk in weeks:
            if wk < start_week:
                continue
            # Ratings from games already played: the streamer cannot see ahead.
            prior = plays[(plays["season"] == season) & (plays["week"] < wk)]
            if prior["game_id"].nunique() < 32:
                continue
            teams_in = sorted(set(prior["posteam"]) | set(prior["defteam"]))
            sub = prior[prior["is_pass"] == 1]
            X, y = R._design(sub, teams_in)
            from sklearn.linear_model import Ridge
            mdl = Ridge(alpha=220.0, fit_intercept=True, solver="cholesky").fit(X, y)
            k = len(teams_in)
            dfn = pd.Series(mdl.coef_[k:2 * k], index=teams_in)

            # Who is unrostered: everyone outside the top rep_rank on season-to-date form.
            std = w[w["week"] < wk]
            if not len(std):
                continue
            form = (std.groupby("pid")["points"].sum() / std.groupby("pid")["week"].nunique()) \
                .sort_values(ascending=False)
            free = set(form.index[rep_rank:])
            this = w[(w["week"] == wk) & (w["pid"].isin(free))].copy()
            if not len(this):
                continue
            # Pick on matchup only: softest pass defense faced.
            this["opp_soft"] = this["posteam"].map(
                lambda t: _opp_def(season, t, wk, dfn))
            pick = this.sort_values("opp_soft", ascending=False).iloc[0]
            got.append(float(pick["points"]))

        if got:
            per_season.append({"season": season, "streamed_ppg": float(np.mean(got)),
                               "naive_ppg": naive, "weeks": len(got)})

    if not per_season:
        return {}
    prem = float(np.mean([p["streamed_ppg"] - p["naive_ppg"] for p in per_season]))
    out = {"position": "QB", "teams": teams, "replacement_rank": rep_rank,
           "premium_ppg": round(prem, 3), "by_season": per_season,
           "method": ("Each week the streamer takes the unrostered starter facing the softest "
                      "pass defense, rated only on games already played. Compared against the "
                      "season-long QB at replacement rank.")}
    with open(CONF / "streaming_premium.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


_SCHED_CACHE: dict = {}


def _opponents(season: int) -> dict:
    """(week, team) -> opponent, from the schedule. Knowing who a team plays is
    not lookahead: the fixture list is published in May."""
    if season not in _SCHED_CACHE:
        g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
        g = g[(g["season"] == season) & (g["game_type"] == "REG")]
        m = {}
        for _, r in g.iterrows():
            h = TEAM_ALIASES.get(r["home_team"], r["home_team"])
            a = TEAM_ALIASES.get(r["away_team"], r["away_team"])
            m[(int(r["week"]), h)] = a
            m[(int(r["week"]), a)] = h
        _SCHED_CACHE[season] = m
    return _SCHED_CACHE[season]


def _opp_def(season: int, team: str, week: int, dfn: pd.Series) -> float:
    """Rating of the defense this team faces this week. Higher means softer,
    because the coefficient is EPA allowed per play."""
    opp = _opponents(season).get((int(week), team))
    if opp is None:
        return float("-inf")
    return float(dfn.get(opp, dfn.mean()))
