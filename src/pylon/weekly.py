"""
PYLON :: weekly
Player projections for a single week, which is a different problem from a season
projection divided by seventeen.

Three things change week to week and none of them survive an annual average.

GAME SCRIPT. A team that is going to trail runs less. Fitting team attempts on
the closing spread across 1,632 team-games shows the effect is real but very
lopsided: rush attempts move -0.25 per point of spread (r = -0.21) while pass
attempts move only +0.07 (r = +0.06). Favourites do not throw less, they run
more, and underdogs do not throw much more, they simply get fewer plays. So a
running back's floor is far more matchup-dependent than a receiver's, which is
the opposite of how most weekly advice is written.

OPPONENT. A season projection faces a league-average defense every week. A real
one faces the team across the line, and the opponent's pass and rush defense are
separately relevant depending on what the player does.

AVAILABILITY. A season projection spreads expected missed games across all
seventeen. A weekly projection needs a binary: is he playing this week.

--------------------------------------------------------------------------
PYLON . NFL projection and market-edge engine
Joshua Powers . joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl . linkedin.com/in/joshuampowerss . @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import RAW, PROC, OUT, CONF, SCORING, TEAM_ALIASES, TARGET_SEASON

# Fitted on 2023-2025 team-games against the closing line. `own_spread` is
# positive when the team is an underdog.
GAME_SCRIPT = {"pass_att_per_point": 0.070, "rush_att_per_point": -0.253}

# How much a defense moves the efficiency of what it faces.
#
# Scaled so the league's best defense suppresses production by about 11% and its
# worst inflates it by about 9%, which is roughly what a decade of weekly
# fantasy outcomes supports. The multiplier is large because the ratings it
# consumes are already heavily shrunk: defensive efficiency carries over at only
# 0.14 year to year, so the projected spread between the best and worst defense
# is under three hundredths of a point per play. Applying a "gentle" coefficient
# on top of that produces a matchup adjustment of plus or minus one percent,
# which is the same as having no matchup adjustment at all.
#
# Capped either way, because one week is a tiny sample and an uncapped
# adjustment turns a projection into a matchup lottery.
OPP_SENSITIVITY = {"pass": 8.0, "rush": 7.0}
OPP_CAP = 0.12

# Weekly scoring is far noisier than a season average. These are the standard
# deviations of a single game's fantasy points as a fraction of the mean,
# measured by position on 2024-2025.
WEEK_CV = {"QB": 0.34, "RB": 0.52, "WR": 0.58, "TE": 0.61}


def _week_context(week: int, season: int = TARGET_SEASON) -> pd.DataFrame:
    """One row per team playing this week: opponent, home flag, own spread."""
    mu_path = OUT / f"matchups_week{week}.csv"
    mu = pd.read_csv(mu_path) if mu_path.exists() else None
    g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    g = g[(g["season"] == season) & (g["game_type"] == "REG") & (g["week"] == week)].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_ALIASES)
    rows = []
    for _, r in g.iterrows():
        sp = None
        if mu is not None:
            hit = mu[(mu["home"] == r["home_team"]) & (mu["away"] == r["away_team"])]
            if len(hit):
                sp = float(hit.iloc[0]["model_spread_home"])
        rows.append({"team": r["home_team"], "opp": r["away_team"], "home": 1,
                     "own_spread": sp if sp is not None else 0.0})
        rows.append({"team": r["away_team"], "opp": r["home_team"], "home": 0,
                     "own_spread": -sp if sp is not None else 0.0})
    return pd.DataFrame(rows)


def project(players_df: pd.DataFrame, ratings: pd.DataFrame, ctx26: pd.DataFrame,
            week: int = 1, season: int = TARGET_SEASON) -> pd.DataFrame:
    """Weekly fantasy points for every player with a role, in full PPR."""
    ctxw = _week_context(week, season)
    if not len(ctxw):
        return pd.DataFrame()
    r = ratings.set_index("team")
    c = ctx26.set_index("team")
    ctxw = ctxw.set_index("team")

    p = players_df[players_df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    p = p[p["team"].isin(ctxw.index)]
    if not len(p):
        return pd.DataFrame()

    tm = p["team"]
    sp = tm.map(ctxw["own_spread"])
    opp = tm.map(ctxw["opp"])

    # Team volume for THIS week, shifted by the projected game script.
    pass_att = tm.map(c["proj_pass_att_per_game"]) + GAME_SCRIPT["pass_att_per_point"] * sp
    rush_all = tm.map(c["proj_rush_att_per_game"]) + GAME_SCRIPT["rush_att_per_point"] * sp
    rush_att = (rush_all * 0.905).clip(lower=8.0)
    td_gm = tm.map(c["proj_off_td_per_game"])

    # Opponent quality. def_final is EPA allowed per play, so a positive value is
    # a soft defense and lifts the efficiency of whoever is facing it.
    # Centre on the league so the adjustment is relative to an average opponent,
    # then scale and cap.
    dp, dr = r["def_pass_prior"], r["def_rush_prior"]
    opp_pass_def = opp.map(dp - dp.mean()).fillna(0.0)
    opp_rush_def = opp.map(dr - dr.mean()).fillna(0.0)
    pass_mult = (1.0 + OPP_SENSITIVITY["pass"] * opp_pass_def).clip(1 - OPP_CAP, 1 + OPP_CAP)
    rush_mult = (1.0 + OPP_SENSITIVITY["rush"] * opp_rush_def).clip(1 - OPP_CAP, 1 + OPP_CAP)

    # Per-game rates come straight from the season model, which already carries
    # the shrunk usage shares and efficiencies.
    gp = p["games"].replace(0, np.nan)
    is_qb = p["position"] == "QB"

    tgt = (p["targets"] / gp).fillna(0) * (pass_att / tm.map(c["proj_pass_att_per_game"]))
    car = (p["rush_att"] / gp).fillna(0) * np.where(is_qb, 1.0, rush_att / (tm.map(c["proj_rush_att_per_game"]) * 0.905))
    ypt = np.where(p["targets"] > 0, p["rec_yards"] / p["targets"].replace(0, np.nan), 0)
    cr = np.where(p["targets"] > 0, p["receptions"] / p["targets"].replace(0, np.nan), 0)
    ypc = np.where(p["rush_att"] > 0, p["rush_yards"] / p["rush_att"].replace(0, np.nan), 0)

    out = pd.DataFrame({
        "week": week, "player": p["player"].to_numpy(), "team": tm.to_numpy(),
        "position": p["position"].to_numpy(), "opponent": opp.to_numpy(),
        "home": tm.map(ctxw["home"]).to_numpy(), "own_spread": sp.to_numpy(),
    })
    out["targets"] = (tgt * pass_mult).to_numpy()
    out["receptions"] = out["targets"] * np.nan_to_num(cr)
    out["rec_yards"] = out["targets"] * np.nan_to_num(ypt) * pass_mult.to_numpy()
    out["rush_att"] = np.asarray(car) * rush_mult.to_numpy()
    out["rush_yards"] = out["rush_att"] * np.nan_to_num(ypc) * rush_mult.to_numpy()

    tdshare_rec = np.where(p["targets"].to_numpy() > 0,
                           p["rec_td"].to_numpy() / np.maximum(p["games"].to_numpy(), 1e-9), 0)
    tdshare_ru = p["rush_td"].to_numpy() / np.maximum(p["games"].to_numpy(), 1e-9)
    scale = (td_gm / tm.map(c["proj_off_td_per_game"])).to_numpy()
    out["rec_td"] = tdshare_rec * scale * pass_mult.to_numpy()
    out["rush_td"] = tdshare_ru * scale * rush_mult.to_numpy()

    pa = (p["pass_att"] / gp).fillna(0).to_numpy() * (pass_att / tm.map(c["proj_pass_att_per_game"])).to_numpy()
    ypa = np.where(p["pass_att"] > 0, p["pass_yards"] / p["pass_att"].replace(0, np.nan), 0)
    out["pass_att"] = np.where(is_qb, pa, 0.0)
    out["pass_yards"] = out["pass_att"] * np.nan_to_num(ypa) * pass_mult.to_numpy()
    out["pass_td"] = np.where(is_qb, (p["pass_td"] / gp).fillna(0).to_numpy() * scale * pass_mult.to_numpy(), 0.0)
    out["interceptions"] = np.where(is_qb, (p["interceptions"] / gp).fillna(0).to_numpy(), 0.0)

    s = SCORING
    out["points"] = (out["pass_yards"] * s["pass_yd"] + out["pass_td"] * s["pass_td"]
                     + out["interceptions"] * s["interception"]
                     + out["rush_yards"] * s["rush_yd"] + out["rush_td"] * s["rush_td"]
                     + out["receptions"] * s["rec"] + out["rec_yards"] * s["rec_yd"]
                     + out["rec_td"] * s["rec_td"])

    cv = out["position"].map(WEEK_CV).fillna(0.55)
    out["sd"] = out["points"] * cv
    out["floor"] = (out["points"] - 1.28 * out["sd"]).clip(lower=0)     # 10th percentile
    out["ceiling"] = out["points"] + 1.28 * out["sd"]                   # 90th percentile
    # What a matchup is worth: this week against a league-average opponent.
    neutral = out["points"] / ((pass_mult.to_numpy() + rush_mult.to_numpy()) / 2)
    out["matchup_delta"] = out["points"] - neutral
    out["pos_rank"] = out.groupby("position")["points"].rank(ascending=False, method="min")
    out = out.sort_values("points", ascending=False)
    out.to_csv(OUT / f"weekly_week{week}.csv", index=False)
    return out


def start_sit(weekly: pd.DataFrame, a: str, b: str) -> dict:
    """P(A outscores B) this week, treating each as lognormal around its mean.

    Lognormal rather than normal because fantasy scoring is bounded below by zero
    and has a long right tail. A normal model gives a 20-point receiver a real
    chance of a negative week, which is not a thing that happens."""
    ra = weekly[weekly["player"] == a]
    rb = weekly[weekly["player"] == b]
    if not len(ra) or not len(rb):
        return {}
    ra, rb = ra.iloc[0], rb.iloc[0]
    out = {}
    for r, k in ((ra, "a"), (rb, "b")):
        mu, sd = max(float(r["points"]), 0.1), max(float(r["sd"]), 0.1)
        v = np.log1p((sd / mu) ** 2)
        out[k] = {"mu": np.log(mu) - v / 2, "sigma": np.sqrt(v)}
    from scipy.stats import norm
    d_mu = out["a"]["mu"] - out["b"]["mu"]
    d_sd = np.sqrt(out["a"]["sigma"] ** 2 + out["b"]["sigma"] ** 2)
    return {"a": a, "b": b, "p_a_beats_b": float(norm.cdf(d_mu / d_sd)),
            "a_points": float(ra["points"]), "b_points": float(rb["points"])}


def build(players_df: pd.DataFrame, ratings: pd.DataFrame, ctx26: pd.DataFrame, week: int = 1):
    with open(CONF / "game_script.json", "w") as f:
        json.dump({**GAME_SCRIPT, "opp_sensitivity": OPP_SENSITIVITY,
                   "week_cv": WEEK_CV, "fitted_on": "2023-2025, 1632 team-games"}, f, indent=2)
    return project(players_df, ratings, ctx26, week=week)
