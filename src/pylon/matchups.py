"""
PYLON :: matchups
Game-level projections: a point spread, a total, and a win probability for every
scheduled game, plus a comparison against the posted market line.

The season simulator only ever needed a margin. A total needs something more:
how many points each side actually scores. So this module fits a scoring model,

    points_scored = a + b * off_rating_team + c * def_rating_opponent
                      + d * pace_team + e * home

on completed games from the training seasons, and then applies it forward. The
spread falls out as the difference of the two expected scores and the total as
their sum, which guarantees the two are internally consistent. A model that
projects a spread and a total from separate equations can and will produce a
pair that implies an impossible pair of team scores.

--------------------------------------------------------------------------
PYLON . NFL projection and market-edge engine
Joshua Powers . joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl . linkedin.com/in/joshuampowerss . @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from .config import RAW, PROC, OUT, CONF, TEAM_ALIASES, TRAIN_SEASONS, TARGET_SEASON


def _games(seasons) -> pd.DataFrame:
    g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    g = g[g["season"].isin(seasons) & (g["game_type"] == "REG")].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_ALIASES)
    return g


def fit_scoring(season_ratings: pd.DataFrame, ctx: pd.DataFrame,
                seasons=TRAIN_SEASONS) -> dict:
    """One row per team per completed game: what they scored, who they played."""
    lg_dr = 0.58
    sr = season_ratings.copy()
    sr["off"] = lg_dr * sr["off_pass"] + (1 - lg_dr) * sr["off_rush"]
    sr["def"] = lg_dr * sr["def_pass"] + (1 - lg_dr) * sr["def_rush"]
    off = sr.set_index(["season", "team"])["off"]
    dfn = sr.set_index(["season", "team"])["def"]
    pace = ctx.set_index(["season", "team"])["plays_per_game"]

    g = _games(seasons).dropna(subset=["home_score", "away_score"])
    rows = []
    for _, r in g.iterrows():
        for side, opp, pts, home in (("home_team", "away_team", "home_score", 1),
                                     ("away_team", "home_team", "away_score", 0)):
            k, ko = (r["season"], r[side]), (r["season"], r[opp])
            if k not in off.index or ko not in dfn.index or k not in pace.index:
                continue
            rows.append({"pts": r[pts], "off": off[k], "def_opp": dfn[ko],
                         "pace": pace[k], "home": home})
    d = pd.DataFrame(rows)
    X = np.column_stack([np.ones(len(d)), d["off"], d["def_opp"], d["pace"], d["home"]])
    y = d["pts"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"intercept": float(beta[0]), "b_off": float(beta[1]), "b_def_opp": float(beta[2]),
            "b_pace": float(beta[3]), "b_home": float(beta[4]),
            "sigma_points": float(resid.std(ddof=5)),
            "r2": float(1 - (resid ** 2).sum() / ss_tot), "n_team_games": int(len(d))}


def _expected_points(sc: dict, off: float, def_opp: float, pace: float, home: int) -> float:
    return (sc["intercept"] + sc["b_off"] * off + sc["b_def_opp"] * def_opp
            + sc["b_pace"] * pace + sc["b_home"] * home)


def project_week(ratings: pd.DataFrame, ctx26: pd.DataFrame, sc: dict, cal: dict,
                 week: int = 1, season: int = TARGET_SEASON) -> pd.DataFrame:
    """Model spread, total and win probability for one week, against the market."""
    r = ratings.set_index("team")
    pace = ctx26.set_index("team")["proj_plays_per_game"]
    snap = json.loads((CONF / "market_snapshot_2026.json").read_text())
    lines = {(l["away"], l["home"]): l for l in snap.get("week1_lines_dk", [])} if week == 1 else {}

    g = _games([season])
    g = g[g["week"] == week]
    out = []
    for _, x in g.iterrows():
        h, a = x["home_team"], x["away_team"]
        if h not in r.index or a not in r.index:
            continue
        raw_h = _expected_points(sc, r.loc[h, "off_final"], r.loc[a, "def_final"], pace[h], 1)
        raw_a = _expected_points(sc, r.loc[a, "off_final"], r.loc[h, "def_final"], pace[a], 0)
        # The margin comes from the calibrated game model, not from the difference
        # of the two fitted scores. Those are two separate regressions and they
        # disagree by a couple of points on lopsided games, which would leave the
        # matchup page quoting a different spread than the simulator that produced
        # every other number on the site. So the scoring model supplies the TOTAL,
        # the game model supplies the MARGIN, and the two team scores are recovered
        # from that pair. Everything stays consistent by construction.
        total = raw_h + raw_a
        margin = cal["hfa"] + cal["points_per_rating"] * (
            r.loc[h, "net_final"] - r.loc[a, "net_final"])
        ph = (total + margin) / 2.0
        pa = (total - margin) / 2.0
        wp = float(norm.cdf(margin / cal["sigma"]))
        mk = lines.get((a, h))
        row = {"week": int(x["week"]), "away": a, "home": h,
               "model_away_pts": pa, "model_home_pts": ph,
               "model_spread_home": -margin,          # negative means home favoured
               "model_total": total,
               "home_win_prob": wp,
               "market_spread_home": mk["spread_home"] if mk else None,
               "market_total": mk["total"] if mk else None,
               "kickoff": mk["date"] if mk else x.get("gameday")}
        if mk:
            # A positive spread edge means the model likes the home side relative
            # to the number. A positive total edge means the model is over it.
            row["spread_edge"] = mk["spread_home"] - row["model_spread_home"]
            row["total_edge"] = row["model_total"] - mk["total"]
            row["market_home_win_prob"] = float(norm.cdf(-mk["spread_home"] / cal["sigma"]))
            row["wp_edge"] = wp - row["market_home_win_prob"]
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(OUT / f"matchups_week{week}.csv", index=False)
    return df


def build(ratings: pd.DataFrame, cal: dict, week: int = 1):
    sr = pd.read_csv(PROC / "season_ratings_full.csv")
    ctx = pd.read_csv(PROC / "team_context.csv")
    ctx26 = pd.read_csv(PROC / "team_context_2026.csv")
    sc = fit_scoring(sr[sr["season"].isin(TRAIN_SEASONS)], ctx)
    with open(CONF / "scoring_model.json", "w") as f:
        json.dump(sc, f, indent=2)
    return project_week(ratings, ctx26, sc, cal, week=week), sc
