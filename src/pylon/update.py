"""
PYLON :: update
In-season re-fitting, and the evidence that it helps.

A preseason rating is a prior. The moment real games exist, the honest thing to
do is combine the two rather than keep quoting week zero all year. This module
does that, and it does not ask to be believed: `weekly_backtest` replays a
completed season week by week, rebuilds the ratings from only the games that had
been played at that point, and scores the updating model against the frozen one
on the games that came next.

The blend is a standard conjugate-style weight,

    rating = w * in_season + (1 - w) * preseason,     w = n / (n + k)

where n is games played and k is the number of games it takes for the in-season
signal to earn half the weight. k is fitted, not chosen: `fit_blend` sweeps it
and keeps whichever value minimises out-of-sample game error across the replay.

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
from .config import (RAW, PROC, OUT, CONF, TEAM_ALIASES, SEASON_WEIGHTS,
                     RIDGE_ALPHA_PASS, RIDGE_ALPHA_RUSH)
from . import ratings as R
from . import calibrate as CAL

LG_DR = 0.58
PHASES = ["off_pass", "off_rush", "def_pass", "def_rush"]


def _net(frame: pd.DataFrame) -> pd.Series:
    off = LG_DR * frame["off_pass"] + (1 - LG_DR) * frame["off_rush"]
    dfn = LG_DR * frame["def_pass"] + (1 - LG_DR) * frame["def_rush"]
    return pd.Series((off - dfn).to_numpy(), index=frame["team"])


def in_season_ratings(plays: pd.DataFrame, season: int, through_week: int) -> pd.DataFrame:
    """Opponent-adjusted ratings from only the games played so far this season."""
    d = plays[(plays["season"] == season) & (plays["week"] <= through_week)]
    if d["game_id"].nunique() < 16:          # too little to fit anything stable
        return pd.DataFrame(columns=["team"] + PHASES)
    out = []
    for phase, alpha in (("pass", RIDGE_ALPHA_PASS), ("rush", RIDGE_ALPHA_RUSH)):
        flag = "is_pass" if phase == "pass" else "is_rush"
        sub = d[d[flag] == 1]
        teams = sorted(set(sub["posteam"]) | set(sub["defteam"]))
        X, y = R._design(sub, teams)
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky").fit(X, y)
        k = len(teams)
        o, f = m.coef_[:k], m.coef_[k:2 * k]
        out.append(pd.DataFrame({"team": teams, f"off_{phase}": o - o.mean(),
                                 f"def_{phase}": f - f.mean()}))
    return out[0].merge(out[1], on="team", how="outer")


def blend(preseason: pd.Series, inseason: pd.Series, games_played: float, k: float) -> pd.Series:
    """Weight the in-season signal by how much of it there is."""
    w = games_played / (games_played + k)
    return (1 - w) * preseason.reindex(inseason.index).fillna(0.0) + w * inseason


def _games(seasons) -> pd.DataFrame:
    g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    g = g[g["season"].isin(seasons) & (g["game_type"] == "REG")].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_ALIASES)
    return g.dropna(subset=["result"])


def weekly_backtest(season: int = 2025, k: float = 4.0, start_week: int = 3) -> pd.DataFrame:
    """Replay a season. At each week, predict that week's games two ways: with a
    frozen preseason rating, and with a rating updated on everything played so
    far. Neither ever sees the week being predicted."""
    plays = pd.read_parquet(PROC / "plays_clean.parquet")
    sr = pd.read_csv(PROC / "season_ratings_full.csv")
    cal = json.loads((CONF / "calibration.json").read_text())
    shrink = {a: b["shrinkage"] for a, b in cal["shrinkage"].items()}
    trans = cal["translation"]

    prior = CAL.blend_prior(sr, season)
    for c in PHASES:
        prior[c] = prior[c] * shrink[c]
    pre_net = _net(prior)

    g = _games([season])
    rows = []
    for wk in range(start_week, int(g["week"].max()) + 1):
        ins = in_season_ratings(plays, season, wk - 1)
        if not len(ins):
            continue
        in_net = _net(ins)
        played = plays[(plays["season"] == season) & (plays["week"] < wk)] \
            .groupby("posteam")["game_id"].nunique()
        n = float(played.mean()) if len(played) else 0.0
        upd_net = blend(pre_net, in_net, n, k)

        for _, x in g[g["week"] == wk].iterrows():
            h, a = x["home_team"], x["away_team"]
            if h not in upd_net.index or a not in upd_net.index:
                continue
            for label, net in (("frozen", pre_net), ("updated", upd_net)):
                if h not in net.index or a not in net.index:
                    continue
                pred = trans["hfa"] + trans["points_per_rating"] * (net[h] - net[a])
                rows.append({"season": season, "week": wk, "model": label,
                             "games_played": n, "pred_margin": pred,
                             "actual_margin": float(x["result"]),
                             "correct": int(np.sign(pred) == np.sign(x["result"])) if x["result"] != 0 else 0,
                             "abs_err": abs(pred - float(x["result"])),
                             "logloss": -np.log(max(1e-9, norm.cdf(pred / trans["sigma"])
                                                    if x["result"] > 0 else 1 - norm.cdf(pred / trans["sigma"])))
                             if x["result"] != 0 else np.nan})
    return pd.DataFrame(rows)


def fit_blend(seasons=(2023, 2024, 2025), ks=(1, 2, 3, 4, 6, 8, 12, 999)) -> dict:
    """Sweep the half-weight constant and keep whichever k predicts best."""
    res = []
    for k in ks:
        parts = [weekly_backtest(s, k=float(k)) for s in seasons]
        d = pd.concat(parts, ignore_index=True)
        u = d[d["model"] == "updated"]
        res.append({"k": k, "mae": float(u["abs_err"].mean()),
                    "accuracy": float(u["correct"].mean()),
                    "logloss": float(u["logloss"].mean(skipna=True))})
    best = min(res, key=lambda r: r["mae"])
    frozen = pd.concat([weekly_backtest(s, k=float(best["k"])) for s in seasons], ignore_index=True)
    fz = frozen[frozen["model"] == "frozen"]
    out = {"k_best": best["k"], "sweep": res, "seasons": list(seasons),
           "frozen": {"mae": float(fz["abs_err"].mean()), "accuracy": float(fz["correct"].mean()),
                      "logloss": float(fz["logloss"].mean(skipna=True))},
           "updated": {"mae": best["mae"], "accuracy": best["accuracy"], "logloss": best["logloss"]}}
    with open(CONF / "inseason.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def curve(seasons=(2023, 2024, 2025), k: float | None = None) -> pd.DataFrame:
    """Accuracy by week, frozen versus updated. This is the chart."""
    if k is None:
        k = json.loads((CONF / "inseason.json").read_text())["k_best"]
    d = pd.concat([weekly_backtest(s, k=float(k)) for s in seasons], ignore_index=True)
    c = d.groupby(["week", "model"]).agg(accuracy=("correct", "mean"),
                                         mae=("abs_err", "mean"),
                                         n=("correct", "size")).reset_index()
    c.to_csv(OUT / "inseason_curve.csv", index=False)
    return c
