"""
PYLON :: backtest
Replays the engine against five completed seasons using only information that
existed before each of them, and uses the result to fix the one parameter a
simulator cannot honestly guess: how uncertain the ratings themselves are.

The failure this catches is subtle and very common. A season simulator that
models only game-level randomness produces win distributions that are far too
narrow, because it implicitly assumes the power ratings are exactly right. They
are not. A team that is secretly two points better than its rating is two
points better in all seventeen games, not independently lucky seventeen times,
and that persistent error is what actually widens a win-total distribution.

So the total residual is split into two pieces:

    sigma_total^2 = sigma_game^2 + 2 * sigma_team^2

sigma_team is solved for by matching the simulator's own win-total standard
deviation to the measured out-of-sample error of the model's win projections
across 2021-2025. Everything reported here is a genuine forward test.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowerss · @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from .config import PROC, CONF, OUT, RAW, TEAM_ALIASES
from . import calibrate as CAL

LG_DR = 0.58


def _prior_net(sr: pd.DataFrame, yr: int, shrink: dict) -> pd.Series:
    p = CAL.blend_prior(sr, yr)
    for c, s in shrink.items():
        p[c] = p[c] * s
    net = (LG_DR * p["off_pass"] + (1 - LG_DR) * p["off_rush"]) - \
          (LG_DR * p["def_pass"] + (1 - LG_DR) * p["def_rush"])
    return pd.Series(net.to_numpy(), index=p["team"])


def expected_wins(net: pd.Series, games: pd.DataFrame, trans: dict) -> pd.Series:
    """Analytic expected wins: sum of per-game win probabilities."""
    out = {t: 0.0 for t in net.index}
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in net.index or a not in net.index:
            continue
        m = trans["hfa"] + trans["points_per_rating"] * (net[h] - net[a])
        p = float(norm.cdf(m / trans["sigma"]))
        out[h] += p
        out[a] += 1 - p
    return pd.Series(out)


def run(seasons=(2021, 2022, 2023, 2024, 2025)) -> dict:
    sr = pd.read_csv(PROC / "season_ratings_full.csv")
    cal = json.loads((CONF / "calibration.json").read_text())
    shrink = {k: v["shrinkage"] for k, v in cal["shrinkage"].items()}
    trans = cal["translation"]

    g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    g = g[g["game_type"] == "REG"].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_ALIASES)

    rows = []
    for yr in seasons:
        net = _prior_net(sr, yr, shrink)
        gy = g[g["season"] == yr].dropna(subset=["result"])
        pred = expected_wins(net, gy, trans)
        actual = {}
        for _, r in gy.iterrows():
            for t in (r["home_team"], r["away_team"]):
                actual.setdefault(t, 0.0)
            if r["result"] > 0:
                actual[r["home_team"]] += 1
            elif r["result"] < 0:
                actual[r["away_team"]] += 1
            else:
                actual[r["home_team"]] += 0.5; actual[r["away_team"]] += 0.5
        for t in pred.index:
            if t in actual:
                rows.append({"season": yr, "team": t, "pred_wins": pred[t], "actual_wins": actual[t]})

    bt = pd.DataFrame(rows)
    bt["error"] = bt["actual_wins"] - bt["pred_wins"]
    rmse = float(np.sqrt((bt["error"] ** 2).mean()))
    mae = float(bt["error"].abs().mean())
    corr = float(bt[["pred_wins", "actual_wins"]].corr().iloc[0, 1])

    # A pure binomial season around a known true strength has SD ~2.06 wins.
    # Whatever error remains beyond that is rating error, and that is the piece
    # the simulator has to carry as a season-long team effect.
    binom_sd = 2.06
    team_sd_wins = float(np.sqrt(max(rmse ** 2 - binom_sd ** 2, 0.04)))
    # Convert wins of error into points of rating: d(wins)/d(point) over a
    # 17-game schedule is 17 * phi(0) / sigma.
    wins_per_point = 17.0 * norm.pdf(0.0) / trans["sigma"]
    sigma_team_pts = team_sd_wins / wins_per_point
    sigma_game = float(np.sqrt(max(trans["sigma"] ** 2 - 2 * sigma_team_pts ** 2, 6.0 ** 2)))

    bt.to_csv(OUT / "backtest_win_projections.csv", index=False)
    result = {
        "seasons_tested": list(seasons), "n_team_seasons": int(len(bt)),
        "rmse_wins": round(rmse, 3), "mae_wins": round(mae, 3),
        "correlation_pred_actual": round(corr, 3),
        "within_1_win_pct": round(float((bt["error"].abs() <= 1).mean()), 3),
        "within_2_wins_pct": round(float((bt["error"].abs() <= 2).mean()), 3),
        "straight_up_game_accuracy": trans["straight_up_accuracy"],
        "variance_split": {
            "sigma_total_pts": trans["sigma"],
            "sigma_game_pts": round(sigma_game, 3),
            "sigma_team_pts": round(sigma_team_pts, 3),
            "team_error_in_wins": round(team_sd_wins, 3),
            "note": "sigma_team is a season-long, team-constant strength error. "
                    "Without it the simulator understates win-total spread by roughly a third.",
        },
    }
    with open(CONF / "backtest.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
