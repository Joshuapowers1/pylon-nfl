"""
PYLON :: game_model
Turns two net ratings into a point spread, and a point spread into a win
probability.

The conversion constant is not assumed. It is fit: regress actual game margin
on (net_rating_home - net_rating_away) across the training seasons, which
returns both the points-per-unit-of-rating slope and the modern home-field
advantage as the intercept. The residual standard deviation of that fit is the
sigma used for every win probability in the simulator, so the uncertainty in the
model is measured rather than guessed.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowerss · @joshuampowers
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from .config import RAW, TEAM_ALIASES, MARGIN_SIGMA, HOME_FIELD_POINTS


def load_games(seasons) -> pd.DataFrame:
    g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    g = g[g["season"].isin(seasons) & (g["game_type"] == "REG")].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_ALIASES)
    return g


def calibrate(season_ratings: pd.DataFrame, games: pd.DataFrame, ctx: pd.DataFrame) -> dict:
    """Fit margin ~ a + b * rating_diff on completed games.

    Ratings are taken from the SAME season as the game, which makes this an
    in-sample descriptive fit of the ratings-to-points scale, not a forecast
    test. That is the correct use here: we want the translation constant, and
    the out-of-sample question is answered separately by the backtest."""
    lg_dr = 0.58
    sr = season_ratings.copy()
    sr["off"] = lg_dr * sr["off_pass"] + (1 - lg_dr) * sr["off_rush"]
    sr["def"] = lg_dr * sr["def_pass"] + (1 - lg_dr) * sr["def_rush"]
    sr["net"] = sr["off"] - sr["def"]
    key = sr.set_index(["season", "team"])["net"]

    g = games.dropna(subset=["result"]).copy()
    g["h_net"] = [key.get((s, t), np.nan) for s, t in zip(g["season"], g["home_team"])]
    g["a_net"] = [key.get((s, t), np.nan) for s, t in zip(g["season"], g["away_team"])]
    g = g.dropna(subset=["h_net", "a_net"])
    g["diff"] = g["h_net"] - g["a_net"]

    X = np.column_stack([np.ones(len(g)), g["diff"].to_numpy()])
    y = g["result"].to_numpy(dtype=float)          # nflverse result = home_score - away_score
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma = float(resid.std(ddof=2))
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"hfa": float(beta[0]), "points_per_rating": float(beta[1]),
            "sigma": sigma, "r2": 1 - ss_res / ss_tot, "n_games": int(len(g))}


def spread(net_home: float, net_away: float, cal: dict, neutral: bool = False) -> float:
    """Expected home margin in points."""
    hfa = 0.0 if neutral else cal.get("hfa", HOME_FIELD_POINTS)
    return hfa + cal.get("points_per_rating", 60.0) * (net_home - net_away)


def win_prob(margin: float, sigma: float | None = None) -> float:
    """P(home wins), treating margin as normal around the predicted spread.
    Ties are handled separately in the simulator; here a 0-point outcome splits."""
    s = sigma or MARGIN_SIGMA
    return float(norm.cdf(margin / s))
