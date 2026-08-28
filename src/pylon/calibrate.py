"""
PYLON :: calibrate
Fits the two things the model refuses to assume:

  1. SHRINKAGE. How much of a team's multi-season blended rating actually
     carries into next year? Answered by regressing year-t opponent-adjusted
     ratings on a blend built only from years t-3..t-1. The slope IS the
     shrinkage coefficient, per phase, measured out of sample.

  2. TRANSLATION AND SIGMA. How many points of game margin one unit of net
     rating buys, the modern home-field advantage, and the standard deviation
     of the residual. Fit on genuine forecasts: prior-years-only ratings
     predicting the following season's games.

Run once. The coefficients are written to config/calibration.json and the
production model reads them from there.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowers · @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import PROC, CONF, SEASON_WEIGHTS, RAW, TEAM_ALIASES
from . import ratings as R

PHASES = ["off_pass", "off_rush", "def_pass", "def_rush"]


def blend_prior(sr: pd.DataFrame, target_year: int, lookback: int = 3) -> pd.DataFrame:
    """Build a team rating using ONLY seasons strictly before target_year."""
    yrs = [target_year - k for k in range(1, lookback + 1)]
    w = {y: list(SEASON_WEIGHTS.values())[i] for i, y in enumerate(yrs)}
    d = sr[sr["season"].isin(yrs)].copy()
    d["w"] = d["season"].map(w)
    rows = []
    for team, g in d.groupby("team"):
        ww = g["w"].to_numpy(dtype=float)
        if ww.sum() == 0:
            continue
        ww = ww / ww.sum()
        rows.append({"team": team, **{c: float(np.nansum(g[c].to_numpy() * ww)) for c in PHASES}})
    return pd.DataFrame(rows)


def fit_shrinkage(sr: pd.DataFrame, test_years: list[int]) -> dict:
    """Slope of actual_t ~ blend_{<t}, per phase. No intercept: both sides are
    already mean-zero by construction, so a fitted intercept would only absorb
    sampling noise."""
    stacked = {c: {"x": [], "y": []} for c in PHASES}
    for yr in test_years:
        prior = blend_prior(sr, yr)
        actual = sr[sr["season"] == yr][["team"] + PHASES]
        m = prior.merge(actual, on="team", suffixes=("_prior", "_act"))
        for c in PHASES:
            stacked[c]["x"].extend(m[c + "_prior"].tolist())
            stacked[c]["y"].extend(m[c + "_act"].tolist())
    out = {}
    for c in PHASES:
        x = np.array(stacked[c]["x"]); y = np.array(stacked[c]["y"])
        slope = float((x @ y) / (x @ x))
        r = float(np.corrcoef(x, y)[0, 1])
        out[c] = {"shrinkage": round(slope, 4), "correlation": round(r, 4), "n": int(len(x))}
    return out


def fit_translation(sr: pd.DataFrame, games: pd.DataFrame, shrink: dict,
                    test_years: list[int]) -> dict:
    """True forward test: prior-years-only ratings vs. the season's real results."""
    lg_dr = 0.58
    rows = []
    for yr in test_years:
        p = blend_prior(sr, yr)
        for c in PHASES:
            p[c] = p[c] * shrink[c]["shrinkage"]
        p["off"] = lg_dr * p["off_pass"] + (1 - lg_dr) * p["off_rush"]
        p["def"] = lg_dr * p["def_pass"] + (1 - lg_dr) * p["def_rush"]
        p["net"] = p["off"] - p["def"]
        key = p.set_index("team")["net"]
        g = games[(games["season"] == yr)].dropna(subset=["result"]).copy()
        g["h"] = g["home_team"].map(key); g["a"] = g["away_team"].map(key)
        g = g.dropna(subset=["h", "a"])
        g["diff"] = g["h"] - g["a"]
        rows.append(g[["season", "diff", "result", "home_team", "away_team"]])
    g = pd.concat(rows, ignore_index=True)

    X = np.column_stack([np.ones(len(g)), g["diff"].to_numpy()])
    y = g["result"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - float((resid ** 2).sum()) / float(((y - y.mean()) ** 2).sum())

    # Straight-up accuracy of the sign of the prediction, the only number a
    # non-technical viewer actually cares about.
    pred = X @ beta
    played = g["result"] != 0
    su = float((np.sign(pred[played]) == np.sign(y[played])).mean())

    return {"hfa": round(float(beta[0]), 3), "points_per_rating": round(float(beta[1]), 2),
            "sigma": round(float(resid.std(ddof=2)), 3), "r2_out_of_sample": round(r2, 4),
            "straight_up_accuracy": round(su, 4), "n_games": int(len(g)),
            "test_years": test_years}


def run(all_seasons=range(2017, 2026), test_years=(2021, 2022, 2023, 2024, 2025)):
    from . import ingest
    raw = ingest.load_raw(list(all_seasons))
    plays = ingest.clean_plays(raw)
    sr = R.season_ratings(plays, seasons=sorted(all_seasons))
    sr.to_csv(PROC / "season_ratings_full.csv", index=False)

    g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    g = g[(g["game_type"] == "REG")].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_ALIASES)

    shrink = fit_shrinkage(sr, list(test_years))
    trans = fit_translation(sr, g, shrink, list(test_years))
    cal = {"shrinkage": shrink, "translation": trans,
           "note": "Fitted out of sample: every rating used to predict season t "
                   "was built only from seasons t-3 through t-1."}
    with open(CONF / "calibration.json", "w") as f:
        json.dump(cal, f, indent=2)
    return cal


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
