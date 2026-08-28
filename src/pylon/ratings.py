"""
PYLON :: ratings
Opponent-adjusted team unit ratings via ridge regression on play-level EPA.

The problem this solves: raw EPA/play leaderboards are schedule artefacts. A
team that played six bottom-five defenses looks elite and is not. So instead of
averaging EPA, we regress it:

    epa_play ~ b0 + off_team + def_team + home_offense

fit separately for pass plays and rush plays, with an L2 penalty. Every team
gets one offensive and one defensive coefficient per phase, expressed in points
of EPA per play above or below league average, with strength of schedule
already divided out.

Then three things happen to those coefficients:
  1. Multi-season blend with exponential recency decay.
  2. Shrinkage toward zero by measured year-over-year stability, applied
     per phase. Defense regresses much harder than offense because it is
     much less repeatable.
  3. Hand-specified roster and scheme deltas (see adjustments.py).

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowers · @joshuampowers
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import json
from .config import (PROC, OUT, CONF, TRAIN_SEASONS, SEASON_WEIGHTS, TEAMS,
                     RIDGE_ALPHA_PASS, RIDGE_ALPHA_RUSH,
                     STABILITY_OFF_PASS, STABILITY_OFF_RUSH,
                     STABILITY_DEF_PASS, STABILITY_DEF_RUSH)


def fitted_stability() -> dict:
    """Read the out-of-sample shrinkage coefficients produced by calibrate.py.
    Falls back to the config defaults if calibration has not been run."""
    f = CONF / "calibration.json"
    if f.exists():
        cal = json.loads(f.read_text())
        return {k: v["shrinkage"] for k, v in cal["shrinkage"].items()}
    return {"off_pass": STABILITY_OFF_PASS, "off_rush": STABILITY_OFF_RUSH,
            "def_pass": STABILITY_DEF_PASS, "def_rush": STABILITY_DEF_RUSH}


def _design(df: pd.DataFrame, teams: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Sparse-ish one-hot design: [off dummies | def dummies | home flag].
    Small enough (65 columns) that a dense matrix is fine and far faster to
    debug than a scipy sparse build."""
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(df), len(teams)
    X = np.zeros((n, 2 * k + 1), dtype=np.float32)
    X[np.arange(n), df["posteam"].map(idx).to_numpy()] = 1.0
    X[np.arange(n), k + df["defteam"].map(idx).to_numpy()] = 1.0
    X[:, -1] = df["home_off"].to_numpy()
    return X, df["epa"].to_numpy(dtype=np.float32)


def fit_phase(plays: pd.DataFrame, season: int, phase: str, alpha: float) -> pd.DataFrame:
    """Fit one ridge model for one season and one phase (pass or rush)."""
    flag = "is_pass" if phase == "pass" else "is_rush"
    d = plays[(plays["season"] == season) & (plays[flag] == 1)]
    teams = sorted(set(d["posteam"]) | set(d["defteam"]))
    X, y = _design(d, teams)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
    model.fit(X, y)
    k = len(teams)
    off = model.coef_[:k]
    dfn = model.coef_[k:2 * k]
    # Re-center so the coefficients read as "vs league average" exactly.
    off = off - off.mean()
    dfn = dfn - dfn.mean()
    return pd.DataFrame({
        "season": season, "team": teams, "phase": phase,
        f"off_{phase}": off, f"def_{phase}": dfn,
        "n_plays": len(d), "hfa_epa": model.coef_[-1], "intercept": model.intercept_,
    })


def season_ratings(plays: pd.DataFrame, seasons=TRAIN_SEASONS) -> pd.DataFrame:
    """Every team-season, both phases, opponent-adjusted."""
    out = []
    for yr in seasons:
        p = fit_phase(plays, yr, "pass", RIDGE_ALPHA_PASS)[["season","team","off_pass","def_pass","hfa_epa"]]
        r = fit_phase(plays, yr, "rush", RIDGE_ALPHA_RUSH)[["season","team","off_rush","def_rush"]]
        out.append(p.merge(r, on=["season", "team"], how="outer"))
    return pd.concat(out, ignore_index=True)


def blend(sr: pd.DataFrame) -> pd.DataFrame:
    """Recency-weighted blend across training seasons, then stability shrinkage.

    A team with only one season of data (relocation, expansion) would get an
    unstable estimate, so weights are renormalised per team on whatever seasons
    that team actually has."""
    sr = sr.copy()
    sr["w"] = sr["season"].map(SEASON_WEIGHTS).astype(float)
    cols = ["off_pass", "off_rush", "def_pass", "def_rush"]
    rows = []
    for team, g in sr.groupby("team"):
        g = g.dropna(subset=cols, how="all")
        w = g["w"].to_numpy()
        w = w / w.sum()
        rows.append({"team": team, **{c: float(np.nansum(g[c].to_numpy() * w)) for c in cols},
                     "seasons_used": len(g)})
    b = pd.DataFrame(rows)

    stability = fitted_stability()
    for c, s in stability.items():
        b[c + "_prior"] = b[c] * s          # shrink toward league average (zero)
    b.attrs["stability_used"] = stability
    return b


def net_ratings(blended: pd.DataFrame, ctx26: pd.DataFrame) -> pd.DataFrame:
    """Collapse phase ratings into a single points-per-play net rating, weighting
    pass and rush by each team's own projected play mix rather than a league
    constant. A run-first team gets less credit for an elite passing grade."""
    b = blended.merge(ctx26[["team", "proj_dropback_rate"]], on="team", how="left")
    dr = b["proj_dropback_rate"].fillna(0.58)
    b["off_rating"] = dr * b["off_pass_prior"] + (1 - dr) * b["off_rush_prior"]
    # Defense faces a league-average mix, not its own offense's mix.
    lg_dr = float(ctx26["proj_dropback_rate"].mean())
    b["def_rating"] = lg_dr * b["def_pass_prior"] + (1 - lg_dr) * b["def_rush_prior"]
    b["net_rating"] = b["off_rating"] - b["def_rating"]
    return b


def build(plays: pd.DataFrame, ctx26: pd.DataFrame):
    sr = season_ratings(plays)
    sr.to_csv(PROC / "season_ratings.csv", index=False)
    bl = blend(sr)
    nr = net_ratings(bl, ctx26)
    nr.to_csv(PROC / "net_ratings_pre_adjustment.csv", index=False)
    return sr, nr
