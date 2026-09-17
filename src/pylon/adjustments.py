"""
PYLON :: adjustments
Two jobs.

1. Project 2026 team CONTEXT (pace, dropback rate, scoring volume) from a
   recency-weighted history, nudged by new play-callers.
2. Apply the 2026 roster and scheme priors to the opponent-adjusted ratings.

On the priors: these are analyst inputs, not model output, and they are the one
subjective layer in the engine. They live in config/offseason_2026.json with a
written justification per team so anyone auditing the model can disagree with a
specific number instead of the whole thing.

Both delta vectors are mean-centred before they are applied. The league cannot
collectively get better on defense. If every team's stated adjustment is
negative, what that actually encodes is relative, not absolute, improvement.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowerss · @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import CONF, PROC, SEASON_WEIGHTS, TEAMS, PRIOR_CONFIDENCE_OFF, PRIOR_CONFIDENCE_DEF


def load_priors() -> dict:
    with open(CONF / "offseason_2026.json") as f:
        return json.load(f)


def project_context(ctx: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """Recency-weighted 2026 pace and play-mix, then a scheme multiplier."""
    ctx = ctx.copy()
    ctx["w"] = ctx["season"].map(SEASON_WEIGHTS).astype(float)
    cols = ["plays_per_game", "pass_att_per_game", "rush_att_per_game",
            "dropback_rate", "off_td_per_game", "pass_td", "rush_td"]
    rows = []
    for team, g in ctx.groupby("team"):
        w = g["w"].to_numpy(); w = w / w.sum()
        rec = {"team": team}
        for c in cols:
            rec["proj_" + c] = float(np.nansum(g[c].to_numpy() * w))
        rows.append(rec)
    p = pd.DataFrame(rows)

    # League-average shrinkage: pace is only moderately repeatable year to year
    # (r ~ 0.45), so pull 30% of the way back to the league mean.
    for c in ["proj_plays_per_game", "proj_dropback_rate", "proj_off_td_per_game"]:
        p[c] = 0.70 * p[c] + 0.30 * p[c].mean()

    sched = priors.get("pace_scheme_2026", {})
    p["scheme_mult"] = p["team"].map(lambda t: sched.get(t, 1.0)).astype(float)
    p["proj_dropback_rate"] = (p["proj_dropback_rate"] * p["scheme_mult"]).clip(0.44, 0.70)

    # Rebuild attempt splits off the adjusted dropback rate so the three
    # volume numbers stay internally consistent.
    # ~7.0% of dropbacks end as sacks and ~5.5% as scrambles, league-wide.
    p["proj_dropbacks_per_game"] = p["proj_plays_per_game"] * p["proj_dropback_rate"]
    p["proj_pass_att_per_game"] = p["proj_dropbacks_per_game"] * 0.875
    p["proj_rush_att_per_game"] = p["proj_plays_per_game"] - p["proj_dropbacks_per_game"]
    p["proj_qb_rush_per_game"] = p["proj_dropbacks_per_game"] * 0.055
    return p


def apply_priors(net: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """Add mean-centred roster deltas to the shrunk opponent-adjusted ratings."""
    t = priors["teams"]
    n = net.copy()
    n["off_delta_raw"] = n["team"].map(lambda x: t.get(x, {}).get("off_delta", 0.0))
    n["def_delta_raw"] = n["team"].map(lambda x: t.get(x, {}).get("def_delta", 0.0))
    n["prior_note"] = n["team"].map(lambda x: t.get(x, {}).get("notes", ""))

    n["off_delta"] = (n["off_delta_raw"] - n["off_delta_raw"].mean()) * PRIOR_CONFIDENCE_OFF
    n["def_delta"] = (n["def_delta_raw"] - n["def_delta_raw"].mean()) * PRIOR_CONFIDENCE_DEF

    n["off_final"] = n["off_rating"] + n["off_delta"]
    n["def_final"] = n["def_rating"] + n["def_delta"]
    n["net_final"] = n["off_final"] - n["def_final"]

    n["off_rank"] = n["off_final"].rank(ascending=False).astype(int)
    n["def_rank"] = n["def_final"].rank(ascending=True).astype(int)   # lower EPA allowed is better
    n["net_rank"] = n["net_final"].rank(ascending=False).astype(int)
    return n.sort_values("net_final", ascending=False).reset_index(drop=True)
