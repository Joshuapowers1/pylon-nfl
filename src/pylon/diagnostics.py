"""
PYLON :: diagnostics
Transparency layer. Two outputs.

1. DECOMPOSITION. For every team, how much of the final rating comes from
   measured play-by-play and how much comes from the hand-written offseason
   prior. A model whose headline calls are driven mostly by its author's
   opinions is an opinion with extra steps, and the only honest way to handle
   that is to publish the split.

2. SENSITIVITY. Re-run the ratings across a grid of prior-confidence weights
   and report how far each team's projected wins moves. Teams that swing hard
   are the ones where the model is least trustworthy.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowers · @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import OUT, CONF, PRIOR_CONFIDENCE_OFF, PRIOR_CONFIDENCE_DEF


def decompose(adj: pd.DataFrame, cal: dict) -> pd.DataFrame:
    ppr = cal["points_per_rating"]
    d = adj.copy()
    d["pts_from_data"] = ppr * (d["off_rating"] - d["def_rating"])
    d["pts_from_prior"] = ppr * (d["off_delta"] - d["def_delta"])
    d["pts_total"] = d["pts_from_data"] + d["pts_from_prior"]
    denom = d["pts_from_data"].abs() + d["pts_from_prior"].abs()
    d["prior_share"] = (d["pts_from_prior"].abs() / denom.replace(0, np.nan)).fillna(0)
    cols = ["team", "pts_total", "pts_from_data", "pts_from_prior", "prior_share", "prior_note"]
    out = d[cols].sort_values("pts_total", ascending=False).round(3)
    out.to_csv(OUT / "rating_decomposition.csv", index=False)
    return out


def sensitivity(sr, ctx26, priors, cal, ratings_mod, adj_mod, sim_mod,
                grid=(0.0, 0.45, 0.85, 1.25), n_sims=2500) -> pd.DataFrame:
    """How much does each team's projected win total move as the confidence in
    the hand-written priors is dialled from zero to 1.25x?"""
    import copy
    from . import config as C
    rows = []
    base_off, base_def = C.PRIOR_CONFIDENCE_OFF, C.PRIOR_CONFIDENCE_DEF
    try:
        for w in grid:
            C.PRIOR_CONFIDENCE_OFF = w
            C.PRIOR_CONFIDENCE_DEF = w * (base_def / base_off)
            import importlib
            importlib.reload(adj_mod)
            a = adj_mod.apply_priors(ratings_mod.net_ratings(ratings_mod.blend(sr), ctx26), priors)
            res, _, _ = sim_mod.simulate(sim_mod.load_schedule(), a, cal, n_sims=n_sims)
            res["prior_weight"] = w
            rows.append(res[["team", "prior_weight", "proj_wins"]])
    finally:
        C.PRIOR_CONFIDENCE_OFF, C.PRIOR_CONFIDENCE_DEF = base_off, base_def
        import importlib; importlib.reload(adj_mod)
    long = pd.concat(rows, ignore_index=True)
    wide = long.pivot(index="team", columns="prior_weight", values="proj_wins").round(2)
    wide.columns = [f"w_{c:.2f}" for c in wide.columns]
    wide["swing"] = (wide.max(axis=1) - wide.min(axis=1)).round(2)
    wide = wide.sort_values("swing", ascending=False).reset_index()
    wide.to_csv(OUT / "sensitivity_priors.csv", index=False)
    return wide
