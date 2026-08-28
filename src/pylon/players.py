"""
PYLON :: players
Player-level fantasy projections, built bottom-up from opportunity.

The chain is deliberately explicit, because every step is a place where a
projection can be wrong and hidden:

    team plays/game        <- pace model (adjustments.project_context)
      x dropback rate      <- scheme model, adjusted for new play-callers
      = team pass attempts
      x player target share <- empirical-Bayes shrunk usage, re-normalised
      = targets
      x catch rate, yards per target <- shrunk efficiency
      = receptions, receiving yards
    + the same chain on the ground
    + a touchdown model driven by red-zone opportunity share, not last year's
      touchdown count
      = fantasy points

Two ideas do most of the work.

EMPIRICAL BAYES. A player's observed share is shrunk toward a positional prior
by sample size: share_hat = (n*observed + k*prior) / (n + k). A back with 40
carries does not get credited with the same conviction as one with 300.

WITHIN-TEAM NORMALISATION. After shrinkage, every team's shares are rescaled to
sum to the league-typical total. This is what makes roster movement work
without hand-writing every depth chart: when New England adds A.J. Brown at a
24.5% target share, everyone else on New England gets squeezed proportionally,
automatically. Competition for touches is modelled as the zero-sum thing it is.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowers · @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import RAW, PROC, OUT, CONF, SCORING, REPLACEMENT_RANK, TEAM_ALIASES, RANDOM_SEED

# Generational suffixes differ across every football data source, so names are
# normalised the same way everywhere in PYLON.
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm_name(s) -> str:
    s = str(s).lower().strip().replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(w for w in s.split() if w not in _SUFFIXES)

# Shrinkage strength: the number of prior "observations" the league average is
# worth. Higher = more regression. Targets are noisier per unit than carries.
K_TARGET, K_RUSH, K_EFF = 45.0, 55.0, 70.0

# Depth-aware priors. An earlier build used one flat prior per position, which
# quietly broke the whole model: a 12-man receiver room times a 5.5% prior sums
# past 100% of the team's targets before a single starter is counted, and the
# within-team normalisation then halves everybody. The prior a player gets has
# to depend on where he sits on the depth chart, and the depth chart is inferred
# from measured history plus explicit role overrides. Values are the league-wide
# average share by within-team positional rank, 2023-2025.
DEPTH_PRIOR = {
    "WR": [0.215, 0.165, 0.110, 0.062, 0.030, 0.014, 0.006],
    "TE": [0.135, 0.052, 0.020, 0.008],
    "RB": [0.520, 0.230, 0.090, 0.035, 0.012],       # of RB carries
    "RB_tgt": [0.105, 0.055, 0.022, 0.010, 0.004],   # of team targets
    "QB": [0.0],
}
ROSTER_DEPTH_CAP = {"WR": 7, "TE": 4, "RB": 5, "QB": 3}
PRIOR_SHARE = {"WR": 0.055, "TE": 0.045, "RB": 0.035, "QB": 0.0, "FB": 0.010}
PRIOR_EFF = {
    "WR": {"catch_rate": 0.635, "yards_per_target": 7.9, "ypc": 5.5},
    "TE": {"catch_rate": 0.700, "yards_per_target": 7.1, "ypc": 4.0},
    "RB": {"catch_rate": 0.760, "yards_per_target": 6.2, "ypc": 4.3},
    "QB": {"catch_rate": 0.650, "yards_per_target": 7.0, "ypc": 4.6},
}
# League totals the within-team shares are normalised to. Every target goes to
# somebody, so the receiving pool sums to ~1.00; the small shortfall is passes
# to linemen and direct-snap gadget plays. The rushing pool is defined below as
# the NON-quarterback share of designed runs, so it also sums to ~1.00 within
# that pool. Setting these below 1.0 was quietly deflating every skill player
# by 5-10% on top of the shrinkage.
TARGET_SHARE_TOTAL, RUSH_SHARE_TOTAL = 0.990, 1.000
# Share of designed (non-dropback) runs that go to quarterbacks league-wide.
# Quarterback rushing is projected separately off dropbacks, so this slice is
# removed from the pool the skill players divide up, and never subtracted twice.
QB_DESIGNED_RUN_SHARE = 0.095
# Expected games by position, before any player-specific override. These are
# risk-adjusted: the number is an expectation across injury outcomes, not a
# healthy-season assumption. It is the main reason PYLON's point totals sit
# below the industry sets, which almost all project a flat 17 games for everyone.
DEFAULT_GAMES = {"QB": 15.8, "RB": 14.9, "WR": 15.4, "TE": 15.3}
SEASON_W = {2025: 1.0, 2024: 0.42}


def load_roster(season=2026) -> pd.DataFrame:
    r = pd.read_csv(RAW / f"roster_{season}.csv", low_memory=False)
    r = r[r["status"].isin(["ACT", "RES"])].copy()
    r["team"] = r["team"].replace({"AZ": "ARI", **TEAM_ALIASES})
    r = r[r["position"].isin(["QB", "RB", "WR", "TE", "FB"])]
    r["position"] = r["position"].replace({"FB": "RB"})
    return r[["gsis_id", "full_name", "team", "position", "birth_date", "depth_chart_position"]] \
        .rename(columns={"gsis_id": "player_id", "full_name": "player"})


LATE_WEIGHT = 0.40      # weight on the weeks 13-18 role when a player has one
LATE_FULL_GAMES = 5     # games in the window before the late role is fully trusted


def apply_late_season(base: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    """Blend the December role into the season-long role.

    This exists because of a specific, repeatable failure. Cam Skattebo's
    full-2025 carry share is 22%, which is what you get by averaging a September
    backup with a December starter. The market drafts him in the fourth round;
    a model that only sees the average has him as a bench body. Same shape for
    every midseason role change in football. Weighting the last six weeks fixes
    it without hand-writing a depth chart for all 32 teams."""
    if late is None or len(late) == 0:
        return base
    lb = blend_usage(late)
    cols = ["target_share", "rush_share", "rz_target_share", "rz_rush_share", "dropback_share"]
    m = base.merge(lb[["player_id"] + cols + ["n_games_played"]],
                   on="player_id", how="left", suffixes=("", "_late"))
    # Trust the late window in proportion to how much of it the player was
    # available for. Three December games is a hint, six is a role.
    gp = m["n_games_played_late"].fillna(0)
    w = np.clip(gp / LATE_FULL_GAMES, 0, 1) * LATE_WEIGHT
    for c in cols:
        late_v = m[c + "_late"].fillna(m[c])
        m[c] = (1 - w) * m[c] + w * late_v
    m["late_games"] = gp
    m["late_role_delta"] = (m["rush_share_late"].fillna(0) - base["rush_share"].fillna(0).to_numpy())                          + (m["target_share_late"].fillna(0) - base["target_share"].fillna(0).to_numpy())
    return m.drop(columns=[c for c in m.columns if c.endswith("_late")], errors="ignore")


def blend_usage(u: pd.DataFrame) -> pd.DataFrame:
    """Recency-weighted per-player career shares and efficiencies."""
    u = u.copy()
    u["w"] = u["season"].map(SEASON_W).astype(float).fillna(0.0)
    share_cols = ["target_share", "rush_share", "rz_target_share", "rz_rush_share", "dropback_share"]
    eff_cols = ["catch_rate", "yards_per_target", "ypc", "ypa", "td_rate_pass", "int_rate"]
    vol_cols = ["targets", "rush_att", "pass_att_p", "receptions", "rec_yards", "rush_yards",
                "rush_td", "scrambles", "scramble_yards", "scramble_td", "dropbacks", "games_played"]
    rows = []
    for pid, g in u.groupby("player_id"):
        w = g["w"].to_numpy()
        if w.sum() == 0:
            continue
        wn = w / w.sum()
        rec = {"player_id": pid, "usage_name": g["player_name"].iloc[-1],
               "last_team": g.sort_values("season")["posteam"].iloc[-1]}
        for c in share_cols + eff_cols:
            rec[c] = float(np.nansum(np.nan_to_num(g[c].to_numpy()) * wn))
        for c in vol_cols:
            rec["n_" + c] = float(np.nansum(g[c].to_numpy() * w))
        rows.append(rec)
    return pd.DataFrame(rows)


def _eb(obs, n, prior, k):
    """Empirical-Bayes shrink toward a prior with pseudo-count k."""
    obs = np.nan_to_num(np.asarray(obs, dtype=float))
    n = np.nan_to_num(np.asarray(n, dtype=float))
    return (n * obs + k * prior) / (n + k)


def active_roles(roles: dict, week: int | None = None) -> dict:
    """Drop role priors that have passed their shelf life.

    Twenty-one of the fifty-eight priors are availability claims: a hamstring, a
    PUP designation, an expected return date. Those describe September. Carried
    into November they are not a prior, they are stale hand-entry fighting real
    usage data, so past `stale_after_week` the assertion is removed and the
    measured share takes over."""
    if week is None:
        return roles
    keep = {k: v for k, v in roles["players"].items()
            if v.get("stale_after_week", 99) >= week}
    return {**roles, "players": keep}


def build_player_frame(usage: pd.DataFrame, roles: dict, usage_late: pd.DataFrame | None = None) -> pd.DataFrame:
    hist = blend_usage(usage)
    hist = apply_late_season(hist, usage_late)
    ros = load_roster()
    p = ros.merge(hist, on="player_id", how="left")

    pos = p["position"]

    # ---- infer a depth chart ----------------------------------------------
    # Role score = measured opportunity, with an explicit override able to
    # promote a rookie who has none.
    ovr_share = p["player"].map(_norm_name).map(
        {_norm_name(k): max(v.get("target_share", 0), v.get("rush_share", 0)) for k, v in roles["players"].items()}
    ).fillna(0.0)
    p["role_score"] = np.maximum(
        np.nan_to_num(p["target_share"]) + np.nan_to_num(p["rush_share"]) * 0.55,
        ovr_share,
    ) + np.nan_to_num(p["n_targets"] + p["n_rush_att"]) * 1e-6
    p["depth"] = p.groupby(["team", "position"])["role_score"].rank(ascending=False, method="first") - 1
    p["depth"] = p["depth"].astype(int)

    def _depth_prior(row, key):
        tbl = DEPTH_PRIOR.get(key, [0.02])
        d = int(row["depth"])
        return tbl[d] if d < len(tbl) else tbl[-1] * 0.4

    p["prior_share"] = p.apply(lambda r: _depth_prior(r, "RB_tgt" if r["position"] == "RB" else r["position"]), axis=1)
    p["prior_rush_share"] = p.apply(lambda r: _depth_prior(r, "RB") if r["position"] == "RB" else 0.004, axis=1)

    # Anyone buried past the fantasy-relevant depth of their room is projected
    # to zero opportunity rather than a small positive share, because a small
    # positive share for forty players per team is where phantom volume comes from.
    cap = pos.map(ROSTER_DEPTH_CAP).fillna(3)
    p["in_pool"] = p["depth"] < cap

    for c, key in [("catch_rate", "catch_rate"), ("yards_per_target", "yards_per_target"), ("ypc", "ypc")]:
        p["prior_" + c] = pos.map(lambda x: PRIOR_EFF.get(x, PRIOR_EFF["WR"])[key])

    p["target_share_eb"] = _eb(p["target_share"], p["n_targets"], p["prior_share"], K_TARGET)
    p["rush_share_eb"] = _eb(p["rush_share"], p["n_rush_att"], p["prior_rush_share"], K_RUSH)
    p["rz_target_share_eb"] = _eb(p["rz_target_share"], p["n_targets"] * 0.14, p["prior_share"], K_TARGET * 0.25)
    p["rz_rush_share_eb"] = _eb(p["rz_rush_share"], p["n_rush_att"] * 0.14,
                                p["prior_rush_share"], K_RUSH * 0.25)
    p["catch_rate_eb"] = _eb(p["catch_rate"], p["n_targets"], p["prior_catch_rate"], K_EFF)
    p["ypt_eb"] = _eb(p["yards_per_target"], p["n_targets"], p["prior_yards_per_target"], K_EFF)
    p["ypc_eb"] = _eb(p["ypc"], p["n_rush_att"], p["prior_ypc"], K_EFF)

    # ---- role overrides ----------------------------------------------------
    p["games"] = pos.map(DEFAULT_GAMES).fillna(15.2)
    p["override"] = ""
    rp = roles["players"]
    by_name = {_norm_name(k): v for k, v in rp.items()}
    for i, row in p.iterrows():
        v = by_name.get(_norm_name(row["player"]))
        if v is None:
            continue
        p.at[i, "override"] = v.get("why", "role prior")
        p.at[i, "games"] = float(v.get("games", p.at[i, "games"]))
        if v.get("target_share", 0) > 0:
            p.at[i, "target_share_eb"] = v["target_share"]
            p.at[i, "rz_target_share_eb"] = v["target_share"] * 1.02
        if v.get("rush_share", 0) > 0:
            p.at[i, "rush_share_eb"] = v["rush_share"]
            p.at[i, "rz_rush_share_eb"] = v["rush_share"] * 1.05
        if v.get("games", 1) == 0.0:
            p.at[i, "target_share_eb"] = 0.0
            p.at[i, "rush_share_eb"] = 0.0

    # Age curve on efficiency: production decays past the positional peak.
    p["age"] = 2026 - pd.to_datetime(p["birth_date"], errors="coerce").dt.year
    peak = pos.map({"RB": 25, "WR": 26, "TE": 27, "QB": 29}).fillna(26)
    decay = pos.map({"RB": 0.020, "WR": 0.010, "TE": 0.008, "QB": 0.006}).fillna(0.010)
    over = (p["age"] - peak).clip(lower=0).fillna(0)
    p["age_mult"] = (1.0 - decay * over).clip(0.80, 1.0)
    p["ypt_eb"] *= p["age_mult"]
    p["ypc_eb"] *= p["age_mult"]
    return p


def normalise_within_team(p: pd.DataFrame) -> pd.DataFrame:
    """Rescale each team's shares so the team's projected opportunity adds up.

    This is the mechanism that prices competition for touches, and the exact
    constraint matters. Because every share is now measured per game the player
    actually played, the right identity is not

        sum(share_i) = 1

    but rather

        sum(share_i * games_i / 17) = 1

    which says the team's season-long opportunity is fully allocated across
    whoever is available each week. Normalising on the naive version taxes every
    healthy alpha to pay for the projected absences of his own backups, and
    quietly shaves 10-15% off the top of every depth chart in the league."""
    p = p.copy()
    avail = (p["games"].clip(0, 17) / 17.0)
    # Zero out everyone outside the fantasy-relevant depth of their room first.
    if "in_pool" in p.columns:
        for c in ("target_share_eb", "rush_share_eb", "rz_target_share_eb", "rz_rush_share_eb"):
            p.loc[~p["in_pool"], c] = 0.0
    skill = p["position"].isin(["WR", "TE", "RB"])
    for col, total in [("target_share_eb", TARGET_SHARE_TOTAL), ("rush_share_eb", RUSH_SHARE_TOTAL),
                       ("rz_target_share_eb", TARGET_SHARE_TOTAL), ("rz_rush_share_eb", RUSH_SHARE_TOTAL)]:
        weighted = p.loc[skill, col] * avail.loc[skill]
        s = weighted.groupby(p.loc[skill, "team"]).transform("sum")
        p.loc[skill, col] = p.loc[skill, col] / s.replace(0, np.nan) * total
    return p.fillna({c: 0 for c in ["target_share_eb", "rush_share_eb", "rz_target_share_eb", "rz_rush_share_eb"]})


def project(p: pd.DataFrame, ctx26: pd.DataFrame, roles: dict) -> pd.DataFrame:
    c = ctx26.set_index("team")
    g = p["games"].clip(0, 17)
    tm = p["team"]

    team_pass = tm.map(c["proj_pass_att_per_game"])
    team_rush_all = tm.map(c["proj_rush_att_per_game"])
    team_rb_rush = (team_rush_all * (1.0 - QB_DESIGNED_RUN_SHARE)).clip(lower=8.0)
    team_td = tm.map(c["proj_off_td_per_game"])
    # League split of offensive touchdowns, 2023-2025: 61% pass, 39% rush.
    team_pass_td, team_rush_td = team_td * 0.61, team_td * 0.39

    p = p.copy()
    p["targets"] = team_pass * p["target_share_eb"] * g
    p["receptions"] = p["targets"] * p["catch_rate_eb"]
    p["rec_yards"] = p["targets"] * p["ypt_eb"]
    p["rush_att"] = np.where(p["position"] == "QB", 0.0, team_rb_rush * p["rush_share_eb"] * g)
    p["rush_yards"] = p["rush_att"] * p["ypc_eb"]

    # Touchdowns from opportunity, not from last year's touchdown total. A 60/40
    # blend of red-zone share and overall share, because red-zone samples are
    # small enough to be badly noisy on their own.
    p["rec_td"] = team_pass_td * g * (0.60 * p["rz_target_share_eb"] + 0.40 * p["target_share_eb"])
    p["rush_td"] = np.where(p["position"] == "QB", 0.0,
                            team_rush_td * g * (0.60 * p["rz_rush_share_eb"] + 0.40 * p["rush_share_eb"]))

    # ---- quarterbacks ------------------------------------------------------
    qd = roles["qb_depth_2026"]
    ov = qd["overrides"]
    is_qb = p["position"] == "QB"
    qb_rank = p[is_qb].assign(sc=lambda d: d["dropback_share"].fillna(0) * 1000 + d["n_pass_att_p"].fillna(0)) \
                      .sort_values("sc", ascending=False).groupby("team").cumcount()
    p["qb_depth"] = 99
    p.loc[qb_rank.index, "qb_depth"] = qb_rank.values
    starter_share = p["team"].map(lambda t: ov.get(t, qd["default_starter_share"]))
    share = np.where(p["qb_depth"] == 0, starter_share,
             np.where(p["qb_depth"] == 1, 1.0 - starter_share, 0.0))
    ovr_qb = {_norm_name(k): v for k, v in roles["players"].items() if v.get("pos") == "QB"}
    for i, row in p[is_qb].iterrows():
        v = ovr_qb.get(_norm_name(row["player"]))
        if v and v.get("games") is not None:
            pass  # games already applied; share stays depth-chart driven

    p["pass_att"] = np.where(is_qb, team_pass * share * g, 0.0)
    ypa = _eb(p["ypa"], p["n_pass_att_p"], 7.10, 160.0)
    tdr = _eb(p["td_rate_pass"], p["n_pass_att_p"], 0.0455, 260.0)
    intr = _eb(p["int_rate"], p["n_pass_att_p"], 0.0245, 260.0)
    p["pass_yards"] = p["pass_att"] * ypa
    p["pass_td"] = p["pass_att"] * tdr
    p["interceptions"] = p["pass_att"] * intr
    # ---- quarterback rushing ----------------------------------------------
    # Carries per dropback, counting BOTH designed runs and scrambles, shrunk
    # toward the league rate. Then a player-specific touchdown rate on those
    # carries, also shrunk: goal-line quarterback rushing is a real, repeatable
    # skill and the single most mispriced input in fantasy quarterback scoring.
    db = p["n_dropbacks"].replace(0, np.nan)
    carries_hist = p["n_rush_att"] + p["n_scrambles"]
    qb_rush_rate = _eb(np.nan_to_num(carries_hist / db), p["n_dropbacks"], 0.095, 220.0)
    proj_db = p["pass_att"] / 0.875
    p["qb_rush_att"] = np.where(is_qb, proj_db * qb_rush_rate, 0.0)

    qb_ypc_hist = np.nan_to_num((p["n_rush_yards"] + p["n_scramble_yards"]) / carries_hist.replace(0, np.nan))
    qb_ypc = _eb(qb_ypc_hist, carries_hist, 4.55, 90.0)
    p["qb_rush_yards"] = p["qb_rush_att"] * np.clip(qb_ypc, 2.0, 7.5)

    qb_td_rate_hist = np.nan_to_num((p["n_rush_td"] + p["n_scramble_td"]) / carries_hist.replace(0, np.nan))
    qb_td_rate = _eb(qb_td_rate_hist, carries_hist, 0.048, 75.0)
    p["qb_rush_td"] = p["qb_rush_att"] * np.clip(qb_td_rate, 0.0, 0.16)
    p.loc[is_qb, "rush_att"] = p.loc[is_qb, "qb_rush_att"]
    p.loc[is_qb, "rush_yards"] = p.loc[is_qb, "qb_rush_yards"]
    p.loc[is_qb, "rush_td"] = p.loc[is_qb, "qb_rush_td"]

    p["fumbles_lost"] = (p["rush_att"] * 0.006 + p["receptions"] * 0.004 + p["pass_att"] * 0.008)

    s = SCORING
    p["proj_points"] = (
        p["pass_yards"] * s["pass_yd"] + p["pass_td"] * s["pass_td"] + p["interceptions"] * s["interception"]
        + p["rush_yards"] * s["rush_yd"] + p["rush_td"] * s["rush_td"]
        + p["receptions"] * s["rec"] + p["rec_yards"] * s["rec_yd"] + p["rec_td"] * s["rec_td"]
        + p["fumbles_lost"] * s["fumble_lost"]
    )
    p["proj_ppg"] = p["proj_points"] / p["games"].replace(0, np.nan)
    # A healthy-season pace figure, for comparison against published projection
    # sets. Almost every industry set assumes 17 games for every player, so
    # PYLON's risk-adjusted totals look low next to them by construction. This
    # column puts the two on the same footing without pretending injuries do
    # not happen: proj_points remains the number the draft board runs on.
    p["proj_points_17g"] = p["proj_ppg"] * 17.0
    return p


def simulate_players(p: pd.DataFrame, n_sims=8000, seed=RANDOM_SEED) -> pd.DataFrame:
    """Per-player Monte Carlo. Three independent sources of variance:
    availability (games actually played), usage (share drift within the season),
    and efficiency (yards and touchdowns per opportunity)."""
    rng = np.random.default_rng(seed)
    n = len(p)
    base = p["proj_points"].to_numpy()
    g = p["games"].to_numpy()

    # Availability: beta-binomial-ish. Expected games are the mean, and the
    # spread widens for players already carrying an injury flag.
    flagged = (p["override"] != "").to_numpy()
    avail_sd = np.where(flagged, 3.4, 2.5)
    games_sim = np.clip(rng.normal(g[:, None], avail_sd[:, None], size=(n, n_sims)), 0, 17)
    avail_mult = np.divide(games_sim, np.maximum(g[:, None], 1e-9))

    # Usage drift is lognormal and heavier for low-share players, whose roles
    # are the least locked in.
    share = p[["target_share_eb", "rush_share_eb"]].max(axis=1).to_numpy()
    usage_sd = np.clip(0.34 - 0.55 * share, 0.14, 0.34)
    usage_mult = rng.lognormal(-0.5 * usage_sd[:, None] ** 2, usage_sd[:, None], size=(n, n_sims))

    eff_sd = np.where(p["position"].to_numpy() == "QB", 0.11, 0.17)
    eff_mult = rng.lognormal(-0.5 * eff_sd[:, None] ** 2, eff_sd[:, None], size=(n, n_sims))

    sims = base[:, None] * avail_mult * usage_mult * eff_mult
    out = p.copy()
    for q, name in [(10, "p10"), (25, "p25"), (50, "p50"), (75, "p75"), (90, "p90")]:
        out[name] = np.percentile(sims, q, axis=1)
    out["sim_mean"] = sims.mean(axis=1)
    out["_sims"] = list(sims)
    return out


def positional_finish_odds(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Share of simulated seasons in which each player finishes top-3, top-5 and
    top-12 at their own position. This is the number that should drive a draft
    board, and almost nobody publishes it."""
    out = sim_df.copy()
    for col in ("p_top3", "p_top5", "p_top12", "p_bust"):
        out[col] = 0.0
    for posn, grp in sim_df.groupby("position"):
        if posn not in REPLACEMENT_RANK:
            continue
        M = np.vstack(grp["_sims"].to_numpy())          # players x sims
        order = (-M).argsort(axis=0)
        ranks = np.empty_like(order)
        rows = np.arange(M.shape[0])[:, None]
        np.put_along_axis(ranks, order, rows, axis=0)
        rep = REPLACEMENT_RANK[posn]
        out.loc[grp.index, "p_top3"] = (ranks < 3).mean(axis=1)
        out.loc[grp.index, "p_top5"] = (ranks < 5).mean(axis=1)
        out.loc[grp.index, "p_top12"] = (ranks < 12).mean(axis=1)
        out.loc[grp.index, "p_bust"] = (ranks >= rep).mean(axis=1)
    return out


# Streaming premium, in fantasy points per game. MEASURED, and the measurement
# went the other way.
#
# The reasoning that produced this constant was: naive value over replacement
# assumes you had to draft the player, and for a streamable position you did
# not, so replacement should sit above the season-long QB14 and quarterbacks
# should therefore fall down the board. That is the conventional wisdom and it
# is why every VORP sheet gets overridden by hand.
#
# `streaming.measure()` replays it instead of assuming it. Each week a simulated
# manager starts whichever unrostered quarterback faces the softest pass
# defense, rated only on games already played, with no sight of the outcome.
# Across 2023-2025 that manager averaged 17.1 points per game against 18.8 for
# the quarterback who finished 14th: a premium of MINUS 1.7.
#
# Streaming is worse than rostering the player who ends up at replacement rank.
# The waiver pool is a real downgrade, not a free option. So the honest premium
# is zero or negative, the fudge factor is deleted, and the board goes back to
# ranking quarterbacks where the arithmetic actually puts them. That disagrees
# with how most people draft, which is the entire reason this project exists.
#
# `stream=True` is kept so the measured value can be applied, and the draft room
# exposes it as a toggle for anyone who trusts the convention over the replay.
# See config/streaming_premium.json for the run.
STREAM_PREMIUM_PPG = {"QB": -1.72, "TE": 0.0, "RB": 0.0, "WR": 0.0}
GAMES_IN_SEASON = 17


def replacement_levels(df: pd.DataFrame, ranks: dict | None = None,
                       stream: bool = False) -> dict:
    """Season points a freely available player at each position is worth."""
    ranks = ranks or REPLACEMENT_RANK
    out = {}
    for posn, rep in ranks.items():
        g = df[df["position"] == posn].sort_values("proj_points", ascending=False)
        if not len(g):
            out[posn] = 0.0
            continue
        naive = float(g["proj_points"].iloc[min(rep, len(g)) - 1])
        prem = STREAM_PREMIUM_PPG.get(posn, 0.0) * GAMES_IN_SEASON if stream else 0.0
        out[posn] = naive + prem
    return out


def add_vorp(df: pd.DataFrame, ranks: dict | None = None, stream: bool = False) -> pd.DataFrame:
    """Value over replacement, the only currency in which a quarterback and a
    running back can be compared."""
    df = df.copy()
    df["vorp"] = np.nan
    base = replacement_levels(df, ranks, stream)
    for posn, baseline in base.items():
        g = df[df["position"] == posn]
        df.loc[g.index, "vorp"] = g["proj_points"] - baseline
    df["pos_rank"] = df.groupby("position")["proj_points"].rank(ascending=False, method="min")
    df["overall_value_rank"] = df["vorp"].rank(ascending=False, method="min")
    df.attrs["replacement"] = base
    return df
