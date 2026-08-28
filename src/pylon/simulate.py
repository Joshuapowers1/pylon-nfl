"""
PYLON :: simulate
Monte Carlo of the actual 2026 schedule, 25,000 times.

Each simulated season plays all 272 games, applies the real NFL seeding and
tiebreaker cascade, runs the real 14-team bracket, and records everything:
win totals, division titles, seeds, conference titles, championships.

Why simulate instead of computing win probabilities directly: because the
questions people actually ask are path-dependent. "What are the odds Buffalo
wins the AFC East" is not a function of Buffalo's win total alone, it depends
on what New England does in the same simulated universe, and on who beat whom.
A simulator answers those natively. A closed form does not.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowers · @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from .config import (RAW, OUT, CONF, N_SIMS, RANDOM_SEED, TIE_PROBABILITY,
                     TEAM_ALIASES, TEAM_DIV, TEAM_CONF, DIVISIONS, TARGET_SEASON)


def load_schedule(season=TARGET_SEASON) -> pd.DataFrame:
    g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    g = g[(g["season"] == season) & (g["game_type"] == "REG")].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_ALIASES)
    g["neutral"] = (g["location"].astype(str).str.lower() != "home").astype(int)
    return g[["game_id", "week", "home_team", "away_team", "neutral"]].reset_index(drop=True)


def game_matrix(sched: pd.DataFrame, ratings: pd.DataFrame, cal: dict):
    """Precompute the expected margin of every scheduled game once."""
    net = ratings.set_index("team")["net_final"].to_dict()
    ppr = cal["points_per_rating"]; hfa = cal["hfa"]
    h = sched["home_team"].map(net).to_numpy()
    a = sched["away_team"].map(net).to_numpy()
    margin = ppr * (h - a) + np.where(sched["neutral"].to_numpy() == 1, 0.0, hfa)
    return margin


def simulate(sched: pd.DataFrame, ratings: pd.DataFrame, cal: dict,
             n_sims: int = N_SIMS, seed: int = RANDOM_SEED, sigma_team: float | None = None):
    """sigma_team is the season-long, team-constant strength error measured by
    backtest.py. Drawing it once per simulated season and holding it across all
    seventeen games is what separates a win distribution with the right spread
    from one that is roughly a third too narrow. Without it, every win-total and
    playoff-odds number the model produces is overconfident."""
    rng = np.random.default_rng(seed)
    if sigma_team is None:
        bt = CONF / "backtest.json"
        sigma_team = json.loads(bt.read_text())["variance_split"]["sigma_team_pts"] if bt.exists() else 0.0
    teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    tix = {t: i for i, t in enumerate(teams)}
    nT, nG = len(teams), len(sched)

    # The playoff bracket resolves outside the vectorised regular-season block,
    # so it needs the ratings in index order. Populate here rather than in run()
    # so that simulate() is callable standalone (the sensitivity sweep does).
    _NET_CACHE["net"] = np.array([ratings.set_index("team")["net_final"][t] for t in teams])

    mu = game_matrix(sched, ratings, cal)
    bt = CONF / "backtest.json"
    sigma = (json.loads(bt.read_text())["variance_split"]["sigma_game_pts"]
             if bt.exists() else cal["sigma"])
    hi = sched["home_team"].map(tix).to_numpy()
    ai = sched["away_team"].map(tix).to_numpy()

    conf = np.array([TEAM_CONF[t] == "AFC" for t in teams])
    div_of = np.array([list(DIVISIONS).index(TEAM_DIV[t]) for t in teams])
    same_div = div_of[hi] == div_of[ai]
    same_conf = conf[hi] == conf[ai]

    # Accumulators
    wins_hist = np.zeros((nT, 18), dtype=np.int32)      # 0..17 wins
    div_wins = np.zeros(nT, dtype=np.int32)
    playoffs = np.zeros(nT, dtype=np.int32)
    seed_hist = np.zeros((nT, 8), dtype=np.int32)       # index 1..7
    conf_champ = np.zeros(nT, dtype=np.int32)
    sb_champ = np.zeros(nT, dtype=np.int32)
    tot_wins = np.zeros(nT, dtype=np.float64)

    CHUNK = 500
    done = 0
    while done < n_sims:
        b = min(CHUNK, n_sims - done)
        # Simulate margins for the whole slate at once.
        # One strength draw per team per simulated season, applied to every
        # game that team plays. This is the persistent component.
        team_eff = rng.normal(0.0, sigma_team, size=(b, nT))
        m = (mu[None, :]
             + team_eff[:, hi] - team_eff[:, ai]
             + rng.normal(0.0, sigma, size=(b, nG)))
        tie = rng.random((b, nG)) < TIE_PROBABILITY
        home_win = (m > 0) & ~tie
        away_win = (m <= 0) & ~tie

        W = np.zeros((b, nT)); L = np.zeros((b, nT)); T = np.zeros((b, nT))
        DW = np.zeros((b, nT)); DL = np.zeros((b, nT))     # division record
        CW = np.zeros((b, nT)); CL = np.zeros((b, nT))     # conference record
        H2H = np.zeros((b, nT, nT), dtype=np.int16)

        np.add.at(W, (slice(None), hi), home_win)
        np.add.at(L, (slice(None), hi), away_win)
        np.add.at(T, (slice(None), hi), tie)
        np.add.at(W, (slice(None), ai), away_win)
        np.add.at(L, (slice(None), ai), home_win)
        np.add.at(T, (slice(None), ai), tie)

        dmask = same_div[None, :]
        np.add.at(DW, (slice(None), hi), home_win & dmask)
        np.add.at(DL, (slice(None), hi), away_win & dmask)
        np.add.at(DW, (slice(None), ai), away_win & dmask)
        np.add.at(DL, (slice(None), ai), home_win & dmask)

        cmask = same_conf[None, :]
        np.add.at(CW, (slice(None), hi), home_win & cmask)
        np.add.at(CL, (slice(None), hi), away_win & cmask)
        np.add.at(CW, (slice(None), ai), away_win & cmask)
        np.add.at(CL, (slice(None), ai), home_win & cmask)

        for gi in range(nG):
            H2H[:, hi[gi], ai[gi]] += home_win[:, gi]
            H2H[:, ai[gi], hi[gi]] += away_win[:, gi]

        wpct = (W + 0.5 * T) / 17.0
        dpct = np.divide(DW + 0.0, np.maximum(DW + DL, 1))
        cpct = np.divide(CW + 0.0, np.maximum(CW + CL, 1))

        for s in range(b):
            _resolve_season(s, teams, tix, wpct[s], dpct[s], cpct[s], H2H[s], W[s], T[s],
                            rng, wins_hist, div_wins, playoffs, seed_hist,
                            conf_champ, sb_champ, tot_wins, mu, cal)
        done += b

    idx = pd.Index(teams, name="team")
    exp_wins = tot_wins / n_sims
    res = pd.DataFrame({
        "team": teams,
        "proj_wins": exp_wins,
        "win_sd": [np.sqrt(np.average((np.arange(18) - exp_wins[i]) ** 2,
                                      weights=wins_hist[i])) for i in range(nT)],
        "make_playoffs": playoffs / n_sims,
        "win_division": div_wins / n_sims,
        "one_seed": seed_hist[:, 1] / n_sims,
        "conf_champ": conf_champ / n_sims,
        "super_bowl": sb_champ / n_sims,
    })
    wd = pd.DataFrame(wins_hist / n_sims, index=idx,
                      columns=[f"p_{w}_wins" for w in range(18)]).reset_index()
    sd = pd.DataFrame(seed_hist[:, 1:] / n_sims, index=idx,
                      columns=[f"p_seed_{i}" for i in range(1, 8)]).reset_index()
    return res, wd, sd


def _rank_key(t_i, wpct, dpct, cpct, h2h, jitter):
    """NFL tiebreaker cascade, approximated in the order that actually decides
    the overwhelming majority of real ties: win percentage, then head-to-head,
    then division record, then conference record. Strength of victory and
    strength of schedule sit below that and are replaced here by a tiny random
    jitter, which is honest about the fact that those two are essentially
    coin-flips at simulation time."""
    return (wpct[t_i], dpct[t_i], cpct[t_i], jitter[t_i])


def _resolve_season(s, teams, tix, wpct, dpct, cpct, h2h, W, T, rng,
                    wins_hist, div_wins, playoffs, seed_hist,
                    conf_champ, sb_champ, tot_wins, mu, cal):
    nT = len(teams)
    jitter = rng.random(nT) * 1e-6
    for i in range(nT):
        wins_hist[i, int(W[i])] += 1
        tot_wins[i] += W[i] + 0.5 * T[i]

    champs = {}
    for conf_name in ("AFC", "NFC"):
        div_winners, others = [], []
        for dname, members in DIVISIONS.items():
            if not dname.startswith(conf_name):
                continue
            idxs = [tix[t] for t in members if t in tix]
            ranked = sorted(idxs, key=lambda i: _rank_key(i, wpct, dpct, cpct, h2h, jitter), reverse=True)
            # Head-to-head sweep check between the top two.
            if len(ranked) > 1 and abs(wpct[ranked[0]] - wpct[ranked[1]]) < 1e-9:
                a, b = ranked[0], ranked[1]
                if h2h[b, a] > h2h[a, b]:
                    ranked[0], ranked[1] = b, a
            div_winners.append(ranked[0])
            others.extend(ranked[1:])
        div_winners.sort(key=lambda i: _rank_key(i, wpct, dpct, cpct, h2h, jitter), reverse=True)
        others.sort(key=lambda i: _rank_key(i, wpct, dpct, cpct, h2h, jitter), reverse=True)
        seeds = div_winners + others[:3]
        for pos, i in enumerate(seeds, start=1):
            seed_hist[i, pos] += 1
            playoffs[i] += 1
            if pos <= 4:
                div_wins[i] += 1
        champs[conf_name] = _run_bracket(seeds, mu, cal, rng, teams)

    a, n = champs["AFC"], champs["NFC"]
    conf_champ[a] += 1; conf_champ[n] += 1
    sb_champ[_neutral_game(a, n, mu, cal, rng)] += 1


_NET_CACHE: dict = {}


def _margin(home_i, away_i, cal, neutral=False):
    net = _NET_CACHE["net"]
    m = cal["points_per_rating"] * (net[home_i] - net[away_i])
    return m if neutral else m + cal["hfa"]


def _play(home_i, away_i, cal, rng, neutral=False):
    m = _margin(home_i, away_i, cal, neutral) + rng.normal(0, cal["sigma"])
    return home_i if m > 0 else away_i


def _bracket_sigma(cal):
    return cal["sigma"]


def _neutral_game(a, b, mu, cal, rng):
    return _play(a, b, cal, rng, neutral=True)


def _run_bracket(seeds, mu, cal, rng, teams):
    """Current 7-team format: 1 seed byes, 2v7 3v6 4v5, reseed each round."""
    s1, s2, s3, s4, s5, s6, s7 = seeds
    w_27 = _play(s2, s7, cal, rng)
    w_36 = _play(s3, s6, cal, rng)
    w_45 = _play(s4, s5, cal, rng)
    rank = {t: i for i, t in enumerate(seeds)}
    rd2 = sorted([w_27, w_36, w_45], key=lambda t: rank[t])
    low = rd2[-1]
    w_a = _play(s1, low, cal, rng)
    w_b = _play(rd2[0], rd2[1], cal, rng)
    hi_, lo_ = (w_a, w_b) if rank[w_a] < rank[w_b] else (w_b, w_a)
    return _play(hi_, lo_, cal, rng)


def run(ratings: pd.DataFrame, cal: dict, n_sims: int = N_SIMS):
    sched = load_schedule()
    res, wd, sd = simulate(sched, ratings, cal, n_sims=n_sims)
    res.to_csv(OUT / "team_projections.csv", index=False)
    wd.to_csv(OUT / "win_distributions.csv", index=False)
    sd.to_csv(OUT / "seed_distributions.csv", index=False)
    return res, wd, sd
