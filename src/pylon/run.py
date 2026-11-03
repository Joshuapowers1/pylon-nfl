"""
PYLON :: run
End-to-end pipeline. One command reproduces every published number.

    python -m pylon.run --sims 25000

Stages: ingest -> ratings -> offseason priors -> calibration -> in-season
blend -> season Monte Carlo (completed games locked) -> player projections
(usage blended, rest of season) -> market edge board -> sealed ledgers ->
diagnostics.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowerss · @joshuampowers
"""
from __future__ import annotations
import argparse, json, warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from . import (ingest, ratings, adjustments, simulate, players, market, diagnostics,
               draft, backtest, matchups, ledger, weekly, inseason, usage as usage_mod)
from .config import PROC, OUT, CONF, TRAIN_SEASONS, N_SIMS, TEAM_DIV, TEAM_CONF, TARGET_SEASON


def current_week(season: int = TARGET_SEASON) -> int:
    """The week the league is actually in.

    This was hardcoded to 1, which was true in August and quietly wrong from
    the moment Week 2 started: the site kept serving Week 1 opponents under a
    heading that implied this week. A weekly job that regenerates last week
    forever is worse than no weekly job, because it looks like it is working.

    A week is "current" until its last game has a result, so Tuesday's refit
    picks up the week that is about to be played rather than the one just gone.
    """
    import pandas as pd
    from .config import RAW
    g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    g = g[(g["season"] == season) & (g["game_type"] == "REG")]
    unplayed = g[g["result"].isna()]
    return int(unplayed["week"].min()) if len(unplayed) else int(g["week"].max())


def _t(msg, t0):
    print(f"  [{time.time()-t0:6.1f}s] {msg}")


def main(n_sims=N_SIMS, rebuild=False, player_sims=8000):
    t0 = time.time()
    print("PYLON 1.1  ::  2026 NFL projection engine")
    print("=" * 72)

    if rebuild or not (PROC / "plays_clean.parquet").exists():
        ingest.build(); _t("ingested play-by-play", t0)
    if rebuild or not (PROC / "player_usage.csv").exists():
        usage_mod.build(); _t("built player usage", t0)

    plays = pd.read_parquet(PROC / "plays_clean.parquet")
    ctx = pd.read_csv(PROC / "team_context.csv")
    priors = adjustments.load_priors()
    roles = json.loads((CONF / "roles_2026.json").read_text())

    ctx26 = adjustments.project_context(ctx, priors)

    srf = PROC / "season_ratings_full.csv"
    sr = pd.read_csv(srf) if srf.exists() else ratings.season_ratings(plays)
    sr_recent = sr[sr["season"].isin(TRAIN_SEASONS)]
    _t("loaded opponent-adjusted ratings", t0)

    cal = json.loads((CONF / "calibration.json").read_text())
    trans = cal["translation"]
    print(f"  calibration: {trans['points_per_rating']} pts per rating unit | "
          f"HFA {trans['hfa']} | sigma {trans['sigma']} | "
          f"out-of-sample SU {trans['straight_up_accuracy']:.1%}")

    adj = adjustments.apply_priors(ratings.net_ratings(ratings.blend(sr_recent), ctx26), priors)

    # ---- in season --------------------------------------------------------
    # Everything above is the August model. Once a week is complete it becomes
    # the prior, and this season's plays are the evidence.
    through = inseason.completed_through()
    wk_no = current_week()
    raw26 = inseason.load_pbp() if through > 0 else None
    status = {"through_week": through, "current_week": wk_no, "applied": False}
    sigma_team = None
    if raw26 is not None and len(raw26):
        k_team = json.loads((CONF / "inseason.json").read_text())["k_best"]
        kp = inseason.player_k()
        games = inseason.team_games_played(through_week=through)
        ctx26 = inseason.blend_context(ctx26, raw26, through, kp)
        plays26 = ingest.clean_plays(raw26)
        adj, info = inseason.blend_team_ratings(adj, plays26, ctx26, through, k_team, games)
        bt = json.loads((CONF / "backtest.json").read_text())["variance_split"]
        sigma_team = inseason.posterior_sigma_team(bt["sigma_team_pts"], adj.set_index("team")["w_inseason"])
        status.update(info, applied=bool(info.get("applied")), k_usage=kp,
                      sigma_team_prior=bt["sigma_team_pts"],
                      sigma_team_mean=round(float(sigma_team.mean()), 3))
        _t(f"blended {info.get('plays_2026', 0):,} plays from weeks 1-{through} "
           f"(mean weight {info.get('w_mean', 0):.2f})", t0)
    ctx26.to_csv(PROC / "team_context_2026.csv", index=False)
    (CONF / "inseason_status.json").write_text(json.dumps(status, indent=2))

    adj["power_rating"] = (trans["points_per_rating"] * adj["net_final"]).round(2)
    adj["division"] = adj["team"].map(TEAM_DIV); adj["conference"] = adj["team"].map(TEAM_CONF)
    adj.to_csv(OUT / "team_power_ratings.csv", index=False)
    _t("built 2026 team ratings", t0)

    res, wd, sd = simulate.run(adj, trans, n_sims=n_sims, sigma_team=sigma_team)
    res = res.merge(adj[["team", "power_rating", "off_rank", "def_rank", "division", "conference",
                         "prior_note"]], on="team")
    res.to_csv(OUT / "team_projections.csv", index=False)
    _t(f"simulated {n_sims:,} seasons", t0)

    u = pd.read_csv(PROC / "player_usage.csv")
    ulate_p = PROC / "player_usage_late.csv"
    ulate = pd.read_csv(ulate_p) if ulate_p.exists() else None
    # Availability priors carry a shelf life; past it the measured share takes over.
    roles_now = players.active_roles(roles, week=wk_no) if status["applied"] else roles
    pf = players.build_player_frame(u, roles_now, ulate)
    td = None
    if status["applied"]:
        cur = inseason.season_usage(raw26, through)
        pf = players.apply_in_season(pf, cur, status["k_usage"])
        left = 17 - inseason.team_games_played(through_week=through)
        pf = players.rest_of_season(pf, left)
        td = inseason.to_date_stats(raw26, through)
    pf = players.normalise_within_team(pf)
    proj = players.project(pf, ctx26, roles_now)
    # Weekly projections need per-game rates, which the rest-of-season frame
    # carries (volume / games are both rest-of-season). Totals are built next.
    ros_frame = proj.copy()
    if td is not None:
        proj = inseason.add_to_date(proj, td)
    sim = players.simulate_players(proj, n_sims=player_sims)
    sim = players.positional_finish_odds(sim)
    sim = players.add_vorp(sim, col="proj_ros" if td is not None else "proj_points")
    keep = ["player", "team", "position", "games", "targets", "receptions", "rec_yards", "rec_td",
            "rush_att", "rush_yards", "rush_td", "pass_att", "pass_yards", "pass_td",
            "interceptions", "proj_points", "proj_ppg", "proj_points_17g", "p10", "p25", "p50", "p75", "p90",
            "p_top3", "p_top5", "p_top12", "p_bust", "vorp", "pos_rank", "overall_value_rank",
            "target_share_eb", "rush_share_eb", "depth", "override"]
    keep += [c for c in ("avail", "status", "games_2026", "games_to_date", "games_ros", "points_to_date",
                         "proj_ros", "receptions_ros", "ppg_to_date") if c in sim.columns]
    out = sim[keep].sort_values("proj_points", ascending=False)
    out.to_csv(OUT / "player_projections.csv", index=False)
    _t(f"projected {len(out):,} players with {player_sims:,} sims each", t0)

    snap = market.load_snapshot(live=status["applied"])
    board = market.edge_board(res, snap)
    market.vig_report(market.load_snapshot(live=False))
    vig = market.vig_report(snap, path=OUT / "market_vig_report_live.csv") if status["applied"] \
        else pd.read_csv(OUT / "market_vig_report.csv")
    _t(f"built market edge board ({'live' if status['applied'] else 'August'} prices)", t0)

    # The draft room is a preseason artifact. Once the season starts the ADP
    # feed thins to a few dozen in-season mock drafts (127 drafts and two
    # players on the day this guard was written), and rebuilding the board on
    # that would silently wipe the ADP from every player on the site.
    if status["applied"] and (OUT / "draft_board.csv").exists():
        dboard = pd.read_csv(OUT / "draft_board.csv")
        dinfo = {"frozen": True, "drafts_sampled": None}
        _t("draft board frozen at its preseason build", t0)
    else:
        dboard, dinfo = draft.board(out)
        _t(f"built draft board vs {dinfo['drafts_sampled']:,} live ADP drafts", t0)

    mu, scoring = matchups.build(adj, trans, week=wk_no)
    _t(f"projected {len(mu)} week {wk_no} matchups", t0)

    wk = weekly.build(ros_frame, adj, ctx26, week=wk_no)
    _t(f"projected {len(wk):,} players for week {wk_no}", t0)

    led = ledger.build(res, board, mu)
    wled = ledger.seal_week(mu, wk_no) if wk_no > 1 else {}
    sealed = sorted(int(w) for w in wled.get("weeks", {})) if wled else []
    grade = ledger.score()
    _t(f"ledger {led['digest']} {'verifies' if ledger.verify() else 'DOES NOT VERIFY'}; "
       f"weeks sealed since: {sealed or 'none'}", t0)

    dec = diagnostics.decompose(adj, trans)
    bt = json.loads((CONF / "backtest.json").read_text()) if (CONF / "backtest.json").exists() else {}
    _t("wrote diagnostics", t0)

    print("=" * 72)
    print(f"done in {time.time()-t0:.1f}s. outputs in {OUT}")
    return dict(ratings=adj, teams=res, wins=wd, seeds=sd, players=out, edges=board,
                vig=vig, decomp=dec, draft=dboard, draft_info=dinfo, backtest=bt,
                matchups=mu, scoring=scoring, ledger=led, weekly=wk, grade=grade, status=status)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=N_SIMS)
    ap.add_argument("--player-sims", type=int, default=8000)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    main(n_sims=a.sims, rebuild=a.rebuild, player_sims=a.player_sims)
