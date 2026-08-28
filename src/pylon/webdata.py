"""
PYLON :: webdata
Packs every model output into one compact JSON bundle for the web app.

Array-of-arrays with a shared header rather than array-of-objects: for 900
players it is roughly a third the bytes, and the app rehydrates it once on load.
Everything is rounded to the precision it is actually reported at, because
storing 14 decimal places of a Monte Carlo estimate is noise with a file size.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowers · @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import OUT, CONF, RAW, PROC, TEAM_ALIASES, TEAM_DIV, TEAM_CONF, DIVISIONS

TEAM_META = {
 "ARI":("Arizona","Cardinals","#97233F","#000000"),   "ATL":("Atlanta","Falcons","#A71930","#000000"),
 "BAL":("Baltimore","Ravens","#241773","#9E7C0C"),    "BUF":("Buffalo","Bills","#00338D","#C60C30"),
 "CAR":("Carolina","Panthers","#0085CA","#101820"),   "CHI":("Chicago","Bears","#0B162A","#C83803"),
 "CIN":("Cincinnati","Bengals","#FB4F14","#000000"),  "CLE":("Cleveland","Browns","#311D00","#FF3C00"),
 "DAL":("Dallas","Cowboys","#041E42","#869397"),      "DEN":("Denver","Broncos","#FB4F14","#002244"),
 "DET":("Detroit","Lions","#0076B6","#B0B7BC"),       "GB":("Green Bay","Packers","#203731","#FFB612"),
 "HOU":("Houston","Texans","#03202F","#A71930"),      "IND":("Indianapolis","Colts","#002C5F","#A2AAAD"),
 "JAX":("Jacksonville","Jaguars","#006778","#D7A22A"),"KC":("Kansas City","Chiefs","#E31837","#FFB81C"),
 "LA":("Los Angeles","Rams","#003594","#FFA300"),     "LAC":("Los Angeles","Chargers","#0080C6","#FFC20E"),
 "LV":("Las Vegas","Raiders","#000000","#A5ACAF"),    "MIA":("Miami","Dolphins","#008E97","#FC4C02"),
 "MIN":("Minnesota","Vikings","#4F2683","#FFC62F"),   "NE":("New England","Patriots","#002244","#C60C30"),
 "NO":("New Orleans","Saints","#D3BC8D","#101820"),   "NYG":("New York","Giants","#0B2265","#A71930"),
 "NYJ":("New York","Jets","#125740","#000000"),       "PHI":("Philadelphia","Eagles","#004C54","#A5ACAF"),
 "PIT":("Pittsburgh","Steelers","#FFB612","#101820"), "SEA":("Seattle","Seahawks","#002244","#69BE28"),
 "SF":("San Francisco","49ers","#AA0000","#B3995D"),  "TB":("Tampa Bay","Buccaneers","#D50A0A","#34302B"),
 "TEN":("Tennessee","Titans","#0C2340","#4B92DB"),    "WAS":("Washington","Commanders","#5A1414","#FFB612"),
}


def _tbl(df: pd.DataFrame, cols: list[str], rounding: dict | None = None) -> dict:
    d = df[cols].copy()
    rounding = rounding or {}
    for c in cols:
        if c in rounding:
            d[c] = d[c].astype(float).round(rounding[c])
            if rounding[c] == 0:
                d[c] = d[c].astype("Int64")
    d = d.replace({np.nan: None})
    return {"cols": cols, "rows": d.to_numpy().tolist()}


def build() -> dict:
    teams = pd.read_csv(OUT / "team_projections.csv")
    power = pd.read_csv(OUT / "team_power_ratings.csv")
    wd = pd.read_csv(OUT / "win_distributions.csv")
    sd = pd.read_csv(OUT / "seed_distributions.csv")
    dec = pd.read_csv(OUT / "rating_decomposition.csv")
    sens = pd.read_csv(OUT / "sensitivity_priors.csv")
    edges = pd.read_csv(OUT / "market_edge_board.csv")
    vig = pd.read_csv(OUT / "market_vig_report.csv")
    players = pd.read_csv(OUT / "player_projections.csv")
    board = pd.read_csv(OUT / "draft_board.csv")
    bt = pd.read_csv(OUT / "backtest_win_projections.csv")
    cal = json.loads((CONF / "calibration.json").read_text())
    inseason = json.loads((CONF / "inseason.json").read_text()) if (CONF / "inseason.json").exists() else {}
    scoring = json.loads((CONF / "scoring_model.json").read_text()) if (CONF / "scoring_model.json").exists() else {}
    led = json.loads((OUT / "prediction_ledger.json").read_text()) if (OUT / "prediction_ledger.json").exists() else {}
    mu = pd.read_csv(OUT / "matchups_week1.csv") if (OUT / "matchups_week1.csv").exists() else pd.DataFrame()
    wk = pd.read_csv(OUT / "weekly_week1.csv") if (OUT / "weekly_week1.csv").exists() else pd.DataFrame()
    stream = json.loads((CONF / "streaming_premium.json").read_text()) if (CONF / "streaming_premium.json").exists() else {}
    gscript = json.loads((CONF / "game_script.json").read_text()) if (CONF / "game_script.json").exists() else {}
    curve = pd.read_csv(OUT / "inseason_curve.csv") if (OUT / "inseason_curve.csv").exists() else pd.DataFrame()
    btm = json.loads((CONF / "backtest.json").read_text())
    priors = json.loads((CONF / "offseason_2026.json").read_text())
    roles = json.loads((CONF / "roles_2026.json").read_text())
    ctx = pd.read_csv(PROC / "team_context_2026.csv")

    sched = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    sched = sched[(sched["season"] == 2026) & (sched["game_type"] == "REG")].copy()
    for c in ("home_team", "away_team"):
        sched[c] = sched[c].replace(TEAM_ALIASES)

    # ---- team block --------------------------------------------------------
    t = teams.merge(power[["team", "off_final", "def_final", "net_final", "off_delta", "def_delta",
                           "seasons_used", "off_pass_prior", "def_pass_prior",
                           "off_rush_prior", "def_rush_prior"]], on="team")
    t = t.merge(dec[["team", "pts_from_data", "pts_from_prior", "prior_share"]], on="team")
    t = t.merge(sens[["team", "swing"]], on="team", how="left")
    t = t.merge(ctx[["team", "proj_plays_per_game", "proj_pass_att_per_game",
                     "proj_rush_att_per_game", "proj_off_td_per_game", "proj_dropback_rate"]], on="team")
    t["name"] = t["team"].map(lambda x: TEAM_META[x][0] + " " + TEAM_META[x][1])
    t["city"] = t["team"].map(lambda x: TEAM_META[x][0])
    t["nick"] = t["team"].map(lambda x: TEAM_META[x][1])
    t["c1"] = t["team"].map(lambda x: TEAM_META[x][2])
    t["c2"] = t["team"].map(lambda x: TEAM_META[x][3])
    t["conf"] = t["team"].map(TEAM_CONF)

    tcols = ["team","name","city","nick","c1","c2","division","conf","proj_wins","win_sd",
             "make_playoffs","win_division","conf_champ","super_bowl","one_seed","power_rating",
             "off_rank","def_rank","off_final","def_final","pts_from_data","pts_from_prior",
             "prior_share","swing","proj_plays_per_game","proj_pass_att_per_game",
             "proj_rush_att_per_game","proj_off_td_per_game","proj_dropback_rate","prior_note"]
    tround = {"proj_wins":2,"win_sd":2,"make_playoffs":4,"win_division":4,"conf_champ":4,
              "super_bowl":4,"one_seed":4,"power_rating":2,"off_rank":0,"def_rank":0,
              "off_final":4,"def_final":4,"pts_from_data":2,"pts_from_prior":2,"prior_share":3,
              "swing":2,"proj_plays_per_game":1,"proj_pass_att_per_game":1,
              "proj_rush_att_per_game":1,"proj_off_td_per_game":2,"proj_dropback_rate":3}

    winsd = {r["team"]: [round(r[f"p_{w}_wins"], 4) for w in range(18)] for _, r in wd.iterrows()}
    seeds = {r["team"]: [round(r[f"p_seed_{i}"], 4) for i in range(1, 8)] for _, r in sd.iterrows()}

    sch = {}
    for tm in TEAM_META:
        g = sched[(sched["home_team"] == tm) | (sched["away_team"] == tm)].sort_values("week")
        sch[tm] = [[int(r["week"]),
                    r["away_team"] if r["home_team"] == tm else r["home_team"],
                    1 if r["home_team"] == tm else 0] for _, r in g.iterrows()]

    # ---- players -----------------------------------------------------------
    p = players.copy()
    p = p[(p["proj_points"] > 0) | (p["override"].notna() & (p["override"] != ""))]
    p["pos_rank"] = p["pos_rank"].fillna(999)
    # Join on a normalised key, not the raw name. The ADP feed writes
    # "James Cook III" where the roster writes "James Cook", and joining on the
    # literal string silently drops the ADP for every player whose sources
    # disagree about a generational suffix.
    from .draft import _norm_name
    bcols = board[["name","adp","adp_formatted","adp_rank","value_rank","arb","verdict","adp_stdev","bye"]].copy()
    bcols["_k"] = bcols["name"].map(_norm_name)
    bcols = bcols.drop_duplicates("_k").drop(columns=["name"])
    p["_k"] = p["player"].map(_norm_name)
    before = len(p)
    p = p.merge(bcols, on="_k", how="left").drop(columns=["_k"])
    assert len(p) == before, "ADP join changed the row count"
    pcols = ["player","team","position","games","proj_points","proj_ppg","proj_points_17g",
             "targets","receptions","rec_yards","rec_td","rush_att","rush_yards","rush_td",
             "pass_att","pass_yards","pass_td","interceptions","p10","p25","p50","p75","p90",
             "p_top3","p_top5","p_top12","p_bust","vorp","pos_rank","target_share_eb",
             "rush_share_eb","depth","adp","adp_formatted","adp_rank","value_rank","arb",
             "verdict","bye","override"]
    pround = {"games":1,"proj_points":1,"proj_ppg":1,"proj_points_17g":1,"targets":1,
              "receptions":1,"rec_yards":0,"rec_td":1,"rush_att":1,"rush_yards":0,"rush_td":1,
              "pass_att":0,"pass_yards":0,"pass_td":1,"interceptions":1,"p10":1,"p25":1,"p50":1,
              "p75":1,"p90":1,"p_top3":3,"p_top5":3,"p_top12":3,"p_bust":3,"vorp":1,"pos_rank":0,
              "target_share_eb":4,"rush_share_eb":4,"depth":0,"adp":1,"adp_rank":0,
              "value_rank":0,"arb":0,"bye":0}
    p = p.sort_values("proj_points", ascending=False)

    # ---- market ------------------------------------------------------------
    e = edges.copy()
    ecols = ["market","team","model_prob","market_price_raw","market_prob_fair","edge_pp",
             "conviction","liquidity_usd","note"]
    eround = {"model_prob":4,"market_price_raw":4,"market_prob_fair":4,"edge_pp":2,"liquidity_usd":0}

    bundle = {
        "meta": {
            "version": "1.0.0", "built": "2026-08-25", "season": 2026,
            "n_sims": 25000, "n_player_sims": 10000,
            "adp_source": "FantasyFootballCalculator 12-team full PPR",
            "adp_drafts": 7726, "adp_window": "2026-08-18 to 2026-08-25",
            "market_pulled": "2026-08-25T07:34:22Z",
            "plays_in_fit": 81389, "n_players": int(len(p)),
        },
        "calibration": cal, "backtest": btm, "inseason": inseason, "scoring": scoring,
        "ledger": {k: v for k, v in led.items() if k != "teams"},
        "matchups": _tbl(mu, [c for c in ["week","away","home","kickoff","model_spread_home",
                              "market_spread_home","spread_edge","model_total","market_total",
                              "total_edge","home_win_prob","market_home_win_prob","wp_edge",
                              "model_home_pts","model_away_pts"] if c in mu.columns],
                         {"model_spread_home":1,"market_spread_home":1,"spread_edge":1,
                          "model_total":1,"market_total":1,"total_edge":1,"home_win_prob":4,
                          "market_home_win_prob":4,"wp_edge":4,"model_home_pts":1,
                          "model_away_pts":1,"week":0}) if len(mu) else {"cols":[],"rows":[]},
        "streaming": stream, "game_script": gscript,
        "weekly": _tbl(wk[wk["points"] > 1.5] if len(wk) else wk,
                       ["player","team","position","opponent","home","own_spread","points",
                        "floor","ceiling","matchup_delta","sd","pos_rank","targets",
                        "rush_att","rec_yards","rush_yards","pass_yards"],
                       {"points":1,"floor":1,"ceiling":1,"matchup_delta":2,"sd":2,"pos_rank":0,
                        "own_spread":1,"home":0,"targets":1,"rush_att":1,"rec_yards":0,
                        "rush_yards":0,"pass_yards":0}) if len(wk) else {"cols":[],"rows":[]},
        "inseason_curve": _tbl(curve, ["week","model","accuracy","mae","n"],
                               {"week":0,"accuracy":4,"mae":3,"n":0}) if len(curve) else {"cols":[],"rows":[]},
        "teams": _tbl(t, tcols, tround),
        "win_dist": winsd, "seed_dist": seeds, "schedule": sch,
        "players": _tbl(p, pcols, pround),
        "edges": _tbl(e, ecols, eround),
        "vig": _tbl(vig, ["book","n_outcomes","raw_sum","overround_pct","power_k"],
                    {"raw_sum":4,"overround_pct":2,"power_k":4,"n_outcomes":0}),
        "backtest_points": _tbl(bt, ["season","team","pred_wins","actual_wins"],
                                {"season":0,"pred_wins":2,"actual_wins":1}),
        "priors": {k: {"off": v["off_delta"], "def": v["def_delta"], "note": v["notes"]}
                   for k, v in priors["teams"].items()},
        "roles": {k: v for k, v in roles["players"].items()},
        "divisions": DIVISIONS,
    }
    out = OUT / "web_bundle.json"
    out.write_text(json.dumps(bundle, separators=(",", ":")))
    return bundle


if __name__ == "__main__":
    b = build()
    import os
    print("bundle:", round(os.path.getsize(OUT / "web_bundle.json") / 1024, 1), "KB")
    print("teams:", len(b["teams"]["rows"]), "| players:", len(b["players"]["rows"]),
          "| edges:", len(b["edges"]["rows"]))
