"""
PYLON :: inseason
The live model. `update.py` proved that blending this season into the preseason
rating beats leaving it frozen; this module is where that blend actually feeds
every published number.

Until this existed, the Tuesday job refit nothing. It reloaded the August
ratings, moved the matchup page forward a week and re-simulated a season that
had already been half played as if no game had a result. Three things change
once games exist, and each is handled here with a constant that was fitted
rather than chosen:

TEAMS. Opponent-adjusted ratings from this season's plays are blended into the
preseason rating at w = n / (n + k), the conjugate weight `update.fit_blend`
selected out of sample (k = 8). Under the same normal-normal model the
posterior variance of a team's true strength is (1 - w) times the prior's, so
the simulator's team-constant draw shrinks by sqrt(1 - w) too. Completed games
are locked to their real result rather than re-rolled.

CONTEXT. Pace, dropback rate and scoring rate blend the same way, with their own
k, because pace stabilises far faster than efficiency does.

PLAYERS. Usage shares blend toward the measured 2026 share with a k fitted on a
replay of 2024 and 2025 (prior = preceding seasons, evidence = weeks 1-8,
scored on weeks 9-18). Efficiency pools the new opportunities with the history
the empirical-Bayes estimate was already built from, which needs no new
constant at all. Season totals become what has happened plus what is left.

--------------------------------------------------------------------------
PYLON . NFL projection and market-edge engine
Joshua Powers . joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl . linkedin.com/in/joshuampowerss . @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import RAW, CONF, TEAM_ALIASES, TARGET_SEASON, SCORING
from . import ingest, usage as usage_mod
from .update import in_season_ratings

INSEASON_PLAYERS = CONF / "inseason_players.json"
# Extra columns the to-date fantasy ledger needs on top of ingest.USE_COLS.
_EXTRA = ["fumbled_1_player_id"]


# --------------------------------------------------------------------------
# where the season is
# --------------------------------------------------------------------------
def schedule(season: int = TARGET_SEASON) -> pd.DataFrame:
    g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    g = g[(g["season"] == season) & (g["game_type"] == "REG")].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_ALIASES)
    return g


def completed_through(season: int = TARGET_SEASON) -> int:
    """The last week in which every game has a final score. 0 before kickoff.

    A week with a game still to play is not complete even if fifteen of its
    sixteen results are in, because the blend weight and the locked results
    would then disagree about how much of the season has happened."""
    g = schedule(season)
    done = 0
    for wk in sorted(g["week"].unique()):
        if g.loc[g["week"] == wk, "result"].notna().all():
            done = int(wk)
        else:
            break
    return done


def team_games_played(season: int = TARGET_SEASON, through_week: int | None = None) -> pd.Series:
    """Completed games per team through `through_week`, byes included as zero."""
    g = schedule(season)
    if through_week is None:
        through_week = completed_through(season)
    d = g[(g["week"] <= through_week) & g["result"].notna()]
    n = pd.concat([d["home_team"], d["away_team"]]).value_counts()
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    return n.reindex(teams).fillna(0).astype(int)


def load_pbp(season: int = TARGET_SEASON) -> pd.DataFrame | None:
    f = RAW / f"pbp_{season}.parquet"
    if not f.exists():
        return None
    import pyarrow.parquet as pq
    have = set(pq.read_schema(f).names)
    cols = [c for c in ingest.USE_COLS + _EXTRA if c in have]
    d = pd.read_parquet(f, columns=cols)
    for c in ("posteam", "defteam", "home_team", "away_team"):
        d[c] = d[c].replace(TEAM_ALIASES)
    return d


# --------------------------------------------------------------------------
# teams
# --------------------------------------------------------------------------
def blend_weight(n, k: float):
    """w = n / (n + k). Zero games means zero weight, never a division error."""
    n = np.asarray(n, dtype=float)
    return n / (n + float(k))


def blend_context(ctx26: pd.DataFrame, raw: pd.DataFrame, through_week: int,
                  k: dict) -> pd.DataFrame:
    """Move pace, dropback rate and touchdown rate toward what this season shows.

    The attempt splits are rebuilt from the blended pace and dropback rate with
    the same identities `adjustments.project_context` uses, so the three volume
    numbers cannot drift apart."""
    d = raw[raw["week"] <= through_week]
    cur = ingest.team_context(d).set_index("team")
    c = ctx26.copy().set_index("team")
    n = cur["games"].reindex(c.index).fillna(0)
    for col, kc in (("plays_per_game", k["plays_per_game"]),
                    ("dropback_rate", k["dropback_rate"]),
                    ("off_td_per_game", k["off_td_per_game"])):
        w = blend_weight(n, kc)
        obs = cur[col].reindex(c.index)
        c["pre_" + col] = c["proj_" + col]
        c["proj_" + col] = np.where(obs.notna(), (1 - w) * c["proj_" + col] + w * obs.fillna(0),
                                    c["proj_" + col])
    c["proj_dropbacks_per_game"] = c["proj_plays_per_game"] * c["proj_dropback_rate"]
    c["proj_pass_att_per_game"] = c["proj_dropbacks_per_game"] * 0.875
    c["proj_rush_att_per_game"] = c["proj_plays_per_game"] - c["proj_dropbacks_per_game"]
    c["proj_qb_rush_per_game"] = c["proj_dropbacks_per_game"] * 0.055
    c["games_2026"] = n.astype(int)
    return c.reset_index()


def blend_team_ratings(adj: pd.DataFrame, plays_clean: pd.DataFrame, ctx26: pd.DataFrame,
                       through_week: int, k: float, games: pd.Series,
                       season: int = TARGET_SEASON) -> tuple[pd.DataFrame, dict]:
    """Blend this season's opponent-adjusted ratings into the preseason ones.

    Every unit is blended with the team's own weight, and the roster-prior delta
    is scaled by (1 - w), so off_final = off_rating + off_delta still holds and
    the data-versus-prior decomposition stays honest: by midseason the August
    opinion carries about half the weight it did in August."""
    ins = in_season_ratings(plays_clean, season, through_week)
    a = adj.copy()
    if not len(ins):
        a["games_2026"] = 0
        a["w_inseason"] = 0.0
        return a, {"through_week": through_week, "w_mean": 0.0, "applied": False}

    ins = ins.set_index("team")
    a = a.set_index("team")
    n = games.reindex(a.index).fillna(0)
    w = pd.Series(blend_weight(n, k), index=a.index)

    dr = ctx26.set_index("team")["proj_dropback_rate"].reindex(a.index).fillna(0.58)
    lg_dr = float(ctx26["proj_dropback_rate"].mean())
    off_ins = dr * ins["off_pass"] + (1 - dr) * ins["off_rush"]
    def_ins = lg_dr * ins["def_pass"] + (1 - lg_dr) * ins["def_rush"]

    for col in ("off_rating", "def_rating", "off_delta", "def_delta", "off_final",
                "def_final", "net_final", "off_pass_prior", "off_rush_prior",
                "def_pass_prior", "def_rush_prior"):
        a["pre_" + col] = a[col]
    for phase in ("off_pass", "off_rush", "def_pass", "def_rush"):
        a[phase + "_prior"] = (1 - w) * a[phase + "_prior"] + w * ins[phase].reindex(a.index).fillna(0)
        a[phase + "_2026"] = ins[phase].reindex(a.index)
    a["off_rating"] = (1 - w) * a["off_rating"] + w * off_ins.reindex(a.index).fillna(0)
    a["def_rating"] = (1 - w) * a["def_rating"] + w * def_ins.reindex(a.index).fillna(0)
    a["net_rating"] = a["off_rating"] - a["def_rating"]
    a["off_delta"] = (1 - w) * a["off_delta"]
    a["def_delta"] = (1 - w) * a["def_delta"]
    a["off_final"] = a["off_rating"] + a["off_delta"]
    a["def_final"] = a["def_rating"] + a["def_delta"]
    a["net_final"] = a["off_final"] - a["def_final"]
    a["games_2026"] = n.astype(int)
    a["w_inseason"] = w
    a = a.reset_index()
    a["off_rank"] = a["off_final"].rank(ascending=False).astype(int)
    a["def_rank"] = a["def_final"].rank(ascending=True).astype(int)
    a["net_rank"] = a["net_final"].rank(ascending=False).astype(int)
    a = a.sort_values("net_final", ascending=False).reset_index(drop=True)
    info = {"through_week": int(through_week), "k": float(k), "applied": True,
            "w_mean": round(float(w.mean()), 4), "w_min": round(float(w.min()), 4),
            "w_max": round(float(w.max()), 4),
            "games_mean": round(float(n.mean()), 3),
            "plays_2026": int(((plays_clean["season"] == season)
                               & (plays_clean["week"] <= through_week)).sum())}
    return a, info


def posterior_sigma_team(sigma_team: float, w: pd.Series) -> pd.Series:
    """Normal-normal: prior variance tau^2, posterior tau^2 (1 - w)."""
    return sigma_team * np.sqrt(1.0 - w.clip(0, 1))


# --------------------------------------------------------------------------
# players
# --------------------------------------------------------------------------
def season_usage(raw: pd.DataFrame, through_week: int, season: int = TARGET_SEASON) -> pd.DataFrame:
    """One row per player for this season, traded players combined.

    `usage.player_usage` keys on (team, player), which is right for a season
    that is over and wrong for one in progress: a receiver traded in October
    would carry two half-sized shares. Counts are summed across teams and the
    shares recomputed over the games he actually appeared in."""
    d = raw[(raw["season"] == season) & (raw["week"] <= through_week)]
    u = usage_mod.player_usage(d, seasons=(season,))
    if not len(u):
        return u
    count_cols = ["targets", "receptions", "rec_yards", "rec_td", "rz_targets", "rush_att",
                  "rush_yards", "rush_td", "rz_rush", "dropbacks", "pass_att_p", "completions",
                  "pass_yards", "pass_td", "ints", "scrambles", "scramble_yards", "scramble_td",
                  "team_pass_att", "team_rush_att", "team_rz_pass_att", "team_rz_rush_att",
                  "team_dropbacks", "games_played"]
    g = u.groupby("player_id", as_index=False)[count_cols].sum()
    s = lambda a, b: g[a] / g[b].replace(0, np.nan)
    g["target_share"] = s("targets", "team_pass_att")
    g["rush_share"] = s("rush_att", "team_rush_att")
    g["rz_target_share"] = s("rz_targets", "team_rz_pass_att")
    g["rz_rush_share"] = s("rz_rush", "team_rz_rush_att")
    g["dropback_share"] = s("dropbacks", "team_dropbacks")
    g["catch_rate"] = s("receptions", "targets")
    g["yards_per_target"] = s("rec_yards", "targets")
    g["ypc"] = s("rush_yards", "rush_att")
    g["ypa"] = s("pass_yards", "pass_att_p")
    g["td_rate_pass"] = s("pass_td", "pass_att_p")
    g["int_rate"] = s("ints", "pass_att_p")
    return g.fillna(0)


def to_date_stats(raw: pd.DataFrame, through_week: int, season: int = TARGET_SEASON) -> pd.DataFrame:
    """What every player has actually scored, full PPR, from the play-by-play.

    Computed on every regular-season scrimmage play, not the competitive-band
    subset the ratings use: a garbage-time touchdown counts on a fantasy
    scoreboard even though it says nothing about team strength."""
    d = raw[(raw["season"] == season) & (raw["season_type"] == "REG")
            & (raw["week"] <= through_week) & (raw["special"] != 1)].copy()
    d["pass_att"] = ((d["qb_dropback"] == 1) & (d["sack"] != 1) & (d["qb_scramble"] != 1)).astype(int)

    pa = d[d["passer_player_id"].notna()].groupby("passer_player_id").agg(
        pass_att=("pass_att", "sum"), pass_yards=("passing_yards", "sum"),
        pass_td=("pass_touchdown", "sum"), interceptions=("interception", "sum"))
    ru = d[d["rusher_player_id"].notna()].groupby("rusher_player_id").agg(
        rush_att=("rush_touchdown", "size"), rush_yards=("rushing_yards", "sum"),
        rush_td=("rush_touchdown", "sum"))
    rc = d[d["receiver_player_id"].notna() & (d["pass_att"] == 1)].groupby("receiver_player_id").agg(
        targets=("pass_att", "sum"), receptions=("complete_pass", "sum"),
        rec_yards=("receiving_yards", "sum"), rec_td=("pass_touchdown", "sum"))
    fl = (d[(d.get("fumble_lost", 0) == 1) & d["fumbled_1_player_id"].notna()]
          .groupby("fumbled_1_player_id").size().rename("fumbles_lost")
          if "fumbled_1_player_id" in d.columns else pd.Series(dtype=float, name="fumbles_lost"))

    ids = []
    for col in ("passer_player_id", "rusher_player_id", "receiver_player_id"):
        ids.append(d[d[col].notna()][[col, "game_id"]].rename(columns={col: "player_id"}))
    gp = pd.concat(ids).drop_duplicates().groupby("player_id").size().rename("games_to_date")

    out = pd.concat([pa, ru, rc, fl, gp], axis=1).fillna(0)
    out.index.name = "player_id"
    s = SCORING
    out["points_to_date"] = (out["pass_yards"] * s["pass_yd"] + out["pass_td"] * s["pass_td"]
                             + out["interceptions"] * s["interception"]
                             + out["rush_yards"] * s["rush_yd"] + out["rush_td"] * s["rush_td"]
                             + out["receptions"] * s["rec"] + out["rec_yards"] * s["rec_yd"]
                             + out["rec_td"] * s["rec_td"] + out["fumbles_lost"] * s["fumble_lost"])
    return out.reset_index()


_VOLUME = ["targets", "receptions", "rec_yards", "rec_td", "rush_att", "rush_yards", "rush_td",
           "pass_att", "pass_yards", "pass_td", "interceptions"]


def add_to_date(proj: pd.DataFrame, td: pd.DataFrame) -> pd.DataFrame:
    """Season total = what has happened + what is projected to happen.

    `proj_points` keeps meaning "full-season points" all year, so the draft
    room, the finish odds and anything else reading it does not change meaning
    in October. The rest-of-season figure is kept beside it as `proj_ros`, and
    `proj_ppg` is the forward rate, because that is what a lineup decision
    needs."""
    p = proj.copy()
    t = td.set_index("player_id")
    j = p["player_id"].map
    p["proj_ros"] = p["proj_points"]
    # Kept so a half-PPR or standard league can strip reception credit from
    # the rest-of-season number without guessing how many catches are left.
    p["receptions_ros"] = p["receptions"]
    p["points_to_date"] = j(t["points_to_date"]).fillna(0.0)
    p["games_to_date"] = j(t["games_to_date"]).fillna(0).astype(int)
    for c in _VOLUME:
        p[c] = p[c].fillna(0) + j(t[c]).fillna(0.0)
    p["proj_points"] = p["points_to_date"] + p["proj_ros"]
    p["games"] = p["games_to_date"] + p["games_ros"]
    p["ppg_to_date"] = p["points_to_date"] / p["games_to_date"].replace(0, np.nan)
    return p


def player_k() -> dict:
    if INSEASON_PLAYERS.exists():
        return json.loads(INSEASON_PLAYERS.read_text())["k"]
    return {"target_share": 6.0, "rush_share": 4.0, "dropback_share": 3.0,
            "plays_per_game": 4.0, "dropback_rate": 6.0, "off_td_per_game": 12.0}


# --------------------------------------------------------------------------
# fitting the constants
# --------------------------------------------------------------------------
_KS = (0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 999)


def _sweep(prior, early, late, n, wt) -> dict:
    """Weighted squared error of the blend against what happened next, per k."""
    res = []
    for k in _KS:
        w = blend_weight(n, k)
        pred = (1 - w) * prior + w * early
        res.append({"k": k, "rmse": float(np.sqrt(np.average((pred - late) ** 2, weights=wt)))})
    best = min(res, key=lambda r: r["rmse"])
    return {"k": best["k"], "rmse": best["rmse"],
            "rmse_prior_only": float(np.sqrt(np.average((prior - late) ** 2, weights=wt))),
            "rmse_season_only": float(np.sqrt(np.average((early - late) ** 2, weights=wt))),
            "n": int(len(late)), "sweep": res}


def fit(seasons=(2024, 2025), split_week: int = 8) -> dict:
    """Replay completed seasons at the same point the live season is at.

    Prior = what the model knew in August (the two preceding seasons, recency
    weighted the way `players.blend_usage` weights them). Evidence = weeks 1 to
    `split_week`. Scored against weeks `split_week`+1 to 18, which the blend
    never sees. A k that does not beat both of its own ingredients would be a
    reason to delete the blend, so both baselines are reported beside it."""
    from .players import SEASON_W
    out = {"split_week": split_week, "seasons": list(seasons), "fits": {}}
    rows = {k: [] for k in ("target_share", "rush_share", "dropback_share")}
    ctx_rows = []
    for S in seasons:
        raw = ingest.load_raw([S - 3, S - 2, S - 1, S])
        # --- usage
        pri = usage_mod.player_usage(raw, seasons=(S - 2, S - 1))
        pri["w"] = pri["season"].map({S - 1: SEASON_W[2025], S - 2: SEASON_W[2024]})
        early = usage_mod.player_usage(raw[(raw["season"] == S) & (raw["week"] <= split_week)], seasons=(S,))
        late = usage_mod.player_usage(raw[(raw["season"] == S) & (raw["week"] > split_week)], seasons=(S,))
        for col, opp, min_late in (("target_share", "team_pass_att", 90),
                                   ("rush_share", "team_rush_att", 60),
                                   ("dropback_share", "team_dropbacks", 90)):
            ww = pri["w"] * pri["games_played"].clip(lower=1)
            p = ((pri[col] * ww).groupby(pri["player_id"]).sum()
                 / ww.groupby(pri["player_id"]).sum()).rename("prior")
            e = early.groupby("player_id").agg(early=(col, "mean"), n=("games_played", "sum"))
            l = late.groupby("player_id").agg(late=(col, "mean"), opp=(opp, "sum"))
            j = e.join(p, how="inner").join(l, how="inner")
            j = j[(j["opp"] >= min_late) & ((j["early"] > 0) | (j["prior"] > 0) | (j["late"] > 0))]
            j["season"] = S
            rows[col].append(j)
        # --- team context
        cpre = ingest.team_context(raw[raw["season"] < S])
        wts = {S - 1: 1.00, S - 2: 0.45, S - 3: 0.18}
        cpre["w"] = cpre["season"].map(wts)
        ce = ingest.team_context(raw[(raw["season"] == S) & (raw["week"] <= split_week)]).set_index("team")
        cl = ingest.team_context(raw[(raw["season"] == S) & (raw["week"] > split_week)]).set_index("team")
        for team, g in cpre.groupby("team"):
            if team not in ce.index or team not in cl.index:
                continue
            w = g["w"].to_numpy() / g["w"].sum()
            r = {"season": S, "team": team, "n": ce.loc[team, "games"], "wt": cl.loc[team, "games"]}
            for col in ("plays_per_game", "dropback_rate", "off_td_per_game"):
                r[col + "_prior"] = float(np.sum(g[col].to_numpy() * w))
                r[col + "_early"] = ce.loc[team, col]
                r[col + "_late"] = cl.loc[team, col]
            ctx_rows.append(r)

    for col, parts in rows.items():
        j = pd.concat(parts)
        out["fits"][col] = _sweep(j["prior"].to_numpy(), j["early"].to_numpy(),
                                  j["late"].to_numpy(), j["n"].to_numpy(), j["opp"].to_numpy())
    c = pd.DataFrame(ctx_rows)
    for col in ("plays_per_game", "dropback_rate", "off_td_per_game"):
        # Match production: the August context is pulled 30% toward the league mean.
        pr = c[col + "_prior"]
        pr = 0.70 * pr + 0.30 * pr.groupby(c["season"]).transform("mean")
        out["fits"][col] = _sweep(pr.to_numpy(), c[col + "_early"].to_numpy(),
                                  c[col + "_late"].to_numpy(), c["n"].to_numpy(), c["wt"].to_numpy())
    out["k"] = {col: f["k"] for col, f in out["fits"].items()}
    INSEASON_PLAYERS.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = fit()
    for col, f in r["fits"].items():
        print(f"{col:16s} k={f['k']:>5}  rmse {f['rmse']:.4f}  "
              f"prior-only {f['rmse_prior_only']:.4f}  season-only {f['rmse_season_only']:.4f}  n={f['n']}")
