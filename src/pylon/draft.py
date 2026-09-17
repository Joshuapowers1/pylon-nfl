"""
PYLON :: draft
Compares the model's player projections to live average draft position and
produces the board that actually matters on draft day.

The comparison is deliberately NOT projected-points-vs-ADP, because points are
not the currency of a draft. A quarterback who scores 300 points is worth less
than a running back who scores 260, if the fourteenth-best quarterback also
scores 285 and the thirtieth-best back scores 150. The right currency is value
over replacement, and the right output is not a ranking but a probability: how
often does this player finish top five at his position, and how often does he
finish outside the startable range entirely.

So every player gets three numbers a normal cheat sheet does not have:
  value_rank   his rank by VORP across all positions
  adp_rank     where the market is actually taking him
  arb          the gap, in draft slots. Positive means the model wants him
               earlier than the room does.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowerss · @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import RAW, OUT

# ADP feeds and roster files use different abbreviations for the same teams.
ADP_TEAM_FIX = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "ARZ": "ARI", "AZ": "ARI"}

# Generational suffixes are inconsistent across every football data source in
# existence -- FFC says "Travis Etienne Jr.", nflverse says "Travis Etienne" --
# so they are stripped from both sides rather than special-cased one at a time.
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
NAME_FIX = {"marquise brown": "hollywood brown"}


def _norm_name(s: str) -> str:
    s = str(s).lower().strip().replace(".", "").replace("'", "").replace("-", " ")
    parts = [w for w in s.split() if w not in SUFFIXES]
    s = " ".join(parts)
    return NAME_FIX.get(s, s)


def load_adp() -> pd.DataFrame:
    d = json.loads((RAW / "adp_ppr_2026.json").read_text())
    meta = d["meta"]
    a = pd.DataFrame(d["players"])
    a["team"] = a["team"].replace(ADP_TEAM_FIX)
    a["key"] = a["name"].map(_norm_name)
    a["adp_rank"] = a["adp"].rank(method="min").astype(int)
    a.attrs["meta"] = meta
    return a[["key", "name", "position", "team", "adp", "adp_formatted", "adp_rank",
              "times_drafted", "high", "low", "stdev", "bye"]], meta


def board(proj: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    adp, meta = load_adp()
    p = proj.copy()
    p["key"] = p["player"].map(_norm_name)
    # Same name, two players: keep the one the model projects more opportunity to.
    p = p.sort_values("proj_points", ascending=False).drop_duplicates("key")

    m = adp.merge(p, on="key", how="left", suffixes=("_adp", ""))
    m["matched"] = m["proj_points"].notna()

    v = m[m["matched"] & m["position_adp"].isin(["QB", "RB", "WR", "TE"])].copy()
    v["value_rank"] = v["vorp"].rank(ascending=False, method="min").astype(int)
    v["adp_rank_pos"] = v.groupby("position_adp")["adp"].rank(method="min").astype(int)
    v["arb"] = v["adp_rank"] - v["value_rank"]      # positive = model wants him earlier
    # Only judge players the market is genuinely drafting. Past roughly pick 180
    # the ADP signal is a handful of drafts deep and the "arbitrage" is noise
    # against noise, which is how cheat sheets end up recommending a team's
    # fourth receiver as a league winner.
    RANKED = 180
    live = (v["adp_rank"] <= RANKED)
    v["verdict"] = np.select(
        [live & (v["arb"] >= 24), live & (v["arb"] >= 10),
         live & (v["arb"] <= -24), live & (v["arb"] <= -10)],
        ["SMASH BUY", "BUY", "HARD FADE", "FADE"], default="FAIR",
    )
    v.loc[~live, "verdict"] = "UNRANKED"
    cols = ["name", "position_adp", "team_adp", "adp", "adp_formatted", "adp_rank", "value_rank",
            "arb", "verdict", "proj_points", "proj_ppg", "vorp", "pos_rank", "p_top3", "p_top5",
            "p_top12", "p_bust", "p10", "p90", "games", "targets", "rush_att", "stdev", "bye"]
    v = v[cols].rename(columns={"position_adp": "pos", "team_adp": "team", "stdev": "adp_stdev"})
    v = v.sort_values("value_rank").reset_index(drop=True)
    v.to_csv(OUT / "draft_board.csv", index=False)

    skill = m["position_adp"].isin(["QB", "RB", "WR", "TE"]) if "position_adp" in m.columns else m["position"].isin(["QB", "RB", "WR", "TE"])
    unmatched = m[(~m["matched"]) & skill]["name_adp" if "name_adp" in m.columns else "name"].tolist()
    info = {"adp_source": "FantasyFootballCalculator 12-team full PPR",
            "drafts_sampled": meta["total_drafts"], "window": f"{meta['start_date']} to {meta['end_date']}",
            "adp_players": len(adp), "matched": int(m["matched"].sum()),
            "unmatched_examples": unmatched[:15]}
    return v, info
