"""
PYLON :: ledger
Locks this season's predictions in writing, before any of them can be checked.

A projection nobody can grade is a hot take with a spreadsheet attached. This
module freezes every preseason call, stamps it, and defines exactly how each one
will be scored when the season resolves. Once a ledger exists on disk with a
timestamp, the record is no longer a claim about the past, it is a commitment
about the future.

Scoring uses Brier score for the probabilistic calls, because it is the metric
that punishes confident wrong answers the way it should:

    Brier = mean((probability - outcome)^2)

A coin flip on everything scores 0.25. Anything above that is worse than knowing
nothing. The ledger records the model's number and the market's number side by
side, so at the end of the season the comparison is already set up.

--------------------------------------------------------------------------
PYLON . NFL projection and market-edge engine
Joshua Powers . joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl . linkedin.com/in/joshuampowers . @joshuampowers
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd
from .config import OUT, CONF, TARGET_SEASON

LEDGER = OUT / "prediction_ledger.json"


def build(teams: pd.DataFrame, edges: pd.DataFrame, matchups: pd.DataFrame | None = None,
          stamped_at: str | None = None) -> dict:
    """Freeze every preseason prediction with its scoring rule."""
    now = stamped_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snap = json.loads((CONF / "market_snapshot_2026.json").read_text())
    mk = {(r["market"], r["team"]): r for _, r in edges.iterrows()}

    entries = []
    for _, t in teams.iterrows():
        wt = snap["win_totals_dk"].get(t["team"])
        entries.append({
            "id": f"{TARGET_SEASON}-{t['team']}",
            "team": t["team"],
            "proj_wins": round(float(t["proj_wins"]), 2),
            "win_total_line": wt,
            "p_over_win_total": round(float(
                mk[(f"Win Total O{wt}", t["team"])]["model_prob"]), 4)
                if (f"Win Total O{wt}", t["team"]) in mk else None,
            "p_make_playoffs": round(float(t["make_playoffs"]), 4),
            "market_p_make_playoffs": round(float(
                mk[("Make Playoffs", t["team"])]["market_prob_fair"]), 4)
                if ("Make Playoffs", t["team"]) in mk else None,
            "p_win_division": round(float(t["win_division"]), 4),
            "p_super_bowl": round(float(t["super_bowl"]), 4),
            "market_p_super_bowl": round(float(
                mk[("Super Bowl LXI", t["team"])]["market_prob_fair"]), 4)
                if ("Super Bowl LXI", t["team"]) in mk else None,
        })

    games = []
    if matchups is not None and len(matchups):
        for _, g in matchups.iterrows():
            games.append({
                "id": f"{TARGET_SEASON}-W{int(g['week'])}-{g['away']}-{g['home']}",
                "week": int(g["week"]), "away": g["away"], "home": g["home"],
                "model_spread_home": round(float(g["model_spread_home"]), 2),
                "model_total": round(float(g["model_total"]), 2),
                "home_win_prob": round(float(g["home_win_prob"]), 4),
                "market_spread_home": g.get("market_spread_home"),
                "market_total": g.get("market_total"),
            })

    payload = {
        "season": TARGET_SEASON,
        "stamped_at": now,
        "model_version": "1.0.0",
        "scoring_rules": {
            "win_totals": "Brier score on P(over the posted line). Resolved by final record.",
            "playoffs": "Brier score on P(makes the playoffs), 14 of 32 resolve true.",
            "division": "Brier score on P(wins the division), 8 of 32 resolve true.",
            "super_bowl": "Brier score on P(wins Super Bowl LXI), 1 of 32 resolves true.",
            "games": ("Straight-up accuracy against the closing line, mean absolute spread "
                      "error, and total over/under record."),
            "benchmark": ("Every entry stores the de-vigged market number alongside the model "
                          "number, so the season-end comparison needs no extra data."),
        },
        "teams": entries,
        "games": games,
    }
    # A hash of the content makes the freeze checkable: if a number is edited
    # later the digest stops matching what was published.
    payload["digest"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "digest"},
                   sort_keys=True).encode()).hexdigest()[:16]
    LEDGER.write_text(json.dumps(payload, indent=2))
    return payload
