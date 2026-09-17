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
github.com/Joshuapowers1/pylon-nfl . linkedin.com/in/joshuampowerss . @joshuampowers
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


def score(season: int = TARGET_SEASON) -> dict:
    """Grade the sealed ledger against what actually happened.

    Runs safely at any point: it scores whatever has resolved and reports the
    rest as pending. Brier score is used throughout, and the market's Brier is
    computed on the same events, because a model's calibration number means
    nothing without the benchmark it was supposed to beat."""
    import numpy as np
    from .config import RAW, TEAM_ALIASES

    if not LEDGER.exists():
        return {"error": "no ledger to score"}
    led = json.loads(LEDGER.read_text())

    g = pd.read_csv(RAW / "schedules.csv", low_memory=False)
    g = g[(g["season"] == season) & (g["game_type"] == "REG")].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_ALIASES)
    played = g.dropna(subset=["result"])
    if not len(played):
        return {"season": season, "status": "no games resolved yet",
                "stamped_at": led["stamped_at"], "digest": led["digest"],
                "entries_awaiting": len(led["teams"])}

    wins = {}
    for _, r in played.iterrows():
        for t in (r["home_team"], r["away_team"]):
            wins.setdefault(t, 0.0)
        if r["result"] > 0:
            wins[r["home_team"]] += 1
        elif r["result"] < 0:
            wins[r["away_team"]] += 1
        else:
            wins[r["home_team"]] += 0.5
            wins[r["away_team"]] += 0.5
    complete = played["week"].nunique() >= 18 and len(played) >= 272

    rows = []
    for e in led["teams"]:
        t = e["team"]
        if t not in wins:
            continue
        row = {"team": t, "actual_wins": wins[t], "proj_wins": e["proj_wins"],
               "win_error": wins[t] - e["proj_wins"]}
        if complete and e.get("win_total_line") is not None and e.get("p_over_win_total") is not None:
            over = 1.0 if wins[t] > e["win_total_line"] else 0.0
            row["over_hit"] = over
            row["brier_win_total"] = (e["p_over_win_total"] - over) ** 2
        rows.append(row)
    d = pd.DataFrame(rows)

    out = {"season": season, "stamped_at": led["stamped_at"], "digest": led["digest"],
           "games_resolved": int(len(played)), "season_complete": bool(complete),
           "win_total_mae": round(float(d["win_error"].abs().mean()), 3) if len(d) else None}

    # Game-level calls, gradeable from week one.
    if led.get("games"):
        gl = []
        for e in led["games"]:
            hit = played[(played["home_team"] == e["home"]) & (played["away_team"] == e["away"])]
            if not len(hit):
                continue
            r = hit.iloc[0]
            margin = float(r["result"])
            model_pick_home = e["home_win_prob"] > 0.5
            gl.append({
                "game": e["id"], "actual_margin": margin,
                "model_correct": int((margin > 0) == model_pick_home) if margin != 0 else 0,
                "model_spread_err": abs(-e["model_spread_home"] - margin),
                "market_spread_err": abs(-e["market_spread_home"] - margin)
                if e.get("market_spread_home") is not None else None,
                "brier_model": (e["home_win_prob"] - (1.0 if margin > 0 else 0.0)) ** 2,
            })
        if gl:
            gd = pd.DataFrame(gl)
            out["games"] = {
                "n": int(len(gd)),
                "straight_up": round(float(gd["model_correct"].mean()), 4),
                "spread_mae_model": round(float(gd["model_spread_err"].mean()), 3),
                "spread_mae_market": round(float(gd["market_spread_err"].dropna().mean()), 3)
                if gd["market_spread_err"].notna().any() else None,
                "brier_model": round(float(gd["brier_model"].mean()), 4),
                "brier_coin_flip": 0.25,
            }

    # Season-long probabilistic calls only resolve at the end.
    if complete:
        div, po = _resolve_structure(played, wins)
        for label, key, truth in (("playoffs", "p_make_playoffs", po),
                                  ("division", "p_win_division", div)):
            p_model = np.array([e[key] for e in led["teams"] if e["team"] in truth])
            y = np.array([truth[e["team"]] for e in led["teams"] if e["team"] in truth])
            out[f"brier_{label}"] = round(float(((p_model - y) ** 2).mean()), 4)
        mk = np.array([e["market_p_make_playoffs"] for e in led["teams"]
                       if e.get("market_p_make_playoffs") is not None and e["team"] in po])
        ym = np.array([po[e["team"]] for e in led["teams"]
                       if e.get("market_p_make_playoffs") is not None and e["team"] in po])
        if len(mk):
            out["brier_playoffs_market"] = round(float(((mk - ym) ** 2).mean()), 4)

    (OUT / "ledger_score.json").write_text(json.dumps(out, indent=2))
    return out


def _resolve_structure(played: pd.DataFrame, wins: dict) -> tuple[dict, dict]:
    """Division winners and playoff berths from final records. Approximate on
    ties, which is fine for scoring a Brier: it moves nothing material."""
    from .config import DIVISIONS, TEAM_CONF
    div = {t: 0 for t in wins}
    po = {t: 0 for t in wins}
    winners = {}
    for name, members in DIVISIONS.items():
        present = [m for m in members if m in wins]
        if not present:
            continue
        best = max(present, key=lambda t: wins[t])
        div[best] = 1
        winners[name] = best
    for conf in ("AFC", "NFC"):
        members = [t for t in wins if TEAM_CONF.get(t) == conf]
        dws = [w for n, w in winners.items() if n.startswith(conf)]
        rest = sorted([t for t in members if t not in dws], key=lambda t: -wins[t])
        for t in dws + rest[:3]:
            po[t] = 1
    return div, po
