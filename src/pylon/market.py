"""
PYLON :: market
Converts prediction-market and sportsbook prices into clean probabilities and
compares them to the model.

Two things people get wrong when they quote market odds.

FIRST, the vig. A sportsbook's 32 Super Bowl prices sum to about 1.23, not 1.00.
That 23% overround is the house edge, and if you compare a model to raw book
prices without removing it you will conclude the model has an edge on all 32
teams, which is arithmetic, not insight. Polymarket is much tighter (about 4.9%
on the Super Bowl book, and its conference books actually sum to LESS than 1.00)
but it still has to be normalised.

The normalisation used here is the power method rather than simple proportional
scaling. Proportional scaling takes the same percentage off a 50-cent favourite
and a 1-cent longshot, which is the wrong shape: the overround is empirically
concentrated in the longshots. The power method solves for k such that
sum(p_i^k) = 1, which strips more from the tail. It is the standard fix in the
odds-modelling literature and it materially changes who looks mispriced.

SECOND, liquidity. A Polymarket division book with $25,000 of liquidity and a
ten-cent bid-ask spread is not a considered market opinion. Those lines are
flagged, not traded against.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowerss · @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from .config import CONF, OUT, EDGE_STRONG, EDGE_LEAN, DIVISIONS, TEAM_DIV


PRESEASON_SNAPSHOT = CONF / "market_snapshot_2026.json"
LIVE_SNAPSHOT = CONF / "market_snapshot_live.json"


def load_snapshot(live: bool = False) -> dict:
    """The August book by default, because that is what the sealed ledger and
    the README's worked de-vig example are about. `live=True` returns the book
    `scripts/snapshot_markets.py` pulled on the day of the refit, falling back
    to August when no refit has pulled one."""
    if live and LIVE_SNAPSHOT.exists():
        return json.loads(LIVE_SNAPSHOT.read_text())
    return json.loads(PRESEASON_SNAPSHOT.read_text())


def american_to_prob(odds: float) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / (-odds + 100.0)


def devig_power(prices: dict[str, float]) -> dict[str, float]:
    """Solve sum(p_i^k) = 1 for k. Handles both overround (k>1) and the
    under-round Polymarket sometimes shows on its conference books (k<1)."""
    p = np.array(list(prices.values()), dtype=float)
    p = np.clip(p / 100.0 if p.max() > 1.5 else p, 1e-6, 0.999999)
    if abs(p.sum() - 1.0) < 1e-9:
        return dict(zip(prices, p))
    f = lambda k: float(np.sum(p ** k) - 1.0)
    lo, hi = 0.20, 6.0
    try:
        k = brentq(f, lo, hi, xtol=1e-10)
    except ValueError:
        k = 1.0
        p = p / p.sum()
        return dict(zip(prices, p))
    q = p ** k
    return dict(zip(prices, q / q.sum()))


def devig_proportional(prices: dict[str, float]) -> dict[str, float]:
    p = np.array(list(prices.values()), dtype=float)
    p = p / 100.0 if p.max() > 1.5 else p
    return dict(zip(prices, p / p.sum()))


def win_total_prob(proj_wins: float, win_sd: float, line: float) -> float:
    """P(team goes over its posted win total). The half-win lines mean no push,
    so this is a clean tail probability off the simulated win distribution."""
    return float(1.0 - norm.cdf(line, loc=proj_wins, scale=max(win_sd, 0.5)))


def edge_board(model: pd.DataFrame, snap: dict, win_dist: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per team per market, model probability against de-vigged price."""
    rows = []

    def add(market, key_map, model_col, liquidity=None, note=""):
        fair = devig_power(key_map)
        raw = {k: (v / 100.0 if max(key_map.values()) > 1.5 else v) for k, v in key_map.items()}
        for team, q in fair.items():
            if team not in set(model["team"]):
                continue
            m = float(model.loc[model["team"] == team, model_col].iloc[0])
            rows.append({
                "market": market, "team": team, "model_prob": m,
                "market_price_raw": raw[team], "market_prob_fair": q,
                "edge_pp": (m - q) * 100.0,
                "edge_ratio": (m / q) if q > 0 else np.nan,
                "liquidity_usd": liquidity, "note": note,
            })

    liq = snap.get("_market_liquidity_usd", {})
    add("Super Bowl LXI", snap["superbowl_lxi"], "super_bowl", liq.get("superbowl", 6_600_483),
        "Polymarket, deepest NFL book")
    afc = {k: v for k, v in snap["afc_champion"].items()}
    nfc = {k: v for k, v in snap["nfc_champion"].items()}
    add("AFC Champion", afc, "conf_champ", liq.get("afc", 1_340_195), "Polymarket")
    add("NFC Champion", nfc, "conf_champ", liq.get("nfc", 1_563_912), "Polymarket")

    for dname, members in DIVISIONS.items():
        book = {t: snap["division_winner"][t] for t in members if t in snap["division_winner"]}
        add(f"Division: {dname}", book, "win_division",
            snap["_division_liquidity_usd"].get(dname), "Polymarket, THIN -- treat as soft")

    # Playoff market is a set of independent yes/no lines, so it is normalised
    # to the structural total of 14 berths rather than to 1.0.
    pl = snap["make_playoffs"]
    scale = 14.0 / (sum(pl.values()) / 100.0)
    for team, v in pl.items():
        if team not in set(model["team"]):
            continue
        q = (v / 100.0) * scale
        m = float(model.loc[model["team"] == team, "make_playoffs"].iloc[0])
        rows.append({"market": "Make Playoffs", "team": team, "model_prob": m,
                     "market_price_raw": v / 100.0, "market_prob_fair": q,
                     "edge_pp": (m - q) * 100.0, "edge_ratio": m / q if q > 0 else np.nan,
                     "liquidity_usd": liq.get("playoffs", 281_987),
                     "note": "Polymarket, normalised to 14 berths"})

    # Win totals are a sportsbook market. Two-way, so the vig comes off the pair.
    # Only the August book has them: once games are played the preseason line is
    # a different bet from the one a book would now offer, so in season they are
    # graded in the ledger rather than traded against here.
    wt, op = snap.get("win_totals_dk", {}), snap.get("win_total_over_price_dk", {})
    for team, line in wt.items():
        if team not in set(model["team"]):
            continue
        row = model.loc[model["team"] == team].iloc[0]
        m = win_total_prob(row["proj_wins"], row["win_sd"], line)
        q = american_to_prob(op[team])
        # Approximate the pair's overround from the standard -110/-110 structure.
        q_fair = q / (q + american_to_prob(-abs(op[team]) if op[team] > 0 else 100 * 100 / abs(op[team])))
        q_fair = min(max(q / 1.045, 0.02), 0.98)
        rows.append({"market": f"Win Total O{line}", "team": team, "model_prob": m,
                     "market_price_raw": q, "market_prob_fair": q_fair,
                     "edge_pp": (m - q_fair) * 100.0, "edge_ratio": m / q_fair,
                     "liquidity_usd": None,
                     "note": f"DraftKings O/U {line} ({op[team]:+d}); model wins {row['proj_wins']:.2f}"})

    df = pd.DataFrame(rows)
    df["conviction"] = np.where(df["edge_pp"].abs() >= EDGE_STRONG * 100, "STRONG",
                        np.where(df["edge_pp"].abs() >= EDGE_LEAN * 100, "LEAN", "-"))
    df["side"] = np.where(df["edge_pp"] > 0, "MODEL HIGHER", "MODEL LOWER")
    df = df.sort_values("edge_pp", key=abs, ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "market_edge_board.csv", index=False)
    return df


def vig_report(snap: dict, path=None) -> pd.DataFrame:
    """Publish the overround on every book. This is the number that explains why
    a market can look beatable when it is not."""
    rows = []
    books = [("Super Bowl LXI (Polymarket)", snap["superbowl_lxi"]),
             ("AFC Champion (Polymarket)", snap["afc_champion"]),
             ("NFC Champion (Polymarket)", snap["nfc_champion"])]
    # The same event on a sportsbook, converted from American odds, so the
    # overround comparison is a computed output rather than a quoted figure.
    if "superbowl_lxi_dk_american" in snap:
        dk = {t: american_to_prob(o) * 100.0 for t, o in snap["superbowl_lxi_dk_american"].items()}
        books.insert(1, ("Super Bowl LXI (DraftKings)", dk))
    for name, book in books:
        s = sum(book.values()) / 100.0
        p = devig_power(book)
        prop = devig_proportional(book)
        biggest = max(p, key=lambda t: abs(p[t] - prop[t]))
        rows.append({"book": name, "n_outcomes": len(book), "raw_sum": round(s, 4),
                     "overround_pct": round((s - 1) * 100, 2),
                     "power_k": round(_solve_k(book), 4),
                     "largest_method_gap_team": biggest,
                     "power_vs_proportional_pp": round((p[biggest] - prop[biggest]) * 100, 3)})
    for dname, members in DIVISIONS.items():
        book = {t: snap["division_winner"][t] for t in members}
        s = sum(book.values()) / 100.0
        rows.append({"book": f"Division {dname} (Polymarket)", "n_outcomes": len(book),
                     "raw_sum": round(s, 4), "overround_pct": round((s - 1) * 100, 2),
                     "power_k": round(_solve_k(book), 4),
                     "largest_method_gap_team": "", "power_vs_proportional_pp": None})
    out = pd.DataFrame(rows)
    out.to_csv(path or (OUT / "market_vig_report.csv"), index=False)
    return out


def _solve_k(prices: dict) -> float:
    p = np.array(list(prices.values()), dtype=float)
    p = np.clip(p / 100.0 if p.max() > 1.5 else p, 1e-6, 0.999999)
    try:
        return float(brentq(lambda k: float(np.sum(p ** k) - 1.0), 0.20, 6.0, xtol=1e-10))
    except ValueError:
        return 1.0
