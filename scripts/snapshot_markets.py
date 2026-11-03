"""Rebuilds config/market_snapshot_live.json from the feeds fetch_data.sh pulled.

The August snapshot (config/market_snapshot_2026.json) is the market the sealed
ledger was graded against, so it is never edited. This one is the market as it
stands on the day of the refit, in the same shape, so the edge board compares
the current model with the current price instead of with a price from before
anybody had played a down.

  Polymarket  futures books from data/raw/polymarket/events_nfl.json, priced at
              the order-book midpoint, the convention the August file used.
  DraftKings  this week's spreads and totals from ESPN's scoreboard feed.

Run after fetch_data.sh:   python scripts/snapshot_markets.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pylon.config import TEAM_ALIASES, DIVISIONS, TARGET_SEASON  # noqa: E402
from pylon.webdata import TEAM_META  # noqa: E402
from pylon.inseason import schedule  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "config" / "market_snapshot_live.json"
NAME_TO_TEAM = {f"{v[0]} {v[1]}".lower(): k for k, v in TEAM_META.items()}
ESPN = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        "?seasontype=2&week={week}&dates={season}")


def _mid(m: dict) -> float | None:
    """Midpoint when both sides of the book are quoted, last trade otherwise.

    A book with a one-cent bid and a dollar ask is not a price, it is an empty
    order book, so that case falls through to the last trade as well."""
    bid, ask = m.get("bestBid"), m.get("bestAsk")
    if bid is not None and ask is not None and 0 < float(ask) - float(bid) < 0.5:
        return (float(bid) + float(ask)) / 2.0
    if m.get("lastTradePrice") not in (None, 0):
        return float(m["lastTradePrice"])
    op = m.get("outcomePrices")
    if op:
        return float(json.loads(op)[0])
    return None


def _book(event: dict) -> dict:
    out = {}
    for m in event.get("markets", []):
        team = NAME_TO_TEAM.get(str(m.get("groupItemTitle", "")).strip().lower())
        if team is None or m.get("closed"):
            continue
        px = _mid(m)
        if px is not None:
            out[team] = round(px * 100.0, 3)
    return out


def polymarket() -> dict:
    ev = json.loads((RAW / "polymarket" / "events_nfl.json").read_text())
    by_title = {e.get("title", "").strip(): e for e in ev}

    def find(prefix):
        hits = [e for t, e in by_title.items() if t.startswith(prefix)]
        return hits[0] if hits else None

    snap = {"_division_liquidity_usd": {}, "_market_liquidity_usd": {}, "division_winner": {}}
    for key, liq, prefix in (("superbowl_lxi", "superbowl", "Pro Football: 2027 Champion"),
                             ("afc_champion", "afc", "Pro Football: 2027 AFC Champion"),
                             ("nfc_champion", "nfc", "Pro Football: 2027 NFC Champion"),
                             ("make_playoffs", "playoffs", "Pro Football: Team to Make Postseason")):
        e = find(prefix)
        if e is None:
            raise SystemExit(f"Polymarket book missing: {prefix!r}. The feed changed shape.")
        snap[key] = _book(e)
        snap["_market_liquidity_usd"][liq] = round(float(e.get("liquidity") or 0))
        if key == "superbowl_lxi":
            snap["source_superbowl"] = f"https://polymarket.com/event/{e.get('slug')}"
    for dname in DIVISIONS:
        e = find(f"Pro Football: {dname} Champion")
        if e is None:
            raise SystemExit(f"Polymarket division book missing: {dname}")
        snap["division_winner"].update(_book(e))
        snap["_division_liquidity_usd"][dname] = round(float(e.get("liquidity") or 0))
    return snap


def week_lines(week: int, season: int = TARGET_SEASON) -> list[dict]:
    req = urllib.request.Request(ESPN.format(week=week, season=season),
                                 headers={"User-Agent": "pylon-nfl weekly refit"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode())
    lines = []
    for ev in d.get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: TEAM_ALIASES.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
                 for c in comp["competitors"]}
        odds = (comp.get("odds") or [{}])[0]
        try:
            spread_home = float(odds["pointSpread"]["home"]["close"]["line"])
        except (KeyError, TypeError, ValueError):
            spread_home = None
        total = odds.get("overUnder")
        if spread_home is None or total is None:
            continue
        lines.append({"date": ev.get("date"), "away": teams["away"], "home": teams["home"],
                      "spread_home": spread_home, "total": float(total),
                      "provider": (odds.get("provider") or {}).get("name")})
    return lines


def main() -> None:
    g = schedule()
    unplayed = g[g["result"].isna()]
    week = int(unplayed["week"].min()) if len(unplayed) else int(g["week"].max())
    snap = {
        "_doc": ("Live market snapshot, rebuilt by every refit. Polymarket prices are "
                 "order-book midpoints (bestBid+bestAsk)/2 from the Gamma API, last trade when "
                 "one side of the book is empty. Game lines are ESPN's DraftKings feed for the "
                 "current week. The August snapshot the ledger was sealed against is "
                 "config/market_snapshot_2026.json and is never rewritten."),
        "pulled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "week": week,
        **polymarket(),
    }
    try:
        snap["week_lines_dk"] = {str(week): week_lines(week)}
    except Exception as exc:          # a missing line is not worth a failed refit
        print(f"  week {week} lines unavailable: {exc}")
        snap["week_lines_dk"] = {str(week): []}
    OUT.write_text(json.dumps(snap, indent=2))
    n = len(snap["week_lines_dk"][str(week)])
    print(f"live snapshot -> {OUT.relative_to(ROOT)}: {len(snap['superbowl_lxi'])} title prices, "
          f"{len(snap['division_winner'])} division prices, {n} week {week} lines")


if __name__ == "__main__":
    main()
