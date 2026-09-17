"""
PYLON :: charts
Broadcast-ready visuals for every published number. Dark, high-contrast, sized
for a 1080x1920 vertical crop or a 16:9 slate.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowerss · @joshuampowers
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .config import OUT, CONF, TEAM_CONF

BG, FG, MUT = "#0B0E14", "#E8ECF4", "#7C8798"
POS, NEG, ACC = "#25D07A", "#FF4D6D", "#4CC9F0"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": "#1E2430", "grid.color": "#161B24", "font.size": 11,
    "axes.titlesize": 15, "axes.titleweight": "bold", "figure.dpi": 150,
})


def _frame(ax, title, sub=""):
    ax.set_title(title, loc="left", pad=34 if sub else 14)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, color=MUT, fontsize=9.5, va="bottom")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def power_ratings(t: pd.DataFrame, path=None):
    d = t.sort_values("power_rating")
    fig, ax = plt.subplots(figsize=(9, 11))
    cols = [POS if v > 0 else NEG for v in d["power_rating"]]
    ax.barh(d["team"], d["power_rating"], color=cols, height=0.72)
    for y, (tm, v) in enumerate(zip(d["team"], d["power_rating"])):
        ax.text(v + (0.16 if v > 0 else -0.16), y, f"{v:+.1f}", va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=FG)
    ax.axvline(0, color=MUT, lw=1)
    ax.set_xlim(d["power_rating"].min() - 1.6, d["power_rating"].max() + 1.6)
    _frame(ax, "PYLON 2026 POWER RATINGS",
           "points per game vs a league-average opponent  ·  opponent-adjusted EPA + offseason priors")
    fig.tight_layout(); fig.savefig(path or OUT / "chart_power_ratings.png"); plt.close(fig)


def market_edges(b: pd.DataFrame, path=None):
    d = b[b["market"].isin(["Super Bowl LXI", "Make Playoffs"])].copy()
    d = d[d["market"] == "Make Playoffs"].sort_values("edge_pp")
    d = pd.concat([d.head(9), d.tail(9)])
    fig, ax = plt.subplots(figsize=(9, 8))
    cols = [POS if v > 0 else NEG for v in d["edge_pp"]]
    ax.barh(d["team"], d["edge_pp"], color=cols, height=0.7)
    for y, (tm, v, m, q) in enumerate(zip(d["team"], d["edge_pp"], d["model_prob"], d["market_prob_fair"])):
        ax.text(v + (0.7 if v > 0 else -0.7), y, f"{m:.0%} vs {q:.0%}", va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=FG)
    ax.axvline(0, color=MUT, lw=1)
    lim = max(abs(d["edge_pp"])) * 1.55
    ax.set_xlim(-lim, lim)
    _frame(ax, "WHERE THE MODEL DISAGREES WITH THE MARKET",
           "playoff odds, model minus de-vigged Polymarket  ·  green = model higher")
    fig.tight_layout(); fig.savefig(path or OUT / "chart_market_edges.png"); plt.close(fig)


def stability(path=None):
    cal = json.loads((CONF / "calibration.json").read_text())["shrinkage"]
    order = ["off_pass", "off_rush", "def_pass", "def_rush"]
    labels = ["Offense\npassing", "Offense\nrushing", "Defense\npassing", "Defense\nrushing"]
    vals = [cal[k]["shrinkage"] for k in order]
    fig, ax = plt.subplots(figsize=(8, 6))
    cols = [ACC, ACC, NEG, NEG]
    ax.bar(labels, vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=14, fontweight="bold", color=FG)
    ax.set_ylim(0, 0.72); ax.set_ylabel("carry-over into the next season")
    ax.set_title("HOW MUCH OF LAST YEAR ACTUALLY CARRIES OVER", loc="left", pad=34)
    ax.text(0, 1.012, "measured out of sample, 2021-2025  ·  1.00 would mean perfectly repeatable",
            transform=ax.transAxes, color=MUT, fontsize=9.5, va="bottom")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", lw=0.6, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path or OUT / "chart_stability.png"); plt.close(fig)


def win_distribution(wd: pd.DataFrame, teams, path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(18)
    palette = [ACC, POS, NEG, "#F9C74F"]
    for i, tm in enumerate(teams):
        row = wd[wd["team"] == tm].iloc[0]
        y = [row[f"p_{w}_wins"] for w in range(18)]
        ax.plot(x, y, lw=2.4, color=palette[i % len(palette)], label=tm)
        ax.fill_between(x, y, alpha=0.10, color=palette[i % len(palette)])
    ax.set_xticks(range(0, 18, 2)); ax.set_xlabel("regular season wins")
    ax.set_ylabel("probability")
    ax.legend(frameon=False, labelcolor=FG)
    ax.set_title("A WIN TOTAL IS NOT A NUMBER, IT IS A DISTRIBUTION", loc="left", pad=34)
    ax.text(0, 1.012, "25,000 simulated seasons on the real 2026 schedule",
            transform=ax.transAxes, color=MUT, fontsize=9.5, va="bottom")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(lw=0.6, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path or OUT / "chart_win_distribution.png"); plt.close(fig)


def draft_arb(board: pd.DataFrame, path=None):
    d = board[board["verdict"].isin(["SMASH BUY", "BUY", "FADE", "HARD FADE"])].copy()
    d = pd.concat([d.nlargest(10, "arb"), d.nsmallest(10, "arb")]).sort_values("arb")
    fig, ax = plt.subplots(figsize=(9.5, 9))
    cols = [POS if v > 0 else NEG for v in d["arb"]]
    lbl = [f"{n}  ({p})" for n, p in zip(d["name"], d["pos"])]
    ax.barh(lbl, d["arb"], color=cols, height=0.7)
    for y, (v, a) in enumerate(zip(d["arb"], d["adp_formatted"])):
        ax.text(v + (2 if v > 0 else -2), y, f"ADP {a}", va="center",
                ha="left" if v > 0 else "right", fontsize=8.5, color=MUT)
    ax.axvline(0, color=MUT, lw=1)
    lim = max(abs(d["arb"])) * 1.35
    ax.set_xlim(-lim, lim); ax.set_xlabel("draft slots of disagreement with the room")
    _frame(ax, "DRAFT BOARD ARBITRAGE",
           "model value rank minus live ADP  ·  green = the model wants him earlier")
    fig.tight_layout(); fig.savefig(path or OUT / "chart_draft_arb.png"); plt.close(fig)


def backtest_scatter(path=None):
    bt = pd.read_csv(OUT / "backtest_win_projections.csv")
    meta = json.loads((CONF / "backtest.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(bt["pred_wins"], bt["actual_wins"], s=26, alpha=0.65, color=ACC, edgecolor="none")
    lims = [2, 15]
    ax.plot(lims, lims, color=MUT, lw=1.2, ls="--")
    ax.set_xlim(3, 13); ax.set_ylim(0, 18)
    ax.set_xlabel("projected wins (built only from prior seasons)")
    ax.set_ylabel("actual wins")
    ax.set_title("THE HONEST BACKTEST", loc="left", pad=34)
    ax.text(0, 1.012, f"{meta['n_team_seasons']} team-seasons, 2021-2025  ·  "
                     f"MAE {meta['mae_wins']} wins  ·  r = {meta['correlation_pred_actual']}",
            transform=ax.transAxes, color=MUT, fontsize=9.5, va="bottom")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(lw=0.6, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path or OUT / "chart_backtest.png"); plt.close(fig)


def build_all(t, b, wd, board):
    power_ratings(t); market_edges(b); stability()
    win_distribution(wd, ["SEA", "ATL", "BUF", "MIA"])
    draft_arb(board); backtest_scatter()
    return sorted(str(p.name) for p in OUT.glob("chart_*.png"))
