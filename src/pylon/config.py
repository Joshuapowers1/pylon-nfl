"""
PYLON :: config
Every tunable lives here. No magic numbers buried in the model code.
Each constant carries the reason it exists and, where relevant, how it was calibrated.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowerss · @joshuampowers
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "out"
CONF = ROOT / "config"
for _p in (RAW, PROC, OUT, CONF):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------------
TARGET_SEASON = 2026
TRAIN_SEASONS = [2023, 2024, 2025]

# Exponential recency weights applied to each training season when blending
# opponent-adjusted ratings. Tuned so the most recent year drives the estimate
# while two prior years still supply stabilising sample.
SEASON_WEIGHTS = {2025: 1.00, 2024: 0.45, 2023: 0.18}

# ---------------------------------------------------------------------------
# Ridge regression (opponent adjustment)
# ---------------------------------------------------------------------------
# One coefficient per team-offense and per team-defense, fit on play-level EPA.
# Alpha is the L2 penalty; higher alpha pulls every unit toward league average.
# Chosen by 5-fold CV on 2023-2025 held-out games (see notebooks/ridge_cv.md).
RIDGE_ALPHA_PASS = 220.0
RIDGE_ALPHA_RUSH = 260.0

# Plays are dropped outside this win-probability band so that garbage time
# (kneel-downs, prevent defense, 4-minute offense) does not pollute the ratings.
WP_FLOOR, WP_CEIL = 0.08, 0.92

# ---------------------------------------------------------------------------
# Year-over-year stability (regression to the mean)
# ---------------------------------------------------------------------------
# Measured autocorrelation of opponent-adjusted EPA/play, 2016-2025.
# Offense is materially stickier than defense, which is the single most
# under-priced fact in preseason NFL markets.
STABILITY_OFF_PASS = 0.58
STABILITY_OFF_RUSH = 0.42
STABILITY_DEF_PASS = 0.36
STABILITY_DEF_RUSH = 0.31

# ---------------------------------------------------------------------------
# Prior confidence
# ---------------------------------------------------------------------------
# The measured shrinkage on defense is brutal (roughly 0.15), which means the
# data layer supplies almost no team-to-team defensive differentiation. Left
# unweighted, the hand-specified roster deltas would then completely determine
# the defensive rankings, which is not a claim the evidence supports either.
# These weights scale the prior layer so it supplies real signal without
# swamping the model. Defense carries a lower weight than offense because
# defensive personnel impact is the harder thing to forecast, not the easier.
# Sensitivity to both weights is reported in out/sensitivity_priors.csv.
PRIOR_CONFIDENCE_OFF = 0.85
PRIOR_CONFIDENCE_DEF = 0.55

# ---------------------------------------------------------------------------
# Game model
# ---------------------------------------------------------------------------
# Calibrated by OLS of actual game margin on net-rating differential.
# See game_model.calibrate() — these are the fallbacks if calibration is skipped.
PLAYS_PER_TEAM_GAME = 63.0
HOME_FIELD_POINTS = 1.7          # post-2020 HFA has compressed from the historic ~2.6
MARGIN_SIGMA = 13.2              # SD of (actual margin - predicted margin)

# ---------------------------------------------------------------------------
# Season simulation
# ---------------------------------------------------------------------------
N_SIMS = 25_000
TIE_PROBABILITY = 0.0022         # empirical rate of regular-season ties, 2012-2025
RANDOM_SEED = 1207               # fixed so every published number is reproducible

# ---------------------------------------------------------------------------
# Fantasy scoring (full PPR)
# ---------------------------------------------------------------------------
SCORING = dict(
    pass_yd=0.04, pass_td=4.0, interception=-2.0,
    rush_yd=0.10, rush_td=6.0,
    rec=1.0, rec_yd=0.10, rec_td=6.0,
    fumble_lost=-2.0,
)

# Fantasy starters per position in a 12-team league, used for replacement level.
REPLACEMENT_RANK = {"QB": 14, "RB": 30, "WR": 38, "TE": 13}

# ---------------------------------------------------------------------------
# Market layer
# ---------------------------------------------------------------------------
# Minimum liquidity (USD) before a Polymarket line is treated as informative
# rather than noise. Division books on Polymarket sit far below this.
MIN_MARKET_LIQUIDITY = 50_000
# Edge thresholds for the published board.
EDGE_STRONG = 0.03
EDGE_LEAN = 0.015

TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LA","LAC","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
]

# nflverse uses LA for the Rams; most market feeds use LAR. Normalise everything.
TEAM_ALIASES = {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "WSH": "WAS", "WFT": "WAS", "JAC": "JAX"}

DIVISIONS = {
    "AFC East": ["BUF","MIA","NE","NYJ"],   "AFC North": ["BAL","CIN","CLE","PIT"],
    "AFC South": ["HOU","IND","JAX","TEN"], "AFC West":  ["DEN","KC","LAC","LV"],
    "NFC East": ["DAL","NYG","PHI","WAS"],  "NFC North": ["CHI","DET","GB","MIN"],
    "NFC South": ["ATL","CAR","NO","TB"],   "NFC West":  ["ARI","LA","SEA","SF"],
}
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}
TEAM_CONF = {t: d.split()[0] for d, ts in DIVISIONS.items() for t in ts}
