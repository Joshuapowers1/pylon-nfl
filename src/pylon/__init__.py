"""
PYLON
Play-level Yield & League Outcome Network

An NFL projection and market-edge engine. Opponent-adjusted ridge ratings on
nine seasons of play-by-play, shrinkage and translation constants fitted out of
sample, a 25,000-season Monte Carlo on the real schedule, opportunity-based
player projections, and a de-vigged comparison against live prediction markets.

--------------------------------------------------------------------------
Joshua Powers
joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl
linkedin.com/in/joshuampowers
instagram.com/joshuampowers

Released under the MIT License. See LICENSE.
"""
__version__ = "1.0.0"
__author__ = "Joshua Powers"
__email__ = "joshpowersbiz@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/Joshuapowers1/pylon-nfl"
__all__ = ["config", "ingest", "usage", "ratings", "calibrate", "backtest",
           "adjustments", "game_model", "simulate", "players", "draft",
           "market", "diagnostics", "charts", "webdata", "run"]
