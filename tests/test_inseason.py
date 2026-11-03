"""
PYLON :: in-season tests
The live model has more ways to be quietly wrong than the August one, because it
mixes two sources of truth: what has happened and what is projected. Each test
pins one place where those two could disagree without anything crashing.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest

from pylon import inseason, ledger, simulate
from pylon.config import OUT, RAW, CONF

needs_raw = pytest.mark.skipif(not (RAW / "schedules.csv").exists(),
                               reason="no data/raw; run ./scripts/fetch_data.sh first")


def status():
    f = CONF / "inseason_status.json"
    return json.loads(f.read_text()) if f.exists() else {}


in_season = pytest.mark.skipif(not status().get("applied"), reason="no completed week yet")


# --------------------------------------------------------------------------
# the blend
# --------------------------------------------------------------------------
def test_blend_weight_is_the_conjugate_form():
    assert inseason.blend_weight(0, 8) == 0.0
    assert inseason.blend_weight(8, 8) == pytest.approx(0.5)
    w = inseason.blend_weight(np.arange(18), 8)
    assert (np.diff(w) > 0).all() and w.max() < 1


def test_posterior_team_error_shrinks_by_root_one_minus_w():
    w = pd.Series([0.0, 0.5, 0.75])
    got = inseason.posterior_sigma_team(4.0, w)
    assert got.tolist() == pytest.approx([4.0, 4.0 * np.sqrt(0.5), 2.0])


def test_every_fitted_constant_beats_both_of_its_ingredients():
    """A blend that cannot beat the prior alone and the season alone is a
    reason to delete the blend, not to ship it."""
    f = CONF / "inseason_players.json"
    if not f.exists():
        pytest.skip("run python -m pylon.inseason first")
    for col, fit in json.loads(f.read_text())["fits"].items():
        assert fit["rmse"] <= fit["rmse_prior_only"], col
        assert fit["rmse"] <= fit["rmse_season_only"], col


@in_season
def test_rating_identity_holds_after_blending():
    r = pd.read_csv(OUT / "team_power_ratings.csv")
    assert np.allclose(r["off_final"], r["off_rating"] + r["off_delta"])
    assert np.allclose(r["def_final"], r["def_rating"] + r["def_delta"])
    assert np.allclose(r["net_final"], r["off_final"] - r["def_final"])
    w = r["w_inseason"]
    assert np.allclose(r["off_delta"], (1 - w) * r["pre_off_delta"])


# --------------------------------------------------------------------------
# the simulation respects what already happened
# --------------------------------------------------------------------------
@needs_raw
@in_season
def test_completed_games_are_locked_in_every_simulated_season():
    sched = simulate.load_schedule()
    done = sched[sched["result"].notna()]
    wins = pd.concat([done.loc[done["result"] > 0, "home_team"],
                      done.loc[done["result"] < 0, "away_team"]]).value_counts()
    played = pd.concat([done["home_team"], done["away_team"]]).value_counts()
    wd = pd.read_csv(OUT / "win_distributions.csv").set_index("team")
    for team, row in wd.iterrows():
        w0 = int(wins.get(team, 0))
        left = 17 - int(played.get(team, 0))
        probs = row.to_numpy()
        assert probs[:w0].sum() == pytest.approx(0.0, abs=1e-12), team
        assert probs[w0 + left + 1:].sum() == pytest.approx(0.0, abs=1e-12), team


@needs_raw
def test_completed_through_means_every_game_is_final():
    g = inseason.schedule()
    through = inseason.completed_through()
    assert g.loc[g["week"] <= through, "result"].notna().all()
    if through < g["week"].max():
        assert g.loc[g["week"] == through + 1, "result"].isna().any()


# --------------------------------------------------------------------------
# players: to date + rest of season
# --------------------------------------------------------------------------
@in_season
def test_season_total_is_banked_plus_remaining():
    p = pd.read_csv(OUT / "player_projections.csv")
    assert np.allclose(p["proj_points"], p["points_to_date"] + p["proj_ros"], atol=1e-6)
    assert (p["p10"] >= p["points_to_date"] - 1e-6).all(), "a simulation cannot take back scored points"


@in_season
def test_reserve_list_players_are_not_counted_back():
    p = pd.read_csv(OUT / "player_projections.csv")
    res = p[p["status"] == "RES"]
    assert (res["proj_ros"].abs() < 1e-9).all()


@in_season
def test_value_is_measured_on_what_is_left():
    p = pd.read_csv(OUT / "player_projections.csv")
    for pos, g in p.groupby("position"):
        top = g.sort_values("vorp", ascending=False).iloc[0]
        assert top["proj_ros"] == g["proj_ros"].max(), pos


# --------------------------------------------------------------------------
# the ledgers
# --------------------------------------------------------------------------
def test_august_ledger_still_verifies():
    assert ledger.verify(), "the sealed ledger was edited after it was stamped"


def test_ledger_build_never_overwrites_a_sealed_season(tmp_path):
    before = ledger.LEDGER.read_bytes()
    out = ledger.build(pd.DataFrame(), pd.DataFrame())
    assert ledger.LEDGER.read_bytes() == before
    assert out["digest"] == json.loads(before)["digest"]


def test_weekly_seals_were_written_before_kickoff():
    f = ledger.WEEKLY_LEDGER
    if not f.exists():
        pytest.skip("no week sealed yet")
    for wk, e in json.loads(f.read_text())["weeks"].items():
        assert e["digest"] == ledger._digest(e), f"week {wk} edited after sealing"
        if e.get("first_kickoff_utc"):
            assert e["stamped_at"] < e["first_kickoff_utc"], f"week {wk} sealed after kickoff"


@in_season
def test_draft_board_stays_frozen_in_season():
    """The in-season ADP feed is a handful of mock drafts. A board rebuilt on it
    strips the ADP from almost every player on the site."""
    assert len(pd.read_csv(OUT / "draft_board.csv")) > 150


@in_season
def test_site_serves_the_week_the_model_projected():
    b = json.loads((OUT / "web_bundle.json").read_text())
    assert b["meta"]["week"] == status()["current_week"]
    wk = b["matchups"]["rows"]
    if wk:
        i = b["matchups"]["cols"].index("week")
        assert {r[i] for r in wk} == {status()["current_week"]}
