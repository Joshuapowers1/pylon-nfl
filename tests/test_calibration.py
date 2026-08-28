"""
PYLON :: tests :: calibration
Guards the fitted constants. These are allowed to move when the model is
retuned, but not to become impossible.
"""
from __future__ import annotations
import json
import pytest
from pylon.config import CONF


@pytest.fixture(scope="module")
def cal():
    return json.loads((CONF / "calibration.json").read_text())


@pytest.fixture(scope="module")
def bt():
    return json.loads((CONF / "backtest.json").read_text())


def test_shrinkage_is_a_valid_weight(cal):
    for phase, v in cal["shrinkage"].items():
        assert 0.0 <= v["shrinkage"] <= 1.0, phase


def test_offense_repeats_more_than_defense(cal):
    """The central empirical claim of the whole project."""
    s = cal["shrinkage"]
    assert s["off_pass"]["shrinkage"] > s["def_pass"]["shrinkage"]
    assert s["off_rush"]["shrinkage"] > s["def_rush"]["shrinkage"]


def test_home_field_is_the_modern_value(cal):
    assert 1.0 < cal["translation"]["hfa"] < 3.0


def test_the_model_beats_a_coin_flip_but_not_the_market(bt):
    acc = bt["straight_up_game_accuracy"]
    assert 0.52 < acc < 0.66, "a preseason model claiming to beat a closing line is a red flag"


def test_variance_split_is_consistent(bt):
    v = bt["variance_split"]
    total = v["sigma_total_pts"] ** 2
    parts = v["sigma_game_pts"] ** 2 + 2 * v["sigma_team_pts"] ** 2
    assert parts == pytest.approx(total, rel=0.02)


def test_in_season_updating_beats_a_frozen_rating():
    f = CONF / "inseason.json"
    if not f.exists():
        pytest.skip("run pylon.update.fit_blend first")
    d = json.loads(f.read_text())
    assert d["updated"]["accuracy"] > d["frozen"]["accuracy"]
    assert d["updated"]["mae"] < d["frozen"]["mae"]
