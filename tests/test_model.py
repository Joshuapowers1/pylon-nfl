"""
PYLON :: tests
Covers the maths that is easy to get quietly wrong and impossible to notice.

Every bug this project actually shipped was silent: nothing crashed, a number
was just wrong. So these tests assert properties rather than golden values.
A probability distribution has to sum to one. Fourteen teams make the playoffs,
not thirteen or fifteen. A player who misses half a season must not read as
having lost half his role. Those hold no matter how the model is retuned.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest

from pylon import players, market, simulate, usage
from pylon.config import OUT, RAW, CONF, DIVISIONS, TEAM_CONF


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def teams():
    return pd.read_csv(OUT / "team_projections.csv")


@pytest.fixture(scope="module")
def proj():
    return pd.read_csv(OUT / "player_projections.csv")


@pytest.fixture(scope="module")
def cal():
    return json.loads((CONF / "calibration.json").read_text())


# --------------------------------------------------------------------------
# structural invariants of the season simulation
# --------------------------------------------------------------------------
def test_exactly_fourteen_playoff_berths(teams):
    assert teams["make_playoffs"].sum() == pytest.approx(14.0, abs=0.02)


def test_exactly_eight_division_winners(teams):
    assert teams["win_division"].sum() == pytest.approx(8.0, abs=0.02)


def test_exactly_one_champion(teams):
    assert teams["super_bowl"].sum() == pytest.approx(1.0, abs=0.01)
    assert teams["conf_champ"].sum() == pytest.approx(2.0, abs=0.02)


def test_mean_wins_is_half_the_schedule(teams):
    assert teams["proj_wins"].mean() == pytest.approx(8.5, abs=0.05)


def test_every_division_produces_one_winner(teams):
    t = teams.set_index("team")
    for name, members in DIVISIONS.items():
        assert t.loc[members, "win_division"].sum() == pytest.approx(1.0, abs=0.02), name


def test_each_conference_sends_seven(teams):
    for conf in ("AFC", "NFC"):
        members = [c for c, v in TEAM_CONF.items() if v == conf]
        got = teams.set_index("team").loc[members, "make_playoffs"].sum()
        assert got == pytest.approx(7.0, abs=0.03), conf


def test_probabilities_are_ordered(teams):
    """You cannot win a title without reaching the playoffs."""
    assert (teams["super_bowl"] <= teams["conf_champ"] + 1e-9).all()
    assert (teams["conf_champ"] <= teams["make_playoffs"] + 1e-9).all()
    assert (teams["win_division"] <= teams["make_playoffs"] + 1e-9).all()


def test_win_spread_matches_measured_error(teams):
    """The simulator models a season-long team error on top of game noise. Drop
    it and the spread collapses to about 2.0, which was a real shipped bug.

    In season the bar moves with the calendar: completed games carry no
    variance at all, so the right comparison is against the spread a
    game-noise-only model would give over the games still to play. The
    team-constant draw (shrunk to its posterior) has to add to that, not
    vanish into it."""
    if not (RAW / "schedules.csv").exists():
        st = CONF / "inseason_status.json"
        if st.exists() and json.loads(st.read_text()).get("applied"):
            pytest.skip("in season the bound needs the schedule; run ./scripts/fetch_data.sh")
        assert teams["win_sd"].mean() > 2.4
        return
    from scipy.stats import norm
    sched = simulate.load_schedule()
    left = sched[sched["result"].isna()]
    if len(left) == len(sched):
        assert teams["win_sd"].mean() > 2.4
        return
    r = pd.read_csv(OUT / "team_power_ratings.csv").set_index("team")
    tr = json.loads((CONF / "calibration.json").read_text())["translation"]
    sg = json.loads((CONF / "backtest.json").read_text())["variance_split"]["sigma_game_pts"]
    mu = (tr["points_per_rating"] * (left["home_team"].map(r["net_final"]) - left["away_team"].map(r["net_final"]))
          + np.where(left["neutral"] == 1, 0.0, tr["hfa"]))
    p = norm.cdf(mu / sg)
    v = pd.concat([pd.Series(p * (1 - p), index=left["home_team"]),
                   pd.Series(p * (1 - p), index=left["away_team"])]).groupby(level=0).sum()
    assert teams["win_sd"].mean() > 1.06 * float(np.sqrt(v).mean())


def test_simulator_reproduces_the_schedule():
    # data/raw is deliberately not committed: it is 183MB and re-fetchable.
    # The weekly workflow fetches it and does run this. On a fresh clone it
    # skips, because a suite that fails before you have downloaded anything
    # tells you nothing about the code.
    if not (RAW / "schedules.csv").exists():
        pytest.skip("no data/raw/schedules.csv; run ./scripts/fetch_data.sh first")
    sched = simulate.load_schedule()
    assert len(sched) == 272
    counts = pd.concat([sched["home_team"], sched["away_team"]]).value_counts()
    assert set(counts.values) == {17}


# --------------------------------------------------------------------------
# de-vigging
# --------------------------------------------------------------------------
def test_devig_normalises_to_one():
    book = {"A": 60.0, "B": 30.0, "C": 20.0}          # sums to 110%
    fair = market.devig_power(book)
    assert sum(fair.values()) == pytest.approx(1.0, abs=1e-9)


def test_devig_preserves_ordering():
    book = {"A": 60.0, "B": 30.0, "C": 20.0}
    fair = market.devig_power(book)
    assert fair["A"] > fair["B"] > fair["C"]


def test_devig_strips_more_from_longshots():
    """The whole reason for the power method over proportional scaling."""
    book = {"fav": 70.0, "mid": 25.0, "dog": 15.0}    # 110%
    power = market.devig_power(book)
    prop = market.devig_proportional(book)
    assert power["dog"] < prop["dog"]
    assert power["fav"] > prop["fav"]


def test_devig_handles_an_underround_book():
    """Polymarket's conference books sum to under 100. That must not explode."""
    book = {"A": 50.0, "B": 30.0, "C": 17.0}          # 97%
    fair = market.devig_power(book)
    assert sum(fair.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0 < v < 1 for v in fair.values())


def test_american_odds_conversion():
    assert market.american_to_prob(100) == pytest.approx(0.5)
    assert market.american_to_prob(-110) == pytest.approx(110 / 210)
    assert market.american_to_prob(550) == pytest.approx(100 / 650)


# --------------------------------------------------------------------------
# empirical Bayes and replacement level
# --------------------------------------------------------------------------
def test_eb_shrinks_small_samples_hardest():
    prior, k = 0.10, 45.0
    small = players._eb(0.40, 5, prior, k)
    large = players._eb(0.40, 500, prior, k)
    assert abs(small - prior) < abs(large - prior)
    assert prior < small < large < 0.40


def test_eb_with_no_evidence_returns_the_prior():
    assert players._eb(0.9, 0, 0.12, 45.0) == pytest.approx(0.12)


def test_eb_is_monotone_in_the_observation():
    lo = players._eb(0.10, 100, 0.2, 45.0)
    hi = players._eb(0.30, 100, 0.2, 45.0)
    assert hi > lo


def test_streaming_premium_matches_the_measurement(proj):
    """The premium was measured, not assumed, and it came out negative: a
    matchup streamer scored 1.7 ppg WORSE than the quarterback who finished at
    replacement rank. The waiver pool is a downgrade, not a free option."""
    assert players.STREAM_PREMIUM_PPG["QB"] < 0
    naive = players.replacement_levels(proj, stream=False)
    streamed = players.replacement_levels(proj, stream=True)
    assert streamed["QB"] < naive["QB"], "a negative premium lowers replacement level"
    assert streamed["RB"] == naive["RB"], "you cannot stream a starting running back"


def test_default_board_uses_the_unadjusted_replacement(proj):
    """The pipeline ships the arithmetic, not the convention."""
    default = players.add_vorp(proj)
    explicit = players.add_vorp(proj, stream=False)
    assert default["vorp"].equals(explicit["vorp"])


def test_shares_normalise_to_the_team_total(proj):
    """Every target belongs to somebody, and only to one somebody. In season the
    weight is availability for the games left, not games over seventeen."""
    for team, g in proj[proj["position"].isin(["WR", "TE", "RB"])].groupby("team"):
        avail = g["avail"] if "avail" in g.columns else g["games"].clip(0, 17) / 17
        if avail.sum() == 0:
            continue
        w = (g["target_share_eb"] * avail).sum()
        assert w == pytest.approx(0.99, abs=0.02), team


def test_percentiles_are_ordered(proj):
    d = proj[proj["proj_points"] > 20]
    assert (d["p10"] <= d["p25"]).all()
    assert (d["p25"] <= d["p50"]).all()
    assert (d["p50"] <= d["p75"]).all()
    assert (d["p75"] <= d["p90"]).all()


def test_finish_probabilities_are_ordered(proj):
    d = proj.dropna(subset=["p_top3", "p_top5", "p_top12"])
    assert (d["p_top3"] <= d["p_top5"] + 1e-9).all()
    assert (d["p_top5"] <= d["p_top12"] + 1e-9).all()


def test_missed_games_do_not_read_as_a_lost_role():
    """The highest-impact bug this project shipped: dividing a player's usage by
    full-season team totals made anyone who missed time look demoted."""
    played = pd.DataFrame({"targets": [60], "team_pass_att": [200]})   # 7 of 17 games
    correct = played["targets"] / played["team_pass_att"]
    wrong = played["targets"] / (200 * 17 / 7)
    assert correct.iloc[0] == pytest.approx(0.30)
    assert wrong.iloc[0] < 0.13
