<div align="center">

# PYLON

**Play-level Yield & League Outcome Network**

An NFL projection and market-edge engine. Opponent-adjusted ridge ratings on nine seasons of play-by-play, shrinkage measured out of sample rather than assumed, 25,000 simulated seasons on the real schedule, and a de-vigged comparison against live prediction markets.

[**Open the live terminal →**](https://joshuapowers1.github.io/pylon-nfl/)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ridge-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-DB5A00)
[![tests](https://github.com/Joshuapowers1/pylon-nfl/actions/workflows/tests.yml/badge.svg)](https://github.com/Joshuapowers1/pylon-nfl/actions/workflows/tests.yml)
![Backtest](https://img.shields.io/badge/backtest-160%20team--seasons-2a78d6)
![Runtime deps](https://img.shields.io/badge/frontend%20dependencies-0-1baf7a)

</div>

![Overview](docs/01-overview.jpg)

---

## The finding

I wanted to know how much of an NFL team's season actually carries into the next one. Not opinions, a number. So I regressed each season's opponent-adjusted efficiency on a blend built only from the three seasons before it, across 2021 through 2025.

| Unit | Carry-over into the next season |
|---|---|
| Offense, passing | **0.57** |
| Offense, rushing | **0.42** |
| Defense, passing | **0.14** |
| Defense, rushing | **0.17** |

Offense is roughly three to four times more repeatable than defense. The betting market prices last year's defense close to face value. That single gap drives nearly every disagreement this model has with the market, including having the defending champions outside the top fifteen.

---

## The honest backtest

Every projection below was built with **zero information from the season it predicts**.

| Metric | Value |
|---|---|
| Team-seasons tested | 160 (2021-2025) |
| Mean absolute error, win totals | **2.39 wins** |
| RMSE | 2.87 wins |
| Correlation, projected vs actual | 0.41 |
| Within 2 wins | 46.9% |
| Straight-up game accuracy | **58.2%** |

Read that honestly. 58.2% straight up is well above a coin flip and well below a closing Vegas line, which lands near 66%. This is a serious estimate that disagrees with the market in specific, defensible places. It is not an edge-printing machine, and any preseason model claiming a much better backtest without publishing this table should be treated with suspicion.

---

## In-season updating

A preseason rating is a prior. The moment real games exist, quoting week zero all
year is the worst version of the model. So it refits, and the refit is measured
rather than asserted: three completed seasons are replayed week by week, with the
updated ratings rebuilt from only the games already played before predicting the
games that came next.

| | Frozen at week zero | Updated weekly |
|---|---|---|
| Straight-up accuracy | 59.0% | **63.2%** |
| Mean absolute error | 10.96 pts | **10.42 pts** |
| Log loss | 0.668 | **0.639** |

**In-season updating is worth 4.2 percentage points**, which closes most of the gap
to a closing Vegas line at roughly 66%.

![Track record](docs/03-track-record.jpg)

The blend is a conjugate-style weight, `rating = w * in-season + (1 - w) * preseason`
with `w = games / (games + k)`. The constant `k` is fitted by sweeping it across
three seasons and keeping whichever value predicted best. It landed on **8**, so this
season earns half the weight after eight games.

The honest detail: through week five the updating model is *worse* than the frozen
one. Four games of data is mostly noise. It takes about six weeks before the season
tells you more than the offseason did, and the weekly curve on the site shows it.

`.github/workflows/weekly-update.yml` runs the whole refit every Tuesday and commits
the result, so the published site does this on its own once the season starts.

---

## Quickstart

```bash
git clone https://github.com/Joshuapowers1/pylon-nfl.git
cd pylon-nfl
pip install -r requirements.txt

./scripts/fetch_data.sh            # play-by-play, rosters, schedule, live ADP, Polymarket
PYTHONPATH=src python -m pylon.run --sims 25000 --player-sims 10000
```

Runs end to end in about 22 seconds and writes every published number to `out/`. Then rebuild the web app:

```bash
PYTHONPATH=src python -m pylon.webdata   # pack outputs into a single JSON bundle
cd web && ./build.sh                     # inline the bundle and logos into index.html
node test.mjs                            # headless check: all pages, both themes, three breakpoints
```

`build.sh` writes the built page to both `web/index.html` and `docs/index.html`. GitHub Pages serves
from `/docs` on `main`, so the published site is always the file that was just built and tested.

---

## What it produces

| Output | File |
|---|---|
| Power ratings, all 32 teams, in points vs an average opponent | `out/team_power_ratings.csv` |
| Win totals, playoff, division and championship odds | `out/team_projections.csv` |
| Full win distribution, 0 through 17, per team | `out/win_distributions.csv` |
| Playoff seed distribution, 1 through 7 | `out/seed_distributions.csv` |
| Fantasy projections for 581 players with percentile bands and finish probabilities | `out/player_projections.csv` |
| Draft board against 7,726 live PPR drafts | `out/draft_board.csv` |
| Model vs de-vigged market, every book | `out/market_edge_board.csv` |
| Overround on every market used | `out/market_vig_report.csv` |
| Five-season out-of-sample backtest | `out/backtest_win_projections.csv` |
| How much of each rating is data vs analyst judgment | `out/rating_decomposition.csv` |
| Sensitivity of every projection to the judgment layer | `out/sensitivity_priors.csv` |
| Game-by-game spread, total and win probability vs the posted line | `out/matchups_week1.csv` |
| Frozen vs updated accuracy, week by week | `out/inseason_curve.csv` |
| This season's predictions, sealed with a timestamp and a digest | `out/prediction_ledger.json` |

---

## How it works

### 1. Strip out the schedule

Raw efficiency leaderboards are schedule artefacts. A team that drew six bottom-five defenses looks elite and is not. So instead of averaging expected points added, Pylon regresses it:

```
epa_play ~ b0 + off_team + def_team + home_offense
```

fit separately for pass and rush plays with an L2 penalty, across 81,389 competitive-win-probability plays. Every team gets one offensive and one defensive coefficient per phase, in points per play above league average, with strength of schedule already divided out. Garbage time is excluded by a win-probability band, because kneel-downs and prevent defense carry no predictive signal.

### 2. Shrink by a measured amount

The shrinkage table above is not chosen. It is the slope of a regression of actual on predicted, fit out of sample. Defense regresses far harder than offense because it repeats far less.

### 3. Price the offseason, and publish the receipts

Play-by-play cannot know that a receiver was traded or that a new coordinator is calling plays. Those are hand-priced in `config/offseason_2026.json`, one entry per team, each with a written justification and a documented magnitude scale.

This is the only subjective layer in the engine, so it ships with two audits. `out/rating_decomposition.csv` reports per team how many points of the final rating came from measured data and how many from the prior. `out/sensitivity_priors.csv` re-runs the whole simulation across a grid of prior-confidence weights. Five teams are more than 80% judgment-driven, and the site names them.

### 4. Translate to points, calibrated on real games

Fit on genuine forecasts, not in-sample:

```
expected home margin = 2.05 + 89.77 * (net_rating_home - net_rating_away)
residual sigma = 13.75 points
```

Home-field advantage of 2.05 points is the post-2020 reality, not the historical 2.6 that most public models still carry.

### 5. Simulate, with the uncertainty that actually matters

25,000 Monte Carlo seasons on the real schedule, with the full NFL seeding cascade and the current seven-team bracket.

The detail that separates this from most public simulators: the residual is split in two.

```
sigma_total² = sigma_game² + 2 · sigma_team²
             = 12.50²      + 2 · 4.06²
```

`sigma_team` is drawn **once per team per simulated season** and applied to all seventeen of that team's games. A team that is secretly two points better than its rating is two points better every week, not independently lucky seventeen times. Modelling only game-level noise produces win distributions roughly a third too narrow, which makes every playoff probability overconfident. That value is solved for by matching the simulator's spread to the model's own measured error.

### 6. Project players from opportunity

```
team plays per game
  × dropback rate            ← scheme model, adjusted for new play-callers
  = team pass attempts
  × player target share      ← empirical-Bayes shrunk usage, re-normalised
  = targets
  × catch rate, yards per target
  = receptions, receiving yards
+ the same chain on the ground
+ touchdowns from red-zone opportunity share, never last year's TD count
  = fantasy points
```

Two ideas do most of the work. **Empirical Bayes**: observed share is shrunk toward a depth-chart-aware prior by sample size. **Availability-weighted normalisation**: every team's shares are rescaled so that `Σ(share × games / 17) = 1`, which prices competition for touches as the zero-sum thing it is and makes roster movement work without hand-writing 32 depth charts.

### 7. De-vig before comparing anything

| Book | Outcomes | Raw sum | Overround |
|---|---|---|---|
| Super Bowl (Polymarket) | 32 | 1.0490 | +4.90% |
| Super Bowl (DraftKings) | 32 | 1.2266 | +22.66% |
| AFC Champion (Polymarket) | 16 | 0.9830 | **-1.70%** |
| NFC Champion (Polymarket) | 16 | 0.9735 | **-2.65%** |

A fair book sums to 1.00. Comparing a model to raw sportsbook prices produces an "edge" on all 32 teams, which is arithmetic rather than insight. De-vigging uses the power method, solving for the exponent that makes prices sum to one, because the margin sits disproportionately in the longshots.

Note the two conference books sum to **under** 100%. The same event priced two ways disagrees with itself.

---

## Week 1, game by game

![Week 1 matchups](docs/02-week1-matchups.jpg)

A projected score, a winner and a total for every game, with the posted line beside
it in plain English. The margin comes from the calibrated game model and the total
from a fitted scoring model, so the two can never imply an impossible pair of team
scores. Across Week 1 the model sits within 1.6 points of the closing line on
average, which is what should happen: a preseason model that routinely disagreed
with the market by a touchdown would be broken, not brilliant.

---

## The web terminal

![Scenario lab](docs/04-scenario-lab.jpg)

A single self-contained HTML file. No framework, no chart library, no network calls at runtime. Every chart is hand-built SVG and the full data bundle plus all 32 logos are inlined.

**Scenario lab.** The season Monte Carlo reimplemented in JavaScript: same team-constant strength draw, per-game noise, real 272-game schedule, division tiebreakers, seven-team bracket. 4,000 seasons in roughly 360ms. It reproduces the Python engine to within about 0.06 wins and one percentage point on playoff odds, which is itself the regression test. Drag a rating and all 32 teams re-solve.

**Draft room.** Value over replacement recomputed for league size and scoring on every pick, so twelve-team full PPR and ten-team standard produce genuinely different boards. Tier breaks, and scarcity measured as startable players remaining above replacement. Persists to `localStorage`.

![Draft room](docs/05-draft-room.jpg)

**Markets.** Power-method de-vig, then expected value and Kelly sizing against an editable price. Stakes are scaled to a deployment cap and the scaling factor is disclosed, because these markets are nested rather than independent and summing line-by-line Kelly across them produces allocations well over 100% of a bankroll.

![Markets](docs/06-markets.jpg)

---

## Repository layout

```
pylon-nfl/
├── config/
│   ├── offseason_2026.json         32 roster and scheme priors, each justified
│   ├── roles_2026.json             58 player role priors: rookies, injuries, role changes
│   ├── market_snapshot_2026.json   Polymarket and sportsbook prices, timestamped
│   ├── calibration.json            fitted shrinkage and translation constants
│   └── backtest.json               out-of-sample results and the variance split
├── src/pylon/
│   ├── config.py                   every tunable, with the reason it exists
│   ├── ingest.py                   play-by-play → clean plays + team context
│   ├── usage.py                    per-player opportunity shares
│   ├── ratings.py                  ridge opponent adjustment
│   ├── calibrate.py                fits shrinkage and translation out of sample
│   ├── backtest.py                 five-season replay, solves the variance split
│   ├── adjustments.py              2026 context projection + analyst priors
│   ├── game_model.py               ratings → spread → win probability
│   ├── simulate.py                 25,000-season Monte Carlo with real tiebreakers
│   ├── players.py                  opportunity-based fantasy projections
│   ├── draft.py                    model vs live ADP
│   ├── market.py                   de-vigging and the edge board
│   ├── diagnostics.py              decomposition and sensitivity
│   ├── charts.py                   broadcast visuals
│   ├── webdata.py                  packs outputs into the web bundle
│   └── run.py                      orchestrator
├── web/
│   ├── app.template.html           source for the terminal
│   ├── build.sh                    injects data + logos → index.html
│   └── test.mjs                    headless Playwright check
└── out/                            every published number
```

---

## Bugs found and fixed during development

Kept here deliberately. Each was silently producing wrong answers, and each is the kind of error a projection system can carry for a full season without anyone noticing.

1. **Season-total denominators.** Target and carry shares were divided by full-season team totals, so any player who missed games looked like he had lost a third of his role. One receiver's target share read 12% instead of 33%. Fixed by computing team opportunity only over the games each player actually played. This single fix moved dozens of players 60+ draft slots.

2. **Vanishing quarterback scrambles.** A scramble is charged as a dropback, so it never lands in the rushing bucket, and nflverse credits it to `rusher_player_id` rather than `passer_player_id`, so it never lands in the passing bucket either. It fell through both. That is roughly 40% of a mobile quarterback's carries and 70% of his rushing yards.

3. **Flat positional priors.** Every rostered receiver was given a 5.5% target-share prior. Twelve receivers × 5.5% sums past 100% of a team's targets before a single starter is counted, and the normalisation step then halved everybody. Replaced with depth-chart-aware priors.

4. **Double-taxed availability.** Quarterback dropback share was set to 0.90 to account for missed time, then multiplied by a projected games figure that already accounted for missed time. Every quarterback was penalised twice.

5. **Naive share normalisation.** Normalising shares to sum to 1.0 taxes every healthy starter to pay for the projected absences of his own backups. Replaced with the availability-weighted identity.

6. **Overconfident simulation.** The original simulator modelled only game-level noise and produced a mean win standard deviation of 2.00 against a measured out-of-sample error of 2.87. It was claiming 44-point probability edges that were mostly its own false precision.

7. **Unsourced vig figure.** The DraftKings overround was quoted in prose rather than computed, so the one number contrasting the two markets was not reproducible from the repo. It is now stored as American odds and converted by the same code path as everything else.

8. **Silent ADP join failure.** The player-to-ADP join matched on raw name, so "James Cook" versus "James Cook III" dropped ADP for around fifty players. Now joins on a normalised key.

9. **Kelly sizing across nested markets.** The market page summed line-by-line Kelly stakes across 160 correlated lines and reported "189% of bankroll deployed". Those markets all resolve off the same season. It now caps deployment and discloses the scaling factor.

---

## Known limitations

1. **The judgment layer is judgment.** Five teams are more than 80% prior-driven. Those projections are opinions with a model attached, and `rating_decomposition.csv` names them.
2. **Defensive ratings are nearly information-free from data alone.** With a measured carry-over of 0.14, play-by-play supplies almost no defensive differentiation, so the prior does proportionally more work there.
3. **No in-season updating.** Every rating is frozen at the start of the season. A live version would re-fit weekly.
4. **Point totals are risk-adjusted, not healthy-season.** Expected games played is projected rather than assumed at 17, which is why totals sit below published industry sets. A 17-game column is provided for like-for-like comparison.
5. **Quarterback VORP overstates draft-day value** in one-quarterback leagues, because the position is streamable in a way replacement-level math cannot capture.
6. **Rookies and role-changers are asserted, not measured.** 58 of them, each with a stated reason.

---

## Data sources

- **Play-by-play, rosters, schedule** - [nflverse](https://github.com/nflverse/nflverse-data), 2017-2025
- **Average draft position** - FantasyFootballCalculator API, 12-team full PPR, 7,726 drafts
- **Prediction markets** - Polymarket Gamma API, order-book midpoints
- **Sportsbook lines** - DraftKings board, converted from American odds

Team names, logos and marks are the property of their respective owners and appear here for identification in an independent analytical project. Not affiliated with or endorsed by the NFL or any club. Projections are estimates and nothing here is financial advice.

---

## Author

**Joshua Powers**

[joshpowersbiz@gmail.com](mailto:joshpowersbiz@gmail.com) · [LinkedIn](https://www.linkedin.com/in/joshuampowers) · [Instagram](https://instagram.com/joshuampowers) · [GitHub](https://github.com/Joshuapowers1)

If you disagree with a number, the code that produced it is right here. I would rather hear the argument than not.

Released under the [MIT License](LICENSE).
