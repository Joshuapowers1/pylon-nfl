# Contributing

Disagreement is the point of this project, so issues that argue with a number are as welcome as issues that report a crash.

## Arguing with a projection

Open an issue with the team or player, the number you disagree with, and why. The most useful version names the specific input you think is wrong: a shrinkage coefficient, an offseason prior in `config/offseason_2026.json`, or a role assumption in `config/roles_2026.json`. Every one of those is a single line you can point at.

Before you open it, check `out/rating_decomposition.csv`. If the team you are arguing about is more than 80% judgment-driven, you are probably right and the fix is a config edit rather than a model change.

## Reporting a bug

Include the command you ran and what came out. Silent wrong answers matter more than crashes here, so if a projection looks impossible, that is a bug report worth filing even without a stack trace.

## Changing the model

Any change to the modelling layer needs to move the backtest, not just the current season's output. Run:

```bash
PYTHONPATH=src python -m pylon.calibrate
PYTHONPATH=src python -m pylon.backtest
```

and include the before and after of `mae_wins` and `straight_up_game_accuracy` in the pull request. A change that improves this year's numbers and worsens the backtest is overfitting.

## Changing the web app

`web/index.html` is generated. Edit `web/app.template.html` and run `web/build.sh`. Then run `node web/test.mjs`, which checks every page in both themes at three breakpoints for JavaScript errors and horizontal overflow.

Design constraints worth preserving:

- Pylon orange is brand chrome only, never data.
- One diverging pair carries all analytic meaning: blue is "model higher than market", red is "model lower". Same language everywhere.
- The categorical palette is capped at four series and every line is direct-labelled.
- The complete light palette lives on bare `:root`, redefined under the dark media query and again under `[data-theme="dark"]` so the toggle wins both ways.
