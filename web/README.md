# Pylon Terminal - web app

A single self-contained HTML file. No build step, no dependencies, no network calls at runtime.
Every chart is hand-built SVG and the full data bundle plus all 32 team logos are inlined.

## Rebuilding after the model changes

```bash
cd .. && python -m pylon.run && python -m pylon.webdata   # regenerate out/web_bundle.json
cd web && ./build.sh                                      # inject data + logos into index.html
node test.mjs                                             # headless check, all pages, both themes
```

`app.template.html` is the source. It carries `__DATA__` and `__LOGOS__` placeholders that
`build.sh` fills. Never hand-edit `index.html`, it is generated.

## What runs client side

- **Scenario lab** reimplements the season Monte Carlo in JavaScript: the same team-constant
  strength draw, per-game noise, real schedule, division tiebreakers and seven-team bracket.
  It reproduces the Python engine to within about a tenth of a win, which is the test. The one
  documented difference is that head-to-head is not used as a tiebreaker client side.
- **Draft room** recomputes value over replacement for the chosen league size and scoring, so
  changing from twelve-team full PPR to ten-team standard reorders the board live. State persists
  to `localStorage`.
- **Markets** de-vigs by the power method, then computes EV and Kelly against an editable price.
  Stakes are scaled to a deployment cap because the markets are nested rather than independent.

## Design constraints worth preserving

- Pylon orange is brand chrome only, never data.
- All analytic meaning runs through one diverging pair: blue is "model higher than market", red is
  "model lower". Same language on the market board and the draft board.
- The categorical palette is capped at four series and every line is direct-labelled.
- Complete light palette on bare `:root`, redefined under the dark media query and again under
  `[data-theme="dark"]` so the toggle wins both ways.

---

Part of [Pylon](https://github.com/Joshuapowers1/pylon-nfl) by Joshua Powers.
[joshpowersbiz@gmail.com](mailto:joshpowersbiz@gmail.com) · [LinkedIn](https://www.linkedin.com/in/joshuampowers) · [@joshuampowers](https://instagram.com/joshuampowers)
